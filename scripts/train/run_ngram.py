#!/usr/bin/env python3
"""NGram adapter: run-level token model in the schema-v2 no-leak framework.

NGramDetector scores whole runs from raw per-run trace CSVs (event tokens like
A_256 / F_2048), not the 90-dim window features the other models consume. This
runner keeps the exact v2 methodology -- the same stratified whole-run split
(common.split_run_groups), the same p99-of-validation-normal threshold policy,
the same gates G7/G8/G10, and the same artifact layout under runs/<id>/ -- but
evaluates at the RUN level (one score per run), so ``sequence_level`` is n/a
and the report renders its seq columns as "-".

Memory: the raw traces total ~1.8GB for final-v2, so every run's token stream
is parsed and reduced to an n-gram Counter (bounded by the ~26-token alphabet)
before the next run is touched; no full trace is ever materialised.

Usage:
    python3 scripts/train/run_ngram.py \
        --attack-data datasets/final-v2/processed/attack \
        --normal-data datasets/final-v2/processed/normal \
        --normal-csv-root datasets/final-v2/csv/normal \
        --attack-csv-root datasets/final-v2/csv/attack \
        [--held-out-cve CVE-2017-11176] --out runs
"""

import argparse
import csv
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config  # noqa: E402
from models import NGramDetector  # noqa: E402
from models.ngram import bucketize_size  # noqa: E402
from scripts.train import common  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_ngram")


def token_iter(csv_path):
    """Yield event tokens (A_<bucket> / F_<bucket>) in timestamp order, streaming.

    Traces are written by ftrace in insertion order, so row order == event order
    (the legacy csv_to_events defensive sort is unnecessary here). FREE tokens
    resolve their size from the live-allocation table of the matching ALLOC,
    exactly like models/ngram.csv_to_events.
    """
    live = {}
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            op = row["op"]
            if op == "ALLOC":
                ptr = row["ptr"]
                size = 0
                try:
                    size = int(row.get("bytes_alloc") or 0) or int(row.get("bytes_req") or 0)
                except ValueError:
                    size = 0
                live[ptr] = size
                yield f"A_{bucketize_size(size) if size else 'unknown'}"
            else:
                size = live.pop(row["ptr"], 0)
                yield f"F_{bucketize_size(size) if size else 'unknown'}"


def ngram_counts(csv_path, n):
    """Per-run n-gram Counter via a sliding window (no token list materialised)."""
    counts = Counter()
    window = deque(maxlen=n)
    for token in token_iter(csv_path):
        window.append(token)
        if len(window) == n:
            counts[tuple(window)] += 1
    return counts


def csv_path_for_run(csv_root, run_id):
    """run_id 'CVE-2017-11176/fs_io/run_000_ab12cd34/trace' -> <root>/CVE/fs_io/run_000_ab12cd34/trace.csv.

    trace2csv nests each run's CSV under a <run_dir>/trace.csv, and the run id
    *is* that path (minus the .csv suffix), so run_id + ".csv" is the file.
    """
    return Path(csv_root) / (str(run_id) + ".csv")


def sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision():
    root = Path(__file__).resolve().parents[1]
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    log.info("wrote %s", path)


def gate_report(gates):
    print("\n=== M5 acceptance gates (ngram, run-level) ===")
    ok = True
    for gate in gates:
        mark = "PASS" if gate["ok"] else "FAIL"
        if not gate["ok"]:
            ok = False
        print(f"[{mark}] {gate['name']}: {gate['detail']}")
    print("\n" + ("ALL M5 GATES PASS" if ok else "SOME M5 GATES FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser(description="NGram run-level train/evaluate (v2 harness adapter)")
    ap.add_argument("--attack-data", required=True)
    ap.add_argument("--normal-data", required=True)
    ap.add_argument("--normal-csv-root", required=True)
    ap.add_argument("--attack-csv-root", required=True)
    ap.add_argument("--dataset-manifest", default="datasets/pilot-v2/dataset_manifest.json")
    ap.add_argument("--out", default=config.RUNS_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-vocab", type=int, default=500)
    ap.add_argument("--val-fraction", type=float, default=common.DEFAULT_VAL_FRACTION)
    ap.add_argument("--test-fraction", type=float, default=common.DEFAULT_TEST_FRACTION)
    ap.add_argument("--target-fpr", type=float, default=common.DEFAULT_TARGET_FPR)
    ap.add_argument("--held-out-cve", default=None)
    ap.add_argument("--train-cves", nargs="+", default=None,
                    help="CVEs whose NORMAL data is used for training (default: all)")
    ap.add_argument("--test-cves", nargs="+", default=None,
                    help="CVEs whose data is used for evaluation (default: all)")
    args = ap.parse_args()

    def cve_of(run_id):
        return str(run_id).split("/", 1)[0]

    if args.val_fraction + args.test_fraction >= 0.5:
        ap.error("val+test fractions must be < 0.5")
    if not (0 < args.target_fpr < 1):
        ap.error("--target-fpr must be in (0,1)")

    experiment = common.experiment_id("ngram", "ngram", args.seed)
    if args.held_out_cve:
        experiment += f"_holdout_{args.held_out_cve.rsplit('-', 1)[-1]}"
    if args.train_cves:
        experiment += "_" + ("".join(c.rsplit("-", 1)[-1] for c in args.train_cves)
                             + "_test_" + "".join(c.rsplit("-", 1)[-1] for c in args.test_cves))
    experiment_dir = Path(args.out) / experiment
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty experiment dir: {experiment_dir}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    log.info("experiment: %s", experiment_dir)

    write_json(experiment_dir / "experiment_config.json", {
        "schema_version": 2,
        "experiment_id": experiment,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "ngram",
        "model_config": {"n": args.n, "max_vocab": args.max_vocab, "alpha": 1e-3},
        "evaluation_level": "run",
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "target_fpr": args.target_fpr,
        "held_out_cve": args.held_out_cve,
        "train_cves": args.train_cves,
        "test_cves": args.test_cves,
        "inputs": {
            "attack_data": args.attack_data,
            "normal_data": args.normal_data,
            "attack_csv_root": args.attack_csv_root,
            "normal_csv_root": args.normal_csv_root,
            "attack_features_sha256": sha256_file(os.path.join(args.attack_data, "features.npz")),
            "normal_features_sha256": sha256_file(os.path.join(args.normal_data, "features.npz")),
        },
        "git_revision": git_revision(),
        "python_version": sys.version.split()[0],
    })

    # ---- run universe: identical to run_experiment.py's ---------------------
    attack, attack_stats = common.load_processed(Path(args.attack_data))
    normal, normal_stats = common.load_processed(Path(args.normal_data))
    normal_groups_all = np.asarray(normal["seq_run_ids"]).astype(str)
    attack_labels = np.asarray(attack["seq_labels"], dtype=np.int8)
    attack_keep = attack_labels >= 0  # drop boundary sequences (plan 5.6)
    attack_groups_used = np.asarray(attack["seq_run_ids"]).astype(str)[attack_keep]
    attack_run_ids = sorted(set(attack_groups_used.tolist()))
    if args.test_cves:
        attack_run_ids = [r for r in attack_run_ids if cve_of(r) in args.test_cves]

    # CVE-aware split (mirrors run_cve_split.py). Default (no --train-cves):
    # train on ALL normal, test = held 15% slice. With CVEs: the train pool is
    # the train-cves' normal (70/15/15); a test-cve NOT in train-cves contributes
    # ALL its normal runs to eval; a train-cve's held 15% slice enters eval only
    # if that cve is also a test-cve.
    all_normal_groups = sorted({str(g) for g in normal_groups_all})
    if args.train_cves:
        train_pool = [g for g in all_normal_groups if cve_of(g) in args.train_cves]
        tr_groups, val_groups, held_test_groups = common.split_run_groups(
            train_pool, args.seed, args.val_fraction, args.test_fraction)
        test_groups = [g for g in held_test_groups if cve_of(g) in args.test_cves]
        for g in all_normal_groups:
            if cve_of(g) in args.test_cves and cve_of(g) not in args.train_cves:
                test_groups.append(g)
        test_groups = sorted(set(test_groups))
    else:
        tr_groups, val_groups, test_groups = common.split_run_groups(
            normal_groups_all, args.seed, args.val_fraction, args.test_fraction)
    train_groups = tr_groups
    g7_ok = (len(set(train_groups) & set(val_groups)) == 0
             and len(set(train_groups) & set(test_groups)) == 0
             and len(set(val_groups) & set(test_groups)) == 0)
    gates = [{
        "name": "G7_run_split_no_overlap",
        "ok": g7_ok,
        "detail": f"train={len(train_groups)} val={len(val_groups)} test={len(test_groups)} "
                  f"runs, pairwise-disjoint={g7_ok}",
    }]
    write_json(experiment_dir / "split_manifest.json", {
        "schema_version": 2, "seed": args.seed,
        "policy": "stratified-by-workload, whole runs, 70/15/15",
        "val_fraction": args.val_fraction, "test_fraction": args.test_fraction,
        "train_groups": train_groups, "val_groups": val_groups, "test_groups": test_groups,
        "held_out_cve": args.held_out_cve,
        "train_cves": args.train_cves, "test_cves": args.test_cves,
    })

    def build_counts(run_ids, csv_root, label):
        out, missing = {}, 0
        for i, rid in enumerate(run_ids):
            path = csv_path_for_run(csv_root, rid)
            if not path.exists():
                missing += 1
                continue
            out[rid] = ngram_counts(path, args.n)
            if (i + 1) % 50 == 0:
                log.info("built counts for %d/%d %s runs", i + 1, len(run_ids), label)
        if missing:
            log.warning("%s: %d/%d run ids had no CSV", label, missing, len(run_ids))
        return out

    # ---- 1. fit on train-normal n-gram counts (streamed, one run at a time) ---
    _t_fit = time.perf_counter()
    train_counts = build_counts(train_groups, args.normal_csv_root, "train-normal")
    model = NGramDetector(n=args.n, max_vocab=args.max_vocab)
    model.fit_counts([train_counts[r] for r in train_groups if r in train_counts])
    pickle.dump(model, open(experiment_dir / "model.pkl", "wb"))
    log.info("fit on %d train runs (%.1fs, vocab=%d)",
             len(train_counts), time.perf_counter() - _t_fit, len(model.vocab))

    # ---- 2. score every run (val/test normal + attack) -----------------------
    _t_score = time.perf_counter()
    normal_counts = build_counts(list(val_groups) + list(test_groups),
                                 args.normal_csv_root, "val/test-normal")
    attack_counts = build_counts(attack_run_ids, args.attack_csv_root, "attack")

    def run_scores(counts, ids):
        return np.asarray([model.anomaly_score_from_counts(counts[r]) for r in ids
                           if r in counts], dtype=np.float64)

    val_scores = run_scores(normal_counts, val_groups)
    threshold = common.threshold_at_fpr(val_scores, args.target_fpr)
    run_threshold = threshold  # run-level model: no separate window/seq threshold
    test_scores = run_scores(normal_counts, test_groups)
    attack_scores = run_scores(attack_counts, attack_run_ids)
    score_seconds = time.perf_counter() - _t_score
    log.info("threshold(p%g normal val, run-level)=%.6f",
             100 * (1 - args.target_fpr), threshold)

    # ---- 3. threshold + LOO assembly -----------------------------------------
    test_labels = np.zeros(len(test_scores), dtype=np.int8)
    attack_labels_run = np.ones(len(attack_scores), dtype=np.int8)
    # attack_scores is aligned to the run ids whose CSV existed.
    kept_ids = [r for r in attack_run_ids if r in attack_counts]
    if args.held_out_cve:
        held_sel = np.asarray([r.startswith(args.held_out_cve + "/") for r in kept_ids])
        dev_sel = ~held_sel
        all_scores = np.concatenate((test_scores, attack_scores[dev_sel]))
        all_labels = np.concatenate((test_labels, attack_labels_run[dev_sel]))
        all_groups = np.concatenate((np.char.add("normal:", np.asarray(test_groups)),
                                     np.char.add("attack:", np.asarray(kept_ids)[dev_sel])))
        held_scores = attack_scores[held_sel]
        held_labels = attack_labels_run[held_sel]
    else:
        all_scores = np.concatenate((test_scores, attack_scores))
        all_labels = np.concatenate((test_labels, attack_labels_run))
        all_groups = np.concatenate((np.char.add("normal:", np.asarray(test_groups)),
                                     np.char.add("attack:", np.asarray(kept_ids))))
        held_scores, held_labels = None, None

    run_metrics = common.classification_metrics(all_scores, all_labels, run_threshold)
    run_ci = common.bootstrap_run_ci(all_scores, all_labels, run_threshold,
                                     n_boot=2000, seed=args.seed)

    # grouped: per-CVE / per-variant detectability = that group's attack runs vs
    # normal test runs (run-level; the window models' within-attack spray-vs-
    # context seq AUC has no run-level analogue, so we use attack-vs-normal).
    def attack_vs_normal(sub_ids):
        idx = np.asarray([r in set(sub_ids) for r in kept_ids])
        s = np.concatenate((attack_scores[idx], test_scores))
        l = np.concatenate((np.ones(int(idx.sum()), dtype=np.int8), test_labels))
        return common.classification_metrics(s, l, run_threshold)

    by_cve, by_variant = {}, {}
    for rid in kept_ids:
        cve = rid.split("/", 1)[0]
        by_cve.setdefault(cve, []).append(rid)
        by_variant.setdefault("/".join(rid.split("/", 2)[:2]), []).append(rid)
    grouped = {"by_cve": {cve: attack_vs_normal(ids) for cve, ids in by_cve.items()},
               "by_variant": {v: attack_vs_normal(ids) for v, ids in by_variant.items()},
               "by_workload": {}, "by_slab": {}}

    evaluation_report = {
        "schema_version": 2,
        "experiment_id": experiment,
        "model": "ngram",
        "evaluation_level": "run",
        "held_out_cve": args.held_out_cve,
        "sequence_level": None,  # run-level model: n/a
        "run_level": run_metrics,
        "run_bootstrap_ci95": run_ci,
        "grouped": grouped,
        "counts": {
            "test_normal_runs": int(len(test_scores)),
            "attack_runs_evaluated": int(len(attack_scores)),
            "val_normal_runs": int(len(val_scores)),
        },
        "inference": {
            "score_seconds": round(score_seconds, 4),
            "runs_per_second": round(float(len(all_scores)) / max(score_seconds, 1e-9), 1),
            "sequences_per_second": None,  # run-level CSV-token model: n/a
            "windows_per_second": None,
        },
    }
    if held_scores is not None:
        evaluation_report["held_out_cve_metrics"] = {
            "cve": args.held_out_cve,
            "run_level": common.classification_metrics(
                np.concatenate((held_scores, test_scores)),
                np.concatenate((held_labels, test_labels)), run_threshold),
        }
    write_json(experiment_dir / "evaluation_report.json", evaluation_report)

    train_report = {
        "schema_version": 2,
        "experiment_id": experiment,
        "model": "ngram",
        "model_config": {"n": args.n, "max_vocab": args.max_vocab},
        "evaluation_level": "run",
        "threshold_policy": f"validation-normal FPR={args.target_fpr} (run-level)",
        "threshold": threshold,
        "run_threshold": run_threshold,
        "val_normal_runs": int(len(val_scores)),
        "seed": args.seed,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(experiment_dir / "train_report.json", train_report)

    # ---- 4. gates G8 / G10 ---------------------------------------------------
    g8_ok = bool(run_metrics.get("roc_auc") is not None)
    gates.append({
        "name": "G8_train_evaluate_end_to_end",
        "ok": g8_ok,
        "detail": f"model=ngram trained and evaluated; test normal={len(test_scores)} "
                  f"runs / attack={int(np.sum(attack_labels_run))} runs",
    })
    base_run_flags = all_scores[
        np.asarray(["/poc_cfh_baseline/" in g for g in all_groups])] > run_threshold
    n_base_runs = int(len(base_run_flags))
    n_baseline = int(base_run_flags.sum()) if n_base_runs else 0
    if n_base_runs == 0:
        gates.append({"name": "G10_baseline_not_flagged", "ok": True,
                      "detail": "no baseline runs in test partition -- not checked"})
    else:
        allowed = 1 if n_base_runs <= 5 else max(1, int(round(0.3 * n_base_runs)))
        g10_ok = n_baseline <= allowed
        gates.append({
            "name": "G10_baseline_not_flagged",
            "ok": g10_ok,
            "detail": f"{n_baseline}/{n_base_runs} baseline runs flagged at run_threshold "
                      f"(allowed <= {allowed})",
        })
    write_json(experiment_dir / "gates.json", gates)
    gate_report(gates)

    import csv as csv_mod
    row = {"experiment_id": experiment, "model": "ngram", "seed": args.seed,
           "target_fpr": args.target_fpr, "held_out_cve": args.held_out_cve or "-",
           "evaluation_level": "run"}
    for key in ("roc_auc", "pr_auc", "f1_at_threshold", "fpr_at_threshold",
                "recall_at_threshold", "precision_at_threshold"):
        row[f"run_{key}"] = run_metrics.get(key, "")
    with open(experiment_dir / "metrics.csv", "w", newline="") as handle:
        writer = csv_mod.DictWriter(handle, fieldnames=sorted(row))
        writer.writeheader()
        writer.writerow(row)

    src = Path(args.dataset_manifest)
    if src.exists():
        (experiment_dir / "dataset_manifest.json").write_text(src.read_text())

    log.info("done: %s", experiment_dir)
    return 0 if all(g["ok"] for g in gates) else 1


if __name__ == "__main__":
    sys.exit(main())
