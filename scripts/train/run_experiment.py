#!/usr/bin/env python3
"""Schema-v2 no-leak train/evaluate harness (M5).

Runs the full pipeline for a baseline anomaly detector and writes every artifact
under runs/<experiment_id>/:

    experiment_config.json   input data + parameters (traceability, plan 4.2)
    split_manifest.json      run-level train/val/test split (G7)
    scaler.npz               Gaussian stats fit on train windows only
    model.pkl                fitted model (train runs only)
    train_report.json        threshold calibrated on validation-normal FPR
    evaluation_report.json   test metrics at the frozen threshold + grouping + CI
    metrics.csv              flat summary
    gates.json               M5 gates G7/G8/G10

Usage:
    python3 scripts/train/run_experiment.py --model ocsvm \
        --attack-data datasets/pilot-v2/processed/attack \
        --normal-data datasets/pilot-v2/processed/normal --out runs
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config  # noqa: E402
from models import OCSVMDetector, TorchAEWrapper  # noqa: E402
from models.event_gru import EventGRUDetector  # noqa: E402
from models.gru_detector import GRUDetector  # noqa: E402
from models.fusion_svdd import FusionSVDDDetector  # noqa: E402
from scripts.train import common  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_experiment")

MODEL_FACTORY = {
    "ocsvm": lambda seed: OCSVMDetector(kernel="rbf", nu=0.05, gamma="scale"),
    "lstm_ae": lambda seed: TorchAEWrapper("lstm_ae", seed=seed,
                                           epochs=25, seq_batch_size=64),
    "lstm_vae": lambda seed: TorchAEWrapper("lstm_vae", seed=seed,
                                            epochs=25, seq_batch_size=64, beta=1.0),
    "gru": lambda seed: GRUDetector(seed=seed, epochs=20, batch_size=256, g=10),
    "event_gru": lambda seed: EventGRUDetector(seed=seed, epochs=20, batch_size=256,
                                               g_size=3),
    "fusion_svdd": lambda seed: FusionSVDDDetector(
        seed=seed, epochs=20, batch_size=256, g=10,
        svdd_loss_weight=0.1, svdd_score_weight=0.3),
}

MODEL_CONFIG = {
    "ocsvm": {"kernel": "rbf", "nu": 0.05, "gamma": "scale"},
    "lstm_ae": {"hidden_dim": 64, "latent_dim": 16, "num_layers": 2, "epochs": 25, "lr": 1e-3},
    "lstm_vae": {"hidden_dim": 64, "latent_dim": 16, "num_layers": 2, "epochs": 25,
                 "lr": 1e-3, "beta": 1.0},
    "gru": {"d_model": 128, "n_layers": 2, "epochs": 20, "lr": 1e-3, "g": 10,
            "vocab_size": 13824},
    "event_gru": {"d_model": 128, "n_layers": 2, "epochs": 20, "lr": 1e-3,
                  "g_size": 3, "cs_vocab": 4096, "w_op": 0.05, "w_size": 0.35,
                  "w_csrep": 0.20, "w_cpu": 0.05, "w_reclaim": 0.05, "w_dt": 0.30},
    "fusion_svdd": {"d_model": 128, "n_layers": 2, "epochs": 20, "lr": 1e-3,
                    "g": 10, "svdd_loss_weight": 0.1, "svdd_score_weight": 0.3,
                    "vocab_size": 13824},
}

# Models that use token sequences (N, L) int32 instead of feature sequences (N, T, F) float32.
TOKEN_MODELS = {"gru", "fusion_svdd"}

# Models that use the event-embedding field view (N, L, 6) float32 and fit an
# end-to-end GRU (fit_sequences) with NO input scaler — like token models, they
# skip the feature scaler; unlike token models, their data is not token ids.
EVENT_MODELS = {"event_gru"}

# Offline sequence models (token or event): never use feature windows / scaler.
SEQUENCE_MODELS = TOKEN_MODELS | EVENT_MODELS

# Default aggregation per model family: GRU/event-GRU/FusionSVDD uses mean
# (violation rate), feature models use max (any-window anomaly).
DEFAULT_AGGREGATION = {"gru": "mean", "event_gru": "mean", "fusion_svdd": "mean"}


def make_experiment_dir(out_root, experiment_id):
    path = Path(out_root) / experiment_id
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty experiment dir: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(exist_ok=True)
    return path


def save_model(model, path):
    with open(path, "wb") as handle:
        pickle.dump(model, handle)


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    log.info("wrote %s", path)


def gate_report(gates):
    print("\n=== M5 acceptance gates ===")
    ok = True
    for gate in gates:
        mark = "PASS" if gate["ok"] else "FAIL"
        if not gate["ok"]:
            ok = False
        print(f"[{mark}] {gate['name']}: {gate['detail']}")
    print("\n" + ("ALL M5 GATES PASS" if ok else "SOME M5 GATES FAILED"))
    return ok


def parse_group(group):
    """Extract (cve, sub) from a scored group id.

    Group ids are 'CLASS:CVE/sub/run_NNN_hash/trace'. For attack groups the
    sub is the PoC variant; for normal groups it is the workload.
    """
    path = str(group).split(":", 1)[1]
    parts = path.split("/")
    cve = parts[0] if parts else "?"
    sub = parts[1] if len(parts) > 1 else "?"
    return cve, sub


def sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision():
    """HEAD revision of the repo, or None when git/unavailable (plan 4.2)."""
    root = Path(__file__).resolve().parents[1]
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="M5 no-leak train/evaluate pipeline")
    parser.add_argument("--model", default="ocsvm",
                        choices=sorted(MODEL_FACTORY))
    parser.add_argument("--attack-data", required=True)
    parser.add_argument("--normal-data", required=True)
    parser.add_argument("--dataset-manifest", default="datasets/dataset_manifest.json")
    parser.add_argument("--out", default=config.RUNS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=common.DEFAULT_VAL_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=common.DEFAULT_TEST_FRACTION)
    parser.add_argument("--target-fpr", type=float, default=common.DEFAULT_TARGET_FPR)
    parser.add_argument("--aggregation", choices=["max", "last", "p90", "mean"], default=None,
                        help="sequence score aggregation; default per model (gru=mean, others=max)")
    parser.add_argument("--held-out-cve", default=None,
                        help="leave-one-CVE-out: exclude this CVE's attack from aggregate eval")
    args = parser.parse_args()

    if args.aggregation is None:
        args.aggregation = DEFAULT_AGGREGATION.get(args.model, "max")

    if args.val_fraction + args.test_fraction >= 0.5:
        parser.error("val+test fractions must be < 0.5")
    if not (0 < args.target_fpr < 1):
        parser.error("--target-fpr must be in (0,1)")

    experiment = common.experiment_id(args.model, args.model, args.seed)
    if args.held_out_cve:
        experiment += f"_holdout_{args.held_out_cve.rsplit('-', 1)[-1]}"
    experiment_dir = make_experiment_dir(args.out, experiment)
    log.info("experiment: %s", experiment_dir)

    # ---- 0. experiment config (traceability, runs/README.md spec) ----------
    write_json(experiment_dir / "experiment_config.json", {
        "schema_version": 2,
        "experiment_id": experiment,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_config": MODEL_CONFIG[args.model],
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "target_fpr": args.target_fpr,
        "aggregation": args.aggregation,
        "held_out_cve": args.held_out_cve,
        "inputs": {
            "attack_data": args.attack_data,
            "normal_data": args.normal_data,
        },
        "git_revision": git_revision(),
        "python_version": sys.version.split()[0],
    })

    # ---- load data (token / event / feature, depending on model) -----------
    is_token_model = args.model in TOKEN_MODELS
    is_event_model = args.model in EVENT_MODELS
    if is_token_model:
        log.info("loading token sequences for model=%s", args.model)
        attack_tokens = common.load_token_data(Path(args.attack_data))
        normal_tokens = common.load_token_data(Path(args.normal_data))
        attack_seqs = attack_tokens["token_seqs"]
        attack_seq_labels = np.asarray(attack_tokens["token_seq_labels"], dtype=np.int8)
        attack_seq_run_ids = np.asarray(attack_tokens["token_seq_run_ids"]).astype(str)
        normal_seqs = normal_tokens["token_seqs"]
        normal_seq_run_ids = np.asarray(normal_tokens["token_seq_run_ids"]).astype(str)
        seq_len = int(normal_seqs.shape[1])
        feat_dim = 0
        attack_hash = sha256_file(os.path.join(args.attack_data, "token_sequences.npz"))
        normal_hash = sha256_file(os.path.join(args.normal_data, "token_sequences.npz"))
    elif is_event_model:
        log.info("loading event fields for model=%s", args.model)
        attack_ev = common.load_event_data(Path(args.attack_data))
        normal_ev = common.load_event_data(Path(args.normal_data))
        attack_seqs = attack_ev["event_fields"]
        attack_seq_labels = attack_ev["event_field_labels"]
        attack_seq_run_ids = attack_ev["event_field_run_ids"]
        normal_seqs = normal_ev["event_fields"]
        normal_seq_run_ids = normal_ev["event_field_run_ids"]
        seq_len = int(normal_seqs.shape[1])
        feat_dim = int(normal_seqs.shape[2])
        attack_hash = sha256_file(os.path.join(args.attack_data, "event_fields.npz"))
        normal_hash = sha256_file(os.path.join(args.normal_data, "event_fields.npz"))
    else:
        attack, attack_stats = common.load_processed(Path(args.attack_data))
        normal, normal_stats = common.load_processed(Path(args.normal_data))
        if attack["sequences"].shape[-1] != normal["sequences"].shape[-1]:
            raise ValueError("attack/normal feature dimension mismatch")
        attack_seqs = attack["sequences"]
        attack_seq_labels = np.asarray(attack["seq_labels"], dtype=np.int8)
        attack_seq_run_ids = np.asarray(attack["seq_run_ids"]).astype(str)
        normal_seqs = normal["sequences"]
        normal_seq_run_ids = np.asarray(normal["seq_run_ids"]).astype(str)
        seq_len = int(normal_seqs.shape[1])
        feat_dim = int(normal_seqs.shape[2])
        attack_hash = sha256_file(os.path.join(args.attack_data, "features.npz"))
        normal_hash = sha256_file(os.path.join(args.normal_data, "features.npz"))

    # update experiment config with actual hashes
    cfg_path = experiment_dir / "experiment_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["inputs"]["attack_data_sha256"] = attack_hash
    cfg["inputs"]["normal_data_sha256"] = normal_hash
    cfg["is_token_model"] = is_token_model
    cfg["is_event_model"] = is_event_model
    write_json(cfg_path, cfg)

    # ---- 1. run split (G7) ------------------------------------------------
    normal_groups_all = normal_seq_run_ids
    train_groups, val_groups, test_groups = common.split_run_groups(
        normal_groups_all, args.seed, args.val_fraction, args.test_fraction)
    train_seq_mask = common.mask_for_groups(normal_groups_all, train_groups)
    val_seq_mask = common.mask_for_groups(normal_groups_all, val_groups)
    test_seq_mask = common.mask_for_groups(normal_groups_all, test_groups)
    if not all(m.any() for m in (train_seq_mask, val_seq_mask, test_seq_mask)):
        raise ValueError("empty partition; adjust run counts")

    # window mask only for feature models (token/event models have no windows)
    train_window_mask = None
    if not is_token_model and not is_event_model:
        train_window_mask = common.mask_for_groups(
            np.asarray(normal["window_run_ids"]).astype(str), train_groups)
        if not train_window_mask.any():
            raise ValueError("empty train window partition; adjust run counts")
    split_payload = {
        "schema_version": 2,
        "seed": args.seed,
        "policy": "stratified-by-workload, whole runs, 70/15/15",
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "test_groups": test_groups,
        "held_out_cve": args.held_out_cve,
    }
    write_json(experiment_dir / "split_manifest.json", split_payload)
    g7_ok = (len(set(train_groups) & set(val_groups)) == 0
             and len(set(train_groups) & set(test_groups)) == 0
             and len(set(val_groups) & set(test_groups)) == 0)
    gates = [{
        "name": "G7_run_split_no_overlap",
        "ok": g7_ok,
        "detail": f"train={len(train_groups)} val={len(val_groups)} test={len(test_groups)} "
                  f"runs, pairwise-disjoint={g7_ok}",
    }]

    # ---- 2. scaler + model fit (token models skip scaler) ------------------
    model = MODEL_FACTORY[args.model](args.seed)
    dev = getattr(model, "_device", lambda: None)()
    log.info("model=%s device=%s", args.model, dev.type if dev is not None else "cpu")

    if args.model in SEQUENCE_MODELS:
        train_dtype = np.int32 if is_token_model else np.float32
        train_seqs = normal_seqs[train_seq_mask].astype(train_dtype)
        model.fit_sequences(train_seqs)
        log.info("model=%s trained on %d train sequences", args.model, len(train_seqs))
    else:
        train_windows = normal["features"][train_window_mask].astype(np.float32)
        mean, std = common.fit_scaler(train_windows)
        np.savez_compressed(experiment_dir / "scaler.npz", mean=mean, std=std)
        if getattr(model, "sequence_model", False):
            train_sequences = normal_seqs[train_seq_mask].astype(np.float32)
            model.fit_sequences(train_sequences)
            log.info("model=%s trained on %d train sequences", args.model, len(train_sequences))
        else:
            model.fit(train_windows)
            log.info("model=%s trained on %d train windows", args.model, len(train_windows))
    save_model(model, experiment_dir / "model.pkl")

    # dtype for sequence arrays: token ids vs float event/feature fields
    seq_dtype = np.int32 if is_token_model else np.float32

    # ---- 3. threshold calibration on validation-normal FPR ------------------
    val_sequences = normal_seqs[val_seq_mask].astype(seq_dtype)
    val_groups_arr = normal_groups_all[val_seq_mask]
    val_seq_scores = common.score_sequences(model, val_sequences, args.aggregation)
    val_run_scores, val_run_ids = common.run_max_scores(val_seq_scores, val_groups_arr)
    run_threshold = common.threshold_at_fpr(val_run_scores, args.target_fpr)
    log.info("run_threshold(p%g normal val)=%.6f", 100 * (1 - args.target_fpr), run_threshold)

    # ---- 4. train report ----------------------------------------------------
    train_report = {
        "schema_version": 2,
        "experiment_id": experiment,
        "model": args.model,
        "model_config": MODEL_CONFIG[args.model],
        "feat_dim": feat_dim,
        "seq_len": seq_len,
        "score_aggregation": args.aggregation,
        "threshold_policy": "validation-normal run-level FPR=%s" % args.target_fpr,
        "run_threshold": run_threshold,
        "val_normal_sequences": int(len(val_seq_scores)),
        "val_normal_runs": int(len(val_run_scores)),
        "scaler_fit_on": "none (sequence model)" if args.model in SEQUENCE_MODELS else "train_runs_only",
        "scaler_npz": "none (sequence model)" if args.model in SEQUENCE_MODELS else "scaler.npz",
        "seed": args.seed,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(experiment_dir / "train_report.json", train_report)

    # ---- 5. evaluation on held-out test -------------------------------------
    _t_score = time.perf_counter()
    test_sequences = normal_seqs[test_seq_mask].astype(seq_dtype)
    test_groups_arr = normal_groups_all[test_seq_mask]
    test_scores = common.score_sequences(model, test_sequences, args.aggregation)
    test_labels = np.zeros(len(test_scores), dtype=np.int8)

    # Score ALL attack sequences: boundary sequences (label -1, partial spray
    # overlap) may contain spray events and often score high, so they must
    # count toward the run-level max. Evaluation is run-level only (the
    # detection unit is "did this run contain a spray attack"); sequence
    # scores are an intermediate aggregation, never reported as metrics.
    attack_sequences_all = attack_seqs.astype(seq_dtype)
    attack_scores_all = common.score_sequences(model, attack_sequences_all, args.aggregation)
    del attack_sequences_all
    score_seconds = time.perf_counter() - _t_score
    scored_sequences = len(test_scores) + len(attack_scores_all)

    if args.held_out_cve:
        # Run pool mirrors the dev subset (held-out CVE stays out of aggregate)
        held_mask_full = np.array([g.startswith(args.held_out_cve + "/")
                                   for g in attack_seq_run_ids])
        run_pool_scores = np.concatenate((test_scores, attack_scores_all[~held_mask_full]))
        run_pool_groups = np.concatenate((
            np.char.add("normal:", test_groups_arr),
            np.char.add("attack:", attack_seq_run_ids[~held_mask_full])))
        run_pool_labels = np.concatenate((test_labels, attack_seq_labels[~held_mask_full]))
    else:
        # Run pool includes ALL attack sequences (boundary included): run-level
        # max must see every sequence score of the run.
        run_pool_scores = np.concatenate((test_scores, attack_scores_all))
        run_pool_groups = np.concatenate((np.char.add("normal:", test_groups_arr),
                                          np.char.add("attack:", attack_seq_run_ids)))
        run_pool_labels = np.concatenate((test_labels, attack_seq_labels))

    run_scores, run_ids = common.run_max_scores(run_pool_scores, run_pool_groups)
    # Run-level label from run identity (data-source class), NOT derived from
    # sequence labels: an attack run stays attack even when none of its
    # sequences carries label 1 (spray window diluted below sequence
    # granularity, or dropped by the boundary policy). The collector fixed the
    # run's class at collection time (PoC executed with spray markers).
    # Baseline runs are collected under normal/ and are intentional negative
    # controls (label 0, checked by gate G10).
    run_labels = np.asarray([1 if str(g).startswith("attack:") else 0
                             for g in run_ids], dtype=np.int8)
    run_metrics = common.classification_metrics(run_scores, run_labels, run_threshold)
    run_ci = common.bootstrap_run_ci(run_scores, run_labels, run_threshold,
                                     n_boot=2000, seed=args.seed)

    # Diagnostics: attack runs whose sequences are all label<=0 ("ghost" runs
    # the old sequence-derived labels miscounted as normal at run level).
    # Computed on the run pool (all sequences incl. boundary).
    run_seq_label_max = {str(g): int(run_pool_labels[run_pool_groups == g].max())
                         for g in run_ids}
    attack_run_ids = [str(g) for g in run_ids if str(g).startswith("attack:")]
    attack_runs_no_spray = sum(1 for g in attack_run_ids
                               if run_seq_label_max[g] < 1)

    # grouped breakdowns at RUN level: flagged-run counts per normal workload
    # (per-workload false alarms) and per attack CVE/variant/slab (detection).
    run_normal_idx = np.asarray([str(g).startswith("normal:") for g in run_ids])
    run_attack_idx = ~run_normal_idx
    attack_cves = np.asarray([parse_group(g)[0] for g in np.asarray(run_ids)[run_attack_idx]])
    attack_variants = np.asarray(["/".join(parse_group(g))
                                  for g in np.asarray(run_ids)[run_attack_idx]])
    attack_slabs = np.asarray([common.TARGET_SLAB.get(parse_group(g)[0], "unknown")
                               for g in np.asarray(run_ids)[run_attack_idx]])
    by_workload = common.grouped_metrics(
        run_scores[run_normal_idx], run_labels[run_normal_idx],
        np.asarray([parse_group(g)[1] for g in np.asarray(run_ids)[run_normal_idx]]),
        run_threshold)
    by_cve = common.grouped_metrics(run_scores[run_attack_idx], run_labels[run_attack_idx],
                                    attack_cves, run_threshold)
    by_variant = common.grouped_metrics(run_scores[run_attack_idx], run_labels[run_attack_idx],
                                        attack_variants, run_threshold)
    by_slab = common.grouped_metrics(run_scores[run_attack_idx], run_labels[run_attack_idx],
                                     attack_slabs, run_threshold)

    evaluation_report = {
        "schema_version": 2,
        "experiment_id": experiment,
        "model": args.model,
        "score_aggregation": args.aggregation,
        "held_out_cve": args.held_out_cve,
        "run_level": run_metrics,
        "run_bootstrap_ci95": run_ci,
        "grouped": {"by_workload": by_workload, "by_cve": by_cve,
                    "by_variant": by_variant, "by_slab": by_slab},
        "counts": {
            "test_normal_sequences": int(len(test_scores)),
            "test_normal_runs": int(len(set(test_groups_arr.tolist()))),
            "attack_sequences_evaluated": int(len(attack_scores_all)),
            "attack_spray_sequences": int(np.sum(attack_seq_labels == 1)),
            "attack_normal_context_sequences": int(np.sum(attack_seq_labels == 0)),
            "ignored_boundary_sequences": int(np.sum(attack_seq_labels < 0)),
            "attack_runs_total": len(attack_run_ids),
            "attack_runs_no_spray_sequence": attack_runs_no_spray,
        },
        "inference": {
            "score_seconds": round(score_seconds, 4),
            "sequences_per_second": round(float(scored_sequences) / max(score_seconds, 1e-9), 1),
            "windows_per_second": round(float(scored_sequences * seq_len) / max(score_seconds, 1e-9), 1),
        },
    }
    if args.held_out_cve:
        # Held-out CVE run-level metrics: held attack runs paired with the same
        # eval normal runs used in the aggregate (run-level AUC is comparable).
        held_pool_scores = attack_scores_all[held_mask_full]
        held_pool_groups = attack_seq_run_ids[held_mask_full]
        if len(held_pool_scores):
            held_run_scores, _ = common.run_max_scores(held_pool_scores, held_pool_groups)
            normal_run_scores = run_scores[run_normal_idx]
            held_metrics = common.classification_metrics(
                np.concatenate((normal_run_scores, held_run_scores)),
                np.concatenate((np.zeros(len(normal_run_scores), dtype=np.int8),
                                np.ones(len(held_run_scores), dtype=np.int8))),
                run_threshold)
        else:
            held_metrics = {"error": "no held-out attack runs"}
        evaluation_report["held_out_cve_metrics"] = {
            "cve": args.held_out_cve,
            "run_level": held_metrics,
        }
    write_json(experiment_dir / "evaluation_report.json", evaluation_report)

    # ---- 6. gates G8 / G10 ---------------------------------------------------
    g8_ok = bool(run_metrics.get("roc_auc") is not None)
    gates.append({
        "name": "G8_train_evaluate_end_to_end",
        "ok": g8_ok,
        "detail": f"model={args.model} trained and evaluated; "
                  f"test normal={len(run_ids) - len(attack_run_ids)} runs / "
                  f"attack={len(attack_run_ids)} runs "
                  f"({int(np.sum(attack_seq_labels == 1))} spray sequences)",
    })

    baseline_run_flags = np.asarray(run_scores)[
        np.asarray(["/poc_cfh_baseline/" in g for g in run_ids])] > run_threshold
    n_base_runs = int(len(baseline_run_flags))
    n_baseline = int(baseline_run_flags.sum()) if n_base_runs else 0
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

    # ---- 7. metrics.csv -------------------------------------------------------
    row = {"experiment_id": experiment, "model": args.model, "seed": args.seed,
           "target_fpr": args.target_fpr, "aggregation": args.aggregation,
           "held_out_cve": args.held_out_cve or "-"}
    for key in ("roc_auc", "pr_auc", "f1_at_threshold", "fpr_at_threshold"):
        row[f"run_{key}"] = run_metrics.get(key, "")
    with open(experiment_dir / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(row))
        writer.writeheader()
        writer.writerow(row)

    # ---- 8. dataset manifest copy (traceability) ------------------------------
    src = Path(args.dataset_manifest)
    if src.exists():
        (experiment_dir / "dataset_manifest.json").write_text(src.read_text())

    log.info("done: %s", experiment_dir)
    return 0 if all(g["ok"] for g in gates) else 1


if __name__ == "__main__":
    sys.exit(main())
