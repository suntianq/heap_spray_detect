#!/usr/bin/env python3
"""CVE-composition train/test experiments (cross-CVE generalization study).

The baseline harness (run_experiment.py) splits ALL normal data into
train/val/test and evaluates on ALL attack data. This script adds explicit
control over which CVEs contribute to training vs testing, enabling four
comparison settings:

    --train-cves {A,B} --test-cves {C}   train on A+B normal, test on C (all data)
    --train-cves {A,B,C} --test-cves {A,B,C}  train on all, test on all (standard)
    --train-cves {A,B,C} --test-cves {C}  train on all, test on C only
    --train-cves {C} --test-cves {A,B}    train on C normal, test on A+B

Split semantics (leak-safe, G7-style):
  * the smallest isolated unit is the run; a run never appears in >1 partition;
  * normal runs of train-cves are split 70/15/15. The train+val slices form the
    training pool (model fit + threshold calibration). The 15% test slice of a
    train-cve is included in the evaluation set ONLY if that CVE is also a
    test-cve, otherwise it is discarded;
  * normal runs of a test-cve that is NOT in train-cves are added whole to the
    evaluation set (unseen normal behavior);
  * attack runs of test-cves are the attack half of the evaluation set.
"""

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config  # noqa: E402
from models import OCSVMDetector, TorchAEWrapper  # noqa: E402
from models.gru_detector import GRUDetector  # noqa: E402
from models.fusion_svdd import FusionSVDDDetector  # noqa: E402
from scripts.train import common  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_cve_split")

MODEL_FACTORY = {
    "ocsvm": lambda seed: OCSVMDetector(kernel="rbf", nu=0.05, gamma="scale"),
    "lstm_ae": lambda seed: TorchAEWrapper("lstm_ae", seed=seed,
                                           epochs=25, seq_batch_size=64),
    "lstm_vae": lambda seed: TorchAEWrapper("lstm_vae", seed=seed,
                                            epochs=25, seq_batch_size=64, beta=1.0),
    "gru": lambda seed: GRUDetector(seed=seed, epochs=20, batch_size=256, g=10),
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
    "fusion_svdd": {"d_model": 128, "n_layers": 2, "epochs": 20, "lr": 1e-3,
                    "g": 10, "svdd_loss_weight": 0.1, "svdd_score_weight": 0.3,
                    "vocab_size": 13824},
}

TOKEN_MODELS = {"gru", "fusion_svdd"}
DEFAULT_AGGREGATION = {"gru": "mean", "fusion_svdd": "mean"}


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def save_model(model, path):
    with open(path, "wb") as handle:
        pickle.dump(model, handle)


def cve_of(run_id):
    return str(run_id).split("/", 1)[0]


def main():
    parser = argparse.ArgumentParser(description="CVE-composition train/test experiments")
    parser.add_argument("--model", default="ocsvm", choices=sorted(MODEL_FACTORY))
    parser.add_argument("--train-cves", nargs="+", required=True,
                        help="CVEs whose NORMAL data is used for training")
    parser.add_argument("--test-cves", nargs="+", required=True,
                        help="CVEs whose data is used for evaluation (normal test + attack)")
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
    parser.add_argument("--name", default=None,
                        help="short tag appended to the experiment id, e.g. 'trainA_testC'")
    args = parser.parse_args()

    if args.aggregation is None:
        args.aggregation = DEFAULT_AGGREGATION.get(args.model, "max")

    if args.val_fraction + args.test_fraction >= 0.5:
        parser.error("val+test fractions must be < 0.5")
    if not (0 < args.target_fpr < 1):
        parser.error("--target-fpr must be in (0,1)")

    train_cves = list(dict.fromkeys(args.train_cves))
    test_cves = list(dict.fromkeys(args.test_cves))

    kind = args.name or ("cvesplit")
    experiment = common.experiment_id(kind, args.model, args.seed)
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
        "model": args.model,
        "model_config": MODEL_CONFIG[args.model],
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "target_fpr": args.target_fpr,
        "aggregation": args.aggregation,
        "train_cves": train_cves,
        "test_cves": test_cves,
        "inputs": {
            "attack_data": args.attack_data,
            "normal_data": args.normal_data,
        },
        "python_version": sys.version.split()[0],
    })

    is_token_model = args.model in TOKEN_MODELS
    if is_token_model:
        attack_tok = common.load_token_data(Path(args.attack_data))
        normal_tok = common.load_token_data(Path(args.normal_data))
        attack_seqs = attack_tok["token_seqs"]
        attack_seq_labels = np.asarray(attack_tok["token_seq_labels"], dtype=np.int8)
        attack_seq_run_ids = np.asarray(attack_tok["token_seq_run_ids"]).astype(str)
        normal_seqs = normal_tok["token_seqs"]
        normal_seq_run_ids = np.asarray(normal_tok["token_seq_run_ids"]).astype(str)
    else:
        attack, _ = common.load_processed(Path(args.attack_data))
        normal, _ = common.load_processed(Path(args.normal_data))
        if attack["sequences"].shape[-1] != normal["sequences"].shape[-1]:
            raise ValueError("attack/normal feature dimension mismatch")
        attack_seqs = attack["sequences"]
        attack_seq_labels = np.asarray(attack["seq_labels"], dtype=np.int8)
        attack_seq_run_ids = np.asarray(attack["seq_run_ids"]).astype(str)
        normal_seqs = normal["sequences"]
        normal_seq_run_ids = np.asarray(normal["seq_run_ids"]).astype(str)
    seq_len = int(normal_seqs.shape[1])

    # ---- normal run split ----------------------------------------------------
    normal_groups_all = normal_seq_run_ids
    train_pool_groups = [g for g in set(normal_groups_all.tolist())
                         if cve_of(g) in train_cves]
    train_pool_groups.sort()
    tr_groups, val_groups, held_test_groups = common.split_run_groups(
        train_pool_groups, args.seed, args.val_fraction, args.test_fraction)

    eval_normal_groups = [g for g in held_test_groups if cve_of(g) in test_cves]
    for g in sorted({x for x in normal_groups_all.tolist()}):
        if cve_of(g) in test_cves and cve_of(g) not in train_cves:
            eval_normal_groups.append(g)
    eval_normal_groups = sorted(set(eval_normal_groups))

    train_seq_mask = common.mask_for_groups(normal_groups_all, tr_groups)
    val_seq_mask = common.mask_for_groups(normal_groups_all, val_groups)
    test_seq_mask = common.mask_for_groups(normal_groups_all, eval_normal_groups)
    train_window_mask = common.mask_for_groups(
        np.asarray(normal["window_run_ids"]).astype(str), tr_groups)
    if not all(m.any() for m in (train_seq_mask, val_seq_mask, test_seq_mask, train_window_mask)):
        raise ValueError("empty partition; adjust run counts / CVE composition")

    split_payload = {
        "schema_version": 2,
        "seed": args.seed,
        "train_cves": train_cves,
        "test_cves": test_cves,
        "train_groups": tr_groups,
        "val_groups": val_groups,
        "eval_normal_groups": eval_normal_groups,
    }
    write_json(experiment_dir / "split_manifest.json", split_payload)
    g7_ok = (len(set(tr_groups) & set(val_groups)) == 0
             and len(set(tr_groups) & set(eval_normal_groups)) == 0
             and len(set(val_groups) & set(eval_normal_groups)) == 0)
    gates = [{
        "name": "G7_run_split_no_overlap",
        "ok": g7_ok,
        "detail": f"train={len(tr_groups)} val={len(val_groups)} eval_normal={len(eval_normal_groups)} "
                  f"runs, pairwise-disjoint={g7_ok}",
    }]

    # ---- scaler + model fit ------------------------------------------------
    model = MODEL_FACTORY[args.model](args.seed)
    dev = getattr(model, "_device", lambda: None)()
    log.info("model=%s device=%s", args.model, dev.type if dev is not None else "cpu")
    if is_token_model:
        train_tokens = normal_seqs[train_seq_mask].astype(np.int32)
        model.fit_sequences(train_tokens)
        del train_tokens
        log.info("model=%s trained on %d token sequences (%d runs)",
                 args.model, int(train_seq_mask.sum()), len(tr_groups))
    else:
        train_window_mask = common.mask_for_groups(
            np.asarray(normal["window_run_ids"]).astype(str), tr_groups)
        if not train_window_mask.any():
            raise ValueError("empty train window partition")
        train_windows = normal["features"][train_window_mask].astype(np.float32)
        mean, std = common.fit_scaler(train_windows)
        np.savez_compressed(experiment_dir / "scaler.npz", mean=mean, std=std)
        if getattr(model, "sequence_model", False):
            train_sequences = normal_seqs[train_seq_mask].astype(np.float32)
            model.fit_sequences(train_sequences)
            del train_sequences
        else:
            model.fit(train_windows)
        log.info("model=%s trained on %d normal runs (%d windows)", args.model,
                 len(tr_groups), len(train_windows))
    save_model(model, experiment_dir / "model.pkl")

    # ---- threshold calibration on validation-normal FPR ----------------------
    if is_token_model:
        val_sequences = normal_seqs[val_seq_mask].astype(np.int32)
    else:
        val_sequences = normal_seqs[val_seq_mask].astype(np.float32)
    val_groups_arr = normal_groups_all[val_seq_mask]
    val_seq_scores = common.score_sequences_batched(model, val_sequences, args.aggregation)
    del val_sequences
    val_run_scores, _ = common.run_max_scores(val_seq_scores, val_groups_arr)
    run_threshold = common.threshold_at_fpr(val_run_scores, args.target_fpr)
    log.info("run_threshold(p%g normal val)=%.6f", 100 * (1 - args.target_fpr), run_threshold)

    write_json(experiment_dir / "train_report.json", {
        "schema_version": 2,
        "experiment_id": experiment,
        "model": args.model,
        "train_cves": train_cves,
        "test_cves": test_cves,
        "score_aggregation": args.aggregation,
        "threshold_policy": "validation-normal run-level FPR=%s" % args.target_fpr,
        "run_threshold": run_threshold,
        "val_normal_sequences": int(len(val_seq_scores)),
        "seed": args.seed,
    })

    # ---- evaluation on test-cves ---------------------------------------------
    _t = time.perf_counter()
    test_groups_arr = normal_groups_all[test_seq_mask]
    test_scores = common.score_sequences_masked(model, normal_seqs,
                                                 test_seq_mask, args.aggregation)
    test_labels = np.zeros(len(test_scores), dtype=np.int8)

    # Score ALL sequences of test-cve attack runs: boundary sequences
    # (label -1, partial spray overlap) may contain spray events and often
    # score high, so they must count toward the run-level max. Evaluation is
    # run-level only; sequence scores are an intermediate aggregation.
    test_attack_mask_all = np.array([cve_of(g) in test_cves
                                     for g in attack_seq_run_ids])
    if is_token_model:
        attack_sequences_all = attack_seqs[test_attack_mask_all].astype(np.int32)
    else:
        attack_sequences_all = attack_seqs[test_attack_mask_all].astype(np.float32)
    attack_scores_all = common.score_sequences_batched(
        model, attack_sequences_all, args.aggregation)
    del attack_sequences_all
    score_seconds = time.perf_counter() - _t

    # Run pool includes ALL test-cve attack sequences (boundary included)
    run_pool_scores = np.concatenate((test_scores, attack_scores_all))
    run_pool_groups = np.concatenate((
        np.char.add("normal:", test_groups_arr),
        np.char.add("attack:", attack_seq_run_ids[test_attack_mask_all])))
    run_pool_labels = np.concatenate((test_labels,
                                      attack_seq_labels[test_attack_mask_all]))
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

    # grouped at RUN level: flagged-run counts per attack CVE / variant
    run_attack_idx = np.asarray([str(g).startswith("attack:") for g in run_ids])
    attack_cves = np.asarray([cve_of(g) for g in np.asarray(run_ids)[run_attack_idx]])
    by_cve = common.grouped_metrics(run_scores[run_attack_idx], run_labels[run_attack_idx],
                                    attack_cves, run_threshold)
    by_variant = common.grouped_metrics(
        run_scores[run_attack_idx], run_labels[run_attack_idx],
        np.asarray([str(g).split(":", 1)[1] for g in np.asarray(run_ids)[run_attack_idx]]),
        run_threshold)

    evaluation_report = {
        "schema_version": 2,
        "experiment_id": experiment,
        "model": args.model,
        "train_cves": train_cves,
        "test_cves": test_cves,
        "score_aggregation": args.aggregation,
        "run_level": run_metrics,
        "run_bootstrap_ci95": run_ci,
        "grouped": {"by_cve": by_cve, "by_variant": by_variant},
        "counts": {
            "test_normal_sequences": int(len(test_scores)),
            "test_normal_runs": int(len(set(test_groups_arr.tolist()))),
            "attack_sequences_evaluated": int(len(attack_scores_all)),
            "attack_spray_sequences": int(np.sum(attack_seq_labels[test_attack_mask_all] == 1)),
            "attack_normal_context_sequences": int(np.sum(attack_seq_labels[test_attack_mask_all] == 0)),
            "attack_runs_total": len(attack_run_ids),
            "attack_runs_no_spray_sequence": attack_runs_no_spray,
        },
        "inference": {
            "score_seconds": round(score_seconds, 4),
            "sequences_per_second": round(float(len(run_pool_scores)) / max(score_seconds, 1e-9), 1),
            "windows_per_second": round(float(len(run_pool_scores) * seq_len) / max(score_seconds, 1e-9), 1),
        },
    }
    write_json(experiment_dir / "evaluation_report.json", evaluation_report)

    # ---- gates G8 / G10 ------------------------------------------------------
    g8_ok = bool(run_metrics.get("roc_auc") is not None)
    gates.append({"name": "G8_train_evaluate_end_to_end", "ok": g8_ok,
                  "detail": f"model={args.model} trained and evaluated; "
                            f"test normal={len(run_ids) - len(attack_run_ids)} runs / "
                            f"attack={len(attack_run_ids)} runs "
                            f"({int(np.sum(attack_seq_labels[test_attack_mask_all] == 1))} spray sequences)"})
    baseline_flags = np.asarray(run_scores)[
        np.asarray(["/poc_cfh_baseline/" in g for g in run_ids])] > run_threshold
    n_base = int(len(baseline_flags))
    n_flagged = int(baseline_flags.sum()) if n_base else 0
    if n_base == 0:
        gates.append({"name": "G10_baseline_not_flagged", "ok": True,
                      "detail": "no baseline runs in test partition"})
    else:
        allowed = 1 if n_base <= 5 else max(1, int(round(0.3 * n_base)))
        gates.append({"name": "G10_baseline_not_flagged", "ok": n_flagged <= allowed,
                      "detail": f"{n_flagged}/{n_base} baseline runs flagged (allowed <= {allowed})"})
    write_json(experiment_dir / "gates.json", gates)

    # metrics.csv
    row = {"experiment_id": experiment, "model": args.model, "seed": args.seed,
           "train_cves": ",".join(train_cves), "test_cves": ",".join(test_cves),
           "target_fpr": args.target_fpr, "aggregation": args.aggregation}
    for key in ("roc_auc", "pr_auc", "f1_at_threshold", "fpr_at_threshold"):
        row[f"run_{key}"] = run_metrics.get(key, "")
    with open(experiment_dir / "metrics.csv", "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=sorted(row))
        w.writeheader()
        w.writerow(row)

    src = Path(args.dataset_manifest)
    if src.exists():
        (experiment_dir / "dataset_manifest.json").write_text(src.read_text())

    log.info("=== run AUC=%.4f run F1=%.4f run P=%.4f run R=%.4f (train=%s test=%s) ===",
             run_metrics.get("roc_auc", float("nan")),
             run_metrics.get("f1_at_threshold", float("nan")),
             run_metrics.get("precision_at_threshold", float("nan")),
             run_metrics.get("recall_at_threshold", float("nan")),
             ",".join(train_cves), ",".join(test_cves))
    return 0 if all(g["ok"] for g in gates) else 1


if __name__ == "__main__":
    sys.exit(main())
