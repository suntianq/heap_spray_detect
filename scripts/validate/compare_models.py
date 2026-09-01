#!/usr/bin/env python3
"""Compare multiple model experiments in a single CSV table.

Scans runs/*/experiment_config.json + evaluation_report.json, extracts
core metrics at both sequence and run level, and writes a flat CSV.

The project has two evaluation levels:
  - seq level: every 128-token sequence is scored independently. Precision is
    low because attack runs have hundreds of "background" (non-spray) sequences
    labeled 0 that can still get high anomaly scores.
  - run level: each run's score = max over its sequences. This is the real
    detection unit — "did this run contain a spray attack?". A run is flagged
    if any one sequence exceeds threshold.

Usage:
    python3 scripts/validate/compare_models.py --runs runs --out results/model_comparison.csv

Optional filters:
    --model-filter ocsvm gru fusion   only include these models
"""
import argparse
import csv
import json
import os
from pathlib import Path


def load_experiments(runs_dir, model_filter=None):
    """Scan runs dir, return list of experiment dicts."""
    experiments = []
    for cfg_file in sorted(Path(runs_dir).glob("*/experiment_config.json")):
        try:
            cfg = json.loads(cfg_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        exp_dir = cfg_file.parent
        report_path = exp_dir / "evaluation_report.json"
        if not report_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        model = cfg.get("model", "?")
        if model_filter and model not in model_filter:
            continue

        experiments.append({
            "dir": exp_dir.name,
            "model": model,
            "seed": cfg.get("seed", ""),
            "held_out_cve": cfg.get("held_out_cve") or "-",
            "report": report,
        })
    return experiments


def safe(d, key, fmt="{:.4f}"):
    if d is None:
        return ""
    v = d.get(key)
    if v is None or (isinstance(v, float) and (v != v)):
        return ""
    return fmt.format(v) if isinstance(v, (int, float)) else str(v)


def main():
    parser = argparse.ArgumentParser(description="Compare model results in CSV")
    parser.add_argument("--runs", required=True, help="runs directory")
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument("--model-filter", nargs="*", default=None,
                        help="only these model names (e.g. ocsvm gru fusion)")
    args = parser.parse_args()

    experiments = load_experiments(
        args.runs, set(args.model_filter) if args.model_filter else None)
    if not experiments:
        print("No experiments found (with evaluation_report.json) matching filters.")
        return 1

    fieldnames = [
        "model",
        "seed",
        "held_out_cve",
        # run level (real detection unit)
        "run_auc",
        "run_precision",
        "run_recall",
        "run_f1",
        "run_fpr",
        # sequence level (per-sequence granularity)
        "seq_auc",
        "seq_precision",
        "seq_recall",
        "seq_f1",
        "seq_fpr",
        # data counts
        "test_normal_seqs",
        "attack_spray_seqs",
        "attack_bg_seqs",
    ]

    rows = []
    for exp in experiments:
        report = exp["report"]
        run = report.get("run_level") or {}
        seq = report.get("sequence_level") or {}
        counts = report.get("counts") or {}
        row = {
            "model": exp["model"],
            "seed": exp["seed"],
            "held_out_cve": exp["held_out_cve"],
            "run_auc": safe(run, "roc_auc"),
            "run_precision": safe(run, "precision_at_threshold"),
            "run_recall": safe(run, "recall_at_threshold"),
            "run_f1": safe(run, "f1_at_threshold"),
            "run_fpr": safe(run, "fpr_at_threshold"),
            "seq_auc": safe(seq, "roc_auc"),
            "seq_precision": safe(seq, "precision_at_threshold"),
            "seq_recall": safe(seq, "recall_at_threshold"),
            "seq_f1": safe(seq, "f1_at_threshold"),
            "seq_fpr": safe(seq, "fpr_at_threshold"),
            "test_normal_seqs": safe(counts, "test_normal_sequences", "{}"),
            "attack_spray_seqs": safe(counts, "attack_spray_sequences", "{}"),
            "attack_bg_seqs": safe(counts, "attack_normal_context_sequences", "{}"),
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["model"], r["held_out_cve"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path} ({len(rows)} experiments)")
    print()
    print(f"{'model':12s} {'run_AUC':>8s} {'run_P':>7s} {'run_R':>7s} {'run_F1':>7s} {'run_FPR':>8s}"
          f" {'seq_AUC':>8s} {'seq_P':>7s} {'seq_R':>7s} {'seq_F1':>7s}")
    print("-" * 90)
    for r in rows:
        print(f"{r['model']:12s} {r['run_auc']:>8s} {r['run_precision']:>7s} {r['run_recall']:>7s} "
              f"{r['run_f1']:>7s} {r['run_fpr']:>8s} {r['seq_auc']:>8s} {r['seq_precision']:>7s} "
              f"{r['seq_recall']:>7s} {r['seq_f1']:>7s}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
