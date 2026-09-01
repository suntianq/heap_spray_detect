#!/usr/bin/env python3
"""Compare multiple model experiments in a single CSV table.

Scans runs/*/experiment_config.json + evaluation_report.json, extracts
common metrics (precision, recall, F1, FPR, AUC, PR-AUC) at both sequence
and run level, and writes a flat CSV. One row per experiment.

Usage:
    python3 scripts/validate/compare_models.py --runs runs --out runs/model_comparison.csv

Optional filters:
    --attack-data PATH    only include experiments whose attack_data matches
    --model-filter NAME   only include specific models (repeatable)
"""
import argparse
import csv
import json
import os
from pathlib import Path


def load_experiments(runs_dir, attack_data_filter=None, model_filter=None):
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
            continue  # skip interrupted runs
        try:
            report = json.loads(report_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        # Filter by attack_data path if specified
        if attack_data_filter:
            actual = cfg.get("inputs", {}).get("attack_data", "")
            if os.path.basename(actual) != os.path.basename(attack_data_filter) \
               and os.path.abspath(actual) != os.path.abspath(attack_data_filter):
                continue

        model = cfg.get("model", "?")
        if model_filter and model not in model_filter:
            continue

        experiments.append({
            "dir": exp_dir.name,
            "model": model,
            "seed": cfg.get("seed", ""),
            "held_out_cve": cfg.get("held_out_cve") or "-",
            "aggregation": report.get("score_aggregation", "-"),
            "report": report,
        })
    return experiments


def extract_metrics(report):
    """Extract all metrics from an evaluation_report into a flat dict."""
    seq = report.get("sequence_level", {})
    run = report.get("run_level", {})
    ci = report.get("run_bootstrap_ci95", {})
    inf = report.get("inference", {})
    counts = report.get("counts", {})

    def safe(d, key, fmt="{:.4f}"):
        if d is None:
            return ""
        v = d.get(key)
        if v is None or (isinstance(v, float) and (v != v)):  # NaN check
            return ""
        return fmt.format(v) if isinstance(v, (int, float)) else str(v)

    return {
        # sequence level
        "seq_auc": safe(seq, "roc_auc"),
        "seq_pr_auc": safe(seq, "pr_auc"),
        "seq_precision": safe(seq, "precision_at_threshold"),
        "seq_recall": safe(seq, "recall_at_threshold"),
        "seq_f1": safe(seq, "f1_at_threshold"),
        "seq_fpr": safe(seq, "fpr_at_threshold"),
        "seq_threshold": safe(seq, "threshold", "{:.6f}"),
        "seq_flagged": safe(seq, "flagged", "{}"),
        "seq_count": safe(seq, "count", "{}"),
        "seq_oracle_f1": safe(seq, "oracle_best_f1_test_only"),
        # run level
        "run_auc": safe(run, "roc_auc"),
        "run_pr_auc": safe(run, "pr_auc"),
        "run_precision": safe(run, "precision_at_threshold"),
        "run_recall": safe(run, "recall_at_threshold"),
        "run_f1": safe(run, "f1_at_threshold"),
        "run_fpr": safe(run, "fpr_at_threshold"),
        "run_threshold": safe(run, "threshold", "{:.6f}"),
        "run_flagged": safe(run, "flagged", "{}"),
        "run_count": safe(run, "count", "{}"),
        "run_oracle_f1": safe(run, "oracle_best_f1_test_only"),
        # bootstrap CI
        "run_auc_ci95_low": safe(ci, "roc_auc_ci95", "[{:.4f}, {:.4f}]") if isinstance(ci.get("roc_auc_ci95"), list) else "",
        "run_f1_ci95": safe(ci, "f1_ci95", "[{:.4f}, {:.4f}]") if isinstance(ci.get("f1_ci95"), list) else "",
        # inference
        "inf_score_seconds": safe(inf, "score_seconds", "{:.2f}"),
        "inf_seqs_per_sec": safe(inf, "sequences_per_second", "{:.1f}"),
        # counts
        "test_normal_seqs": safe(counts, "test_normal_sequences", "{}"),
        "attack_spray_seqs": safe(counts, "attack_spray_sequences", "{}"),
        "attack_bg_seqs": safe(counts, "attack_normal_context_sequences", "{}"),
    }


def fmt_ci(ci_list):
    if isinstance(ci_list, list) and len(ci_list) == 2:
        return f"[{ci_list[0]:.4f}, {ci_list[1]:.4f}]"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Compare model results in CSV")
    parser.add_argument("--runs", required=True, help="runs directory")
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument("--attack-data", default=None,
                        help="filter: only experiments with this attack_data path")
    parser.add_argument("--model-filter", nargs="*", default=None,
                        help="filter: only these model names (e.g. ocsvm gru fusion)")
    args = parser.parse_args()

    experiments = load_experiments(args.runs, args.attack_data, set(args.model_filter) if args.model_filter else None)
    if not experiments:
        print("No experiments found (with evaluation_report.json) matching filters.")
        return 1

    # Build rows
    fieldnames = [
        "experiment_dir", "model", "seed", "held_out_cve", "aggregation",
        # sequence level
        "seq_auc", "seq_pr_auc", "seq_precision", "seq_recall",
        "seq_f1", "seq_fpr", "seq_threshold", "seq_flagged", "seq_count", "seq_oracle_f1",
        # run level
        "run_auc", "run_pr_auc", "run_precision", "run_recall",
        "run_f1", "run_fpr", "run_threshold", "run_flagged", "run_count", "run_oracle_f1",
        # CI
        "run_auc_ci95", "run_f1_ci95",
        # inference
        "inf_score_seconds", "inf_seqs_per_sec",
        # counts
        "test_normal_seqs", "attack_spray_seqs", "attack_bg_seqs",
    ]

    rows = []
    for exp in experiments:
        m = extract_metrics(exp["report"])
        # Fix CI fields (extract_metrics returns them under different keys)
        ci = exp["report"].get("run_bootstrap_ci95", {})
        m["run_auc_ci95"] = fmt_ci(ci.get("roc_auc_ci95"))
        m["run_f1_ci95"] = fmt_ci(ci.get("f1_ci95"))
        row = {
            "experiment_dir": exp["dir"],
            "model": exp["model"],
            "seed": exp["seed"],
            "held_out_cve": exp["held_out_cve"],
            "aggregation": exp["aggregation"],
        }
        row.update({k: m.get(k, "") for k in fieldnames if k not in row})
        rows.append(row)

    # Sort: by model name, then held_out_cve
    rows.sort(key=lambda r: (r["model"], r["held_out_cve"]))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out} ({len(rows)} experiments)")
    print()
    # Print a quick summary table to stdout
    print(f"{'model':12s} {'run_AUC':>8s} {'run_P':>7s} {'run_R':>7s} {'run_F1':>7s} {'run_FPR':>8s} {'seq_AUC':>8s} {'seq_F1':>7s}")
    print("-" * 75)
    for r in rows:
        print(f"{r['model']:12s} {r['run_auc']:>8s} {r['run_precision']:>7s} {r['run_recall']:>7s} "
              f"{r['run_f1']:>7s} {r['run_fpr']:>8s} {r['seq_auc']:>8s} {r['seq_f1']:>7s}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
