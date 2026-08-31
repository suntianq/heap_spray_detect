#!/usr/bin/env python3
"""Aggregate the 5-models x 4-scenarios cross-CVE runs into one performance table.

Reads each runs/<exp>/evaluation_report.json for the 4 scenario tags and all
frozen models, and prints a markdown table of run-level AUC / Recall / Precision
/ F1 / FPR / seq AUC plus the G10 gate status (from gates.json when present).

Usage:
  python3 scripts/validate/cross_cve_aggregate.py --runs runs [--out table.md]

Requires the numpy-only build environment is fine, but sklearn is not needed:
metrics are read from the JSON reports, not recomputed.
"""

import argparse
import glob
import json
import os
import re
import sys

SCENARIOS = {
    "cveAB_testC":     ("AB→C",     "零样本"),
    "cveABC_testABC":  ("ABC→ABC",  "标准 3-CVE"),
    "cveABC_testC":    ("ABC→C",    "数据增强"),
    "cveC_testAB":     ("C→AB",     "反向迁移"),
}
MODELS = ["ocsvm", "mlp_ae", "lstm_ae", "lstm_vae", "ngram"]

DIR_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(cve[^_]+_[^_]+)_v2_([a-z_0-9]+)_s(\d+)_(\d{6})$")
# ngram experiment dirs carry the CVE numbers as a suffix:
# <date>_ngram_v2_ngram_s<seed>_<HHMMSS>_<trainNums>_test_<testNums>
NGRAM_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_ngram_v2_ngram_s(\d+)_(\d{6})_(\d+)_test_(\d+)$")
# suffix (train nums, test nums) -> scenario tag
NGRAM_TAG = {
    ("111767308", "2636"): "cveAB_testC",
    ("1117673082636", "1117673082636"): "cveABC_testABC",
    ("1117673082636", "2636"): "cveABC_testC",
    ("2636", "111767308"): "cveC_testAB",
}


def load_gates(exp_dir):
    """Return G10 verdict string from gates.json, or '?' when absent."""
    path = os.path.join(exp_dir, "gates.json")
    if not os.path.isfile(path):
        return "?"
    try:
        with open(path) as f:
            gates = json.load(f)
    except Exception:
        return "?"
    if isinstance(gates, dict):
        for key in ("G10", "G10_no_shortcut", "gates", "results"):
            if key in gates:
                gates = gates[key]
                break
    if isinstance(gates, dict):
        if "verdict" in gates:
            return gates["verdict"]
        return "✅" if gates.get("pass", False) else "❌"
    if isinstance(gates, list):  # [{name, ok}, ...]
        for entry in gates:
            if isinstance(entry, dict) and str(entry.get("name", "")).startswith("G10"):
                return "✅" if entry.get("ok") else "❌"
        return "?"
    return str(gates)[:20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", help="write markdown table to this file")
    args = ap.parse_args()

    found = {}  # (model, scenario) -> report dict
    for path in sorted(glob.glob(os.path.join(args.runs, "*"))):
        name = os.path.basename(path)
        if not os.path.isfile(os.path.join(path, "evaluation_report.json")):
            continue
        m = DIR_RE.match(name)
        if m:
            _, tag, model, _seed, _stamp = m.groups()
        else:
            nm = NGRAM_RE.match(name)
            if not nm:
                continue
            model = "ngram"
            tag = NGRAM_TAG.get((nm.group(4), nm.group(5)))
            if tag is None:
                continue
        if tag not in SCENARIOS or model not in MODELS:
            continue
        with open(os.path.join(path, "evaluation_report.json")) as f:
            rep = json.load(f)
        found[(model, tag)] = {
            "run": rep.get("run_level", {}) or {},
            "seq": rep.get("sequence_level") or {},
            "ci": rep.get("run_bootstrap_ci95", {}) or {},
            "g10": load_gates(path),
        }

    # Table 1: run-level metrics, one row per (model, scenario).
    rows = []
    headers = ["模型", "场景", "run AUC (CI95)", "seq AUC", "Recall", "Precision",
               "F1", "FPR", "G10"]
    for model in MODELS:
        for tag, (label, note) in SCENARIOS.items():
            key = (model, tag)
            if key not in found:
                rows.append([model, f"{label}({note})", "—", "—", "—", "—", "—", "—", "—"])
                continue
            rep = found[key]
            r, s = rep["run"], rep["seq"]
            ci = rep["ci"].get("roc_auc_ci95", [None, None])
            auc_ci = (f"{r.get('roc_auc', float('nan')):.4f} "
                      f"({ci[0]:.3f}–{ci[1]:.3f})" if ci[0] is not None
                      else f"{r.get('roc_auc', float('nan')):.4f}")
            seq_auc = s.get("roc_auc")
            rows.append([
                model,
                f"{label}({note})",
                auc_ci,
                f"{seq_auc:.4f}" if seq_auc is not None else "—",
                f"{r.get('recall_at_threshold', float('nan')):.3f}",
                f"{r.get('precision_at_threshold', float('nan')):.3f}",
                f"{r.get('f1_at_threshold', float('nan')):.3f}",
                f"{r.get('fpr_at_threshold', float('nan')):.3f}",
                rep["g10"],
            ])

    # Table 2: matrix AUC view (rows=scenario, cols=model) for quick scan.
    auc_rows = [["场景"] + MODELS]
    for tag, (label, note) in SCENARIOS.items():
        line = [f"{label}({note})"]
        for model in MODELS:
            key = (model, tag)
            if key in found:
                line.append(f"{found[key]['run'].get('roc_auc', float('nan')):.4f}")
            else:
                line.append("—")
        auc_rows.append(line)

    def md_table(header, body):
        widths = [max(len(str(r[i])) for r in [header] + body) for i in range(len(header))]
        fmt = " | ".join(f"{{:<{w}}}" for w in widths)
        sep = " | ".join("-" * w for w in widths)
        out = [fmt.format(*header), sep]
        out += [fmt.format(*[str(c) for c in r]) for r in body]
        return "\n".join(out)

    text = "## 五模型 × 四场景汇总（run 级）\n\n"
    text += md_table(headers, rows) + "\n\n"
    text += "### run AUC 矩阵（场景 × 模型）\n\n"
    text += md_table(auc_rows[0], auc_rows[1:]) + "\n"
    text += "\n* run AUC = run 级 max 聚合 ROC-AUC；Recall/Precision/F1/FPR 为冻结阈值\n"
    text += "下 run 级判定；seq AUC = 序列级 ROC-AUC；G10 = baseline 门禁状态。*\n"

    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"\n[wrote] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
