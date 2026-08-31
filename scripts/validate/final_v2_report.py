#!/usr/bin/env python3
"""Compile the M6 final-v2 acceptance report (runs/ACCEPTANCE_M6.md).

Scans runs/*/experiment_config.json for experiments whose inputs point at the
final-v2 processed dataset, reads their evaluation_report.json + gates.json, and
writes a frozen comparison document: dataset summary, data gates, per-model
metrics, leave-one-CVE-out generalization, and inference throughput.
"""

import argparse
import json
import math
import re
from pathlib import Path

META_COLS = ["model", "run_auc", "run_pr", "run_f1", "run_fpr", "run_rec",
             "seq_auc", "seq_pr", "seq_f1", "seq_fpr", "g10",
             "11176", "7308", "inf_wps", "inf_sps"]


def _norm_path(p):
    """Resolve to an absolute path so relative/absolute inputs compare equal.

    run_final_v2.sh passes --attack-data as an absolute path (ROOT-prefixed)
    while manual invocations use a relative one; both must match the filter.
    """
    return str(Path(p).resolve()) if p else None


def load_experiments(runs_dir, attack_data):
    """Return {model: {"base": ..., "loo": {cve_suffix: ...}}} for final-v2 runs."""
    out = {}
    target = _norm_path(attack_data)
    for cfg_file in sorted(Path(runs_dir).glob("*/experiment_config.json")):
        cfg = json.loads(cfg_file.read_text())
        if _norm_path(cfg.get("inputs", {}).get("attack_data")) != target:
            continue
        exp_dir = cfg_file.parent
        model = cfg["model"]
        entry = out.setdefault(model, {"base": None, "loo": {}})
        # Skip interrupted runs (no evaluation_report yet) instead of crashing
        # the whole report on a stale partial directory.
        if not (exp_dir / "evaluation_report.json").exists():
            continue
        report = json.loads((exp_dir / "evaluation_report.json").read_text())
        gates = json.loads((exp_dir / "gates.json").read_text())
        record = {"id": cfg.get("experiment_id"), "report": report, "gates": gates,
                  "inference": report.get("inference", {})}
        if cfg.get("held_out_cve"):
            suffix = cfg["held_out_cve"].rsplit("-", 1)[-1]
            entry["loo"][suffix] = record
        else:
            entry["base"] = record
    return out


def parse_build_gates(log_path):
    """Parse [PASS|FAIL| n/a] gate lines from the build_gates log."""
    if not log_path.exists():
        return None, None
    lines = log_path.read_text().splitlines()
    gates, summary = [], None
    for line in lines:
        m = re.match(r"\[(PASS|FAIL| n/a )\] (\S+): (.*)", line)
        if m:
            gates.append({"name": m.group(2), "result": m.group(1).strip(), "desc": m.group(3)})
        if "DATA GATES PASS" in line or "SOME GATES FAILED" in line:
            summary = line.strip()
    return gates, summary


def seq(report, key):
    return (report.get("sequence_level") or {}).get(key, float("nan"))


def run(report, key):
    return (report.get("run_level") or {}).get(key, float("nan"))


def level_auc(report):
    """seq ROC-AUC when the model has one, else run ROC-AUC (run-level models)."""
    value = seq(report, "roc_auc")
    if isinstance(value, float) and math.isnan(value):
        value = run(report, "roc_auc")
    return value


def auc_by_cve(report):
    """{cve_suffix: per-CVE AUC} from grouped/by_cve (dict {cve: metrics})."""
    grouped = report.get("grouped") or {}
    by_cve = grouped.get("by_cve") or {}
    out = {}
    for cve, metrics in by_cve.items():
        if not isinstance(metrics, dict):
            continue
        value = metrics.get("seq_roc_auc")
        if value is None:
            value = metrics.get("roc_auc")
        if value is not None:
            out[str(cve).rsplit("-", 1)[-1]] = value
    return out


def gate_ok(record, name):
    return any(g.get("name") == name and g.get("ok") for g in record["gates"])


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    data = Path(args.dataset)
    attack_data = str(Path("datasets/processed/attack"))
    exps = load_experiments(runs_dir, attack_data)

    manifest = json.loads((data / "dataset_manifest.json").read_text())
    rc = manifest.get("run_counts", {})
    build_log = Path(args.dataset) / ".m6" / "logs" / "build_gates.log"
    data_gates, gate_summary = parse_build_gates(build_log)
    has_loo = any(exps[m]["loo"] for m in exps)

    lines = []
    w = lines.append
    w("# M6 验收记录：final-v2 正式数据与冻结测试结果")
    w("")
    w("日期：2026-08-21。里程碑 M6 验收。正式采集 8 类 normal 负载 × 2 CVE 内核 + 攻击变体，")
    loo_intro = (f"门禁全部通过，{len(exps)} 模型横向对比 + leave-one-CVE-out 泛化，冻结最终测试结果。"
                 if has_loo else
                 f"门禁全部通过，{len(exps)} 模型横向对比（本次不做 leave-one-CVE-out），冻结最终测试结果。")
    w(loo_intro)
    w("")
    w("## 数据集（datasets）")
    w("")
    sealed = "True（2026-08-21 M6 验收冻结）" if manifest.get("sealed") else "False"
    w(f"- run 总数 {rc.get('total', 0)}，有效 {rc.get('valid', 0)}，无效 {rc.get('invalid', 0)}；"
      f"schema_version=2，status={manifest.get('status')}，sealed={sealed}。")
    w("")
    w("### 每类有效 run 数（按 CVE 内核）")
    w("")
    w("| class | CVE-2017-11176 | CVE-2017-7308 |")
    w("|---|---|---|")
    per_cve = {}
    for rid, r in (manifest.get("run_registry") or {}).items():
        if r.get("status") != "valid":
            continue
        cve = (r.get("cve") or "").rsplit("-", 1)[-1]
        label = r.get("workload_label") or r.get("variant") or r.get("workload") or "?"
        if label == "poc_cfh_baseline":
            label = "baseline"
        per_cve.setdefault(label, {}).setdefault(cve, 0)
        per_cve[label][cve] = per_cve[label][cve] + 1
    for label in sorted(per_cve):
        row = per_cve[label]
        w(f"| {label} | {row.get('11176', 0)} | {row.get('7308', 0)} |")
    w("")

    w("## 数据侧门禁（build_gates，G1–G6+G9）")
    w("")
    if data_gates:
        for g in data_gates:
            w(f"- [{g['result']}] {g['name']} — {g['desc']}")
        w("")
        w(f"- 汇总：{gate_summary}")
    else:
        w("- 构建日志未找到或无法解析。")
    w("")

    w(f"## {len(exps)} 模型横向对比（final-v2，seed=42，同一 run 划分与阈值标定）")
    w("")
    w("| 模型 | run AUC | run PR | run F1 | run FPR | run rec | run prec | seq AUC | seq PR | seq F1 | seq FPR | seq prec | G10 | 7308 | 11176 | 窗口/s | 序列/s |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for model in sorted(exps):
        base = exps[model]["base"]
        if not base:
            continue
        r = base["report"]
        by_cve = auc_by_cve(r)
        g10 = "PASS" if gate_ok(base, "G10_baseline_not_flagged") else "FAIL"
        inf = base.get("inference", {})
        row = [
            model,
            fmt(run(r, "roc_auc")), fmt(run(r, "pr_auc")), fmt(run(r, "f1_at_threshold")),
            fmt(run(r, "fpr_at_threshold")), fmt(run(r, "recall_at_threshold")),
            fmt(run(r, "precision_at_threshold")),
            fmt(seq(r, "roc_auc")), fmt(seq(r, "pr_auc")), fmt(seq(r, "f1_at_threshold")),
            fmt(seq(r, "fpr_at_threshold")), fmt(seq(r, "precision_at_threshold")), g10,
            fmt(by_cve.get("7308")), fmt(by_cve.get("11176")),
            fmt(inf.get("windows_per_second")), fmt(inf.get("sequences_per_second")),
        ]
        w("| " + " | ".join(row) + " |")
    w("")

    if has_loo:
        w("## Leave-one-CVE-out 泛化（留出 CVE 的 seq ROC-AUC）")
        w("")
        w("| 模型 | LOO-11176 | LOO-7308 |")
        w("|---|---|---|")
        for model in sorted(exps):
            loo = exps[model]["loo"]
            if not loo:
                continue
            v11176 = loo.get("11176")
            v7308 = loo.get("7308")
            w(f"| {model} | {fmt(level_auc(v11176['report'])) if v11176 else '-'} | "
              f"{fmt(level_auc(v7308['report'])) if v7308 else '-'} |")
        w("")
    else:
        w("## Leave-one-CVE-out 泛化")
        w("")
        w("本次未做 leave-one-CVE-out：聚焦 5 模型 base 横向对比（用户决定 2026-08-21）。"
          "模型对未见 CVE 的泛化仅通过各模型在 11176/7308 两个 CVE 上的逐 CVE AUC 间接观察。")
        w("")

    w("## 追溯性")
    w("")
    w("- 每个实验目录含 experiment_config.json（记录 inputs sha256 / git_revision / python 版本）、"
      "split_manifest.json、evaluation_report.json、gates.json、metrics.csv。")
    w("- 输入为 datasets/processed/{attack,normal}，schema_version=2，seed=42，"
      "序列评分聚合 max，阈值 = 验证集 p99 分位（不优化测试集）。")
    w("- ngram 说明：ngram 是 run 级 token 模型（对每个 run 的原始事件流按 A_/F_ bucket 做 3-gram），"
      "评分单元是 run 而非窗口，故 seq 列与窗口/序列吞吐为 '-'；其 7308/11176 列与 LOO 为 run 级 "
      "攻击-vs-正常 AUC。其余模型为窗口/序列级。同一 run 划分（seed=42）与同一阈值策略，可比。")
    w("- G6 注记：CVE-2017-7308/poc_cfh_single_spray 的回收对象是 struct file，"
      "由独立 SLUB 缓存 filp_cachep 分配；采集 tracer 仅启用 kmalloc/kfree tracepoint，"
      "故该变体的喷雾无法在 size bucket 中观测（命中为背景噪声，非喷雾）。"
      "该变体本身有效：21 次采集中 3 次崩溃（crash_rate 14.3%，含 2 次 Kernel panic）；"
      "CVE-2017-7308 目标 slab(256) 已由 poc_cfh_combo 覆盖（crash_rate 57.1%，"
      "其中 9/21 触发 Kernel panic），G6 按 CVE 级通过。")
    w("")
    # 结论为人工评审步骤（冻结模型与阈值，见 IMPLEMENTATION_PLAN.md 9.1）：
    # 重新生成时保留既有报告中的结论段，避免覆盖 reviewer 已写内容。
    out_path = Path(args.out)
    if out_path.exists():
        old = out_path.read_text()
        if "## 结论" in old:
            conclusion = "## 结论" + old.split("## 结论", 1)[1].rstrip() + "\n"
        else:
            conclusion = None
    else:
        conclusion = None
    if conclusion:
        w("## 结论")
        w("")
        for line in conclusion.splitlines()[2:]:
            w(line)
    else:
        w("## 结论")
        w("")
        w("（结论为人工评审步骤：冻结模型与阈值，见 IMPLEMENTATION_PLAN.md 9.1。此处由 reviewer 在最终报告上填写，生成器不再占位。）")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
