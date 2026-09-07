"""M5 end-to-end integration test (IMPLEMENTATION_PLAN.md 10.2).

Synthetic traces -> trace2csv -> csv2features -> train/evaluate harness, then
asserts the no-leak guarantees and determinism:
  * run split partitions are pairwise disjoint (G7);
  * the baseline model trains and evaluates end-to-end with finite metrics (G8);
  * repeating the experiment produces identical reports (deterministic hash);
  * the decision threshold is the validation-normal percentile, not a test-set
    optimum.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "train"))
from scripts.train import common

ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = ROOT / "scripts" / "train" / "run_experiment.py"
BUILD_PILOT = ROOT / "scripts" / "validate" / "build_pilot_dataset.py"
VENV_PY = ROOT / ".venv" / "bin" / "python3"


def run(cmd, cwd):
    result = subprocess.run([str(c) for c in cmd], cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def kmalloc_line(comm, pid, ts_ns, ptr, bytes_alloc):
    return ("  {}-{:<5} [000] ...1 {:.6f}: kmalloc: "
            "call_site=ffffffff8139c7f1 ptr={:016x} bytes_req={} bytes_alloc={} gfp_flags=GFP_KERNEL"
            .format(comm, pid, ts_ns / 1e9, ptr, bytes_alloc, bytes_alloc))


def kfree_line(comm, pid, ts_ns, ptr):
    return "  {}-{:<5} [000] ...1 {:.6f}: kfree: call_site=ffffffff8136ccd1 ptr={:016x}".format(
        comm, pid, ts_ns / 1e9, ptr)


def write_trace(path, background_size, spray=None, spray_start_ns=None, spray_end_ns=None):
    """Write an ftrace-style trace.log.

    background_size: allocation bytes used at ~10ms cadence over ~3.5s.
    spray: (interval_ns, alloc_bytes, count) burst written between markers when given.
    """
    lines = []
    ts = 500_000_000  # start well inside the first second
    ptr = 0x1000
    end = 4_000_000_000
    while ts < end:
        lines.append(kmalloc_line("wl", 100, ts, ptr, background_size))
        ptr += 0x100
        ts += 10_000_000
        # occasional free keeps ptr reuse realistic
        if ts % 30_000_000 == 0:
            lines.append(kfree_line("wl", 100, ts, ptr - 0x80))
    if spray:
        interval, size, count = spray
        ts = spray_start_ns
        for i in range(count):
            lines.append(kmalloc_line("poc", 200, ts, ptr, size))
            ptr += 0x100
            ts += interval
    with open(path, "w") as handle:
        for line in lines:
            handle.write(line + "\n")


def build_synthetic_root(tmp):
    """Create a CVE-first raw root with 2 attack, 4 idle, 2 churn, 3 baseline runs.

    Layout (datasets restructure): raw/<CVE>/{attack,normal,baseline}/<variant|workload>/run/...
    """
    root = Path(tmp)
    spray_start, spray_end = 2_000_000_000, 2_200_000_000
    attack_dir = root / "raw" / "CVE-SYN-ATT" / "attack" / "poc_spray"
    normal_dir = root / "raw" / "CVE-SYN" / "normal" / "idle"
    churn_dir = root / "raw" / "CVE-SYN" / "normal" / "msg_msg_256"
    base_dir = root / "raw" / "CVE-SYN" / "baseline" / "poc_cfh_baseline"

    for i in range(2):
        d = attack_dir / f"run_00{i}_synatt{i}"
        d.mkdir(parents=True)
        write_trace(d / "trace.log", 96, spray=(500_000, 512, 300),
                    spray_start_ns=spray_start, spray_end_ns=spray_end)
        with open(d / "manifest.json", "w") as f:
            json.dump({"status": "valid", "class": "attack", "cve": "CVE-SYN-ATT",
                       "variant": "poc_spray", "spray_start_ns": spray_start,
                       "spray_end_ns": spray_end}, f)

    for i in range(4):
        d = normal_dir / f"run_00{i}_syn{i}"
        d.mkdir(parents=True)
        write_trace(d / "trace.log", 96)
        with open(d / "manifest.json", "w") as f:
            json.dump({"status": "valid", "class": "normal", "cve": "CVE-SYN",
                       "workload": "idle"}, f)

    for i in range(2):
        d = churn_dir / f"run_00{i}_syn{i}"
        d.mkdir(parents=True)
        write_trace(d / "trace.log", 96)
        with open(d / "manifest.json", "w") as f:
            # churn workloads are normal-collector runs (spray-shaped allocation
            # storms); the harness holds them out by the msg_msg_* run_id segment.
            json.dump({"status": "valid", "class": "normal", "cve": "CVE-SYN",
                       "workload": "msg_msg_256"}, f)

    for i in range(3):
        d = base_dir / f"run_00{i}_syn{i}"
        d.mkdir(parents=True)
        write_trace(d / "trace.log", 96)
        with open(d / "manifest.json", "w") as f:
            # baseline PoCs are collected by the attack collector -> class=attack,
            # and flipped to normal by build_run_meta via /poc_cfh_baseline/.
            json.dump({"status": "valid", "class": "attack", "cve": "CVE-SYN",
                       "variant": "poc_cfh_baseline", "workload": "poc_cfh_baseline"}, f)

    return root / "raw"


class TestM5Pipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m5_pipe_")
        self.raw = Path(self.tmp) / "raw"
        self.out = Path(self.tmp) / "pilot"
        build_synthetic_root(self.tmp)
        self.run_pipeline()

    def run_pipeline(self):
        out = self.out
        # New build entry point (datasets restructure): --raw (CVE-first root)
        # -> processed/{attack,normal} + dataset_manifest.json; csv staging is
        # transient (.tmp) and removed by the build.
        # --skip-gates: synthetic traces cannot satisfy the data-quality gates
        # (G3 empty windows, G5 PoC comm, G9 duration overlap); the M5 test
        # validates the model-level gates (G7/G8/G10) via run_experiment.
        run([sys.executable, BUILD_PILOT, "--raw", self.raw, "--out", out,
             "--skip-gates"], cwd=ROOT)

        proc_attack = out / "processed" / "attack"
        proc_normal = out / "processed" / "normal"

        runs_dir = Path(self.tmp) / "runs"
        run([VENV_PY, RUN_EXPERIMENT, "--model", "ocsvm",
             "--attack-data", proc_attack, "--normal-data", proc_normal,
             "--out", runs_dir], cwd=ROOT)
        self.experiments = sorted(p for p in runs_dir.iterdir() if p.is_dir())

    def test_end_to_end_artifacts_and_gates(self):
        self.assertEqual(len(self.experiments), 1)
        exp = self.experiments[0]
        for name in ("split_manifest.json", "scaler.npz", "model.pkl",
                     "train_report.json", "evaluation_report.json",
                     "metrics.csv", "gates.json"):
            self.assertTrue((exp / name).exists(), f"missing {name}")
        gates = json.loads((exp / "gates.json").read_text())
        for gate in gates:
            self.assertTrue(gate["ok"], f"gate failed: {gate['name']}: {gate['detail']}")

        report = json.loads((exp / "evaluation_report.json").read_text())
        self.assertNotIn("sequence_level", report)  # run-level-only evaluation
        run = report["run_level"]
        self.assertTrue(all(k in run for k in ("roc_auc", "pr_auc", "f1_at_threshold",
                                               "fpr_at_threshold")))
        self.assertTrue(np.isfinite(run["roc_auc"]))
        # the synthetic spray is dense/large enough that the baseline must beat chance
        self.assertGreater(run["roc_auc"], 0.5)

    def test_run_split_disjoint(self):
        split = json.loads((self.experiments[0] / "split_manifest.json").read_text())
        for a, b in (("train_groups", "val_groups"), ("train_groups", "test_groups"),
                     ("val_groups", "test_groups")):
            self.assertEqual(set(split[a]) & set(split[b]), set(), f"{a}/{b} overlap")

    def test_churn_family_held_out(self):
        """Retired msg_msg/keyctl runs are held out like baseline: excluded from
        train/val/test, scored at the frozen threshold as churn_near_attack (G11)."""
        exp = self.experiments[0]
        split = json.loads((exp / "split_manifest.json").read_text())
        report = json.loads((exp / "evaluation_report.json").read_text())
        gates = json.loads((exp / "gates.json").read_text())
        churn_ids = split["churn_held_out"]
        self.assertEqual(len(churn_ids), 2)
        self.assertTrue(all("/msg_msg_256/" in g for g in churn_ids))
        for pool in ("train_groups", "val_groups", "test_groups"):
            self.assertEqual(set(split[pool]) & set(churn_ids), set(),
                             f"churn run leaked into {pool}")
        churn = report["churn_near_attack"]
        self.assertEqual(churn["runs_total"], 2)
        self.assertTrue(0 <= churn["runs_flagged"] <= churn["runs_total"])
        self.assertIn(round(churn["detection_rate"] * 2), (0, 1, 2))
        g11 = [g for g in gates if g["name"] == "G11_churn_held_out"]
        self.assertEqual(len(g11), 1)
        self.assertTrue(g11[0]["ok"])
        # baseline behaviour unchanged: still held out and reported (G10)
        self.assertEqual(report["baseline_near_attack"]["runs_total"], 3)
        self.assertTrue(any(g["name"] == "G10_baseline_held_out" for g in gates))

    def test_threshold_is_validation_percentile_not_test_optimum(self):
        exp = self.experiments[0]
        train = json.loads((exp / "train_report.json").read_text())
        report = json.loads((exp / "evaluation_report.json").read_text())
        run_threshold = train["run_threshold"]
        # run_threshold must equal the p(1-fpr) percentile of the calibration
        # scores; recompute it from the val set through the saved model to
        # prove no test information entered calibration.
        normal = np.load(self.out / "processed" / "normal" / "features.npz",
                         allow_pickle=True)
        split = json.loads((exp / "split_manifest.json").read_text())
        val_groups = set(split["val_groups"])
        val_mask = np.isin(normal["seq_run_ids"].astype(str), list(val_groups))
        val_seq = normal["sequences"][val_mask].astype(np.float32)
        val_run_ids = normal["seq_run_ids"].astype(str)[val_mask]
        # Unpickle the ocsvm model and replay the harness score path:
        # anomaly_score per window, "max" aggregation over the sequence,
        # then max over each val run's sequences (run-level scores).
        # (The venv interpreter has torch, so unpickling through the models
        # package is safe.)
        import pickle
        with open(exp / "model.pkl", "rb") as fh:
            model = pickle.load(fh)
        n, t, f = val_seq.shape
        per_window = np.asarray(model.anomaly_score(val_seq.reshape(n * t, f)),
                                dtype=np.float64).reshape(n, t)
        val_seq_scores = per_window.max(axis=1)  # ocsvm, "max" seq aggregation
        val_run_scores, _ = common.run_max_scores(val_seq_scores, val_run_ids)
        expected = common.threshold_at_fpr(val_run_scores, common.DEFAULT_TARGET_FPR)
        self.assertAlmostEqual(run_threshold, expected, places=4)
        # oracle best-F1 is the max over ALL thresholds, so it must be >= the
        # F1 at the frozen (val-calibrated) threshold. (The old "F1 < 1.0"
        # heuristic assumed imperfect separability; ocsvm separates the
        # synthetic spray perfectly, so F1 can legitimately be 1.0.)
        self.assertGreaterEqual(
            report["run_level"]["oracle_best_f1_test_only"],
            report["run_level"]["f1_at_threshold"] - 1e-9)

    def test_deterministic(self):
        """Same inputs + config -> identical metrics.csv and evaluation report
        (modulo the per-invocation experiment_id time stamp and inference
        timing, which varies run to run)."""
        exp = self.experiments[0]
        run([VENV_PY, RUN_EXPERIMENT, "--model", "ocsvm",
             "--attack-data", self.out / "processed" / "attack",
             "--normal-data", self.out / "processed" / "normal",
             "--out", exp.parent], cwd=ROOT)
        exp2 = sorted(p for p in exp.parent.iterdir() if p.is_dir() and p != exp)[0]

        def load_metrics(path):
            with open(path, newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row.pop("experiment_id", None)
            return rows

        def load_report(path):
            report = json.loads(path.read_text())
            report.pop("experiment_id", None)
            report.pop("inference", None)  # wall-clock timing, not deterministic
            return report

        self.assertEqual(load_metrics(exp / "metrics.csv"),
                         load_metrics(exp2 / "metrics.csv"))
        self.assertEqual(load_report(exp / "evaluation_report.json"),
                         load_report(exp2 / "evaluation_report.json"))

    def test_no_nan_inf_in_metrics(self):
        metrics = (self.experiments[0] / "metrics.csv").read_text()
        self.assertNotIn("nan", metrics.lower())
        self.assertNotIn("inf", metrics.lower())


class ControlSplitTest(unittest.TestCase):
    """Unit tests for the two near-attack control families (baseline + churn)."""

    def test_is_churn_run_matches_retired_workloads(self):
        for rid in ("CVE-2017-11176/msg_msg_256/run_000_x/trace",
                    "CVE-2017-11176/msg_msg_2048/run_001_x/trace",
                    "CVE-2017-11176/msg_msg/run_002_x/trace",
                    "CVE-2017-11176/keyctl/run_003_x/trace"):
            self.assertTrue(common.is_churn_run(rid), rid)
        for workload in ("idle", "business", "net_busy", "fs_io",
                         "fork_stress", "mem_pressure"):
            rid = f"CVE-2017-11176/{workload}/run_000_x/trace"
            self.assertFalse(common.is_churn_run(rid), rid)
        # attack/baseline variants never look like churn
        self.assertFalse(common.is_churn_run("CVE-2017-11176/poc_cfh_baseline/run_000_x/trace"))
        self.assertFalse(common.is_churn_run("CVE-2017-11176/poc_cfh_single_spray/run_000_x/trace"))
        self.assertFalse(common.is_churn_run("CVE-2017-11176/poc_cfh_combo/run_000_x/trace"))

    def test_split_control_groups_three_way(self):
        groups = np.array([
            "CVE-A/idle/run_000_x/trace",
            "CVE-A/msg_msg_256/run_001_x/trace",
            "CVE-A/keyctl/run_002_x/trace",
            "CVE-A/poc_cfh_baseline/run_003_x/trace",
        ])
        base, churn, pure = common.split_control_groups(groups)
        self.assertEqual(int(base.sum()), 1)
        self.assertEqual(int(churn.sum()), 2)
        self.assertEqual(int(pure.sum()), 1)
        # disjoint and exhaustive
        self.assertFalse((base & churn).any())
        self.assertFalse((base & pure).any())
        self.assertFalse((churn & pure).any())
        self.assertEqual(int((base | churn | pure).sum()), len(groups))

    def test_split_control_groups_baseline_wins_over_churn(self):
        # a poc_cfh_baseline id can never be claimed by the churn family
        groups = np.array(["CVE-A/poc_cfh_baseline/run_000_x/trace"])
        base, churn, pure = common.split_control_groups(groups)
        self.assertEqual(int(base.sum()), 1)
        self.assertEqual(int(churn.sum()), 0)
        self.assertEqual(int(pure.sum()), 0)


class AggregateScoresTest(unittest.TestCase):
    """The top-K mean must expose short bursts that a single order statistic hides."""

    def test_topk_exposes_short_burst_p90_misses(self):
        scores = np.array([
            [1.0] * 128,                      # flat background
            [1.0] * 123 + [50.0] * 5,         # 5-position burst in 128
        ])
        # p90 (rank ~14 from top) cannot see a 5-position burst
        self.assertAlmostEqual(common.aggregate_scores(scores, "p90")[1], 1.0, places=6)
        # topk8 lets the burst occupy 5 of 8 slots without touching flat noise
        topk = common.aggregate_scores(scores, "topk8")
        self.assertAlmostEqual(topk[0], 1.0, places=6)
        self.assertAlmostEqual(topk[1], (5 * 50.0 + 3 * 1.0) / 8, places=6)

    def test_topk_larger_than_sequence_falls_back_to_mean(self):
        scores = np.array([[3.0, 1.0, 2.0]])
        self.assertAlmostEqual(common.aggregate_scores(scores, "topk10")[0], 2.0, places=6)

    def test_topk_invalid_k_rejected(self):
        with self.assertRaises(ValueError):
            common.aggregate_scores(np.zeros((1, 4)), "topk0")
        with self.assertRaises(ValueError):
            common.aggregate_scores(np.zeros((1, 4)), "topkx")


if __name__ == "__main__":
    unittest.main()
