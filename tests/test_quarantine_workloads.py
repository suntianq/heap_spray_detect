"""Tests for scripts/validate/quarantine_workloads.py.

Covers the raw-tree migration contract: dry-run makes no changes, --apply
moves churn workload dirs into raw/<CVE>/quarantine/ preserving the run
layout, writes a per-CVE quarantine manifest, and re-runs are idempotent.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate" / "quarantine_workloads.py"


def build_raw(tmp):
    """raw/<CVE>/normal/{idle,msg_msg_256,keyctl} + attack sibling, 2 runs each."""
    root = Path(tmp)
    cve_dir = root / "CVE-SYN"
    for workload in ("idle", "msg_msg_256", "keyctl"):
        for i in range(2):
            run_dir = cve_dir / "normal" / workload / f"run_00{i}_syn{i}"
            run_dir.mkdir(parents=True)
            (run_dir / "trace.log").write_text("trace\n")
            (run_dir / "manifest.json").write_text(
                json.dumps({"status": "valid", "class": "normal",
                            "cve": "CVE-SYN", "workload": workload}))
    attack_run = cve_dir / "attack" / "poc_spray" / "run_000_synatt"
    attack_run.mkdir(parents=True)
    (attack_run / "trace.log").write_text("trace\n")
    (attack_run / "manifest.json").write_text(
        json.dumps({"status": "valid", "class": "attack",
                    "cve": "CVE-SYN", "variant": "poc_spray"}))
    return root


class QuarantineWorkloadsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="quarantine_")
        self.raw = build_raw(self.tmp)
        self.script = [sys.executable, str(SCRIPT), "--raw", str(self.raw)]

    def run_script(self, *extra):
        result = subprocess.run(self.script + list(extra), capture_output=True,
                                text=True, check=False)
        self.assertEqual(result.returncode, 0,
                         f"rc={result.returncode}\n{result.stdout}\n{result.stderr}")
        return result.stdout

    def test_churn_runs_exist(self):
        self.assertTrue((self.raw / "CVE-SYN" / "normal" / "msg_msg_256"
                         / "run_000_syn0" / "trace.log").exists())

    def test_dry_run_moves_nothing(self):
        out = self.run_script()
        self.assertIn("DRY RUN", out)
        # plan covers exactly the churn workloads, never clean ones
        plan_lines = [line for line in out.splitlines() if " -> " in line]
        self.assertEqual(len(plan_lines), 2)
        self.assertTrue(all("msg_msg_256" in line or "keyctl" in line
                            for line in plan_lines))
        # nothing moved
        self.assertTrue((self.raw / "CVE-SYN" / "normal" / "msg_msg_256").is_dir())
        self.assertFalse((self.raw / "CVE-SYN" / "quarantine").exists())

    def test_apply_moves_churn_only_and_is_idempotent(self):
        self.run_script("--apply")
        q = self.raw / "CVE-SYN" / "quarantine"
        # churn workloads moved with run layout intact
        self.assertTrue((q / "msg_msg_256" / "run_000_syn0" / "trace.log").is_file())
        self.assertTrue((q / "keyctl" / "run_001_syn1" / "manifest.json").is_file())
        self.assertFalse((self.raw / "CVE-SYN" / "normal" / "msg_msg_256").exists())
        self.assertFalse((self.raw / "CVE-SYN" / "normal" / "keyctl").exists())
        # clean workloads and attack stay put
        self.assertTrue((self.raw / "CVE-SYN" / "normal" / "idle" / "run_000_syn0").is_dir())
        self.assertTrue((self.raw / "CVE-SYN" / "attack" / "poc_spray").is_dir())
        # per-CVE manifest records both moves
        manifest = json.loads((q / "quarantine_manifest.json").read_text())
        moved_to = sorted(m["workload"] for m in manifest["moves"])
        self.assertEqual(moved_to, ["keyctl", "msg_msg_256"])
        self.assertEqual(manifest["moves"][0]["runs"], 2)

        # second run: nothing to move, nothing duplicated
        out = self.run_script("--apply")
        self.assertIn("nothing to move", out)
        manifest2 = json.loads((q / "quarantine_manifest.json").read_text())
        self.assertEqual(len(manifest2["moves"]), 2)

    def test_workload_subset(self):
        self.run_script("--apply", "--workloads", "msg_msg_256")
        self.assertFalse((self.raw / "CVE-SYN" / "normal" / "msg_msg_256").exists())
        self.assertTrue((self.raw / "CVE-SYN" / "normal" / "keyctl").is_dir())
        manifest = json.loads(
            (self.raw / "CVE-SYN" / "quarantine" / "quarantine_manifest.json").read_text())
        self.assertEqual([m["workload"] for m in manifest["moves"]], ["msg_msg_256"])


if __name__ == "__main__":
    unittest.main()
