"""Unit tests for the schema v2 preprocessing pipeline.

Covers IMPLEMENTATION_PLAN.md section 10.1 items for preprocessing:
- ptr reuse, unknown FREE, cross-window FREE size recovery
- fixed-cadence windows with empty windows preserved
- 50%-overlap window label boundary
- endpoint/any sequence labels and boundary policy
- sequences never cross runs; sequence span is 1650 ms
- workload/CVE/variant metadata parsing
- marker fail-closed behaviour (missing / reversed order)
- no NaN/Inf in output
- deterministic output hash
- non-empty output directory is refused
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "preprocess"))

import config
import csv2features as c2f
from synthetic import csv_row, write_csv_file, write_spray_markers

CSV2FEATURES = os.path.join(os.path.dirname(__file__), "..", "scripts", "preprocess", "csv2features.py")


class TestBucketIndex(unittest.TestCase):
    def test_sizes_map_to_buckets(self):
        for size, expected in [(32, 0), (33, 1), (64, 1), (96, 2), (128, 3),
                               (192, 4), (256, 5), (512, 6), (1024, 7),
                               (2048, 8), (4096, 9), (8192, 10)]:
            self.assertEqual(c2f.bucket_index(size), expected, f"size={size}")

    def test_overflow_bucket(self):
        self.assertEqual(c2f.bucket_index(8193), c2f.OVERFLOW_BUCKET)
        self.assertEqual(c2f.bucket_index(1 << 30), c2f.OVERFLOW_BUCKET)


class TestResolveFreeSizes(unittest.TestCase):
    def test_resolution_and_lifetime(self):
        events = [
            {"op": "ALLOC", "ptr": "0x100", "bytes_alloc": 256, "bytes_req": 0, "timestamp_ns": 1000},
            {"op": "ALLOC", "ptr": "0x200", "bytes_alloc": 64, "bytes_req": 0, "timestamp_ns": 2000},
            {"op": "FREE", "ptr": "0x100", "bytes_alloc": 0, "bytes_req": 0, "timestamp_ns": 3000},
            {"op": "FREE", "ptr": "0x300", "bytes_alloc": 0, "bytes_req": 0, "timestamp_ns": 4000},
        ]
        stats = c2f.resolve_free_sizes(events)
        self.assertEqual(stats["resolved"], 1)
        self.assertEqual(stats["unresolved"], 1)  # free of unknown ptr
        free = events[2]
        self.assertTrue(free["size_resolved"])
        self.assertEqual(free["resolved_bytes_alloc"], 256)
        self.assertEqual(free["allocation_timestamp_ns"], 1000)
        self.assertEqual(free["object_lifetime_ns"], 2000)
        unknown_free = events[3]
        self.assertFalse(unknown_free["size_resolved"])
        self.assertIsNone(unknown_free["resolved_bytes_alloc"])

    def test_ptr_reuse_across_free(self):
        """A freed pointer is re-allocated; the second FREE resolves to the new size."""
        events = [
            {"op": "ALLOC", "ptr": "0x100", "bytes_alloc": 256, "bytes_req": 0, "timestamp_ns": 1000},
            {"op": "FREE", "ptr": "0x100", "bytes_alloc": 0, "bytes_req": 0, "timestamp_ns": 2000},
            {"op": "ALLOC", "ptr": "0x100", "bytes_alloc": 512, "bytes_req": 0, "timestamp_ns": 3000},
            {"op": "FREE", "ptr": "0x100", "bytes_alloc": 0, "bytes_req": 0, "timestamp_ns": 4000},
        ]
        c2f.resolve_free_sizes(events)
        self.assertEqual(events[1]["resolved_bytes_alloc"], 256)
        self.assertEqual(events[3]["resolved_bytes_alloc"], 512)  # new allocation wins


class TestFixedWindows(unittest.TestCase):
    def test_empty_windows_preserved(self):
        """Events at 0 and 1000 ms on a 100 ms / 50 ms cadence -> empty windows kept.

        Overlapping windows (stride 50 < window 100) mean an event can appear in
        up to two windows: event@0 lands in [0,100); event@1000ms lands in both
        [950,1050) and [1000,1100). So 21 windows total, 3 non-empty, 18 empty.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.csv")
            write_csv_file(path, [
                csv_row(0, 1, 1, "bash", "ALLOC", "0000000000000001", 32, 32, "ffffffff8139c7f1"),
                csv_row(1_000_000_000, 1, 1, "bash", "FREE", "0000000000000001", 0, 0, "ffffffff8136ccd1"),
            ])
            features, labels, starts, _, _, _, empty_ratio = c2f.process_csv(path, 100, 50)
            names = c2f.feature_names()
            is_empty = features[:, names.index("is_empty")]
            event_count = features[:, names.index("event_count")]
            self.assertEqual(len(features), 21)  # windows at 0, 50, ..., 1000 ms
            self.assertEqual(int(is_empty.sum()), 18)  # empty windows preserved
            self.assertEqual(int((1 - is_empty).sum()), 3)
            self.assertEqual(int(event_count.sum()), 3)  # event@1000 counted in 2 windows
            self.assertGreater(empty_ratio, 0.5)
            # empty windows have zero counts
            alloc_count_idx = names.index("alloc_count_32")
            empty_mask = is_empty == 1
            self.assertTrue((event_count[empty_mask] == 0).all())
            self.assertTrue((features[empty_mask, alloc_count_idx] == 0).all())

    def test_rate_fixed_over_window(self):
        """alloc_rate must equal count / window_ms (100), not span between events."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.csv")
            # two allocations in the first 100 ms window
            write_csv_file(path, [
                csv_row(1_000_000, 1, 1, "bash", "ALLOC", "0000000000000001", 32, 32, "ffffffff8139c7f1"),
                csv_row(30_000_000, 1, 1, "bash", "ALLOC", "0000000000000002", 32, 32, "ffffffff8139c7f1"),
            ])
            features, labels, starts, *_ = c2f.process_csv(path, 100, 50)
            idx = c2f.feature_names().index("alloc_rate_32")
            # features are stored as float32; compare with tolerance
            self.assertAlmostEqual(float(features[0, idx]), 2.0 / 100.0, places=6)


class TestWindowLabel(unittest.TestCase):
    def test_boundary_policy_long_spray(self):
        # Spray interval exactly [100ms, 300ms) (200ms, spanning several windows).
        # A window is 1 if it overlaps >=50% of the window OR >=50% of the spray.
        start, end = 100_000_000, 300_000_000
        # window [0,100ms): overlap 0 -> 0
        self.assertEqual(c2f.window_label(0, 100_000_000, True, start, end), 0)
        # window [100ms,200ms): overlap 100ms = 100% of window -> 1
        self.assertEqual(c2f.window_label(100_000_000, 200_000_000, True, start, end), 1)
        # window [150ms,250ms): overlap 100ms = 100% of window -> 1
        self.assertEqual(c2f.window_label(150_000_000, 250_000_000, True, start, end), 1)
        # window [250ms,350ms): overlap 50ms = 50% of window -> 1
        self.assertEqual(c2f.window_label(250_000_000, 350_000_000, True, start, end), 1)
        # window [260ms,360ms): overlap 40ms = 40% of window AND 20% of spray -> -1 boundary
        self.assertEqual(c2f.window_label(260_000_000, 360_000_000, True, start, end), -1)
        # window [300ms,400ms): overlap 0 -> 0
        self.assertEqual(c2f.window_label(300_000_000, 400_000_000, True, start, end), 0)
        # normal trace is always normal
        self.assertEqual(c2f.window_label(0, 100_000_000, False, start, end), 0)

    def test_short_spray_single_window(self):
        # A sub-window spray (0.2ms) must still produce a label-1 window: the
        # containing window covers ~100% of the spray (spray-ratio), which is
        # what a detector would flag. This was the pilot data failure under a
        # window-length-only ratio.
        start, end = 50_000_000, 50_200_000  # 0.2 ms spray fully inside [0,100ms)
        # window [0ms,100ms) contains the whole spray -> 1
        self.assertEqual(c2f.window_label(0, 100_000_000, True, start, end), 1)
        # window [50ms,150ms) contains the whole spray -> 1
        self.assertEqual(c2f.window_label(50_000_000, 150_000_000, True, start, end), 1)
        # window [100ms,200ms) has no overlap -> 0
        self.assertEqual(c2f.window_label(100_000_000, 200_000_000, True, start, end), 0)

    def test_zero_length_spray_is_boundary(self):
        start = end = 100_000_000
        self.assertEqual(c2f.window_label(0, 100_000_000, True, start, end), -1)


class TestSequenceLabels(unittest.TestCase):
    def test_endpoint_policy(self):
        self.assertEqual(c2f.label_sequence(np.array([0, 0, 1]), "endpoint", "drop"), 1)
        self.assertEqual(c2f.label_sequence(np.array([0, 0, -1]), "endpoint", "drop"), -1)
        self.assertEqual(c2f.label_sequence(np.array([0, 0, 0]), "endpoint", "drop"), 0)

    def test_any_policy(self):
        self.assertEqual(c2f.label_sequence(np.array([0, 0, 1]), "any", "drop"), 1)
        self.assertEqual(c2f.label_sequence(np.array([0, 0, 0]), "any", "drop"), 0)

    def test_boundary_drop_ignores_mixed(self):
        """A sequence containing any boundary window is -1 under drop, even with attack windows."""
        self.assertEqual(c2f.label_sequence(np.array([1, -1, 0]), "any", "drop"), -1)
        self.assertEqual(c2f.label_sequence(np.array([1, -1, 0]), "endpoint", "drop"), -1)
        # keep policy preserves the old behaviour for comparison
        self.assertEqual(c2f.label_sequence(np.array([1, -1, 0]), "any", "keep"), 1)


class TestBuildSequences(unittest.TestCase):
    def test_span_and_single_run(self):
        """32 windows on a 50 ms stride -> start interval 1550 ms, covered range 1650 ms."""
        rng = np.random.default_rng(0)
        features = rng.random((40, config.FEAT_DIM)).astype(np.float32)
        labels = np.zeros(40, np.int8)
        starts = np.arange(40, dtype=np.int64) * 50_000_000
        seqs, seq_labels, groups, seq_starts, short = c2f.build_sequences(
            features, labels, starts, 32, "msg_msg_run_000", "endpoint", "drop", 50, 100)
        self.assertFalse(short)
        self.assertEqual(seqs.shape, (9, 32, config.FEAT_DIM))
        # start-to-start interval is (32-1)*50ms
        self.assertEqual(int(seq_starts[1] - seq_starts[0]), 50_000_000)
        self.assertEqual(int(starts[31] - starts[0]), 31 * 50_000_000)
        # all sequences belong to the single run
        self.assertTrue((groups == "msg_msg_run_000").all())

    def test_short_run_returns_empty(self):
        features = np.zeros((5, config.FEAT_DIM), np.float32)
        labels = np.zeros(5, np.int8)
        starts = np.arange(5, dtype=np.int64) * 50_000_000
        seqs, _, _, _, short = c2f.build_sequences(features, labels, starts, 32, "idle_run_000",
                                                   "endpoint", "drop", 50, 100)
        self.assertTrue(short)
        self.assertEqual(len(seqs), 0)


class TestRunMetadata(unittest.TestCase):
    def test_parse_variants(self):
        meta = c2f.parse_run_metadata("CVE-2010-2959/poc_cfh_single_spray_run_001")
        self.assertEqual(meta["class"], "attack")
        self.assertEqual(meta["cve"], "CVE-2010-2959")
        self.assertEqual(meta["variant"], "poc_cfh_single_spray")
        self.assertIsNone(meta["workload"])

    def test_parse_normal(self):
        meta = c2f.parse_run_metadata("msg_msg_run_000")
        self.assertEqual(meta["class"], "normal")
        self.assertEqual(meta["workload"], "msg_msg")
        self.assertIsNone(meta["cve"])

    def test_parse_normal_v2_dir_structure(self):
        meta = c2f.parse_run_metadata("msg_msg_256/run_000_abcd1234/trace")
        self.assertEqual(meta["class"], "normal")
        self.assertEqual(meta["workload"], "msg_msg_256")
        self.assertIsNone(meta["cve"])

    def test_parse_baseline_under_cve_is_attack(self):
        # A baseline PoC lives under a CVE dir, so path parsing marks it attack;
        # --run-meta must be able to flip it back to normal.
        meta = c2f.parse_run_metadata("CVE-2017-11176/poc_cfh_baseline/run_000_x/trace")
        self.assertEqual(meta["class"], "attack")
        self.assertEqual(meta["variant"], "poc_cfh_baseline")

    def test_parse_normal_under_cve_is_normal(self):
        # Matched normal controls are collected under a CVE dir too (plan 7.2);
        # the CVE segment alone must not imply an attack run.
        meta = c2f.parse_run_metadata("CVE-2017-11176/idle/run_000_x/trace")
        self.assertEqual(meta["class"], "normal")
        self.assertEqual(meta["cve"], "CVE-2017-11176")
        self.assertEqual(meta["workload"], "idle")
        self.assertIsNone(meta["variant"])

    def test_parse_msg_msg_under_cve(self):
        meta = c2f.parse_run_metadata("CVE-2017-7308/msg_msg_256/run_003_abcd/trace")
        self.assertEqual(meta["class"], "normal")
        self.assertEqual(meta["cve"], "CVE-2017-7308")
        self.assertEqual(meta["workload"], "msg_msg_256")

    def test_merge_run_meta_syncs_top_level_fields(self):
        records = [{
            "run_id": "CVE-2017-11176/poc_cfh_baseline/run_000_x/trace",
            "metadata": c2f.parse_run_metadata("CVE-2017-11176/poc_cfh_baseline/run_000_x/trace"),
            "class": "attack", "cve": "CVE-2017-11176", "variant": "poc_cfh_baseline",
            "workload": None,
        }]
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "runs_meta.json")
            json.dump({records[0]["run_id"]: {"class": "normal", "workload": "baseline"}}, open(meta_path, "w"))
            c2f.merge_run_meta(records, meta_path)
        self.assertEqual(records[0]["class"], "normal")
        self.assertEqual(records[0]["workload"], "baseline")
        self.assertEqual(records[0]["metadata"]["class"], "normal")
        self.assertEqual(records[0]["cve"], "CVE-2017-11176")


class TestEndToEnd(unittest.TestCase):
    """Runs the actual CLI on synthetic traces."""

    def _write_normal_run(self, root, name, seed):
        rng = np.random.default_rng(seed)
        rows = []
        ts = 0
        for i in range(400):
            rows.append(csv_row(ts, 100 + i, 100 + i, "bash", "ALLOC",
                                "{:016x}".format(0x1000 + i), 32, 32, "ffffffff8139c7f1"))
            ts += 2_500_000  # 2.5 ms apart
        write_csv_file(os.path.join(root, name + ".csv"), rows)

    def _write_attack_run(self, root, cve, name, seed):
        rng = np.random.default_rng(seed)
        rows = []
        spray_start, spray_end = 150_000_000, 650_000_000
        ts = 0
        while ts < spray_start:
            rows.append(csv_row(ts, 200, 200, "bash", "ALLOC", "{:016x}".format(0x1000 + ts),
                                32, 32, "ffffffff8139c7f1"))
            ts += 5_000_000
        # dense spray
        for i in range(300):
            rows.append(csv_row(spray_start + i * 1_000_000, 3000, 3000, "poc",
                                "ALLOC", "{:016x}".format(0x2000 + i), 256, 256, "ffffffff81234567"))
        write_csv_file(os.path.join(root, cve, name + ".csv"), rows)
        write_spray_markers(os.path.join(root, "spray_markers.json"),
                            {f"{cve}/{name}.log": {"SPRAY_START": spray_start, "SPRAY_END": spray_end}})

    def test_full_pipeline_attack_and_normal(self):
        with tempfile.TemporaryDirectory() as tmp:
            attack_root = os.path.join(tmp, "attack")
            normal_root = os.path.join(tmp, "normal")
            os.makedirs(os.path.join(attack_root, "CVE-2010-2959"))
            self._write_attack_run(attack_root, "CVE-2010-2959", "poc_cfh_single_spray_run_001", 1)
            self._write_normal_run(normal_root, "idle_run_000", 2)

            out_a = os.path.join(tmp, "out_attack")
            out_n = os.path.join(tmp, "out_normal")
            self.assertEqual(subprocess.run(
                [sys.executable, CSV2FEATURES, "-i", attack_root, "-o", out_a,
                 "--is-attack", "--sequence-label", "any"], capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run(
                [sys.executable, CSV2FEATURES, "-i", normal_root, "-o", out_n],
                capture_output=True).returncode, 0)

            data_a = np.load(os.path.join(out_a, "features.npz"), allow_pickle=True)
            data_n = np.load(os.path.join(out_n, "features.npz"), allow_pickle=True)
            self.assertEqual(data_a["schema_version"], 2)
            self.assertEqual(data_a["features"].shape[1], config.FEAT_DIM)
            self.assertTrue((data_a["labels"] >= -1).all() and (data_a["labels"] <= 1).all())
            # attack run must contain attack windows
            self.assertIn(1, set(data_a["labels"].tolist()))
            # normal run contains no attack and no boundary
            self.assertEqual(set(data_n["labels"].tolist()), {0})
            # no NaN/Inf
            self.assertTrue(np.isfinite(data_a["features"]).all())
            self.assertTrue(np.isfinite(data_n["features"]).all())
            # output manifests exist
            for d in (out_a, out_n):
                self.assertTrue(os.path.exists(os.path.join(d, "dataset_manifest.json")))
                self.assertTrue(os.path.exists(os.path.join(d, "runs_meta.json")))
                self.assertTrue(os.path.exists(os.path.join(d, "feature_schema.json")))
                self.assertTrue(os.path.exists(os.path.join(d, "stats.json")))

    def test_attack_fails_without_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_normal_run(tmp, "idle_run_000", 1)
            out = os.path.join(tmp, "out")
            result = subprocess.run(
                [sys.executable, CSV2FEATURES, "-i", tmp, "-o", out, "--is-attack"],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("markers", result.stderr.lower())

    def test_reversed_marker_order_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "a")
            os.makedirs(root)
            write_csv_file(os.path.join(root, "run.csv"), [
                csv_row(1_000_000, 1, 1, "poc", "ALLOC", "0000000000000001", 32, 32, "ffffffff8139c7f1"),
            ])
            write_spray_markers(os.path.join(root, "spray_markers.json"),
                                {"run.log": {"SPRAY_START": 5_000_000_000, "SPRAY_END": 1_000_000_000}})
            out = os.path.join(tmp, "out")
            result = subprocess.run(
                [sys.executable, CSV2FEATURES, "-i", root, "-o", out, "--is-attack"],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SPRAY_END", result.stderr)

    def test_deterministic_output_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_normal_run(tmp, "idle_run_000", 1)
            out1, out2 = os.path.join(tmp, "o1"), os.path.join(tmp, "o2")
            self.assertEqual(subprocess.run(
                [sys.executable, CSV2FEATURES, "-i", tmp, "-o", out1],
                capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run(
                [sys.executable, CSV2FEATURES, "-i", tmp, "-o", out2],
                capture_output=True).returncode, 0)
            h1 = json.load(open(os.path.join(out1, "dataset_manifest.json")))["output_features_hash"]
            h2 = json.load(open(os.path.join(out2, "dataset_manifest.json")))["output_features_hash"]
            self.assertEqual(h1, h2)

    def test_nonempty_output_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_normal_run(tmp, "idle_run_000", 1)
            out = os.path.join(tmp, "out")
            os.makedirs(out)
            with open(os.path.join(out, "existing"), "w") as fh:
                fh.write("x")
            result = subprocess.run(
                [sys.executable, CSV2FEATURES, "-i", tmp, "-o", out],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not empty", result.stderr)

    def test_free_unknown_quality_flag(self):
        """A run where most frees are unresolvable is flagged high_free_unknown."""
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "a")
            os.makedirs(root)
            rows = []
            for i in range(50):
                rows.append(csv_row(i * 1_000_000, 1, 1, "bash", "ALLOC",
                                    "{:016x}".format(0x1000 + i), 32, 32, "ffffffff8139c7f1"))
                rows.append(csv_row(i * 1_000_000 + 500_000, 1, 1, "bash", "FREE",
                                    "{:016x}".format(0x5000 + i), 0, 0, "ffffffff8136ccd1"))
            write_csv_file(os.path.join(root, "msg_msg_run_000.csv"), rows)
            out = os.path.join(tmp, "out")
            self.assertEqual(subprocess.run(
                [sys.executable, CSV2FEATURES, "-i", root, "-o", out,
                 "--free-unknown-threshold", "0.05"], capture_output=True).returncode, 0)
            meta = json.load(open(os.path.join(out, "runs_meta.json")))
            self.assertEqual(meta["runs"]["msg_msg_run_000"]["quality"], "high_free_unknown")


if __name__ == "__main__":
    unittest.main()
