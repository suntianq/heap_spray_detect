"""Unit tests for the FusionDetector (phase 3 dual-axis fusion).

Verifies:
- FusionDetector trains both GRU and ocsvm sub-models
- fuse_scores produces quantile-aligned weighted combination
- Save/load round-trip preserves config
- _quantile_rank maps scores correctly to [0, 1]
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.fusion import FusionDetector, _run_max


class TestQuantileRank(unittest.TestCase):
    def test_basic_ranking(self):
        det = FusionDetector(seed=42, gru_config={"epochs": 1, "batch_size": 4, "g": 3,
                                                   "vocab_size": 50, "d_model": 16, "n_layers": 1})
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # sorted
        scores = np.array([0.5, 2.5, 6.0])
        ranks = det._quantile_rank(scores, ref)
        # 0.5 < 1.0 -> 0/5 = 0.0
        # 2.5 is between 2.0 and 3.0 -> 2/5 = 0.4
        # 6.0 > 5.0 -> 5/5 = 1.0
        self.assertAlmostEqual(ranks[0], 0.0)
        self.assertAlmostEqual(ranks[1], 0.4)
        self.assertAlmostEqual(ranks[2], 1.0)

    def test_empty_ref(self):
        det = FusionDetector(seed=42, gru_config={"epochs": 1, "batch_size": 4, "g": 3,
                                                   "vocab_size": 50, "d_model": 16, "n_layers": 1})
        ranks = det._quantile_rank(np.array([1.0, 2.0]), np.array([]))
        np.testing.assert_array_equal(ranks, np.array([0.0, 0.0]))


class TestRunMax(unittest.TestCase):
    def test_run_max(self):
        scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        groups = np.array(["run_a", "run_a", "run_b", "run_b", "run_c"])
        max_scores, ids = _run_max(scores, groups)
        max_map = dict(zip(ids, max_scores))
        self.assertAlmostEqual(max_map["run_a"], 0.5)
        self.assertAlmostEqual(max_map["run_b"], 0.9)
        self.assertAlmostEqual(max_map["run_c"], 0.2)


class TestFusionDetector(unittest.TestCase):
    """Integration test: train both axes and fuse scores."""

    @classmethod
    def setUpClass(cls):
        cls.det = FusionDetector(
            seed=42,
            gru_config={"epochs": 3, "batch_size": 8, "g": 3,
                        "vocab_size": 50, "d_model": 16, "n_layers": 1},
            ocsvm_config={"kernel": "rbf", "nu": 0.1, "gamma": "scale"},
            w_gru=0.6, w_ocsvm=0.4)

        # Normal token sequences: diverse tokens (0-9), each differs from prev
        rng = np.random.default_rng(42)
        n_normal = 100
        seq_len = 32
        feat_dim = 10
        cls.normal_tokens = np.zeros((n_normal, seq_len), dtype=np.int32)
        for i in range(n_normal):
            for t in range(seq_len):
                if t == 0:
                    cls.normal_tokens[i, t] = rng.integers(0, 10)
                else:
                    prev = cls.normal_tokens[i, t - 1]
                    cls.normal_tokens[i, t] = (prev + 1 + rng.integers(0, 9)) % 10

        # Normal window features: 2 clusters in feature space
        cls.normal_windows = rng.normal(0, 1, (n_normal, feat_dim)).astype(np.float32)

        # Train GRU
        cls.det.fit_sequences(cls.normal_tokens)
        # Train ocsvm
        cls.det.fit_windows(cls.normal_windows)

        # Fit fusion: score val normal (use more data for meaningful quantiles)
        cls.val_tokens = cls.normal_tokens[:50]
        cls.val_windows = cls.normal_windows[:50]
        gru_val = cls.det.sequence_anomaly_score(cls.val_tokens).mean(axis=1)
        ocs_val = cls.det.window_anomaly_score(cls.val_windows)
        cls.det.fit_fusion(gru_val, ocs_val)

    def test_fuse_scores_shape(self):
        """fuse_scores returns (run_scores, run_ids) with correct length."""
        test_tokens = self.normal_tokens[20:30]
        test_windows = self.normal_windows[20:30]
        run_ids_token = np.array(["run_0"] * 10)
        run_ids_window = np.array(["run_0"] * 10)

        gru_scores = self.det.sequence_anomaly_score(test_tokens).mean(axis=1)
        ocs_scores = self.det.window_anomaly_score(test_windows)
        fused, ids = self.det.fuse_scores(gru_scores, ocs_scores,
                                           run_ids_token, run_ids_window)
        self.assertEqual(len(fused), 1)  # one run
        self.assertEqual(len(ids), 1)
        self.assertTrue(np.isfinite(fused[0]))
        self.assertGreaterEqual(fused[0], 0.0)
        self.assertLessEqual(fused[0], 1.0)

    def test_attack_higher_than_normal(self):
        """Attack sequences should have higher GRU violation + ocsvm distance
        than normal sequences (before run-level aggregation)."""
        attack_tokens = np.full((10, 32), 40, dtype=np.int32)
        attack_windows = np.random.default_rng(99).normal(5, 2, (10, 10)).astype(np.float32)

        rng = np.random.default_rng(77)
        nrm_tokens = np.zeros((10, 32), dtype=np.int32)
        for i in range(10):
            for t in range(32):
                if t == 0:
                    nrm_tokens[i, t] = rng.integers(0, 10)
                else:
                    prev = nrm_tokens[i, t - 1]
                    nrm_tokens[i, t] = (prev + 1 + rng.integers(0, 9)) % 10
        nrm_windows = rng.normal(0, 1, (10, 10)).astype(np.float32)

        # Compare per-sequence scores (not run-level fused)
        normal_gru = self.det.sequence_anomaly_score(nrm_tokens).mean(axis=1)
        attack_gru = self.det.sequence_anomaly_score(attack_tokens).mean(axis=1)
        self.assertGreater(attack_gru.mean(), normal_gru.mean(),
                           f"gru attack={attack_gru.mean():.3f} normal={normal_gru.mean():.3f}")

        normal_ocs = self.det.window_anomaly_score(nrm_windows)
        attack_ocs = self.det.window_anomaly_score(attack_windows)
        self.assertGreater(attack_ocs.mean(), normal_ocs.mean(),
                           f"ocs attack={attack_ocs.mean():.3f} normal={normal_ocs.mean():.3f}")


class TestFusionSaveLoad(unittest.TestCase):
    def test_save_load_preserves_config(self):
        det = FusionDetector(seed=42,
                             gru_config={"epochs": 1, "batch_size": 4, "g": 3,
                                         "vocab_size": 50, "d_model": 16, "n_layers": 1},
                             w_gru=0.7, w_ocsvm=0.3)
        det.gru_val_scores = np.array([0.1, 0.2, 0.3])
        det.ocsvm_val_scores = np.array([1.0, 2.0, 3.0])

        # Train minimal GRU + ocsvm so save has something to serialize
        tokens = np.random.default_rng(42).integers(0, 50, (10, 16), dtype=np.int32)
        det.fit_sequences(tokens)
        windows = np.random.default_rng(42).normal(0, 1, (10, 4)).astype(np.float32)
        det.fit_windows(windows)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            det.save(f.name)
            loaded = FusionDetector.load(f.name)
        os.unlink(f.name)

        self.assertEqual(loaded.w_gru, 0.7)
        self.assertEqual(loaded.w_ocsvm, 0.3)
        np.testing.assert_array_equal(loaded.gru_val_scores, [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
