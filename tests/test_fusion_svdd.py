"""Unit tests for FusionSVDDDetector (unified GRU + Deep SVDD).

Verifies:
- Net forward returns correct shapes (next_logits + svdd_emb)
- Training completes without error
- sequence_anomaly_score returns (N, L) per-position scores
- Attack sequences score higher than normal
- Save/load round-trip
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.fusion_svdd import FusionSVDDDetector, FusionSVDDNet


def make_normal_sequences(n=200, seq_len=64, vocab=100, seed=42):
    """Normal: diverse tokens, each position differs from previous."""
    rng = np.random.default_rng(seed)
    seqs = np.zeros((n, seq_len), dtype=np.int32)
    for i in range(n):
        for t in range(seq_len):
            if t == 0:
                seqs[i, t] = rng.integers(0, 20)
            else:
                prev = seqs[i, t - 1]
                seqs[i, t] = (prev + 1 + rng.integers(0, 19)) % 20
    return seqs


def make_attack_sequences(n=20, seq_len=64, vocab=100, seed=99):
    """Attack: unseen token (50-59) repeated throughout."""
    rng = np.random.default_rng(seed)
    seqs = np.zeros((n, seq_len), dtype=np.int32)
    for i in range(n):
        seqs[i, :] = rng.integers(50, 60)
    return seqs


class TestFusionSVDDNet(unittest.TestCase):
    def test_forward_shapes(self):
        net = FusionSVDDNet(vocab_size=100, d_model=32, n_layers=1, dropout=0.0, svdd_dim=16)
        x = torch.randint(0, 100, (4, 16))
        next_logits, svdd_emb = net(x)
        self.assertEqual(next_logits.shape, (4, 16, 100))
        self.assertEqual(svdd_emb.shape, (4, 16))  # svdd_dim


class TestFusionSVDDDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.det = FusionSVDDDetector(
            seed=42, epochs=5, batch_size=16, g=3,
            vocab_size=100, d_model=32, n_layers=1,
            svdd_loss_weight=0.1, svdd_score_weight=0.3)
        cls.normal = make_normal_sequences(n=200, seq_len=32, vocab=100)
        cls.det.fit_sequences(cls.normal)

    def test_score_shape(self):
        test = make_normal_sequences(n=10, seq_len=32, vocab=100, seed=7)
        scores = self.det.sequence_anomaly_score(test)
        self.assertEqual(scores.shape, (10, 32))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_attack_higher_than_normal(self):
        """Attack (unseen repeated token) should score higher than normal."""
        normal_test = make_normal_sequences(n=20, seq_len=32, vocab=100, seed=7)
        attack_test = make_attack_sequences(n=20, seq_len=32, vocab=100, seed=7)

        normal_scores = self.det.sequence_anomaly_score(normal_test).mean(axis=1)
        attack_scores = self.det.sequence_anomaly_score(attack_test).mean(axis=1)

        self.assertGreater(attack_scores.mean(), normal_scores.mean(),
                           f"attack={attack_scores.mean():.4f} normal={normal_scores.mean():.4f}")

    def test_svdd_center_initialized(self):
        """SVDD center should be a non-zero vector after training."""
        self.assertIsNotNone(self.det.svdd_center)
        self.assertGreater(self.det.svdd_center.norm().item(), 0.0)

    def test_svdd_p99_set(self):
        """svdd_p99 should be set after training for normalization."""
        self.assertGreater(self.det.svdd_p99, 0.0)


class TestFusionSVDDSaveLoad(unittest.TestCase):
    def test_save_load_roundtrip(self):
        det = FusionSVDDDetector(
            seed=42, epochs=2, batch_size=8, g=3,
            vocab_size=50, d_model=16, n_layers=1)
        normal = make_normal_sequences(n=20, seq_len=16, vocab=50)
        det.fit_sequences(normal)

        test = make_normal_sequences(n=5, seq_len=16, vocab=50)
        scores_before = det.sequence_anomaly_score(test)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            det.save(f.name)
            loaded = FusionSVDDDetector.load(f.name)
        os.unlink(f.name)

        scores_after = loaded.sequence_anomaly_score(test)
        np.testing.assert_array_almost_equal(scores_before, scores_after, decimal=5)


if __name__ == "__main__":
    unittest.main()
