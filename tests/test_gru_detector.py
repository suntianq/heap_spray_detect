"""Unit tests for the GRU event-level anomaly detector.

Verifies:
- GRU trains on synthetic normal token sequences without error
- Top-g violation rate is higher for anomalous sequences (repeated tokens)
  than for normal sequences (diverse tokens)
- Shape contract: sequence_anomaly_score returns (N, L) per-position scores
- Save/load round-trip preserves scoring
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.gru_detector import GRUDetector, GRUNet


def make_normal_sequences(n=200, seq_len=64, vocab=1536, seed=42):
    """Normal sequences: diverse tokens from a broad vocabulary (0-19),
    each position differs from previous (mimicking diverse kernel events)."""
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


def make_attack_sequences(n=20, seq_len=64, vocab=1536, seed=99):
    """Attack sequences: a single token (50-59, outside normal range) repeated.
    This mimics spray: a call_site/size combo never seen in normal data,
    repeated hundreds of times."""
    rng = np.random.default_rng(seed)
    seqs = np.zeros((n, seq_len), dtype=np.int32)
    for i in range(n):
        token = rng.integers(50, 60)  # tokens never seen in normal data
        seqs[i, :] = token
    return seqs


class TestGRUShape(unittest.TestCase):
    def test_grunet_forward_shape(self):
        import torch
        net = GRUNet(vocab_size=100, d_model=32, n_layers=1, dropout=0.0)
        x = torch.randint(0, 100, (4, 16))
        out = net(x)
        self.assertEqual(out.shape, (4, 16, 100))

    def test_sequence_anomaly_score_shape(self):
        """sequence_anomaly_score returns (N, L) per-position violations."""
        det = GRUDetector(seed=42, epochs=2, batch_size=8, g=5,
                          vocab_size=100, d_model=16, n_layers=1)
        normal = make_normal_sequences(n=50, seq_len=32, vocab=100)
        det.fit_sequences(normal)
        test = make_normal_sequences(n=10, seq_len=32, vocab=100)
        scores = det.sequence_anomaly_score(test)
        self.assertEqual(scores.shape, (10, 32))
        # violations are 0.0 or 1.0
        self.assertTrue(np.all((scores == 0.0) | (scores == 1.0)))


class TestGRUViolationSensitivity(unittest.TestCase):
    """The core test: violation rate should be higher for spray-like sequences."""

    @classmethod
    def setUpClass(cls):
        """Train a small GRU on normal (diverse) sequences."""
        cls.det = GRUDetector(seed=42, epochs=15, batch_size=16, g=3,
                              vocab_size=100, d_model=32, n_layers=1)
        cls.normal_train = make_normal_sequences(n=200, seq_len=32, vocab=100)
        cls.det.fit_sequences(cls.normal_train)

    def test_attack_higher_than_normal(self):
        """Spray sequences (repeated token) should have higher violation rate."""
        normal_test = make_normal_sequences(n=30, seq_len=32, vocab=100, seed=7)
        attack_test = make_attack_sequences(n=30, seq_len=32, vocab=100, seed=7)

        normal_scores = self.det.sequence_anomaly_score(normal_test)
        attack_scores = self.det.sequence_anomaly_score(attack_test)

        normal_rate = normal_scores.mean(axis=1)
        attack_rate = attack_scores.mean(axis=1)

        self.assertGreater(attack_rate.mean(), normal_rate.mean(),
                           f"attack={attack_rate.mean():.3f} normal={normal_rate.mean():.3f}")


class TestGRUSaveLoad(unittest.TestCase):
    def test_save_load_roundtrip(self):
        det = GRUDetector(seed=42, epochs=2, batch_size=8, g=3,
                          vocab_size=50, d_model=16, n_layers=1)
        normal = make_normal_sequences(n=20, seq_len=16, vocab=50)
        det.fit_sequences(normal)

        test = make_normal_sequences(n=5, seq_len=16, vocab=50)
        scores_before = det.sequence_anomaly_score(test)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            det.save(f.name)
            loaded = GRUDetector.load(f.name)
        os.unlink(f.name)

        scores_after = loaded.sequence_anomaly_score(test)
        np.testing.assert_array_equal(scores_before, scores_after)


class TestTokenEncoding(unittest.TestCase):
    """Verify token encoding is bijective (no collisions)."""

    def test_encode_decode(self):
        from scripts.preprocess.trace2tokens import encode_token
        seen = set()
        for op in range(2):
            for sb in range(12):
                for bt in range(4):
                    for fr in range(4):
                        for dt in range(4):
                            for cpu_b in range(3):
                                for reclaim in range(3):
                                    tid = encode_token(op, sb, bt, fr, dt, cpu_b, reclaim)
                                    self.assertNotIn(tid, seen, f"collision at ({op},{sb},{bt},{fr},{dt},{cpu_b},{reclaim})")
                                    seen.add(tid)
        self.assertEqual(len(seen), 2 * 12 * 4 * 4 * 4 * 3 * 3)
        self.assertEqual(max(seen), 13823)
        self.assertEqual(min(seen), 0)


if __name__ == "__main__":
    unittest.main()
