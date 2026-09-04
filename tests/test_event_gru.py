"""Unit tests for the field-decomposed event GRU detector.

Verifies:
- EventGRUNet emits one logit tensor per field with the right per-field vocab
- The detector rejects the legacy 6-channel field matrix with a clear message
- Training on synthetic normal event matrices produces finite scores of shape (N, L)
- Spray-like sequences (one call_site + one size repeated at <2us) score higher
  than diverse normal sequences
- The lifecycle channel carries signal on its own: a groom-like REUSE cycle
  scores above a NEW-only baseline
- Save/load round-trip preserves the calibration stats and the scores
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.event_gru import (  # noqa: E402
    CS_VOCAB, DT_VOCAB, EVENT_FIELD_SIZE, FIELD_KEYS, LIFE_VOCAB,
    EventGRUDetector, EventGRUNet,
)
from scripts.preprocess.trace2tokens import (  # noqa: E402
    DT_FIELD_VOCAB, LIFE_FREE_LONG, LIFE_FREE_MEDIUM, LIFE_FREE_SHORT,
    LIFE_NEW, LIFE_REUSE, LIFE_UNKNOWN, dt_to_field_bucket, lifecycle_state,
)

# Channel indices of the (N, L, 8) field matrix.
C_OP, C_SIZE, C_CS, C_CPU, C_RECLAIM, C_LIFE, C_DT_B, C_DT_C = range(8)


def make_normal_events(n=120, seq_len=32, seed=42):
    """Diverse events: varied sizes / call_sites / CPUs, alloc-free pairs with
    medium delays. Mimics ordinary kernel allocation traffic."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, seq_len, EVENT_FIELD_SIZE), dtype=np.float32)
    for i in range(n):
        for t in range(seq_len):
            op = int(rng.integers(0, 2))
            x[i, t, C_OP] = op
            x[i, t, C_SIZE] = rng.integers(0, 12)
            x[i, t, C_CS] = rng.integers(0, 40)
            x[i, t, C_CPU] = rng.integers(0, 3)
            x[i, t, C_RECLAIM] = 0
            x[i, t, C_LIFE] = LIFE_NEW if op == 0 else LIFE_FREE_LONG
            x[i, t, C_DT_B] = rng.integers(2, DT_VOCAB)  # >=10us: unhurried
            x[i, t, C_DT_C] = rng.random() * 0.5 + 0.4
    return x


def make_spray_events(n=40, seq_len=32, seed=7):
    """Spray: one call_site, one size class, sub-2us deltas, CPU-pinned."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, seq_len, EVENT_FIELD_SIZE), dtype=np.float32)
    for i in range(n):
        call_site = int(rng.integers(0, 40))
        size = int(rng.integers(0, 12))
        x[i, :, C_OP] = 0            # all ALLOC
        x[i, :, C_SIZE] = size       # one size class
        x[i, :, C_CS] = call_site    # one call_site
        x[i, :, C_CPU] = 0           # pinned
        x[i, :, C_RECLAIM] = 0
        x[i, :, C_LIFE] = LIFE_NEW
        x[i, :, C_DT_B] = 0          # <2us burst
        x[i, :, C_DT_C] = 0.02
    return x


def make_groom_events(n=40, seq_len=32, seed=11):
    """Grooming: ALLOC/ALLOC/FREE_SHORT/REUSE cycle on one size class.

    Timing and size stay in the normal range so the only strong deviation is
    the lifecycle channel (REUSE + FREE_SHORT), isolating its contribution.
    """
    rng = np.random.default_rng(seed)
    cycle_op = [0, 0, 1, 0]
    cycle_life = [LIFE_NEW, LIFE_NEW, LIFE_FREE_SHORT, LIFE_REUSE]
    x = np.zeros((n, seq_len, EVENT_FIELD_SIZE), dtype=np.float32)
    for i in range(n):
        size = int(rng.integers(0, 12))
        call_site = int(rng.integers(0, 40))
        for t in range(seq_len):
            phase = t % len(cycle_op)
            x[i, t, C_OP] = cycle_op[phase]
            x[i, t, C_SIZE] = size
            x[i, t, C_CS] = call_site
            x[i, t, C_CPU] = int(rng.integers(0, 3))
            x[i, t, C_RECLAIM] = 1 if cycle_life[phase] == LIFE_REUSE else 0
            x[i, t, C_LIFE] = cycle_life[phase]
            x[i, t, C_DT_B] = rng.integers(2, DT_VOCAB)  # normal-speed
            x[i, t, C_DT_C] = rng.random() * 0.5 + 0.4
    return x


class TestFieldHelpers(unittest.TestCase):
    """The preprocessing helpers that populate the two new channels."""

    def test_dt_bucket_boundaries(self):
        # Spray burst regime gets its own class 0, boundary is exclusive.
        self.assertEqual(dt_to_field_bucket(0), 0)
        self.assertEqual(dt_to_field_bucket(1_999), 0)      # 1.999us
        self.assertEqual(dt_to_field_bucket(2_000), 1)      # 2us
        self.assertEqual(dt_to_field_bucket(9_999), 1)
        self.assertEqual(dt_to_field_bucket(10_000), 2)     # 10us
        self.assertEqual(dt_to_field_bucket(100_000), 3)    # 100us
        self.assertEqual(dt_to_field_bucket(1_000_000), 4)  # 1ms
        self.assertEqual(dt_to_field_bucket(10_000_000), 5)  # 10ms
        self.assertEqual(dt_to_field_bucket(10 ** 12), DT_FIELD_VOCAB - 1)
        # Negative deltas (clock skew) clamp into the burst bucket, not below 0.
        self.assertEqual(dt_to_field_bucket(-5), 0)

    def test_dt_bucket_in_range(self):
        for dt_ns in (0, 1, 5_000, 250_000, 3_000_000, 10 ** 11):
            self.assertIn(dt_to_field_bucket(dt_ns), range(DT_FIELD_VOCAB))

    def test_lifecycle_alloc(self):
        self.assertEqual(lifecycle_state({"op": "ALLOC"}), LIFE_NEW)
        self.assertEqual(
            lifecycle_state({"op": "ALLOC", "reclaim_from_free": True}),
            LIFE_REUSE)

    def test_lifecycle_free_by_lifetime(self):
        def free(lifetime_ns):
            return lifecycle_state({"op": "FREE", "object_lifetime_ns": lifetime_ns})

        self.assertEqual(free(0), LIFE_FREE_SHORT)
        self.assertEqual(free(99_000), LIFE_FREE_SHORT)       # 99us
        self.assertEqual(free(100_000), LIFE_FREE_MEDIUM)     # 100us
        self.assertEqual(free(9_000_000), LIFE_FREE_MEDIUM)   # 9ms
        self.assertEqual(free(10_000_000), LIFE_FREE_LONG)    # 10ms
        # An unresolved free (allocation never observed) is UNKNOWN, not SHORT.
        self.assertEqual(free(None), LIFE_UNKNOWN)

    def test_lifecycle_values_in_vocab(self):
        events = [{"op": "ALLOC"}, {"op": "ALLOC", "reclaim_from_free": True},
                  {"op": "FREE", "object_lifetime_ns": 1},
                  {"op": "FREE", "object_lifetime_ns": None}]
        for event in events:
            self.assertIn(lifecycle_state(event), range(LIFE_VOCAB))


class TestEventGRUNetShape(unittest.TestCase):
    def test_forward_emits_one_head_per_field(self):
        import torch
        net = EventGRUNet(d_model=16, n_layers=1, dropout=0.0, cs_vocab=64)
        x = torch.zeros(4, 12, EVENT_FIELD_SIZE)
        x[..., C_CS] = torch.randint(0, 64, (4, 12)).float()
        logits = net(x)
        self.assertEqual(set(logits), set(FIELD_KEYS))
        expected = {"op": 2, "size": 12, "csrep": 2, "cpu": 3, "reclaim": 3,
                    "life": LIFE_VOCAB, "dt": DT_VOCAB}
        for key, n_classes in expected.items():
            # (B, L-1, n_classes): position t predicts event t+1
            self.assertEqual(logits[key].shape, (4, 11, n_classes), key)

    def test_out_of_range_call_site_is_clamped(self):
        """cs_hash beyond the embedding table must not raise (hash collisions)."""
        import torch
        net = EventGRUNet(d_model=8, n_layers=1, dropout=0.0, cs_vocab=16)
        x = torch.zeros(2, 6, EVENT_FIELD_SIZE)
        x[..., C_CS] = 9999.0
        logits = net(x)
        self.assertTrue(torch.isfinite(logits["op"]).all())


class TestInputValidation(unittest.TestCase):
    def test_legacy_six_channel_input_rejected(self):
        """The 6-channel layout predates lifecycle/dt_bucket; fail loudly."""
        det = EventGRUDetector(seed=0, epochs=1, batch_size=4, d_model=8, n_layers=1)
        legacy = np.zeros((3, 8, 6), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            det.fit_sequences(legacy)
        self.assertIn("trace2tokens", str(ctx.exception))


class TestEventGRUScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.det = EventGRUDetector(seed=42, epochs=12, batch_size=32, d_model=32,
                                   n_layers=1, cs_vocab=CS_VOCAB, aggregation="p90")
        cls.det.fit_sequences(make_normal_events(n=120, seq_len=32))

    def test_score_shape_and_finiteness(self):
        test = make_normal_events(n=10, seq_len=32, seed=3)
        scores = self.det.sequence_anomaly_score(test)
        self.assertEqual(scores.shape, (10, 32))
        self.assertTrue(np.isfinite(scores).all())
        # The first position has no prediction and must contribute nothing.
        np.testing.assert_array_equal(scores[:, 0], 0.0)

    def test_calibration_stats_cover_every_field(self):
        stats = self.det._score_stats
        self.assertIsNotNone(stats)
        self.assertEqual(set(stats), set(FIELD_KEYS))
        for key, (mean, std) in stats.items():
            self.assertTrue(np.isfinite(mean), key)
            self.assertGreater(std, 0.0, key)

    def test_spray_scores_above_normal(self):
        normal = self.det.sequence_anomaly_score(make_normal_events(n=30, seq_len=32, seed=5))
        spray = self.det.sequence_anomaly_score(make_spray_events(n=30, seq_len=32))
        normal_p90 = np.percentile(normal, 90, axis=1)
        spray_p90 = np.percentile(spray, 90, axis=1)
        self.assertGreater(
            spray_p90.mean(), normal_p90.mean(),
            f"spray={spray_p90.mean():.3f} normal={normal_p90.mean():.3f}")

    def test_groom_lifecycle_scores_above_normal(self):
        """REUSE/FREE_SHORT cycling is anomalous even at normal speed and size."""
        normal = self.det.sequence_anomaly_score(make_normal_events(n=30, seq_len=32, seed=5))
        groom = self.det.sequence_anomaly_score(make_groom_events(n=30, seq_len=32))
        normal_p90 = np.percentile(normal, 90, axis=1)
        groom_p90 = np.percentile(groom, 90, axis=1)
        self.assertGreater(
            groom_p90.mean(), normal_p90.mean(),
            f"groom={groom_p90.mean():.3f} normal={normal_p90.mean():.3f}")


class TestEventGRUSaveLoad(unittest.TestCase):
    def test_save_load_roundtrip(self):
        det = EventGRUDetector(seed=42, epochs=3, batch_size=16, d_model=16,
                               n_layers=1)
        det.fit_sequences(make_normal_events(n=40, seq_len=16))
        test = make_normal_events(n=5, seq_len=16, seed=9)
        before = det.sequence_anomaly_score(test)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as handle:
            det.save(handle.name)
            loaded = EventGRUDetector.load(handle.name)
        os.unlink(handle.name)

        self.assertEqual(loaded._score_stats, det._score_stats)
        self.assertEqual(loaded.w_life, det.w_life)
        np.testing.assert_allclose(before, loaded.sequence_anomaly_score(test),
                                   rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
