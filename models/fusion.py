"""Dual-axis fusion detector (phase 3).

Combines the GRU event-level detector (structure axis) with an ocsvm
window-feature detector (quantity axis) via quantile-aligned score
fusion. The design rationale:

  - Fast spray (<1ms): burst is invisible in 100ms windows but dominates
    the event-level top-g violation rate -> GRU catches it.
  - Slow spray (>50ms): spreads across windows, visible in window features
    (burst_dominant_callsite_ratio, cpu_alloc_entropy) -> ocsvm catches it.
  - msg_msg_2048 false positives: normal and attack share the same
    call_site, so ocsvm sees similar window features; but GRU distinguishes
    the temporal pattern (dispersed vs continuous) -> fusion reduces FPs.

Fusion method: quantile alignment. Both axes produce scores on different
scales (GRU = violation rate [0,1], ocsvm = -score_samples). We align them
to a common scale by mapping each to its percentile rank in the validation
normal distribution, then combine:

  score_final = w1 * quantile_rank(gru_score) + w2 * quantile_rank(ocsvm_score)

This is leak-safe: the percentile mapping is fit on validation normal only,
and the final threshold is calibrated on validation normal at target FPR.
"""

import numpy as np
import pickle

from .gru_detector import GRUDetector
from .ocsvm import OCSVMDetector


class FusionDetector:
    """Dual-axis fusion: GRU (structure) + ocsvm (quantity).

    Exposes sequence_model=True so the harness dispatches to
    fit_sequences / sequence_anomaly_score. Internally it trains both
    sub-models: GRU on token sequences, ocsvm on window features.

    The harness must provide BOTH token sequences and window features.
    This is handled by run_experiment.py which loads token_sequences.npz
    alongside features.npz for fusion models.
    """

    def __init__(self, seed=42, gru_config=None, ocsvm_config=None,
                 w_gru=0.6, w_ocsvm=0.4, device=None):
        self.seed = seed
        self.w_gru = w_gru
        self.w_ocsvm = w_ocsvm
        self.sequence_model = True
        gru_config = gru_config or {}
        ocsvm_config = ocsvm_config or {}
        # Avoid passing seed twice: GRUDetector already takes seed from here
        gru_config = {k: v for k, v in gru_config.items() if k != "seed"}
        self.gru = GRUDetector(seed=seed, **gru_config)
        self.ocsvm = OCSVMDetector(**ocsvm_config)
        self.device = device
        # Percentile lookup arrays (fit on val normal, used for score alignment)
        self.gru_val_scores = None
        self.ocsvm_val_scores = None

    def _device(self):
        if self.device is None:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device

    # ---- training ----------------------------------------------------------

    def fit_sequences(self, token_seqs):
        """Train GRU on token sequences. OCSVM is trained separately
        via fit_windows() — the harness calls both."""
        self.gru.fit_sequences(token_seqs)
        return self

    def fit_windows(self, windows):
        """Train ocsvm on window features (called by harness after fit_sequences)."""
        self.ocsvm.fit(windows)
        return self

    def fit_fusion(self, gru_val_scores, ocsvm_val_scores):
        """Store validation normal scores for quantile alignment.

        Called by the harness after both sub-models are trained and scored
        on the validation set. These score arrays define the percentile
        mapping used to align the two axes.
        """
        self.gru_val_scores = np.sort(np.asarray(gru_val_scores, dtype=np.float64))
        self.ocsvm_val_scores = np.sort(np.asarray(ocsvm_val_scores, dtype=np.float64))

    # ---- scoring -----------------------------------------------------------

    def _quantile_rank(self, scores, ref_sorted):
        """Map each score to its percentile rank in the reference distribution.

        Uses binary search: rank = fraction of ref values below the score.
        Output is in [0, 1], where 0.99 means "higher than 99% of normal".
        """
        scores = np.asarray(scores, dtype=np.float64)
        if len(ref_sorted) == 0:
            return np.zeros_like(scores)
        ranks = np.searchsorted(ref_sorted, scores, side="right") / len(ref_sorted)
        return np.clip(ranks, 0.0, 1.0)

    def sequence_anomaly_score(self, token_seqs):
        """Score token sequences with the GRU axis. Returns (N, L) per-position.

        The harness aggregates via "mean" to get per-sequence violation rate.
        """
        return self.gru.sequence_anomaly_score(token_seqs)

    def window_anomaly_score(self, windows):
        """Score window features with the ocsvm axis. Returns (N,)."""
        return np.asarray(self.ocsvm.anomaly_score(
            np.asarray(windows, dtype=np.float32)), dtype=np.float64)

    def fuse_scores(self, gru_seq_scores, ocsvm_window_scores, gru_run_ids, ocsvm_run_ids):
        """Fuse GRU sequence scores and ocsvm window scores into run-level scores.

        Both inputs are per-sequence / per-window arrays. We aggregate to
        run-level (max per run) for both axes, then quantile-align and
        weight-combine.

        Returns (run_scores, run_ids).
        """
        gru_run_scores, gru_ids = _run_max(gru_seq_scores, gru_run_ids)
        ocs_run_scores, ocs_ids = _run_max(ocsvm_window_scores, ocsvm_run_ids)

        # Align: both run_ids should be the same set, but order may differ
        # Build a unified run_id -> fused score map
        all_run_ids = sorted(set(gru_ids.tolist()) | set(ocs_ids.tolist()))
        gru_map = dict(zip(gru_ids, gru_run_scores))
        ocs_map = dict(zip(ocs_ids, ocs_run_scores))

        gru_aligned = np.array([gru_map.get(rid, 0.0) for rid in all_run_ids])
        ocs_aligned = np.array([ocs_map.get(rid, 0.0) for rid in all_run_ids])

        # Quantile-align each axis to [0, 1] using val normal reference
        gru_q = self._quantile_rank(gru_aligned, self.gru_val_scores)
        ocs_q = self._quantile_rank(ocs_aligned, self.ocsvm_val_scores)

        # Weighted combination
        fused = self.w_gru * gru_q + self.w_ocsvm * ocs_q
        return fused, np.array(all_run_ids, dtype=object)

    # ---- persistence -------------------------------------------------------

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "gru": {"net_state": self.gru.net.state_dict() if self.gru.net else None,
                         "config": {"vocab_size": self.gru.vocab_size,
                                    "d_model": self.gru.d_model,
                                    "n_layers": self.gru.n_layers,
                                    "dropout": self.gru.dropout,
                                    "g": self.gru.g,
                                    "seed": self.gru.seed}},
                "ocsvm": {"model": self.ocsvm.model,
                           "mean": self.ocsvm.mean,
                           "std": self.ocsvm.std},
                "gru_val_scores": self.gru_val_scores,
                "ocsvm_val_scores": self.ocsvm_val_scores,
                "config": {"w_gru": self.w_gru, "w_ocsvm": self.w_ocsvm,
                           "seed": self.seed},
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        cfg = data["config"]
        gru_cfg = data["gru"]["config"]
        ocs_cfg = {"kernel": "rbf", "nu": 0.05, "gamma": "scale"}
        obj = cls(seed=cfg["seed"], gru_config=gru_cfg, ocsvm_config=ocs_cfg,
                  w_gru=cfg["w_gru"], w_ocsvm=cfg["w_ocsvm"])
        if data["gru"]["net_state"] is not None:
            obj.gru.net = obj.gru._make_net()
            obj.gru.net.load_state_dict(data["gru"]["net_state"])
            obj.gru.net.eval()
        obj.ocsvm.model = data["ocsvm"]["model"]
        obj.ocsvm.mean = data["ocsvm"]["mean"]
        obj.ocsvm.std = data["ocsvm"]["std"]
        obj.gru_val_scores = data["gru_val_scores"]
        obj.ocsvm_val_scores = data["ocsvm_val_scores"]
        return obj


def _run_max(scores, groups):
    """Per-run max score aggregation (same logic as common.run_max_scores)."""
    out, ids = [], []
    for group in sorted({str(g) for g in groups}):
        mask = np.asarray(groups).astype(str) == group
        out.append(float(np.asarray(scores)[mask].max()))
        ids.append(group)
    return np.asarray(out, dtype=np.float64), np.asarray(ids, dtype=object)
