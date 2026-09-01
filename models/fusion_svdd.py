"""Unified GRU + Deep SVDD detector (all-GPU, single training pass).

Replaces the two-model fusion (GRU + OCSVM) with a single network that has
two heads sharing the same GRU encoder:

  Head 1 — Next-token prediction (structural axis):
    GRU hidden state → Linear → vocab logits → top-g violation rate.
    Learns "what token comes next" on normal data. Spray sequences
    (repeated tokens) cause frequent violations.

  Head 2 — Deep SVDD distance (quantity axis):
    GRU hidden state → mean-pool → Linear → embedding → ||emb - center||².
    Learns a hypersphere boundary around normal embeddings. Anomalous
    sequences land outside the sphere (large distance).

Training: joint loss = L_next_token + λ × L_svdd
  - L_next_token = cross-entropy (predict token t+1 given t)
  - L_svdd = mean ||embedding - center||² (pull normal toward center)
  - center c = frozen mean embedding of first training batch (standard DeepSVDD)

Scoring: per-position combined score = violation(0/1) + α × normalized_svdd
  - With "mean" aggregation → violation_rate + α × normalized_svdd_distance
  - SVDD distance normalized by training p99 to [0, 1]

Advantages over fusion (GRU + OCSVM):
  - Single training pass (no separate OCSVM fit)
  - Fully on GPU (no sklearn in the loop)
  - Single input (token sequences only, no window features needed)
  - Joint optimization (two heads learn complementary features together)
"""

import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionSVDDDetector:
    """Unified GRU + Deep SVDD detector.

    Exposes sequence_model=True so the harness dispatches to
    fit_sequences / sequence_anomaly_score.

    sequence_anomaly_score returns (N, L) per-position combined scores:
    violation_indicator + score_svdd_weight * normalized_svdd_distance.
    The harness aggregates via "mean" to get per-sequence:
    violation_rate + score_svdd_weight * normalized_svdd_distance.
    """

    def __init__(self, seed=42, vocab_size=1536, d_model=128, n_layers=2,
                 dropout=0.1, lr=1e-3, epochs=20, batch_size=256, g=10,
                 svdd_loss_weight=0.1, svdd_score_weight=0.3,
                 svdd_dim=None, device=None):
        self.seed = seed
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.g = g
        self.svdd_loss_weight = svdd_loss_weight
        self.svdd_score_weight = svdd_score_weight
        self.svdd_dim = svdd_dim or d_model  # projection dim for SVDD
        self.device = device
        self.sequence_model = True
        self.net = None
        self.svdd_center = None        # frozen center (svdd_dim,)
        self.svdd_p99 = 1.0            # normalization factor (updated after training)

    def _make_net(self):
        torch.manual_seed(self.seed)
        device = self._device()
        if device.type == "cpu":
            torch.set_num_threads(1)
        net = FusionSVDDNet(self.vocab_size, self.d_model, self.n_layers,
                            self.dropout, self.svdd_dim)
        return net.to(device)

    def _device(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device

    # ---- training ----------------------------------------------------------

    def fit_sequences(self, token_seqs):
        """Train with joint next-token + SVDD loss on normal token sequences."""
        self.net = self._make_net()
        device = self._device()

        x = torch.from_numpy(np.asarray(token_seqs, dtype=np.int64))
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # ---- Initialize SVDD center from first batch (standard DeepSVDD) ----
        with torch.no_grad():
            first_batch = x[:min(self.batch_size, len(x))].to(device)
            _, svdd_emb = self.net(first_batch)
            self.svdd_center = svdd_emb.mean(dim=0).detach().clone()

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()

        from tqdm import tqdm
        epoch_pbar = tqdm(range(self.epochs), desc="FusionSVDD training",
                          unit="epoch", leave=True)

        for epoch in epoch_pbar:
            total_loss = 0.0
            total_ce = 0.0
            total_svdd = 0.0
            n_batches = 0

            for (xb,) in loader:
                xb = xb.to(device)
                inp = xb[:, :-1]
                tgt = xb[:, 1:]

                next_logits, svdd_emb = self.net(inp)  # (B, L-1, vocab), (B, svdd_dim)

                # Head 1: next-token cross-entropy
                ce_loss = F.cross_entropy(
                    next_logits.reshape(-1, self.vocab_size), tgt.reshape(-1))

                # Head 2: SVDD distance (pull toward frozen center)
                svdd_loss = ((svdd_emb - self.svdd_center) ** 2).sum(dim=1).mean()

                # Joint loss
                loss = ce_loss + self.svdd_loss_weight * svdd_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_ce += ce_loss.item()
                total_svdd += svdd_loss.item()
                n_batches += 1

            epoch_pbar.set_postfix(
                loss=f"{total_loss/max(n_batches,1):.4f}",
                ce=f"{total_ce/max(n_batches,1):.4f}",
                svdd=f"{total_svdd/max(n_batches,1):.4f}")

            # Collapse guard: if SVDD distance drops to ~0, embeddings collapsed
            if (epoch + 1) % 5 == 0 and total_svdd / max(n_batches, 1) < 1e-4:
                import logging
                logging.getLogger("fusion_svdd").warning(
                    "SVDD collapse: mean distance %.2e < 1e-4; "
                    "SVDD head may be uninformative", total_svdd / max(n_batches, 1))

        epoch_pbar.close()
        self.net.eval()

        # ---- Compute SVDD p99 on training data for score normalization ----
        all_svdd_dist = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device)
                inp = xb[:, :-1]
                _, svdd_emb = self.net(inp)
                dist = ((svdd_emb - self.svdd_center) ** 2).sum(dim=1)
                all_svdd_dist.append(dist.cpu().numpy())
        if all_svdd_dist:
            combined = np.concatenate(all_svdd_dist)
            self.svdd_p99 = float(np.percentile(combined, 99))
            if self.svdd_p99 < 1e-8:
                self.svdd_p99 = 1.0  # avoid division by zero

        return self

    # ---- scoring -----------------------------------------------------------

    def sequence_anomaly_score(self, token_seqs):
        """Score token sequences: combined violation + SVDD distance.

        Returns (N, L) per-position scores. With "mean" aggregation:
        per-sequence = violation_rate + score_svdd_weight * normalized_svdd
        """
        device = self._device()
        x = torch.from_numpy(np.asarray(token_seqs, dtype=np.int64))
        from torch.utils.data import DataLoader, TensorDataset
        batch = min(self.batch_size, len(x))
        loader = DataLoader(TensorDataset(x), batch_size=batch, shuffle=False)

        from tqdm import tqdm
        all_scores = []
        with torch.no_grad():
            for (xb,) in tqdm(loader, desc="FusionSVDD scoring", unit="batch", leave=True):
                xb = xb.to(device)
                inp = xb[:, :-1]
                tgt = xb[:, 1:]

                next_logits, svdd_emb = self.net(inp)

                # Head 1: top-g violation
                top_g = next_logits.topk(self.g, dim=-1).indices  # (B, L-1, g)
                tgt_expanded = tgt.unsqueeze(-1)
                in_top_g = (top_g == tgt_expanded).any(dim=-1)  # (B, L-1)
                violations = (~in_top_g).float()  # (B, L-1) 0.0 or 1.0

                # Head 2: SVDD distance (per-sequence, broadcast to positions)
                svdd_dist = ((svdd_emb - self.svdd_center) ** 2).sum(dim=1)  # (B,)
                svdd_normalized = (svdd_dist / self.svdd_p99).clamp(0, 1)  # (B,) in [0,1]

                # Combine: violation + alpha * normalized_svdd (broadcast to L-1 positions)
                combined = violations + self.svdd_score_weight * svdd_normalized.unsqueeze(1)

                # Pad first position (no prediction for token 0)
                first = torch.full((xb.size(0), 1),
                                   self.svdd_score_weight * svdd_normalized.mean().item(),
                                   device=device)
                full = torch.cat([first, combined], dim=1)  # (B, L)
                all_scores.append(full.cpu().numpy())

        if not all_scores:
            return np.zeros((len(x), x.size(1)), dtype=np.float64)
        return np.concatenate(all_scores, axis=0)

    # ---- persistence -------------------------------------------------------

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "net_state": self.net.state_dict() if self.net else None,
                "svdd_center": self.svdd_center.cpu() if self.svdd_center is not None else None,
                "svdd_p99": self.svdd_p99,
                "config": {
                    "vocab_size": self.vocab_size,
                    "d_model": self.d_model,
                    "n_layers": self.n_layers,
                    "dropout": self.dropout,
                    "g": self.g,
                    "svdd_loss_weight": self.svdd_loss_weight,
                    "svdd_score_weight": self.svdd_score_weight,
                    "svdd_dim": self.svdd_dim,
                    "seed": self.seed,
                },
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        cfg = data["config"]
        obj = cls(**cfg)
        if data["net_state"] is not None:
            obj.net = obj._make_net()
            obj.net.load_state_dict(data["net_state"])
            obj.net.eval()
        if data["svdd_center"] is not None:
            obj.svdd_center = data["svdd_center"].to(obj._device())
        obj.svdd_p99 = data["svdd_p99"]
        return obj


class FusionSVDDNet(nn.Module):
    """Bidirectional GRU with two heads: next-token prediction + SVDD embedding."""

    def __init__(self, vocab_size, d_model, n_layers, dropout, svdd_dim):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(
            d_model, d_model, num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        # Head 1: next-token prediction
        self.next_token_head = nn.Linear(2 * d_model, vocab_size)
        # Head 2: SVDD projection (project to lower-dim for tighter hypersphere)
        self.svdd_head = nn.Linear(2 * d_model, svdd_dim)

    def forward(self, x):
        """(B, L) int64 -> (next_logits (B,L,vocab), svdd_emb (B, svdd_dim))"""
        emb = self.embedding(x)          # (B, L, d_model)
        out, _ = self.gru(emb)           # (B, L, 2*d_model)
        next_logits = self.next_token_head(out)  # (B, L, vocab)
        svdd_emb = self.svdd_head(out)           # (B, L, svdd_dim)
        # Mean-pool over time for sequence-level SVDD embedding
        svdd_seq = svdd_emb.mean(dim=1)          # (B, svdd_dim)
        return next_logits, svdd_seq
