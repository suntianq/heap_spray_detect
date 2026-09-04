"""Event-level GRU anomaly detector (phase 2).

DeepLog-style next-token prediction with top-g violation scoring:

  Training: GRU predicts the next token given the previous tokens in the
  sequence. Loss = cross-entropy. Pure self-supervision on normal data.

  Scoring: for each position, if the real token is NOT in the model's
  top-g predicted candidates, it's a "violation". The per-sequence score
  is the violation rate (fraction of positions that violate).

Token sequences come from trace2tokens.py: (N, L) int32 arrays where each
token encodes (op, size_bucket, behavior_type, frequency_rank, dt_bucket).

Why this works for heap spray: the GRU learns that normal event sequences
are diverse (different call_sites, sizes, and timing patterns alternate).
During spray, the same token repeats hundreds of times in a row -- the
model predicts "diverse next tokens" but the real next token is "the same
one again" -- frequent top-g violations.

The dt_bucket field gives the token itself local temporal discriminative
power (spray has <2us intervals, normal has >2us), reducing reliance on
the GRU alone for temporal pattern learning.
"""

import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUDetector:
    """Event-level next-token prediction anomaly detector.

    Architecture: Embedding(vocab -> d_model) + N-layer CAUSAL (unidirectional)
    GRU + Linear(d_model -> vocab). Trained with next-token cross-entropy on
    normal token sequences. Scored with top-g violation rate.

    The GRU must be unidirectional: a bidirectional GRU's output at position t
    concatenates a backward state that has already read token t+1 -- the
    prediction target -- so the head can decode the answer instead of learning
    the normal distribution, flattening the violation signal on normal and
    anomalous input alike.

    The detector exposes sequence_model=True so run_experiment.py dispatches
    to fit_sequences / sequence_anomaly_score. sequence_anomaly_score returns
    (N, L) per-position violation indicators (0.0 or 1.0), which the
    harness aggregates via "mean" (violation rate per sequence).
    """

    def __init__(self, seed=42, vocab_size=13824, d_model=128, n_layers=2,
                 dropout=0.1, lr=1e-3, epochs=20, batch_size=256, g=10,
                 device=None):
        self.seed = seed
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.g = g
        self.device = device
        # harness dispatch marker: sequence_model -> fit_sequences / sequence_anomaly_score
        self.sequence_model = True
        self.net = None

    def _make_net(self):
        torch.manual_seed(self.seed)
        device = self._device()
        if device.type == "cpu":
            torch.set_num_threads(1)
        net = GRUNet(self.vocab_size, self.d_model, self.n_layers, self.dropout)
        return net.to(device)

    def _device(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device

    # ---- training ----------------------------------------------------------

    def fit_sequences(self, token_seqs):
        """Train GRU with next-token cross-entropy on normal token sequences.

        Args:
            token_seqs: (N, L) int32 array of token IDs.
        """
        self.net = self._make_net()
        device = self._device()

        x = torch.from_numpy(np.asarray(token_seqs, dtype=np.int64))
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            drop_last=False)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()

        from tqdm import tqdm
        epoch_pbar = tqdm(range(self.epochs), desc="GRU training", unit="epoch",
                          leave=True)
        for epoch in epoch_pbar:
            total_loss = 0.0
            n_batches = 0
            for (xb,) in loader:
                xb = xb.to(device)
                inp = xb[:, :-1]
                tgt = xb[:, 1:]
                logits = self.net(inp)
                loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)
            epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}")
        epoch_pbar.close()
        self.net.eval()
        return self

    # ---- scoring -----------------------------------------------------------

    def sequence_anomaly_score(self, token_seqs):
        """Score token sequences with top-g violation rate.

        For each position, check if the real token is in the model's top-g
        predicted candidates. If not, it's a violation (1.0), else 0.0.

        Returns (N, L) per-position violation indicators. The harness
        aggregates via "mean" to get per-sequence violation rate.
        """
        device = self._device()
        x = torch.from_numpy(np.asarray(token_seqs, dtype=np.int64))
        from torch.utils.data import DataLoader, TensorDataset
        batch = min(self.batch_size, len(x))
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch, shuffle=False)

        from tqdm import tqdm
        all_violations = []
        with torch.no_grad():
            for (xb,) in tqdm(loader, desc="GRU scoring", unit="batch", leave=True):
                xb = xb.to(device)
                inp = xb[:, :-1]
                tgt = xb[:, 1:]
                logits = self.net(inp)  # (B, L-1, vocab)
                top_g = logits.topk(self.g, dim=-1).indices  # (B, L-1, g)
                tgt_expanded = tgt.unsqueeze(-1)  # (B, L-1, 1)
                in_top_g = (top_g == tgt_expanded).any(dim=-1)  # (B, L-1)
                violations = (~in_top_g).float()  # (B, L-1) 1.0=violation
                first = torch.zeros(xb.size(0), 1, device=device)
                full = torch.cat([first, violations], dim=1)  # (B, L)
                all_violations.append(full.cpu().numpy())

        if not all_violations:
            return np.zeros((len(x), x.size(1)), dtype=np.float64)
        return np.concatenate(all_violations, axis=0)

    # ---- persistence -------------------------------------------------------

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"net_state": self.net.state_dict() if self.net else None,
                         "config": {"vocab_size": self.vocab_size,
                                    "d_model": self.d_model,
                                    "n_layers": self.n_layers,
                                    "dropout": self.dropout,
                                    "g": self.g,
                                    "seed": self.seed}},
                        f)

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
        return obj


class GRUNet(nn.Module):
    """Bidirectional GRU with embedding + prediction head."""

    def __init__(self, vocab_size, d_model, n_layers, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=None)
        self.gru = nn.GRU(
            d_model, d_model, num_layers=n_layers, batch_first=True,
            bidirectional=False, dropout=dropout if n_layers > 1 else 0.0)
        # Causal GRU -> hidden size is d_model; position t sees tokens 0..t only
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """(B, L) int64 -> (B, L, vocab) logits for next-token prediction."""
        emb = self.embedding(x)  # (B, L, d_model)
        out, _ = self.gru(emb)  # (B, L, d_model), causal: out[:, t] sees 0..t
        logits = self.fc(out)  # (B, L, vocab)
        return logits
