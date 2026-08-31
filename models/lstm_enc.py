"""LSTM-encoder + density-head anomaly model (M6.5).

The user's proposal: instead of scoring by reconstruction error (LSTM-AE), use an
LSTM *encoder* to learn window embeddings from the 90-dim features and output an
anomaly score per window via a one-class density head. Only normal data is used
for training (unsupervised, one-class), and there is no decoder reconstruction
in the scoring path.

The encoder keeps the per-timestep hidden states `(N, T, 2*hidden)` -- the exact
tensor `LSTMAE.encode` discards (models/lstm_ae.py:24) -- and a density head
(OC-SVM / GMM / IsolationForest) converts each window embedding into a score,
producing the `(N, T)` per-window matrix the harness aggregates (max/last/p90).

Encoder training modes (train_mode):
  * ae_pretrain (default): train the full LSTMAE with the reconstruction
    objective using the memory-safe zero-copy per-batch loop (same as
    TorchAEWrapper._train), then keep only the bidirectional encoder LSTM and
    drop decoder/latent. Scoring NEVER uses reconstruction error, so this
    isolates the hypothesis "embedding density > reconstruction error".
  * svdd: Deep-SVDD one-class objective `loss = mean_t ||h_t - c||^2` directly
    on LSTMEnc, with a frozen center and a collapse guard.

Density heads (density): ocsvm (default, same as the current-best model so the
only variable is the embedding), gmm (literal probability semantics via negative
log-likelihood), isolation_forest (robust/scalable). All score as
`-score_samples(X)` (higher = more anomalous), matching models/ocsvm.py.

Leak safety: mean/std statistics and the density head are fit on TRAINING data
only; the decision threshold is calibrated on validation normal by the harness.
Memory: head fit uses a seeded subsample; scoring batches at embed_batch so the
full (N,T,F) base array is never copied (the harness streams it via
score_sequences_masked).
"""

import numpy as np
import pickle

import torch
import torch.nn as nn


class LSTMEnc(nn.Module):
    """Bidirectional LSTM encoder keeping per-timestep hidden states."""

    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Identical config to LSTMAE.encoder_lstm (lstm_ae.py:13-14) so weights
        # transfer 1:1 in ae_pretrain mode.
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        """(N, T, F) -> (N, T, 2*hidden_dim) per-window hidden embeddings."""
        outputs, _ = self.lstm(x)
        return outputs


class LSTMEncDetector:
    def __init__(self, seed=42, epochs=25, seq_batch_size=64, hidden_dim=64,
                 latent_dim=16, num_layers=2, lr=1e-3, density="ocsvm",
                 train_mode="ae_pretrain", head_subsample=8000,
                 embed_batch=512, svdd_center_batch=64, gmm_components=4,
                 device=None):
        if density not in ("ocsvm", "gmm", "isolation_forest"):
            raise ValueError(f"unknown density head: {density}")
        if train_mode not in ("ae_pretrain", "svdd"):
            raise ValueError(f"unknown train_mode: {train_mode}")
        self.seed = seed
        self.epochs = epochs
        self.seq_batch_size = seq_batch_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.lr = lr
        self.density = density
        self.train_mode = train_mode
        self.head_subsample = int(head_subsample)
        self.embed_batch = int(embed_batch)
        self.svdd_center_batch = int(svdd_center_batch)
        self.gmm_components = gmm_components
        # harness dispatch marker (run_cve_split.py:205 calls fit_sequences)
        self.sequence_model = True
        self.net = None
        self.head = None
        self.mean = None
        self.std = None
        self.device = device

    # ---- training ----------------------------------------------------------

    def fit_sequences(self, sequences):
        """Train the encoder (no decoder in the scoring path), then fit the
        density head on a seeded subsample of TRAINING window embeddings."""
        self._set_stats(sequences)
        if self.train_mode == "ae_pretrain":
            self._train_reconstruction(sequences)
        else:
            self._train_svdd(sequences)
        self.net.eval()
        embeddings = self._collect_head_embeddings(sequences)
        self._fit_head(embeddings)
        return self

    def _train_reconstruction(self, sequences):
        """Train the full LSTMAE (reconstruction), keep only the encoder.

        Memory-safe: zero-copy torch view of the float32 numpy array, frozen
        train stats applied per batch (same pattern as TorchAEWrapper._train).
        """
        from .lstm_ae import LSTMAE
        torch.manual_seed(self.seed)
        device = self._device()
        if device.type == "cpu":
            torch.set_num_threads(1)
        seq_len = int(sequences.shape[1])
        feat_dim = int(sequences.shape[2])
        ae = LSTMAE(feat_dim, self.hidden_dim, self.latent_dim, seq_len,
                    num_layers=self.num_layers).to(device)

        x = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
        mean_t = torch.from_numpy(self.mean.astype(np.float32))
        std_t = torch.from_numpy(self.std.astype(np.float32))
        from torch.utils.data import DataLoader, TensorDataset
        optimizer = torch.optim.Adam(ae.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(x, x), batch_size=self.seq_batch_size,
                            shuffle=True)
        ae.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = (xb - mean_t) / std_t
                yb = (yb - mean_t) / std_t
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                x_recon, _ = ae(xb)
                loss = ae.loss_function(yb, x_recon)
                loss.backward()
                optimizer.step()
        ae.eval()

        # transfer the bidirectional encoder weights 1:1 into the scoring net
        self.net = LSTMEnc(feat_dim, self.hidden_dim,
                           num_layers=self.num_layers).to(device)
        self.net.lstm.load_state_dict(ae.encoder_lstm.state_dict())
        del ae, x
        torch.cuda.empty_cache() if device.type == "cuda" else None

    def _train_svdd(self, sequences):
        """Deep-SVDD: pull window embeddings toward a frozen center.

        Center is the mean embedding over one fixed warmup batch computed with
        init weights (standard DeepSVDD) and never updated. Collapse guard:
        every 5 epochs, if mean distance to center over a probe batch drops
        below 1e-4 the embeddings collapsed (scores constant -> AUC 0.5), so we
        log a warning; the harness should fall back to ae_pretrain.
        """
        torch.manual_seed(self.seed)
        device = self._device()
        if device.type == "cpu":
            torch.set_num_threads(1)
        feat_dim = int(sequences.shape[2])
        self.net = LSTMEnc(feat_dim, self.hidden_dim,
                           num_layers=self.num_layers).to(device)

        x = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
        mean_t = torch.from_numpy(self.mean.astype(np.float32))
        std_t = torch.from_numpy(self.std.astype(np.float32))
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(x), batch_size=self.seq_batch_size,
                            shuffle=True)

        # frozen center from one fixed warmup batch (init weights)
        warmup = torch.from_numpy(
            np.asarray(sequences[:self.svdd_center_batch], dtype=np.float32))
        with torch.no_grad():
            h = self.net((warmup - mean_t) / std_t)
        center = h.reshape(-1, h.shape[-1]).mean(dim=0, keepdim=True).to(device)

        probe_batch = torch.from_numpy(
            np.asarray(sequences[:self.svdd_center_batch], dtype=np.float32))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for epoch in range(self.epochs):
            for (xb,) in loader:
                xb = (xb - mean_t) / std_t
                xb = xb.to(device)
                optimizer.zero_grad()
                h = self.net(xb)
                loss = (h - center).pow(2).mean()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    hb = self.net((probe_batch - mean_t) / std_t)
                dist = (hb - center).pow(2).mean().item()
                if dist < 1e-4:
                    import logging
                    logging.getLogger("lstm_enc").warning(
                        "SVDD collapse: mean dist to center %.2e < 1e-4; "
                        "scores will be constant; fall back to ae_pretrain", dist)
        self.net.eval()
        del x

    # ---- scoring -----------------------------------------------------------

    def sequence_anomaly_score(self, sequences):
        """(N, T, F) -> (N, T) per-window anomaly scores (higher = anomalous)."""
        count = int(sequences.shape[0])
        seq_len = int(sequences.shape[1])
        out = np.empty((count, seq_len), dtype=np.float64)
        for i in range(0, count, self.embed_batch):
            emb = self._embed(sequences[i:i + self.embed_batch])  # (b, T, D)
            b = int(emb.shape[0])
            scores = self._head_score(emb.reshape(-1, emb.shape[-1]))
            out[i:i + b] = scores.reshape(b, seq_len)
        return out

    # ---- internals ----------------------------------------------------------

    def _embed(self, chunk):
        with torch.no_grad():
            x = self._tensor(chunk)
            return self.net(x).cpu().numpy()

    def _tensor(self, values):
        values = np.asarray(values, dtype=np.float32)
        if self.mean is not None:
            values = (values - self.mean) / self.std
        return torch.from_numpy(values).to(self._device())

    def _set_stats(self, data):
        flat = data.reshape(-1, data.shape[-1]) if data.ndim == 3 else data
        self.mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0

    def _device(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.net is not None and next(self.net.parameters()).device.type != self.device.type:
                self.net = self.net.to(self.device)
        return self.device

    def _collect_head_embeddings(self, sequences):
        """Seeded subsample of training window embeddings for the density head.

        The full train set would be ~160k seqs x 32 x 128 x 4B ~ 2.6GB (too big
        on the 7GB VM) and OC-SVM fit is ~O(n^2), so we cap at head_subsample
        windows. All sampling is seeded -> reproducible.
        """
        rng = np.random.default_rng(self.seed)
        n_seqs = int(sequences.shape[0])
        seq_cap = max(1, int(self.head_subsample // 32))
        idx = rng.choice(n_seqs, size=min(n_seqs, seq_cap), replace=False)
        parts = []
        for i in range(0, len(idx), self.embed_batch):
            emb = self._embed(sequences[idx[i:i + self.embed_batch]])
            parts.append(emb.reshape(-1, emb.shape[-1]))
        out = np.concatenate(parts)
        if len(out) > self.head_subsample:
            keep = rng.choice(len(out), self.head_subsample, replace=False)
            out = out[keep]
        return out.astype(np.float32)

    def _make_head(self):
        if self.density == "ocsvm":
            from sklearn.svm import OneClassSVM
            self.head = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
        elif self.density == "gmm":
            from sklearn.mixture import GaussianMixture
            self.head = GaussianMixture(n_components=self.gmm_components,
                                        covariance_type="diag", random_state=self.seed)
        else:
            from sklearn.ensemble import IsolationForest
            self.head = IsolationForest(contamination=0.01, n_estimators=200,
                                        random_state=self.seed)

    def _fit_head(self, embeddings):
        self._make_head()
        self.head.fit(embeddings)

    def _head_score(self, X):
        """-score_samples: higher = more anomalous (matches ocsvm convention)."""
        return -self.head.score_samples(np.asarray(X, dtype=np.float32))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"net": self.net, "head": self.head,
                         "mean": self.mean, "std": self.std}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.net = data["net"]
        obj.head = data["head"]
        obj.mean = data["mean"]
        obj.std = data["std"]
        return obj
