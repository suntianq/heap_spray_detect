"""M5 adapter: train/score the torch autoencoders through the window/sequence API.

The raw nn.Modules (MLPAE / LSTMAE / LSTMVAE) expose forward + loss_function but no
fit loop, and their anomaly_score consumes torch tensors rather than the numpy
window/sequence matrices the harness feeds. TorchAEWrapper adds the missing glue
without changing the models:

  * fit(windows)            -- window-level training (mlp_ae); scores per window (N,)
  * fit_sequences(sequences) -- sequence-level training (lstm_ae / lstm_vae)
  * anomaly_score(windows)            -- (N, F) numpy -> per-window scores (N,)
  * sequence_anomaly_score(sequences) -- (N, T, F) numpy -> per-window-in-seq (N, T)
    so common.score_sequences can dispatch without the harness knowing the family.

Leak-safety: normalization statistics (mean, std) are computed from the training
input only (identical formula to scripts.train.common.fit_scaler), never from val
or test data. Scoring normalizes with those frozen statistics.

Device: the wrapper auto-detects the compute device at network construction
(torch.device("cuda" if torch.cuda.is_available() else "cpu")) and moves both
parameters and batches there, so deep models use the GPU when present. Classical
sklearn models (ocsvm, isolation_forest, ...) never enter this path and stay on
CPU by design. On CPU, execution stays single-threaded so a fixed seed
reproduces the fit; CUDA training is not bit-reproducible across runs.
"""

import numpy as np

from .lstm_ae import LSTMAE
from .lstm_vae import LSTMVAE
from .mlp_ae import MLPAE

__all__ = ["TorchAEWrapper"]


class TorchAEWrapper:
    def __init__(self, model_kind, seed=42, latent_dim=16, hidden_dims=(128, 64),
                 hidden_dim=64, num_layers=2, epochs=30, lr=1e-3, batch_size=128,
                 beta=1.0, seq_batch_size=64):
        if model_kind not in ("mlp_ae", "lstm_ae", "lstm_vae"):
            raise ValueError(f"unknown torch model kind: {model_kind}")
        self.model_kind = model_kind
        self.seed = seed
        # dispatch marker for the harness: sequence models train on sequences via
        # fit_sequences; window models (mlp_ae) train on windows via fit().
        self.sequence_model = model_kind in ("lstm_ae", "lstm_vae")
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.seq_batch_size = seq_batch_size
        self.net = None
        self.mean = None
        self.std = None
        self.device = None

    # ---- training ----------------------------------------------------------

    def fit(self, windows):
        """Window-level fit (mlp_ae): scaler + net statistics from train windows."""
        self._set_stats(windows)
        feat_dim = int(windows.shape[1])
        self.net = self._make_net(feat_dim)
        self._train(windows, self.batch_size)
        return self

    def fit_sequences(self, sequences):
        """Sequence-level fit (lstm_ae / lstm_vae)."""
        self._set_stats(sequences)
        feat_dim = int(sequences.shape[2])
        seq_len = int(sequences.shape[1])
        self.net = self._make_net(feat_dim, seq_len)
        self._train(sequences, self.seq_batch_size)
        return self

    # ---- scoring -----------------------------------------------------------

    def anomaly_score(self, windows):
        """(N, F) numpy -> per-window anomaly scores (N,)."""
        if self.model_kind != "mlp_ae":
            raise TypeError("anomaly_score(windows) is only for window-level models")
        import torch
        x = self._tensor(windows)
        with torch.no_grad():
            return self.net.anomaly_score(x).cpu().numpy().reshape(-1)

    def sequence_anomaly_score(self, sequences):
        """(N, T, F) numpy -> per-window-in-sequence errors (N, T)."""
        import torch
        x = self._tensor(sequences)
        with torch.no_grad():
            return self.net.anomaly_score(x).cpu().numpy()

    # ---- internals ----------------------------------------------------------

    def _set_stats(self, data):
        flat = data.reshape(-1, data.shape[-1]) if data.ndim == 3 else data
        self.mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0

    def _device(self):
        """Compute device: CUDA when available, else CPU (auto-detected once).

        Self-heals pickles saved before device support: those have no
        self.device attribute, so detect it here and move a cached net to the
        detected device so scoring still works (e.g. a CPU-trained model loaded
        on a GPU machine gets promoted to CUDA).
        """
        if getattr(self, "device", None) is None:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.net is not None:
                net = self.net
                if next(net.parameters()).device.type != self.device.type:
                    self.net = net.to(self.device)
        return self.device

    def _tensor(self, values):
        import torch
        values = np.asarray(values, dtype=np.float32)
        if self.mean is not None:
            values = (values - self.mean) / self.std
        return torch.from_numpy(values).to(self._device())

    def _make_net(self, feat_dim, seq_len=None):
        import torch
        torch.manual_seed(self.seed)
        device = self._device()
        if device.type == "cpu":
            # Keep the CPU path single-threaded so a fixed seed reproduces the fit.
            torch.set_num_threads(1)
        if self.model_kind == "mlp_ae":
            net = MLPAE(feat_dim, hidden_dims=list(self.hidden_dims),
                        latent_dim=self.latent_dim)
        elif self.model_kind == "lstm_ae":
            net = LSTMAE(feat_dim, self.hidden_dim, self.latent_dim, seq_len,
                         num_layers=self.num_layers)
        else:
            net = LSTMVAE(feat_dim, self.hidden_dim, self.latent_dim, seq_len,
                          num_layers=self.num_layers)
        return net.to(device)

    def _train(self, data, batch_size):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        # Zero-copy torch view of the caller's float32 numpy array; normalization
        # happens per-batch below so the full (N,T,F) train tensor is never
        # duplicated as a second multi-GB copy (the 7GB-VM OOM risk).
        x = torch.from_numpy(np.asarray(data, dtype=np.float32))
        mean_t = std_t = None
        if self.mean is not None:
            mean_t = torch.from_numpy(self.mean.astype(np.float32))
            std_t = torch.from_numpy(self.std.astype(np.float32))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(x, x), batch_size=batch_size, shuffle=True)
        self.net.train()
        from tqdm import tqdm
        for _ in tqdm(range(self.epochs), desc=f"{self.model_kind} training", unit="epoch", leave=True):
            for xb, yb in loader:
                if mean_t is not None:  # apply frozen train stats per batch
                    xb = (xb - mean_t) / std_t
                    yb = (yb - mean_t) / std_t
                xb, yb = xb.to(self._device()), yb.to(self._device())
                optimizer.zero_grad()
                if self.model_kind == "lstm_vae":
                    x_recon, mu, logvar = self.net(xb)
                    loss, _, _ = self.net.loss_function(yb, x_recon, mu, logvar, beta=self.beta)
                else:
                    x_recon, _ = self.net(xb)
                    loss = self.net.loss_function(yb, x_recon)
                loss.backward()
                optimizer.step()
        self.net.eval()
