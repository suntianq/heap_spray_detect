"""Event-embedding GRU anomaly detector (field-decomposed, phase 3).

Replaces the discrete-symbol GRU (gru_detector.py). Where the old model
collapsed every event into ONE opaque token id over a 13824-word vocab, this
model decomposes each event into its constituent FIELDS, each mapped to its
own learnable embedding (or a continuous scalar), and feeds the concatenated
vector to a bidirectional GRU.

Not an autoencoder. Training is still next-event self-supervision -- predict
the next event's fields from the preceding events -- but the prediction target
is a FIELD-WISE decomposition instead of one opaque symbol. Every head is a
classifier over its own small per-field vocabulary:

  Field    classes  encoding
  -------  -------  ---------------------------------------------------------
  op       2        ALLOC / FREE
  size     12       kmalloc size class (32,64,...,8192,gt_8192)
  csrep    2        "does the next event reuse the current call_site?"
  cpu      3        CPU concentration bucket
  reclaim  3        none / same-site / cross-site reclaim
  life     6        NEW, REUSE, FREE_SHORT, FREE_MEDIUM, FREE_LONG, UNKNOWN
  dt       6        inter-event delta bucket (<2us, ..., >10ms)

Why field decomposition fixes the old model's weaknesses:

  1. Vocab 13824 -> per-field vocab <=12. Normal next-event predictions rarely
     miss; the fragile top-10-over-13824 test (high FPR) is gone.
  2. behavior_type / frequency_rank are DROPPED. They were computed from the
     global normal distribution and leaked a "rare == anomalous" prior into the
     token, which is exactly what produced the run_FPR=0.09 on allocation-heavy
     workloads (fork_stress > idle > mem_pressure).
  3. call_site becomes a stable-hash embedding slot (CS_VOCAB), so unseen
     call_sites are a fresh embedding, not a "rare" penalty. No global vocab.
  4. The csrep head literally models "the next event repeats the same
     call_site", which is the dominant spray signature (same call_site, same
     size, <2us apart, hundreds of times).
  5. The life head carries the object lifecycle recovered from ptr tracking.
     Measured on the CVE-first dataset, spray sequences run 53% NEW / 4% REUSE
     against 21% NEW / 26% REUSE for normal traffic -- spray allocates without
     ever giving objects back, which op+size+dt cannot express on their own.

Why dt is a CLASSIFIER, not a regressor: log1p compresses the short-delta
regime into a sliver of the output range, so an MSE objective spends most of
its gradient on the common millisecond-scale gaps. A 6-way bucket head gives
each decade its own logit, so a misprediction produces a large, well-calibrated
NLL. (The separating band is the 100us-1ms bucket, ~6% of normal deltas vs ~30%
during spray; sub-2us alone does not separate -- normal traffic is ~52% sub-2us
as well.) The continuous log1p(dt) is still fed as an INPUT channel -- it is the
finer-grained signal -- it is simply no longer the prediction target.

Input layout (from trace2tokens.cut_event_sequences): (N, L, 8) float32
  [op, size_bucket, call_site_hash, cpu_bucket, reclaim_flag,
   lifecycle, dt_bucket, dt_cont]
Channels 0-6 are integer indices (stored as float), channel 7 is the
continuous dt. The model converts channels 0-6 to long for embedding lookup.

Interface is identical to GRUDetector: exposes sequence_model=True with
fit_sequences / sequence_anomaly_score, so run_experiment.py dispatches to it
unchanged. sequence_anomaly_score returns (N, L) per-position NLL-surprise
scores; the harness aggregates via "p90" by default (see DEFAULT_AGGREGATION).

Scoring (unlike the old top-g hit/miss vote): each field's per-position
cross-entropy NLL is z-scored against its training distribution and
relu-clipped, then weighted-summed. A 0.99-confidence misprediction scores far
above a 0.51 one, so the score is a continuous "surprise" rather than a binary
vote. See sequence_anomaly_score.
"""

import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

# Per-field vocabulary sizes. Must match the layout trace2tokens writes.
OP_VOCAB = 2
SIZE_VOCAB = 12
CPU_VOCAB = 3
RECLAIM_VOCAB = 3
LIFE_VOCAB = 6
DT_VOCAB = 6

# Field embedding sizes. Chosen small: the fields are few-valued, and the
# GRU is what carries temporal structure. Total = 76 + 1 (dt_cont) = 77.
OP_DIM = 4
SIZE_DIM = 16
CS_DIM = 32
CPU_DIM = 4
RECLAIM_DIM = 4
LIFE_DIM = 8
DT_DIM = 8
FIELD_EMB_DIM = (OP_DIM + SIZE_DIM + CS_DIM + CPU_DIM + RECLAIM_DIM
                 + LIFE_DIM + DT_DIM)  # 76

# Number of channels in the input field matrix (see module docstring).
EVENT_FIELD_SIZE = 8

# All scored fields, in the order their weights are declared.
FIELD_KEYS = ("op", "size", "csrep", "cpu", "reclaim", "life", "dt")

# call_site hash table size. Actual call_sites number in the dozens; 4096
# slots keep collisions negligible while never demanding a global vocab.
CS_VOCAB = 4096

# dt continuation scale: log1p(dt_us) / log1p(1e9). 1e9 us = ~16.7 min, a
# generous upper bound; maps dt into roughly [0, 1] with no data-dependent
# statistics (leak-safe). dt=0 for a sequence's first event.
LOG_DT_NORM = float(np.log1p(1e9))


# ---------------------------------------------------------------------------
# Field helpers (mirror the layout produced by trace2tokens)
# ---------------------------------------------------------------------------

def dt_to_continuous(dt_ns):
    """Map an inter-event delta (ns) to a continuous scalar in ~[0, 1]."""
    dt_us = max(dt_ns, 0) / 1000.0
    return float(np.log1p(dt_us) / LOG_DT_NORM)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EventGRUNet(nn.Module):
    """Field-embedding GRU with per-field next-event prediction heads.

    x: (B, L, 8) float32 with channels
    [op, size, cs_hash, cpu, reclaim, life, dt_bucket, dt_cont].
    Returns a dict of per-field logits over positions 0..L-2 (predicting
    events 1..L-1).
    """

    def __init__(self, d_model, n_layers, dropout, cs_vocab=CS_VOCAB):
        super().__init__()
        self.d_model = d_model
        self.cs_vocab = cs_vocab

        self.op_emb = nn.Embedding(OP_VOCAB, OP_DIM)
        self.size_emb = nn.Embedding(SIZE_VOCAB, SIZE_DIM)
        self.cs_emb = nn.Embedding(cs_vocab, CS_DIM)
        self.cpu_emb = nn.Embedding(CPU_VOCAB, CPU_DIM)
        self.reclaim_emb = nn.Embedding(RECLAIM_VOCAB, RECLAIM_DIM)
        self.life_emb = nn.Embedding(LIFE_VOCAB, LIFE_DIM)
        self.dt_emb = nn.Embedding(DT_VOCAB, DT_DIM)

        # Project field-embedding concat + continuous dt to GRU input dim.
        self.field_proj = nn.Linear(FIELD_EMB_DIM + 1, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.gru = nn.GRU(
            d_model, d_model, num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)

        # Prediction heads (bidirectional GRU -> 2*d_model)
        self.op_head = nn.Linear(2 * d_model, OP_VOCAB)
        self.size_head = nn.Linear(2 * d_model, SIZE_VOCAB)
        self.csrep_head = nn.Linear(2 * d_model, 2)
        self.cpu_head = nn.Linear(2 * d_model, CPU_VOCAB)
        self.reclaim_head = nn.Linear(2 * d_model, RECLAIM_VOCAB)
        self.life_head = nn.Linear(2 * d_model, LIFE_VOCAB)
        self.dt_head = nn.Linear(2 * d_model, DT_VOCAB)

    def _embed_fields(self, x):
        """x (B, L, 8) -> (B, L, FIELD_EMB_DIM+1)."""
        op = x[..., 0].long()
        size = x[..., 1].long()
        cs = x[..., 2].long().clamp(0, self.cs_vocab - 1)
        cpu = x[..., 3].long()
        reclaim = x[..., 4].long()
        life = x[..., 5].long()
        dt_bucket = x[..., 6].long()
        dt_cont = x[..., 7:8]  # keep channel dim

        emb = torch.cat([
            self.op_emb(op),
            self.size_emb(size),
            self.cs_emb(cs),
            self.cpu_emb(cpu),
            self.reclaim_emb(reclaim),
            self.life_emb(life),
            self.dt_emb(dt_bucket),
            dt_cont,
        ], dim=-1)  # (B, L, 77)
        return emb

    def forward(self, x):
        """(B, L, 8) -> dict of next-field logits. Positions 0..L-2 predict 1..L-1."""
        emb = self._embed_fields(x)
        h = self.norm(self.field_proj(emb))  # (B, L, d_model)
        out, _ = self.gru(h)      # (B, L, 2*d_model)
        # Predict next event: out[..., :-1, :] -> logits for events[1:]
        hn = out[:, :-1]
        return {
            "op": self.op_head(hn),
            "size": self.size_head(hn),
            "csrep": self.csrep_head(hn),
            "cpu": self.cpu_head(hn),
            "reclaim": self.reclaim_head(hn),
            "life": self.life_head(hn),
            "dt": self.dt_head(hn),
        }


class EventGRUDetector:
    """Field-decomposed next-event prediction anomaly detector.

    Exposes sequence_model=True; fit_sequences / sequence_anomaly_score take
    (N, L, 8) float32 field matrices. sequence_anomaly_score returns (N, L)
    per-position surprise scores; the harness aggregates via "p90".
    """

    def __init__(self, seed=42, d_model=128, n_layers=2, dropout=0.1,
                 lr=1e-3, epochs=20, batch_size=256, g_size=3,
                 cs_vocab=CS_VOCAB, dt_loss_weight=1.0,
                 w_op=0.05, w_size=0.25, w_csrep=0.15, w_cpu=0.05,
                 w_reclaim=0.05, w_life=0.20, w_dt=0.25, device=None,
                 label_smoothing=0.05, aggregation="p90"):
        self.seed = seed
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.g_size = g_size  # retained for config compatibility; scoring is NLL-based
        self.cs_vocab = cs_vocab
        self.dt_loss_weight = dt_loss_weight
        # Score weights, one per FIELD_KEYS entry. life carries the grooming /
        # UAF reuse fingerprint, so it takes a share comparable to size and dt.
        self.w_op = w_op
        self.w_size = w_size
        self.w_csrep = w_csrep
        self.w_cpu = w_cpu
        self.w_reclaim = w_reclaim
        self.w_life = w_life
        self.w_dt = w_dt
        self.device = device
        self.label_smoothing = label_smoothing
        # Per-position -> per-sequence aggregation. max is fragile (one noisy
        # position flags a whole burst); mean washes a short spray window out of
        # a long run. An upper quantile keeps "part of the run was anomalous"
        # while discarding single-position spikes.
        self.aggregation = aggregation
        self.sequence_model = True
        self.net = None
        self.feat_dim = EVENT_FIELD_SIZE  # (N, L, 8) field matrix
        self.n_fields = len(FIELD_KEYS)
        # Per-field NLL calibration stats (mean/std over training normal runs).
        # Filled by fit_sequences / calibrate_scoring. None until then.
        self._score_stats = None

    def field_weights(self):
        """Score weight per FIELD_KEYS entry, in order."""
        return {
            "op": self.w_op, "size": self.w_size, "csrep": self.w_csrep,
            "cpu": self.w_cpu, "reclaim": self.w_reclaim,
            "life": self.w_life, "dt": self.w_dt,
        }

    def _as_input(self, event_fields):
        """Validate and coerce an (N, L, 8) field matrix to a float32 tensor."""
        array = np.asarray(event_fields, dtype=np.float32)
        if array.ndim != 3 or array.shape[-1] != EVENT_FIELD_SIZE:
            raise ValueError(
                f"event_gru expects (N, L, {EVENT_FIELD_SIZE}) field matrices "
                f"[op, size, cs_hash, cpu, reclaim, life, dt_bucket, dt_cont], "
                f"got shape {array.shape}. Regenerate event_fields.npz with "
                f"scripts/preprocess/trace2tokens.py.")
        return torch.from_numpy(array)

    def _make_net(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = self._device()
        if device.type == "cpu":
            torch.set_num_threads(1)
        net = EventGRUNet(self.d_model, self.n_layers, self.dropout,
                          self.cs_vocab)
        return net.to(device)

    def _device(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)
        return self.device

    # ---- training ----------------------------------------------------------

    def fit_sequences(self, event_fields):
        """Train field-wise next-event prediction on normal event matrices.

        Args:
            event_fields: (N, L, 8) float32 field matrix.
        """
        self.net = self._make_net()
        device = self._device()

        x = self._as_input(event_fields)
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            drop_last=False)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3)

        # dt is weighted separately so dt_loss_weight keeps its meaning now that
        # the head is a classifier rather than a regressor.
        loss_weights = {k: 1.0 for k in FIELD_KEYS}
        loss_weights["dt"] = self.dt_loss_weight

        from tqdm import tqdm
        epoch_pbar = tqdm(range(self.epochs), desc="EventGRU training",
                          unit="epoch", leave=True)
        for epoch in epoch_pbar:
            total_loss = 0.0
            n_batches = 0
            for (xb,) in loader:
                xb = xb.to(device)
                logits = self.net(xb)  # predict events[1:]
                targets = self._field_targets(xb)

                loss = sum(
                    loss_weights[key] * F.cross_entropy(
                        logits[key].reshape(-1, logits[key].shape[-1]),
                        targets[key].reshape(-1),
                        label_smoothing=self.label_smoothing)
                    for key in FIELD_KEYS)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)
            epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}")
            scheduler.step(avg_loss)
        epoch_pbar.close()
        self.net.eval()
        # Calibrate per-field z-score stats on the same normal training set.
        self._compute_score_stats(event_fields)
        return self

    # ---- scoring -----------------------------------------------------------

    def _compute_score_stats(self, event_fields):
        """Fit per-field NLL calibration stats on normal training sequences.

        For each field, collect the per-position NLL over the reference set and
        store (mean, std). Scoring then z-scores each field's surprise so the
        seven fields are on a comparable scale before the weighted sum -- a
        binary hit/miss vote cannot do this (it treats a 0.51-confidence miss
        the same as a 0.99 miss).

        Called at the end of fit_sequences. Stats are saved/loaded with the
        model so scoring never needs the training data again.
        """
        device = self._device()
        x = self._as_input(event_fields)
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)

        sums = {k: 0.0 for k in FIELD_KEYS}
        sumsq = {k: 0.0 for k in FIELD_KEYS}
        counts = {k: 0 for k in FIELD_KEYS}
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device)
                out = self._field_surprise(xb)
                for k in FIELD_KEYS:
                    v = out[k].detach().cpu().numpy()
                    sums[k] += float(v.sum())
                    sumsq[k] += float((v * v).sum())
                    counts[k] += v.size
        if not any(counts.values()):
            self._score_stats = None
            return self
        stats = {}
        for k in FIELD_KEYS:
            n = max(counts[k], 1)
            mean = sums[k] / n
            var = sumsq[k] / n - mean * mean
            stats[k] = (float(mean), float(max(var, 1e-12) ** 0.5))
        self._score_stats = stats
        return self

    def _field_targets(self, xb):
        """Per-field next-event targets for a batch (B, L, 8).

        Every field is categorical: the target is the next event's class index,
        except csrep, which is the derived "next event reuses this call_site".
        Returns a dict of (B, L-1) long tensors keyed by FIELD_KEYS.
        """
        inp = xb[:, :-1]      # (B, L-1, 8)
        tgt = xb[:, 1:]       # (B, L-1, 8)
        return {
            "op": tgt[..., 0].long(),
            "size": tgt[..., 1].long(),
            "csrep": (tgt[..., 2].long() == inp[..., 2].long()).long(),
            "cpu": tgt[..., 3].long(),
            "reclaim": tgt[..., 4].long(),
            "life": tgt[..., 5].long(),
            "dt": tgt[..., 6].long(),
        }

    def _field_surprise(self, xb):
        """Per-position per-field surprise for a batch (B, L, 8).

        Returns a dict of (B, L-1) float tensors, one per FIELD_KEYS entry,
        each holding that head's cross-entropy NLL for the observed next event.
        """
        logits = self.net(xb)
        targets = self._field_targets(xb)
        out = {}
        for key in FIELD_KEYS:
            head = logits[key]
            target = targets[key]
            out[key] = F.cross_entropy(
                head.reshape(-1, head.shape[-1]), target.reshape(-1),
                reduction="none").reshape(target.shape)
        return out

    def _zscore(self, values, key):
        """z-score a per-position surprise tensor for one field, relu-clipped."""
        stats = self._score_stats
        if not stats or key not in stats:
            return values  # uncalibrated fallback (model loaded w/o stats)
        mean, std = stats[key]
        z = (values - mean) / std
        return torch.relu(z)

    def sequence_anomaly_score(self, event_fields):
        """Score field matrices with calibrated per-field NLL surprise.

        Per-position score = weighted sum of per-field z-scored NLL surprise
        (relu-clipped so below-normal surprise contributes 0). Unlike the old
        top-g hit/miss vote, this keeps the magnitude of a misprediction: a
        0.99-confidence miss scores far higher than a 0.51 one. Returns (N, L).
        """
        device = self._device()
        x = self._as_input(event_fields)
        from torch.utils.data import DataLoader, TensorDataset
        batch = min(self.batch_size, len(x))
        loader = DataLoader(TensorDataset(x), batch_size=batch, shuffle=False)

        weights = self.field_weights()
        from tqdm import tqdm
        all_scores = []
        with torch.no_grad():
            for (xb,) in tqdm(loader, desc="EventGRU scoring", unit="batch", leave=True):
                xb = xb.to(device)
                out = self._field_surprise(xb)

                per_pos = sum(weights[key] * self._zscore(out[key], key)
                              for key in FIELD_KEYS)  # (B, L-1)

                # Pad the first position (no prediction available)
                first = torch.zeros(xb.size(0), 1, device=device)
                full = torch.cat([first, per_pos], dim=1)  # (B, L)
                all_scores.append(full.cpu().numpy())

        if not all_scores:
            return np.zeros((len(x), x.size(1)), dtype=np.float64)
        return np.concatenate(all_scores, axis=0)

    # ---- persistence -------------------------------------------------------

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {"net_state": self.net.state_dict() if self.net else None,
                 "score_stats": self._score_stats,
                 "config": {"seed": self.seed, "d_model": self.d_model,
                            "n_layers": self.n_layers, "dropout": self.dropout,
                            "lr": self.lr, "epochs": self.epochs,
                            "batch_size": self.batch_size, "g_size": self.g_size,
                            "cs_vocab": self.cs_vocab,
                            "dt_loss_weight": self.dt_loss_weight,
                            "w_op": self.w_op, "w_size": self.w_size,
                            "w_csrep": self.w_csrep, "w_cpu": self.w_cpu,
                            "w_reclaim": self.w_reclaim, "w_life": self.w_life,
                            "w_dt": self.w_dt,
                            "label_smoothing": self.label_smoothing,
                            "aggregation": self.aggregation}},
                f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        cfg = data["config"]
        obj = cls(**cfg)
        obj._score_stats = data.get("score_stats")
        if data["net_state"] is not None:
            obj.net = obj._make_net()
            obj.net.load_state_dict(data["net_state"])
            obj.net.eval()
        return obj
