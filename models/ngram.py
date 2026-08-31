import numpy as np
from collections import Counter, defaultdict
import pickle
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


SIZE_BUCKETS = config.SIZE_BUCKETS


def event_to_token(event):
    op = event["op"]
    if op == "ALLOC":
        b = event.get("_bucket", 0)
        return f"A_{b}"
    else:
        b = event.get("_bucket", 0)
        return f"F_{b}"


def bucketize_size(size):
    for b in SIZE_BUCKETS:
        if size <= b:
            return b
    return f"gt_{SIZE_BUCKETS[-1]}"


def csv_to_events(csv_path):
    import csv as csv_mod
    events = []
    live_allocations = {}
    with open(csv_path, "r") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            try:
                op = row["op"]
                ptr = row["ptr"]
                if op == "ALLOC":
                    size = int(row.get("bytes_alloc", 0)) or int(row["bytes_req"])
                    live_allocations[ptr] = size
                else:
                    size = live_allocations.pop(ptr, 0)
                events.append({
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "op": op,
                    "_bucket": bucketize_size(size) if size else "unknown",
                    "ptr": ptr,
                })
            except (ValueError, KeyError):
                continue
    events.sort(key=lambda e: e["timestamp_ns"])
    return events


def events_to_tokens(events):
    return [event_to_token(e) for e in events]


class NGramDetector:
    def __init__(self, n=3, max_vocab=500):
        self.n = n
        self.max_vocab = max_vocab
        self.vocab = None
        self.normal_profile = None
        self.normal_freq_norm = None
        self.alpha = 1e-3

    def _build_ngram_counts(self, tokens):
        counts = Counter()
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            counts[ngram] += 1
        return counts

    def fit(self, all_token_lists):
        return self.fit_counts([self._build_ngram_counts(tokens) for tokens in all_token_lists])

    def fit_counts(self, counts_list):
        """Fit from per-run n-gram Counters (streaming-safe).

        Same profile as fit() over the same underlying n-grams, but the caller
        can produce one Counter per run and discard each run's token list, so a
        dataset of ~400 runs / ~1.8GB of traces (final-v2) fits in memory. The
        Counter of distinct n-grams is bounded by the token vocabulary (~26
        event kinds, n<=3), so a per-run Counter is a few KB.
        """
        all_ngrams = Counter()
        for counts in counts_list:
            all_ngrams.update(counts)

        top = all_ngrams.most_common(self.max_vocab)
        self.vocab = {ng: i for i, (ng, _) in enumerate(top)}

        # One additional dimension is reserved for n-grams not observed in the
        # normal vocabulary. Without it, an entirely novel trace scored 0.
        profile = np.zeros(self.max_vocab + 1)
        for counts in counts_list:
            for ng, c in counts.items():
                if ng in self.vocab:
                    profile[self.vocab[ng]] += c

        total = profile.sum()
        self.normal_freq_norm = (profile + self.alpha) / (total + self.alpha * len(profile))
        self.normal_profile = profile
        return self

    def anomaly_score(self, tokens_list):
        return np.asarray([self.anomaly_score_from_counts(self._build_ngram_counts(tokens))
                           for tokens in tokens_list], dtype=np.float64)

    def anomaly_score_from_counts(self, counts):
        """Score one run from its n-gram Counter (same formula as anomaly_score)."""
        vec = np.zeros(self.max_vocab + 1)
        for ng, c in counts.items():
            if ng in self.vocab:
                vec[self.vocab[ng]] += c
            else:
                vec[-1] += c
        total = vec.sum()
        freq = (vec + self.alpha) / (total + self.alpha * len(vec))
        return float(np.sum(freq * np.log(freq / self.normal_freq_norm)))

    def loss_function(self, *args, **kwargs):
        return 0.0, 0.0, 0.0

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "n": self.n,
                "max_vocab": self.max_vocab,
                "vocab": self.vocab,
                "normal_profile": self.normal_profile,
                "normal_freq_norm": self.normal_freq_norm,
                "alpha": self.alpha,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(n=data["n"], max_vocab=data["max_vocab"])
        obj.vocab = data["vocab"]
        obj.normal_profile = data["normal_profile"]
        obj.normal_freq_norm = data["normal_freq_norm"]
        obj.alpha = data.get("alpha", 1e-3)
        return obj
