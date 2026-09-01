import numpy as np
from sklearn.svm import OneClassSVM
import pickle
import logging

log = logging.getLogger("ocsvm")

# Batch size for chunked scoring (avoids materializing a huge float64 output
# array for hundreds of thousands of sequences at once).
_SCORE_BATCH = 65536


class OCSVMDetector:
    def __init__(self, kernel="rbf", nu=0.05, gamma="scale"):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.mean = None
        self.std = None

    def fit(self, features, labels=None):
        if labels is not None:
            normal_mask = labels == 0
            if normal_mask.any():
                self.mean = features[normal_mask].mean(axis=0)
                self.std = features[normal_mask].std(axis=0)
        else:
            self.mean = features.mean(axis=0)
            self.std = features.std(axis=0)
        self.std[self.std < 1e-8] = 1.0

        norm_feats = (features - self.mean) / self.std
        log.info("ocsvm fit: %d samples, %d features", norm_feats.shape[0], norm_feats.shape[1])
        self.model.fit(norm_feats)
        log.info("ocsvm fit done")
        return self

    def anomaly_score(self, features):
        norm_feats = (features - self.mean) / self.std
        n = len(norm_feats)
        if n <= _SCORE_BATCH:
            return -self.model.score_samples(norm_feats)
        # Chunk to show progress on large eval sets
        from tqdm import tqdm
        chunks = range(0, n, _SCORE_BATCH)
        parts = []
        for i in tqdm(chunks, total=(n + _SCORE_BATCH - 1) // _SCORE_BATCH,
                      desc="ocsvm scoring", unit="chunk", leave=True):
            parts.append(-self.model.score_samples(norm_feats[i:i + _SCORE_BATCH]))
        return np.concatenate(parts)

    def loss_function(self, *args, **kwargs):
        return 0.0, 0.0, 0.0

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "mean": self.mean, "std": self.std}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.model = data["model"]
        obj.mean = data["mean"]
        obj.std = data["std"]
        return obj
