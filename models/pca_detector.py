import numpy as np
from sklearn.decomposition import PCA
import pickle


class PCADetector:
    def __init__(self, n_components=0.95):
        self.n_components = n_components
        self.model = None
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
        self.model = PCA(n_components=self.n_components)
        self.model.fit(norm_feats)
        return self

    def anomaly_score(self, features):
        norm_feats = (features - self.mean) / self.std
        projected = self.model.transform(norm_feats)
        reconstructed = self.model.inverse_transform(projected)
        return np.mean((norm_feats - reconstructed) ** 2, axis=1)

    def loss_function(self, *args, **kwargs):
        return 0.0, 0.0, 0.0

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "mean": self.mean, "std": self.std, "n_components": self.n_components}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(n_components=data["n_components"])
        obj.model = data["model"]
        obj.mean = data["mean"]
        obj.std = data["std"]
        return obj
