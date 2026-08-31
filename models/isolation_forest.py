import numpy as np
from sklearn.ensemble import IsolationForest
import pickle


class IsolationForestDetector:
    def __init__(self, contamination=0.01, n_estimators=100, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
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
        self.model.fit(norm_feats)
        return self

    def anomaly_score(self, features):
        norm_feats = (features - self.mean) / self.std
        scores = -self.model.score_samples(norm_feats)
        return scores

    def predict(self, features):
        norm_feats = (features - self.mean) / self.std
        return self.model.predict(norm_feats)

    def loss_function(self, *args, **kwargs):
        return 0.0, 0.0, 0.0

    def feature_anomaly(self, features):
        norm_feats = (features - self.mean) / self.std
        return np.abs(norm_feats)

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
