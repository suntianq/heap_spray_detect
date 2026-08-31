import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle


class RandomForestDetector:
    def __init__(self, n_estimators=200, max_depth=15, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, class_weight="balanced",
        )
        self.mean = None
        self.std = None

    def fit(self, features, labels):
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        norm_feats = (features - self.mean) / self.std
        self.model.fit(norm_feats, labels)
        return self

    def anomaly_score(self, features):
        norm_feats = (features - self.mean) / self.std
        return self.model.predict_proba(norm_feats)[:, 1]

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
