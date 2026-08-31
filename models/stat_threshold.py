import numpy as np


class StatisticalThresholdDetector:
    def __init__(self, n_sigma=3.0):
        self.n_sigma = n_sigma
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
        return self

    def anomaly_score(self, features):
        norm_feats = (features - self.mean) / self.std
        max_deviations = np.max(np.abs(norm_feats), axis=1)
        return max_deviations

    def predict(self, features, threshold=None):
        if threshold is None:
            threshold = self.n_sigma
        scores = self.anomaly_score(features)
        return np.where(scores > threshold, -1, 1)

    def loss_function(self, *args, **kwargs):
        return 0.0, 0.0, 0.0

    def feature_anomaly(self, features):
        return np.abs((features - self.mean) / self.std)
