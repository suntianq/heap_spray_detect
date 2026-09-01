import os
import numpy as np
import pickle
import logging

log = logging.getLogger("ocsvm")

# Batch size for chunked scoring (avoids materializing a huge float64 output
# array for hundreds of thousands of sequences at once).
_SCORE_BATCH = 65536


def _probe_thundersvm():
    """Import probe for ThunderSVM.

    Any failure means unavailable: ImportError (not installed) or OSError
    (installed but broken, e.g. wheel built against an older CUDA than the
    host's, missing libcusparse/libcublas .so).
    """
    import thundersvm  # noqa: F401


def _resolve_backend():
    """Resolve SVM backend: ThunderSVM (GPU) when available, else sklearn.

    ThunderSVM provides a sklearn-compatible OneClassSVM that runs on CUDA.
    Set env HEAPSPRAY_SVM_BACKEND=sklearn to force the CPU backend even when
    ThunderSVM is installed. A broken ThunderSVM install (CUDA lib mismatch)
    also falls back to sklearn instead of crashing the run.
    """
    forced = os.environ.get("HEAPSPRAY_SVM_BACKEND", "").strip().lower()
    if forced in ("sklearn", "cpu"):
        return "sklearn"
    try:
        _probe_thundersvm()
        return "thundersvm"
    except Exception as error:
        log.warning("thundersvm unavailable (%s: %s); using sklearn CPU backend",
                    type(error).__name__, error)
        return "sklearn"


def _make_model(kernel, nu, gamma, backend):
    if backend == "thundersvm":
        from thundersvm import OneClassSVM
        return OneClassSVM(kernel=kernel, nu=nu, gamma=gamma, verbose=False)
    from sklearn.svm import OneClassSVM
    return OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)


class OCSVMDetector:
    """One-Class SVM detector with automatic GPU backend.

    Backend selection at fit time: thundersvm (CUDA) if installed, else
    sklearn (CPU). gamma='scale' is resolved to its numeric value so both
    backends compute identical kernels.
    """

    def __init__(self, kernel="rbf", nu=0.05, gamma="scale"):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.backend = _resolve_backend()
        self.model = None
        self.mean = None
        self.std = None

    def _effective_gamma(self, X):
        """Resolve string gamma to numeric (ThunderSVM requires a float).

        'scale' = 1 / (n_features * X.var()), matching sklearn semantics.
        """
        g = self.gamma
        if isinstance(g, str):
            if g == "scale":
                var = float(X.var())
                return 1.0 / (X.shape[1] * var) if var > 0 else 1.0
            if g == "auto":
                return 1.0 / X.shape[1]
            raise ValueError(f"unsupported gamma: {g!r}")
        return float(g)

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
        gamma = self._effective_gamma(norm_feats)
        self.model = _make_model(self.kernel, self.nu, gamma, self.backend)
        log.info("ocsvm fit: backend=%s %d samples, %d features",
                 self.backend, norm_feats.shape[0], norm_feats.shape[1])
        self.model.fit(norm_feats)
        log.info("ocsvm fit done")
        return self

    def _score_samples(self, X):
        """score_samples with backend fallback.

        ThunderSVM may not implement score_samples; decision_function differs
        only by a constant offset (offset_), which is monotonic-equivalent:
        AUC ranking is unchanged and the p99 threshold is calibrated on the
        same backend's scores, so the constant shift is absorbed.
        """
        if hasattr(self.model, "score_samples"):
            return self.model.score_samples(X)
        return self.model.decision_function(X)

    def anomaly_score(self, features):
        norm_feats = (features - self.mean) / self.std
        n = len(norm_feats)
        if n <= _SCORE_BATCH:
            return -self._score_samples(norm_feats)
        # Chunk to show progress on large eval sets
        from tqdm import tqdm
        chunks = range(0, n, _SCORE_BATCH)
        parts = []
        for i in tqdm(chunks, total=(n + _SCORE_BATCH - 1) // _SCORE_BATCH,
                      desc=f"ocsvm scoring ({self.backend})", unit="chunk", leave=True):
            parts.append(-self._score_samples(norm_feats[i:i + _SCORE_BATCH]))
        return np.concatenate(parts)

    def loss_function(self, *args, **kwargs):
        return 0.0, 0.0, 0.0

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "mean": self.mean, "std": self.std,
                         "kernel": self.kernel, "nu": self.nu, "gamma": self.gamma,
                         "backend": self.backend}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(kernel=data.get("kernel", "rbf"), nu=data.get("nu", 0.05),
                  gamma=data.get("gamma", "scale"))
        obj.backend = data.get("backend", obj.backend)
        obj.model = data["model"]
        obj.mean = data["mean"]
        obj.std = data["std"]
        return obj
