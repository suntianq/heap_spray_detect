from .lstm_vae import LSTMVAE
from .lstm_ae import LSTMAE
from .mlp_ae import MLPAE
from .torch_ae import TorchAEWrapper
from .isolation_forest import IsolationForestDetector
from .stat_threshold import StatisticalThresholdDetector
from .ocsvm import OCSVMDetector
from .random_forest import RandomForestDetector
from .lof_detector import LOFDetector
from .pca_detector import PCADetector

MODEL_REGISTRY = {
    "lstm_vae": LSTMVAE,
    "lstm_ae": LSTMAE,
    "mlp_ae": MLPAE,
    "torch_ae": TorchAEWrapper,
    "isolation_forest": IsolationForestDetector,
    "stat_threshold": StatisticalThresholdDetector,
    "ocsvm": OCSVMDetector,
    "random_forest": RandomForestDetector,
    "lof": LOFDetector,
    "pca": PCADetector,
}
