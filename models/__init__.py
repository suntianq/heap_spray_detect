from .lstm_vae import LSTMVAE
from .lstm_ae import LSTMAE
from .torch_ae import TorchAEWrapper
from .ocsvm import OCSVMDetector

MODEL_REGISTRY = {
    "lstm_vae": LSTMVAE,
    "lstm_ae": LSTMAE,
    "torch_ae": TorchAEWrapper,
    "ocsvm": OCSVMDetector,
}
