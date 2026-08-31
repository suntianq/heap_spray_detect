import numpy as np
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class MLPAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, latent_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        h = hidden_dims or [128, 64]

        enc_layers = []
        prev = input_dim
        for dim in h:
            enc_layers.extend([nn.Linear(prev, dim), nn.ReLU()])
            prev = dim
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = latent_dim
        for dim in reversed(h):
            dec_layers.extend([nn.Linear(prev, dim), nn.ReLU()])
            prev = dim
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, orig_shape[-1])
        z = self.encoder(x)
        x_recon = self.decoder(z)
        if len(orig_shape) == 3:
            x_recon = x_recon.reshape(orig_shape)
        return x_recon, z

    def loss_function(self, x, x_recon):
        return nn.functional.mse_loss(x_recon, x, reduction="mean")

    def anomaly_score(self, x, beta=1.0):
        with torch.no_grad():
            x_recon, z = self.forward(x)
            per_feat = (x - x_recon) ** 2
            if x.dim() == 3:
                per_window = per_feat.mean(dim=-1)
            else:
                per_window = per_feat.mean(dim=-1, keepdim=True)
            return per_window
