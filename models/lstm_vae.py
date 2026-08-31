import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.latent_to_h = nn.Linear(latent_dim, num_layers * hidden_dim)
        self.latent_to_c = nn.Linear(latent_dim, num_layers * hidden_dim)
        self.input_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        batch_size = z.size(0)

        h0 = F.relu(self.latent_to_h(z)).view(batch_size, self.num_layers, self.hidden_dim)
        h0 = h0.permute(1, 0, 2).contiguous()
        c0 = F.relu(self.latent_to_c(z)).view(batch_size, self.num_layers, self.hidden_dim)
        c0 = c0.permute(1, 0, 2).contiguous()

        token = self.input_token.expand(batch_size, self.seq_len, -1)
        h, _ = self.lstm(token, (h0, c0))
        x_recon = self.fc_out(h)
        return x_recon


class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len, num_layers=2):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers)
        self.seq_len = seq_len

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def loss_function(self, x, x_recon, mu, logvar, beta=1.0, free_bits=0.1):
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        kl_loss = torch.mean(kl_per_dim)

        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def anomaly_score(self, x, beta=1.0):
        with torch.no_grad():
            x_recon, mu, logvar = self.forward(x)
            recon_err = F.mse_loss(x_recon, x, reduction="none").mean(dim=2)
            kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) / self.seq_len
            combined = recon_err + beta * kl_per_sample.unsqueeze(1).expand_as(recon_err)
            return combined

    def feature_anomaly(self, x):
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
            feat_err = (x_recon - x) ** 2
            return feat_err.mean(dim=1)
