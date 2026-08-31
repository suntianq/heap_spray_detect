import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                                    batch_first=True, bidirectional=True)
        self.fc_enc = nn.Linear(hidden_dim * 2, latent_dim)

        self.latent_to_h = nn.Linear(latent_dim, num_layers * hidden_dim)
        self.latent_to_c = nn.Linear(latent_dim, num_layers * hidden_dim)
        self.input_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        # h[:, -1, :] does not contain the complete backward sequence state.
        # Concatenate the final forward/backward states from the last layer.
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        z = self.fc_enc(h_last)
        return z

    def decode(self, z):
        batch_size = z.size(0)

        h0 = F.relu(self.latent_to_h(z)).view(batch_size, self.num_layers, self.hidden_dim)
        h0 = h0.permute(1, 0, 2).contiguous()
        c0 = F.relu(self.latent_to_c(z)).view(batch_size, self.num_layers, self.hidden_dim)
        c0 = c0.permute(1, 0, 2).contiguous()

        token = self.input_token.expand(batch_size, self.seq_len, -1)
        h, _ = self.decoder_lstm(token, (h0, c0))
        return self.fc_out(h)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def loss_function(self, x, x_recon, *args, **kwargs):
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        return recon_loss

    def anomaly_score(self, x, **kwargs):
        with torch.no_grad():
            x_recon, _ = self.forward(x)
            recon_err = F.mse_loss(x_recon, x, reduction="none")
            return recon_err.mean(dim=2)

    def feature_anomaly(self, x):
        with torch.no_grad():
            x_recon, _ = self.forward(x)
            return (x_recon - x) ** 2
