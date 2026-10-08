import math
from typing import Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: list = [256, 128, 64]
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])

            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(
            hidden_dims[-1],
            latent_dim
        )

        self.fc_log_var = nn.Linear(
            hidden_dims[-1],
            latent_dim
        )

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])

            prev_dim = hidden_dim

        decoder_layers.extend([
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()
        ])

        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        h = self.encoder(x)

        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)

        z = self.reparameterize(mu, log_var)

        reconstruction = self.decoder(z)

        return reconstruction, mu, log_var

    def generate_synthetic(
        self,
        num_samples: int,
        device: str = "cuda"
    ) -> torch.Tensor:

        z = torch.randn(
            num_samples,
            self.latent_dim
        ).to(device)

        return self.decoder(z)


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        self.pos_encoder = PositionalEncoding(
            d_model,
            dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers
        )

        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_projection(x)

        x = x * math.sqrt(self.d_model)

        x = self.pos_encoder(x)

        output = self.transformer_encoder(x)

        return output[:, -1, :]


class SpatialCNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        num_classes: int = 2
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv1d(
                input_channels,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Dropout(0.2),

            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Dropout(0.2),

            nn.Conv1d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.AdaptiveMaxPool1d(1)
        )

        self.flat_size = 256

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                self.flat_size,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # (batch, sequence, features)
        # ->
        # (batch, features, sequence)

        x = x.permute(0, 2, 1)

        features = self.conv_layers(x)

        return self.classifier(features)


class HybridAnomalyDetector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 10,
        num_classes: int = 2,
        latent_dim: int = 64
    ):
        super().__init__()

        self.input_dim = input_dim
        self.sequence_length = sequence_length

        # VAE
        self.vae = VariationalAutoencoder(
            input_dim,
            latent_dim
        )

        # Transformer
        self.temporal = TemporalTransformer(
            input_dim
        )

        # CNN
        self.spatial = SpatialCNN(
            input_dim,
            sequence_length,
            num_classes
        )

        # Fusion
        fusion_input_dim = (
            latent_dim + 128 + num_classes
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                fusion_input_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                num_classes
            )
        )

        # Anomaly scorer
        self.anomaly_scorer = nn.Sequential(

            nn.Linear(
                latent_dim + 128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        batch_size, seq_len, features = x.shape

        # Last timestep for VAE
        x_flat = x[:, -1, :]

        reconstruction, mu, log_var = self.vae(
            x_flat
        )

        z = self.vae.reparameterize(
            mu,
            log_var
        )

        # Transformer
        temp_features = self.temporal(x)

        # CNN
        cnn_logits = self.spatial(x)

        # Fusion
        combined = torch.cat(
            [
                z,
                temp_features,
                cnn_logits
            ],
            dim=1
        )

        classification = self.fusion(
            combined
        )

        # Anomaly score
        anomaly_features = torch.cat(
            [
                z,
                temp_features
            ],
            dim=1
        )

        anomaly_score = self.anomaly_scorer(
            anomaly_features
        )

        return {
            "classification": classification,
            "anomaly_score": anomaly_score,
            "reconstruction": reconstruction,
            "mu": mu,
            "log_var": log_var,
            "vae_latent": z,
            "temporal_features": temp_features,
            "cnn_logits": cnn_logits
        }