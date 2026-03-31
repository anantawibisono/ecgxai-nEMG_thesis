from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Causal1D(nn.Module):
    def __init__(self, padding: int):
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.padding].contiguous() if self.padding > 0 else x


class SpatialDropout1D(nn.Module):
    """Drop entire feature maps instead of individual time samples."""

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout2d = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2d(x.unsqueeze(2)).squeeze(2)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                stride=stride,
            ),
            Causal1D(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                stride=stride,
            ),
            Causal1D(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            SpatialDropout1D(dropout),
        )

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding=0, stride=stride
            )

        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.block(x)
        x_res = self.downsample(x) if self.downsample is not None else x
        return self.out_relu(x_out + x_res)


class TCNNBetaVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        seed_len: int = 32,
        kernel_size: int = 5,
        encoder_channels: tuple[int, ...] = (2, 4, 8, 16, 16, 32, 32, 64, 64),
        encoder_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256),
        encoder_dropouts: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3),
        decoder_channels: tuple[int, ...] = (64, 32, 32, 16, 16, 8, 4, 2, 2),
        decoder_dilations: tuple[int, ...] = (256, 128, 64, 32, 16, 8, 4, 2, 1),
        decoder_dropouts: tuple[float, ...] = (0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        output_channels: int = 2,
    ):
        super().__init__()

        if not (
            len(encoder_channels)
            == len(encoder_dilations)
            == len(encoder_dropouts)
            == len(decoder_channels)
            == len(decoder_dilations)
            == len(decoder_dropouts)
        ):
            raise ValueError("Encoder/decoder channel, dilation, and dropout lists must align.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seed_len = seed_len
        self.enc_out_channels = encoder_channels[-1]

        encoder_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, dilation, dropout in zip(
            encoder_channels, encoder_dilations, encoder_dropouts
        ):
            encoder_layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)

        self.enc_pool = nn.AdaptiveAvgPool1d(seed_len)
        flat_dim = self.enc_out_channels * seed_len

        self.fc_hidden = nn.Linear(flat_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.fc_decode_1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_decode_2 = nn.Linear(hidden_dim, flat_dim)

        decoder_layers: list[nn.Module] = []
        in_ch = self.enc_out_channels
        for out_ch, dilation, dropout in zip(
            decoder_channels, decoder_dilations, decoder_dropouts
        ):
            decoder_layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.decoder = nn.Sequential(*decoder_layers)

        self.output_conv = nn.Sequential(
            nn.Conv1d(in_ch, output_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(output_channels, 1, kernel_size=1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L] -> [B, 1, L]
        if x.ndim == 2:
            x = x.unsqueeze(1)

        h = self.encoder(x)
        h = self.enc_pool(h)
        h = h.flatten(start_dim=1)
        h = self.fc_hidden(h)

        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode_1(z)
        h = self.fc_decode_2(h)
        h = h.view(z.size(0), self.enc_out_channels, self.seed_len)

        h = self.decoder(h)
        h = F.interpolate(h, size=self.input_dim, mode="linear", align_corners=False)
        x_hat = self.output_conv(h).squeeze(1)
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
