from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 2):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock1D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
        stride: int = 2,
        output_padding: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.ConvTranspose1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv1DBetaVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        channels: tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 7,
        stride: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        enc_layers = []
        in_ch = 1
        current_len = input_dim

        for out_ch in channels:
            enc_layers.append(ConvBlock1D(in_ch, out_ch, kernel_size=kernel_size, stride=stride))
            in_ch = out_ch
            current_len = self._conv_out_len(current_len, kernel_size, stride, kernel_size // 2)

        self.encoder = nn.Sequential(*enc_layers)
        self.enc_out_channels = channels[-1]
        self.enc_out_len = current_len
        flat_dim = self.enc_out_channels * self.enc_out_len

        self.fc_hidden = nn.Linear(flat_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc_decode_1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_decode_2 = nn.Linear(hidden_dim, flat_dim)

        dec_layers = []
        rev_channels = list(channels[::-1])

        for i in range(len(rev_channels) - 1):
            dec_layers.append(
                DeconvBlock1D(
                    rev_channels[i],
                    rev_channels[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    output_padding=1,
                )
            )

        self.decoder_blocks = nn.Sequential(*dec_layers)
        self.final_layer = nn.ConvTranspose1d(
            rev_channels[-1],
            1,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            output_padding=1,
        )

    @staticmethod
    def _conv_out_len(length: int, kernel_size: int, stride: int, padding: int) -> int:
        return (length + 2 * padding - kernel_size) // stride + 1

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L] -> [B, 1, L]
        x = x.unsqueeze(1)
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        h = self.fc_hidden(h)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode_1(z)
        h = self.fc_decode_2(h)
        h = h.view(z.size(0), self.enc_out_channels, self.enc_out_len)
        h = self.decoder_blocks(h)
        x_hat = self.final_layer(h)

        # x_hat: [B, 1, L'] -> crop/pad to original length, then [B, L]
        x_hat = x_hat.squeeze(1)

        if x_hat.size(1) > self.input_dim:
            x_hat = x_hat[:, : self.input_dim]
        elif x_hat.size(1) < self.input_dim:
            pad = self.input_dim - x_hat.size(1)
            x_hat = torch.nn.functional.pad(x_hat, (0, pad))

        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar