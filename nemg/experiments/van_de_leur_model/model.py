from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Softplus(nn.Module):
    def __init__(self, eps: float = 1.0e-4):
        super().__init__()
        self.eps = eps
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus(x) + self.eps


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(2)


class CausalConvolutionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        final: bool = False,
        forward: bool = True,
    ):
        super().__init__()
        Conv1d = nn.Conv1d
        padding = (kernel_size - 1) * dilation

        conv1 = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        chomp1 = Chomp1d(padding)
        relu1 = nn.LeakyReLU()

        conv2 = Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        chomp2 = Chomp1d(padding)
        relu2 = nn.LeakyReLU()

        self.causal = nn.Sequential(conv1, chomp1, relu1, conv2, chomp2, relu2)
        self.upordownsample = Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.LeakyReLU() if final else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        out = out_causal + res
        return out if self.relu is None else self.relu(out)


class CausalCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        depth: int,
        out_channels: int,
        kernel_size: int,
        forward: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dilation_size = 1 if forward else 2**depth

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers.append(
                CausalConvolutionBlock(
                    in_channels_block,
                    channels,
                    kernel_size,
                    dilation_size,
                    forward=forward,
                )
            )
            dilation_size = dilation_size * 2 if forward else dilation_size // 2

        layers.append(
            CausalConvolutionBlock(
                channels,
                out_channels,
                kernel_size,
                dilation_size,
                forward=forward,
            )
        )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CausalCNNVEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        depth: int,
        reduced_size: int,
        out_channels: int,
        kernel_size: int,
        softplus_eps: float,
        dropout: float = 0.0,
        sd_output: bool = True,
    ):
        super().__init__()
        causal_cnn = CausalCNN(in_channels, channels, depth, reduced_size, kernel_size)
        reduce_size = nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()
        self.network = nn.Sequential(causal_cnn, reduce_size, squeeze)
        self.linear_mean = nn.Linear(reduced_size, out_channels)
        self.sd_output = sd_output
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.sd_output:
            self.linear_sd = nn.Sequential(
                nn.Linear(reduced_size, out_channels),
                Softplus(softplus_eps),
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        out = self.dropout(self.network(x))
        if self.sd_output:
            return self.linear_mean(out), self.linear_sd(out)
        return self.linear_mean(out).squeeze()


class CausalCNNVDecoder(nn.Module):
    def __init__(
        self,
        k: int,
        width: int,
        in_channels: int,
        channels: int,
        depth: int,
        out_channels: int,
        kernel_size: int,
        gaussian_out: bool,
        softplus_eps: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.gaussian_out = gaussian_out

        self.linear1 = nn.Linear(k, in_channels)
        self.linear2 = nn.Linear(in_channels, in_channels * width)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.causal_cnn = CausalCNN(
            in_channels,
            channels,
            depth,
            out_channels,
            kernel_size,
            forward=False,
        )

        if self.gaussian_out:
            flat_out = out_channels * width
            self.linear_mean = nn.Linear(flat_out, flat_out)
            self.linear_sd = nn.Sequential(
                nn.Linear(flat_out, flat_out),
                Softplus(softplus_eps),
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        batch_size, _ = x.shape
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.dropout(out)
        out = out.view(batch_size, self.in_channels, self.width)
        out = self.causal_cnn(out)

        if self.gaussian_out:
            out_shape = out.shape
            out = torch.flatten(out, start_dim=1)
            mean = self.linear_mean(out).reshape(out_shape)
            sd = self.linear_sd(out).reshape(out_shape)
            return mean, sd
        return out


class Conv1DBetaVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 21,
        channels: int = 128,
        depth: int = 7,
        reduced_size: int = 64,
        decoder_in_channels: int = 64,
        kernel_size: int = 5,
        softplus_eps: float = 1.0e-4,
        dropout: float = 0.0,
        gaussian_out: bool = True,
        recon_loss_type: str | None = None,
        lambda_fdd: float = 1.0,
        lambda_cosine: float = 1.0,
        lambda_spectral: float = 1.0,
        huber_delta: float = 1.0,
        spectral_use_log_magnitude: bool = False,
        event_weight_alpha: float = 2.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.softplus_eps = softplus_eps
        self.gaussian_out = gaussian_out
        self.lambda_fdd = lambda_fdd
        self.lambda_cosine = lambda_cosine
        self.lambda_spectral = lambda_spectral
        self.huber_delta = huber_delta
        self.spectral_use_log_magnitude = spectral_use_log_magnitude
        self.event_weight_alpha = event_weight_alpha

        if recon_loss_type is None:
            self.recon_loss_type = "gaussian" if gaussian_out else "mse"
        else:
            self.recon_loss_type = recon_loss_type

        if self.recon_loss_type == "gaussian" and not gaussian_out:
            raise ValueError("recon_loss_type='gaussian' requires gaussian_out=True")

        if self.recon_loss_type in {"mse", "fdd", "huber_cosine", "mse_spectral", "weighted_huber_cosine"} and gaussian_out:
            print(
                f"Warning: gaussian_out=True but recon_loss_type='{self.recon_loss_type}'. "
                "recon_std will be produced by the decoder but ignored by the loss."
            )

        self.encoder = CausalCNNVEncoder(
            in_channels=1,
            channels=channels,
            depth=depth,
            reduced_size=reduced_size,
            out_channels=latent_dim,
            kernel_size=kernel_size,
            softplus_eps=softplus_eps,
            dropout=dropout,
            sd_output=True,
        )
        self.decoder = CausalCNNVDecoder(
            k=latent_dim,
            width=input_dim,
            in_channels=decoder_in_channels,
            channels=channels,
            depth=depth,
            out_channels=1,
            kernel_size=kernel_size,
            gaussian_out=gaussian_out,
            softplus_eps=softplus_eps,
            dropout=dropout,
        )

    def _ensure_3d_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        raise ValueError(f"Expected input of shape [B, L] or [B, C, L], got {tuple(x.shape)}")

    def _crop_or_pad(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) > self.input_dim:
            return x[..., : self.input_dim]
        if x.size(-1) < self.input_dim:
            pad = self.input_dim - x.size(-1)
            return F.pad(x, (0, pad))
        return x

    def encode_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._ensure_3d_input(x)
        mu, sd = self.encoder(x)
        logvar = torch.log(sd.pow(2) + self.softplus_eps)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, sd, logvar

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, _sd, logvar = self.encode_distribution(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        z: torch.Tensor,
        return_std: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.decoder(z)
        if self.gaussian_out:
            mean, sd = out
            mean = self._crop_or_pad(mean).squeeze(1)
            sd = self._crop_or_pad(sd).squeeze(1).clamp_min(self.softplus_eps)
            return (mean, sd) if return_std else mean
        x_hat = self._crop_or_pad(out).squeeze(1)
        return x_hat

    def forward(
        self,
        x: torch.Tensor,
        return_decoder_stats: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        mu, _sd, logvar = self.encode_distribution(x)
        z = self.reparameterize(mu, logvar)

        if self.gaussian_out:
            x_hat, recon_std = self.decode(z, return_std=True)
            if return_decoder_stats:
                return x_hat, mu, logvar, recon_std
            return x_hat, mu, logvar

        x_hat = self.decode(z)
        return x_hat, mu, logvar
