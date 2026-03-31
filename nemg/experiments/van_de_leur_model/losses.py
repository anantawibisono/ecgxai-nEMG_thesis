from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def beta_for_epoch(epoch: int, beta: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return beta
    scale = min(epoch / warmup_epochs, 1.0)
    return beta * scale


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def _flatten_signal(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(start_dim=1)


def _cosine_term(x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    x = _flatten_signal(x)
    x_hat = _flatten_signal(x_hat)
    cos_sim = F.cosine_similarity(x_hat, x, dim=1, eps=eps)
    return ((1.0 - cos_sim) / 2.0).mean()


def gaussian_recon_loss(
    x: torch.Tensor,
    recon_mean: torch.Tensor,
    recon_std: torch.Tensor,
    min_std: float = 5.0e-2,
    max_std: float = 1.0,
) -> torch.Tensor:
    x = _flatten_signal(x)
    recon_mean = _flatten_signal(recon_mean)
    recon_std = _flatten_signal(recon_std).clamp(min=min_std, max=max_std)

    nll = (
        0.5 * math.log(2.0 * math.pi)
        + torch.log(recon_std)
        + 0.5 * ((x - recon_mean) / recon_std).pow(2)
    )
    return nll.mean(dim=1).mean()


def mse_recon_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(_flatten_signal(x_hat), _flatten_signal(x), reduction="mean")


def huber_recon_loss(x: torch.Tensor, x_hat: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(_flatten_signal(x_hat), _flatten_signal(x), reduction="mean", delta=delta)


def spectral_mse_loss(x: torch.Tensor, x_hat: torch.Tensor, use_log_magnitude: bool = False) -> torch.Tensor:
    x = _flatten_signal(x)
    x_hat = _flatten_signal(x_hat)

    x_fft = torch.fft.rfft(x, dim=1)
    x_hat_fft = torch.fft.rfft(x_hat, dim=1)

    x_mag = torch.abs(x_fft)
    x_hat_mag = torch.abs(x_hat_fft)

    if use_log_magnitude:
        x_mag = torch.log1p(x_mag)
        x_hat_mag = torch.log1p(x_hat_mag)

    return F.mse_loss(x_hat_mag, x_mag, reduction="mean")


def fdd_recon_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    lambda_fdd: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    mse = mse_recon_loss(x, x_hat)
    cos_term = _cosine_term(x, x_hat, eps=eps)
    return mse + lambda_fdd * cos_term


def huber_cosine_recon_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    lambda_cosine: float = 1.0,
    huber_delta: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    huber = huber_recon_loss(x, x_hat, delta=huber_delta)
    cos_term = _cosine_term(x, x_hat, eps=eps)
    return huber + lambda_cosine * cos_term


def mse_spectral_recon_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    lambda_spectral: float = 1.0,
    spectral_use_log_magnitude: bool = False,
) -> torch.Tensor:
    mse = mse_recon_loss(x, x_hat)
    spec = spectral_mse_loss(x, x_hat, use_log_magnitude=spectral_use_log_magnitude)
    return mse + lambda_spectral * spec


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    recon_std: torch.Tensor | None = None,
    recon_loss_type: str = "mse",
    lambda_fdd: float = 1.0,
    lambda_cosine: float = 1.0,
    lambda_spectral: float = 1.0,
    huber_delta: float = 1.0,
    spectral_use_log_magnitude: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if recon_loss_type == "gaussian":
        if recon_std is None:
            raise ValueError("recon_std must be provided when recon_loss_type='gaussian'")
        recon = gaussian_recon_loss(x, x_hat, recon_std)
    elif recon_loss_type == "mse":
        recon = mse_recon_loss(x, x_hat)
    elif recon_loss_type == "fdd":
        recon = fdd_recon_loss(x, x_hat, lambda_fdd=lambda_fdd)
    elif recon_loss_type == "huber_cosine":
        recon = huber_cosine_recon_loss(
            x,
            x_hat,
            lambda_cosine=lambda_cosine,
            huber_delta=huber_delta,
        )
    elif recon_loss_type == "mse_spectral":
        recon = mse_spectral_recon_loss(
            x,
            x_hat,
            lambda_spectral=lambda_spectral,
            spectral_use_log_magnitude=spectral_use_log_magnitude,
        )
    else:
        raise ValueError(f"Unsupported recon_loss_type: {recon_loss_type}")

    kl = kl_divergence(mu, logvar).mean()
    total = recon + beta * kl
    return total, recon, kl
