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


def gaussian_recon_loss(
    x: torch.Tensor,
    recon_mean: torch.Tensor,
    recon_std: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    x = x.flatten(start_dim=1)
    recon_mean = recon_mean.flatten(start_dim=1)
    recon_std = recon_std.flatten(start_dim=1).clamp_min(eps)

    nll = (
        0.5 * math.log(2.0 * math.pi)
        + torch.log(recon_std)
        + 0.5 * ((x - recon_mean) / recon_std).pow(2)
    )
    return nll.sum(dim=1).mean()


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    recon_std: torch.Tensor | None = None,
    recon_loss_type: str = "mse",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if recon_loss_type == "gaussian":
        if recon_std is None:
            raise ValueError("recon_std must be provided when recon_loss_type='gaussian'")
        recon = gaussian_recon_loss(x, x_hat, recon_std)
    elif recon_loss_type == "mse":
        recon = F.mse_loss(x_hat, x, reduction="mean")
    else:
        raise ValueError(f"Unsupported recon_loss_type: {recon_loss_type}")

    kl = kl_divergence(mu, logvar).mean()
    total = recon + beta * kl
    return total, recon, kl
