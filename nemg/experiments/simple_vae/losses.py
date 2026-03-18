from __future__ import annotations

import torch
import torch.nn.functional as F


def beta_for_epoch(epoch: int, beta: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return beta
    scale = min(epoch / warmup_epochs, 1.0)
    return beta * scale


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # per-sample KL, shape: (batch,)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = kl_divergence(mu, logvar).mean()
    total = recon + beta * kl
    return total, recon, kl