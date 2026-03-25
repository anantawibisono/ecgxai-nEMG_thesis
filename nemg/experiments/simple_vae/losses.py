from __future__ import annotations

from typing import Literal

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


def cosine_similarity_term(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Mean angular-difference term used in FDD:
        (1 - cosine_similarity) / 2

    Returns a scalar averaged over the batch.
    """
    x_flat = x.reshape(x.size(0), -1)
    x_hat_flat = x_hat.reshape(x_hat.size(0), -1)

    cs = F.cosine_similarity(x_hat_flat, x_flat, dim=1, eps=eps)  # shape: (batch,)
    return ((1.0 - cs) / 2.0).mean()


def fdd_reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    lambda_fdd: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Feature-Direction Difference (FDD):
        FDD = MSE + lambda * ((1 - CS) / 2)

    Returns:
        fdd: total FDD reconstruction loss
        mse: MSE component
        cs_term: ((1 - CS) / 2) component
    """
    mse = F.mse_loss(x_hat, x, reduction="mean")
    cs_term = cosine_similarity_term(x, x_hat, eps=eps)
    fdd = mse + lambda_fdd * cs_term
    return fdd, mse, cs_term


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    recon_loss: Literal["mse", "fdd"] = "fdd",
    lambda_fdd: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss with selectable reconstruction term.

    Returns:
        total: reconstruction + beta * KL
        recon: selected reconstruction loss (MSE or FDD)
        kl: KL divergence
    """
    if recon_loss == "mse":
        recon = F.mse_loss(x_hat, x, reduction="mean")
    elif recon_loss == "fdd":
        recon, _, _ = fdd_reconstruction_loss(
            x=x,
            x_hat=x_hat,
            lambda_fdd=lambda_fdd,
            eps=eps,
        )
    else:
        raise ValueError(f"Unsupported recon_loss: {recon_loss}")

    kl = kl_divergence(mu, logvar).mean()
    total = recon + beta * kl
    return total, recon, kl