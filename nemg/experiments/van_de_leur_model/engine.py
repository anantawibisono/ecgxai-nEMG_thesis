from __future__ import annotations
from typing import Any

import time

import torch
from torch.nn.utils import clip_grad_norm_

from nemg.experiments.van_de_leur_model.losses import vae_loss


def _to_float_dict(d: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        out[k] = float(v.item()) if hasattr(v, "item") else float(v)
    return out


def _forward_vae(model: torch.nn.Module, x: torch.Tensor):
    try:
        out = model(x, return_decoder_stats=True)
    except TypeError:
        out = model(x)

    if not isinstance(out, tuple):
        raise TypeError(f"Expected tuple output from VAE model, got {type(out)!r}")

    if len(out) == 3:
        x_hat, mu, logvar = out
        recon_std = None
    elif len(out) == 4:
        x_hat, mu, logvar, recon_std = out
    else:
        raise ValueError(f"Unexpected number of outputs from model: {len(out)}")

    recon_loss_type = getattr(model, "recon_loss_type", "fdd")
    loss_kwargs = {
        "lambda_fdd": getattr(model, "lambda_fdd", 1.0),
        "lambda_cosine": getattr(model, "lambda_cosine", 1.0),
        "lambda_spectral": getattr(model, "lambda_spectral", 1.0),
        "huber_delta": getattr(model, "huber_delta", 1.0),
        "spectral_use_log_magnitude": getattr(model, "spectral_use_log_magnitude", False),
    }
    return x_hat, mu, logvar, recon_std, recon_loss_type, loss_kwargs


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    metrics,
    device: torch.device,
    beta: float,
    grad_clip_norm: float | None = None,
    max_batches: int | None = None,
    print_every: int = 50,
) -> dict[str, float]:
    model.train()
    metrics.reset()
    start_time = time.time()

    for batch_idx, (x, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)

        x_hat, mu, logvar, recon_std, recon_loss_type, loss_kwargs = _forward_vae(model, x)
        loss, recon, kl = vae_loss(
            x=x,
            x_hat=x_hat,
            mu=mu,
            logvar=logvar,
            beta=beta,
            recon_std=recon_std,
            recon_loss_type=recon_loss_type,
            **loss_kwargs,
        )

        if not torch.isfinite(loss):
            raise ValueError(
                f"Non-finite loss detected: loss={loss.item()}, "
                f"recon={recon.item()}, kl={kl.item()}"
            )

        loss.backward()
        if grad_clip_norm is not None:
            clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        metrics["loss"].update(loss.detach())
        metrics["recon"].update(recon.detach())
        metrics["kl"].update(kl.detach())

    elapsed = time.time() - start_time
    stats = _to_float_dict(metrics.compute())
    stats["time_sec"] = elapsed
    return stats


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    metrics,
    device: torch.device,
    beta: float,
    max_batches: int | None = None,
    print_every: int = 50,
) -> dict[str, float]:
    model.eval()
    metrics.reset()
    start_time = time.time()

    for batch_idx, (x, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device, non_blocking=True).float()

        x_hat, mu, logvar, recon_std, recon_loss_type, loss_kwargs = _forward_vae(model, x)

        if recon_std is not None and batch_idx == 0:
            print(
                f"recon_std mean={recon_std.mean().item():.6f} "
                f"min={recon_std.min().item():.6f} "
                f"max={recon_std.max().item():.6f}"
            )

        loss, recon, kl = vae_loss(
            x=x,
            x_hat=x_hat,
            mu=mu,
            logvar=logvar,
            beta=beta,
            recon_std=recon_std,
            recon_loss_type=recon_loss_type,
            **loss_kwargs,
        )

        if not torch.isfinite(loss):
            raise ValueError(
                f"Non-finite loss detected: loss={loss.item()}, "
                f"recon={recon.item()}, kl={kl.item()}"
            )

        metrics["loss"].update(loss.detach())
        metrics["recon"].update(recon.detach())
        metrics["kl"].update(kl.detach())

    elapsed = time.time() - start_time
    stats = _to_float_dict(metrics.compute())
    stats["time_sec"] = elapsed
    return stats
