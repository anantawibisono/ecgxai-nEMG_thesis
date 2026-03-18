from __future__ import annotations

from typing import Any
import time

import torch
from torch.nn.utils import clip_grad_norm_

from nemg.experiments.simple_vae.losses import vae_loss

def _to_float_dict(d: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            out[k] = float(v.item())
        else:
            out[k] = float(v)
    return out


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

        # if batch_idx % print_every == 0:
        #     print(f"[train] batch {batch_idx}/{len(loader)}")

        x = x.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        x_hat, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta=beta)

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

        # if batch_idx % print_every == 0:
        #     print(f"[val] batch {batch_idx}/{len(loader)}")

        x = x.to(device, non_blocking=True).float()

        x_hat, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta=beta)

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