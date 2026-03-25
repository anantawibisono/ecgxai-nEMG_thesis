from __future__ import annotations

import time
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from nemg.dataset.dataset import build_dataloaders
from nemg.experiments.simple_vae.engine import train_one_epoch, validate
from nemg.experiments.simple_vae.losses import beta_for_epoch
from nemg.experiments.simple_vae.metrics import build_metrics
from nemg.experiments.simple_vae.utils import count_parameters, resolve_device, set_seed
from nemg.experiments.conv1d_beta_vae.model import Conv1DBetaVAE


@torch.no_grad()
def save_reconstruction_plot(model, loader, device, save_path: Path, n: int = 3) -> None:
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(device).float()
    x_hat, _, _ = model(x)

    x = x.cpu()
    x_hat = x_hat.cpu()

    n = min(n, x.size(0))
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), squeeze=False)

    for i in range(n):
        ax = axes[i, 0]
        ax.plot(x[i].numpy(), label="input")
        ax.plot(x_hat[i].numpy(), label="recon")
        ax.set_title(f"Sample {i}")
        ax.legend(loc="upper right")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_loss_plot(
    train_losses: list[float],
    val_losses: list[float],
    save_path: Path,
    title: str = "Training and Validation Loss",
) -> None:
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, label="train loss")
    ax.plot(epochs, val_losses, label="val loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def maybe_init_wandb(cfg: DictConfig):
    if not cfg.logger.use_wandb:
        return None
    import wandb
    return wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        mode=cfg.logger.mode,
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    device = resolve_device(cfg.trainer.device)

    train_loader, val_loader = build_dataloaders(
        fold_dir=cfg.data.fold_dir,
        windows_dir=cfg.data.windows_dir,
        use_downsampled=cfg.data.use_downsampled,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        cache=cfg.data.cache,
        drop_last_train=cfg.data.drop_last_train,
    )

    input_dim = train_loader.dataset.win_len

    model = Conv1DBetaVAE(
        input_dim=input_dim,
        latent_dim=cfg.model.latent_dim,
        channels=tuple(cfg.model.channels),
        kernel_size=cfg.model.kernel_size,
        stride=cfg.model.stride,
        hidden_dim=cfg.model.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.trainer.lr,
        weight_decay=cfg.trainer.weight_decay,
    )

    train_metrics = build_metrics().to(device)
    val_metrics = build_metrics().to(device)

    run = maybe_init_wandb(cfg)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    recon_dir = output_dir / "reconstructions"
    plot_dir = output_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_ckpt_path = ckpt_dir / "best.pt"

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    print(f"Device: {device}")
    print(f"Input dim: {input_dim}")
    print(f"Trainable params: {count_parameters(model):,}")
    print(f"Output dir: {output_dir}")

    for epoch in range(1, cfg.trainer.epochs + 1):
        epoch_start = time.time()
        beta = beta_for_epoch(epoch, cfg.model.beta, cfg.model.beta_warmup_epochs)

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            metrics=train_metrics,
            device=device,
            beta=beta,
            grad_clip_norm=cfg.trainer.grad_clip_norm,
            max_batches=cfg.trainer.max_train_batches,
        )
        val_stats = validate(
            model=model,
            loader=val_loader,
            metrics=val_metrics,
            device=device,
            beta=beta,
            max_batches=cfg.trainer.max_val_batches,
        )

        epoch_time = time.time() - epoch_start
        log_dict = {
            "epoch": epoch,
            "beta": beta,
            "epoch_time_sec": epoch_time,
            **{f"train/{k}": v for k, v in train_stats.items()},
            **{f"val/{k}": v for k, v in val_stats.items()},
        }

        train_loss_history.append(float(train_stats["loss"]))
        val_loss_history.append(float(val_stats["loss"]))

        print(
            f"Epoch {epoch:03d} | beta={beta:.4f} | "
            f"train_loss={log_dict['train/loss']:.6f} | "
            f"train_recon={log_dict['train/recon']:.6f} | "
            f"train_kl={log_dict['train/kl']:.6f} | "
            f"val_loss={log_dict['val/loss']:.6f} | "
            f"val_recon={log_dict['val/recon']:.6f} | "
            f"val_kl={log_dict['val/kl']:.6f} | "
            f"time={epoch_time:.2f}s"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "input_dim": input_dim,
            },
            ckpt_dir / "last.pt",
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "input_dim": input_dim,
                },
                best_ckpt_path,
            )

        if (
            cfg.trainer.plot_recons_every_n_epochs is not None
            and epoch % cfg.trainer.plot_recons_every_n_epochs == 0
        ):
            save_reconstruction_plot(
                model,
                val_loader,
                device,
                recon_dir / f"epoch_{epoch:03d}.png",
                n=cfg.trainer.num_plot_examples,
            )

        # update loss plot every epoch
        save_loss_plot(
            train_losses=train_loss_history,
            val_losses=val_loss_history,
            save_path=plot_dir / "train_val_loss.png",
        )

        if run is not None:
            run.log(log_dict)

    save_reconstruction_plot(
        model,
        val_loader,
        device,
        recon_dir / "final.png",
        n=cfg.trainer.num_plot_examples,
    )

    save_loss_plot(
        train_losses=train_loss_history,
        val_losses=val_loss_history,
        save_path=plot_dir / "train_val_loss_final.png",
    )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()