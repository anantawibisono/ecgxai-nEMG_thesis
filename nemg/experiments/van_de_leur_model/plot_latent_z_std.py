from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from nemg.dataset.dataset import build_dataloaders, build_test_loader
from nemg.experiments.van_de_leur_model.model import Conv1DBetaVAE


SPLIT_CHOICES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-dimension expected posterior std E_x[sigma_j] for the "
            "van_de_leur_model VAE, with optional per-dimension KL as a second "
            "diagnostic."
        )
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt or last.pt")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=SPLIT_CHOICES,
        help="One or more splits to compare (default: train val)",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="Required only if 'test' is included in --splits",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from checkpoint config",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override num_workers from checkpoint config",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optionally cap the number of batches per split for faster debugging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="latent_std",
        help="Directory to save plots and CSVs",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI",
    )
    return parser.parse_args()



def should_use_pin_memory(device: torch.device, cfg_pin_memory: bool) -> bool:
    return bool(cfg_pin_memory) and device.type == "cuda" and torch.cuda.is_available()



def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> Conv1DBetaVAE:
    cfg = ckpt["cfg"]
    model_cfg = cfg["model"]
    input_dim = ckpt["input_dim"]

    model = Conv1DBetaVAE(
        input_dim=input_dim,
        latent_dim=model_cfg["latent_dim"],
        channels=model_cfg["channels"],
        depth=model_cfg["depth"],
        reduced_size=model_cfg["reduced_size"],
        decoder_in_channels=model_cfg["decoder_in_channels"],
        kernel_size=model_cfg["kernel_size"],
        softplus_eps=model_cfg["softplus_eps"],
        dropout=model_cfg["dropout"],
        gaussian_out=model_cfg.get("gaussian_out", True),
        recon_loss_type=model_cfg.get("recon_loss_type", None),
        lambda_fdd=model_cfg.get("lambda_fdd", 1.0),
        lambda_cosine=model_cfg.get("lambda_cosine", 1.0),
        lambda_spectral=model_cfg.get("lambda_spectral", 1.0),
        huber_delta=model_cfg.get("huber_delta", 1.0),
        spectral_use_log_magnitude=model_cfg.get("spectral_use_log_magnitude", False),
        event_weight_alpha=model_cfg.get("event_weight_alpha", 2.0),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model



def build_loader_from_checkpoint_cfg(
    ckpt_cfg: dict,
    split: str,
    batch_size_override: int | None,
    num_workers_override: int | None,
    test_csv: str | None,
    device: torch.device,
):
    data_cfg = ckpt_cfg["data"]

    batch_size = batch_size_override or data_cfg["batch_size"]
    num_workers = num_workers_override if num_workers_override is not None else data_cfg["num_workers"]
    pin_memory = should_use_pin_memory(device, data_cfg.get("pin_memory", True))

    if split in {"train", "val"}:
        train_loader, val_loader = build_dataloaders(
            fold_dir=data_cfg["fold_dir"],
            windows_dir=data_cfg["windows_dir"],
            use_downsampled=data_cfg["use_downsampled"],
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            cache=data_cfg.get("cache", True),
            drop_last_train=data_cfg.get("drop_last_train", True),
        )
        return train_loader if split == "train" else val_loader

    if split == "test":
        if test_csv is None:
            raise ValueError("--test-csv is required when 'test' is included in --splits")
        return build_test_loader(
            test_csv=test_csv,
            windows_dir=data_cfg["windows_dir"],
            use_downsampled=data_cfg["use_downsampled"],
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            cache=data_cfg.get("cache", True),
        )

    raise ValueError(f"Unsupported split: {split}")


@torch.no_grad()
def collect_latent_stats(
    model: Conv1DBetaVAE,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, np.ndarray | int]:
    sum_std = None
    sum_kl = None
    sum_abs_mu = None
    mu_batches: list[torch.Tensor] = []
    n_samples = 0

    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    for batch_idx, (x, _y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if batch_idx == 0 or (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(f"Processing batch {batch_idx + 1}/{total_batches}")

        x = x.to(device, non_blocking=True).float()
        mu, sd, logvar = model.encode_distribution(x)

        # Per-dimension KL for a diagonal Gaussian posterior q(z|x)=N(mu, sigma^2)
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)

        batch_n = x.size(0)
        if sum_std is None:
            latent_dim = mu.size(1)
            sum_std = torch.zeros(latent_dim, dtype=torch.float64)
            sum_kl = torch.zeros(latent_dim, dtype=torch.float64)
            sum_abs_mu = torch.zeros(latent_dim, dtype=torch.float64)

        sum_std += sd.sum(dim=0).detach().cpu().double()
        sum_kl += kl_per_dim.sum(dim=0).detach().cpu().double()
        sum_abs_mu += mu.abs().sum(dim=0).detach().cpu().double()
        mu_batches.append(mu.detach().cpu())
        n_samples += batch_n

    if n_samples == 0:
        raise RuntimeError("No samples were processed. Check your loader or --max-batches.")

    mu_all = torch.cat(mu_batches, dim=0)
    expected_std = (sum_std / n_samples).numpy()
    mean_kl = (sum_kl / n_samples).numpy()
    mean_abs_mu = (sum_abs_mu / n_samples).numpy()
    mu_dataset_std = mu_all.std(dim=0, unbiased=False).numpy()

    return {
        "expected_std": expected_std,
        "mean_kl": mean_kl,
        "mean_abs_mu": mean_abs_mu,
        "mu_dataset_std": mu_dataset_std,
        "n_samples": n_samples,
    }



def make_long_dataframe(stats_by_split: dict[str, dict[str, np.ndarray | int]]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for split, stats in stats_by_split.items():
        latent_dim = len(stats["expected_std"])
        for dim_idx in range(latent_dim):
            rows.append(
                {
                    "split": split,
                    "dimension": dim_idx + 1,
                    "expected_std": float(stats["expected_std"][dim_idx]),
                    "mean_kl": float(stats["mean_kl"][dim_idx]),
                    "mean_abs_mu": float(stats["mean_abs_mu"][dim_idx]),
                    "mu_dataset_std": float(stats["mu_dataset_std"][dim_idx]),
                    "n_samples": int(stats["n_samples"]),
                }
            )
    return pd.DataFrame(rows)



def _plot_grouped_bars(
    stats_by_split: dict[str, dict[str, np.ndarray | int]],
    key: str,
    ylabel: str,
    title: str,
    save_path: Path,
    dpi: int,
    ref_line: float | None = None,
) -> None:
    split_names = list(stats_by_split.keys())
    latent_dim = len(stats_by_split[split_names[0]][key])

    x = np.arange(latent_dim)
    width = 0.8 / max(len(split_names), 1)
    fig_width = max(10, 0.55 * latent_dim + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for idx, split in enumerate(split_names):
        offset = (idx - (len(split_names) - 1) / 2.0) * width
        values = np.asarray(stats_by_split[split][key])
        ax.bar(x + offset, values, width=width, alpha=0.85, label=split)

    if ref_line is not None:
        ax.axhline(ref_line, linestyle="--", linewidth=1.2, color="black", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(1, latent_dim + 1)])
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def plot_expected_std(
    stats_by_split: dict[str, dict[str, np.ndarray | int]],
    save_path: Path,
    dpi: int,
) -> None:
    _plot_grouped_bars(
        stats_by_split=stats_by_split,
        key="expected_std",
        ylabel=r"Expected posterior std $E_x[\sigma_j]$",
        title="Per-dimension expected posterior std of the latent distribution",
        save_path=save_path,
        dpi=dpi,
        ref_line=1.0,
    )



def plot_mean_kl(
    stats_by_split: dict[str, dict[str, np.ndarray | int]],
    save_path: Path,
    dpi: int,
) -> None:
    _plot_grouped_bars(
        stats_by_split=stats_by_split,
        key="mean_kl",
        ylabel=r"Mean KL per dimension $E_x[KL_j]$",
        title="Per-dimension KL divergence to the N(0,1) prior",
        save_path=save_path,
        dpi=dpi,
        ref_line=0.0,
    )



def plot_mu_dataset_std(
    stats_by_split: dict[str, dict[str, np.ndarray | int]],
    save_path: Path,
    dpi: int,
) -> None:
    _plot_grouped_bars(
        stats_by_split=stats_by_split,
        key="mu_dataset_std",
        ylabel=r"Dataset std of latent mean $std_x[\mu_j(x)]$",
        title="Per-dimension spread of encoder means across the dataset",
        save_path=save_path,
        dpi=dpi,
        ref_line=None,
    )



def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    model = build_model_from_checkpoint(ckpt, device=device)

    stats_by_split: dict[str, dict[str, np.ndarray | int]] = {}

    for split in args.splits:
        print(f"\n=== Collecting latent stats for split='{split}' ===")
        loader = build_loader_from_checkpoint_cfg(
            ckpt_cfg=cfg,
            split=split,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
            test_csv=args.test_csv,
            device=device,
        )
        stats = collect_latent_stats(
            model=model,
            loader=loader,
            device=device,
            max_batches=args.max_batches,
        )
        stats_by_split[split] = stats
        print(f"Processed {stats['n_samples']} samples for split='{split}'")
        print("Expected posterior std:", np.round(stats["expected_std"], 4))
        print("Mean per-dim KL:     ", np.round(stats["mean_kl"], 4))

    df = make_long_dataframe(stats_by_split)
    csv_path = outdir / "latent_stats_per_dim.csv"
    df.to_csv(csv_path, index=False)

    std_plot_path = outdir / "latent_expected_std.png"
    kl_plot_path = outdir / "latent_per_dim_kl.png"
    mu_std_plot_path = outdir / "latent_mu_dataset_std.png"

    plot_expected_std(stats_by_split, std_plot_path, dpi=args.dpi)
    plot_mean_kl(stats_by_split, kl_plot_path, dpi=args.dpi)
    plot_mu_dataset_std(stats_by_split, mu_std_plot_path, dpi=args.dpi)

    print(f"\nSaved per-dim CSV to:     {csv_path}")
    print(f"Saved std plot to:        {std_plot_path}")
    print(f"Saved KL plot to:         {kl_plot_path}")
    print(f"Saved mu-std plot to:     {mu_std_plot_path}")


if __name__ == "__main__":
    main()

# Example:
# python -m nemg.experiments.van_de_leur_model.run_latent_std \
#   --ckpt multirun/.../checkpoints/best.pt \
#   --splits train val \
#   --outdir outputs/van_de_leur_model/run_01/latent_std
