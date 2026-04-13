from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap.umap_ as umap
except ImportError:
    umap = None

from nemg.dataset.dataset import build_dataloaders, build_test_loader
from nemg.experiments.van_de_leur_model.model import Conv1DBetaVAE


LABEL_NAMES = {
    0: "ALS",
    1: "Normal",
    2: "Myopathy",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PCA (2D), t-SNE (2D), and UMAP (3D) of VAE latent means for van_de_leur_model."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt or last.pt")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split to visualize",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="Required only when --split test. Path to test.csv",
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="latent_projections",
        help="Directory to save plots and CSVs",
    )

    # New options
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Use raw latent means (mu) instead of StandardScaler-normalized latents.",
    )
    parser.add_argument(
        "--save-both",
        action="store_true",
        help="Save both standardized and raw projections side by side.",
    )

    # UMAP options
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist",
    )
    parser.add_argument(
        "--umap-metric",
        type=str,
        default="euclidean",
        help="UMAP metric",
    )
    parser.add_argument(
        "--umap-random-state",
        type=int,
        default=42,
        help="UMAP random_state",
    )

    # t-SNE options
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity",
    )
    parser.add_argument(
        "--tsne-learning-rate",
        type=float,
        default=200.0,
        help="t-SNE learning rate",
    )
    parser.add_argument(
        "--tsne-n-iter",
        type=int,
        default=1000,
        help="t-SNE number of iterations",
    )
    parser.add_argument(
        "--tsne-random-state",
        type=int,
        default=42,
        help="t-SNE random_state",
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
    ckpt_cfg,
    split: str,
    batch_size_override: int | None,
    num_workers_override: int | None,
    test_csv: str | None,
    device: torch.device,
):
    data_cfg = ckpt_cfg["data"]
    batch_size = batch_size_override or data_cfg["batch_size"]
    num_workers = (
        num_workers_override if num_workers_override is not None
        else data_cfg["num_workers"]
    )
    pin_memory = should_use_pin_memory(device, data_cfg["pin_memory"])

    if split in {"train", "val"}:
        train_loader, val_loader = build_dataloaders(
            fold_dir=data_cfg["fold_dir"],
            windows_dir=data_cfg["windows_dir"],
            use_downsampled=data_cfg["use_downsampled"],
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            cache=data_cfg["cache"],
            drop_last_train=data_cfg["drop_last_train"],
        )
        return train_loader if split == "train" else val_loader

    if split == "test":
        if test_csv is None:
            raise ValueError("--test-csv is required when --split test")

        return build_test_loader(
            test_csv=test_csv,
            windows_dir=data_cfg["windows_dir"],
            use_downsampled=data_cfg["use_downsampled"],
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            cache=data_cfg["cache"],
        )

    raise ValueError(f"Unsupported split: {split}")


@torch.no_grad()
def extract_mu_and_labels(model: Conv1DBetaVAE, loader, device: torch.device):
    all_mu = []
    all_labels = []
    total_batches = len(loader)

    for i, (x, y) in enumerate(loader):
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(f"Processing batch {i + 1}/{total_batches}")

        x = x.to(device).float()
        y = y.cpu().numpy()
        mu, logvar = model.encode(x)

        all_mu.append(mu.cpu().numpy())
        all_labels.append(y)

    all_mu = np.concatenate(all_mu, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_mu, all_labels


def standardize_latents(mu: np.ndarray):
    scaler = StandardScaler()
    mu_scaled = scaler.fit_transform(mu)
    return scaler, mu_scaled


def get_projection_inputs(mu: np.ndarray, args):
    runs = []

    if args.save_both:
        _, mu_scaled = standardize_latents(mu)
        runs.append(("standardized", mu_scaled, "standardized"))
        runs.append(("raw", mu, "raw"))
    elif args.no_standardize:
        runs.append(("raw", mu, "raw"))
    else:
        _, mu_scaled = standardize_latents(mu)
        runs.append(("standardized", mu_scaled, "standardized"))

    return runs


def mode_suffix(mode_name: str, save_both: bool, no_standardize: bool) -> str:
    if save_both or no_standardize:
        return f"_{mode_name}"
    return ""


def run_pca(latents: np.ndarray):
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(latents)
    return pca, pcs


def resolve_tsne_perplexity(n_samples: int, requested_perplexity: float) -> float:
    if n_samples <= 3:
        raise ValueError(f"Need more than 3 samples for t-SNE, got {n_samples}.")

    max_valid = max(2.0, min(50.0, float(n_samples - 1) / 3.0))
    perplexity = min(requested_perplexity, max_valid)

    if perplexity != requested_perplexity:
        print(
            f"Adjusted t-SNE perplexity from {requested_perplexity} to {perplexity:.2f} "
            f"because n_samples={n_samples}."
        )

    return perplexity


def run_tsne_2d(
    latents: np.ndarray,
    perplexity: float,
    learning_rate: float,
    n_iter: int,
    random_state: int,
):
    perplexity = resolve_tsne_perplexity(latents.shape[0], perplexity)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        init="pca",
        random_state=random_state,
    )
    embedding = tsne.fit_transform(latents)
    return tsne, embedding, perplexity


def run_umap_3d(
    latents: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
):
    if umap is None:
        raise ImportError(
            "UMAP is not installed. Please install it with:\n"
            "pip install umap-learn"
        )

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(latents)
    return reducer, embedding


def save_2d_plot(
    projection: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    x_label: str,
    y_label: str,
    title: str,
):
    plt.figure(figsize=(8, 6))

    for label_id in sorted(np.unique(labels)):
        idx = labels == label_id
        plt.scatter(
            projection[idx, 0],
            projection[idx, 1],
            s=14,
            alpha=0.75,
            label=LABEL_NAMES.get(int(label_id), str(label_id)),
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_umap_3d_plot(
    embedding: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    mode_label: str,
):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for label_id in sorted(np.unique(labels)):
        idx = labels == label_id
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            embedding[idx, 2],
            s=14,
            alpha=0.75,
            label=LABEL_NAMES.get(int(label_id), str(label_id)),
        )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    ax.set_title(
        f"3D UMAP of VAE latent means (mu) [{mode_label}]\n"
        f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_projection_csv(
    projection: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    coord_names: list[str],
):
    data = {name: projection[:, i] for i, name in enumerate(coord_names)}
    data["label_id"] = labels
    data["label_name"] = [LABEL_NAMES.get(int(y), str(y)) for y in labels]
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)


def main():
    args = parse_args()
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]

    model = build_model_from_checkpoint(ckpt, device=device)
    loader = build_loader_from_checkpoint_cfg(
        ckpt_cfg=cfg,
        split=args.split,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
        test_csv=args.test_csv,
        device=device,
    )

    mu, labels = extract_mu_and_labels(model, loader, device=device)
    projection_runs = get_projection_inputs(mu, args)

    for mode_name, latents, mode_label in projection_runs:
        suffix = mode_suffix(mode_name, args.save_both, args.no_standardize)

        print(f"\nRunning projections with {mode_label} latent means...")

        # PCA
        pca, pcs = run_pca(latents)
        pca_plot_path = outdir / f"latent_pca_{args.split}{suffix}.png"
        pca_csv_path = outdir / f"latent_pca_{args.split}{suffix}.csv"

        save_2d_plot(
            projection=pcs,
            labels=labels,
            save_path=pca_plot_path,
            x_label=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
            y_label=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)",
            title=f"PCA of VAE latent means (mu) [{mode_label}]",
        )
        save_projection_csv(
            projection=pcs,
            labels=labels,
            save_path=pca_csv_path,
            coord_names=["pc1", "pc2"],
        )

        # t-SNE
        tsne, tsne_2d, used_perplexity = run_tsne_2d(
            latents=latents,
            perplexity=args.tsne_perplexity,
            learning_rate=args.tsne_learning_rate,
            n_iter=args.tsne_n_iter,
            random_state=args.tsne_random_state,
        )
        tsne_plot_path = outdir / f"latent_tsne_{args.split}{suffix}.png"
        tsne_csv_path = outdir / f"latent_tsne_{args.split}{suffix}.csv"

        save_2d_plot(
            projection=tsne_2d,
            labels=labels,
            save_path=tsne_plot_path,
            x_label="t-SNE-1",
            y_label="t-SNE-2",
            title=(
                f"t-SNE of VAE latent means (mu) [{mode_label}]\n"
                f"perplexity={used_perplexity:.2f}, "
                f"lr={args.tsne_learning_rate}, n_iter={args.tsne_n_iter}"
            ),
        )
        save_projection_csv(
            projection=tsne_2d,
            labels=labels,
            save_path=tsne_csv_path,
            coord_names=["tsne1", "tsne2"],
        )

        # UMAP
        reducer, umap_3d = run_umap_3d(
            latents=latents,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.umap_random_state,
        )
        umap_plot_path = outdir / f"latent_umap3d_{args.split}{suffix}.png"
        umap_csv_path = outdir / f"latent_umap3d_{args.split}{suffix}.csv"

        save_umap_3d_plot(
            embedding=umap_3d,
            labels=labels,
            save_path=umap_plot_path,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            mode_label=mode_label,
        )
        save_projection_csv(
            projection=umap_3d,
            labels=labels,
            save_path=umap_csv_path,
            coord_names=["umap1", "umap2", "umap3"],
        )

        print(f"Saved PCA plot to: {pca_plot_path}")
        print(f"Saved PCA CSV to: {pca_csv_path}")
        print(f"Saved t-SNE plot to: {tsne_plot_path}")
        print(f"Saved t-SNE CSV to: {tsne_csv_path}")
        print(f"Saved UMAP 3D plot to: {umap_plot_path}")
        print(f"Saved UMAP 3D CSV to: {umap_csv_path}")
        print(
            f"PCA explained variance [{mode_label}]: "
            f"PC1={pca.explained_variance_ratio_[0]:.4f}, "
            f"PC2={pca.explained_variance_ratio_[1]:.4f}"
        )
        print(
            f"t-SNE settings [{mode_label}]: "
            f"perplexity={used_perplexity:.2f}, "
            f"learning_rate={args.tsne_learning_rate}, "
            f"n_iter={args.tsne_n_iter}, "
            f"random_state={args.tsne_random_state}"
        )
        print(
            f"UMAP settings [{mode_label}]: "
            f"n_neighbors={args.umap_n_neighbors}, "
            f"min_dist={args.umap_min_dist}, "
            f"metric={args.umap_metric}, "
            f"random_state={args.umap_random_state}"
        )


if __name__ == "__main__":
    main()

# python -m nemg.experiments.van_de_leur_model.plot_latent_pca \
#   --ckpt multirun/multirun_past/van_de_leur_model_FDD_loss/2026-03-26/19-47-20/2/checkpoints/best.pt \
#   --split val \
#   --outdir outputs/van_de_leur_model/run_01/latent_pca