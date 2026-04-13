from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from nemg.dataset.dataset import EMGWindowDataset, LABEL_TO_INT
from nemg.experiments.van_de_leur_model.model import Conv1DBetaVAE


INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
SPLIT_CHOICES = ("train", "val", "test")
FEATURE_CHOICES = ("mu", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a recording-level multinomial logistic regression classifier on "
            "the latent space of the van de Leur VAE. Window latents are extracted "
            "per recording and mean-pooled into one recording-level vector."
        )
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt or last.pt")
    parser.add_argument(
        "--feature",
        type=str,
        default="mu",
        choices=FEATURE_CHOICES,
        help="Latent feature to use before recording-level pooling: mu or z (default: mu)",
    )
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=["val"],
        choices=SPLIT_CHOICES,
        help="Extra splits to evaluate after fitting the classifier (default: val)",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="Required only if 'test' is included in --eval-splits",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override feature-extraction batch size from checkpoint config",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override feature-extraction num_workers from checkpoint config",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optionally cap extracted batches per split for quick debugging",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for LogisticRegression (default: 1.0)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum iterations for LogisticRegression (default: 2000)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"],
        help="Optimizer for LogisticRegression (default: lbfgs)",
    )
    parser.add_argument(
        "--penalty",
        type=str,
        default="l2",
        choices=["l2", "l1", "elasticnet", "none"],
        help="Penalty for LogisticRegression. Use 'none' for no penalty.",
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=None,
        help="l1_ratio for elasticnet penalty (only used with --penalty elasticnet and solver saga)",
    )
    parser.add_argument(
        "--no-class-weight-balance",
        action="store_true",
        help="Disable class_weight='balanced'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu for latent extraction",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--outdir",
        type=str,
        default="latent_recording_multinomial_logreg",
        help="Directory to save the fitted classifier, metrics, and plots",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


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
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_dataset_and_loader_from_checkpoint_cfg(
    ckpt_cfg: dict,
    split: str,
    batch_size_override: int | None,
    num_workers_override: int | None,
    test_csv: str | None,
    device: torch.device,
) -> tuple[EMGWindowDataset, DataLoader, Path]:
    data_cfg = ckpt_cfg["data"]
    batch_size = batch_size_override or data_cfg["batch_size"]
    num_workers = num_workers_override if num_workers_override is not None else data_cfg["num_workers"]
    pin_memory = should_use_pin_memory(device, data_cfg.get("pin_memory", True))

    fold_dir = Path(data_cfg["fold_dir"])
    windows_dir = Path(data_cfg["windows_dir"])
    use_downsampled = bool(data_cfg["use_downsampled"])
    cache = bool(data_cfg.get("cache", True))

    if split == "train":
        csv_path = fold_dir / "train.csv"
    elif split == "val":
        csv_path = fold_dir / "val.csv"
    elif split == "test":
        if test_csv is None:
            raise ValueError("--test-csv is required when 'test' is included in --eval-splits")
        csv_path = Path(test_csv)
    else:
        raise ValueError(f"Unsupported split: {split}")

    dataset = EMGWindowDataset.from_csv(
        csv_path=csv_path,
        windows_dir=windows_dir,
        use_downsampled=use_downsampled,
        cache=cache,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return dataset, loader, csv_path


@torch.no_grad()
def extract_recording_level_features(
    vae: Conv1DBetaVAE,
    dataset: EMGWindowDataset,
    loader: DataLoader,
    device: torch.device,
    feature_type: str = "mu",
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    if not hasattr(dataset, "_index"):
        raise AttributeError("Dataset does not expose _index; cannot recover recording IDs for aggregation.")

    sums: OrderedDict[str, np.ndarray] = OrderedDict()
    counts: OrderedDict[str, int] = OrderedDict()
    labels: OrderedDict[str, int] = OrderedDict()

    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    offset = 0
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if batch_idx == 0 or (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(f"Extracting {feature_type} features: batch {batch_idx + 1}/{total_batches}")

        batch_size = int(y.shape[0])
        batch_meta = dataset._index[offset : offset + batch_size]
        if len(batch_meta) != batch_size:
            raise RuntimeError(
                f"Metadata length mismatch while aggregating recordings: got {len(batch_meta)} metadata rows "
                f"for batch size {batch_size}."
            )
        offset += batch_size

        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()

        mu, _sd, logvar = vae.encode_distribution(x)
        if feature_type == "mu":
            feats = mu
        elif feature_type == "z":
            feats = vae.reparameterize(mu, logvar)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        feats_np = feats.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        for feat, y_item, meta in zip(feats_np, y_np, batch_meta):
            npz_path, _w_idx, label_int = meta
            rec_id = str(npz_path)

            if int(y_item) != int(label_int):
                raise RuntimeError(
                    f"Label mismatch for recording {rec_id}: batch label={int(y_item)} metadata label={int(label_int)}"
                )

            if rec_id not in sums:
                sums[rec_id] = feat.astype(np.float64, copy=True)
                counts[rec_id] = 1
                labels[rec_id] = int(label_int)
            else:
                sums[rec_id] += feat
                counts[rec_id] += 1
                if labels[rec_id] != int(label_int):
                    raise RuntimeError(
                        f"Inconsistent labels inside recording {rec_id}: {labels[rec_id]} vs {int(label_int)}"
                    )

    if not sums:
        raise RuntimeError("No recording-level features were extracted. Check your dataloader or --max-batches.")

    record_ids = list(sums.keys())
    x_rec = np.stack([sums[r] / counts[r] for r in record_ids], axis=0).astype(np.float32)
    y_rec = np.asarray([labels[r] for r in record_ids], dtype=np.int64)
    n_windows = np.asarray([counts[r] for r in record_ids], dtype=np.int64)
    return x_rec, y_rec, record_ids, n_windows


def evaluate_classifier(clf: LogisticRegression, x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    y_pred = clf.predict(x)
    cm = confusion_matrix(y, y_pred, labels=np.arange(len(LABEL_TO_INT)))
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path, title: str, dpi: int) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    max_value = cm.max() if cm.size > 0 else 0
    threshold = max_value / 2.0 if max_value > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def save_coefficient_heatmap(coef: np.ndarray, class_names: list[str], out_path: Path, dpi: int) -> None:
    latent_dims = np.arange(1, coef.shape[1] + 1)
    plt.figure(figsize=(max(8, coef.shape[1] * 0.35), 3.5))
    plt.imshow(coef, aspect="auto")
    plt.colorbar(label="Coefficient")
    plt.xticks(np.arange(coef.shape[1]), latent_dims)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel("Latent dimension")
    plt.ylabel("Class")
    plt.title("Recording-level multinomial logistic regression coefficients")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt)
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    vae = build_model_from_checkpoint(ckpt, device)

    eval_splits = list(dict.fromkeys(args.eval_splits))
    if "val" not in eval_splits:
        eval_splits = ["val"] + eval_splits

    split_specs: dict[str, tuple[EMGWindowDataset, DataLoader, Path]] = {}
    for split_name in ["train", "val"] + (["test"] if "test" in eval_splits else []):
        dataset, loader, csv_path = build_dataset_and_loader_from_checkpoint_cfg(
            ckpt["cfg"],
            split=split_name,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
            test_csv=args.test_csv,
            device=device,
        )
        split_specs[split_name] = (dataset, loader, csv_path)
        print(f"Prepared {split_name} dataset from {csv_path}")

    split_features: dict[str, np.ndarray] = {}
    split_labels: dict[str, np.ndarray] = {}
    split_record_ids: dict[str, list[str]] = {}
    split_n_windows: dict[str, np.ndarray] = {}

    for split_name, (dataset, loader, _csv_path) in split_specs.items():
        x_rec, y_rec, record_ids, n_windows = extract_recording_level_features(
            vae=vae,
            dataset=dataset,
            loader=loader,
            device=device,
            feature_type=args.feature,
            max_batches=args.max_batches,
        )
        split_features[split_name] = x_rec
        split_labels[split_name] = y_rec
        split_record_ids[split_name] = record_ids
        split_n_windows[split_name] = n_windows
        print(
            f"{split_name}: recordings={x_rec.shape[0]}, latent_dim={x_rec.shape[1]}, "
            f"mean_windows_per_recording={float(n_windows.mean()):.2f}"
        )

        pd.DataFrame(
            {
                "recording_id": record_ids,
                "label_int": y_rec,
                "label_name": [INT_TO_LABEL[int(v)] for v in y_rec],
                "n_windows": n_windows,
            }
        ).to_csv(outdir / f"recordings_{split_name}.csv", index=False)

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(split_features["train"])
    split_features_std = {"train": x_train_std}
    for split_name, x in split_features.items():
        if split_name == "train":
            continue
        split_features_std[split_name] = scaler.transform(x)

    class_weight = None if args.no_class_weight_balance else "balanced"

    penalty = None if args.penalty == "none" else args.penalty
    if penalty == "elasticnet" and args.solver != "saga":
        raise ValueError("elasticnet penalty requires --solver saga")
    if penalty == "l1" and args.solver not in {"saga"}:
        raise ValueError("l1 penalty requires solver saga for multinomial logistic regression")

    clf = LogisticRegression(
        multi_class="multinomial",
        solver=args.solver,
        penalty=penalty,
        C=args.C,
        l1_ratio=args.l1_ratio,
        class_weight=class_weight,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    clf.fit(split_features_std["train"], split_labels["train"])

    num_classes = len(LABEL_TO_INT)
    class_names = [INT_TO_LABEL[i] for i in range(num_classes)]

    results_rows: list[dict[str, float | str | int]] = []
    for split_name in ["train"] + eval_splits:
        result = evaluate_classifier(clf, split_features_std[split_name], split_labels[split_name])
        print(
            f"[{split_name}] acc={result['accuracy']:.4f} "
            f"balanced_acc={result['balanced_accuracy']:.4f} macro_f1={result['macro_f1']:.4f}"
        )
        results_rows.append(
            {
                "split": split_name,
                "n_recordings": int(split_features_std[split_name].shape[0]),
                "accuracy": result["accuracy"],
                "balanced_accuracy": result["balanced_accuracy"],
                "macro_f1": result["macro_f1"],
            }
        )
        cm = result["confusion_matrix"]
        np.save(outdir / f"confusion_matrix_{split_name}.npy", cm)
        save_confusion_matrix(
            cm,
            class_names=class_names,
            out_path=outdir / f"confusion_matrix_{split_name}.png",
            title=f"Recording-level latent multinomial logistic regression ({split_name})",
            dpi=args.dpi,
        )

        pred_df = pd.DataFrame(
            {
                "recording_id": split_record_ids[split_name],
                "y_true": split_labels[split_name],
                "y_true_name": [INT_TO_LABEL[int(v)] for v in split_labels[split_name]],
                "y_pred": result["y_pred"],
                "y_pred_name": [INT_TO_LABEL[int(v)] for v in result["y_pred"]],
                "n_windows": split_n_windows[split_name],
            }
        )
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(split_features_std[split_name])
            for class_idx, class_name in enumerate(class_names):
                pred_df[f"prob_{class_name}"] = probs[:, class_idx]
        pred_df.to_csv(outdir / f"predictions_{split_name}.csv", index=False)

    metrics_df = pd.DataFrame(results_rows)
    metrics_df.to_csv(outdir / "metrics.csv", index=False)

    coef_df = pd.DataFrame(
        clf.coef_,
        index=class_names,
        columns=[f"latent_{i}" for i in range(1, clf.coef_.shape[1] + 1)],
    )
    coef_df.to_csv(outdir / "coefficients.csv")
    intercept_df = pd.DataFrame({"class": class_names, "intercept": clf.intercept_})
    intercept_df.to_csv(outdir / "intercepts.csv", index=False)

    abs_importance = np.abs(clf.coef_).mean(axis=0)
    importance_df = pd.DataFrame(
        {
            "latent_dim": np.arange(1, clf.coef_.shape[1] + 1),
            "mean_abs_coef": abs_importance,
        }
    ).sort_values("mean_abs_coef", ascending=False)
    importance_df.to_csv(outdir / "latent_importance_mean_abs_coef.csv", index=False)

    save_coefficient_heatmap(clf.coef_, class_names, outdir / "coefficients_heatmap.png", dpi=args.dpi)

    with open(outdir / "multinomial_logreg.pkl", "wb") as f:
        pickle.dump(
            {
                "classifier": clf,
                "scaler": scaler,
                "feature_type": args.feature,
                "pooling": "mean_per_recording",
                "class_names": class_names,
                "label_to_int": LABEL_TO_INT,
                "ckpt_path": str(ckpt_path),
                "metrics": results_rows,
            },
            f,
        )

    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
