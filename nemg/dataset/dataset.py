#!/usr/bin/env python3
"""
Dataset and DataLoader utilities for windowed nEMG signals (EMGLab/AMC).

Supports:
  - Raw windowed .npz files (from windowing.py)
  - MinMax-downsampled .npz files (from downsample_windows.py)
  - 5-fold cross-validation splits (train.csv / val.csv / test.csv)
  - Flexible window-level sampling across different window sizes
  - Optional weighted sampling for imbalanced training sets

Key design choice:
  The dataset uses the actual number of windows inside each .npz file
  as the source of truth and completely ignores any n_windows column
  in the CSV. This makes it compatible with different windowing
  configurations (e.g. w400, w2000, etc.) as long as the CSV still
  points to the correct files.

Usage example:
    from dataset import EMGWindowDataset, build_dataloaders

    train_ds = EMGWindowDataset.from_csv(
        csv_path="data/emglab/splits/fold_0/train.csv",
        windows_dir="data/emglab/windows_w400_h100",
        use_downsampled=False,
    )

    train_loader, val_loader = build_dataloaders(
        fold_dir="data/emglab/splits/fold_0",
        windows_dir="data/emglab/windows_w400_h100",
        use_downsampled=False,
        batch_size=64,
    )
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

# Keep the original label strings because your AMC split CSVs already use them.
LABEL_TO_INT: dict[str, int] = {
    "ALS": 0,
    "Normal": 1,
    "Myopathy": 2,
}


def encode_label(label: str) -> int:
    """Map string label to integer class index. Raises KeyError on unknown."""
    if label not in LABEL_TO_INT:
        raise KeyError(
            f"Unknown label '{label}'. Known labels: {list(LABEL_TO_INT.keys())}"
        )
    return LABEL_TO_INT[label]


# ---------------------------------------------------------------------------
# Core Dataset
# ---------------------------------------------------------------------------


class EMGWindowDataset(Dataset):
    """
    A window-level dataset for nEMG recordings.

    Each item is a single window (1-D signal) drawn from a recording's .npz file.
    Windows are lazily loaded per recording the first time a window from that
    recording is requested, then optionally cached in memory.

    Required CSV columns (raw mode):
        - raw_npz
        - label

    Required CSV columns (downsampled mode):
        - ds_npz
        - ds_exists
        - label

    Notes:
        - Any n_windows column in the CSV is ignored.
        - The actual number of windows is always read from the .npz file.
    """

    def __init__(
        self,
        records: pd.DataFrame,
        windows_dir: Union[str, Path],
        use_downsampled: bool = False,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        label_transform: Optional[Callable[[int], int]] = None,
        cache: bool = True,
    ):
        self.windows_dir = Path(windows_dir)
        self.use_downsampled = use_downsampled
        self.transform = transform
        self.label_transform = label_transform
        self.cache = cache

        # Flat index: list of (npz_path, local_window_idx, label_int)
        self._index: list[tuple[Path, int, int]] = []

        # Optional in-memory cache: npz_path -> ndarray of shape (n_windows, win_len)
        self._npz_cache: dict[Path, np.ndarray] = {}

        self._build_index(records)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _get_array_key(self) -> str:
        return "windows_downsampled" if self.use_downsampled else "windows"

    def _get_file_n_windows(self, npz_path: Path) -> int:
        """Read the true number of windows directly from the NPZ file."""
        key = self._get_array_key()
        with np.load(str(npz_path), allow_pickle=False) as data:
            if key not in data:
                available = list(data.keys())
                raise KeyError(
                    f"Key '{key}' not found in {npz_path}. "
                    f"Available keys: {available}"
                )
            return int(data[key].shape[0])

    def _build_index(self, records: pd.DataFrame) -> None:
        npz_col = "ds_npz" if self.use_downsampled else "raw_npz"

        required = {npz_col, "label"}
        if self.use_downsampled:
            required.add("ds_exists")

        missing = required - set(records.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        for _, row in records.iterrows():
            if self.use_downsampled and not bool(row["ds_exists"]):
                continue

            npz_rel = row[npz_col]
            npz_path = self.windows_dir / str(npz_rel)

            if not npz_path.exists():
                raise FileNotFoundError(
                    f"NPZ file not found: {npz_path}\n"
                    f"  (windows_dir={self.windows_dir}, npz_col={npz_col})"
                )

            label_int = encode_label(str(row["label"]))
            actual_n_windows = self._get_file_n_windows(npz_path)

            for w_idx in range(actual_n_windows):
                self._index.append((npz_path, w_idx, label_int))

    def _load_windows(self, npz_path: Path) -> np.ndarray:
        """Load and optionally cache the full windows array for one recording."""
        if npz_path in self._npz_cache:
            return self._npz_cache[npz_path]

        key = self._get_array_key()
        with np.load(str(npz_path), allow_pickle=False) as data:
            if key not in data:
                available = list(data.keys())
                raise KeyError(
                    f"Key '{key}' not found in {npz_path}. "
                    f"Available keys: {available}"
                )
            arr = data[key].astype(np.float32, copy=False)  # (n_windows, win_len)

        if self.cache:
            self._npz_cache[npz_path] = arr

        return arr

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        npz_path, w_idx, label_int = self._index[idx]
        windows = self._load_windows(npz_path)

        if w_idx >= windows.shape[0]:
            raise IndexError(
                f"Window index {w_idx} out of bounds for file {npz_path}. "
                f"File contains {windows.shape[0]} windows."
            )

        window = torch.from_numpy(windows[w_idx])  # (win_len,)

        if self.transform is not None:
            window = self.transform(window)

        if self.label_transform is not None:
            label_int = self.label_transform(label_int)

        return window, label_int

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def win_len(self) -> int:
        """Length of each window in samples (inferred from first file)."""
        if not self._index:
            raise RuntimeError("Dataset is empty.")
        npz_path = self._index[0][0]
        arr = self._load_windows(npz_path)
        return int(arr.shape[1])

    @property
    def n_classes(self) -> int:
        return len(LABEL_TO_INT)

    @property
    def labels(self) -> list[int]:
        """All integer labels in index order."""
        return [label for _, _, label in self._index]

    # ------------------------------------------------------------------
    # Factory: build from a CSV path
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        windows_dir: Union[str, Path],
        use_downsampled: bool = False,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        label_transform: Optional[Callable[[int], int]] = None,
        cache: bool = True,
    ) -> "EMGWindowDataset":
        """
        Build a dataset directly from a split CSV file.

        Args:
            csv_path: path to train.csv / val.csv / test.csv
            windows_dir: root directory for .npz files
            use_downsampled: use downsampled windows if True
            transform: optional window-level transform
            label_transform: optional label transform
            cache: cache loaded arrays in RAM
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        records = pd.read_csv(csv_path)

        return cls(
            records=records,
            windows_dir=windows_dir,
            use_downsampled=use_downsampled,
            transform=transform,
            label_transform=label_transform,
            cache=cache,
        )


# ---------------------------------------------------------------------------
# Sampler helper
# ---------------------------------------------------------------------------


def build_weighted_sampler(ds: EMGWindowDataset) -> WeightedRandomSampler:
    """
    Build a window-level weighted sampler from dataset labels.
    Minority classes get sampled more often.
    """
    counts = Counter(ds.labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = torch.DoubleTensor([class_weights[label] for label in ds.labels])

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------


def build_dataloaders(
    fold_dir: Union[str, Path],
    windows_dir: Union[str, Path],
    use_downsampled: bool = False,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transform: Optional[Callable[[Tensor], Tensor]] = None,
    val_transform: Optional[Callable[[Tensor], Tensor]] = None,
    cache: bool = True,
    drop_last_train: bool = True,
    use_weighted_sampler: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders for one fold.
    """
    fold_dir = Path(fold_dir)

    train_ds = EMGWindowDataset.from_csv(
        csv_path=fold_dir / "train.csv",
        windows_dir=windows_dir,
        use_downsampled=use_downsampled,
        transform=train_transform,
        cache=cache,
    )
    val_ds = EMGWindowDataset.from_csv(
        csv_path=fold_dir / "val.csv",
        windows_dir=windows_dir,
        use_downsampled=use_downsampled,
        transform=val_transform,
        cache=cache,
    )

    train_loader_kwargs = dict(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        persistent_workers=num_workers > 0,
    )

    if use_weighted_sampler:
        train_loader_kwargs["sampler"] = build_weighted_sampler(train_ds)
        train_loader_kwargs["shuffle"] = False
    else:
        train_loader_kwargs["shuffle"] = True

    train_loader = DataLoader(**train_loader_kwargs)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


def build_test_loader(
    test_csv: Union[str, Path],
    windows_dir: Union[str, Path],
    use_downsampled: bool = False,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    cache: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for the held-out test set.
    """
    test_ds = EMGWindowDataset.from_csv(
        csv_path=test_csv,
        windows_dir=windows_dir,
        use_downsampled=use_downsampled,
        cache=cache,
    )

    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Sanity-check helper
# ---------------------------------------------------------------------------


def dataset_info(ds: EMGWindowDataset) -> None:
    """Print a quick summary of the dataset."""
    label_int_to_str = {v: k for k, v in LABEL_TO_INT.items()}
    counts = Counter(ds.labels)

    print(f"  Total windows   : {len(ds):,}")
    print(f"  Window length   : {ds.win_len} samples")
    print(f"  Use downsampled : {ds.use_downsampled}")
    print("  Class distribution:")
    for label_int, count in sorted(counts.items()):
        name = label_int_to_str.get(label_int, str(label_int))
        print(f"    {name:12s} ({label_int}): {count:,} windows")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    use_ds = "--ds" in args
    args = [a for a in args if a != "--ds"]

    if len(args) < 2:
        print(
            "Usage: python dataset.py <fold_dir> <windows_dir> [--ds]\n"
            "  --ds  use downsampled windows"
        )
        sys.exit(1)

    fold_dir, windows_dir = args[0], args[1]

    print(f"\n=== Train set ({'downsampled' if use_ds else 'raw'}) ===")
    train_ds = EMGWindowDataset.from_csv(
        csv_path=Path(fold_dir) / "train.csv",
        windows_dir=windows_dir,
        use_downsampled=use_ds,
    )
    dataset_info(train_ds)

    print(f"\n=== Val set ({'downsampled' if use_ds else 'raw'}) ===")
    val_ds = EMGWindowDataset.from_csv(
        csv_path=Path(fold_dir) / "val.csv",
        windows_dir=windows_dir,
        use_downsampled=use_ds,
    )
    dataset_info(val_ds)

    print("\n=== Checking first batch ===")
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    windows, labels = next(iter(loader))
    print(f"  windows shape : {windows.shape}  dtype={windows.dtype}")
    print(f"  labels        : {labels.tolist()}")
    print("\nAll checks passed.")