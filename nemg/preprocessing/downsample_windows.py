#!/usr/bin/env python3
"""
Apply MinMax downsampling to windowed .npz files produced by windowing.py.

Expected input .npz format:
  - windows: shape (n_windows, win_len)
  - fs
  - win_ms
  - hop_ms
  - win_samp
  - hop_samp
  - source_filename

Example:
  python nemg/preprocessing/downsample_windows.py \
    --input_dir "data/emglab/windows_w400_h100" \
    --output_dir "data/emglab/windows_w400_h100_minmax_f25" \
    --factor 25 \
    --limit 10 \
    --write_manifest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tsdownsample import MinMaxDownsampler


def list_npzs(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(input_dir.rglob("*.npz"))
    return sorted(input_dir.glob("*.npz"))


def safe_relpath(path: Path, start: Path) -> Path:
    try:
        return path.relative_to(start)
    except ValueError:
        return Path(path.name)


def compute_n_out(win_len: int, factor: int) -> int:
    if factor <= 1:
        return win_len

    n_out = win_len // factor

    # MinMaxDownsampler expects an even output size
    if n_out % 2 != 0:
        n_out -= 1

    if n_out < 2:
        raise ValueError(
            f"Window too short for factor={factor}: "
            f"win_len={win_len} -> n_out={n_out}"
        )

    return n_out


def downsample_windows_minmax(
    windows: np.ndarray,
    factor: int,
    dtype_out: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample each window independently using MinMaxDownsampler.

    Args:
        windows: shape (n_windows, win_len)
        factor: downsampling factor
        dtype_out: dtype of output downsampled windows

    Returns:
        windows_ds: shape (n_windows, n_out)
        selected_indices: shape (n_windows, n_out), indices within each original window
    """
    if windows.ndim != 2:
        raise ValueError(f"Expected 2D windows array, got shape {windows.shape}")

    n_windows, win_len = windows.shape
    n_out = compute_n_out(win_len, factor)

    if factor <= 1:
        idx = np.arange(win_len, dtype=np.int32)
        selected_indices = np.broadcast_to(idx, (n_windows, win_len)).copy()
        windows_ds = np.array(windows, dtype=dtype_out, copy=True)
        return windows_ds, selected_indices

    downsampler = MinMaxDownsampler()

    windows_ds = np.empty((n_windows, n_out), dtype=dtype_out)
    selected_indices = np.empty((n_windows, n_out), dtype=np.int32)

    for i in range(n_windows):
        w = windows[i]
        idx = downsampler.downsample(w, n_out=n_out)
        idx = np.asarray(idx, dtype=np.int32)

        # Ensure time order inside each window
        idx.sort()

        selected_indices[i] = idx
        windows_ds[i] = w[idx].astype(dtype_out, copy=False)

    return windows_ds, selected_indices


def process_one_file(
    npz_path: Path,
    input_root: Path,
    output_root: Path,
    factor: int,
    dtype_out: np.dtype = np.float32,
    overwrite: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Returns:
      (success, message_if_skipped_or_error)
    """
    rel = safe_relpath(npz_path, input_root)
    out_path = output_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return False, f"SKIP exists: {out_path}"

    with np.load(str(npz_path), allow_pickle=False) as data:
        if "windows" not in data:
            return False, f"ERROR missing 'windows' array in {npz_path}"

        windows = data["windows"]

        if windows.ndim != 2:
            return False, f"ERROR expected windows shape (n_windows, win_len), got {windows.shape}"

        windows = windows.astype(dtype_out, copy=False)
        n_windows, win_len = windows.shape
        n_out = compute_n_out(win_len, factor)

        windows_ds, selected_indices = downsample_windows_minmax(
            windows=windows,
            factor=factor,
            dtype_out=dtype_out,
        )

        save_dict = {
            "windows_downsampled": windows_ds,
            "selected_indices": selected_indices,
            "factor": np.int32(factor),
            "n_windows": np.int32(n_windows),
            "original_win_len": np.int32(win_len),
            "downsampled_win_len": np.int32(n_out),
        }

        # Copy over known metadata if present
        for key in ["fs", "win_ms", "hop_ms", "win_samp", "hop_samp", "source_filename"]:
            if key in data:
                save_dict[key] = data[key]

    np.savez_compressed(str(out_path), **save_dict)
    return True, None


def main():
    parser = argparse.ArgumentParser(
        description="Apply MinMax downsampling to windowed .npz files."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing windowed .npz files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write downsampled .npz files.")
    parser.add_argument("--factor", type=int, required=True, help="Downsampling factor (e.g. 25).")
    parser.add_argument("--recursive", action="store_true", help="Search for .npz files recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files (0 = no limit).")
    parser.add_argument("--write_manifest", action="store_true", help="Write a manifest JSON in output_dir.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = list_npzs(input_dir, args.recursive)
    if args.limit and args.limit > 0:
        npz_files = npz_files[: args.limit]

    if not npz_files:
        raise SystemExit(f"No .npz files found under {input_dir}")

    n_ok = 0
    n_skip = 0
    n_err = 0
    errors = []

    for i, npz_path in enumerate(npz_files, 1):
        ok, msg = process_one_file(
            npz_path=npz_path,
            input_root=input_dir,
            output_root=output_dir,
            factor=args.factor,
            overwrite=args.overwrite,
        )

        if ok:
            n_ok += 1
            if i % 50 == 0 or i == len(npz_files):
                print(f"[{i}/{len(npz_files)}] OK ({n_ok} ok, {n_skip} skip, {n_err} err)")
        else:
            if msg and msg.startswith("SKIP"):
                n_skip += 1
            else:
                n_err += 1
                errors.append({"file": str(npz_path), "error": msg})
                print(f"[{i}/{len(npz_files)}] {msg}")

    print("\nDone.")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"factor={args.factor}")
    print(f"Processed: {len(npz_files)} files -> {n_ok} ok, {n_skip} skipped, {n_err} errors")

    if args.write_manifest:
        manifest = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "factor": args.factor,
            "recursive": bool(args.recursive),
            "num_files_seen": len(npz_files),
            "num_ok": n_ok,
            "num_skipped": n_skip,
            "num_errors": n_err,
            "errors": errors[:50],
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()


# Example:
# python nemg/preprocessing/downsample_windows.py \
#   --input_dir "data/emglab/windows_w400_h100" \
#   --output_dir "data/emglab/windows_w400_h100_minmax_f25" \
#   --factor 25 \
#   --limit 10 \
#   --write_manifest