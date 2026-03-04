#!/usr/bin/env python3
"""
Window EMGLAB WAV recordings into fixed-length overlapping segments.

Example:
  python nemg/preprocessing/windowing.py \
    --input_dir /path/to/EMGlabDatabaseWav \
    --output_dir /path/to/EMGlab_windows_400_100 \
    --win_ms 400 --hop_ms 100 --recursive
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from scipy.io import wavfile


def ms_to_samples(ms: float, fs: int) -> int:
    return int(round((ms / 1000.0) * fs))


def make_windows_strided_1d(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """
    Create overlapping windows using NumPy stride tricks.

    Args:
        x: shape (N,)
        win: window length in samples
        hop: hop length in samples

    Returns:
        windows: shape (n_windows, win) as a view (no copy). Caller can copy if needed.
    """
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {x.shape}")
    if win <= 0 or hop <= 0:
        raise ValueError("win and hop must be positive")
    N = x.shape[0]
    if N < win:
        return np.empty((0, win), dtype=x.dtype)

    n_windows = 1 + (N - win) // hop
    stride = x.strides[0]
    # shape: (n_windows, win)
    windows = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_windows, win),
        strides=(hop * stride, stride),
        writeable=False,
    )
    return windows


def list_wavs(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(input_dir.rglob("*.wav"))
    return sorted(input_dir.glob("*.wav"))


def safe_relpath(path: Path, start: Path) -> Path:
    # Produce a relative path even if on different drives (Windows edge cases)
    try:
        return path.relative_to(start)
    except ValueError:
        return Path(path.name)


def process_one_file(
    wav_path: Path,
    input_root: Path,
    output_root: Path,
    win_ms: float,
    hop_ms: float,
    dtype_out: np.dtype = np.float32,
    scale_int16: bool = True,
    overwrite: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Returns:
      (success, message_if_skipped_or_error)
    """
    rel = safe_relpath(wav_path, input_root)
    out_path = (output_root / rel).with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return False, f"SKIP exists: {out_path}"

    fs, sig = wavfile.read(str(wav_path))

    # Expect mono (N,) based on your earlier check
    if sig.ndim != 1:
        return False, f"ERROR non-mono WAV {wav_path} shape={sig.shape}"

    # Convert to float
    x = sig.astype(dtype_out, copy=False)
    if scale_int16 and sig.dtype == np.int16:
        x = x / 32768.0

    win = ms_to_samples(win_ms, fs)
    hop = ms_to_samples(hop_ms, fs)

    windows = make_windows_strided_1d(x, win, hop)

    # Save a compact NPZ; copy to ensure contiguous array on disk
    np.savez_compressed(
        str(out_path),
        windows=np.array(windows, dtype=dtype_out, copy=True),
        fs=np.int32(fs),
        win_ms=np.float32(win_ms),
        hop_ms=np.float32(hop_ms),
        win_samp=np.int32(win),
        hop_samp=np.int32(hop),
        source_filename=str(wav_path.name),
    )

    return True, None


def main():
    parser = argparse.ArgumentParser(description="Window EMGLAB WAV dataset into .npz window files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing WAV files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write windowed .npz files.")
    parser.add_argument("--win_ms", type=float, default=400.0, help="Window length in milliseconds.")
    parser.add_argument("--hop_ms", type=float, default=100.0, help="Hop length in milliseconds.")
    parser.add_argument("--recursive", action="store_true", help="Search for WAVs recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files (0 = no limit).")
    parser.add_argument("--write_manifest", action="store_true", help="Write a manifest JSON in output_dir.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wavs = list_wavs(input_dir, args.recursive)
    if args.limit and args.limit > 0:
        wavs = wavs[: args.limit]

    if not wavs:
        raise SystemExit(f"No .wav files found under {input_dir}")

    n_ok = 0
    n_skip = 0
    n_err = 0
    errors = []

    for i, wav_path in enumerate(wavs, 1):
        ok, msg = process_one_file(
            wav_path=wav_path,
            input_root=input_dir,
            output_root=output_dir,
            win_ms=args.win_ms,
            hop_ms=args.hop_ms,
            overwrite=args.overwrite,
        )

        if ok:
            n_ok += 1
            if i % 50 == 0 or i == len(wavs):
                print(f"[{i}/{len(wavs)}] OK ({n_ok} ok, {n_skip} skip, {n_err} err)")
        else:
            if msg and msg.startswith("SKIP"):
                n_skip += 1
            else:
                n_err += 1
                errors.append({"file": str(wav_path), "error": msg})
                print(f"[{i}/{len(wavs)}] {msg}")

    print("\nDone.")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"win_ms={args.win_ms} hop_ms={args.hop_ms}")
    print(f"Processed: {len(wavs)} files -> {n_ok} ok, {n_skip} skipped, {n_err} errors")

    if args.write_manifest:
        manifest = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "win_ms": args.win_ms,
            "hop_ms": args.hop_ms,
            "recursive": bool(args.recursive),
            "num_files_seen": len(wavs),
            "num_ok": n_ok,
            "num_skipped": n_skip,
            "num_errors": n_err,
            "errors": errors[:50],  # cap
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()