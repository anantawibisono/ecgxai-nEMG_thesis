#!/usr/bin/env python3
"""
Create patient-level 5-fold CV splits for windowed nEMG .npz files.

Input: directory of windowed .npz produced by windowing.py, where each file contains:
  - windows: (n_windows, win_len)
  - fs, win_ms, hop_ms, win_samp, hop_samp, source_filename (optional)

We parse EMGLab filenames like: N2001A01BB02(.npz/.wav)

Output:
  output_dir/
    all_files.csv
    test.csv                  (if --test_frac > 0)
    fold_0/train.csv
    fold_0/val.csv
    ...
    fold_4/train.csv
    fold_4/val.csv
    split_summary.json

Example:
  python nemg/preprocessing/make_patient_splits.py \
    --raw_npz_dir "data/emglab/windows_w400_h100" \
    --ds_npz_dir  "data/emglab/windows_w400_h100_minmax_f25" \
    --output_dir  "data/emglab/splits_w400_h100" \
    --k 5 --test_frac 0.10 --recursive
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd


LABEL_MAP = {"A": "ALS", "M": "Myopathy", "C": "Normal"}


def parse_args():
    p = argparse.ArgumentParser(description="Create patient-level K-fold splits for nEMG windowed NPZ files.")
    p.add_argument("--raw_npz_dir", type=str, required=True, help="Directory with windowed RAW .npz files.")
    p.add_argument("--ds_npz_dir", type=str, default="", help="Optional directory with downsampled .npz files.")
    p.add_argument("--output_dir", type=str, required=True, help="Where to write split CSVs.")
    p.add_argument("--k", type=int, default=5, help="Number of CV folds (default 5).")
    p.add_argument("--test_frac", type=float, default=0.10, help="Fraction of subjects per prefix to hold out as test (0 disables).")
    p.add_argument("--seed_test", type=int, default=0, help="Seed for test subject selection.")
    p.add_argument("--seed_balance", type=int, default=42, help="Seed for greedy balancing tie-breaks.")
    p.add_argument("--recursive", action="store_true", help="Search for .npz recursively.")
    p.add_argument("--limit", type=int, default=0, help="Process only first N files (0=no limit).")
    p.add_argument("--allowed_prefixes", type=str, default="A,M,C", help="Comma-separated allowed prefixes (default A,M,C).")
    p.add_argument(
        "--filename_regex",
        type=str,
        default=r"N2001([AMC])(\d+)([A-Z]{2})(\d+)$",
        help="Regex (without extension) to parse EMGLab filename stem.",
    )
    p.add_argument(
        "--count_mode",
        type=str,
        default="windows",
        choices=["windows", "files"],
        help="Balance by 'windows' (recommended) or by 'files' per subject.",
    )
    return p.parse_args()


def list_npzs(root: Path, recursive: bool) -> List[Path]:
    return sorted(root.rglob("*.npz")) if recursive else sorted(root.glob("*.npz"))


def safe_relpath(path: Path, start: Path) -> Path:
    try:
        return path.relative_to(start)
    except ValueError:
        return Path(path.name)


def parse_emglab_stem(stem: str, pattern: re.Pattern) -> Optional[Dict]:
    """
    Parses stems like 'N2001A01BB02' (no extension).
    Returns: label_code/prefix, subject, label, muscle, rec_id
    """
    m = pattern.match(stem)
    if not m:
        return None
    label_code = m.group(1)  # A/M/C
    subject_id = m.group(2)  # digits
    muscle = m.group(3)      # BB etc
    rec_id = int(m.group(4))
    return {
        "prefix": label_code,
        "subject": f"{label_code}{subject_id}",
        "label": LABEL_MAP.get(label_code, label_code),
        "muscle": muscle,
        "rec_id": rec_id,
    }


def get_n_windows(npz_path: Path) -> int:
    """
    Reads only what's needed to get number of windows.
    Tries 'windows' first, then 'windows_downsampled'.
    """
    with np.load(str(npz_path), allow_pickle=False) as d:
        if "windows" in d:
            return int(d["windows"].shape[0])
        if "windows_downsampled" in d:
            return int(d["windows_downsampled"].shape[0])
        if "n_windows" in d:
            return int(d["n_windows"])
    raise ValueError(f"Cannot infer n_windows from {npz_path.name} (missing windows/windows_downsampled/n_windows).")


def greedy_balance_subjects(subject_weights: pd.Series, k: int, random_state: int = 0) -> Dict[str, int]:
    """
    Assign subjects into k bins to balance total weight (greedy: heaviest -> lightest bin).
    subject_weights: index=subject, value=weight (e.g. total windows)
    """
    rng = np.random.default_rng(random_state)

    tmp = subject_weights.sort_values(ascending=False).reset_index()
    tmp.columns = ["subject", "w"]
    tmp["tie"] = rng.random(len(tmp))
    tmp = tmp.sort_values(["w", "tie"], ascending=[False, True])

    totals = np.zeros(k, dtype=np.int64)
    subj2bin: Dict[str, int] = {}

    for subj, w in zip(tmp["subject"], tmp["w"]):
        b = int(np.argmin(totals))
        subj2bin[str(subj)] = b
        totals[b] += int(w)

    return subj2bin


def main():
    args = parse_args()

    raw_root = Path(args.raw_npz_dir).expanduser().resolve()
    ds_root = Path(args.ds_npz_dir).expanduser().resolve() if args.ds_npz_dir else None
    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    allowed_prefixes = tuple([s.strip() for s in args.allowed_prefixes.split(",") if s.strip()])
    pat = re.compile(args.filename_regex)

    raw_files = list_npzs(raw_root, args.recursive)
    if args.limit and args.limit > 0:
        raw_files = raw_files[: args.limit]
    if not raw_files:
        raise SystemExit(f"No .npz files found in {raw_root}")

    rows = []
    n_bad = 0

    for p in raw_files:
        rel = safe_relpath(p, raw_root)
        stem = p.stem  # filename without .npz

        meta = parse_emglab_stem(stem, pat)
        if meta is None:
            n_bad += 1
            continue
        if meta["prefix"] not in allowed_prefixes:
            continue

        # link downsampled if provided
        ds_rel = rel
        ds_path = (ds_root / ds_rel) if ds_root else None
        ds_exists = bool(ds_path.exists()) if ds_path else False

        # weights for balancing
        if args.count_mode == "windows":
            n_windows = get_n_windows(p)
        else:
            n_windows = 1

        rows.append({
            "raw_npz": str(rel).replace("\\", "/"),
            "ds_npz": (str(ds_rel).replace("\\", "/") if ds_root else ""),
            "ds_exists": ds_exists,
            "filename_stem": stem,
            **meta,
            "n_windows": n_windows,
        })

    if not rows:
        raise SystemExit("No files matched your regex/prefix filter. Check --filename_regex / --allowed_prefixes.")

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "all_files.csv", index=False)

    # -----------------------------
    # 1) Hold out TEST subjects (per prefix)
    # -----------------------------
    rng_test = np.random.default_rng(args.seed_test)
    test_subjects = set()

    if args.test_frac and args.test_frac > 0:
        for pr in allowed_prefixes:
            subjects_pr = sorted(df.loc[df["prefix"] == pr, "subject"].unique().tolist())
            if not subjects_pr:
                continue
            n_test = max(1, int(round(args.test_frac * len(subjects_pr))))
            n_test = min(n_test, len(subjects_pr))
            chosen = rng_test.choice(subjects_pr, size=n_test, replace=False).tolist()
            test_subjects.update(chosen)

    df["split"] = "cv"
    if test_subjects:
        df.loc[df["subject"].isin(test_subjects), "split"] = "test"

    df_test = df[df["split"] == "test"].copy()
    df_cv = df[df["split"] == "cv"].copy()

    if test_subjects:
        df_test.to_csv(out_root / "test.csv", index=False)

    # -----------------------------
    # 2) Assign CV subjects to fold bins, stratified by prefix + balanced by weight
    # -----------------------------
    subject_weights_cv = (
        df_cv.groupby(["prefix", "subject"])["n_windows"]
        .sum()
        .reset_index()
    )

    subj_to_fold: Dict[str, int] = {}

    for pr in allowed_prefixes:
        sub_df = subject_weights_cv[subject_weights_cv["prefix"] == pr].copy()
        if sub_df.empty:
            continue

        weights = pd.Series(sub_df["n_windows"].values, index=sub_df["subject"].astype(str))
        subj2bin = greedy_balance_subjects(weights, k=args.k, random_state=args.seed_balance)

        for s, b in subj2bin.items():
            subj_to_fold[s] = b

    df_cv["fold_id"] = df_cv["subject"].map(subj_to_fold)
    if df_cv["fold_id"].isna().any():
        missing = sorted(df_cv.loc[df_cv["fold_id"].isna(), "subject"].unique().tolist())
        raise ValueError(f"Some CV subjects did not get a fold assignment: {missing}")

    df_cv["fold_id"] = df_cv["fold_id"].astype(int)

    # -----------------------------
    # 3) Write fold train/val CSVs
    # -----------------------------
    fold_stats = []
    for f in range(args.k):
        fold_dir = out_root / f"fold_{f}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_df = df_cv[df_cv["fold_id"] == f].copy()
        train_df = df_cv[df_cv["fold_id"] != f].copy()

        train_df.to_csv(fold_dir / "train.csv", index=False)
        val_df.to_csv(fold_dir / "val.csv", index=False)

        fold_stats.append({
            "fold": f,
            "train_files": int(len(train_df)),
            "val_files": int(len(val_df)),
            "train_windows": int(train_df["n_windows"].sum()),
            "val_windows": int(val_df["n_windows"].sum()),
            "val_prefix_counts": val_df.groupby("prefix")["raw_npz"].count().to_dict(),
        })

    # -----------------------------
    # 4) Summary JSON
    # -----------------------------
    summary = {
        "raw_npz_dir": str(raw_root),
        "ds_npz_dir": str(ds_root) if ds_root else "",
        "output_dir": str(out_root),
        "k": args.k,
        "test_frac": args.test_frac,
        "allowed_prefixes": list(allowed_prefixes),
        "count_mode": args.count_mode,
        "num_files_total": int(len(df)),
        "num_files_test": int(len(df_test)),
        "num_files_cv": int(len(df_cv)),
        "num_subjects_total": int(df["subject"].nunique()),
        "num_subjects_test": int(len(test_subjects)),
        "num_subjects_cv": int(df_cv["subject"].nunique()),
        "test_subjects": sorted(list(test_subjects)),
        "fold_stats": fold_stats,
    }
    (out_root / "split_summary.json").write_text(json.dumps(summary, indent=2))

    # -----------------------------
    # 5) Print quick overview
    # -----------------------------
    print("Done.")
    print(f"Parsed files: {len(df)} (skipped non-matching stems: {n_bad})")
    print(f"Subjects total: {df['subject'].nunique()}")
    print(f"Test subjects: {len(test_subjects)} | Test files: {len(df_test)}")
    print(f"CV subjects: {df_cv['subject'].nunique()} | CV files: {len(df_cv)}")
    print(f"Splits written to: {out_root}")
    print("Per-fold summary (files/windows):")
    for s in fold_stats:
        print(f"  fold {s['fold']}: train {s['train_files']}/{s['train_windows']} | val {s['val_files']}/{s['val_windows']} | val prefixes {s['val_prefix_counts']}")


if __name__ == "__main__":
    main()