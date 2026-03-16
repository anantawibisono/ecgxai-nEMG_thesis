#!/usr/bin/env python3
"""
Run dataset smoke tests across all folds and all window directories.

It executes commands like:
    python nemg/dataset/dataset.py data/emglab/splits/fold_0 data/emglab/windows_w1000_h100
    python nemg/dataset/dataset.py data/emglab/splits/fold_0 data/emglab/windows_w1000_h100_minmax_f25 --ds

Features:
- tests every fold_* directory
- tests every windows_* directory
- auto-detects downsampled folders and adds --ds
- captures stdout/stderr for every run
- prints a console summary
- writes a comprehensive .txt and .json report

Usage:
    source ~/venvs/nemg/bin/activate
    python nemg/check_dataset_matrix.py

Optional:
    python nemg/check_dataset_matrix.py \
        --splits-root data/emglab/splits \
        --windows-root data/emglab \
        --dataset-script nemg/dataset/dataset.py \
        --output-dir reports/dataset_checks
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def discover_folds(splits_root: Path) -> list[Path]:
    folds = [p for p in splits_root.iterdir() if p.is_dir() and p.name.startswith("fold_")]
    return sorted(folds, key=lambda p: p.name)


def discover_window_dirs(windows_root: Path) -> list[Path]:
    dirs = [p for p in windows_root.iterdir() if p.is_dir() and p.name.startswith("windows_")]
    return sorted(dirs, key=lambda p: p.name)


def is_downsampled_dir(path: Path) -> bool:
    name = path.name.lower()
    return "minmax" in name or "downsampled" in name


def run_one(
    python_exe: str,
    dataset_script: Path,
    fold_dir: Path,
    windows_dir: Path,
) -> dict[str, Any]:
    use_ds = is_downsampled_dir(windows_dir)

    cmd = [
        python_exe,
        str(dataset_script),
        str(fold_dir),
        str(windows_dir),
    ]
    if use_ds:
        cmd.append("--ds")

    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )
    duration_sec = time.perf_counter() - start

    result = {
        "fold": fold_dir.name,
        "fold_path": str(fold_dir),
        "windows_dir": windows_dir.name,
        "windows_path": str(windows_dir),
        "use_downsampled": use_ds,
        "command": cmd,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "duration_sec": duration_sec,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    return result


def short_status(result: dict[str, Any]) -> str:
    status = "PASS" if result["passed"] else "FAIL"
    ds_tag = "DS" if result["use_downsampled"] else "RAW"
    return (
        f"[{status}] "
        f"{result['fold']:>6} | "
        f"{result['windows_dir']:<32} | "
        f"{ds_tag:<3} | "
        f"{result['duration_sec']:.2f}s"
    )


def build_text_report(
    results: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
    args: argparse.Namespace,
) -> str:
    total = len(results)
    passed = sum(r["passed"] for r in results)
    failed = total - passed

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("DATASET SMOKE TEST MATRIX REPORT")
    lines.append("=" * 100)
    lines.append(f"Started at : {started_at}")
    lines.append(f"Finished at: {finished_at}")
    lines.append(f"Python     : {sys.executable}")
    lines.append(f"Script     : {args.dataset_script}")
    lines.append(f"Splits root: {args.splits_root}")
    lines.append(f"Windows root: {args.windows_root}")
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 100)
    lines.append(f"Total runs : {total}")
    lines.append(f"Passed     : {passed}")
    lines.append(f"Failed     : {failed}")
    lines.append("")

    lines.append("RESULTS OVERVIEW")
    lines.append("-" * 100)
    for r in results:
        lines.append(short_status(r))
    lines.append("")

    fail_results = [r for r in results if not r["passed"]]
    if fail_results:
        lines.append("FAILED RUNS")
        lines.append("-" * 100)
        for r in fail_results:
            lines.append(short_status(r))
        lines.append("")

    lines.append("DETAILED OUTPUT")
    lines.append("-" * 100)
    for idx, r in enumerate(results, start=1):
        lines.append(f"Run #{idx}")
        lines.append(f"Status       : {'PASS' if r['passed'] else 'FAIL'}")
        lines.append(f"Fold         : {r['fold']}")
        lines.append(f"Windows dir  : {r['windows_dir']}")
        lines.append(f"Downsampled  : {r['use_downsampled']}")
        lines.append(f"Duration     : {r['duration_sec']:.4f}s")
        lines.append(f"Return code  : {r['returncode']}")
        lines.append(f"Command      : {' '.join(r['command'])}")
        lines.append("")
        lines.append("[STDOUT]")
        lines.append(r["stdout"].rstrip() or "<empty>")
        lines.append("")
        lines.append("[STDERR]")
        lines.append(r["stderr"].rstrip() or "<empty>")
        lines.append("")
        lines.append("-" * 100)

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset smoke tests across all folds and window dirs.")
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path("data/emglab/splits"),
        help="Directory containing fold_* subdirectories.",
    )
    parser.add_argument(
        "--windows-root",
        type=Path,
        default=Path("data/emglab"),
        help="Directory containing windows_* subdirectories.",
    )
    parser.add_argument(
        "--dataset-script",
        type=Path,
        default=Path("nemg/dataset/dataset.py"),
        help="Path to the dataset smoke-test script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/dataset_checks"),
        help="Directory where reports will be written.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable to use. Default: current interpreter.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.splits_root.exists():
        print(f"ERROR: splits root does not exist: {args.splits_root}", file=sys.stderr)
        return 2

    if not args.windows_root.exists():
        print(f"ERROR: windows root does not exist: {args.windows_root}", file=sys.stderr)
        return 2

    if not args.dataset_script.exists():
        print(f"ERROR: dataset script does not exist: {args.dataset_script}", file=sys.stderr)
        return 2

    folds = discover_folds(args.splits_root)
    window_dirs = discover_window_dirs(args.windows_root)

    if not folds:
        print(f"ERROR: no fold_* directories found in {args.splits_root}", file=sys.stderr)
        return 2

    if not window_dirs:
        print(f"ERROR: no windows_* directories found in {args.windows_root}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    started_at_dt = datetime.now()
    started_at = started_at_dt.isoformat(timespec="seconds")
    timestamp = started_at_dt.strftime("%Y%m%d_%H%M%S")

    total_runs = len(folds) * len(window_dirs)
    print("=" * 100)
    print("DATASET SMOKE TEST MATRIX")
    print("=" * 100)
    print(f"Folds found      : {len(folds)}")
    for f in folds:
        print(f"  - {f.name}")
    print(f"Window dirs found: {len(window_dirs)}")
    for w in window_dirs:
        mode = "DS" if is_downsampled_dir(w) else "RAW"
        print(f"  - {w.name} ({mode})")
    print(f"Total runs       : {total_runs}")
    print("")

    results: list[dict[str, Any]] = []
    run_idx = 0

    for fold_dir in folds:
        for windows_dir in window_dirs:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] Running {fold_dir.name} x {windows_dir.name} ...")
            result = run_one(
                python_exe=args.python_exe,
                dataset_script=args.dataset_script,
                fold_dir=fold_dir,
                windows_dir=windows_dir,
            )
            results.append(result)
            print(short_status(result))

            if result["passed"]:
                stdout_lines = [line for line in result["stdout"].splitlines() if line.strip()]
                tail = stdout_lines[-6:] if stdout_lines else []
                if tail:
                    print("  stdout tail:")
                    for line in tail:
                        print(f"    {line}")
            else:
                print("  stdout:")
                for line in (result["stdout"].splitlines() or ["<empty>"]):
                    print(f"    {line}")
                print("  stderr:")
                for line in (result["stderr"].splitlines() or ["<empty>"]):
                    print(f"    {line}")

            print("")

    finished_at = datetime.now().isoformat(timespec="seconds")

    txt_report = build_text_report(results, started_at, finished_at, args)

    txt_path = args.output_dir / f"dataset_matrix_report_{timestamp}.txt"
    json_path = args.output_dir / f"dataset_matrix_report_{timestamp}.json"

    txt_path.write_text(txt_report, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "started_at": started_at,
                "finished_at": finished_at,
                "python": sys.executable,
                "dataset_script": str(args.dataset_script),
                "splits_root": str(args.splits_root),
                "windows_root": str(args.windows_root),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total = len(results)
    passed = sum(r["passed"] for r in results)
    failed = total - passed

    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"Total runs : {total}")
    print(f"Passed     : {passed}")
    print(f"Failed     : {failed}")
    print(f"Text report: {txt_path}")
    print(f"JSON report: {json_path}")

    if failed:
        print("")
        print("Failed combinations:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['fold']} x {r['windows_dir']}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())