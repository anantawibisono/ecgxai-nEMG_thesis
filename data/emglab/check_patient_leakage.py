from pathlib import Path
import pandas as pd
import re

ROOT = Path("splits")

# Adjust this regex only if your filename format changes
PATIENT_RE = re.compile(r"^([A-Za-z]\d{4}[A-Za-z]\d{2})")

def extract_patient_id(x: str) -> str:
    stem = Path(str(x)).stem  # removes .npz if present
    m = PATIENT_RE.match(stem)
    if not m:
        raise ValueError(f"Could not extract patient_id from: {x}")
    return m.group(1)

def load_split(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Prefer filename_stem if present, otherwise raw_npz
    if "filename_stem" in df.columns:
        source_col = "filename_stem"
    elif "raw_npz" in df.columns:
        source_col = "raw_npz"
    else:
        raise ValueError(f"No filename column found in {csv_path}")

    df = df.copy()
    df["patient_id"] = df[source_col].map(extract_patient_id)
    return df

def check_same_patient_not_split(df: pd.DataFrame, split_name: str):
    # If a patient appears with multiple recordings/windows, they should still map to one patient_id
    counts = df.groupby("patient_id").size().sort_values(ascending=False)
    print(f"{split_name}: {len(counts)} unique patients, {len(df)} rows")
    print(counts.head(), "\n")

# Load held-out test set
test_csv = ROOT / "test.csv"
test_df = load_split(test_csv)
test_patients = set(test_df["patient_id"])

print(f"TEST: {len(test_patients)} unique patients\n")

fold_patient_sets = {}

for fold_dir in sorted(ROOT.glob("fold_*")):
    train_csv = fold_dir / "train.csv"
    val_csv = fold_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        continue

    train_df = load_split(train_csv)
    val_df = load_split(val_csv)

    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])

    # 1) Core leakage check inside this fold
    overlap = train_patients & val_patients

    print(f"=== {fold_dir.name} ===")
    print(f"train patients: {len(train_patients)}")
    print(f"val patients:   {len(val_patients)}")
    print(f"train∩val:      {len(overlap)}")

    if overlap:
        print("LEAKED patients between train and val:")
        print(sorted(overlap))
    else:
        print("OK: no patient leakage between train and val")

    # 2) Test set must not overlap with train or val
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients

    print(f"train∩test:     {len(train_test_overlap)}")
    print(f"val∩test:       {len(val_test_overlap)}")

    if train_test_overlap:
        print("LEAKED patients between train and test:")
        print(sorted(train_test_overlap))

    if val_test_overlap:
        print("LEAKED patients between val and test:")
        print(sorted(val_test_overlap))

    # 3) Optional sanity prints
    check_same_patient_not_split(train_df, f"{fold_dir.name} train")
    check_same_patient_not_split(val_df, f"{fold_dir.name} val")

    fold_patient_sets[fold_dir.name] = {
        "train": train_patients,
        "val": val_patients,
    }

# Optional: check that each patient appears in validation exactly once across folds
all_val_counts = {}
for fold_name, sets in fold_patient_sets.items():
    for pid in sets["val"]:
        all_val_counts[pid] = all_val_counts.get(pid, 0) + 1

bad_val_counts = {pid: c for pid, c in all_val_counts.items() if c != 1}

print("=== Cross-fold validation coverage check ===")
if bad_val_counts:
    print("Patients not appearing exactly once in validation across folds:")
    print(bad_val_counts)
else:
    print("OK: every patient appears exactly once in validation across folds")