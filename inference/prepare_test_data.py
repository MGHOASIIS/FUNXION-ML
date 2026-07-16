"""
prepare_test_data.py
====================
Convert a single new patient's raw CSV files into per-task pickle files
that are compatible with inference.py.

Mirrors the pipeline in scripts/pre/generate_pickled_datasets.py, but for
one subject only.

Input directory layout (same as existing patient/control data)
--------------------------------------------------------------
    <data-dir>/
    ├── Master.csv          ← task timestamps
    └── PlayerMovement.csv  ← continuous motion capture recording

Output
------
    <output-dir>/test_patient_task1.pkl   → {"<subject-id>": np.ndarray (T, 19)}
    <output-dir>/test_patient_task2.pkl
    ...

The pkl format is identical to patient_data_task{N}.pkl so it can be passed
directly to inference.py via --test-data.

Usage
-----
    # Extract all 6 tasks from data/test_data/ and save to data/test_data/pickled/
    python prepare_test_data.py --data-dir data/test_data/ --subject-id PX_NEW

    # Extract only tasks 1 and 3
    python prepare_test_data.py --data-dir data/test_data/ --subject-id PX_NEW --tasks 1 3

    # Dry run to check what would be extracted (no files written)
    python prepare_test_data.py --data-dir data/test_data/ --subject-id PX_NEW --dry-run

    # Specify a custom output directory
    python prepare_test_data.py --data-dir data/test_data/ --subject-id PX_NEW \\
        --output-dir my_output/
"""

import argparse
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Allow running as `python inference/prepare_test_data.py` from the project
# root — the script's own directory (inference/) is on sys.path by default,
# not the project root, so top-level packages wouldn't otherwise be importable.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.ingestion import load_dataset_config


# ---------------------------------------------------------------------------
# Constants (kept in sync with generate_pickled_datasets.py)
#
# TASK_NAMES here are the literal row labels used in Master.csv's "name"
# column — a raw-ingestion-format detail, not the same as the human-readable
# task names (e.g. "jar_opening") in datasets/{dataset}/dataset.yaml.
# MOVEMENT_COLS / ROT_COLS describe XDash's specific 3-sensor CSV schema and
# are similarly a raw-format detail rather than dataset-config-driven.
# ---------------------------------------------------------------------------

TASK_NAMES = {
    1: "Task 1",
    2: "Task 2",
    3: "Task 3",
    4: "Task 4",
    5: "Task 5",
    6: "Task 6",
}

MOVEMENT_COLS = [
    "TimeElapsed",
    "HPosX", "HPosY", "HPosZ", "HRotX", "HRotY", "HRotZ",
    "LPosX", "LPosY", "LPosZ", "LRotX", "LRotY", "LRotZ",
    "RPosX", "RPosY", "RPosZ", "RRotX", "RRotY", "RRotZ",
]

N_FEATURES = 18   # excludes TimeElapsed

# Rotation column indices in the (T, 19) array (col 0 = TimeElapsed)
ROT_COLS = [4, 5, 6,    # HRotX, HRotY, HRotZ
            10, 11, 12, # LRotX, LRotY, LRotZ
            16, 17, 18] # RRotX, RRotY, RRotZ


# ---------------------------------------------------------------------------
# Master.csv parsing
# ---------------------------------------------------------------------------

def parse_master(master_path: Path) -> dict:
    """
    Parse Master.csv and return task time boundaries.

    Returns
    -------
    dict : {task_num: (start_time, end_time)}
           end_time is float('inf') for Task 6 when no row follows it.
           value is None when the task row is missing.
    """
    df = pd.read_csv(master_path)
    df.columns = [c.strip() for c in df.columns]

    # Drop unnamed columns created by trailing commas in data rows
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

    # Map to canonical names — use exact lowercase match to avoid
    # "name" matching "Unnamed: 2" as a substring
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "timestamp":
            col_map[c] = "timestamp"
        elif cl == "name":
            col_map[c] = "name"
    df = df.rename(columns=col_map)

    if "timestamp" not in df.columns or "name" not in df.columns:
        raise ValueError(
            f"Master.csv must have Timestamp and Name columns. "
            f"Found: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["name"]      = df["name"].astype(str).str.strip()

    boundaries = {}
    for task_num, task_label in TASK_NAMES.items():
        start_rows = df[df["name"] == task_label]
        if start_rows.empty:
            boundaries[task_num] = None
            continue

        start_time = float(start_rows.iloc[0]["timestamp"])

        if task_num < 6:
            next_label = TASK_NAMES[task_num + 1]
            end_rows = df[df["name"] == next_label]
            if end_rows.empty:
                boundaries[task_num] = None
                continue
            end_time = float(end_rows.iloc[0]["timestamp"])
        else:
            # Task 6 ends at the next row in Master.csv (whatever it is)
            task6_idx   = start_rows.index[0]
            rows_after  = df[df.index > task6_idx]
            end_time    = float(rows_after.iloc[0]["timestamp"]) \
                          if not rows_after.empty else float("inf")

        boundaries[task_num] = (start_time, end_time)

    return boundaries


# ---------------------------------------------------------------------------
# PlayerMovement.csv slicing
# ---------------------------------------------------------------------------

def load_movement_slice(movement_path: Path,
                        start_time: float,
                        end_time: float) -> np.ndarray | None:
    """
    Load PlayerMovement.csv and return rows within [start_time, end_time).

    Returns
    -------
    np.ndarray (T, 19) float32, or None on failure.
    """
    if not movement_path.exists():
        print(f"  ✗  PlayerMovement.csv not found at {movement_path}")
        return None

    df = pd.read_csv(movement_path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in MOVEMENT_COLS if c not in df.columns]
    if missing:
        print(f"  ✗  Missing columns in PlayerMovement.csv: {missing}")
        print(f"     Found: {list(df.columns)}")
        return None

    df = df[MOVEMENT_COLS]
    df["TimeElapsed"] = pd.to_numeric(df["TimeElapsed"], errors="coerce")
    df = df.dropna(subset=["TimeElapsed"])

    if end_time == float("inf"):
        mask = df["TimeElapsed"] >= start_time
    else:
        mask = (df["TimeElapsed"] >= start_time) & (df["TimeElapsed"] < end_time)

    sliced = df[mask].reset_index(drop=True)

    if len(sliced) == 0:
        print(f"  ✗  No frames in window [{start_time:.2f}s, "
              f"{'∞' if end_time == float('inf') else f'{end_time:.2f}s'})")
        return None

    return sliced.values.astype(np.float32)


# ---------------------------------------------------------------------------
# Rotation fix (identical to generate_pickled_datasets.py)
# ---------------------------------------------------------------------------

def fix_rotation_cols(arr: np.ndarray) -> np.ndarray:
    """
    Fix rotation discontinuities in a (T, 19) array.

    1. np.unwrap — resolves 0/360 boundary jumps so the signal is continuous.
    2. Shift  — if the first frame > 180°, subtract 360 so it sits near 0°.
    """
    arr = arr.copy()
    for col in ROT_COLS:
        arr[:, col] = np.rad2deg(np.unwrap(np.deg2rad(arr[:, col])))
        if arr[0, col] > 180:
            arr[:, col] -= 360
    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a single patient's raw XDash CSVs to per-task pickle files."
    )
    parser.add_argument(
        "--dataset", default="xdash",
        help="Dataset name (must match datasets/ folder). Default: xdash"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing Master.csv and PlayerMovement.csv "
             "(e.g. data/test_data/)"
    )
    parser.add_argument(
        "--subject-id", required=True,
        help="Subject ID to use as the dict key in the pkl file (e.g. PX_NEW)"
    )
    parser.add_argument(
        "--tasks", type=int, nargs="+", default=list(TASK_NAMES.keys()),
        help="Task numbers to extract (default: 1 2 3 4 5 6)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for pkl files "
             "(default: <data-dir>/pickled/)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print extraction info without writing any files"
    )
    args = parser.parse_args()

    sampling_rate = load_dataset_config(args.dataset).get("sampling_rate", 50)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir \
                 else data_dir / "pickled"
    master_path   = data_dir / "Master.csv"
    movement_path = data_dir / "PlayerMovement.csv"

    # Validate input files
    if not master_path.exists():
        raise FileNotFoundError(f"Master.csv not found in {data_dir}")
    if not movement_path.exists():
        raise FileNotFoundError(f"PlayerMovement.csv not found in {data_dir}")

    print(f"\n{'='*60}")
    print(f"Subject     : {args.subject_id}")
    print(f"Data dir    : {data_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Tasks       : {args.tasks}")
    print(f"Dry run     : {args.dry_run}")
    print(f"{'='*60}\n")

    # Parse Master.csv once
    print("[1/2] Parsing Master.csv...")
    boundaries = parse_master(master_path)

    for task_num, bounds in boundaries.items():
        if bounds:
            s, e = bounds
            label = f"{s:.2f}s → {'∞' if e == float('inf') else f'{e:.2f}s'}"
        else:
            label = "NOT FOUND"
        print(f"  Task {task_num}: {label}")

    # Extract and save each requested task
    print(f"\n[2/2] Extracting tasks...")
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for task_num in args.tasks:
        print(f"\n  Task {task_num} ({TASK_NAMES[task_num]})")

        bounds = boundaries.get(task_num)
        if bounds is None:
            print(f"    ✗  Skipped — Task {task_num} not found in Master.csv")
            continue

        start_time, end_time = bounds
        arr = load_movement_slice(movement_path, start_time, end_time)

        if arr is None:
            print(f"    ✗  Skipped — no data extracted")
            continue

        arr = fix_rotation_cols(arr)

        n_frames   = len(arr)
        duration_s = n_frames / sampling_rate
        print(f"    ✓  {n_frames} frames  ({duration_s:.1f}s @ {sampling_rate}Hz)  "
              f"shape={arr.shape}")

        if args.dry_run:
            print(f"    → Dry run: would save test_patient_task{task_num}.pkl")
            continue

        out_path = output_dir / f"test_patient_task{task_num}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({args.subject_id: arr}, f)
        print(f"    → Saved: {out_path}")
        saved.append(out_path)

    # Summary
    print(f"\n{'='*60}")
    if args.dry_run:
        print("Dry run complete — no files written.")
    else:
        print(f"Done. {len(saved)} file(s) written to {output_dir}/")
        for p in saved:
            print(f"  {p.name}")
        if saved:
            print(f"\nNext step — run inference on any of these files:")
            example = saved[0]
            task_n  = example.stem.split("task")[-1]
            print(f"  python inference.py \\")
            print(f"      --checkpoint experiments/task{task_n}/paradigm1/.../best_model_BA*.pt \\")
            print(f"      --task {task_n} --paradigm 1 --model cnn --method truncate \\")
            print(f"      --test-data {example}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
