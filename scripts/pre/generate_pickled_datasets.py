"""
create_pickled_datasets.py
==========================
Creates per-task pickled datasets from raw PlayerMovement.csv files,
using Master.csv to extract the correct time window for each task.

Pipeline
--------
1. Load Master.csv per subject → extract task start/end timestamps
2. Load PlayerMovement.csv per subject → slice rows within [task_start, task_end)
3. Store raw numpy arrays (NO normalization, NO unwrap — done in separate script)
4. Save as:
       pickled_datasets/patient_data_task{1-6}.pkl
       pickled_datasets/control_data_task{1-6}.pkl

Output format
-------------
Each pkl file is a dict:
    {
        "PX01": np.ndarray (T, 19),   # 19 = TimeElapsed + 18 features
        "PX02": np.ndarray (T, 19),
        ...
    }

Master.csv schema (per subject, relative timestamps from session start)
------------------------------------------------------------------------
    Timestamp, Name
    158.21,    Task 1
    291.81,    Task 2
    ...
    542.61,    Chapter 3   ← marks end of Task 6

PlayerMovement.csv column layout (19 columns total)
-----------------------------------------------------
    Col  0 : TimeElapsed          ← relative seconds from session start
    Col  1 : HPosX                ← head position x
    Col  2 : HPosY
    Col  3 : HPosZ
    Col  4 : HRotX                ← head rotation x   (0-360 deg)
    Col  5 : HRotY
    Col  6 : HRotZ
    Col  7 : LPosX                ← left hand position x
    Col  8 : LPosY
    Col  9 : LPosZ
    Col 10 : LRotX                ← left hand rotation x  (0-360 deg)
    Col 11 : LRotY
    Col 12 : LRotZ
    Col 13 : RPosX                ← right hand position x
    Col 14 : RPosY
    Col 15 : RPosZ
    Col 16 : RRotX                ← right hand rotation x (0-360 deg)
    Col 17 : RRotY
    Col 18 : RRotZ

Rotation column indices in the 18-feature signal (TimeElapsed stripped):
    Head rotation  : 3, 4, 5
    Left rotation  : 9, 10, 11
    Right rotation : 15, 16, 17

Usage
-----
    python create_pickled_datasets.py --base-dir /path/to/data
    python create_pickled_datasets.py --base-dir /path/to/data --dry-run
    python create_pickled_datasets.py --base-dir /path/to/data --tasks 1 2
    python create_pickled_datasets.py --base-dir /path/to/data --output-dir /path/to/output
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

TASK_NAMES = {
    1: "Task 1",
    2: "Task 2",
    3: "Task 3",
    4: "Task 4",
    5: "Task 5",
    6: "Task 6",
}

# Expected PlayerMovement.csv column names
MOVEMENT_COLS = [
    "TimeElapsed",
    "HPosX", "HPosY", "HPosZ", "HRotX", "HRotY", "HRotZ",
    "LPosX", "LPosY", "LPosZ", "LRotX", "LRotY", "LRotZ",
    "RPosX", "RPosY", "RPosZ", "RRotX", "RRotY", "RRotZ",
]

SAMPLING_RATE = 50   # Hz
N_FEATURES    = 18   # excludes TimeElapsed
TASKS         = [1, 2, 3, 4, 5, 6]

# Rotation column indices in the full 19-col array (TimeElapsed in col 0)
# PlayerMovement.csv order: Head, Left, Right
# Layout per sensor: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
ROT_COLS = [4,  5,  6,   # HRotX/Y/Z  — head rotation
            10, 11, 12,  # LRotX/Y/Z  — left hand rotation
            16, 17, 18]  # RRotX/Y/Z  — right hand rotation


# =============================================================================
# Master.csv parsing
# =============================================================================

def parse_master(master_path: Path) -> dict:
    """
    Parse Master.csv and return task time boundaries.

    Returns
    -------
    dict : {task_num: (start_time, end_time)} or {task_num: None} if missing
    """
    df = pd.read_csv(master_path)
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names flexibly
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if 'time' in cl:
            col_map[c] = 'timestamp'
        elif 'name' in cl:
            col_map[c] = 'name'
    df = df.rename(columns=col_map)

    if 'timestamp' not in df.columns or 'name' not in df.columns:
        raise ValueError(
            f"Master.csv must have Timestamp and Name columns. "
            f"Found: {list(df.columns)} in {master_path}"
        )

    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['name']      = df['name'].astype(str).str.strip()

    boundaries = {}

    for task_num, task_label in TASK_NAMES.items():
        start_rows = df[df['name'] == task_label]
        if start_rows.empty:
            boundaries[task_num] = None
            continue
        start_time = float(start_rows.iloc[0]['timestamp'])

        if task_num < 6:
            next_label = TASK_NAMES[task_num + 1]
            end_rows = df[df['name'] == next_label]
            if end_rows.empty:
                boundaries[task_num] = None
                continue
            end_time = float(end_rows.iloc[0]['timestamp'])
        else:
            # End of Task 6 = whatever row comes immediately AFTER Task 6
            # in Master.csv, regardless of its name (Chapter 3, Chapter 4, etc.)
            task6_idx = start_rows.index[0]
            rows_after = df[df.index > task6_idx]
            if not rows_after.empty:
                end_time = float(rows_after.iloc[0]['timestamp'])
            else:
                # Task 6 is the last row — use end of PlayerMovement recording
                end_time = float('inf')

        boundaries[task_num] = (start_time, end_time)

    return boundaries


# =============================================================================
# PlayerMovement.csv slicing
# =============================================================================

def load_movement_slice(movement_path: Path,
                        start_time: float,
                        end_time: float) -> np.ndarray:
    """
    Load PlayerMovement.csv and slice rows within [start_time, end_time).

    Returns
    -------
    np.ndarray shape (T, 19) — TimeElapsed + 18 features, float32
    None if file missing, unreadable, or no rows in window
    """
    if not movement_path.exists():
        return None

    df = pd.read_csv(movement_path)
    df.columns = [c.strip() for c in df.columns]

    # Validate columns
    missing = [c for c in MOVEMENT_COLS if c not in df.columns]
    if missing:
        print(f"      ⚠️  Missing columns {missing} in {movement_path.name}")
        print(f"         Found: {list(df.columns)}")
        return None

    # Keep only the expected columns in the correct order
    df = df[MOVEMENT_COLS]
    df['TimeElapsed'] = pd.to_numeric(df['TimeElapsed'], errors='coerce')
    df = df.dropna(subset=['TimeElapsed'])

    # Slice task window — inclusive start, exclusive end
    # end_time may be inf for Task 6 when no row follows it in Master.csv
    if end_time == float('inf'):
        mask = df['TimeElapsed'] >= start_time
    else:
        mask = (df['TimeElapsed'] >= start_time) & (df['TimeElapsed'] < end_time)
    sliced = df[mask].reset_index(drop=True)

    if len(sliced) == 0:
        return None

    return sliced.values.astype(np.float32)   # (T, 19)


# =============================================================================
# Rotation fix
# =============================================================================

def fix_rotation_cols(arr: np.ndarray) -> np.ndarray:
    """
    Fix rotation columns in a (T, 19) array:

    Step 1 — np.unwrap
        Resolves 0/360 boundary discontinuities so the signal is continuous.
        e.g. 358, 359, 0, 1 → 358, 359, 360, 361

    Step 2 — shift if first frame > 180°
        If the first frame is near 360° (> 180°), subtract 360 so it sits
        near 0° instead. Preserves true angular value relative to 0.
        e.g. first=358 → 358-360=-2   (2° below zero)
             first=5   → no change    (5° above zero)

    Parameters
    ----------
    arr : np.ndarray, shape (T, 19)  — TimeElapsed + 18 features

    Returns
    -------
    np.ndarray, same shape
    """
    arr = arr.copy()

    for col in ROT_COLS:
        # Step 1: unwrap to fix 0/360 boundary crossings
        arr[:, col] = np.rad2deg(np.unwrap(np.deg2rad(arr[:, col])))

        # Step 2: if first frame is near 360, shift entire column down by 360
        if arr[0, col] > 180:
            arr[:, col] -= 360

    return arr


# =============================================================================
# Per-subject processing
# =============================================================================

def process_subject(subject_dir: Path, task_num: int) -> tuple:
    """
    Process one subject for one task.

    Returns
    -------
    (subject_id, arr)  — arr is np.ndarray (T, 19) or None on failure
    """
    subject_id    = subject_dir.name
    master_path   = subject_dir / "Master.csv"
    movement_path = subject_dir / "PlayerMovement.csv"

    # Parse Master.csv
    if not master_path.exists():
        print(f"    [{subject_id}] ⚠️  Master.csv not found")
        return subject_id, None

    try:
        boundaries = parse_master(master_path)
    except Exception as e:
        print(f"    [{subject_id}] ⚠️  Master.csv error: {e}")
        return subject_id, None

    bounds = boundaries.get(task_num)
    if bounds is None:
        print(f"    [{subject_id}] ⚠️  Task {task_num} not in Master.csv")
        return subject_id, None

    start_time, end_time = bounds
    duration = end_time - start_time

    # Slice PlayerMovement.csv
    arr = load_movement_slice(movement_path, start_time, end_time)

    if arr is None or len(arr) == 0:
        print(f"    [{subject_id}] ⚠️  No frames in window "
              f"[{start_time:.2f}s, {end_time:.2f}s)")
        return subject_id, None

    # Fix rotation columns: unwrap + shift first frame near 360 to near 0
    arr = fix_rotation_cols(arr)

    n_frames = len(arr)
    actual_duration = n_frames / SAMPLING_RATE

    print(f"    [{subject_id}]  "
          f"window=[{start_time:.2f}s, {end_time:.2f}s]  "
          f"task_duration={duration:.1f}s  "
          f"frames={n_frames}  "
          f"actual={actual_duration:.1f}s")

    return subject_id, arr


# =============================================================================
# Main runner
# =============================================================================

def create_datasets(base_dir: Path,
                    output_dir: Path,
                    tasks: list,
                    dry_run: bool):

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover subject directories
    patient_dirs = sorted(
        [p for p in (base_dir / "px").glob("PX*") if p.is_dir()]
        or [p for p in base_dir.glob("PX*") if p.is_dir()],
        key=lambda p: int(''.join(filter(str.isdigit, p.name)) or 0)
    )
    control_dirs = sorted(
        [c for c in (base_dir / "fx").glob("fx*") if c.is_dir()]
        or [c for c in base_dir.glob("fx*") if c.is_dir()],
        key=lambda c: int(''.join(filter(str.isdigit, c.name)) or 0)
    )

    print(f"\nPatient dirs : {len(patient_dirs)}")
    print(f"Control dirs : {len(control_dirs)}")
    print(f"Output dir   : {output_dir}\n")

    for task_num in tasks:
        print(f"\n{'='*65}")
        print(f"Task {task_num}  ({TASK_NAMES[task_num]})")
        print(f"{'='*65}")

        patient_data = {}
        control_data = {}

        print(f"\n  Patients:")
        for subject_dir in patient_dirs:
            sid, arr = process_subject(subject_dir, task_num)
            if arr is not None:
                patient_data[sid] = arr

        print(f"\n  Controls:")
        for subject_dir in control_dirs:
            sid, arr = process_subject(subject_dir, task_num)
            if arr is not None:
                control_data[sid] = arr

        # Summary
        print(f"\n  Summary:")
        print(f"    Patients : {len(patient_data)} / {len(patient_dirs)} loaded")
        print(f"    Controls : {len(control_data)} / {len(control_dirs)} loaded")

        if patient_data:
            frames = [v.shape[0] for v in patient_data.values()]
            print(f"    Patient frames — min={min(frames)}  max={max(frames)}  "
                  f"mean={int(np.mean(frames))}  "
                  f"({min(frames)/SAMPLING_RATE:.0f}s – {max(frames)/SAMPLING_RATE:.0f}s)")
        if control_data:
            frames = [v.shape[0] for v in control_data.values()]
            print(f"    Control frames — min={min(frames)}  max={max(frames)}  "
                  f"mean={int(np.mean(frames))}  "
                  f"({min(frames)/SAMPLING_RATE:.0f}s – {max(frames)/SAMPLING_RATE:.0f}s)")

        if dry_run:
            print(f"  → Dry run: no files written")
            continue

        patient_pkl = output_dir / f"patient_data_task{task_num}.pkl"
        control_pkl = output_dir / f"control_data_task{task_num}.pkl"

        with open(patient_pkl, "wb") as f:
            pickle.dump(patient_data, f)
        print(f"  → Saved: {patient_pkl}")

        with open(control_pkl, "wb") as f:
            pickle.dump(control_data, f)
        print(f"  → Saved: {control_pkl}")

    print(f"\n{'='*65}")
    print("Done." if not dry_run else "Dry run complete.")
    print(f"{'='*65}\n")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create per-task pickled datasets from raw XDash CSV files"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default='data/',
        help="Root directory containing px/ and fx/ subject folders "
             "(or PX*/fx* directly)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for pkl files "
             "(default: <base-dir>/pickled_datasets)"
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=TASKS,
        help="Task numbers to process (default: 1 2 3 4 5 6)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print extraction info without writing any files"
    )
    args = parser.parse_args()

    base_dir   = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir \
                 else base_dir / "pickled_datasets"

    create_datasets(
        base_dir=base_dir,
        output_dir=output_dir,
        tasks=args.tasks,
        dry_run=args.dry_run,
    )