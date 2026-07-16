"""
datasets/xdash/ingest.py
========================
XDash-specific raw → pickled ingestion.

Reads per-subject CSV files from storage/raw/xdash/ and writes
per-task pickled dicts to storage/pickled/xdash/.

Called by data/ingestion.py:  ingest(dataset="xdash", ...)
Can also be run as a standalone CLI:
    python -m datasets.xdash.ingest --raw-dir storage/raw/xdash --output-dir storage/pickled/xdash
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# CSV column layout
# ---------------------------------------------------------------------------

MOVEMENT_COLS = [
    "TimeElapsed",
    "HPosX", "HPosY", "HPosZ", "HRotX", "HRotY", "HRotZ",
    "LPosX", "LPosY", "LPosZ", "LRotX", "LRotY", "LRotZ",
    "RPosX", "RPosY", "RPosZ", "RRotX", "RRotY", "RRotZ",
]

# Rotation column indices in the 19-col array (col 0 = TimeElapsed)
ROT_COLS = [4, 5, 6, 10, 11, 12, 16, 17, 18]


# ---------------------------------------------------------------------------
# Master.csv → task time boundaries
# ---------------------------------------------------------------------------

def parse_master(master_path: Path, task_labels: dict) -> dict:
    """Return {task_num: (start, end)} from Master.csv."""
    df = pd.read_csv(master_path)
    df.columns = [c.strip() for c in df.columns]

    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if "time" in cl:
            col_map[c] = "timestamp"
        elif "name" in cl:
            col_map[c] = "name"
    df = df.rename(columns=col_map)

    if "timestamp" not in df.columns or "name" not in df.columns:
        raise ValueError(
            f"Master.csv must have Timestamp and Name columns. "
            f"Found: {list(df.columns)} in {master_path}"
        )

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["name"] = df["name"].astype(str).str.strip()

    task_nums = sorted(task_labels.keys())
    boundaries = {}

    for task_num in task_nums:
        task_label = task_labels[task_num]
        start_rows = df[df["name"] == task_label]
        if start_rows.empty:
            boundaries[task_num] = None
            continue
        start_time = float(start_rows.iloc[0]["timestamp"])

        # End = start of next task, or first row after last task
        next_tasks = [n for n in task_nums if n > task_num]
        if next_tasks:
            next_label = task_labels[next_tasks[0]]
            end_rows = df[df["name"] == next_label]
            if end_rows.empty:
                boundaries[task_num] = None
                continue
            end_time = float(end_rows.iloc[0]["timestamp"])
        else:
            task_idx = start_rows.index[0]
            rows_after = df[df.index > task_idx]
            end_time = float(rows_after.iloc[0]["timestamp"]) if not rows_after.empty else float("inf")

        boundaries[task_num] = (start_time, end_time)

    return boundaries


# ---------------------------------------------------------------------------
# PlayerMovement.csv slicing
# ---------------------------------------------------------------------------

def load_movement_slice(movement_path: Path, start_time: float, end_time: float):
    """Return (T, 19) float32 array, or None on failure."""
    if not movement_path.exists():
        return None

    df = pd.read_csv(movement_path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in MOVEMENT_COLS if c not in df.columns]
    if missing:
        print(f"      Warning: missing columns {missing} in {movement_path.name}")
        return None

    df = df[MOVEMENT_COLS]
    df["TimeElapsed"] = pd.to_numeric(df["TimeElapsed"], errors="coerce")
    df = df.dropna(subset=["TimeElapsed"])

    if end_time == float("inf"):
        mask = df["TimeElapsed"] >= start_time
    else:
        mask = (df["TimeElapsed"] >= start_time) & (df["TimeElapsed"] < end_time)

    sliced = df[mask].reset_index(drop=True)
    return sliced.values.astype(np.float32) if len(sliced) > 0 else None


# ---------------------------------------------------------------------------
# Rotation fix: unwrap + shift frames near 360° to near 0°
# ---------------------------------------------------------------------------

def fix_rotation_cols(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    for col in ROT_COLS:
        arr[:, col] = np.rad2deg(np.unwrap(np.deg2rad(arr[:, col])))
        if arr[0, col] > 180:
            arr[:, col] -= 360
    return arr


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

def process_subject(subject_dir: Path, task_num: int, task_labels: dict, sampling_rate: int):
    subject_id = subject_dir.name
    master_path = subject_dir / "Master.csv"
    movement_path = subject_dir / "PlayerMovement.csv"

    if not master_path.exists():
        print(f"    [{subject_id}] Warning: Master.csv not found")
        return subject_id, None

    try:
        boundaries = parse_master(master_path, task_labels)
    except Exception as e:
        print(f"    [{subject_id}] Warning: Master.csv error: {e}")
        return subject_id, None

    bounds = boundaries.get(task_num)
    if bounds is None:
        print(f"    [{subject_id}] Warning: task {task_num} not in Master.csv")
        return subject_id, None

    start_time, end_time = bounds
    arr = load_movement_slice(movement_path, start_time, end_time)

    if arr is None:
        print(f"    [{subject_id}] Warning: no frames in window [{start_time:.2f}s, {end_time:.2f}s)")
        return subject_id, None

    arr = fix_rotation_cols(arr)
    duration = end_time - start_time
    n_frames = len(arr)

    print(f"    [{subject_id}]  window=[{start_time:.2f}s, {end_time:.2f}s]  "
          f"task={duration:.1f}s  frames={n_frames}  actual={n_frames / sampling_rate:.1f}s")

    return subject_id, arr


# ---------------------------------------------------------------------------
# Main ingestion function (called by dataio/ingestion.py)
# ---------------------------------------------------------------------------

def ingest(config: dict, raw_dir: Path, out_dir: Path,
           tasks: list = None, dry_run: bool = False):
    """
    Generate per-task pickled datasets from raw XDash CSV files.

    Parameters
    ----------
    config : dict
        Loaded dataset.yaml contents.
    raw_dir : Path
        storage/raw/xdash/  — must contain px/ and fx/ subdirs (or PX*/fx* directly).
    out_dir : Path
        storage/pickled/xdash/  — output destination.
    tasks : list[int] | None
        Task numbers to process; None means all tasks in config.
    dry_run : bool
        If True, print summary without writing files.
    """
    sampling_rate = config.get("sampling_rate", 50)
    # Build task label map: {1: "Task 1", ...} using the task name as the CSV marker
    # XDash Master.csv uses "Task 1", "Task 2", etc. as markers
    task_ids = sorted(config["tasks"].keys())
    task_labels = {t: f"Task {t}" for t in task_ids}

    if tasks is None:
        tasks = task_ids

    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover subject directories
    patient_dirs = sorted(
        [p for p in (raw_dir / "px").glob("PX*") if p.is_dir()]
        or [p for p in raw_dir.glob("PX*") if p.is_dir()],
        key=lambda p: int("".join(filter(str.isdigit, p.name)) or 0),
    )
    control_dirs = sorted(
        [c for c in (raw_dir / "fx").glob("fx*") if c.is_dir()]
        or [c for c in raw_dir.glob("fx*") if c.is_dir()],
        key=lambda c: int("".join(filter(str.isdigit, c.name)) or 0),
    )

    print(f"\nPatient dirs : {len(patient_dirs)}")
    print(f"Control dirs : {len(control_dirs)}")
    print(f"Output dir   : {out_dir}\n")

    for task_num in tasks:
        print(f"\n{'='*65}")
        print(f"Task {task_num}  ({config['tasks'][task_num]})")
        print(f"{'='*65}")

        patient_data, control_data = {}, {}

        print("\n  Patients:")
        for d in patient_dirs:
            sid, arr = process_subject(d, task_num, task_labels, sampling_rate)
            if arr is not None:
                patient_data[sid] = arr

        print("\n  Controls:")
        for d in control_dirs:
            sid, arr = process_subject(d, task_num, task_labels, sampling_rate)
            if arr is not None:
                control_data[sid] = arr

        print(f"\n  Summary: patients={len(patient_data)}/{len(patient_dirs)}  "
              f"controls={len(control_data)}/{len(control_dirs)}")

        if dry_run:
            print("  → Dry run: no files written")
            continue

        with open(out_dir / f"patient_data_task{task_num}.pkl", "wb") as f:
            pickle.dump(patient_data, f)
        with open(out_dir / f"control_data_task{task_num}.pkl", "wb") as f:
            pickle.dump(control_data, f)
        print(f"  → Saved task {task_num} pickled datasets")

    print(f"\n{'='*65}")
    print("Done." if not dry_run else "Dry run complete.")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(description="Create xdash pickled datasets from raw CSV files")
    parser.add_argument("--raw-dir", default="storage/raw/xdash",
                        help="Root dir containing px/ and fx/ subject folders")
    parser.add_argument("--output-dir", default="storage/pickled/xdash",
                        help="Output dir for pkl files")
    parser.add_argument("--tasks", type=int, nargs="+", default=None,
                        help="Task numbers to process (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print info without writing files")
    args = parser.parse_args()

    config_path = _Path(__file__).parent / "dataset.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ingest(
        config=cfg,
        raw_dir=_Path(args.raw_dir),
        out_dir=_Path(args.output_dir),
        tasks=args.tasks,
        dry_run=args.dry_run,
    )
