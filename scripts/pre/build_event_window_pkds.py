"""
build_task_datasets.py
======================
Builds per-task, per-group event-window pickle files from unified_dataset_raw.pkl.

Output files (task × group  =  6 × 2 = 12 files):
    data/pickled_datasets/event_window/g0_data_task1.pkl   ← control (healthy)
    data/pickled_datasets/event_window/g1_data_task1.pkl   ← patient (injured)
    ... (tasks 1-6)

Groups:
    g0 → is_control == True  (healthy / fx subjects)
    g1 → is_control == False (patient / px subjects)

Each file is a dict keyed by window_id:
    {
        'fx07_task4_trial1': np.ndarray (T, 18),
        'fx07_task4_trial2': np.ndarray (T, 18),
        ...
    }

Cross-validation (LOSO) is applied on these files at training time.
"""

import pickle
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR   = Path("data")
SOURCE_PKL = DATA_DIR / "pickled_datasets" / "unified_dataset_raw.pkl"
OUT_DIR    = DATA_DIR / "pickled_datasets" / "event_window"

TASK_NAMES = {
    # 1: "jar_lid_opening",
    # 2: "key_turning",
    # 3: "wall_cleaning",
    4: "back_washing",
    # 5: "knife_slicing",
    # 6: "hammer_nailing",
}
GROUPS = ["g0", "g1"]   # g0 = control, g1 = patient


# ── Path helper (importable) ──────────────────────────────────────────────────

def get_pickled_dataset_path(task: int, group: str) -> Path:
    """
    Get path to pickled dataset.

    Args:
        task:  task number (1-6)
        group: 'g0' (control) or 'g1' (patient)
    """
    return OUT_DIR / f"{group}_data_task{task}.pkl"


# ── Main ──────────────────────────────────────────────────────────────────────

def build(source_pkl: Path = SOURCE_PKL):
    print(f"Loading {source_pkl} ...")
    with open(source_pkl, 'rb') as f:
        all_trials = pickle.load(f)
    print(f"  Total trials: {len(all_trials)}")

    frames = [t['movement_data'].shape[0] for t in all_trials if 'movement_data' in t]
    frames = np.array(frames)
    p = [10, 25, 50, 75, 90]
    percentiles = np.percentile(frames, p).astype(int)
    print(f"  Frame distribution:")
    print(f"    min={frames.min()}  max={frames.max()}  mean={frames.mean():.0f}  std={frames.std():.0f}")
    print(f"    p10={percentiles[0]}  p25={percentiles[1]}  p50={percentiles[2]}"
          f"  p75={percentiles[3]}  p90={percentiles[4]}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group by (task_number, group)
    buckets: dict[tuple, dict] = {}
    skipped = 0

    for trial in all_trials:
        if 'movement_data' not in trial:
            skipped += 1
            continue
        task = trial.get('task_number')
        wid  = trial.get('window_id')
        if task is None or wid is None:
            skipped += 1
            continue

        group = 'g0' if trial.get('is_control') else 'g1'
        buckets.setdefault((task, group), {})[wid] = trial['movement_data']

    if skipped:
        print(f"  Skipped {skipped} trials (missing movement_data / task / window_id)\n")

    # Save
    total_written = 0
    for task in sorted(TASK_NAMES):
        print(f"Task {task}  ({TASK_NAMES[task]})")
        for group in GROUPS:
            windows  = buckets.get((task, group), {})
            out_path = get_pickled_dataset_path(task, group)
            with open(out_path, 'wb') as f:
                pickle.dump(windows, f)
            total_written += len(windows)
            print(f"  [{group}]  {len(windows):4d} windows  →  {out_path.name}")
        print()

    print(f"Done. {total_written} windows across "
          f"{len(TASK_NAMES) * len(GROUPS)} files.")


if __name__ == "__main__":
    build()
