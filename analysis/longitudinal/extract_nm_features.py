"""
extract_nm_features.py
======================
Extract features from longitudinal sessions 1-4 (test_sub_NM/).

All 8 trackers are used:
    Head, LeftHand, RightHand, Back,
    LeftHip, RightHip, LeftKnee, RightKnee

For each tracker the following feature groups are extracted:
    - Time-domain stats  (per channel: 19 stats × 6 channels)
    - Frequency features (on position magnitude)
    - Movement quality   (jerk, SPARC on velocity magnitude)
    - Wavelet features   (on position magnitude)
    - Complexity/entropy (on position magnitude)
    - Joint kinematics   (ROM, mean, min, max per rotation axis)

Bilateral asymmetry is computed for Left/Right pairs:
    hand (LeftHand vs RightHand), hip (LeftHip vs RightHip),
    knee (LeftKnee vs RightKnee)

Session metadata (dominant side, injured side, eye height)
is pulled from Events.csv.

Usage:
    python scripts/extract_nm_features.py
    python scripts/extract_nm_features.py --sessions 1 2 3 4 --out results/nm_features.csv
"""

import sys
import argparse
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Resolve import paths
# ------------------------------------------------------------------
HERE      = Path(__file__).resolve().parent          # analysis/longitudinal/
REPO_ROOT = HERE.parent.parent                       # repo root
DATA_DIR  = HERE / "test_sub_NM"                     # sessions 1-4
FEATURES_DIR = REPO_ROOT / "features"
sys.path.insert(0, str(FEATURES_DIR))

from time_domain_stats import extract_time_domain_features          # noqa: E402
from frequency_domain import extract_frequency_features             # noqa: E402
from movement_quality import calculate_smoothness_jerk, calculate_sparc  # noqa: E402
from wavelet_features import extract_wavelet_features               # noqa: E402
from complexity_entropy import extract_complexity_features          # noqa: E402
from asymmetry_compensation_metrics import calculate_asymmetry      # noqa: E402

# ------------------------------------------------------------------
# Sensor layout  (column indices in UserMovement.csv after 3-row header)
# Layout: Timestamp | T0(Head) 6cols | T1(LeftHand) 6cols | ... | T7(RightKnee) 6cols
# ------------------------------------------------------------------
TIMESTAMP_COL = 0
SAMPLING_RATE = 50.0

SENSORS = [
    "Head",
    "LeftHand",
    "RightHand",
    "Back",
    "LeftHip",
    "RightHip",
    "LeftKnee",
    "RightKnee",
]

# Each sensor occupies 6 consecutive columns starting at 1 + sensor_index*6
SENSOR_COL_START = {name: 1 + i * 6 for i, name in enumerate(SENSORS)}

# Bilateral pairs for asymmetry
BILATERAL_PAIRS = [
    ("hand",  "LeftHand",  "RightHand"),
    ("hip",   "LeftHip",   "RightHip"),
    ("knee",  "LeftKnee",  "RightKnee"),
]

CHANNEL_NAMES = ["PosX", "PosY", "PosZ", "RotX", "RotY", "RotZ"]


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_user_movement(path):
    """
    Load UserMovement.csv (UTF-16-LE, 3-row header).

    Returns
    -------
    timestamps : (T,) float array
    sensor_data : dict  sensor_name -> (T, 6) float array
    """
    with open(path, encoding="utf-16-le") as f:
        lines = f.readlines()

    # Skip the 3 header rows
    raw = pd.read_csv(
        StringIO("".join(lines[3:])),
        header=None,
        dtype=float,
        na_values=["", " "],
    ).dropna(how="all")

    timestamps = raw.iloc[:, TIMESTAMP_COL].to_numpy(dtype=float)

    sensor_data = {}
    for name in SENSORS:
        start = SENSOR_COL_START[name]
        sensor_data[name] = raw.iloc[:, start:start + 6].to_numpy(dtype=float)

    return timestamps, sensor_data


def load_events(path):
    """Parse Events.csv (UTF-16-LE) → flat metadata dict."""
    df = pd.read_csv(path, encoding="utf-16-le")
    df.columns = [c.strip() for c in df.columns]

    meta = {}
    for _, row in df.iterrows():
        event = str(row.get("Event Name", "")).strip()
        value = str(row.get("Event Value", "")).strip()
        if event == "Dominant Side":
            meta["dominant_side"] = value
        elif event == "Injured Side":
            meta["injured_side"] = value
        elif event == "Eye height level":
            try:
                meta["eye_height"] = float(value)
            except ValueError:
                meta["eye_height"] = value
    return meta


# ------------------------------------------------------------------
# Feature extraction helpers
# ------------------------------------------------------------------

def _pos_velocity_mag(sensor_arr, fs):
    """Velocity magnitude from 3-D positions."""
    vel = np.linalg.norm(np.diff(sensor_arr[:, :3], axis=0), axis=1) * fs
    return np.concatenate([[vel[0]], vel])


def extract_sensor_features(name, arr, fs):
    """
    All per-sensor features for one tracker.

    Parameters
    ----------
    name : str   sensor label
    arr  : (T, 6) array  [PosX, PosY, PosZ, RotX, RotY, RotZ]
    fs   : float  sampling rate

    Returns
    -------
    dict of feature_name -> scalar
    """
    feats = {}
    prefix = name

    # ---- 1. Time-domain stats for each of the 6 channels ----
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        signal = arr[:, ch_idx]
        try:
            td = extract_time_domain_features(signal)
            for k, v in td.items():
                feats[f"{prefix}_{ch_name}_{k}"] = v
        except Exception:
            pass

    # ---- 2. Position magnitude as a single signal ----
    pos_mag = np.linalg.norm(arr[:, :3], axis=1)

    # Frequency
    try:
        fq = extract_frequency_features(pos_mag, fs=fs)
        for k, v in fq.items():
            feats[f"{prefix}_freq_{k}"] = v
    except Exception:
        pass

    # Wavelet
    try:
        wv = extract_wavelet_features(pos_mag, wavelet="db4", level=5)
        for k, v in wv.items():
            feats[f"{prefix}_wavelet_{k}"] = v
    except Exception:
        pass

    # Complexity/entropy
    try:
        cx = extract_complexity_features(pos_mag)
        for k, v in cx.items():
            feats[f"{prefix}_entropy_{k}"] = v
    except Exception:
        pass

    # ---- 3. Velocity-based movement quality ----
    vel = _pos_velocity_mag(arr, fs)
    dt = 1.0 / fs

    try:
        jk = calculate_smoothness_jerk(vel, dt=dt)
        for k, v in jk.items():
            feats[f"{prefix}_vel_{k}"] = v
    except Exception:
        pass

    try:
        feats[f"{prefix}_vel_sparc"] = calculate_sparc(vel, fs=fs)
    except Exception:
        pass

    # ---- 4. Rotation ROM / mean / min / max per axis ----
    rot_labels = ["RotX", "RotY", "RotZ"]
    for ax_idx, ax_name in enumerate(rot_labels):
        rot = arr[:, 3 + ax_idx]
        feats[f"{prefix}_{ax_name}_rom"]  = float(np.ptp(rot))
        feats[f"{prefix}_{ax_name}_mean"] = float(np.mean(rot))
        feats[f"{prefix}_{ax_name}_min"]  = float(np.min(rot))
        feats[f"{prefix}_{ax_name}_max"]  = float(np.max(rot))

    return feats


def extract_bilateral_features(pair_name, left_arr, right_arr):
    """Asymmetry features for one Left/Right pair."""
    feats = {}
    try:
        asym = calculate_asymmetry(left_arr, right_arr)
        for k, v in asym.items():
            feats[f"{pair_name}_{k}"] = v
    except Exception:
        pass
    return feats


# ------------------------------------------------------------------
# Per-session extraction
# ------------------------------------------------------------------

def extract_session(session_id, fs=SAMPLING_RATE):
    session_dir = DATA_DIR / str(session_id)

    print(f"\n[Session {session_id}] Loading ...", end=" ", flush=True)
    timestamps, sensor_data = load_user_movement(session_dir / "UserMovement.csv")
    n_frames = timestamps.shape[0]
    duration = float(timestamps[-1] - timestamps[0])
    print(f"{n_frames} frames, {duration:.1f}s")

    row = {
        "session_id": session_id,
        "n_frames":   n_frames,
        "duration_s": duration,
    }

    # Session metadata
    events_path = session_dir / "Events.csv"
    if events_path.exists():
        row.update(load_events(events_path))

    # Per-sensor features
    for name in SENSORS:
        print(f"  {name} ...", end=" ", flush=True)
        feats = extract_sensor_features(name, sensor_data[name], fs)
        row.update(feats)
        print(f"{len(feats)} features")

    # Bilateral asymmetry
    for pair_name, left_name, right_name in BILATERAL_PAIRS:
        print(f"  asymmetry {pair_name} ...", end=" ", flush=True)
        feats = extract_bilateral_features(
            pair_name, sensor_data[left_name], sensor_data[right_name]
        )
        row.update(feats)
        print(f"{len(feats)} features")

    total = len(row) - 4  # subtract metadata keys
    print(f"  -> Session {session_id} done: {total} features")
    return row


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract features from NM longitudinal sessions."
    )
    parser.add_argument(
        "--sessions", nargs="+", type=int, default=[1, 2, 3, 4],
        help="Session IDs to process (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--out", type=str,
        default=str(HERE / "nm_features.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    rows = []
    for sid in args.sessions:
        session_dir = DATA_DIR / str(sid)
        if not session_dir.is_dir():
            print(f"[Warning] Session directory not found: {session_dir} — skipping.")
            continue
        try:
            rows.append(extract_session(sid))
        except Exception as e:
            import traceback
            print(f"[Error] Session {sid}: {e}")
            traceback.print_exc()

    if not rows:
        print("No sessions processed.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Put metadata columns first
    meta_cols = ["session_id", "dominant_side", "injured_side", "eye_height",
                 "n_frames", "duration_s"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feature_cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df)} sessions x {len(feature_cols)} features -> {out_path}")


if __name__ == "__main__":
    main()
