"""
compare_push_down.py
====================
Longitudinal comparison of all features for "Push Down Exercise"
across sessions 1-4.

Output: push_down_longitudinal.csv
  One row per feature, columns:
    feature               name of the feature
    session_1 .. 4        raw value per session
    delta_1_to_4          session_4 - session_1
    pct_change_1_to_4     % change from session 1 to session 4
    direction             increasing / decreasing / stable / mixed
    monotonic             True if values change in the same direction every step

Rows sorted by abs(pct_change_1_to_4) descending so the most-changed
features appear first.

Usage
-----
    python analysis/longitudinal/compare_push_down.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

HERE     = Path(__file__).resolve().parent
IN_CSV   = HERE / "nm_features.csv"
OUT_CSV  = HERE / "push_down_longitudinal.csv"

TASK     = "Push Down Exercise"
SESSIONS = [1, 2, 3, 4]

META_COLS = [
    "session_id", "task_name",
    "task_start_s", "task_end_s", "task_duration_s",
    "dominant_side", "injured_side",
    "eye_height", "torso_width", "torso_height",
    "leg_height", "arm_length", "shoulder_height",
    "n_reps", "n_pickups", "n_stands",
    "furthest_rotation_deg", "piston_distance_m", "pain_score",
    "n_frames",
]


def direction_label(values):
    """
    Classify how a feature changes across 4 sessions.
    values : list/array of 4 floats (may contain NaN)
    """
    v = [x for x in values if pd.notna(x)]
    if len(v) < 2:
        return "insufficient_data"
    diffs = np.diff(v)
    if all(d > 0 for d in diffs):
        return "increasing"
    if all(d < 0 for d in diffs):
        return "decreasing"
    if all(d == 0 for d in diffs):
        return "stable"
    return "mixed"


def is_monotonic(values):
    v = [x for x in values if pd.notna(x)]
    if len(v) < 2:
        return False
    diffs = np.diff(v)
    return bool(all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs))


def main():
    df = pd.read_csv(IN_CSV)

    push = (
        df[df["task_name"] == TASK]
        .sort_values("session_id")
        .reset_index(drop=True)
    )

    if len(push) == 0:
        raise ValueError(f"No rows found for task: {TASK!r}")

    found_sessions = push["session_id"].tolist()
    print(f"Sessions found: {found_sessions}")
    print(f"Task duration (s): {push['task_duration_s'].tolist()}")

    # Feature columns only (drop meta + all-NaN cols)
    feat_cols = [c for c in push.columns if c not in META_COLS]
    feat_data = push[feat_cols].select_dtypes(include=np.number)
    feat_data = feat_data.loc[:, feat_data.notna().any()]  # drop all-NaN cols
    print(f"Feature columns after cleaning: {len(feat_data.columns)}")

    # Build comparison table
    records = []
    for feat in feat_data.columns:
        vals = feat_data[feat].tolist()           # one value per session row
        session_vals = {
            f"session_{sid}": v
            for sid, v in zip(found_sessions, vals)
        }

        v1 = vals[0] if pd.notna(vals[0]) else None
        v4 = vals[-1] if pd.notna(vals[-1]) else None

        if v1 is not None and v4 is not None:
            delta     = v4 - v1
            pct_change = (delta / abs(v1) * 100) if v1 != 0 else np.nan
        else:
            delta      = np.nan
            pct_change = np.nan

        records.append({
            "feature":           feat,
            **session_vals,
            "delta_1_to_4":      delta,
            "pct_change_1_to_4": pct_change,
            "direction":         direction_label(vals),
            "monotonic":         is_monotonic(vals),
        })

    result = pd.DataFrame(records)

    # Sort by absolute percent change (largest movers first)
    result = result.sort_values(
        "pct_change_1_to_4",
        key=lambda s: s.abs(),
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    result.to_csv(OUT_CSV, index=False)

    # ---- summary print ----
    print(f"\nSaved -> {OUT_CSV}")
    print(f"Total features:  {len(result)}")
    dir_counts = result["direction"].value_counts()
    print(f"\nDirection breakdown:\n{dir_counts.to_string()}")
    print(f"\nMonotonic features: {result['monotonic'].sum()}")

    print(f"\nTop 20 most-changed features (by |% change session 1→4|):")
    top = result[result["pct_change_1_to_4"].notna()].head(20)
    print(
        top[["feature", "pct_change_1_to_4", "direction", "monotonic"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
