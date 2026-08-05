"""
pipeline/io.py — data loading and result persistence.
"""
import json
import pickle
from pathlib import Path

import pandas as pd

from config.paths import get_pickled_dataset_path, get_event_window_path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(task: int, dataset: str, dataset_config: dict) -> tuple:
    """Load subject-level patient and control pickled datasets."""
    patient_path = get_pickled_dataset_path(task, "patient", dataset=dataset)
    control_path = get_pickled_dataset_path(task, "control", dataset=dataset)

    with open(patient_path, "rb") as f:
        patient_data = pickle.load(f)
    with open(control_path, "rb") as f:
        control_data = pickle.load(f)

    task_name = dataset_config["tasks"].get(task, task)
    print(f"\n[Data Loaded]")
    print(f"  Task:     {task} ({task_name})")
    print(f"  Patients: {len(patient_data)}")
    print(f"  Controls: {len(control_data)}")
    return patient_data, control_data


def load_event_window_data(task: int, dataset: str, dataset_config: dict) -> tuple:
    """Load event-window pickled datasets (one entry per window)."""
    g1_path = get_event_window_path(task, "g1", dataset=dataset)
    g0_path = get_event_window_path(task, "g0", dataset=dataset)

    with open(g1_path, "rb") as f:
        g1_data = pickle.load(f)
    with open(g0_path, "rb") as f:
        g0_data = pickle.load(f)

    task_name = dataset_config["tasks"].get(task, task)
    print(f"\n[Data Loaded] event-window")
    print(f"  Task:          {task} ({task_name})")
    print(f"  g1 (patients): {len(g1_data)} windows")
    print(f"  g0 (controls): {len(g0_data)} windows")
    return g1_data, g0_data


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_results(results, task: int, paradigm: int, model_name: str,
                 method: str, save_dir: Path, dataset_config: dict) -> dict:
    """Serialise metrics and best params to JSON."""
    task_names = dataset_config.get("tasks", {})
    paradigm_names = {
        int(k): v.get("name", f"paradigm_{k}")
        for k, v in dataset_config.get("paradigms", {}).items()
    }
    results_dict = {
        "task": task,
        "task_name": task_names.get(task, str(task)),
        "paradigm": paradigm,
        "paradigm_name": paradigm_names.get(paradigm, str(paradigm)),
        "model": model_name,
        "preprocessing_method": method,
        "metrics": results.metrics,
        "best_params": results.best_params,
        "feature_importance": results.feature_importance,
        "X_shape": list(results.X_shape),
    }
    filename = f"results_T{task}_P{paradigm}_{model_name}_{method}.json"
    filepath = save_dir / filename
    with open(filepath, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n[Results Saved] {filepath}")
    return results_dict


def save_predictions(results, subject_ids, task: int, paradigm: int,
                     model_name: str, method: str, save_dir: Path) -> pd.DataFrame:
    """Save per-sample predictions to CSV."""
    sids = results.subject_ids if results.subject_ids is not None else subject_ids

    # Guard against the IDs and predictions coming from differently-ordered
    # arrays (e.g. one built patients-first, the other controls-first) — this
    # single check would have caught that exact class of bug.
    if results.subject_ids is not None:
        assert len(sids) == len(results.y_true), (
            f"subject_ids length {len(sids)} != y_true length {len(results.y_true)}"
        )
        for sid, yt in zip(sids, results.y_true):
            expected = 1 if str(sid).startswith("g1_") else 0
            assert expected == int(yt), (
                f"subject_id/y_true mismatch: {sid} implies label {expected}, "
                f"got y_true={yt}"
            )

    rows = []
    for sid, yt, yp, ypr in zip(sids, results.y_true, results.y_pred, results.y_proba):
        sid_str = str(sid)
        parts = sid_str.split("__", 1)
        raw_subject = parts[0].split("_", 2)[-1] if "_" in parts[0] else parts[0]
        window_desc = parts[1] if len(parts) > 1 else ""
        correct = int(yt) == int(yp)
        error_type = (
            "correct" if correct
            else ("false_positive" if int(yp) == 1 else "false_negative")
        )
        rows.append({
            "subject_id": sid_str,
            "subject": raw_subject,
            "window_desc": window_desc,
            "y_true": int(yt),
            "y_pred": int(yp),
            "y_proba": round(float(ypr), 4),
            "correct": correct,
            "error_type": error_type,
        })

    df = pd.DataFrame(rows)
    filename = f"predictions_T{task}_P{paradigm}_{model_name}_{method}.csv"
    filepath = save_dir / filename
    df.to_csv(filepath, index=False)

    n_correct = df["correct"].sum()
    n_fp = (df["error_type"] == "false_positive").sum()
    n_fn = (df["error_type"] == "false_negative").sum()
    print(f"\n[Predictions Saved] {filepath}")
    print(f"  Correct: {n_correct}/{len(df)}  |  FP: {n_fp}  |  FN: {n_fn}")
    return df
