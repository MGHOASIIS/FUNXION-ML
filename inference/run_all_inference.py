"""
run_all_inference.py
====================
Batch inference runner: auto-discovers every checkpoint in experiments_from_hpc/
and runs inference for each task × paradigm × model combination.

Training data preprocessing is cached per (task, paradigm) so training data is
loaded only once per group — not once per model.

Usage
-----
    # Run all available checkpoints for one test subject
    python run_all_inference.py --test-subject-dir data/test_data/PX41 --subject-id PX_41

    # Filter to specific tasks or paradigms
    python run_all_inference.py --test-subject-dir data/test_data/PX41 --subject-id PX_41 \\
        --tasks 1 2 --paradigms 1 2

    # Custom experiments and output paths
    python run_all_inference.py \\
        --test-subject-dir data/test_data/PX41 --subject-id PX_41 \\
        --experiments-dir results/experiments_from_hpc \\
        --output results/PX41_all_results.csv
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# Allow running as `python inference/run_all_inference.py` from the project
# root — the script's own directory (inference/) is on sys.path by default,
# not the project root, so top-level packages wouldn't otherwise be importable.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.ingestion import load_dataset_config
from data.paradigms import ParadigmSelector
from data.preprocessors import PreprocessorFactory
from inference import (
    load_training_data,
    build_model,
    train_final_model,
    preprocess_test_truncate,
    run_inference,
)

# Set by main() before any helper runs
_DATASET_CONFIG: dict = {}


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(experiments_dir: Path) -> list[dict]:
    """
    Walk experiments_dir and return a list of checkpoint metadata dicts.

    Expected path structure:
        experiments_dir/task{N}/paradigm{P}/{MODEL}_{TIMESTAMP}/model_checkpoints/best_model_BA*.pt

    Returns
    -------
    List of dicts with keys: task, paradigm, model, ba, path
    """
    records = []
    pattern = re.compile(r"^(CNN|RNN|TRANSFORMER)_\d{8}_\d{6}$", re.IGNORECASE)

    for ckpt_path in sorted(experiments_dir.rglob("best_model_BA*.pt")):
        parts = ckpt_path.parts
        try:
            # Locate 'task{N}' and 'paradigm{P}' in the path parts
            task_part     = next(p for p in parts if p.startswith("task") and p[4:].isdigit())
            paradigm_part = next(p for p in parts if p.startswith("paradigm") and p[8:].isdigit())
            exp_dir_name  = ckpt_path.parent.parent.name   # e.g. CNN_20260303_133635

            if not pattern.match(exp_dir_name):
                print(f"  [skip] Unexpected experiment folder name: {exp_dir_name}")
                continue

            task_num     = int(task_part[4:])
            paradigm_num = int(paradigm_part[8:])
            model_name   = exp_dir_name.split("_")[0].upper()  # CNN / RNN / TRANSFORMER

            # Extract BA from filename
            ba_match = re.search(r"BA([\d.]+).pt", ckpt_path.name)
            ba       = float(ba_match.group(1)) if ba_match else None

            records.append({
                "task":     task_num,
                "paradigm": paradigm_num,
                "model":    model_name,
                "ba":       ba,
                "path":     ckpt_path,
            })
        except (StopIteration, ValueError, IndexError):
            print(f"  [skip] Could not parse path: {ckpt_path}")

    return records


# ---------------------------------------------------------------------------
# Training-data cache
# ---------------------------------------------------------------------------

def build_train_cache(
    checkpoints: list[dict],
    method: str = "truncate",
    resample_rate: int = 50,
) -> dict:
    """
    Pre-load and preprocess training data for every unique (task, paradigm)
    combination found in the checkpoint list.

    Returns
    -------
    dict keyed by (task, paradigm) →
        {
            "X_train":       np.ndarray,
            "y_train":       np.ndarray,
            "scaler":        fitted StandardScaler (CNN),
            "scaler_3d":     fitted StandardScaler (RNN/Transformer),
            "T_seq_cf":      int — sequence length for channels-first (CNN),
            "T_seq_3d":      int — sequence length for 3D (RNN/Transformer),
        }
    """
    unique_combos = sorted({(c["task"], c["paradigm"]) for c in checkpoints})
    cache = {}
    selector = ParadigmSelector(_DATASET_CONFIG)

    print(f"\n[Cache] Pre-loading training data for "
          f"{len(unique_combos)} task×paradigm combinations...\n")

    for task, paradigm in unique_combos:
        key = (task, paradigm)
        print(f"  Loading task={task}  paradigm={paradigm} ...", end="  ", flush=True)

        try:
            patient_data, control_data = load_training_data(task)
            g1, g0 = selector.select_paradigm(patient_data, control_data,
                                               paradigm=paradigm)

            # Preprocess for CNN (channels_first)
            pp_cf = PreprocessorFactory.create(
                method=method, model_type="cnn",
                resample_rate=resample_rate, original_rate=_DATASET_CONFIG.get("sampling_rate", 50),
            )
            X_cf, y_cf, _ = pp_cf.prepare_data(g1, g0)
            inner_cf = getattr(pp_cf, "inner", pp_cf)
            T_seq_cf = X_cf.shape[2]    # (N, C, T)

            # Preprocess for RNN/Transformer (3d)
            pp_3d = PreprocessorFactory.create(
                method=method, model_type="rnn",
                resample_rate=resample_rate, original_rate=_DATASET_CONFIG.get("sampling_rate", 50),
            )
            X_3d, y_3d, _ = pp_3d.prepare_data(g1, g0)
            inner_3d = getattr(pp_3d, "inner", pp_3d)
            T_seq_3d = X_3d.shape[1]    # (N, T, C)

            cache[key] = {
                "X_cf":     X_cf,
                "y_cf":     y_cf,
                "scaler_cf": inner_cf.scaler,
                "T_seq_cf": T_seq_cf,
                "X_3d":     X_3d,
                "y_3d":     y_3d,
                "scaler_3d": inner_3d.scaler,
                "T_seq_3d": T_seq_3d,
            }
            print(f"OK  (g1={len(g1)}, g0={len(g0)}, "
                  f"T_cf={T_seq_cf}, T_3d={T_seq_3d})")

        except Exception as e:
            print(f"FAILED — {e}")
            cache[key] = None

    return cache


# ---------------------------------------------------------------------------
# Single inference run
# ---------------------------------------------------------------------------

def run_one(
    ckpt_meta: dict,
    test_data: dict,
    train_cache: dict,
    expected_subject_id: str,
) -> dict | None:
    """
    Run inference for one checkpoint against the test subject.

    test_data is expected to contain exactly the one subject named by
    expected_subject_id (--subject-id) — prepare_test_data.py guarantees
    this, but we verify it rather than silently using whichever subject
    happens to be first in the dict.

    Returns a result dict, or None if the run could not complete.
    """
    task     = ckpt_meta["task"]
    paradigm = ckpt_meta["paradigm"]
    model_name = ckpt_meta["model"]      # CNN / RNN / TRANSFORMER
    ckpt_path  = ckpt_meta["path"]

    cache = train_cache.get((task, paradigm))
    if cache is None:
        return {"error": "training data unavailable"}

    # Determine format for this model
    is_cnn = model_name == "CNN"
    if is_cnn:
        X_train   = cache["X_cf"]
        y_train   = cache["y_cf"]
        scaler    = cache["scaler_cf"]
        T_seq     = cache["T_seq_cf"]
        out_fmt   = "channels_first"
    else:
        X_train   = cache["X_3d"]
        y_train   = cache["y_3d"]
        scaler    = cache["scaler_3d"]
        T_seq     = cache["T_seq_3d"]
        out_fmt   = "3d"

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        best_params = ckpt["hyperparameters"]

        # Build and train final model on all training data
        model = build_model(model_name, best_params)
        model = train_final_model(model, X_train, y_train, best_params)

        # Preprocess test subject
        X_test, subject_ids = preprocess_test_truncate(
            test_data, scaler, T_seq, out_fmt
        )

        if subject_ids != [expected_subject_id]:
            raise ValueError(
                f"test_data for task={task} contains subjects {subject_ids}, "
                f"expected exactly ['{expected_subject_id}'] (--subject-id). "
                f"Check the pkl file passed to prepare_test_data.py."
            )

        # Inference
        preds, probs = run_inference(model, X_test)

        return {
            "task":       task,
            "paradigm":   paradigm,
            "model":      model_name,
            "train_ba":   ckpt_meta["ba"],
            "subject_id": subject_ids[0],
            "prediction": int(preds[0]),
            "label":      "patient" if preds[0] == 1 else "control",
            "prob_patient": round(float(probs[0]), 4),
            "checkpoint": str(ckpt_path),
        }

    except Exception as e:
        return {
            "task": task, "paradigm": paradigm, "model": model_name,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    """Print a grouped summary table to stdout."""
    rows = [r for r in results if "error" not in r]
    if not rows:
        print("\nNo successful runs.")
        return

    df = pd.DataFrame(rows)
    task_names     = _DATASET_CONFIG.get("tasks", {})
    paradigm_names = {k: v.get("name", f"paradigm_{k}")
                      for k, v in _DATASET_CONFIG.get("paradigms", {}).items()}
    df["task_name"]     = df["task"].map(task_names)
    df["paradigm_name"] = df["paradigm"].map(paradigm_names)

    print(f"\n{'='*85}")
    print(f"{'Task':<5} {'Paradigm':<8} {'Model':<13} {'Train BA':>8}  "
          f"{'Prediction':<22} {'P(patient)':>10}")
    print("-" * 85)

    for _, grp in df.groupby(["task", "paradigm"]):
        first = grp.iloc[0]
        print(f"\n  Task {first['task']} — {first['task_name']}  |  "
              f"Paradigm {first['paradigm']} — {first['paradigm_name']}")
        for _, row in grp.iterrows():
            flag = "(*)" if row["prediction"] == 1 else "   "
            print(f"    {row['model']:<13}  BA={row['train_ba']:.4f}"
                  f"  {flag} {row['label']:<20}  {row['prob_patient']:>10.4f}")

    print(f"\n{'='*85}")

    # Aggregate: majority vote and mean probability per task
    print("\nMajority vote summary (across models, per task×paradigm):")
    print(f"{'Task':<5} {'Paradigm':<8}  {'Patient votes':>14}  "
          f"{'Total models':>13}  {'Mean P(patient)':>16}  {'Verdict':>10}")
    print("-" * 75)
    for (task, paradigm), grp in df.groupby(["task", "paradigm"]):
        n_patient = int((grp["prediction"] == 1).sum())
        n_total   = len(grp)
        mean_prob = grp["prob_patient"].mean()
        verdict   = "PATIENT" if n_patient > n_total / 2 else "control"
        print(f"  {task:<4} {paradigm:<8}  {n_patient:>14}  {n_total:>13}  "
              f"{mean_prob:>16.4f}  {verdict:>10}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch inference over all task×paradigm×model checkpoints."
    )
    parser.add_argument("--dataset", default="xdash",
                        help="Dataset name (must match datasets/ folder). Default: xdash")
    parser.add_argument("--test-subject-dir", required=True,
                        help="Directory containing pickled/ subdir with test pkl files "
                             "(e.g. data/test_data/PX41)")
    parser.add_argument("--subject-id", required=True,
                        help="Subject ID used in the pkl dict (e.g. PX_41)")
    parser.add_argument("--experiments-dir", default="results/experiments_from_hpc",
                        help="Root directory of checkpoints (default: results/experiments_from_hpc)")
    parser.add_argument("--method", default="truncate",
                        choices=["truncate", "sliding_window", "padding", "phase_shift"],
                        help="Preprocessing method (default: truncate)")
    parser.add_argument("--tasks", type=int, nargs="+", default=None,
                        help="Restrict to specific task numbers (default: all found)")
    parser.add_argument("--paradigms", type=int, nargs="+", default=None,
                        help="Restrict to specific paradigm numbers (default: all found)")
    parser.add_argument("--models", nargs="+",
                        default=["CNN", "RNN", "TRANSFORMER"],
                        help="Restrict to specific models (default: CNN RNN TRANSFORMER)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: results/inference/all_inference_results.csv)")
    args = parser.parse_args()

    global _DATASET_CONFIG
    _DATASET_CONFIG = load_dataset_config(args.dataset)

    experiments_dir   = Path(args.experiments_dir)
    test_subject_dir  = Path(args.test_subject_dir)
    pkl_dir           = test_subject_dir / "pickled"
    models_upper      = [m.upper() for m in args.models]

    if not experiments_dir.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")
    if not pkl_dir.exists():
        raise FileNotFoundError(
            f"Pickled test data not found at {pkl_dir}. "
            f"Run prepare_test_data.py first."
        )

    # ------------------------------------------------------------------
    # 1. Discover checkpoints
    # ------------------------------------------------------------------
    print(f"\n[1/4] Discovering checkpoints in {experiments_dir}/...")
    all_ckpts = discover_checkpoints(experiments_dir)

    # Filter by user args
    if args.tasks:
        all_ckpts = [c for c in all_ckpts if c["task"] in args.tasks]
    if args.paradigms:
        all_ckpts = [c for c in all_ckpts if c["paradigm"] in args.paradigms]
    all_ckpts = [c for c in all_ckpts if c["model"] in models_upper]

    print(f"  Found {len(all_ckpts)} checkpoints after filtering.")
    if not all_ckpts:
        print("  Nothing to run. Check --tasks / --paradigms / --models flags.")
        return

    # ------------------------------------------------------------------
    # 2. Build training-data cache
    # ------------------------------------------------------------------
    print("\n[2/4] Building training data cache...")
    train_cache = build_train_cache(all_ckpts, method=args.method)

    # ------------------------------------------------------------------
    # 3. Load test pkl files (one per task, reused across paradigms/models)
    # ------------------------------------------------------------------
    print("\n[3/4] Loading test pkl files...")
    test_pkls: dict[int, dict] = {}
    for task_num in sorted({c["task"] for c in all_ckpts}):
        pkl_path = pkl_dir / f"test_patient_task{task_num}.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                test_pkls[task_num] = pickle.load(f)
            print(f"  task {task_num}: {pkl_path.name}")
        else:
            print(f"  task {task_num}: NOT FOUND ({pkl_path}) — skipping")

    # ------------------------------------------------------------------
    # 4. Run inference for each checkpoint
    # ------------------------------------------------------------------
    print(f"\n[4/4] Running inference ({len(all_ckpts)} checkpoints)...\n")
    results = []
    errors  = []

    for i, ckpt_meta in enumerate(all_ckpts, 1):
        task = ckpt_meta["task"]
        if task not in test_pkls:
            continue

        print(f"  [{i:>3}/{len(all_ckpts)}]  "
              f"task={task}  paradigm={ckpt_meta['paradigm']}  "
              f"model={ckpt_meta['model']:<12}  "
              f"BA={ckpt_meta['ba']:.4f}", end="  ... ", flush=True)

        result = run_one(ckpt_meta, test_pkls[task], train_cache, args.subject_id)

        if result and "error" not in result:
            flag = "PATIENT (*)" if result["prediction"] == 1 else "control"
            print(f"{flag}  p={result['prob_patient']:.4f}")
            results.append(result)
        else:
            err = result.get("error", "unknown") if result else "unknown"
            print(f"ERROR — {err}")
            errors.append({**ckpt_meta, "error": err, "path": str(ckpt_meta["path"])})

    # ------------------------------------------------------------------
    # 5. Display and save
    # ------------------------------------------------------------------
    print_summary(results)

    # Save results CSV
    out_path = Path(args.output) if args.output else Path("results/inference/all_inference_results.csv")
    if results:
        pd.DataFrame(results).sort_values(
            ["task", "paradigm", "model"]
        ).to_csv(out_path, index=False)
        print(f"Results saved → {out_path}")

    # Save errors if any
    if errors:
        err_path = out_path.with_name(out_path.stem + "_errors.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"Errors  saved → {err_path}  ({len(errors)} failed runs)")

    # Save full JSON for downstream use
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "subject_id": args.subject_id,
            "timestamp":  datetime.now().isoformat(),
            "method":     args.method,
            "n_runs":     len(results),
            "n_errors":   len(errors),
            "results":    results,
        }, f, indent=2)
    print(f"Full JSON saved → {json_path}\n")


if __name__ == "__main__":
    main()
