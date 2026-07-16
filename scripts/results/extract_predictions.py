"""
extract_predictions.py
======================
Extracts per-sample predictions from saved model checkpoints and produces
a misclassification analysis CSV.

For old checkpoints (no subject_ids saved), re-runs preprocessing using
the experiment config.json to recover the original subject/window identity.
Preprocessing is deterministic so sample order is guaranteed to match.

Checkpoint formats:
    HMM  → .json   (.../model_checkpoints/HMM_T*_P*_BA*.json)
    CNN / RNN / Transformer → .pt

Output:
    predictions_analysis/
        predictions_<exp_name>.csv       ← per-sample detail
        all_predictions.csv              ← all experiments combined
        summary_misclassifications.csv   ← one row per experiment

Columns in per-sample CSV:
    subject_id    raw preprocessor key
    subject       PX03 / fx07 etc.
    window_desc   event phase (event_window data only, else '')
    group         g1 | g0
    y_true        ground truth (0/1)
    y_pred        predicted label (0/1)
    y_proba       probability for class 1
    correct       True / False
    error_type    correct | false_positive | false_negative
    confidence    abs(y_proba - 0.5) * 2   (0=uncertain, 1=certain)

Usage:
    python extract_predictions.py --exp-dir /home/singh.vishwa/xdash2/experiments
    python extract_predictions.py --exp-dir experiments --name-filter ABL_EW
    python extract_predictions.py --exp-dir experiments --task 4 --paradigm 2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Make sure project root is on sys.path so dataio/ imports work ────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _infer_data_source(exp_name: str) -> str:
    """Infer data source from experiment name. ABL_EW_ → event_window, else subject."""
    if "ABL_EW_" in exp_name.upper():
        return "event_window"
    return "subject"


# =============================================================================
# 1. CHECKPOINT LOADING
# =============================================================================

def load_checkpoint(ckpt_path: Path) -> Optional[dict]:
    try:
        if ckpt_path.suffix == ".json":
            with open(ckpt_path) as f:
                return json.load(f)
        elif ckpt_path.suffix == ".pt":
            import torch
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return None
    except Exception as e:
        print(f"  [SKIP] Could not load {ckpt_path.name}: {e}")
        return None


def find_checkpoints(exp_dir: str, name_filter: Optional[str] = None) -> list[Path]:
    patterns = [
        os.path.join(exp_dir, "**", "model_checkpoints", "*.json"),
        os.path.join(exp_dir, "**", "model_checkpoints", "*.pt"),
        os.path.join(exp_dir, "**", "HMM_T*_P*_BA*.json"),
    ]
    files = list(set(f for pat in patterns for f in glob.glob(pat, recursive=True)))

    if name_filter:
        files = [f for f in files if name_filter in str(Path(f).parts[-3])]

    return sorted(Path(f) for f in files)


def _load_config(ckpt_path: Path) -> dict:
    """Load config.json from the experiment directory."""
    exp_dir = ckpt_path.parent.parent
    for candidate in [exp_dir / "config.json",
                      exp_dir.parent / "config.json"]:
        if candidate.exists():
            try:
                with open(candidate) as f:
                    cfg = json.load(f)
                # Backfill data_source if missing (old experiments)
                # infer from experiment directory name
                if "data_source" not in cfg:
                    cfg["data_source"] = _infer_data_source(exp_dir.name)
                return cfg
            except Exception:
                pass
    # No config.json found — infer what we can from the experiment name
    return {"data_source": _infer_data_source(exp_dir.name)}


# =============================================================================
# 2. EVENT METADATA LOOKUP FROM unified_dataset_raw.pkl
# =============================================================================

def load_event_lookup(project_dir: Path) -> dict:
    """
    Build a lookup dict for event metadata from unified_dataset_raw.pkl.

    Returns:
        dict keyed by (subject_id, task_number) → list of event metadata dicts,
        sorted by event_number so positional index matches trial order in the pickle.

    e.g. lookup[("PX34", 4)] = [
        {'start_event': 'Jar picked up', 'end_event': 'Lid grabbed', ...},  # trial1
        {'start_event': 'Lid grabbed',   'end_event': 'Lid removed', ...},  # trial2
        ...
    ]
    """
    candidates = [
        project_dir / "data" / "pickled_datasets" / "unified_dataset_raw.pkl",
        project_dir / "data" / "unified_dataset_raw.pkl",
        project_dir / "unified_dataset_raw.pkl",
    ]
    pkl_path = next((p for p in candidates if p.exists()), None)
    if pkl_path is None:
        print("  [WARN] unified_dataset_raw.pkl not found — event names unavailable")
        return {}

    try:
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)

        # Group entries by (subject_id, task_number), sorted by event_number
        from collections import defaultdict
        grouped = defaultdict(list)
        for entry in raw:
            sid   = entry.get("subject_id", "")
            task  = entry.get("task_number", "")
            if not sid:
                continue
            em = entry.get("event_metadata", {}) or {}
            grouped[(sid, task)].append({
                "event_number":     entry.get("event_number", ""),
                "start_event":      em.get("start_event", ""),
                "end_event":        em.get("end_event", ""),
                "hand_used":        em.get("hand_used", ""),
                "duration_s":       entry.get("duration_seconds", ""),
                "task_type":        entry.get("task_type", ""),
                "injury_type":      entry.get("injury_type", ""),
                "diagnosis":        entry.get("diagnosis", ""),
                "post_task_survey": entry.get("post_task_survey", ""),
                "pain_rating":      entry.get("pain_rating", ""),
                "age":              entry.get("age", ""),
                "gender":           entry.get("gender", ""),
            })

        # Sort each subject's entries by event_number
        lookup = {}
        for key, entries in grouped.items():
            lookup[key] = sorted(entries, key=lambda x: x["event_number"]
                                 if isinstance(x["event_number"], (int, float)) else 0)

        n_subjects = len(set(k[0] for k in lookup))
        print(f"  [Event Lookup] Loaded {len(raw)} entries, "
              f"{n_subjects} subjects from {pkl_path.name}")
        sample_keys = list(lookup.keys())[:3]
        print(f"  [Event Lookup] Sample keys: {sample_keys}")
        return lookup

    except Exception as e:
        print(f"  [WARN] Could not load event lookup: {e}")
        return {}


def recover_subject_ids(cfg: dict, n_expected: int,
                        project_dir: Path = None,
                        dataset: str = "xdash") -> Optional[list]:
    """
    Re-run preprocessing using the experiment config to recover subject_ids
    in the same order as the checkpoint predictions.

    Returns list of subject_id strings, or None if preprocessing fails.
    """
    task        = cfg.get("task")
    paradigm    = cfg.get("paradigm")
    method      = cfg.get("method", "truncate")
    model_type  = cfg.get("model", "hmm").lower()
    freq        = cfg.get("freq", 50)
    data_source = cfg.get("data_source", "subject")

    if task is None or paradigm is None:
        print("  [WARN] config.json missing task/paradigm — cannot recover subject_ids")
        return None

    print(f"  [DEBUG] data_source={data_source}  task={task}  paradigm={paradigm}  "
          f"method={method}  freq={freq}")

    try:
        # Try nested layout first (HPC: config/paths.py, data/paradigms.py)
        # then fall back to flat layout (local: paths.py, paradigms.py)
        try:
            from config.paths import get_pickled_dataset_path, get_event_window_path
            from dataio.paradigms import ParadigmSelector
            from dataio.preprocessors import PreprocessorFactory
            from dataio.ingestion import load_dataset_config
        except ModuleNotFoundError:
            from paths import get_pickled_dataset_path, get_event_window_path
            from paradigms import ParadigmSelector
            from preprocessors import PreprocessorFactory
            from ingestion import load_dataset_config

        # If project_dir given, build data paths directly to avoid
        # paths.py's parent.parent resolution which may be wrong locally
        def _pkl_path(task_, dtype):
            if project_dir:
                return project_dir / "data" / "pickled_datasets" / \
                       f"{dtype}_data_task{task_}.pkl"
            return get_pickled_dataset_path(task_, dtype)

        def _ew_path(task_, group):
            if project_dir:
                return project_dir / "data" / "pickled_datasets" / \
                       "event_window" / f"{group}_data_task{task_}.pkl"
            return get_event_window_path(task_, group)

        # ── Load data ────────────────────────────────────────────────────────
        if data_source == "event_window":
            g1_path = _ew_path(task, "g1")
            g0_path = _ew_path(task, "g0")
            print(f"  [DEBUG] Loading EW data: {g1_path}")
            with open(g1_path, "rb") as f: patient_data = pickle.load(f)
            with open(g0_path, "rb") as f: control_data = pickle.load(f)
        else:
            p_path = _pkl_path(task, "patient")
            c_path = _pkl_path(task, "control")
            print(f"  [DEBUG] Loading subject data: {p_path}")
            with open(p_path, "rb") as f: patient_data = pickle.load(f)
            with open(c_path, "rb") as f: control_data = pickle.load(f)

        # ── Paradigm selection ────────────────────────────────────────────────
        selector = ParadigmSelector(load_dataset_config(dataset))
        g1, g0 = selector.select_paradigm(patient_data, control_data, paradigm)

        # ── Build preprocessor kwargs from config ─────────────────────────────
        kwargs = {}
        if method == "sliding_window":
            kwargs["window_size"] = cfg.get("window_size", 300)
            kwargs["overlap"]     = cfg.get("overlap", 0.3)
        elif method == "dtw_embedding":
            kwargs["n_components"] = cfg.get("n_components", 10)
            kwargs["dtw_method"]   = cfg.get("dtw_method", "mds")
        elif method == "phase_shift":
            kwargs["shift_fraction"] = cfg.get("shift_fraction", 0.1)

        preprocessor = PreprocessorFactory.create(
            method=method,
            model_type=model_type,
            resample_rate=freq,
            original_rate=50,
            data_source=data_source,
            **kwargs
        )

        _, _, subject_ids = preprocessor.prepare_data(g1, g0)
        sids = list(subject_ids)

        if len(sids) != n_expected:
            print(f"  [WARN] Recovered {len(sids)} subject_ids but checkpoint has "
                  f"{n_expected} predictions — mismatch, falling back to indices.")
            return None

        print(f"  [OK] Recovered {len(sids)} subject_ids via preprocessing")
        return sids

    except Exception as e:
        import traceback
        print(f"  [WARN] Could not recover subject_ids: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# 3. PARSE SUBJECT ID KEY
# =============================================================================

def _parse_subject_id(sid: str) -> dict:
    """
    Parse a raw subject_id key into component fields.

    Key formats observed:
        Subject-level:
            "g1_0_PX03"         → subject=PX03, task='', trial='', group=g1
            "PX03"              → subject=PX03, task='', trial='', group=unknown

        Event-window (from pickle):
            "g1_0_PX34_task4_trial2"  → subject=PX34, task=4, trial=2, group=g1
            "PX34_task4_trial2"       → subject=PX34, task=4, trial=2, group=unknown
            "fx18_task4_trial1"       → subject=fx18, task=4, trial=1, group=unknown
    """
    sid   = str(sid)
    group = "g1" if sid.startswith("g1_") else ("g0" if sid.startswith("g0_") else "unknown")

    # Strip group prefix:  g1_PX34_task4_trial2 → PX34_task4_trial2
    #                      g0_fx18_task4_trial1  → fx18_task4_trial1
    core = sid[3:] if sid.startswith(("g1_", "g0_")) else sid

    # Parse subject_task_trial format:  PX34_task4_trial2
    subject  = core
    task_id  = ""
    trial_id = ""

    parts = core.split("_")
    # Look for 'task' and 'trial' tokens
    for i, part in enumerate(parts):
        if part.startswith("task") and part[4:].isdigit():
            task_id = part[4:]
            subject = "_".join(parts[:i])   # everything before 'taskN'
        elif part.startswith("trial") and part[5:].isdigit():
            trial_id = part[5:]

    return {
        "subject":  subject if subject else core,
        "task_id":  task_id,
        "trial_id": trial_id,
        "group":    group,
    }


# =============================================================================
# 4. BUILD PREDICTIONS DATAFRAME
# =============================================================================

def build_predictions_df(ckpt: dict, ckpt_path: Path, cfg: dict,
                         project_dir: Path = None,
                         event_lookup: dict = None,
                         dataset: str = "xdash") -> Optional[pd.DataFrame]:
    preds = ckpt.get("predictions", {})
    if not preds:
        print(f"  [SKIP] No predictions in {ckpt_path.name}")
        return None

    y_true  = preds.get("y_true",  [])
    y_pred  = preds.get("y_pred",  [])
    y_proba = preds.get("y_proba", [])
    sids    = preds.get("subject_ids", [])

    n = len(y_true)
    if n == 0:
        return None

    # ── Recover subject_ids if missing ───────────────────────────────────────
    if not sids:
        print(f"  [INFO] No subject_ids in checkpoint — recovering via preprocessing...")
        sids = recover_subject_ids(cfg, n, project_dir, dataset)

    if not sids:
        print(f"  [WARN] Using positional indices")
        sids = [f"sample_{i}" for i in range(n)]

    event_lookup = event_lookup or {}

    # Per-subject trial counter for positional event matching
    # subject_id → how many times we've seen it so far
    subject_trial_counter: dict = {}

    rows = []
    for sid, yt, yp, ypr in zip(sids, y_true, y_pred, y_proba):
        parsed     = _parse_subject_id(sid)
        correct    = int(yt) == int(yp)
        confidence = round(abs(float(ypr) - 0.5) * 2, 4)
        subject    = parsed["subject"]
        task_num   = cfg.get("task")

        # Increment per-subject counter to get trial index (0-based)
        trial_idx = subject_trial_counter.get(subject, 0)
        subject_trial_counter[subject] = trial_idx + 1

        # Look up event metadata positionally
        ev_list = event_lookup.get((subject, task_num), [])
        ev = ev_list[trial_idx] if trial_idx < len(ev_list) else {}

        rows.append({
            "subject_id":       str(sid),
            "subject":          subject,
            "task_id":          task_num,
            "trial_id":         trial_idx + 1,   # 1-based
            "group":            parsed["group"],
            # event metadata from unified_dataset_raw.pkl
            "start_event":      ev.get("start_event", ""),
            "end_event":        ev.get("end_event", ""),
            "hand_used":        ev.get("hand_used", ""),
            "event_number":     ev.get("event_number", ""),
            "duration_s":       ev.get("duration_s", ""),
            "task_type":        ev.get("task_type", ""),
            "injury_type":      ev.get("injury_type", ""),
            "diagnosis":        ev.get("diagnosis", ""),
            "post_task_survey": ev.get("post_task_survey", ""),
            "pain_rating":      ev.get("pain_rating", ""),
            "age":              ev.get("age", ""),
            "gender":           ev.get("gender", ""),
            # predictions
            "y_true":           int(yt),
            "y_pred":           int(yp),
            "y_proba":          round(float(ypr), 4),
            "correct":          correct,
            "error_type":       "correct" if correct else
                                ("false_positive" if int(yp) == 1 else "false_negative"),
            "confidence":       confidence,
        })

    return pd.DataFrame(rows)


# =============================================================================
# 5. SUMMARY ROW
# =============================================================================

def build_summary_row(df: pd.DataFrame, ckpt: dict,
                      ckpt_path: Path, cfg: dict) -> dict:
    n         = len(df)
    n_fp      = (df["error_type"] == "false_positive").sum()
    n_fn      = (df["error_type"] == "false_negative").sum()
    n_correct = df["correct"].sum()
    wrong_df  = df[~df["correct"]]

    top_wrong = (wrong_df.groupby("subject").size()
                         .sort_values(ascending=False).head(5).to_dict())
    top_win   = {}
    if "trial_id" in df.columns and df["trial_id"].any():
        top_win = {
            f"{s} trial {t}": n
            for (s, t), n in (
                wrong_df.groupby(["subject", "trial_id"]).size()
                        .sort_values(ascending=False).head(5).items()
            )
        }

    borderline = (df[df["correct"]].nsmallest(3, "confidence")["subject_id"].tolist())
    metrics    = ckpt.get("metrics", {})

    return {
        "experiment":                  ckpt_path.parent.parent.name,
        "model":                       ckpt.get("model_name", cfg.get("model", "")),
        "method":                      cfg.get("method", ""),
        "freq":                        cfg.get("freq", 50),
        "data_source":                 cfg.get("data_source", "subject"),
        "task":                        cfg.get("task", ckpt.get("task")),
        "paradigm":                    cfg.get("paradigm", ckpt.get("paradigm")),
        "ba":                          metrics.get("balanced_accuracy", metrics.get("ba")),
        "auc":                         metrics.get("auc"),
        "n_samples":                   n,
        "n_correct":                   int(n_correct),
        "n_false_positive":            int(n_fp),
        "n_false_negative":            int(n_fn),
        "accuracy":                    round(n_correct / n, 4) if n else None,
        "top_misclassified_subjects":  json.dumps(top_wrong),
        "top_misclassified_windows":   json.dumps(top_win),
        "borderline_correct":          json.dumps(borderline),
    }


# =============================================================================
# 6. CLI + MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract per-sample predictions from model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--exp-dir",      default="experiments",
                   help="Root experiment dir (default: experiments). "
                        "HPC: /home/singh.vishwa/xdash2/experiments")
    p.add_argument("--project-dir",  default=None,
                   help="Project root directory. Defaults to current working "
                        "directory. Set this if running from outside the project "
                        "root so imports and data paths resolve correctly.")
    p.add_argument("--out-dir",      default="results/predictions")
    p.add_argument("--name-filter",  default=None,
                   help="Only experiments whose folder name contains this "
                        "(e.g. 'ABL_EW')")
    p.add_argument("--task",         type=int, default=None)
    p.add_argument("--paradigm",     type=int, default=None)
    p.add_argument("--dataset",      default="xdash",
                   help="Dataset name (must match datasets/ folder). Default: xdash")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve project root and fix sys.path ─────────────────────────────────
    project_dir = Path(args.project_dir).resolve() if args.project_dir \
                  else Path.cwd()
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    print(f"\n{'='*70}")
    print("  Extract Predictions from Checkpoints")
    print(f"  Exp dir     : {args.exp_dir}")
    print(f"  Project dir : {project_dir}")
    print(f"  Name filter : {args.name_filter or '(none)'}")
    print(f"  Output      : {out_dir}")
    print(f"{'='*70}\n")

    ckpt_paths = find_checkpoints(args.exp_dir, name_filter=args.name_filter)
    print(f"Found {len(ckpt_paths)} checkpoints\n")
    if not ckpt_paths:
        print("No checkpoints found. Check --exp-dir.")
        return

    # Load event metadata lookup once — shared across all checkpoints
    print("[Event Lookup] Loading unified_dataset_raw.pkl...")
    event_lookup = load_event_lookup(project_dir)
    print()

    summary_rows = []
    all_dfs      = []

    for ckpt_path in ckpt_paths:
        ckpt = load_checkpoint(ckpt_path)
        if ckpt is None:
            continue

        cfg      = _load_config(ckpt_path)
        task     = cfg.get("task",     ckpt.get("task"))
        paradigm = cfg.get("paradigm", ckpt.get("paradigm"))

        if args.task     and task     != args.task:     continue
        if args.paradigm and paradigm != args.paradigm: continue

        exp_name = ckpt_path.parent.parent.name
        print(f"── {exp_name}")

        df = build_predictions_df(ckpt, ckpt_path, cfg, project_dir, event_lookup,
                                   dataset=args.dataset)
        if df is None:
            continue

        df["experiment"] = exp_name
        df["model"]      = ckpt.get("model_name", cfg.get("model", ""))
        df["method"]     = cfg.get("method", "")
        df["freq"]       = cfg.get("freq", 50)
        df["data_source"]= cfg.get("data_source", "subject")
        df["task"]       = task
        df["paradigm"]   = paradigm

        out_csv = out_dir / f"predictions_{exp_name}.csv"
        df.to_csv(out_csv, index=False)

        n_wrong = (~df["correct"]).sum()
        print(f"   {len(df)} samples  |  {n_wrong} misclassified  "
              f"|  saved → {out_csv.name}\n")

        summary_rows.append(build_summary_row(df, ckpt, ckpt_path, cfg))
        all_dfs.append(df)

    if not all_dfs:
        print("No predictions extracted.")
        return

    # ── Combined CSV ──────────────────────────────────────────────────────────
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(out_dir / "all_predictions.csv", index=False)
    print(f"\n[Combined] {len(combined)} rows → all_predictions.csv")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(["task", "paradigm", "model", "method"], inplace=True)
    summary_df.to_csv(out_dir / "summary_misclassifications.csv", index=False)
    print(f"[Summary]  {len(summary_df)} experiments → summary_misclassifications.csv")

    # ── Console: most misclassified subjects ──────────────────────────────────
    wrong = combined[~combined["correct"]]
    if not wrong.empty:
        print(f"\n{'='*70}")
        print("  TOP MISCLASSIFIED SUBJECTS")
        print(f"{'='*70}")
        top = (wrong.groupby(["subject", "group"]).size()
                    .reset_index(name="n_errors")
                    .sort_values("n_errors", ascending=False).head(15))
        print(top.to_string(index=False))

    # ── Event phase accuracy breakdown ───────────────────────────────────────
    ew = combined[combined["start_event"] != ""].copy()
    if not ew.empty:
        phase_stats = (
            ew.groupby(["start_event", "end_event"])
              .agg(
                  total        = ("correct", "count"),
                  n_correct    = ("correct", "sum"),
                  n_fp         = ("error_type", lambda x: (x == "false_positive").sum()),
                  n_fn         = ("error_type", lambda x: (x == "false_negative").sum()),
                  mean_conf    = ("confidence", "mean"),
              )
              .reset_index()
        )
        phase_stats["n_wrong"]    = phase_stats["total"] - phase_stats["n_correct"]
        phase_stats["accuracy"]   = (phase_stats["n_correct"] / phase_stats["total"]).round(3)
        phase_stats["mean_conf"]  = phase_stats["mean_conf"].round(3)
        phase_stats = phase_stats.sort_values("accuracy")

        # ── Save full phase breakdown CSV ─────────────────────────────────────
        phase_path = out_dir / "event_phase_accuracy.csv"
        phase_stats.to_csv(phase_path, index=False)
        print(f"\n[Event Phases] Saved breakdown → {phase_path.name}")

        # ── Hardest phases (lowest accuracy) ─────────────────────────────────
        print(f"\n{'='*70}")
        print("  HARDEST EVENT PHASES  (lowest classification accuracy)")
        print(f"{'='*70}")
        cols = ["start_event", "end_event", "total", "n_correct", "n_wrong",
                "n_fp", "n_fn", "accuracy", "mean_conf"]
        print(phase_stats[cols].head(10).to_string(index=False))

        # ── Easiest phases (highest accuracy) ─────────────────────────────────
        print(f"\n{'='*70}")
        print("  EASIEST EVENT PHASES  (highest classification accuracy)")
        print(f"{'='*70}")
        print(phase_stats[cols].tail(10).to_string(index=False))

        # ── Phases with most false positives ──────────────────────────────────
        print(f"\n{'='*70}")
        print("  MOST FALSE POSITIVES BY PHASE  (controls classified as patients)")
        print(f"{'='*70}")
        fp_phases = phase_stats.sort_values("n_fp", ascending=False).head(10)
        print(fp_phases[["start_event", "end_event", "total", "n_fp", "accuracy"]].to_string(index=False))

        # ── Phases with most false negatives ──────────────────────────────────
        print(f"\n{'='*70}")
        print("  MOST FALSE NEGATIVES BY PHASE  (patients classified as controls)")
        print(f"{'='*70}")
        fn_phases = phase_stats.sort_values("n_fn", ascending=False).head(10)
        print(fn_phases[["start_event", "end_event", "total", "n_fn", "accuracy"]].to_string(index=False))

    print(f"\n{'='*70}")
    print(f"  Done. All outputs in: {out_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()