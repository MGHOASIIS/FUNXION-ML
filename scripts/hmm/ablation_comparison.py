"""
ablation_comparison.py
======================
Parses all ablation experiment summary.json files and produces:

  1. Per-method plots  — for each method, one figure comparing BA across
                         sampling frequencies (x-axis = freq, lines = models)

  2. Per-frequency plots — for each frequency, one figure comparing BA across
                           methods (x-axis = method, lines = models)

  3. Best hyperparameters table — for each (method, freq, model) combination,
                                   the best-performing hyperparameter set, saved
                                   as CSV and pretty-printed. Use this to fix
                                   hyperparameters for future experiments.

Usage:
    python ablation_comparison.py                          # default exp dir
    python ablation_comparison.py --exp-dir /path/to/logs/ablations
    python ablation_comparison.py --task 1 --paradigm 2   # filter
    python ablation_comparison.py --metric auc            # plot AUC instead of BA
    python ablation_comparison.py --out-dir my_plots      # custom output dir

Output:
    ablation_results/
        data/
            ablation_results.csv          ← all parsed rows
        plots/
            by_method/
                <method>_freq_comparison.png
            by_freq/
                freq<N>hz_method_comparison.png
        tables/
            best_hyperparameters.csv      ← best HP per (method, freq, model)
            best_hyperparameters.txt      ← human-readable version
"""

from __future__ import annotations

import argparse
import json
import glob
import os
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── Constants ─────────────────────────────────────────────────────────────────

TASK_NAMES = {
    1: "jar_opening", 2: "key_turning", 3: "cleaning",
    4: "back_washing", 5: "cutting", 6: "hammering",
}
PARADIGM_NAMES = {
    1: "patients_vs_controls", 2: "rct_vs_controls",
    3: "other_conditions_vs_controls", 4: "rct_vs_other_conditions",
}

MODELS   = ["HMM", "CNN", "RNN", "TRANSFORMER"]
METHODS  = ["variable_length", "truncate", "padding", "sliding_window",
            "phase_shift", "dtw_embedding"]
FREQS    = [50, 40, 30, 20, 10]

MODEL_COLORS = {
    "HMM":         "#E07B39",
    "CNN":         "#2E86AB",
    "RNN":         "#4CAF50",
    "TRANSFORMER": "#9C27B0",
}
MODEL_MARKERS = {
    "HMM": "o", "CNN": "s", "RNN": "^", "TRANSFORMER": "D",
}

METHOD_COLORS = {
    "variable_length": "#2E86AB",
    "truncate":        "#E07B39",
    "padding":         "#4CAF50",
    "sliding_window":  "#9C27B0",
    "phase_shift":     "#F44336",
    "dtw_embedding":   "#795548",
}

METRIC_LABELS = {
    "ba":        "Balanced Accuracy",
    "auc":       "AUC-ROC",
    "recall":    "Recall / Sensitivity",
    "precision": "Precision",
    "f1":        "F1 Score",
}


# =============================================================================
# 1. PARSING
# =============================================================================

def _extract_freq_from_tag(job_tag: str) -> Optional[int]:
    """Extract frequency from job tag, e.g. 'truncate_f30' → 30."""
    m = re.search(r"_f(\d+)", job_tag)
    return int(m.group(1)) if m else None


def load_ablation_summaries(
    exp_dir: str,
    task_filter: Optional[int] = None,
    paradigm_filter: Optional[int] = None,
    model_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Walk exp_dir recursively, load every summary.json, parse into a flat
    DataFrame. Handles both ablation and standard experiment layouts.
    """
    pattern = os.path.join(exp_dir, "**", "summary.json")
    files = glob.glob(pattern, recursive=True)
    print(f"[Parser] Found {len(files)} summary.json files in {exp_dir}")

    rows = []
    for fpath in sorted(files):
        try:
            with open(fpath) as f:
                d = json.load(f)
        except Exception as e:
            print(f"  [SKIP] {fpath}: {e}")
            continue

        cfg     = d.get("config", {})
        res     = d.get("results", {})
        metrics = res.get("metrics", {})
        bp      = res.get("best_params", {})
        evl     = d.get("evaluation", {})

        task     = cfg.get("task",     res.get("task"))
        paradigm = cfg.get("paradigm", res.get("paradigm"))
        model    = cfg.get("model",    res.get("model", "")).upper()
        method   = cfg.get("method",   res.get("preprocessing_method", ""))
        freq     = cfg.get("freq", 50)  # default 50 if not set (pre-ablation runs)
        exp_name = d.get("experiment_name", Path(fpath).parent.name)

        # Try to recover freq from experiment name if missing from config
        if freq is None or freq == 50:
            f_from_tag = _extract_freq_from_tag(exp_name)
            if f_from_tag is not None:
                freq = f_from_tag

        # Apply filters
        if task_filter    and task    != task_filter:    continue
        if paradigm_filter and paradigm != paradigm_filter: continue
        if model_filter   and model   != model_filter.upper(): continue

        ba  = metrics.get("ba")
        auc = metrics.get("auc")

        if ba is None:
            # try evaluation block
            ba  = evl.get("balanced_accuracy")
            auc = evl.get("auc_roc")

        rows.append({
            "experiment_name": exp_name,
            "task":            task,
            "task_name":       TASK_NAMES.get(task, f"task{task}"),
            "paradigm":        paradigm,
            "paradigm_name":   PARADIGM_NAMES.get(paradigm, f"p{paradigm}"),
            "model":           model,
            "method":          method,
            "freq":            int(freq) if freq is not None else 50,
            # metrics
            "ba":              float(ba)  if ba  is not None else np.nan,
            "auc":             float(auc) if auc is not None else np.nan,
            "recall":          float(metrics.get("recall",    np.nan)),
            "precision":       float(metrics.get("precision", np.nan)),
            "f1":              float(metrics.get("f1",        np.nan)),
            # best hyperparameters
            "best_params":     json.dumps(bp),
            "best_params_dict": bp,
            # provenance
            "summary_path":    fpath,
        })

    if not rows:
        print("[Parser] WARNING: No rows parsed. Check --exp-dir path.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values(["model", "method", "freq", "task", "paradigm"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[Parser] Parsed {len(df)} rows")
    print(f"  Methods : {sorted(df['method'].unique())}")
    print(f"  Freqs   : {sorted(df['freq'].unique())}")
    print(f"  Models  : {sorted(df['model'].unique())}")
    return df


# =============================================================================
# 2. PLOTTING HELPERS
# =============================================================================

def _finish_plot(fig, ax, title: str, xlabel: str, ylabel: str,
                 save_path: Path, legend_handles=None):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0.3, 1.02)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Chance (0.5)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    if legend_handles:
        handles = legend_handles + handles
    ax.legend(handles=handles, fontsize=9, framealpha=0.7,
              loc="lower right", ncol=2)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {save_path}")


# =============================================================================
# 3. PLOT 1 — Per-method: BA vs Frequency
# =============================================================================

def plot_by_method(df: pd.DataFrame, metric: str, out_dir: Path,
                   task: Optional[int], paradigm: Optional[int]):
    """
    One figure per method. X-axis = frequency, one line per model.
    If task/paradigm filters are set, plots that specific slice.
    Otherwise averages across all tasks and paradigms.
    """
    save_dir = out_dir / "by_method"
    ylabel   = METRIC_LABELS.get(metric, metric.upper())
    scope    = _scope_label(task, paradigm)

    methods_present = sorted(df["method"].unique())

    for method in methods_present:
        sub = df[df["method"] == method].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        legend_handles = []

        for model in MODELS:
            msub = sub[sub["model"] == model]
            if msub.empty:
                continue

            # Group by freq — mean ± std across tasks/paradigms
            grp = msub.groupby("freq")[metric].agg(["mean", "std"]).reset_index()
            grp = grp.sort_values("freq")

            color  = MODEL_COLORS.get(model, "#333333")
            marker = MODEL_MARKERS.get(model, "o")

            ax.plot(grp["freq"], grp["mean"], marker=marker, color=color,
                    linewidth=2, markersize=7, label=model)
            ax.fill_between(grp["freq"],
                            grp["mean"] - grp["std"].fillna(0),
                            grp["mean"] + grp["std"].fillna(0),
                            color=color, alpha=0.12)

            legend_handles.append(
                Line2D([0], [0], color=color, marker=marker,
                       linewidth=2, markersize=7, label=model)
            )

        ax.set_xticks(sorted(sub["freq"].unique()))
        ax.set_xticklabels([f"{f} Hz" for f in sorted(sub["freq"].unique())])

        _finish_plot(
            fig, ax,
            title=f"Method: {method}  —  {ylabel} vs Sampling Frequency\n{scope}",
            xlabel="Sampling Frequency (Hz)",
            ylabel=ylabel,
            save_path=save_dir / f"{method}_freq_comparison.png",
            legend_handles=legend_handles,
        )


# =============================================================================
# 4. PLOT 2 — Per-frequency: BA vs Method
# =============================================================================

def plot_by_freq(df: pd.DataFrame, metric: str, out_dir: Path,
                 task: Optional[int], paradigm: Optional[int]):
    """
    One figure per frequency. X-axis = method, one line per model.
    """
    save_dir = out_dir / "by_freq"
    ylabel   = METRIC_LABELS.get(metric, metric.upper())
    scope    = _scope_label(task, paradigm)

    freqs_present = sorted(df["freq"].unique())

    for freq in freqs_present:
        sub = df[df["freq"] == freq].copy()
        if sub.empty:
            continue

        methods_in_freq = sorted(sub["method"].unique())
        x_pos = {m: i for i, m in enumerate(methods_in_freq)}

        fig, ax = plt.subplots(figsize=(max(8, len(methods_in_freq) * 1.8), 5))
        legend_handles = []

        for model in MODELS:
            msub = sub[sub["model"] == model]
            if msub.empty:
                continue

            grp = msub.groupby("method")[metric].agg(["mean", "std"]).reset_index()

            xs     = [x_pos[m] for m in grp["method"]]
            ys     = grp["mean"].values
            yerr   = grp["std"].fillna(0).values
            color  = MODEL_COLORS.get(model, "#333333")
            marker = MODEL_MARKERS.get(model, "o")

            ax.errorbar(xs, ys, yerr=yerr, marker=marker, color=color,
                        linewidth=2, markersize=7, capsize=4, label=model)

            legend_handles.append(
                Line2D([0], [0], color=color, marker=marker,
                       linewidth=2, markersize=7, label=model)
            )

        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels(list(x_pos.keys()), rotation=20, ha="right", fontsize=9)

        _finish_plot(
            fig, ax,
            title=f"Frequency: {freq} Hz  —  {ylabel} vs Preprocessing Method\n{scope}",
            xlabel="Preprocessing Method",
            ylabel=ylabel,
            save_path=save_dir / f"freq{freq}hz_method_comparison.png",
            legend_handles=legend_handles,
        )


# =============================================================================
# 5. BEST HYPERPARAMETERS TABLE
# =============================================================================

def build_best_hp_table(df: pd.DataFrame, metric: str, out_dir: Path):
    """
    For each (method, freq, model) combination, find the row with the
    highest mean metric across tasks/paradigms and extract its best_params.

    Saves:
        tables/best_hyperparameters.csv  — machine-readable
        tables/best_hyperparameters.txt  — human-readable
    """
    save_dir = out_dir / "tables"
    save_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for (method, freq, model), grp in df.groupby(["method", "freq", "model"]):
        if grp[metric].isna().all():
            continue

        # Best row by metric
        best_idx = grp[metric].idxmax()
        best_row = grp.loc[best_idx]

        mean_metric = grp[metric].mean()
        std_metric  = grp[metric].std()

        try:
            bp = best_row["best_params_dict"]
            if not isinstance(bp, dict):
                bp = json.loads(best_row["best_params"])
        except Exception:
            bp = {}

        record = {
            "method":           method,
            "freq_hz":          freq,
            "model":            model,
            f"mean_{metric}":   round(mean_metric, 4),
            f"std_{metric}":    round(std_metric,  4) if not np.isnan(std_metric) else 0.0,
            f"best_{metric}":   round(best_row[metric], 4),
            "best_task":        best_row["task"],
            "best_paradigm":    best_row["paradigm"],
            "best_params_json": json.dumps(bp),
            **{f"hp_{k}": v for k, v in bp.items()},
        }
        records.append(record)

    if not records:
        print("[HP Table] No records — nothing to save.")
        return pd.DataFrame()

    hp_df = pd.DataFrame(records)
    hp_df.sort_values(["method", "freq_hz", "model"], inplace=True)
    hp_df.reset_index(drop=True, inplace=True)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = save_dir / "best_hyperparameters.csv"
    hp_df.to_csv(csv_path, index=False)
    print(f"  [Saved] {csv_path}")

    # ── Save human-readable TXT ───────────────────────────────────────────────
    txt_path = save_dir / "best_hyperparameters.txt"
    lines = [
        "=" * 80,
        "BEST HYPERPARAMETERS PER (METHOD × FREQ × MODEL)",
        f"Metric used for ranking: {METRIC_LABELS.get(metric, metric.upper())}",
        "=" * 80,
    ]
    prev_method = None
    for _, row in hp_df.iterrows():
        if row["method"] != prev_method:
            lines.append(f"\n{'─'*80}")
            lines.append(f"  METHOD: {row['method'].upper()}")
            lines.append(f"{'─'*80}")
            prev_method = row["method"]

        lines.append(
            f"\n  {row['model']:<14} @ {row['freq_hz']:>2} Hz  │  "
            f"{METRIC_LABELS.get(metric, metric)} = "
            f"{row[f'mean_{metric}']:.4f} ± {row[f'std_{metric}']:.4f}  "
            f"(best: {row[f'best_{metric}']:.4f})"
        )
        try:
            bp = json.loads(row["best_params_json"])
            for k, v in bp.items():
                lines.append(f"    {k:<22}: {v}")
        except Exception:
            lines.append(f"    {row['best_params_json']}")

    lines.append("\n" + "=" * 80)
    txt_path.write_text("\n".join(lines))
    print(f"  [Saved] {txt_path}")

    # ── Print summary to console ──────────────────────────────────────────────
    print("\n" + lines[0])
    print(lines[1])
    print(lines[2])
    print(lines[3])
    display_cols = ["method", "freq_hz", "model",
                    f"mean_{metric}", f"std_{metric}", f"best_{metric}",
                    "best_params_json"]
    display_cols = [c for c in display_cols if c in hp_df.columns]
    print(hp_df[display_cols].to_string(index=False))

    return hp_df


# =============================================================================
# 6. HELPERS
# =============================================================================

def _scope_label(task: Optional[int], paradigm: Optional[int]) -> str:
    parts = []
    if task:
        parts.append(f"Task {task} ({TASK_NAMES.get(task, '')})")
    else:
        parts.append("Task 4")
    if paradigm:
        parts.append(f"Paradigm {paradigm} ({PARADIGM_NAMES.get(paradigm, '')})")
    else:
        parts.append("Paradigm 2")
    return " | ".join(parts)


def save_data(df: pd.DataFrame, out_dir: Path):
    save_dir = out_dir / "data"
    save_dir.mkdir(parents=True, exist_ok=True)
    # drop non-serialisable dict column before saving
    csv_df = df.drop(columns=["best_params_dict"], errors="ignore")
    path = save_dir / "ablation_results.csv"
    csv_df.to_csv(path, index=False)
    print(f"  [Saved] {path}  ({len(df)} rows)")


# =============================================================================
# 7. CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Ablation comparison plots and best hyperparameter table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--exp-dir",   default="logs/ablations",
                   help="Root directory containing ablation experiment folders "
                        "(default: logs/ablations)")
    p.add_argument("--out-dir",   default="ablation_results",
                   help="Output directory for plots and tables (default: ablation_results)")
    p.add_argument("--metric",    default="ba",
                   choices=["ba", "auc", "recall", "precision", "f1"],
                   help="Metric to plot and rank by (default: ba)")
    p.add_argument("--task",      type=int, default=None, choices=range(1, 7),
                   help="Filter to a single task (default: all)")
    p.add_argument("--paradigm",  type=int, default=None, choices=range(1, 5),
                   help="Filter to a single paradigm (default: all)")
    p.add_argument("--model",     type=str, default=None,
                   choices=["hmm", "cnn", "rnn", "transformer"],
                   help="Filter to a single model (default: all)")
    return p.parse_args()


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)

    print(f"\n{'='*70}")
    print("  Ablation Comparison")
    print(f"  Exp dir : {args.exp_dir}")
    print(f"  Metric  : {args.metric} ({METRIC_LABELS.get(args.metric, '')})")
    print(f"  Filters : task={args.task}  paradigm={args.paradigm}  model={args.model}")
    print(f"  Output  : {out_dir}")
    print(f"{'='*70}\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_ablation_summaries(
        exp_dir=args.exp_dir,
        task_filter=args.task,
        paradigm_filter=args.paradigm,
        model_filter=args.model,
    )
    if df.empty:
        print("No data found — exiting.")
        return

    save_data(df, out_dir)

    # ── Plot 1: per-method, BA vs freq ────────────────────────────────────────
    print("\n[1/3] Generating per-method plots (BA vs frequency)...")
    plot_by_method(df, args.metric, out_dir, args.task, args.paradigm)

    # ── Plot 2: per-freq, BA vs method ───────────────────────────────────────
    print("\n[2/3] Generating per-frequency plots (BA vs method)...")
    plot_by_freq(df, args.metric, out_dir, args.task, args.paradigm)

    # ── Table 3: best hyperparameters ─────────────────────────────────────────
    print("\n[3/3] Building best hyperparameter table...")
    build_best_hp_table(df, args.metric, out_dir)

    print(f"\n{'='*70}")
    print(f"  Done.  All outputs in: {out_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()