"""
ablation_comparison.py
======================
Parses all ablation experiment summary.json files and produces:

  1. Per-method plots  — BA vs sampling frequency, one line per model
  2. Per-frequency plots — BA vs method, one line per model
     Both split into subject/ and event_window/ subfolders.

  3. Best hyperparameters table — per (data_source, method, freq, model),
     saved as CSV and human-readable TXT. Both data sources in one table.

Output structure:
    ablation_results/
        data/
            ablation_results.csv            ← all rows (both sources)
            ablation_results_subject.csv
            ablation_results_event_window.csv
        plots/
            subject/
                by_method/<method>_freq_comparison.png
                by_freq/freq<N>hz_method_comparison.png
            event_window/
                by_method/<method>_freq_comparison.png
                by_freq/freq<N>hz_method_comparison.png
        tables/
            best_hyperparameters.csv        ← both sources, data_source column
            best_hyperparameters.txt        ← human-readable, grouped by source

Usage:
    python ablation_comparison.py --exp-dir /home/singh.vishwa/xdash2/experiments
    python ablation_comparison.py --exp-dir experiments --task 4 --paradigm 2
    python ablation_comparison.py --exp-dir experiments --metric auc
"""

from __future__ import annotations

import argparse
import json
import glob
import os
import re
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

MODELS  = ["HMM", "CNN", "RNN", "TRANSFORMER"]
METHODS = ["variable_length", "truncate", "padding", "sliding_window",
           "phase_shift", "dtw_embedding"]
FREQS   = [50, 40, 30, 20, 10]

MODEL_COLORS  = {"HMM": "#E07B39", "CNN": "#2E86AB",
                 "RNN": "#4CAF50", "TRANSFORMER": "#9C27B0"}
MODEL_MARKERS = {"HMM": "o", "CNN": "s", "RNN": "^", "TRANSFORMER": "D"}

METRIC_LABELS = {
    "ba":        "Balanced Accuracy",
    "auc":       "AUC-ROC",
    "recall":    "Recall / Sensitivity",
    "precision": "Precision",
    "f1":        "F1 Score",
}

# experiment name prefixes → data_source label
# ABL_EW_  → event_window
# ABL_SBJ_ → subject
# ABL_     → subject (legacy, no DS tag)
def _infer_data_source(exp_name: str) -> str:
    if "ABL_EW_" in exp_name.upper():
        return "event_window"
    return "subject"


# =============================================================================
# 1. PARSING
# =============================================================================

def _extract_freq_from_tag(tag: str) -> Optional[int]:
    m = re.search(r"_f(\d+)", tag)
    return int(m.group(1)) if m else None


def load_ablation_summaries(
    exp_dir: str,
    task_filter: Optional[int] = None,
    paradigm_filter: Optional[int] = None,
    model_filter: Optional[str] = None,
    name_filter: Optional[str] = None,
) -> pd.DataFrame:
    pattern = os.path.join(exp_dir, "**", "summary.json")
    files   = glob.glob(pattern, recursive=True)
    print(f"[Parser] Found {len(files)} summary.json files in {exp_dir}")

    if name_filter:
        files = [f for f in files if name_filter in Path(f).parent.name]
        print(f"[Parser] After name filter '{name_filter}': {len(files)} files")

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
        freq     = cfg.get("freq", 50)
        exp_name = d.get("experiment_name", Path(fpath).parent.name)

        # recover freq from name if missing
        if freq is None or freq == 50:
            f_tag = _extract_freq_from_tag(exp_name)
            if f_tag is not None:
                freq = f_tag

        # infer data source from experiment name
        data_source = _infer_data_source(exp_name)

        if task_filter     and task     != task_filter:          continue
        if paradigm_filter and paradigm != paradigm_filter:      continue
        if model_filter    and model    != model_filter.upper(): continue

        ba  = metrics.get("ba")
        auc = metrics.get("auc")
        if ba is None:
            ba  = evl.get("balanced_accuracy")
            auc = evl.get("auc_roc")

        rows.append({
            "experiment_name": exp_name,
            "data_source":     data_source,
            "task":            task,
            "task_name":       TASK_NAMES.get(task, f"task{task}"),
            "paradigm":        paradigm,
            "paradigm_name":   PARADIGM_NAMES.get(paradigm, f"p{paradigm}"),
            "model":           model,
            "method":          method,
            "freq":            int(freq) if freq is not None else 50,
            "ba":              float(ba)  if ba  is not None else np.nan,
            "auc":             float(auc) if auc is not None else np.nan,
            "recall":          float(metrics.get("recall",    np.nan)),
            "precision":       float(metrics.get("precision", np.nan)),
            "f1":              float(metrics.get("f1",        np.nan)),
            "best_params":     json.dumps(bp),
            "best_params_dict": bp,
            "summary_path":    fpath,
        })

    if not rows:
        print("[Parser] WARNING: No rows parsed. Check --exp-dir path.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values(["data_source", "model", "method", "freq", "task", "paradigm"],
                   inplace=True)
    df.reset_index(drop=True, inplace=True)

    for src in ["subject", "event_window"]:
        sub = df[df["data_source"] == src]
        if not sub.empty:
            print(f"\n[{src}]  {len(sub)} rows")
            print(f"  Methods : {sorted(sub['method'].unique())}")
            print(f"  Freqs   : {sorted(sub['freq'].unique())}")
            print(f"  Models  : {sorted(sub['model'].unique())}")
    return df


# =============================================================================
# 2. PLOTTING HELPERS
# =============================================================================

def _finish_plot(fig, ax, title, xlabel, ylabel, save_path, legend_handles=None):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0.3, 1.02)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    chance  = Line2D([0], [0], color="grey", linestyle="--",
                     linewidth=0.8, alpha=0.5, label="Chance (0.5)")
    handles = (legend_handles or []) + [chance]
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

def plot_by_method(df: pd.DataFrame, metric: str, save_dir: Path,
                   scope: str, source_label: str):
    ylabel = METRIC_LABELS.get(metric, metric.upper())

    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        legend_handles = []

        for model in MODELS:
            msub = sub[sub["model"] == model]
            if msub.empty:
                continue

            grp    = msub.groupby("freq")[metric].agg(["mean", "std"]).reset_index()
            grp    = grp.sort_values("freq")
            color  = MODEL_COLORS.get(model, "#333333")
            marker = MODEL_MARKERS.get(model, "o")

            ax.plot(grp["freq"], grp["mean"], marker=marker, color=color,
                    linewidth=2, markersize=7)
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
            title=f"[{source_label}]  {method}  —  {ylabel} vs Frequency\n{scope}",
            xlabel="Sampling Frequency (Hz)",
            ylabel=ylabel,
            save_path=save_dir / "by_method" / f"{method}_freq_comparison.png",
            legend_handles=legend_handles,
        )


# =============================================================================
# 4. PLOT 2 — Per-frequency: BA vs Method
# =============================================================================

def plot_by_freq(df: pd.DataFrame, metric: str, save_dir: Path,
                 scope: str, source_label: str):
    ylabel = METRIC_LABELS.get(metric, metric.upper())

    for freq in sorted(df["freq"].unique()):
        sub = df[df["freq"] == freq]
        if sub.empty:
            continue

        methods_here = sorted(sub["method"].unique())
        x_pos = {m: i for i, m in enumerate(methods_here)}

        fig, ax = plt.subplots(figsize=(max(8, len(methods_here) * 1.8), 5))
        legend_handles = []

        for model in MODELS:
            msub = sub[sub["model"] == model]
            if msub.empty:
                continue

            grp    = msub.groupby("method")[metric].agg(["mean", "std"]).reset_index()
            xs     = [x_pos[m] for m in grp["method"]]
            color  = MODEL_COLORS.get(model, "#333333")
            marker = MODEL_MARKERS.get(model, "o")

            ax.errorbar(xs, grp["mean"].values,
                        yerr=grp["std"].fillna(0).values,
                        marker=marker, color=color,
                        linewidth=2, markersize=7, capsize=4)
            legend_handles.append(
                Line2D([0], [0], color=color, marker=marker,
                       linewidth=2, markersize=7, label=model)
            )

        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels(list(x_pos.keys()), rotation=20, ha="right", fontsize=9)

        _finish_plot(
            fig, ax,
            title=f"[{source_label}]  {freq} Hz  —  {ylabel} vs Method\n{scope}",
            xlabel="Preprocessing Method",
            ylabel=ylabel,
            save_path=save_dir / "by_freq" / f"freq{freq}hz_method_comparison.png",
            legend_handles=legend_handles,
        )


# =============================================================================
# 5. BEST HYPERPARAMETERS TABLE  (both data sources combined)
# =============================================================================

def build_best_hp_table(df: pd.DataFrame, metric: str, out_dir: Path):
    save_dir = out_dir / "tables"
    save_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for (data_source, method, freq, model), grp in df.groupby(
            ["data_source", "method", "freq", "model"]):
        if grp[metric].isna().all():
            continue

        best_row    = grp.loc[grp[metric].idxmax()]
        mean_metric = grp[metric].mean()
        std_metric  = grp[metric].std()

        try:
            bp = best_row["best_params_dict"]
            if not isinstance(bp, dict):
                bp = json.loads(best_row["best_params"])
        except Exception:
            bp = {}

        records.append({
            "data_source":      data_source,
            "method":           method,
            "freq_hz":          freq,
            "model":            model,
            f"mean_{metric}":   round(mean_metric, 4),
            f"std_{metric}":    round(std_metric, 4) if not np.isnan(std_metric) else 0.0,
            f"best_{metric}":   round(best_row[metric], 4),
            "best_task":        best_row["task"],
            "best_paradigm":    best_row["paradigm"],
            "best_params_json": json.dumps(bp),
            **{f"hp_{k}": v for k, v in bp.items()},
        })

    if not records:
        print("[HP Table] No records — nothing to save.")
        return pd.DataFrame()

    hp_df = pd.DataFrame(records)
    hp_df.sort_values(["data_source", "method", "freq_hz", "model"], inplace=True)
    hp_df.reset_index(drop=True, inplace=True)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = save_dir / "best_hyperparameters.csv"
    hp_df.to_csv(csv_path, index=False)
    print(f"  [Saved] {csv_path}")

    # ── TXT ───────────────────────────────────────────────────────────────────
    txt_path = save_dir / "best_hyperparameters.txt"
    lines = [
        "=" * 80,
        "BEST HYPERPARAMETERS PER (DATA SOURCE × METHOD × FREQ × MODEL)",
        f"Metric: {METRIC_LABELS.get(metric, metric.upper())}",
        "=" * 80,
    ]

    prev_src, prev_method = None, None
    for _, row in hp_df.iterrows():
        if row["data_source"] != prev_src:
            lines.append(f"\n{'█'*80}")
            lines.append(f"  DATA SOURCE: {row['data_source'].upper()}")
            lines.append(f"{'█'*80}")
            prev_src    = row["data_source"]
            prev_method = None

        if row["method"] != prev_method:
            lines.append(f"\n  {'─'*76}")
            lines.append(f"  METHOD: {row['method'].upper()}")
            lines.append(f"  {'─'*76}")
            prev_method = row["method"]

        lines.append(
            f"\n    {row['model']:<14} @ {row['freq_hz']:>2} Hz  │  "
            f"{METRIC_LABELS.get(metric, metric)} = "
            f"{row[f'mean_{metric}']:.4f} ± {row[f'std_{metric}']:.4f}  "
            f"(best: {row[f'best_{metric}']:.4f}  "
            f"T{row['best_task']} P{row['best_paradigm']})"
        )
        try:
            bp = json.loads(row["best_params_json"])
            for k, v in bp.items():
                lines.append(f"      {k:<24}: {v}")
        except Exception:
            lines.append(f"      {row['best_params_json']}")

    lines.append("\n" + "=" * 80)
    txt_path.write_text("\n".join(lines))
    print(f"  [Saved] {txt_path}")

    # console summary
    print("\n" + "=" * 80)
    print("BEST HYPERPARAMETERS SUMMARY")
    print("=" * 80)
    display_cols = ["data_source", "method", "freq_hz", "model",
                    f"mean_{metric}", f"std_{metric}", f"best_{metric}"]
    display_cols = [c for c in display_cols if c in hp_df.columns]
    print(hp_df[display_cols].to_string(index=False))

    return hp_df


# =============================================================================
# 6. DATA SAVE
# =============================================================================

def save_data(df: pd.DataFrame, out_dir: Path):
    save_dir = out_dir / "data"
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_df = df.drop(columns=["best_params_dict"], errors="ignore")

    # combined
    p = save_dir / "ablation_results.csv"
    csv_df.to_csv(p, index=False)
    print(f"  [Saved] {p}  ({len(df)} rows)")

    # per source
    for src in ["subject", "event_window"]:
        sub = csv_df[csv_df["data_source"] == src]
        if not sub.empty:
            p2 = save_dir / f"ablation_results_{src}.csv"
            sub.to_csv(p2, index=False)
            print(f"  [Saved] {p2}  ({len(sub)} rows)")


# =============================================================================
# 7. CLI
# =============================================================================

def _scope_label(task, paradigm):
    parts = []
    parts.append(f"Task {task} ({TASK_NAMES.get(task, '')})" if task else "All Tasks")
    parts.append(f"Paradigm {paradigm}" if paradigm else "All Paradigms")
    return " | ".join(parts)


def parse_args():
    p = argparse.ArgumentParser(
        description="Ablation comparison — subject vs event_window plots + HP tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--exp-dir",     default="experiments_abl",
                   help="Root experiment dir (default: experiments). "
                        "HPC: /home/singh.vishwa/xdash2/experiments")
    p.add_argument("--out-dir",     default="ablation_results")
    p.add_argument("--metric",      default="ba",
                   choices=["ba", "auc", "recall", "precision", "f1"])
    p.add_argument("--task",        type=int, default=None, choices=range(1, 7))
    p.add_argument("--paradigm",    type=int, default=None, choices=range(1, 5))
    p.add_argument("--model",       type=str, default=None,
                   choices=["hmm", "cnn", "rnn", "transformer"])
    p.add_argument("--name-filter", type=str, default="ABL_",
                   help="Only include experiments whose dir name contains this "
                        "(default: 'ABL_'). Pass '' for all.")
    return p.parse_args()


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    scope   = _scope_label(args.task, args.paradigm)

    print(f"\n{'='*70}")
    print("  Ablation Comparison")
    print(f"  Exp dir     : {args.exp_dir}")
    print(f"  Metric      : {args.metric} ({METRIC_LABELS.get(args.metric, '')})")
    print(f"  Name filter : '{args.name_filter}'")
    print(f"  Filters     : task={args.task}  paradigm={args.paradigm}  model={args.model}")
    print(f"  Output      : {out_dir}")
    print(f"{'='*70}\n")

    df = load_ablation_summaries(
        exp_dir=args.exp_dir,
        task_filter=args.task,
        paradigm_filter=args.paradigm,
        model_filter=args.model,
        name_filter=args.name_filter if args.name_filter else None,
    )
    if df.empty:
        print("No data found — exiting.")
        return

    # ── Save raw data ─────────────────────────────────────────────────────────
    print("\n[Data] Saving CSVs...")
    save_data(df, out_dir)

    # ── Plots — split by data source ─────────────────────────────────────────
    SOURCE_LABELS = {"subject": "Subject-level", "event_window": "Event Window"}

    for src in ["subject", "event_window"]:
        sub = df[df["data_source"] == src]
        if sub.empty:
            print(f"\n[Plots] No {src} data — skipping plots.")
            continue

        plot_dir = out_dir / "plots" / src
        label    = SOURCE_LABELS[src]

        print(f"\n[Plots: {label}] by_method...")
        plot_by_method(sub, args.metric, plot_dir, scope, label)

        print(f"[Plots: {label}] by_freq...")
        plot_by_freq(sub, args.metric, plot_dir, scope, label)

    # ── HP table — both sources combined ─────────────────────────────────────
    print("\n[Tables] Building best hyperparameter table (both sources)...")
    build_best_hp_table(df, args.metric, out_dir)

    print(f"\n{'='*70}")
    print(f"  Done.  Outputs in: {out_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()