"""
extract_hpc_results.py
======================
Reads every summary.json under experiments_from_hpc/ and produces a tidy
results table with metrics, diagnostics, and per-feature importance.

Output (written to nn-models-results/):
    hpc_results.csv     — one row per experiment
    hpc_results.xlsx    — same, with auto-column widths

Usage:
    python scripts/extract_hpc_results.py
    python scripts/extract_hpc_results.py --exp-dir /path/to/experiments_from_hpc
    python scripts/extract_hpc_results.py --out-dir results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FEATURE_COLS = [
    "head_pos_x", "head_pos_y", "head_pos_z",
    "head_rot_x", "head_rot_y", "head_rot_z",
    "left_hand_pos_x", "left_hand_pos_y", "left_hand_pos_z",
    "left_hand_rot_x", "left_hand_rot_y", "left_hand_rot_z",
    "right_hand_pos_x", "right_hand_pos_y", "right_hand_pos_z",
    "right_hand_rot_x", "right_hand_rot_y", "right_hand_rot_z",
]


def extract_row(summary_path: Path) -> dict | None:
    try:
        with open(summary_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [SKIP] {summary_path}: {e}")
        return None

    results = data.get("results", {})
    metrics = results.get("metrics", {})
    diag    = data.get("diagnostics", {})
    fi      = results.get("feature_importance", {}) or {}

    row = {
        # Identity
        "experiment":           data.get("experiment_name", summary_path.parent.name),
        "task":                 results.get("task",          data.get("config", {}).get("task")),
        "task_name":            results.get("task_name",     ""),
        "paradigm":             results.get("paradigm",      data.get("config", {}).get("paradigm")),
        "paradigm_name":        results.get("paradigm_name", ""),
        "model":                results.get("model",         data.get("config", {}).get("model", "")).upper(),
        "preprocessing_method": results.get("preprocessing_method",
                                            data.get("config", {}).get("method", "")),
        # Core metrics
        "ba":           metrics.get("ba"),
        "auc":          metrics.get("auc"),
        "auc_ci_low":   metrics.get("auc_ci_low"),
        "auc_ci_high":  metrics.get("auc_ci_high"),
        "recall":       metrics.get("recall"),
        "precision":    metrics.get("precision"),
        "f1":           metrics.get("f1"),
        "accuracy":     data.get("evaluation", {}).get("accuracy"),
        # Diagnostics
        "overfitting_risk":   diag.get("overfitting_risk"),
        "generalization_gap": diag.get("generalization_gap"),
    }

    # Feature importance — one column per feature
    for feat in FEATURE_COLS:
        row[f"fi_{feat}"] = fi.get(feat)

    # Top feature by importance
    if fi:
        row["top_feature"] = max(fi, key=fi.get)
    else:
        row["top_feature"] = None

    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-dir", default="results/experiments_from_hpc",
                        help="Root directory of HPC experiments (default: results/experiments_from_hpc)")
    parser.add_argument("--out-dir", default="results/nn_models",
                        help="Output directory (default: results/nn_models)")
    args = parser.parse_args()

    exp_root = Path(args.exp_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_files = sorted(exp_root.glob("task*/paradigm*/*/summary.json"))
    print(f"Found {len(summary_files)} summary.json files under {exp_root}/\n")

    rows = []
    for sf in summary_files:
        row = extract_row(sf)
        if row:
            rows.append(row)
            print(f"  OK  task={row['task']}  paradigm={row['paradigm']}"
                  f"  model={row['model']:<12}  ba={row['ba']}  auc={row['auc']}")

    if not rows:
        print("No results extracted.")
        return

    df = pd.DataFrame(rows)
    df.sort_values(["task", "paradigm", "model"], inplace=True, ignore_index=True)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = out_dir / "hpc_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[CSV]  {len(df)} rows  →  {csv_path}")

    # ── Excel (auto column widths) ────────────────────────────────────────────
    xlsx_path = out_dir / "hpc_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
        ws = writer.sheets["Results"]
        for col_cells in ws.columns:
            max_len = max(
                len(str(c.value)) if c.value is not None else 0
                for c in col_cells
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 40)
    print(f"[XLSX] {len(df)} rows  →  {xlsx_path}")

    # ── Console summary table ─────────────────────────────────────────────────
    cols = ["task", "task_name", "paradigm", "paradigm_name",
            "model", "preprocessing_method",
            "ba", "auc", "auc_ci_low", "auc_ci_high",
            "recall", "precision", "f1", "accuracy",
            "overfitting_risk", "generalization_gap", "top_feature"]
    print(f"\n{'='*120}")
    print("  RESULTS SUMMARY")
    print(f"{'='*120}")
    print(df[cols].to_string(index=False))
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()
