"""
scrape_results.py
-----------------
Scrapes experiment results for HMM, CNN, RNN, and Transformer models
across all tasks (1-6) and paradigms (1-4).

HMM note: uses variable_length preprocessing (full-length sequences).
          X_shape may be absent or a list — handled gracefully.

Expected directory structure:
    experiments_from_hpc/
        task{1-6}/
            paradigm{1-4}/
                HMM*/
                CNN*/
                RNN*/
                TRANSFORMER*/
                    diagnostics/reports/comprehensive_analysis.json
                    results/results_T{task}_P{paradigm}_{model}*.json
                    summary.json

Output:
    results_summary.csv   — one row per experiment (flat metrics)
    results_summary.json  — full structured data
"""

import json
import glob
import os
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path("experiments_from_hpc")
MODELS = ["CNN", "RNN", "TRANSFORMER"]
TASKS = range(1, 7)
PARADIGMS = range(1, 5)

TASK_NAMES = {
    1: "jar_opening", 2: "key_turning", 3: "cleaning",
    4: "back_washing", 5: "cutting", 6: "hammering"
}
PARADIGM_NAMES = {
    1: "patients_vs_controls", 2: "rct_vs_controls",
    3: "other_conditions_vs_controls", 4: "rct_vs_other_conditions"
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | None:
    """Load a JSON file, return None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


def find_experiment_dirs(base: Path, task: int, paradigm: int, model: str) -> list[Path]:
    """Glob for model experiment directories (handles timestamps in folder name)."""
    pattern = str(base / f"task{task}" / f"paradigm{paradigm}" / f"{model}*")
    return [Path(p) for p in glob.glob(pattern) if Path(p).is_dir()]


def extract_summary(exp_dir: Path) -> dict | None:
    return load_json(exp_dir / "summary.json")


def extract_comprehensive(exp_dir: Path) -> dict | None:
    return load_json(exp_dir / "diagnostics" / "reports" / "comprehensive_analysis.json")


def extract_results_json(exp_dir: Path, task: int, paradigm: int, model: str) -> dict | None:
    pattern = str(exp_dir / "results" / f"results_T{task}_P{paradigm}_{model}*.json")
    matches = glob.glob(pattern)
    if not matches:
        # Try case-insensitive fallback
        pattern2 = str(exp_dir / "results" / f"results_T{task}_P{paradigm}_*")
        matches = [m for m in glob.glob(pattern2)
                   if model.lower() in Path(m).name.lower()]
    return load_json(Path(matches[0])) if matches else None

# ── Main scraper ──────────────────────────────────────────────────────────────

def scrape_all() -> list[dict]:
    records = []

    for task in TASKS:
        for paradigm in PARADIGMS:
            for model in MODELS:
                exp_dirs = find_experiment_dirs(BASE_DIR, task, paradigm, model)

                if not exp_dirs:
                    print(f"[MISSING] T{task} P{paradigm} {model} — no directory found")
                    continue

                # If multiple runs exist (re-runs), take the most recently modified
                exp_dir = max(exp_dirs, key=lambda p: p.stat().st_mtime)
                print(f"[FOUND]   T{task} P{paradigm} {model} → {exp_dir.name}")

                # ── Load data sources ─────────────────────────────────────
                summary      = extract_summary(exp_dir)
                comprehensive = extract_comprehensive(exp_dir)
                results_data  = extract_results_json(exp_dir, task, paradigm, model)

                # Prefer summary.json as primary source (contains everything)
                primary = summary or results_data
                if primary is None:
                    print(f"  [SKIP] No usable JSON found in {exp_dir}")
                    continue

                # ── Resolve nested keys ───────────────────────────────────
                res     = primary.get("results", primary)        # summary wraps in "results"
                metrics = res.get("metrics", {})
                diag    = primary.get("diagnostics", {})
                evalu   = primary.get("evaluation", {})
                config  = primary.get("config", {})

                # If comprehensive_analysis.json has extra fields, merge them
                comp_extra = {}
                if comprehensive:
                    comp_extra = {
                        "comp_overfitting_risk":    comprehensive.get("overfitting_risk"),
                        "comp_generalization_gap":  comprehensive.get("generalization_gap"),
                        # add any other fields from comprehensive_analysis here
                    }

                record = {
                    # ── Identity ──────────────────────────────────────────
                    "experiment_name":  primary.get("experiment_name", exp_dir.name),
                    "task":             task,
                    "task_name":        TASK_NAMES[task],
                    "paradigm":         paradigm,
                    "paradigm_name":    PARADIGM_NAMES[paradigm],
                    "model":            model,
                    "preprocessing":    res.get("preprocessing_method")
                                        or config.get("method"),

                    # ── Core metrics ──────────────────────────────────────
                    "ba":               metrics.get("ba"),
                    "recall":           metrics.get("recall"),
                    "precision":        metrics.get("precision"),
                    "f1":               metrics.get("f1"),
                    "auc":              metrics.get("auc"),
                    "auc_ci_low":       metrics.get("auc_ci_low"),
                    "auc_ci_high":      metrics.get("auc_ci_high"),

                    # ── Evaluation block (from summary) ───────────────────
                    "accuracy":         evalu.get("accuracy"),
                    "balanced_accuracy":evalu.get("balanced_accuracy"),
                    "auc_roc":          evalu.get("auc_roc"),
                    "auc_roc_ci_low":   (evalu.get("auc_roc_ci") or [None, None])[0],
                    "auc_roc_ci_high":  (evalu.get("auc_roc_ci") or [None, None])[1],

                    # ── Overfitting diagnostics ───────────────────────────
                    "overfitting_risk":     diag.get("overfitting_risk"),
                    "generalization_gap":   diag.get("generalization_gap"),

                    # ── Sample info ───────────────────────────────────────
                    # HMM: variable-length sequences — X_shape handled gracefully
                    "n_samples":        (res.get("X_shape") or [None])[0],
                    "n_features":       18 if res.get("model","").upper() == "HMM"
                                        else ((res.get("X_shape") or [None, None])[1]
                                              if res.get("X_shape") and len(res["X_shape"]) > 1 else None),
                    "seq_length":       "variable" if res.get("model","").upper() == "HMM"
                                        else ((res.get("X_shape") or [None, None, None])[2]
                                              if res.get("X_shape") and len(res["X_shape"]) > 2 else None),

                    # ── Top features (top 3 by importance) ───────────────
                    **{f"top_feature_{i+1}": feat
                       for i, feat in enumerate(
                           sorted(res.get("feature_importance", {}).items(),
                                  key=lambda x: x[1], reverse=True)[:3]
                       )},   # yields top_feature_1..3 as (name, value) tuples

                    # ── Merge comprehensive extras ────────────────────────
                    **comp_extra,

                    # ── Source path ───────────────────────────────────────
                    "source_dir": str(exp_dir),
                }

                # Flatten top_feature tuples → separate name/score columns
                for i in range(1, 4):
                    key = f"top_feature_{i}"
                    val = record.get(key)
                    if isinstance(val, tuple):
                        record[f"top_feature_{i}_name"]  = val[0]
                        record[f"top_feature_{i}_score"] = val[1]
                    else:
                        record[f"top_feature_{i}_name"]  = None
                        record[f"top_feature_{i}_score"] = None
                    record.pop(key, None)

                records.append(record)

    return records


# ── Run & save ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nScraping experiments under: {BASE_DIR.resolve()}\n")
    records = scrape_all()

    if not records:
        print("\n[ERROR] No records collected. Check that BASE_DIR is correct.")
    else:
        df = pd.DataFrame(records)

        # Sort for readability
        df.sort_values(["model", "task", "paradigm"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # ── Save CSV ──────────────────────────────────────────────────────
        csv_path = "nn-models-results/results_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ CSV  saved → {csv_path}  ({len(df)} rows)")

        # ── Save JSON ─────────────────────────────────────────────────────
        json_path = "nn-models-results/results_summary.json"
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        print(f"✓ JSON saved → {json_path}")

        # ── Quick summary table ───────────────────────────────────────────
        print("\n── Per-model mean metrics ──────────────────────────────────")
        numeric_cols = ["ba", "auc", "f1", "recall", "generalization_gap"]
        available_num = [c for c in numeric_cols if c in df.columns]
        num_df = df[["model"] + available_num].copy()
        for col in available_num:
            num_df[col] = pd.to_numeric(num_df[col], errors="coerce")
        print(num_df.groupby("model")[available_num].mean().round(3).to_string())

        print("\n── Overfitting risk counts ─────────────────────────────────")
        if "overfitting_risk" in df.columns:
            print(df.groupby(["model", "overfitting_risk"]).size().to_string())