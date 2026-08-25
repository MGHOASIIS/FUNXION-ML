"""
extract_best_params.py
----------------------
Walks experiments_from_hpc/task{i}/paradigm{j}/{MODEL}_*/summary.json
and extracts results.best_params for each task × paradigm combination.

Output: nn-models-results/transformer_best_params.csv

Usage:
    python extract_best_params.py
    python extract_best_params.py --base path/to/experiments_from_hpc
    python extract_best_params.py --model ALL   # CNN | RNN | TRANSFORMER | ALL
    python extract_best_params.py --out my_params.csv
"""

import argparse
import csv
import glob
import json
from pathlib import Path

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--base",  default="results/experiments_from_hpc")
parser.add_argument("--model", default="TRANSFORMER",
                    help="CNN | RNN | TRANSFORMER | ALL")
parser.add_argument("--out",   default="results/nn_models/transformer_best_params.csv")
args = parser.parse_args()

BASE = Path(args.base)
if not BASE.exists():
    raise FileNotFoundError(f"Not found: {BASE.resolve()}")

TASK_NAMES = {
    1: "jar_opening",  2: "key_turning",  3: "cleaning",
    4: "back_washing", 5: "cutting",      6: "hammering",
}
PARADIGM_NAMES = {
    1: "patients_vs_controls",         2: "rct_vs_controls",
    3: "other_conditions_vs_controls", 4: "rct_vs_other_conditions",
}

models = (
    ["CNN", "RNN", "TRANSFORMER"]
    if args.model.upper() == "ALL"
    else [args.model.upper()]
)

# ── Walk ──────────────────────────────────────────────────────────────────────
records = []
missing = []

for task_id in range(1, 7):
    for par_id in range(1, 5):
        for model in models:
            pattern  = str(BASE / f"task{task_id}" / f"paradigm{par_id}" / f"{model}_*")
            exp_dirs = [Path(p) for p in glob.glob(pattern) if Path(p).is_dir()]

            if not exp_dirs:
                missing.append((task_id, par_id, model))
                print(f"[MISSING]    T{task_id} P{par_id} {model}")
                continue

            # most recent run wins
            exp_dir = max(exp_dirs, key=lambda p: p.stat().st_mtime)

            summary_path = exp_dir / "summary.json"
            if not summary_path.exists():
                missing.append((task_id, par_id, model))
                print(f"[NO SUMMARY] T{task_id} P{par_id} {model} → {exp_dir.name}")
                continue

            try:
                with open(summary_path) as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                missing.append((task_id, par_id, model))
                print(f"[JSON ERROR] {summary_path}: {e}")
                continue

            # ── Extract ───────────────────────────────────────────────────
            results     = data.get("results", {})
            metrics     = results.get("metrics", {})
            best_params = results.get("best_params", {})
            diagnostics = data.get("diagnostics", {})

            record = {
                "experiment":         data.get("experiment_name", exp_dir.name),
                "task":               task_id,
                "task_name":          TASK_NAMES[task_id],
                "paradigm":           par_id,
                "paradigm_name":      PARADIGM_NAMES[par_id],
                "model":              model,
                "ba":                 metrics.get("ba"),
                "recall":             metrics.get("recall"),
                "precision":          metrics.get("precision"),
                "f1":                 metrics.get("f1"),
                "auc":                metrics.get("auc"),
                "auc_ci_low":         metrics.get("auc_ci_low"),
                "auc_ci_high":        metrics.get("auc_ci_high"),
                "overfitting_risk":   diagnostics.get("overfitting_risk"),
                "generalization_gap": diagnostics.get("generalization_gap"),
            }

            # flatten best_params → param_* columns
            for k, v in best_params.items():
                record[f"param_{k}"] = v

            records.append(record)
            print(f"[OK] T{task_id} P{par_id} {model:12} → {exp_dir.name}  "
                  f"BA={metrics.get('ba')}  best_params={best_params}")

# ── Write CSV ─────────────────────────────────────────────────────────────────
if not records:
    print("\n[ERROR] No records found. Check --base path.")
else:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_keys = list(dict.fromkeys(k for r in records for k in r.keys()))

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in all_keys})

    print(f"\n✓ {len(records)} rows → {out_path.resolve()}")

if missing:
    print(f"\n[MISSING] {len(missing)} combinations:")
    for t, p, m in missing:
        print(f"  T{t} ({TASK_NAMES[t]}) | P{p} ({PARADIGM_NAMES[p]}) | {m}")