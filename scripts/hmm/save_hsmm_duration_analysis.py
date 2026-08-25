"""
save_hsmm_duration_analysis.py
===============================
Computes and persists the HSMM's class-conditional duration-distribution
results per task x paradigm: the Poisson rate lambda_u (mean sojourn
duration, in seconds) per hidden state u, for both the control- and
patient/RCT-conditional models. This is the "Table 2"-style result
described in the paper's HSMM duration analysis section.

lambda_u is estimated post-hoc from the mean run length of Viterbi-decoded
state sequences (GaussianHSMM._estimate_duration_rates, models/hsmm_model.py),
computed here by refitting each class-conditional model on the full group
(no CV) via fit_for_analysis, at the checkpoint's already-known best
hyperparameters -- exactly the same read-only, post-hoc pattern used by
save_emission_importance.py and generate_all_state_seqs.py. It does not
change any checkpoint, results, or figures.

For each state, also reports:
  - empirical occupancy: fraction of all decoded training frames assigned
    to that state (NOT the transition matrix's theoretical stationary
    distribution, which is degenerate -- collapses to ~100% on any
    quasi-absorbing state, i.e. any state with a self-transition
    probability close to 1. Empirical, finite-trial occupancy is the
    correct quantity for interpreting real ~1-2 minute trials.)
  - outlier flag: True if this state's dwell time exceeds every other
    state's dwell time in the SAME class-conditional model by a factor
    of >= 2.5 (the criterion used for the bolded entries in Table 2).

Output (per task x paradigm):
    storage/results/<dataset>/hsmm/duration_analysis/
        duration_analysis_T{t}_P{p}.csv

Columns: task, paradigm, class (control/patient), state, lambda_s,
         empirical_occupancy, is_outlier

Usage:
    python scripts/hmm/save_hsmm_duration_analysis.py --task 1 --paradigm 2
    python scripts/hmm/save_hsmm_duration_analysis.py --all
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def setup_project_path() -> bool:
    script_dir = Path(__file__).resolve().parent
    for root in (Path.cwd(), script_dir, script_dir.parent, script_dir.parent.parent):
        if (root / "models" / "hsmm_model.py").exists():
            sys.path.insert(0, str(root))
            return True
    return False


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute + save HSMM class-conditional duration distributions "
                    "(Poisson lambda_u per state) per T x P")
    p.add_argument("--dataset", default="xdash")
    p.add_argument("--task", type=int, choices=range(1, 7))
    p.add_argument("--paradigm", type=int, choices=range(1, 5))
    p.add_argument("--all", action="store_true",
                   help="Process all 24 task x paradigm combinations "
                        "(skips combos with no HSMM checkpoint found)")
    p.add_argument("--hmm-dir", default=None,
                   help="Root of experiment folders "
                        "(default: storage/results/<dataset>/experiments)")
    p.add_argument("--out", default=None,
                   help="Output directory "
                        "(default: storage/results/<dataset>/hsmm/duration_analysis)")
    p.add_argument("--outlier-factor", type=float, default=2.5,
                   help="Flag a state as an outlier if its dwell time exceeds "
                        "every other state's by at least this factor (default: 2.5)")
    return p.parse_args()


def find_checkpoint(hmm_dir: Path, task: int, paradigm: int) -> Path | None:
    patterns = [
        str(hmm_dir / f"task{task}" / f"paradigm{paradigm}" / "HSMM*"
            / "model_checkpoints" / f"HSMM_T{task}_P{paradigm}_BA*.json"),
        str(hmm_dir / "**" / f"HSMM_T{task}_P{paradigm}_BA*.json"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return Path(matches[-1])
    return None


def load_xy(task: int, paradigm: int, dataset: str, sampling_rate: int):
    from pipeline.io import load_data
    from dataio.paradigms import ParadigmSelector
    from dataio.preprocessors import PreprocessorFactory

    from dataio.ingestion import load_dataset_config
    dataset_config = load_dataset_config(dataset)
    patient_data, control_data = load_data(task, dataset, dataset_config)
    selector = ParadigmSelector(dataset_config)
    g1, g0 = selector.select_paradigm(patient_data, control_data, paradigm=paradigm)

    preprocessor = PreprocessorFactory.create(
        method="variable_length", model_type="hsmm",
        resample_rate=sampling_rate, original_rate=sampling_rate,
    )
    X, y, sids = preprocessor.prepare_data(g1, g0)
    return X, y


def empirical_occupancy(fitted, seqs) -> np.ndarray:
    """Fraction of decoded frames (geometric Viterbi, same path used
    internally by _estimate_duration_rates) assigned to each state,
    across all sequences in `seqs`."""
    n_states = fitted.n_components
    counts = np.zeros(n_states)
    total = 0
    for seq in seqs:
        states = fitted.predict(seq)
        total += len(states)
        for s in range(n_states):
            counts[s] += int((states == s).sum())
    return counts / total


def run_one(task: int, paradigm: int, hmm_dir: Path, out_dir: Path,
            dataset: str, outlier_factor: float):
    tag = f"T{task}_P{paradigm}"
    ckpt_path = find_checkpoint(hmm_dir, task, paradigm)
    if ckpt_path is None:
        print(f"[SKIP] {tag} — no HSMM checkpoint found in {hmm_dir}")
        return False

    ckpt = json.load(open(ckpt_path))
    hp = ckpt["hyperparameters"]
    print(f"\n{'='*60}\n  HSMM {tag}  |  checkpoint: {ckpt_path.name}\n"
          f"  hyperparameters: {hp}\n{'='*60}")

    from dataio.ingestion import load_dataset_config
    from models.hsmm_model import HSMMModel

    dataset_config = load_dataset_config(dataset)
    sampling_rate = dataset_config.get("sampling_rate", 50)
    X, y = load_xy(task, paradigm, dataset, sampling_rate)

    model = HSMMModel(task=task, paradigm=paradigm)
    model.fit_for_analysis(
        X=X, y=y,
        n_components=hp["n_components"], covariance_type=hp["covariance_type"],
        n_iter=hp["n_iter"], max_duration=hp.get("max_duration", 200),
    )

    seqs_0 = [X[i] for i in range(len(X)) if y[i] == 0]
    seqs_1 = [X[i] for i in range(len(X)) if y[i] == 1]

    rows = []
    for label, fitted, seqs in [("control", model.fitted_hsmm0, seqs_0),
                                ("patient", model.fitted_hsmm1, seqs_1)]:
        lambdas_s = fitted._duration_rates / sampling_rate
        occ = empirical_occupancy(fitted, seqs)
        n_states = len(lambdas_s)

        for s in range(n_states):
            others = np.delete(lambdas_s, s)
            is_outlier = bool(others.size and lambdas_s[s] >= outlier_factor * others.max())
            rows.append({
                "task": task, "paradigm": paradigm, "class": label, "state": s,
                "lambda_s": round(float(lambdas_s[s]), 3),
                "empirical_occupancy": round(float(occ[s]), 4),
                "is_outlier": is_outlier,
            })
        top = lambdas_s.argmax()
        print(f"  [{label}] lambda (s): {[round(x,2) for x in lambdas_s]}  "
              f"occupancy: {[round(x,3) for x in occ]}  "
              f"outlier state: {top if rows[-n_states+top]['is_outlier'] else 'none'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"duration_analysis_{tag}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  wrote {out_path}")
    return True


def main():
    if not setup_project_path():
        print("[ERROR] Could not locate project root (models/hsmm_model.py not found).")
        sys.exit(1)

    args = parse_args()
    if not args.all and (args.task is None or args.paradigm is None):
        print("[ERROR] Provide --task and --paradigm, or use --all")
        sys.exit(1)

    from config.paths import get_experiments_dir, get_results_dir
    hmm_dir = Path(args.hmm_dir) if args.hmm_dir else get_experiments_dir(args.dataset)
    out_dir = (Path(args.out) if args.out
               else get_results_dir(args.dataset) / "hsmm" / "duration_analysis")

    combos = ([(t, p) for t in range(1, 7) for p in range(1, 5)]
              if args.all else [(args.task, args.paradigm)])

    done, skipped = [], []
    for task, paradigm in combos:
        try:
            ok = run_one(task, paradigm, hmm_dir, out_dir, args.dataset, args.outlier_factor)
            (done if ok else skipped).append(f"T{task}_P{paradigm}")
        except Exception as e:
            import traceback
            print(f"[ERROR] T{task}_P{paradigm} failed: {e}")
            traceback.print_exc()
            skipped.append(f"T{task}_P{paradigm}")

    print(f"\n{'='*60}\nDone. {len(done)} combo(s) written -> {out_dir}/")
    print(f"Skipped/failed: {skipped}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
