"""
save_emission_importance.py
============================
Computes and persists emission-matrix-derived channel importance for each
class-conditional HMM/HSMM model, per task x paradigm — "state-averaged
emission importance", epsilon_d = (1/K) * sum_u delta~(u,d): the unweighted
average, across a model's K hidden states, of each channel's per-state
normalised importance delta~(u,d) (itself: |mean shift from cross-state
average| * cross-state range, normalised to sum to 1 within each state).
Because every delta~(u,*) already sums to 1, epsilon_d also sums to 1 across
the 18 channels.

epsilon_d is NOT the same as compute_state_specific_importance()'s own
returned `global_importance` (gamma~_d, based on cross-state RANGE alone,
never averaged over states) -- that quantity is computed here too, for
reference/diagnostic purposes, but epsilon_d is the one written to the
"global_importance" column for downstream compatibility (see below), since
epsilon_d is what the paper reports as "state-averaged emission importance".

Note on non-independence (documented, not fixed here): gamma_d enters every
state's delta(u,d) as a common multiplicative factor, so a single state with
an outlying emission mean elevates that channel's delta~(u,d) -- and hence
epsilon_d -- across the ENTIRE state repertoire, not just in the state where
it's actually extreme. Per-state and state-averaged importance are therefore
not independent; interpret rankings with that caveat.

This is a read-only, post-hoc analysis over an EXISTING checkpoint's
hyperparameters (best_params) and the ORIGINAL training data — it refits
each class-conditional model on the full group (no CV) via fit_for_analysis,
exactly as pipeline/runner.py's diagnostics stage and generate_all_state_seqs.py
already do, then calls compute_state_specific_importance() on each fitted
model. It does not change any checkpoint, results, or figures.

Output (per task x paradigm x class):
    storage/results/<dataset>/hmm/emission_importance/
        emission_importance_T{t}_P{p}_control.csv
        emission_importance_T{t}_P{p}_patient.csv

Each CSV has one row per channel with columns:
    feature, global_importance, state_averaged_importance,
    range_based_importance, state0_importance, state1_importance, ...
"global_importance" == "state_averaged_importance" == epsilon_d (kept as
"global_importance" too so generate_hmm_analysis_table.py's --emission-csv
consumer, which checks for that exact column name, works unchanged).
"range_based_importance" is gamma~_d, included for reference only -- it is
NOT epsilon_d and is not used by any downstream consumer.

Usage:
    python scripts/hmm/save_emission_importance.py --task 1 --paradigm 2
    python scripts/hmm/save_emission_importance.py --model hsmm --task 4 --paradigm 2
    python scripts/hmm/save_emission_importance.py --all
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def setup_project_path() -> bool:
    script_dir = Path(__file__).resolve().parent
    for root in (Path.cwd(), script_dir, script_dir.parent, script_dir.parent.parent):
        if (root / "models" / "hmm_model.py").exists():
            sys.path.insert(0, str(root))
            return True
    return False


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute + save state-averaged emission importance "
                    "(gamma~_d) and per-state emission importance per T x P")
    p.add_argument("--dataset", default="xdash")
    p.add_argument("--model", choices=["hmm", "hsmm"], default="hmm")
    p.add_argument("--task", type=int, choices=range(1, 7))
    p.add_argument("--paradigm", type=int, choices=range(1, 5))
    p.add_argument("--all", action="store_true",
                   help="Process all 24 task x paradigm combinations "
                        "(skips combos with no checkpoint found)")
    p.add_argument("--hmm-dir", default=None,
                   help="Root of experiment folders "
                        "(default: storage/results/<dataset>/experiments)")
    p.add_argument("--out", default=None,
                   help="Output directory "
                        "(default: storage/results/<dataset>/hmm/emission_importance)")
    return p.parse_args()


def find_checkpoint(hmm_dir: Path, task: int, paradigm: int, model: str) -> Path | None:
    prefix = model.upper()
    patterns = [
        str(hmm_dir / f"task{task}" / f"paradigm{paradigm}" / f"{prefix}*"
            / "model_checkpoints" / f"{prefix}_T{task}_P{paradigm}_BA*.json"),
        str(hmm_dir / "**" / f"{prefix}_T{task}_P{paradigm}_BA*.json"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return Path(matches[-1])
    return None


def load_xy(task: int, paradigm: int, dataset: str, model: str):
    from pipeline.io import load_data
    from dataio.paradigms import ParadigmSelector
    from dataio.preprocessors import PreprocessorFactory
    from dataio.ingestion import load_dataset_config

    dataset_config = load_dataset_config(dataset)
    patient_data, control_data = load_data(task, dataset, dataset_config)
    selector = ParadigmSelector(dataset_config)
    g1, g0 = selector.select_paradigm(patient_data, control_data, paradigm=paradigm)

    sampling_rate = dataset_config.get("sampling_rate", 50)
    preprocessor = PreprocessorFactory.create(
        method="variable_length", model_type=model,
        resample_rate=sampling_rate, original_rate=sampling_rate,
    )
    X, y, sids = preprocessor.prepare_data(g1, g0)
    return X, y, dataset_config


def run_one(task: int, paradigm: int, hmm_dir: Path, out_dir: Path,
            dataset: str, model: str):
    tag = f"T{task}_P{paradigm}"
    ckpt_path = find_checkpoint(hmm_dir, task, paradigm, model)
    if ckpt_path is None:
        print(f"[SKIP] {tag} — no {model.upper()} checkpoint found in {hmm_dir}")
        return False

    import json
    ckpt = json.load(open(ckpt_path))
    hp = ckpt["hyperparameters"]
    print(f"\n{'='*60}\n  {model.upper()} {tag}  |  checkpoint: {ckpt_path.name}\n"
          f"  hyperparameters: {hp}\n{'='*60}")

    X, y, dataset_config = load_xy(task, paradigm, dataset, model)
    channel_names = dataset_config.get("channels", [])

    if model == "hsmm":
        from models.hsmm_model import HSMMModel as ModelCls
    else:
        from models.hmm_model import HMMModel as ModelCls
    model_obj = ModelCls(task=task, paradigm=paradigm)

    fit_kwargs = dict(
        X=X, y=y,
        n_components=hp["n_components"],
        covariance_type=hp["covariance_type"],
        n_iter=hp["n_iter"],
    )
    if model == "hsmm":
        fit_kwargs["max_duration"] = hp.get("max_duration", 200)
    model_obj.fit_for_analysis(**fit_kwargs)

    out_dir.mkdir(parents=True, exist_ok=True)
    for label, fitted in [("control", model_obj._get_fitted(0)),
                          ("patient", model_obj._get_fitted(1))]:
        range_imp, state_imp = model_obj.compute_state_specific_importance(
            model=fitted, channel_names=channel_names
        )
        n_states = len(state_imp)

        # epsilon_d = (1/K) * sum_u delta~(u,d) -- the true state-averaged
        # importance. NOT range_imp (gamma~_d), which compute_state_specific_
        # importance returns based on cross-state range alone and never
        # averages over states.
        eps = {
            ch: float(np.mean([state_imp[s][ch] for s in range(n_states)]))
            for ch in channel_names
        }
        eps_sum = sum(eps.values())
        eps = {ch: v / eps_sum for ch, v in eps.items()}

        rows = []
        for ch in channel_names:
            row = {
                "feature": ch,
                "global_importance": eps[ch],          # == epsilon_d (see module docstring)
                "state_averaged_importance": eps[ch],   # explicit alias, same values
                "range_based_importance": range_imp[ch],  # gamma~_d, reference only
            }
            for s in range(n_states):
                row[f"state{s}_importance"] = state_imp[s][ch]
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("global_importance", ascending=False)

        out_path = out_dir / f"emission_importance_{tag}_{label}.csv"
        df.to_csv(out_path, index=False)
        print(f"  [{label}] wrote {out_path}  "
              f"(top channel: {df.iloc[0]['feature']} = {df.iloc[0]['global_importance']:.4f})")

    return True


def main():
    if not setup_project_path():
        print("[ERROR] Could not locate project root (models/hmm_model.py not found).")
        sys.exit(1)

    args = parse_args()
    if not args.all and (args.task is None or args.paradigm is None):
        print("[ERROR] Provide --task and --paradigm, or use --all")
        sys.exit(1)

    from config.paths import get_experiments_dir, get_results_dir
    hmm_dir = Path(args.hmm_dir) if args.hmm_dir else get_experiments_dir(args.dataset)
    out_dir = (Path(args.out) if args.out
               else get_results_dir(args.dataset) / "hmm" / "emission_importance")

    combos = ([(t, p) for t in range(1, 7) for p in range(1, 5)]
              if args.all else [(args.task, args.paradigm)])

    done, skipped = [], []
    for task, paradigm in combos:
        try:
            ok = run_one(task, paradigm, hmm_dir, out_dir, args.dataset, args.model)
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
