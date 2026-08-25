"""
run_hsmm_staged.py
===================
Why this script exists
-----------------------
HSMMModel.train_and_evaluate() (models/hsmm_model.py) needs ~19 independent
full LOO-CV passes to finish: one for the (single, non-searched) hyperparameter
combo, one baseline pass for feature importance, and one more per permuted
channel (18 channels). Each pass is expensive because HSMM's explicit-duration
score()/predict() are O(T * max_duration * n_components) pure-Python nested
loops — nothing like the C-optimized geometric HMM. A single pass for T4/P2 at
n_components=8 took 6-14 hours; running the whole set as one job (even
8-way-parallel across channels) meant ~50h of real work, which does not fit
in a single job. Two full submissions of the monolithic job timed out at the
cluster's 48h wall-time cap with nothing to show — no checkpoint, no partial
credit, because train_and_evaluate() only writes output once, at the very end.

The fix isn't more cores per job — it's recognising that the 19 passes don't
depend on each other at all (same fixed best_params every time; see
config/hyperparameter.py's HSMM_PARAM_GRID/HSMM_PARAM_OVERRIDES). So instead
of one job doing 19 passes internally, this script submits up to 19 separate
SLURM jobs (one "grid" + N "channel"), each comfortably inside 48h on its own,
then a fast "merge" step assembles them into the exact checkpoint format
train_and_evaluate() would have written — indistinguishable to every
downstream consumer (generate_all_state_seqs.py, the analysis-table/laterality
scripts, etc.). This is what actually got HSMM T4_P2 (n_components=8) to
complete: ~23h total wall-clock across parallel jobs, vs. two prior timeouts.

A separate "diagnostics" stage exists because everything downstream of a
finished checkpoint — emission plots, transition matrices, duration
distributions, state-importance heatmaps, event alignment — only needs ONE
full-data fit at the checkpoint's already-known hyperparameters, not another
LOO-CV sweep. Splitting it out means you never have to re-pay the expensive
training cost just to regenerate plots.

Stages
------
grid    — grid search (HSMM_PARAM_GRID has a single combo per task/paradigm,
          so this is really "the one combo") + baseline LOO pass + a
          full-data fit for downstream analysis. Writes grid.json.
channel — one channel's permutation-importance LOO pass. Writes
          channel_{idx:02d}.json. Submit one job per channel (0..n_channels-1).
merge   — combines grid.json + all channel_*.json into the final checkpoint,
          in the exact format HSMMModel.train_and_evaluate() writes, so every
          downstream script (generate_all_state_seqs.py, generate_hmm_
          analysis_table.py-style tools) finds it exactly as it would a
          normal run's checkpoint.

Determinism: `channel` uses the identical per-channel seed derivation as the
non-staged compute_feature_importance() (np.random.SeedSequence(42).spawn),
so results are byte-identical whether computed in one process or split
across many.

Usage:
    python scripts/hmm/run_hsmm_staged.py --stage grid --task 4 --paradigm 2 \\
        --partial-dir storage/results/xdash/hsmm_staged/T4_P2

    python scripts/hmm/run_hsmm_staged.py --stage channel --task 4 --paradigm 2 \\
        --channel-idx 0 --partial-dir storage/results/xdash/hsmm_staged/T4_P2
    # ... repeat for --channel-idx 1..n_channels-1, independently, in parallel

    python scripts/hmm/run_hsmm_staged.py --stage merge --task 4 --paradigm 2 \\
        --partial-dir storage/results/xdash/hsmm_staged/T4_P2
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def setup_project_path() -> bool:
    script_dir = Path(__file__).resolve().parent
    for root in (Path.cwd(), script_dir, script_dir.parent, script_dir.parent.parent):
        if (root / "models" / "hsmm_model.py").exists():
            sys.path.insert(0, str(root))
            return True
    return False


def parse_args():
    p = argparse.ArgumentParser(description="Staged HSMM training / feature-importance runner")
    p.add_argument("--stage", required=True, choices=["grid", "channel", "merge", "diagnostics"])
    p.add_argument("--dataset", default="xdash")
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--paradigm", type=int, required=True)
    p.add_argument("--channel-idx", type=int, default=None,
                   help="Required for --stage channel (0-based index into dataset channels)")
    p.add_argument("--partial-dir", default=None,
                   help="Directory to write/read partial-result JSON files (grid/channel/merge)")
    p.add_argument("--experiments-dir", default=None,
                   help="Root for the final checkpoint (default: storage/results/<dataset>/experiments)")
    p.add_argument("--checkpoint", default=None,
                   help="Path to the merged checkpoint JSON (required for --stage diagnostics)")
    p.add_argument("--hmm-csv-dir", default=None,
                   help="Directory with consolidated_task{n}.csv event files, for alignment analysis")
    return p.parse_args()


def load_xy(task, paradigm, dataset):
    """
    Mirrors pipeline/runner.py's run_experiment() data path exactly (load_data
    -> ParadigmSelector -> PreprocessorFactory('variable_length').prepare_data),
    NOT generate_all_state_seqs.py's simplified preprocess() — that helper
    tags subject_ids as bare "PX01"/"fx01", but build_loo_splits/resolve_fold_
    masks (and the g1_/g0_ consistency assertion) require the real pipeline's
    "g1_PX01"/"g0_fx01" tagging from VariableLengthPreprocessor._collect_
    sequences(). Using the wrong helper here would silently desync from what
    a normal (non-staged) train_and_evaluate() run actually does.
    """
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
        method="variable_length", model_type="hsmm",
        resample_rate=sampling_rate, original_rate=sampling_rate,
    )
    X, y, sids = preprocessor.prepare_data(g1, g0)
    return X, y, np.asarray(sids), dataset_config


def get_params(task, paradigm):
    """HSMM_PARAM_GRID entries are single-element lists (no real grid search —
    see config/hyperparameter.py) — take the combo directly, applying the
    same per-(task,paradigm) override HSMMModel.train_and_evaluate() applies."""
    from config.hyperparameter import HSMM_PARAM_GRID, HSMM_PARAM_OVERRIDES
    grid = dict(HSMM_PARAM_GRID)
    override = HSMM_PARAM_OVERRIDES.get((task, paradigm))
    if override:
        print(f"[params] Applying override for T{task}_P{paradigm}: {override}")
        grid.update(override)
    return {k: v[0] for k, v in grid.items()}


def stage_grid(args):
    from models.hsmm_model import HSMMModel
    from utils.training import build_loo_splits

    X, y, sids, _ = load_xy(args.task, args.paradigm, args.dataset)
    params = get_params(args.task, args.paradigm)
    print(f"[grid] T{args.task}_P{args.paradigm} params: {params}")

    model = HSMMModel(task=args.task, paradigm=args.paradigm)
    cv_splits, unique_subjects = build_loo_splits(len(X), sids, "HSMM")

    (ba, _, y_true, y_pred, y_proba,
     _per_fold, subject_order) = model._loo_score(
        params, X, y, cv_splits, sids, unique_subjects
    )
    print(f"[grid] LOO balanced accuracy: {ba:.4f}")

    # Same invariant HSMMModel.train_and_evaluate() checks.
    for sid, yt in zip(subject_order, y_true):
        expected = 1 if str(sid).startswith("g1_") else 0
        assert expected == int(yt), f"subject_id/y_true mismatch: {sid} vs y_true={yt}"
    print("[grid] subject_id/y_true consistency check passed")

    print("[grid] fitting full-data model for downstream analysis ...")
    model.fit_for_analysis(
        X=X, y=y,
        n_components=params["n_components"],
        covariance_type=params["covariance_type"],
        n_iter=params["n_iter"],
        max_duration=params.get("max_duration", 200),
    )

    out = {
        "stage": "grid",
        "task": args.task,
        "paradigm": args.paradigm,
        "best_params": params,
        "best_score": ba,
        "baseline_ba": ba,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
        "subject_order": [str(s) for s in subject_order],
        "n_channels": int(X[0].shape[1]),
        "timestamp": datetime.now().isoformat(),
    }
    partial_dir = Path(args.partial_dir)
    partial_dir.mkdir(parents=True, exist_ok=True)
    out_path = partial_dir / "grid.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[grid] wrote {out_path}")


def stage_channel(args):
    from models.hsmm_model import HSMMModel
    from utils.training import build_loo_splits

    if args.channel_idx is None:
        raise SystemExit("--channel-idx is required for --stage channel")

    partial_dir = Path(args.partial_dir)
    grid_path = partial_dir / "grid.json"
    if not grid_path.exists():
        raise SystemExit(f"Run --stage grid first — missing {grid_path}")
    grid = json.load(open(grid_path))
    params = grid["best_params"]
    n_channels = grid["n_channels"]
    if not (0 <= args.channel_idx < n_channels):
        raise SystemExit(f"--channel-idx must be in [0, {n_channels - 1}]")

    X, y, sids, dataset_config = load_xy(args.task, args.paradigm, args.dataset)
    ch_names = dataset_config.get("channels", [f"ch{i}" for i in range(n_channels)])

    model = HSMMModel(task=args.task, paradigm=args.paradigm)
    cv_splits, unique_subjects = build_loo_splits(len(X), sids, "HSMM")

    # Identical derivation to the non-staged compute_feature_importance() path
    # (models/hsmm_model.py) — same seed regardless of execution order.
    seed = np.random.SeedSequence(42).spawn(n_channels)[args.channel_idx]
    rng = np.random.default_rng(seed)
    seqs_perm = model._permute_channel(X, args.channel_idx, rng)

    ba_d, *_ = model._loo_score(params, seqs_perm, y, cv_splits, sids, unique_subjects)
    drop = grid["baseline_ba"] - ba_d
    print(f"[channel {args.channel_idx}] {ch_names[args.channel_idx]}: "
          f"ba_d={ba_d:.4f}  drop={drop:+.4f}")

    out = {
        "stage": "channel",
        "channel_idx": args.channel_idx,
        "channel_name": ch_names[args.channel_idx],
        "ba_d": ba_d,
    }
    out_path = partial_dir / f"channel_{args.channel_idx:02d}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[channel {args.channel_idx}] wrote {out_path}")


def stage_merge(args):
    from utils.metrics import compute_metrics
    from config.paths import get_experiments_dir

    partial_dir = Path(args.partial_dir)
    grid = json.load(open(partial_dir / "grid.json"))
    n_channels = grid["n_channels"]

    channel_files = sorted(partial_dir.glob("channel_*.json"))
    if len(channel_files) != n_channels:
        found = [json.load(open(f))["channel_idx"] for f in channel_files]
        missing = sorted(set(range(n_channels)) - set(found))
        raise SystemExit(
            f"Expected {n_channels} channel partial files, found {len(channel_files)} "
            f"in {partial_dir}. Missing channel indices: {missing}"
        )

    drops = np.zeros(n_channels)
    names = [None] * n_channels
    baseline_ba = grid["baseline_ba"]
    for f in channel_files:
        d = json.load(open(f))
        idx = d["channel_idx"]
        drops[idx] = baseline_ba - d["ba_d"]
        names[idx] = d["channel_name"]

    importance = np.clip(drops, 0, None)
    denom = importance.sum()
    importance = importance / denom if denom > 1e-12 else np.ones(n_channels) / n_channels

    feature_imp = {
        names[i]: float(importance[i])
        for i in np.argsort(importance)[::-1]
    }

    y_true = np.array(grid["y_true"])
    y_pred = np.array(grid["y_pred"])
    y_proba = np.array(grid["y_proba"])
    subject_order = grid["subject_order"]
    metrics = compute_metrics(y_true, y_pred, y_proba)
    best_score = grid["best_score"]
    best_params = grid["best_params"]

    experiments_dir = Path(args.experiments_dir) if args.experiments_dir else get_experiments_dir(args.dataset)
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = experiments_dir / f"task{args.task}" / f"paradigm{args.paradigm}" / f"HSMM_{timestamp_tag}"
    checkpoints_dir = experiment_dir / "model_checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Config snapshot, matching pipeline/runner.py's run_experiment() (config.json
    # written as soon as the experiment dir exists) so staged and monolithic runs
    # are discoverable the same way by downstream tooling.
    with open(experiment_dir / "config.json", "w") as f:
        json.dump({
            "dataset": args.dataset,
            "task": args.task,
            "paradigm": args.paradigm,
            "model": "hsmm",
            "method": "variable_length",
            "timestamp": timestamp_tag,
            "diagnostics_enabled": True,
        }, f, indent=2)

    best_path = checkpoints_dir / f"HSMM_T{args.task}_P{args.paradigm}_BA{best_score:.4f}_{timestamp_tag}.json"
    with open(best_path, "w") as f:
        json.dump({
            "model_name": "HSMM",
            "task": args.task,
            "paradigm": args.paradigm,
            "hyperparameters": best_params,
            "metrics": {"balanced_accuracy": best_score, **metrics},
            "feature_importance": feature_imp,
            "predictions": {
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "y_proba": y_proba.tolist(),
                "subject_ids": subject_order,
            },
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"[merge] wrote final checkpoint -> {best_path}")
    print(f"[merge] BA={best_score:.4f}  AUC={metrics.get('auc')}")
    print("[merge] Feature importance (top 5):")
    for i, (k, v) in enumerate(list(feature_imp.items())[:5]):
        print(f"  {i+1}. {k}: {v:.4f}")


def stage_diagnostics(args):
    """
    Everything downstream of "we already have a merged checkpoint": figures/
    (confusion matrix, ROC, probability distribution), results/ (predictions
    CSV + results JSON), and diagnostics/ (emission plots, transition
    matrices, duration distributions, state-importance heatmaps, event
    alignment). All of this only needs ONE full-data fit at the checkpoint's
    hyperparameters — not a re-run of the LOO-CV training — mirroring exactly
    what pipeline/runner.py's _run_diagnostics() does for a normal (non-
    staged) run, so the output layout matches a regular experiment folder.
    """
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required for --stage diagnostics")

    import numpy as np
    from models.hsmm_model import HSMMModel
    from models.base_model import ModelResults
    from utils.evaluator import ModelEvaluator, Visualizer
    from pipeline.io import save_results, save_predictions

    ckpt_path = Path(args.checkpoint)
    ckpt = json.load(open(ckpt_path))
    params = ckpt["hyperparameters"]
    preds = ckpt["predictions"]
    y_true = np.array(preds["y_true"])
    y_pred = np.array(preds["y_pred"])
    y_proba = np.array(preds["y_proba"])
    subject_ids = preds["subject_ids"]

    X, y, sids, dataset_config = load_xy(args.task, args.paradigm, args.dataset)
    channel_names = dataset_config.get("channels", [])
    sampling_rate = dataset_config.get("sampling_rate", 50)

    # model_checkpoints/HSMM_T{t}_P{p}_BA*.json -> experiment_dir is its grandparent
    experiment_dir = ckpt_path.resolve().parent.parent
    figures_dir = experiment_dir / "figures"
    diagnostics_dir = experiment_dir / "diagnostics"
    results_dir = experiment_dir / "results"
    for d in (figures_dir, diagnostics_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    model = HSMMModel(task=args.task, paradigm=args.paradigm)
    print("[diagnostics] Fitting full-data model at checkpoint hyperparameters "
          f"({params}) ...")
    model.fit_for_analysis(
        X=X, y=y,
        n_components=params["n_components"],
        covariance_type=params["covariance_type"],
        n_iter=params["n_iter"],
        max_duration=params.get("max_duration", 200),
    )

    evaluator = ModelEvaluator()
    eval_results = evaluator.evaluate(
        y_true=y_true, y_pred=y_pred, y_proba=y_proba, subject_ids=subject_ids
    )
    evaluator.print_report(eval_results)

    viz = Visualizer()
    tag = f"Task-{args.task} P{args.paradigm} HSMM"
    viz.plot_confusion_matrix(
        cm=eval_results.confusion_matrix, class_names=["Group0", "Group1"],
        normalize=True, title=f"{tag} - Confusion Matrix",
        save_path=figures_dir / "confusion_matrix.png",
    )
    viz.plot_roc_curve(
        y_true=y_true, y_proba=y_proba, title=f"{tag} - ROC Curve",
        save_path=figures_dir / "roc_curve.png",
    )
    viz.plot_probability_distribution(
        y_true=y_true, y_proba=y_proba, title=f"{tag} - Probability Distribution",
        save_path=figures_dir / "probability_distribution.png",
    )

    results_obj = ModelResults(
        metrics=ckpt["metrics"], best_params=params,
        feature_importance=ckpt["feature_importance"],
        y_true=y_true, y_pred=y_pred, y_proba=y_proba,
        X_shape=(len(X), X[0].shape[1]), subject_ids=subject_ids, per_fold_results=[],
    )
    results_dict = save_results(results_obj, args.task, args.paradigm, "HSMM", "variable_length",
                 results_dir, dataset_config)
    save_predictions(results_obj, subject_ids, args.task, args.paradigm, "HSMM",
                      "variable_length", results_dir)

    seq_sids = [str(sid).split("_", 2)[-1] for sid in subject_ids]
    fitted0, fitted1 = model.fitted_hsmm0, model.fitted_hsmm1

    model.plot_emission_distributions(
        model=fitted1, channel_names=channel_names,
        title_suffix=f"Patient — Task {args.task} P{args.paradigm}",
        save_path=diagnostics_dir / "emissions_patient.png",
    )
    model.plot_emission_distributions(
        model=fitted0, channel_names=channel_names,
        title_suffix=f"Control — Task {args.task} P{args.paradigm}",
        save_path=diagnostics_dir / "emissions_control.png",
    )
    model.plot_transition_matrix(
        model=fitted1, title_suffix=f"Patient — Task {args.task} P{args.paradigm}",
        save_path=diagnostics_dir / "transition_patient.png",
    )
    model.plot_transition_matrix(
        model=fitted0, title_suffix=f"Control — Task {args.task} P{args.paradigm}",
        save_path=diagnostics_dir / "transition_control.png",
    )
    model.plot_duration_distributions(save_path=diagnostics_dir / "duration_distributions.png")

    global_imp, state_imp = model.compute_state_specific_importance(
        model=fitted1, channel_names=channel_names
    )
    model.plot_state_importance_heatmap(
        state_importance=state_imp, channel_names=channel_names,
        title=f"HSMM State Feature Importance (Patient) T{args.task} P{args.paradigm}",
        save_path=diagnostics_dir / "state_importance_patient.png",
    )
    global_imp_ctrl, state_imp_ctrl = model.compute_state_specific_importance(
        model=fitted0, channel_names=channel_names
    )
    model.plot_state_importance_heatmap(
        state_importance=state_imp_ctrl, channel_names=channel_names,
        title=f"HSMM State Feature Importance (Control) T{args.task} P{args.paradigm}",
        save_path=diagnostics_dir / "state_importance_control.png",
    )

    try:
        model.compare_patient_control_emissions(
            channel_names=channel_names, save_path=diagnostics_dir / "emission_diff.png"
        )
    except ValueError as e:
        print(f"  Emission comparison skipped — {e}")

    if args.hmm_csv_dir:
        csv_path = Path(args.hmm_csv_dir) / f"consolidated_task{args.task}.csv"
        if csv_path.exists():
            model.run_alignment_analysis(
                sequences=X, subject_ids=seq_sids, csv_path=csv_path,
                task_id=args.task, paradigm_id=args.paradigm, tolerance_s=0.5,
                sampling_rate=sampling_rate,
                save_path=diagnostics_dir / f"alignment_T{args.task}_P{args.paradigm}.csv",
            )
        else:
            print(f"  Event CSV not found at {csv_path} — alignment skipped")

    diagnostic_results = {
        "hsmm_analysis": {
            "n_components":           params.get("n_components", 2),
            "covariance_type":        params.get("covariance_type", "diag"),
            "global_importance":      global_imp,
            "global_importance_ctrl": global_imp_ctrl,
        }
    }

    # Summary, matching pipeline/runner.py's run_experiment() (summary.json
    # written once evaluation/figures/results/diagnostics all complete) so
    # staged and monolithic runs are discoverable the same way downstream.
    summary = {
        "experiment_name": experiment_dir.name,
        "dataset": args.dataset,
        "config": {"task": args.task, "paradigm": args.paradigm,
                   "model": "hsmm", "method": "variable_length"},
        "results": results_dict,
        "evaluation": {
            "accuracy":          getattr(eval_results, "accuracy", None),
            "balanced_accuracy": getattr(eval_results, "balanced_accuracy", None),
            "auc_roc":           getattr(eval_results, "auc_roc", None),
            "auc_roc_ci":        getattr(eval_results, "auc_roc_ci", None),
        },
    }
    if diagnostic_results:
        summary["diagnostics"] = {
            "overfitting_risk":    diagnostic_results.get("overfitting", {}).get("risk", "N/A"),
            "generalization_gap":  diagnostic_results.get("overfitting", {}).get("generalization_gap", "N/A"),
        }

    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
            return super().default(obj)

    with open(experiment_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=_NpEncoder)

    print(f"\n[diagnostics] Complete — outputs in {experiment_dir}")


def main():
    if not setup_project_path():
        print("[ERROR] Could not locate project root (models/hsmm_model.py not found).")
        sys.exit(1)
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # for generate_all_state_seqs import

    args = parse_args()
    {
        "grid": stage_grid,
        "channel": stage_channel,
        "merge": stage_merge,
        "diagnostics": stage_diagnostics,
    }[args.stage](args)


if __name__ == "__main__":
    main()
