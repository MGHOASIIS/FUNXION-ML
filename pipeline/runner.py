"""
pipeline/runner.py — experiment orchestration.

run_experiment(args, dataset_config) is the single entry point called by main.py.
"""
import json
from datetime import datetime
from pathlib import Path

from dataio.paradigms import ParadigmSelector
from dataio.preprocessors import PreprocessorFactory, AugmentedPreprocessor
from models.hmm_model import HMMModel
from models.hsmm_model import HSMMModel
from models.cnn_model import CNNModel
from models.rnn_model import RNNModel
from models.transformer_model import TransformerModel
from utils.evaluator import ModelEvaluator, Visualizer
from config.paths import get_experiments_dir
from pipeline.io import (
    load_data, load_event_window_data,
    save_results, save_predictions,
)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(model_type: str, checkpoints_dir=None,
                 patience=None, min_delta=None, task=None, paradigm=None,
                 channel_names=None, multilabel=False, label_names=None):
    model_type = model_type.lower()
    if model_type == "hmm":
        return HMMModel(checkpoints_dir=checkpoints_dir, task=task, paradigm=paradigm,
                        channel_names=channel_names, multilabel=multilabel,
                        label_names=label_names)
    if model_type == "hsmm":
        return HSMMModel(checkpoints_dir=checkpoints_dir, task=task, paradigm=paradigm,
                         channel_names=channel_names, multilabel=multilabel,
                         label_names=label_names)
    if model_type == "cnn":
        return CNNModel(checkpoints_dir, patience=patience, min_delta=min_delta,
                        task=task, paradigm=paradigm, channel_names=channel_names,
                        multilabel=multilabel, label_names=label_names)
    if model_type == "rnn":
        return RNNModel(checkpoints_dir, patience=patience, min_delta=min_delta,
                        task=task, paradigm=paradigm, channel_names=channel_names,
                        multilabel=multilabel, label_names=label_names)
    if model_type == "transformer":
        return TransformerModel(checkpoints_dir=checkpoints_dir, patience=patience,
                                min_delta=min_delta, task=task, paradigm=paradigm,
                                channel_names=channel_names, multilabel=multilabel,
                                label_names=label_names)
    raise ValueError(f"Unknown model type: {model_type!r}")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(args, dataset_config: dict):
    channel_names = dataset_config.get("channels", [])
    task_names    = dataset_config.get("tasks", {})
    paradigm_cfgs = dataset_config.get("paradigms", {})
    paradigm_names = {k: v.get("name", f"paradigm_{k}") for k, v in paradigm_cfgs.items()}
    sampling_rate  = dataset_config.get("sampling_rate", 50)

    # ── Experiment directory setup ────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = (
        f"{args.experiment_name}_{timestamp}" if args.experiment_name
        else f"{args.model.upper()}_{timestamp}"
    )

    experiments_dir = get_experiments_dir(args.dataset)
    experiment_dir  = experiments_dir / f"task{args.task}" / f"paradigm{args.paradigm}" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    results_dir     = experiment_dir / "results"
    figures_dir     = experiment_dir / "figures"
    checkpoints_dir = experiment_dir / "model_checkpoints"
    diagnostics_dir = experiment_dir / "diagnostics"

    for d in [results_dir, figures_dir, checkpoints_dir]:
        d.mkdir(exist_ok=True)

    if args.diagnostics:
        diag_path = Path(args.diagnostics_dir) if args.diagnostics_dir else diagnostics_dir
        diag_path.mkdir(parents=True, exist_ok=True)
        diagnostics_dir = diag_path

    # ── Print header ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Dataset:    {args.dataset}")
    print(f"Experiment: {experiment_name}")
    print(f"Task:       {args.task} ({task_names.get(args.task, '?')})")
    print(f"Paradigm:   {args.paradigm} ({paradigm_names.get(args.paradigm, '?')})")
    print(f"Model:      {args.model.upper()}")
    print(f"Preprocess: {args.method}")
    if args.method == "sliding_window":
        print(f"  Window: {args.window_size}  Overlap: {args.overlap:.1%}")
    elif args.method == "downsample_truncate":
        print(f"  {args.original_rate} Hz -> {args.target_rate} Hz")
    elif args.method == "dtw_embedding":
        print(f"  Components: {args.n_components}  Method: {args.dtw_method}")
    elif args.method == "phase_shift":
        print(f"  Shift: {args.shift_fraction:.3f}")
    if args.freq < sampling_rate:
        print(f"  Resample: {sampling_rate} Hz -> {args.freq} Hz")
    print(f"Diagnostics: {'ENABLED' if args.diagnostics else 'disabled'}")
    print("=" * 70)

    # Save config snapshot
    with open(experiment_dir / "config.json", "w") as f:
        json.dump({
            "dataset": args.dataset,
            "task": args.task,
            "paradigm": args.paradigm,
            "model": args.model,
            "method": args.method,
            "timestamp": timestamp,
            "diagnostics_enabled": args.diagnostics,
        }, f, indent=2)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.data_source == "event_window":
        patient_data, control_data = load_event_window_data(
            args.task, args.dataset, dataset_config
        )
    else:
        patient_data, control_data = load_data(args.task, args.dataset, dataset_config)

    selector = ParadigmSelector(dataset_config)
    is_multilabel = selector.get_paradigm_type(args.paradigm) == "multilabel"

    label_names = None
    g1 = g0 = groups = None
    if is_multilabel:
        label_names = list(paradigm_cfgs[args.paradigm]["labels"].keys())
        groups = selector.select_labels(patient_data, control_data, paradigm=args.paradigm)
    else:
        g1, g0 = selector.select_paradigm(patient_data, control_data, paradigm=args.paradigm)

    # ── Preprocess ────────────────────────────────────────────────────────────
    preproc_kwargs = {}
    resample_rate = args.freq
    preproc_original_rate = sampling_rate
    if args.method == "sliding_window":
        preproc_kwargs = {"window_size": args.window_size, "overlap": args.overlap}
    elif args.method == "downsample_truncate":
        # Dedicated --target-rate/--original-rate flags, independent of the
        # generic --freq resampling mechanism used by every other method.
        resample_rate = args.target_rate
        preproc_original_rate = args.original_rate
    elif args.method == "dtw_embedding":
        preproc_kwargs = {"n_components": args.n_components, "dtw_method": args.dtw_method}
    elif args.method == "phase_shift":
        preproc_kwargs = {"shift_fraction": args.shift_fraction}

    preprocessor = PreprocessorFactory.create(
        method=args.method,
        model_type=args.model,
        resample_rate=resample_rate,
        original_rate=preproc_original_rate,
        data_source=args.data_source,
        # This is the LOSO cross-validation training path: truncation
        # length and the z-score scaler must be fit per fold (training
        # subjects only), not once over the full dataset. Only affects
        # cnn/rnn/transformer with method='truncate'/'downsample_truncate'
        # — see PreprocessorFactory.create()'s fold_safe docstring.
        fold_safe=True,
        **preproc_kwargs,
    )
    if args.augment:
        preprocessor = AugmentedPreprocessor(
            base_preprocessor=preprocessor,
            augmentations=args.augment_methods,
            n_augmentations=args.n_augmentations,
        )

    if is_multilabel:
        X, y, subject_ids = preprocessor.prepare_data_multilabel(groups, label_names)
    else:
        X, y, subject_ids = preprocessor.prepare_data(g1, g0)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = create_model(
        args.model,
        checkpoints_dir=checkpoints_dir if args.save_checkpoints else None,
        patience=args.patience,
        min_delta=args.min_delta,
        task=args.task,
        paradigm=args.paradigm,
        channel_names=channel_names,
        multilabel=is_multilabel,
        label_names=label_names,
    )
    if is_multilabel:
        results = model.fit_multilabel(
            groups=groups, label_names=label_names,
            preprocessor=preprocessor, paradigm=args.paradigm,
        )
    else:
        results = model.fit(g1=g1, g0=g0, preprocessor=preprocessor, paradigm=args.paradigm)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluator = ModelEvaluator()
    viz = Visualizer()
    tag = f"Task-{args.task} P{args.paradigm} {args.model.upper()}"

    if is_multilabel:
        eval_results = evaluator.evaluate_multilabel(
            y_true=results.y_true, y_pred=results.y_pred,
            y_proba=results.y_proba, label_names=label_names,
            subject_ids=subject_ids,
        )
        evaluator.print_report_multilabel(eval_results)

        viz.plot_multilabel_confusion_matrices(
            confusion_matrices=eval_results.confusion_matrices,
            title=f"{tag} - Confusion Matrices",
            save_path=figures_dir / "confusion_matrices.png",
        )
        # Per-label ROC/probability plots reuse the existing binary plot
        # functions, once per label column — no new plotting code needed.
        for li, name in enumerate(label_names):
            viz.plot_roc_curve(
                y_true=results.y_true[:, li], y_proba=results.y_proba[:, li],
                title=f"{tag} - ROC Curve ({name})",
                save_path=figures_dir / f"roc_curve_{name}.png",
            )
            viz.plot_probability_distribution(
                y_true=results.y_true[:, li], y_proba=results.y_proba[:, li],
                title=f"{tag} - Probability Distribution ({name})",
                save_path=figures_dir / f"probability_distribution_{name}.png",
            )
    else:
        eval_results = evaluator.evaluate(
            y_true=results.y_true, y_pred=results.y_pred,
            y_proba=results.y_proba, subject_ids=subject_ids,
        )
        evaluator.print_report(eval_results)

        viz.plot_confusion_matrix(
            cm=eval_results.confusion_matrix, class_names=["Group0", "Group1"],
            normalize=True, title=f"{tag} - Confusion Matrix",
            save_path=figures_dir / "confusion_matrix.png",
        )
        viz.plot_roc_curve(
            y_true=results.y_true, y_proba=results.y_proba,
            title=f"{tag} - ROC Curve", save_path=figures_dir / "roc_curve.png",
        )
        viz.plot_probability_distribution(
            y_true=results.y_true, y_proba=results.y_proba,
            title=f"{tag} - Probability Distribution",
            save_path=figures_dir / "probability_distribution.png",
        )

    # ── Save results ──────────────────────────────────────────────────────────
    results_dict = save_results(
        results, args.task, args.paradigm, args.model.upper(),
        args.method, results_dir, dataset_config,
    )
    save_predictions(
        results, subject_ids, args.task, args.paradigm,
        args.model.upper(), args.method, results_dir,
        label_names=label_names,
    )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    diagnostic_results = None
    if args.diagnostics and is_multilabel:
        print("\n[Diagnostics] Multi-label diagnostics not yet supported — "
              "skipping --diagnostics for this run.")
    elif args.diagnostics:
        diagnostic_results = _run_diagnostics(
            args, model, X, y, subject_ids, results,
            channel_names, experiment_name, diagnostics_dir,
            sampling_rate,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "experiment_name": experiment_name,
        "dataset": args.dataset,
        "config": {"task": args.task, "paradigm": args.paradigm,
                   "model": args.model, "method": args.method},
        "results": results_dict,
        "evaluation": (
            {
                "subset_accuracy":         eval_results.subset_accuracy,
                "hamming_loss":            eval_results.hamming_loss,
                "macro_f1":                eval_results.macro_f1,
                "micro_f1":                eval_results.micro_f1,
                "macro_balanced_accuracy": eval_results.macro_balanced_accuracy,
            } if is_multilabel else {
                "accuracy":          eval_results.accuracy,
                "balanced_accuracy": eval_results.balanced_accuracy,
                "auc_roc":           eval_results.auc_roc,
                "auc_roc_ci":        eval_results.auc_roc_ci,
            }
        ),
    }
    if diagnostic_results:
        summary["diagnostics"] = {
            "overfitting_risk":    diagnostic_results.get("overfitting", {}).get("risk", "N/A"),
            "generalization_gap":  diagnostic_results.get("overfitting", {}).get("generalization_gap", "N/A"),
        }

    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
            return super().default(obj)

    with open(experiment_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=_NpEncoder)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    if is_multilabel:
        print(f"Subset Accuracy: {results.metrics['subset_accuracy']:.3f}")
        print(f"Macro F1:        {results.metrics['macro_f1']:.3f}")
        print(f"Hamming Loss:    {results.metrics['hamming_loss']:.3f}")
    else:
        print(f"Balanced Accuracy: {results.metrics['ba']:.3f}")
        print(f"AUC:               {results.metrics['auc']:.3f}")
        print(f"Recall:            {results.metrics['recall']:.3f}")
    print(f"\nOutputs: {experiment_dir}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Diagnostics (HMM/HSMM and NN paths, unchanged from original main.py)
# ---------------------------------------------------------------------------

def _run_diagnostics(args, model, X, y, subject_ids, results,
                     channel_names, experiment_name, diagnostics_dir,
                     sampling_rate: int = 50):
    print(f"\n{'='*70}\nRUNNING COMPREHENSIVE DIAGNOSTICS\n{'='*70}\n")

    if args.model.lower() in ("hmm", "hsmm"):
        is_hsmm = args.model.lower() == "hsmm"
        tag = "HSMM" if is_hsmm else "HMM"
        print(f"[{tag} Diagnostics] Running interpretability analysis ...")

        seq_sids = [str(sid).split("_", 2)[-1] for sid in subject_ids]
        fitted0 = model.fitted_hsmm0 if is_hsmm else model.fitted_hmm0
        fitted1 = model.fitted_hsmm1 if is_hsmm else model.fitted_hmm1

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
            model=fitted1,
            title_suffix=f"Patient — Task {args.task} P{args.paradigm}",
            save_path=diagnostics_dir / "transition_patient.png",
        )
        model.plot_transition_matrix(
            model=fitted0,
            title_suffix=f"Control — Task {args.task} P{args.paradigm}",
            save_path=diagnostics_dir / "transition_control.png",
        )
        if is_hsmm:
            model.plot_duration_distributions(
                save_path=diagnostics_dir / "duration_distributions.png"
            )

        global_imp, state_imp = model.compute_state_specific_importance(
            model=fitted1, channel_names=channel_names
        )
        model.plot_state_importance_heatmap(
            state_importance=state_imp, channel_names=channel_names,
            title=f"{tag} State Feature Importance (Patient) T{args.task} P{args.paradigm}",
            save_path=diagnostics_dir / "state_importance_patient.png",
        )
        global_imp_ctrl, state_imp_ctrl = model.compute_state_specific_importance(
            model=fitted0, channel_names=channel_names
        )
        model.plot_state_importance_heatmap(
            state_importance=state_imp_ctrl, channel_names=channel_names,
            title=f"{tag} State Feature Importance (Control) T{args.task} P{args.paradigm}",
            save_path=diagnostics_dir / "state_importance_control.png",
        )

        try:
            model.compare_patient_control_emissions(
                channel_names=channel_names,
                save_path=diagnostics_dir / "emission_diff.png",
            )
        except ValueError as e:
            print(f"  Emission comparison skipped — {e}")

        if hasattr(args, "hmm_csv_dir") and args.hmm_csv_dir:
            from pathlib import Path
            csv_path = Path(args.hmm_csv_dir) / f"consolidated_task{args.task}.csv"
            if csv_path.exists():
                # run_alignment_analysis decodes sequences with self.fitted_hmm0/1,
                # which were fit on the globally-scaled X_analysis inside
                # train_and_evaluate() — X here is raw (VariableLengthPreprocessor
                # no longer scales), so it needs the same global scaling to match
                # the scale the fitted models actually expect.
                from utils.training import scale_sequences_global
                X_aligned = scale_sequences_global(list(X))
                model.run_alignment_analysis(
                    sequences=X_aligned, subject_ids=seq_sids, csv_path=csv_path,
                    task_id=args.task, paradigm_id=args.paradigm,
                    tolerance_s=0.5, sampling_rate=sampling_rate,
                    save_path=diagnostics_dir / f"alignment_T{args.task}_P{args.paradigm}.csv",
                )
            else:
                print(f"  Event CSV not found at {csv_path} — alignment skipped")

        diagnostic_results = {
            f"{args.model.lower()}_analysis": {
                "n_components":           results.best_params.get("n_components", 2),
                "covariance_type":        results.best_params.get("covariance_type", "diag"),
                "global_importance":      global_imp,
                "global_importance_ctrl": global_imp_ctrl,
            }
        }
        print(f"\n[{tag} Diagnostics] Complete — outputs in {diagnostics_dir}")
        return diagnostic_results

    else:
        from utils.comprehensive_monitor import run_complete_monitoring
        return run_complete_monitoring(
            model=model, X=X, y=y, subject_ids=subject_ids,
            fold_results=results.per_fold_results,
            experiment_name=experiment_name,
            save_dir=diagnostics_dir,
            hyperparameters=results.best_params,
            feature_names=channel_names,
        )
