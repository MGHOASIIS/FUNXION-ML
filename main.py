"""
Main entry point for running XDash classification experiments.

Usage:
    # Basic (fast)
    python main.py --task 1 --paradigm 1 --model rnn --method truncate
    
    # With comprehensive diagnostics (slower but complete)
    python main.py --task 1 --paradigm 1 --model rnn --method truncate --diagnostics
    
    # Full example with all options
    python main.py --task 1 --paradigm 1 --model rnn --method sliding_window \
        --window-size 300 --overlap 0.3 --diagnostics --save-figures
"""
import argparse
import pickle
import json
from pathlib import Path
from datetime import datetime

from config.constants import TASK_NAMES, PARADIGM_NAMES, CHAN_NAME
from config.paths import get_pickled_dataset_path, EXPERIMENTS_DIR
from data.paradigms import ParadigmSelector
from data.preprocessors import PreprocessorFactory, AugmentedPreprocessor
from models.hmm_model import HMMModel
from models.cnn_model import CNNModel
from models.rnn_model import RNNModel
from models.transformer_model import TransformerModel

def load_data(task: int):
    """Load patient and control data for given task."""
    patient_path = get_pickled_dataset_path(task, "patient")
    control_path = get_pickled_dataset_path(task, "control")
    
    with open(patient_path, "rb") as f:
        patient_data = pickle.load(f)
    
    with open(control_path, "rb") as f:
        control_data = pickle.load(f)
    
    print(f"\n[Data Loaded]")
    print(f"  Task: {TASK_NAMES.get(task, task)}")
    print(f"  Patients: {len(patient_data)}")
    print(f"  Controls: {len(control_data)}")
    
    return patient_data, control_data


def create_model(model_type: str, checkpoints_dir=None, patience=None, min_delta=None, task=None, paradigm=None):
    """Create model instance."""
    model_type = model_type.lower()
    
    if model_type == "hmm":
        return HMMModel(
            checkpoints_dir=checkpoints_dir, 
            task=task, 
            paradigm=paradigm
        )
    elif model_type == "cnn":
        return CNNModel(
            checkpoints_dir, 
            patience=patience, 
            min_delta=min_delta, 
            task=task, 
            paradigm=paradigm
        )
    elif model_type == "rnn":
        return RNNModel(
            checkpoints_dir, 
            patience=patience, 
            min_delta=min_delta, 
            task=task, 
            paradigm=paradigm
        )
    elif model_type == "transformer":
        return TransformerModel(
            checkpoints_dir=checkpoints_dir,
            patience=patience,
            min_delta=min_delta,
            task=task,
            paradigm=paradigm
        )


def save_results(results, task: int, paradigm: int, model_name: str, method: str, save_dir: Path):
    """Save results to JSON file."""
    results_dict = {
        "task": task,
        "task_name": TASK_NAMES.get(task, str(task)),
        "paradigm": paradigm,
        "paradigm_name": PARADIGM_NAMES.get(paradigm, str(paradigm)),
        "model": model_name,
        "preprocessing_method": method,
        "metrics": results.metrics,
        "best_params": results.best_params,
        "feature_importance": results.feature_importance,
        "X_shape": list(results.X_shape)
    }
    
    # Save to file
    filename = f"results_T{task}_P{paradigm}_{model_name}_{method}.json"
    filepath = save_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n[Results Saved] {filepath}")
    
    return results_dict


# def run_diagnostics(
#     model,
#     X,
#     y,
#     subject_ids,
#     results,
#     experiment_name: str,
#     diagnostics_dir: Path
# ):
#     """
#     Run comprehensive diagnostics.
    
#     Parameters
#     ----------
#     model : BaseModel
#         Trained model
#     X : np.ndarray
#         Features
#     y : np.ndarray
#         Labels
#     subject_ids : np.ndarray
#         Subject IDs
#     results : ModelResults
#         Training results
#     experiment_name : str
#         Experiment name
#     diagnostics_dir : Path
#         Directory to save diagnostics
#     """
#     print(f"\n{'='*70}")
#     print("RUNNING COMPREHENSIVE DIAGNOSTICS")
#     print(f"{'='*70}\n")
    
#     from utils.comprehensive_monitor import run_complete_monitoring
#     import numpy as np
    
#     # Run complete monitoring
#     diagnostic_results = run_complete_monitoring(
#         model=model,
#         X=X,
#         y=y,
#         subject_ids=subject_ids,
#         fold_results=results.per_fold_results,
#         experiment_name=experiment_name,
#         save_dir=diagnostics_dir,
#         hyperparameters=results.best_params,
#         feature_names=CHAN_NAME
#     )
    
#     return diagnostic_results


def main():
    parser = argparse.ArgumentParser(
        description="Run XDash classification experiments with optional comprehensive diagnostics"
    )
    
    # Required arguments
    parser.add_argument(
        "-t",
        "--task",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5, 6],
        help="Task number (1=jar_opening, 2=key_turning, etc.)"
    )
    
    parser.add_argument(
        "-p",
        "--paradigm",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Classification paradigm (1=patients_vs_controls, 2=rct_vs_controls, etc.)"
    )
    
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        choices=["hmm", "cnn", "rnn", "transformer"],
        help="Model type"
    )
    
    # Preprocessing arguments
    parser.add_argument(
        "-pre",
        "--method",
        type=str,
        default="truncate",
        choices=["truncate", "sliding_window", "padding", "dtw_embedding", "downsample_truncate"],
        help="Preprocessing method"
    )
    
    # sliding window arguments
    parser.add_argument(
        "--window-size",
        type=int,
        default=300,
        help="Window size for sliding_window method (default: 300)"
    )
    
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.30,
        help="Overlap for sliding_window method (default: 0.30)"
    )
    
    # downsample truncating arguments
    parser.add_argument(
        "--target-rate",
        type=int,
        default=25,
        help="Target sampling rate for downsample_truncate method (default: 25 Hz)"
    )
    
    parser.add_argument(
        "--original-rate",
        type=int,
        default=50,
        help="Original sampling rate for downsample_truncate method (default: 50 Hz)"
    )
    
    # dtw_embedding arguments
    parser.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="Number of components for dtw_embedding method (default: 10)"
    )
    
    parser.add_argument(
        "--dtw-method",
        type=str,
        default="mds",
        choices=["mds", "isomap", "tsne"],
        help="Embedding method for dtw_embedding (default: mds)"
    )

    # Early stopping arguments
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (default: 15 epochs)"
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum delta for early stopping (default: 1e-4)"
    )

    # Augmentation arguments
    parser.add_argument(
        "--augment", 
        action="store_true"
    )
    parser.add_argument(
        "--augment-methods", 
        nargs='+', 
        default=['jitter', 'time_warp', 'magnitude_warp']
    )
    parser.add_argument(
        "--n-augmentations", 
        type=int, 
        default=2
    )
    
    # Diagnostic arguments
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run comprehensive diagnostics (overfitting detection, gradient analysis, etc.)"
    )
    
    parser.add_argument(
        "--diagnostics-dir",
        type=str,
        default=None,
        help="Custom directory for diagnostics (default: auto-generated in experiments/)"
    )
    
    # Output arguments
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save probability density figures"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom experiment name (default: auto-generated)"
    )

    # Checkpointing arguments
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        default=False,
        help="Save model checkpoints during training"
    )

    # HMM-specific analysis arguments
    parser.add_argument(
        "--hmm-csv-dir",
        type=str,
        default=None,
        help="Directory containing consolidated_task{N}.csv event files "
             "for HMM state-to-event alignment (used with --diagnostics --model hmm)"
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    # Generate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name:
        experiment_name = f"{args.experiment_name}_{timestamp}"
    else:
        experiment_name = f"{args.model.upper()}_{timestamp}"
    
    # Create experiment directory
    experiment_dir = EXPERIMENTS_DIR / f"task{args.task}" / f"paradigm{args.paradigm}" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    results_dir = experiment_dir / "results"
    figures_dir = experiment_dir / "figures"
    checkpoints_dir = experiment_dir / "model_checkpoints"
    
    for dir in [results_dir, figures_dir, checkpoints_dir]:
        dir.mkdir(exist_ok=True)
    
    # Diagnostics directory
    if args.diagnostics:
        if args.diagnostics_dir:
            diagnostics_dir = Path(args.diagnostics_dir)
        else:
            diagnostics_dir = experiment_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("\n" + "="*70)
    print("XDash Classification Experiment")
    print("="*70)
    print(f"Experiment: {experiment_name}")
    print(f"Task:       {args.task} ({TASK_NAMES.get(args.task, 'Unknown')})")
    print(f"Paradigm:   {args.paradigm} ({PARADIGM_NAMES.get(args.paradigm, 'Unknown')})")
    print(f"Model:      {args.model.upper()}")
    print(f"Preprocess: {args.method}")
    if args.method == "sliding_window":
        print(f"  Window:   {args.window_size}")
        print(f"  Overlap:  {args.overlap:.1%}")
    elif args.method == "downsample_truncate":
        print(f"  Original: {args.original_rate} Hz")
        print(f"  Target:   {args.target_rate} Hz")
    elif args.method == "dtw_embedding":
        print(f"  Components: {args.n_components}")
        print(f"  Method:     {args.dtw_method}")
    
    if args.diagnostics:
        print(f"\n🔬 Diagnostics: ENABLED")
        print(f"   Output:     {diagnostics_dir}")
    else:
        print(f"\n🔬 Diagnostics: DISABLED (use --diagnostics to enable)")
    
    print("="*70)
    
    # Save config
    config = {
        'task': args.task,
        'paradigm': args.paradigm,
        'model': args.model,
        'method': args.method,
        'timestamp': timestamp,
        'diagnostics_enabled': args.diagnostics
    }
    
    with open(experiment_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    patient_data, control_data = load_data(args.task)
    
    # Select paradigm
    selector = ParadigmSelector()
    g1, g0 = selector.select_paradigm(
        patient_data=patient_data,
        control_data=control_data,
        paradigm=args.paradigm
    )
    
    # ========================================================================
    # PREPROCESS
    # ========================================================================
    
    # Create preprocessor
    preproc_kwargs = {}
    if args.method == "sliding_window":
        preproc_kwargs["window_size"] = args.window_size
        preproc_kwargs["overlap"] = args.overlap
    elif args.method == "downsample_truncate":
        preproc_kwargs["target_rate"] = args.target_rate
        preproc_kwargs["original_rate"] = args.original_rate
    elif args.method == "dtw_embedding":
        preproc_kwargs["n_components"] = args.n_components
        preproc_kwargs["method"] = args.dtw_method
    
    preprocessor = PreprocessorFactory.create(
        method=args.method,
        model_type=args.model,
        **preproc_kwargs
    )

    if args.augment:
        preprocessor = AugmentedPreprocessor(
            base_preprocessor=preprocessor,
            augmentations=args.augment_methods,
            n_augmentations=args.n_augmentations
        )
    else:
        preprocessor = preprocessor
    
    X, y, subject_ids = preprocessor.prepare_data(g1, g0)
    
    # ========================================================================
    # TRAIN MODEL
    # ========================================================================
    
    model = create_model(args.model, 
                        checkpoints_dir = checkpoints_dir if args.save_checkpoints else None,         
                        patience=args.patience,
                        min_delta=args.min_delta,
                        task=args.task,
                        paradigm=args.paradigm
                        )

    results = model.fit(
        g1=g1,
        g0=g0,
        preprocessor=preprocessor,
        paradigm=args.paradigm
    )
    
    # ========================================================================
    # EVALUATE
    # ========================================================================
    
    from training.evaluator import ModelEvaluator, Visualizer
    
    evaluator = ModelEvaluator()
    eval_results = evaluator.evaluate(
        y_true=results.y_true,
        y_pred=results.y_pred,
        y_proba=results.y_proba,
        subject_ids=subject_ids
    )
    evaluator.print_report(eval_results)
    
    # Generate visualizations
    viz = Visualizer()
    
    viz.plot_confusion_matrix(
        cm=eval_results.confusion_matrix,
        class_names=['Group0', 'Group1'],
        normalize=True,
        title=f"Task-{args.task} - Paradigm{args.paradigm} - {args.model.upper()} - Confusion Matrix",
        save_path=figures_dir / "confusion_matrix.png"
    )
    
    viz.plot_roc_curve(
        y_true=results.y_true,
        y_proba=results.y_proba,
        title=f"Task-{args.task} - Paradigm{args.paradigm} - {args.model.upper()} - ROC Curve",
        save_path=figures_dir / "roc_curve.png"
    )
    
    viz.plot_probability_distribution(
        y_true=results.y_true,
        y_proba=results.y_proba,
        title=f"Task-{args.task} - Paradigm{args.paradigm} - {args.model.upper()} - Probability Distribution",
        save_path=figures_dir / "probability_distribution.png"
    )
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    results_dict = save_results(
        results=results,
        task=args.task,
        paradigm=args.paradigm,
        model_name=args.model.upper(),
        method=args.method,
        save_dir=results_dir
    )
    
    # ========================================================================
    # RUN DIAGNOSTICS
    # ========================================================================

    diagnostic_results = None
    if args.diagnostics:
        print(f"\n{'='*70}")
        print("RUNNING COMPREHENSIVE DIAGNOSTICS")
        print(f"{'='*70}\n")

        if args.model.lower() == "hmm":
            # ── HMM-specific post-processing analysis ────────────────────────
            # HMM has no gradients or activations — instead we run the
            # interpretability analysis built into HMMModel:
            #   - fit_for_analysis() on full dataset using LOO CV best params
            #   - emission distributions per state
            #   - transition matrix visualisation
            #   - state-specific feature importance + heatmap
            #   - patient vs control emission comparison
            #   - event alignment with CSV markers (if csv_dir provided)
            print("[HMM Diagnostics] Running HMM interpretability analysis ...")

            # g1/g0 and y are already in scope from preprocessing above
            # sequences_raw: variable-length list needed by HMM methods
            from sklearn.preprocessing import StandardScaler
            import numpy as np

            # Extract raw variable-length sequences (same as TruncatePreprocessor
            # but without stacking — HMM methods need a list not a 3D array)
            seqs_raw = []
            seq_labels = []
            seq_sids   = []
            for sid_key, tensor in {**g1, **g0}.items():
                arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
                arr = arr[:, 1:] if arr.shape[1] > 18 else arr
                seqs_raw.append(arr)
                seq_labels.append(1 if sid_key in g1 else 0)
                seq_sids.append(str(sid_key))

            y_raw   = np.array(seq_labels, dtype=np.int32)
            all_raw = np.vstack(seqs_raw)
            scaler  = StandardScaler().fit(all_raw)
            seqs_scaled = [scaler.transform(s) for s in seqs_raw]

            best = results.best_params    # from LOO CV — n_components, covariance_type, n_iter
            n_comp  = best.get("n_components",   2)
            cov_typ = best.get("covariance_type", "diag")
            n_iter  = best.get("n_iter",          100)

            print(f"  Using LOO CV best params: n_components={n_comp}, "
                  f"covariance_type={cov_typ}")

            # Fit on full dataset for interpretation
            model.fit_for_analysis(
                X=seqs_raw, y=y_raw,
                n_components=n_comp,
                covariance_type=cov_typ,
                n_iter=n_iter
            )

            # Emission distributions
            model.plot_emission_distributions(
                model=model.fitted_hmm1, channel_names=CHAN_NAME,
                title_suffix=f"Patient — Task {args.task} P{args.paradigm}",
                save_path=diagnostics_dir / "emissions_patient.png"
            )
            model.plot_emission_distributions(
                model=model.fitted_hmm0, channel_names=CHAN_NAME,
                title_suffix=f"Control — Task {args.task} P{args.paradigm}",
                save_path=diagnostics_dir / "emissions_control.png"
            )

            # Transition matrices
            model.plot_transition_matrix(
                model=model.fitted_hmm1,
                title_suffix=f"Patient — Task {args.task} P{args.paradigm}",
                save_path=diagnostics_dir / "transition_patient.png"
            )
            model.plot_transition_matrix(
                model=model.fitted_hmm0,
                title_suffix=f"Control — Task {args.task} P{args.paradigm}",
                save_path=diagnostics_dir / "transition_control.png"
            )

            # State-specific feature importance
            global_imp, state_imp = model.compute_state_specific_importance(
                model=model.fitted_hmm1, channel_names=CHAN_NAME
            )
            model.plot_state_importance_heatmap(
                state_importance=state_imp, channel_names=CHAN_NAME,
                title=f"State Feature Importance (Patient) T{args.task} P{args.paradigm}",
                save_path=diagnostics_dir / "state_importance_patient.png"
            )
            global_imp_ctrl, state_imp_ctrl = model.compute_state_specific_importance(
                model=model.fitted_hmm0, channel_names=CHAN_NAME
            )
            model.plot_state_importance_heatmap(
                state_importance=state_imp_ctrl, channel_names=CHAN_NAME,
                title=f"State Feature Importance (Control) T{args.task} P{args.paradigm}",
                save_path=diagnostics_dir / "state_importance_control.png"
            )

            # Patient vs control emission comparison
            try:
                model.compare_patient_control_emissions(
                    channel_names=CHAN_NAME,
                    save_path=diagnostics_dir / "emission_diff.png"
                )
            except ValueError as e:
                print(f"  Emission comparison skipped — {e}")

            # Decode + plot state sequence for first patient and first control
            for i, (seq_s, sid, lbl) in enumerate(
                zip(seqs_scaled, seq_sids, y_raw)
            ):
                if i == 0 or (lbl == 0 and all(y_raw[:i] == 1)):
                    group = "patient" if lbl == 1 else "control"
                    states, _, _ = model.decode_sequence(
                        seq_s, model.fitted_hmm1, sampling_rate=50
                    )
                    model.plot_state_sequence_over_time(
                        sequence=seq_s, state_sequence=states,
                        channel_idx=0,
                        title=f"State Seq — {sid} ({group}) T{args.task} P{args.paradigm}",
                        save_path=diagnostics_dir / f"state_seq_{group}_{sid}.png"
                    )

            # Event alignment — only if CSV dir is provided via --hmm-csv-dir
            if hasattr(args, "hmm_csv_dir") and args.hmm_csv_dir:
                csv_path = Path(args.hmm_csv_dir) / f"consolidated_task{args.task}.csv"
                if csv_path.exists():
                    model.run_alignment_analysis(
                        sequences=seqs_scaled,
                        subject_ids=seq_sids,
                        csv_path=csv_path,
                        task_id=args.task,
                        paradigm_id=args.paradigm,
                        tolerance_s=0.5,
                        sampling_rate=50,
                        save_path=diagnostics_dir / f"alignment_T{args.task}_P{args.paradigm}.csv"
                    )
                else:
                    print(f"  Event CSV not found at {csv_path} — alignment skipped")

            diagnostic_results = {
                "hmm_analysis": {
                    "n_components":          n_comp,
                    "covariance_type":       cov_typ,
                    "param_source":          "loo_cv_best_params",
                    "global_importance":     global_imp,
                    "global_importance_ctrl": global_imp_ctrl,
                }
            }
            print(f"\n[HMM Diagnostics] Complete — outputs in {diagnostics_dir}")

        else:
            # ── CNN / RNN / Transformer diagnostics ───────────────
            from utils.comprehensive_monitor import run_complete_monitoring
            import numpy as np
            diagnostic_results = run_complete_monitoring(
                model=model,
                X=X,
                y=y,
                subject_ids=subject_ids,
                fold_results=results.per_fold_results,
                experiment_name=experiment_name,
                save_dir=diagnostics_dir,
                hyperparameters=results.best_params,
                feature_names=CHAN_NAME
            )
    
    # ========================================================================
    # SAVE SUMMARY
    # ========================================================================
    
    summary = {
        'experiment_name': experiment_name,
        'config': config,
        'results': results_dict,
        'evaluation': {
            'accuracy': eval_results.accuracy,
            'balanced_accuracy': eval_results.balanced_accuracy,
            'auc_roc': eval_results.auc_roc,
            'auc_roc_ci': eval_results.auc_roc_ci
        }
    }
    
    if diagnostic_results:
        summary['diagnostics'] = {
            'overfitting_risk': diagnostic_results.get('overfitting', {}).get('risk', 'N/A'),
            'generalization_gap': diagnostic_results.get('overfitting', {}).get('generalization_gap', 'N/A')
        }
    
    with open(experiment_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # ========================================================================
    # PRINT FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Experiment:       {experiment_name}")
    print(f"Balanced Accuracy: {results.metrics['ba']:.3f}")
    print(f"AUC:              {results.metrics['auc']:.3f}")
    print(f"Recall:           {results.metrics['recall']:.3f}")
    
    if diagnostic_results:
        print(f"\n🔬 Diagnostics:")
        print(f"   Overfitting Risk: {diagnostic_results.get('overfitting', {}).get('risk', 'N/A')}")
        print(f"   Saved to:         {diagnostics_dir}")
    
    print(f"\n📁 All outputs:")
    print(f"   Experiment dir: {experiment_dir}")
    print(f"   Results:       {results_dir}")
    print(f"   Figures:       {figures_dir}")
    print(f"   Checkpoints:   {checkpoints_dir}")
    if args.diagnostics:
        print(f"   Diagnostics:   {diagnostics_dir}")
    
    print("\nTop 6 Features:")
    for i, (feat, imp) in enumerate(list(results.feature_importance.items())[:6]):
        print(f"  {i+1}. {feat}: {imp:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()