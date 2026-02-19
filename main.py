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


def create_model(model_type, checkpoints_dir=None, patience=None,
                 min_delta=None, task=None, paradigm=None):
    model_type = model_type.lower()
    if model_type == "hmm":
        return HMMModel(...)
    elif model_type == "cnn":
        return CNNModel(...)
    elif model_type == "rnn":
        return RNNModel(...)
    elif model_type == "transformer":          # ADD THIS
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
        default = False,
        action="store_true",
        help="Save model checkpoints during training"
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
        
        from utils.comprehensive_monitor import run_complete_monitoring
        import numpy as np
        # diagnostic_results = run_diagnostics(
        #     model=model,
        #     X=X,
        #     y=y,
        #     subject_ids=subject_ids,
        #     results=results,
        #     experiment_name=experiment_name,
        #     diagnostics_dir=diagnostics_dir
        # )
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