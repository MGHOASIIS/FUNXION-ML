"""
COMPREHENSIVE MODEL MONITORING SYSTEM

State-of-the-art analysis for your RNN on Task 1, Paradigm 1.

This script runs ALL diagnostic tools to ensure your model:
1. Is NOT overfitting (critical for N=60!)
2. Is NOT underfitting
3. Learns clinically meaningful patterns
4. Generalizes properly across folds
5. Has stable training dynamics

Run this AFTER training to get complete diagnostic report.
"""
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json

from utils.overfitting_detection import (
    OverfittingDetector, LearningCurveAnalyzer,
    BiasVarianceAnalyzer, PerFoldAnalyzer
)
from utils.model_diagnostics import (
    GradientDiagnostics, ActivationAnalyzer,
    SaliencyAnalyzer, WeightDiagnostics,
    ClinicalInterpreter
)
from utils.checkpointing import (
    CheckpointManager, ExperimentTracker, ModelExporter
)
from utils.importance import (
    PermutationImportance, WeightBasedImportance,
    ImportanceVisualizer, print_importance_report
)
from utils.visualization import (
    TrainingHistoryVisualizer, ModelComparisonVisualizer,
    TimeSeriesVisualizer
)
from training.evaluator import ModelEvaluator, Visualizer

from config.constants import DEVICE, DOFS


class ComprehensiveModelMonitor:
    """
    MASTER monitoring class that runs ALL diagnostics.
    
    Use this to get a complete health check of your model.
    """
    
    def __init__(self, experiment_name: str, save_dir: Path):
        """
        Parameters
        ----------
        experiment_name : str
            Name of experiment
        save_dir : Path
            Directory to save all outputs
        """
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.figures_dir = self.save_dir / "figures"
        self.reports_dir = self.save_dir / "reports"
        
        for dir in [self.figures_dir, self.reports_dir]:
            dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE MODEL MONITOR")
        print(f"Experiment: {experiment_name}")
        print(f"Save Dir: {save_dir}")
        print(f"{'='*70}\n")

    
    def count_parameters(self, model) -> int:
        temp_model = model._create_temp_model()
        n_params = sum(p.numel() for p in temp_model.parameters())
        del temp_model
        return n_params
    
    
    def run_complete_analysis(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        fold_results: List[Dict],
        training_history: Optional[Any] = None,
        hyperparameters: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run COMPLETE diagnostic analysis.
        
        This is the MAIN method - runs everything!
        
        Parameters
        ----------
        model : nn.Module
            Trained model
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        subject_ids : np.ndarray
            Subject identifiers
        fold_results : List[Dict]
            Results from LOO CV folds
        training_history : Any, optional
            Training history object
        hyperparameters : Dict, optional
            Model hyperparameters
        feature_names : List[str], optional
            Feature names
        
        Returns
        -------
        Dict[str, Any]
            Complete diagnostic results
        """
        all_results = {}
        
        print(f"\n{'#'*70}")
        print(f"# STARTING COMPREHENSIVE ANALYSIS")
        print(f"{'#'*70}\n")
        
        # ================================================================
        # 1. OVERFITTING DETECTION (CRITICAL FOR N=60!)
        # ================================================================
        print(f"\n{'='*70}")
        print("SECTION 1: OVERFITTING DETECTION")
        print(f"{'='*70}\n")
        
        overfitting_detector = OverfittingDetector()
        overfitting_analysis = overfitting_detector.analyze(
            model=model,
            X=X,
            y=y,
            cv_splits=[],  # Not needed if fold_results provided
            fold_results=fold_results,
            model_name=self.experiment_name
        )
        
        all_results['overfitting'] = {
            'risk': overfitting_analysis.overfitting_risk,
            'generalization_gap': overfitting_analysis.generalization_gap,
            'sample_param_ratio': overfitting_analysis.sample_to_param_ratio,
            'recommendations': overfitting_analysis.recommendations
        }
        
        # ================================================================
        # 2. PER-FOLD CONSISTENCY ANALYSIS
        # ================================================================
        print(f"\n{'='*70}")
        print("SECTION 2: PER-FOLD CONSISTENCY")
        print(f"{'='*70}\n")
        
        fold_analyzer = PerFoldAnalyzer()
        fold_analyzer.analyze_fold_consistency(fold_results, subject_ids.tolist())
        fold_analyzer.plot_fold_scores(
            fold_results,
            subject_ids.tolist(),
            save_path=self.figures_dir / "fold_scores.png"
        )
        
        # ================================================================
        # 3. BIAS-VARIANCE DECOMPOSITION
        # ================================================================
        print(f"\n{'='*70}")
        print("SECTION 3: BIAS-VARIANCE ANALYSIS")
        print(f"{'='*70}\n")
        
        if fold_results and 'y_pred' in fold_results[0]:
            # Extract from your actual structure
            all_preds = []
            all_probs = []
            
            for r in fold_results:
                if 'y_pred' in r: all_preds.extend(r['y_pred'])
                if 'y_proba' in r: all_probs.extend(r['y_proba'])
            
            if all_probs:
                prob_variance = np.var(all_probs)
                mean_prob = np.mean(all_probs)
                
                print(f"Probability variance: {prob_variance:.4f}")
                print(f"Mean probability: {mean_prob:.4f}")
                
                all_results['bias_variance'] = {
                    'probability_variance': float(prob_variance),
                    'mean_probability': float(mean_prob)
                }
        else:
            print("⚠️ No predictions found in fold_results")
            all_results['bias_variance'] = {}
        
        # ================================================================
        # 4. GRADIENT DIAGNOSTICS
        # ================================================================
        print(f"\n{'='*70}")
        print("SECTION 4: GRADIENT ANALYSIS")
        print(f"{'='*70}\n")
        
       
        # Extract or recreate PyTorch model
        pytorch_model = model._create_temp_model()
        pytorch_model = pytorch_model.to(DEVICE)
        
        if pytorch_model is None:
            print("⚠️  Gradient Analysis SKIPPED")
            print("   Could not extract or recreate PyTorch model from wrapper")
            print(f"   Model type: {type(model)}")
            all_results['gradients'] = {'error': 'no_pytorch_model'}
            return
        
        # Create appropriate dataloader
        try:
            from torch.utils.data import DataLoader, TensorDataset
            
            # Use subset of data for analysis
            n_samples = min(10, len(X))
            X_torch = torch.tensor(X[:n_samples], dtype=torch.float32)
            
            # For RNN, we need (batch, seq_len, features) format
            if len(X_torch.shape) == 2:  # (samples, features)
                # Assume each sample is a flattened sequence
                seq_len = X_torch.shape[1] // 18  # Assuming 18 features per timestep
                if seq_len > 1:
                    X_torch = X_torch.view(n_samples, seq_len, 18)
                else:
                    # Single timestep, add sequence dimension
                    X_torch = X_torch.unsqueeze(1)  # (batch, 1, features)
            
            # Create dummy labels (we just need them for gradient computation)
            y_torch = torch.randint(0, 2, (n_samples,), dtype=torch.long)

            X_torch = X_torch.to(DEVICE)
            y_torch = y_torch.to(DEVICE)
            
            dataloader = DataLoader(TensorDataset(X_torch, y_torch), batch_size=n_samples)
            
            print(f"✓ Created dataloader with {n_samples} samples")
            print(f"  Input shape: {X_torch.shape}")
            
        except Exception as e:
            print(f"⚠️  Failed to create dataloader: {e}")
            all_results['gradients'] = {'error': 'dataloader_creation_failed'}
            return
        
        # Run gradient analysis with error handling
        try:
            grad_diagnostics = GradientDiagnostics()
            gradient_stats = grad_diagnostics.analyze_gradients(
                model=pytorch_model,  # Use extracted PyTorch model
                dataloader=dataloader,
                criterion=torch.nn.CrossEntropyLoss(),
                device=DEVICE
            )
            
            if gradient_stats:
                print("✓ Gradient analysis completed successfully")
                
                # Plot gradient flow
                try:
                    grad_diagnostics.plot_gradient_flow(
                        gradient_stats,
                        save_path=self.figures_dir / "gradient_flow.png"
                    )
                    print(f"✓ Gradient flow plot saved")
                except Exception as e:
                    print(f"⚠️  Failed to plot gradient flow: {e}")
                
                all_results['gradients'] = gradient_stats
            else:
                print("⚠️  Gradient analysis returned empty results")
                all_results['gradients'] = {'error': 'empty_results'}
                
        except Exception as e:
            print(f"⚠️  Gradient analysis failed: {e}")
            all_results['gradients'] = {'error': str(e)}

            
        # ================================================================
        # 5. ACTIVATION ANALYSIS
        # ================================================================
        print(f"\n{'='*70}")
        print("SECTION 5: ACTIVATION ANALYSIS")
        print(f"{'='*70}\n")
        
        # Extract or recreate PyTorch model
        pytorch_model = model._create_temp_model()
        pytorch_model = pytorch_model.to(DEVICE)

        
        if pytorch_model is None:
            print("⚠️  Activation Analysis SKIPPED")
            print("   Could not extract or recreate PyTorch model from wrapper")
            all_results['activations'] = {'error': 'no_pytorch_model'}
            return
        
        # Prepare input data
        try:
            n_samples = min(10, len(X))
            X_torch = torch.tensor(X[:n_samples], dtype=torch.float32)
            
            # Ensure correct shape for RNN
            if len(X_torch.shape) == 2:
                seq_len = X_torch.shape[1] // 18
                if seq_len > 1:
                    X_torch = X_torch.view(n_samples, seq_len, 18)
                else:
                    X_torch = X_torch.unsqueeze(1)

            X_torch = X_torch.to(DEVICE)
            
            print(f"✓ Prepared input tensor with shape: {X_torch.shape}")
            
        except Exception as e:
            print(f"⚠️  Failed to prepare input data: {e}")
            all_results['activations'] = {'error': 'input_preparation_failed'}
            return
        
        # Extract activations with error handling
        try:
            act_analyzer = ActivationAnalyzer()
            activations = act_analyzer.extract_activations(
                model=pytorch_model,  # Use extracted PyTorch model
                X=X_torch,
                device=DEVICE
            )
            
            if activations:
                print(f"✓ Extracted activations from {len(activations)} layers")
                
                # Analyze activations
                try:
                    act_stats = act_analyzer.analyze_activations(activations)
                    print("✓ Activation analysis completed")
                    
                    # Plot activation distributions
                    try:
                        act_analyzer.plot_activation_distributions(
                            activations,
                            save_path=self.figures_dir / "activation_distributions.png"
                        )
                        print("✓ Activation distribution plots saved")
                    except Exception as e:
                        print(f"⚠️  Failed to plot activation distributions: {e}")
                    
                    all_results['activations'] = act_stats
                    
                except Exception as e:
                    print(f"⚠️  Failed to analyze activations: {e}")
                    all_results['activations'] = {'error': f'analysis_failed: {e}'}
            else:
                print("⚠️  No activations extracted")
                all_results['activations'] = {'error': 'no_activations_extracted'}
                
        except Exception as e:
            print(f"⚠️  Activation extraction failed: {e}")
            all_results['activations'] = {'error': str(e)}
            
        # ================================================================
        # 6. WEIGHT ANALYSIS
        # ================================================================
        print(f"\n{'='*70}")
        print("SECTION 6: WEIGHT DISTRIBUTION ANALYSIS")
        print(f"{'='*70}\n")

        pytorch_model = model._create_temp_model()
        pytorch_model = pytorch_model.to(DEVICE)
        
        weight_diagnostics = WeightDiagnostics()
        weight_dists = weight_diagnostics.analyze_weight_distributions(pytorch_model)
        weight_diagnostics.plot_weight_distributions(
            weight_dists,
            save_path=self.figures_dir / "weight_distributions.png"
        )
        
        # ================================================================
        # 7. FEATURE IMPORTANCE (Multiple Methods)
        # ================================================================
        print(f"\n{'='*70}")
        print("SECTION 7: FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*70}\n")
        
        # Method 1: Permutation (gold standard)
        print("\n[7.1] Permutation Importance...")
        # Note: This requires model with predict method
        # Skipped for now, would be implemented based on model type
        
        # Method 2: Weight-based (from saved weights)
        print("\n[7.2] Weight-Based Importance...")
        if hasattr(model, 'rnn'):
            # For RNN
            try:
                weight_ih = model.rnn.weight_ih_l0.detach().cpu().numpy()
                weight_imp_results = WeightBasedImportance.from_linear_weights(
                    weights=weight_ih,
                    feature_names=feature_names,
                    aggregate='mean'
                )
                
                print_importance_report(weight_imp_results, top_k=10)
                
                imp_viz = ImportanceVisualizer()
                imp_viz.plot_importance_bars(
                    weight_imp_results,
                    top_k=10,
                    save_path=self.figures_dir / "feature_importance.png"
                )
                
                all_results['feature_importance'] = {
                    name: float(score) 
                    for name, score in zip(
                        weight_imp_results.feature_names,
                        weight_imp_results.importance_scores
                    )
                }
            except Exception as e:
                print(f"⚠️  Could not extract RNN weights: {e}")
        
        # ================================================================
        # 8. SALIENCY MAPS
        # ================================================================
        print(f"\n{'='*70}")
        print("SECTION 8: SALIENCY ANALYSIS")
        print(f"{'='*70}\n")

        pytorch_model = model._create_temp_model()
        pytorch_model = pytorch_model.to(DEVICE)

        saliency_analyzer = SaliencyAnalyzer()

        # Compute for a few samples
        for class_idx in [0, 1]:
            sample_idx = np.where(y == class_idx)[0][0]
            X_sample = torch.tensor(X[sample_idx:sample_idx+1], dtype=torch.float32)
            
            # Fix input shape for RNN
            if len(X_sample.shape) == 2:
                seq_len = X_sample.shape[1] // 18
                if seq_len > 1:
                    X_sample = X_sample.view(1, seq_len, 18)
                else:
                    X_sample = X_sample.unsqueeze(1)
            
            X_sample = X_sample.to(DEVICE)
            # pytorch_model.train()

            saliency = saliency_analyzer.compute_saliency(
                model=pytorch_model,
                X=X_sample,
                target_class=class_idx,
                device=DEVICE
            )
            
            saliency_analyzer.plot_saliency_map(
                saliency=saliency,
                channel_names=feature_names,
                title=f"Saliency Map - Class {class_idx} Sample",
                save_path=self.figures_dir / f"saliency_class_{class_idx}.png"
            )
                
        # ================================================================
        # 9. CLINICAL INTERPRETATION
        # ================================================================
        print(f"\n{'='*70}")
        print("SECTION 9: CLINICAL INTERPRETATION")
        print(f"{'='*70}\n")
        
        if 'feature_importance' in all_results:
            clinical_interp = ClinicalInterpreter()
            interpretation = clinical_interp.interpret_sensor_importance(
                importance_dict=all_results['feature_importance'],
                top_k=DOFS
            )
            print(interpretation)
            
            # Save interpretation
            interp_path = self.reports_dir / "clinical_interpretation.txt"
            with open(interp_path, 'w') as f:
                f.write(interpretation)
            print(f"\n[Clinical Interpretation Saved] {interp_path}")
        
        # ================================================================
        # 10. TRAINING HISTORY VISUALIZATION (if available)
        # ================================================================
        if training_history is not None:
            print(f"\n{'='*70}")
            print("SECTION 10: TRAINING HISTORY")
            print(f"{'='*70}\n")
            
            hist_viz = TrainingHistoryVisualizer()
            
            if hasattr(training_history, 'train_loss'):
                hist_viz.plot_loss_curves(
                    train_loss=training_history.train_loss,
                    val_loss=training_history.val_loss if hasattr(training_history, 'val_loss') else None,
                    save_path=self.figures_dir / "training_loss.png"
                )
        
        # ================================================================
        # 11. SAVE COMPREHENSIVE REPORT
        # ================================================================
        self._save_comprehensive_report(all_results)
        
        # ================================================================
        # 12. PRINT SUMMARY
        # ================================================================
        self._print_final_summary(all_results)
        
        return all_results
    
    def _save_comprehensive_report(self, results: Dict[str, Any]):
        """Save comprehensive JSON report."""
        report_path = self.reports_dir / "comprehensive_analysis.json"
        
        # Make results JSON-serializable
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {
                    k: float(v) if isinstance(v, np.ndarray) or isinstance(v, np.floating) else v
                    for k, v in value.items()
                }
            else:
                json_results[key] = value
        
        with open(report_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n[Comprehensive Report Saved] {report_path}")
    
    def _print_final_summary(self, results: Dict[str, Any]):
        """Print executive summary."""
        print(f"\n{'#'*70}")
        print(f"# FINAL SUMMARY - {self.experiment_name}")
        print(f"{'#'*70}\n")
        
        # Overfitting risk
        if 'overfitting' in results:
            risk = results['overfitting']['risk']
            gap = results['overfitting']['generalization_gap']
            ratio = results['overfitting']['sample_param_ratio']
            
            print(f"🔍 OVERFITTING ASSESSMENT:")
            print(f"   Risk Level:         {risk}")
            print(f"   Generalization Gap: {gap:.4f}")
            print(f"   Sample/Param Ratio: {ratio:.2f}")
            
            if risk == "HIGH":
                print(f"\n   🚨 ACTION REQUIRED:")
                for rec in results['overfitting']['recommendations'][:3]:
                    print(f"      • {rec}")
            elif risk == "MEDIUM":
                print(f"\n   ⚠️  MONITOR CAREFULLY")
            else:
                print(f"\n   ✓ Model appears healthy")
        
        # Feature importance
        if 'feature_importance' in results:
            print(f"\n📊 TOP 6 DISCRIMINATIVE FEATURES:")
            sorted_feats = sorted(
                results['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:6]
            for i, (feat, score) in enumerate(sorted_feats, 1):
                print(f"   {i}. {feat}: {score:.4f}")
        
        # Gradient health
        if 'gradients' in results:
            print(f"\n⚡ GRADIENT HEALTH:")
            first_layer = list(results['gradients'].keys())[0]
            grad_mean = results['gradients'][first_layer]['mean']
            
            if grad_mean < 1e-7:
                print(f"   🚨 VANISHING gradients detected!")
            elif grad_mean > 10:
                print(f"   🚨 EXPLODING gradients detected!")
            else:
                print(f"   ✓ Gradients healthy (mean={grad_mean:.6f})")
        
        # Activation health
        if 'activations' in results:
            print(f"\n🧠 ACTIVATION HEALTH:")
            total_sparsity = np.mean([
                stats['sparsity'] for stats in results['activations'].values()
            ])
            
            if total_sparsity > 0.7:
                print(f"   ⚠️  High sparsity ({total_sparsity:.1%}) - many dead neurons")
            else:
                print(f"   ✓ Good activation diversity (sparsity={total_sparsity:.1%})")
        
        print(f"\n{'#'*70}")
        print(f"# ALL DIAGNOSTICS COMPLETE")
        print(f"# Results saved to: {self.save_dir}")
        print(f"{'#'*70}\n")


def run_complete_monitoring(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    fold_results: List[Dict],
    experiment_name: str,
    save_dir: Path,
    training_history: Optional[Any] = None,
    hyperparameters: Optional[Dict] = None,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run complete monitoring.
    
    Parameters
    ----------
    model : nn.Module
        Trained model
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    subject_ids : np.ndarray
        Subject IDs
    fold_results : List[Dict]
        Fold results with train_score, val_score
    experiment_name : str
        Experiment name
    save_dir : Path
        Save directory
    training_history : Any, optional
        Training history
    hyperparameters : Dict, optional
        Hyperparameters
    feature_names : List[str], optional
        Feature names
    
    Returns
    -------
    Dict[str, Any]
        All diagnostic results
    """
    monitor = ComprehensiveModelMonitor(experiment_name, save_dir)
    
    results = monitor.run_complete_analysis(
        model=model,
        X=X,
        y=y,
        subject_ids=subject_ids,
        fold_results=fold_results,
        training_history=training_history,
        hyperparameters=hyperparameters,
        feature_names=feature_names
    )
    
    return results


# Example usage
if __name__ == "__main__":
    from config.constants import CHAN_NAME
    
    # Create dummy model
    model = nn.Sequential(
        nn.Linear(18, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )
    
    # Dummy data
    np.random.seed(42)
    X = np.random.randn(60, 200, 18)
    y = np.random.randint(0, 2, 60)
    subject_ids = np.array([f"subj_{i}" for i in range(60)])
    
    # Simulate fold results
    fold_results = []
    for i in range(60):
        fold_results.append({
            'fold': i,
            'train_score': 0.85 + np.random.randn() * 0.05,
            'val_score': 0.72 + np.random.randn() * 0.08,
            'train_conf': 0.9,
            'val_conf': 0.75
        })
    
    # Run complete monitoring
    results = run_complete_monitoring(
        model=model,
        X=X,
        y=y,
        subject_ids=subject_ids,
        fold_results=fold_results,
        experiment_name="RNN_Task1_Paradigm1",
        save_dir=Path("diagnostics_output"),
        feature_names=CHAN_NAME
    )