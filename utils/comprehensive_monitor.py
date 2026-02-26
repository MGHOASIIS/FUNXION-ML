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
    OverfittingDetector, PerFoldAnalyzer
)
from utils.model_diagnostics import (
    GradientDiagnostics, ActivationAnalyzer,
    SaliencyAnalyzer, WeightDiagnostics,
    ClinicalInterpreter
)
from utils.importance import (
    WeightBasedImportance,
    ImportanceVisualizer, print_importance_report
)
from utils.visualization import (
    TrainingHistoryVisualizer
)

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
        if temp_model is None or not isinstance(temp_model, nn.Module):
            return 0
        n_params = sum(p.numel() for p in temp_model.parameters())
        del temp_model
        return n_params

    @staticmethod
    def _get_pytorch_model(model) -> Optional[nn.Module]:
        """
        Safely extract a PyTorch nn.Module from a model wrapper.

        Returns None for non-PyTorch models (e.g. HMM) so all
        PyTorch-specific sections (gradient, activation, weight, saliency)
        can gate themselves with a single check instead of crashing on .to().

        Parameters
        ----------
        model : BaseModel subclass
            Any wrapper with a _create_temp_model() method.

        Returns
        -------
        nn.Module or None
            Model moved to DEVICE, or None if not a PyTorch model.
        """
        temp = model._create_temp_model()
        if temp is None or not isinstance(temp, nn.Module):
            return None
        return temp.to(DEVICE)
    
    
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
        consistency_stats = fold_analyzer.analyze_fold_consistency(fold_results, subject_ids.tolist())
        all_results['fold_consistency'] = consistency_stats
        fold_analyzer.plot_fold_scores(
            fold_results,
            subject_ids.tolist(),
            save_path=self.figures_dir / "fold_scores.png"
        )
        
        # ================================================================
        # ================================================================
        # 3. PREDICTION CONFIDENCE ANALYSIS
        # ================================================================
        print(f"\n{'='*70}")
        print("SECTION 3: PREDICTION CONFIDENCE ANALYSIS")
        print(f"{'='*70}\n")

        if fold_results and 'y_pred' in fold_results[0] and 'y_proba' in fold_results[0]:

            # Collect per-subject data from fold results
            records = []
            for r in fold_results:
                if not r.get('y_pred') or not r.get('y_proba') or not r.get('y_true'):
                    continue
                true_label = int(r['y_true'][0])
                pred_label = int(r['y_pred'][0])
                pred_prob  = float(r['y_proba'][0])  # prob of class 1
                correct    = (pred_label == true_label)
                # Calibration error: distance between predicted prob and true label
                calib_error = abs(true_label - pred_prob)
                records.append({
                    'true_label':   true_label,
                    'pred_label':   pred_label,
                    'pred_prob':    pred_prob,
                    'correct':      correct,
                    'calib_error':  calib_error
                })

            if records:
                all_probs   = [r['pred_prob']   for r in records]
                all_correct = [r['correct']      for r in records]
                all_errors  = [r['calib_error']  for r in records]
                all_labels  = [r['true_label']   for r in records]

                # ── 1. Overall calibration ────────────────────────────────────
                mean_calib_error = float(np.mean(all_errors))
                std_calib_error  = float(np.std(all_errors))

                print(f"1. CALIBRATION ERROR (|true_label - pred_prob|)")
                print(f"   Mean:  {mean_calib_error:.4f}  (0=perfect, 1=worst)")
                print(f"   Std:   {std_calib_error:.4f}  (high = inconsistent across subjects)")
                if mean_calib_error < 0.2:
                    calib_verdict = "WELL CALIBRATED"
                    print(f"   [OK] {calib_verdict}")
                elif mean_calib_error < 0.4:
                    calib_verdict = "MODERATELY CALIBRATED"
                    print(f"   [!] {calib_verdict}")
                else:
                    calib_verdict = "POORLY CALIBRATED"
                    print(f"   [!!] {calib_verdict}")

                # ── 2. Confidence when correct vs wrong ───────────────────────
                conf_correct = [r['pred_prob'] if r['true_label']==1 else 1-r['pred_prob']
                                for r in records if r['correct']]
                conf_wrong   = [r['pred_prob'] if r['true_label']==1 else 1-r['pred_prob']
                                for r in records if not r['correct']]

                mean_conf_correct = float(np.mean(conf_correct)) if conf_correct else 0.0
                mean_conf_wrong   = float(np.mean(conf_wrong))   if conf_wrong   else 0.0

                print(f"\n2. CONFIDENCE WHEN CORRECT vs WRONG")
                print(f"   Mean confidence when CORRECT: {mean_conf_correct:.4f}")
                print(f"   Mean confidence when WRONG:   {mean_conf_wrong:.4f}")
                if mean_conf_correct > mean_conf_wrong:
                    print(f"   [OK] Model is more confident when correct (healthy)")
                else:
                    print(f"   [!!] Model equally/more confident when WRONG (overconfident on errors)")

                # ── 3. Patient vs control confidence split ────────────────────
                prob_patients  = [r['pred_prob'] for r in records if r['true_label'] == 1]
                prob_controls  = [r['pred_prob'] for r in records if r['true_label'] == 0]

                mean_prob_pat  = float(np.mean(prob_patients)) if prob_patients  else 0.0
                mean_prob_ctrl = float(np.mean(prob_controls)) if prob_controls  else 0.0
                separation     = float(abs(mean_prob_pat - mean_prob_ctrl))

                print(f"\n3. PATIENT vs CONTROL CONFIDENCE SPLIT")
                print(f"   Mean prob(class=1) for PATIENTS:  {mean_prob_pat:.4f}")
                print(f"   Mean prob(class=1) for CONTROLS:  {mean_prob_ctrl:.4f}")
                print(f"   Separation:                        {separation:.4f}")
                if separation > 0.3:
                    sep_verdict = "GOOD SEPARATION — model distinguishes groups"
                    print(f"   [OK] {sep_verdict}")
                elif separation > 0.1:
                    sep_verdict = "MODERATE SEPARATION"
                    print(f"   [!] {sep_verdict}")
                else:
                    sep_verdict = "POOR SEPARATION — model struggles to distinguish groups"
                    print(f"   [!!] {sep_verdict}")

                # ── 4. Overconfident errors ───────────────────────────────────
                overconf_threshold = 0.85
                overconf_errors = [
                    r for r in records
                    if not r['correct'] and (
                        (r['true_label']==1 and r['pred_prob'] < (1 - overconf_threshold)) or
                        (r['true_label']==0 and r['pred_prob'] > overconf_threshold)
                    )
                ]
                n_overconf = len(overconf_errors)
                print(f"\n4. OVERCONFIDENT ERRORS (wrong but prob > {overconf_threshold})")
                print(f"   Count: {n_overconf} / {len(records)} subjects")
                if n_overconf == 0:
                    overconf_verdict = "NONE — no dangerous overconfident errors"
                    print(f"   [OK] {overconf_verdict}")
                elif n_overconf <= 3:
                    overconf_verdict = f"{n_overconf} overconfident errors — monitor these subjects"
                    print(f"   [!] {overconf_verdict}")
                else:
                    overconf_verdict = f"{n_overconf} overconfident errors — model dangerously overconfident"
                    print(f"   [!!] {overconf_verdict}")

                # ── Save results ──────────────────────────────────────────────
                all_results['prediction_confidence'] = {
                    'n_subjects':             len(records),
                    'mean_calibration_error': mean_calib_error,
                    'std_calibration_error':  std_calib_error,
                    'calibration_verdict':    calib_verdict,
                    'mean_conf_when_correct': mean_conf_correct,
                    'mean_conf_when_wrong':   mean_conf_wrong,
                    'mean_prob_patients':     mean_prob_pat,
                    'mean_prob_controls':     mean_prob_ctrl,
                    'group_separation':       separation,
                    'separation_verdict':     sep_verdict,
                    'n_overconfident_errors': n_overconf,
                    'overconfidence_verdict': overconf_verdict,
                }
            else:
                print("[!] No valid per-subject records found in fold_results")
                all_results['prediction_confidence'] = {}
        else:
            print("[!] No predictions found in fold_results — skipping")
            all_results['prediction_confidence'] = {}

        # ================================================================
        # 4. GRADIENT DIAGNOSTICS
        # ================================================================
        print(f"\n{'='*70}")
        print("SECTION 4: GRADIENT ANALYSIS")
        print(f"{'='*70}\n")
        
        # Extract or recreate PyTorch model
        pytorch_model = self._get_pytorch_model(model)

        if pytorch_model is None:
            print("âš ï¸  Gradient Analysis SKIPPED")
            print("   Model is not a PyTorch nn.Module (e.g. HMM â€” no gradients)")
            print(f"   Model type: {type(model).__name__}")
            all_results['gradients'] = {'error': 'no_pytorch_model'}
        else:
            try:
                from torch.utils.data import DataLoader, TensorDataset

                n_samples = min(10, len(X))
                X_torch = torch.tensor(X[:n_samples], dtype=torch.float32)

                if len(X_torch.shape) == 2:
                    seq_len = X_torch.shape[1] // 18
                    if seq_len > 1:
                        X_torch = X_torch.view(n_samples, seq_len, 18)
                    else:
                        X_torch = X_torch.unsqueeze(1)

                y_torch = torch.randint(0, 2, (n_samples,), dtype=torch.long)
                X_torch = X_torch.to(DEVICE)
                y_torch = y_torch.to(DEVICE)

                dataloader = DataLoader(TensorDataset(X_torch, y_torch), batch_size=n_samples)
                print(f"âœ“ Created dataloader with {n_samples} samples, shape: {X_torch.shape}")

                grad_diagnostics = GradientDiagnostics()
                gradient_stats = grad_diagnostics.analyze_gradients(
                    model=pytorch_model,
                    dataloader=dataloader,
                    criterion=torch.nn.CrossEntropyLoss(),
                    device=DEVICE
                )

                if gradient_stats:
                    print("âœ“ Gradient analysis completed successfully")
                    try:
                        grad_diagnostics.plot_gradient_flow(
                            gradient_stats,
                            save_path=self.figures_dir / "gradient_flow.png"
                        )
                        print("âœ“ Gradient flow plot saved")
                    except Exception as e:
                        print(f"âš ï¸  Failed to plot gradient flow: {e}")
                    all_results['gradients'] = gradient_stats
                else:
                    print("âš ï¸  Gradient analysis returned empty results")
                    all_results['gradients'] = {'error': 'empty_results'}

            except Exception as e:
                print(f"âš ï¸  Gradient analysis failed: {e}")
                all_results['gradients'] = {'error': str(e)}

            
        # ================================================================
        # 5. ACTIVATION ANALYSIS
        # ================================================================
        print(f"\n{'='*70}")
        print("SECTION 5: ACTIVATION ANALYSIS")
        print(f"{'='*70}\n")
        
        # Extract or recreate PyTorch model
        pytorch_model = self._get_pytorch_model(model)

        if pytorch_model is None:
            print("âš ï¸  Activation Analysis SKIPPED")
            print("   Model is not a PyTorch nn.Module (e.g. HMM â€” no activations)")
            all_results['activations'] = {'error': 'no_pytorch_model'}
        else:
            try:
                n_samples = min(10, len(X))
                X_torch = torch.tensor(X[:n_samples], dtype=torch.float32)

                if len(X_torch.shape) == 2:
                    seq_len = X_torch.shape[1] // 18
                    if seq_len > 1:
                        X_torch = X_torch.view(n_samples, seq_len, 18)
                    else:
                        X_torch = X_torch.unsqueeze(1)

                X_torch = X_torch.to(DEVICE)
                print(f"âœ“ Prepared input tensor with shape: {X_torch.shape}")

                act_analyzer = ActivationAnalyzer()
                activations = act_analyzer.extract_activations(
                    model=pytorch_model,
                    X=X_torch,
                    device=DEVICE
                )

                if activations:
                    print(f"âœ“ Extracted activations from {len(activations)} layers")
                    try:
                        act_stats = act_analyzer.analyze_activations(activations)
                        print("âœ“ Activation analysis completed")
                        try:
                            act_analyzer.plot_activation_distributions(
                                activations,
                                save_path=self.figures_dir / "activation_distributions.png"
                            )
                            print("âœ“ Activation distribution plots saved")
                        except Exception as e:
                            print(f"âš ï¸  Failed to plot activation distributions: {e}")
                        all_results['activations'] = act_stats
                    except Exception as e:
                        print(f"âš ï¸  Failed to analyze activations: {e}")
                        all_results['activations'] = {'error': f'analysis_failed: {e}'}
                else:
                    print("âš ï¸  No activations extracted")
                    all_results['activations'] = {'error': 'no_activations_extracted'}

            except Exception as e:
                print(f"âš ï¸  Activation extraction failed: {e}")
                all_results['activations'] = {'error': str(e)}
            
        # ================================================================
        # 6. WEIGHT ANALYSIS
        # ================================================================
        print(f"\n{'='*70}")
        print("SECTION 6: WEIGHT DISTRIBUTION ANALYSIS")
        print(f"{'='*70}\n")

        pytorch_model = self._get_pytorch_model(model)

        if pytorch_model is None:
            print("âš ï¸  Weight Distribution Analysis SKIPPED")
            print("   Model is not a PyTorch nn.Module (e.g. HMM â€” no weight tensors)")
            all_results['weights'] = {'error': 'no_pytorch_model'}
        else:
            try:
                weight_diagnostics = WeightDiagnostics()
                weight_dists = weight_diagnostics.analyze_weight_distributions(pytorch_model)
                weight_diagnostics.plot_weight_distributions(
                    weight_dists,
                    save_path=self.figures_dir / "weight_distributions.png"
                )
                all_results['weights'] = weight_dists
            except Exception as e:
                print(f"âš ï¸  Weight analysis failed: {e}")
                all_results['weights'] = {'error': str(e)}
        
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
                
                print_importance_report(weight_imp_results, top_k=DOFS)
                
                imp_viz = ImportanceVisualizer()
                imp_viz.plot_importance_bars(
                    weight_imp_results,
                    top_k=DOFS,
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
                print(f"âš ï¸  Could not extract RNN weights: {e}")
        
        # ================================================================
        # 8. SALIENCY MAPS
        # ================================================================
        print(f"\n{'='*70}")
        print("SECTION 8: SALIENCY ANALYSIS")
        print(f"{'='*70}\n")

        pytorch_model = self._get_pytorch_model(model)

        if pytorch_model is None:
            print("âš ï¸  Saliency Analysis SKIPPED")
            print("   Model is not a PyTorch nn.Module (e.g. HMM â€” no gradient-based saliency)")
            all_results['saliency'] = {'error': 'no_pytorch_model'}
        else:
            try:
                saliency_analyzer = SaliencyAnalyzer()

                for class_idx in [0, 1]:
                    sample_idx = np.where(y == class_idx)[0][0]
                    X_sample = torch.tensor(X[sample_idx:sample_idx+1], dtype=torch.float32)

                    if len(X_sample.shape) == 2:
                        seq_len = X_sample.shape[1] // 18
                        if seq_len > 1:
                            X_sample = X_sample.view(1, seq_len, 18)
                        else:
                            X_sample = X_sample.unsqueeze(1)

                    X_sample = X_sample.to(DEVICE)

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

            except Exception as e:
                print(f"âš ï¸  Saliency analysis failed: {e}")
                all_results['saliency'] = {'error': str(e)}
                
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

        def _make_serialisable(v):
            """
            Recursively convert any value to a JSON-safe type.

            Handles the full range of types that end up in all_results:
            - nested dicts and lists (recurse)
            - numpy arrays of any shape (tolist() â€” works for scalars and arrays)
            - numpy scalar types (int32, float64, etc.)
            - Python float inf/nan (not valid JSON â€” map to None)
            - everything else passes through unchanged
            """
            if isinstance(v, dict):
                return {kk: _make_serialisable(vv) for kk, vv in v.items()}
            if isinstance(v, list):
                return [_make_serialisable(vv) for vv in v]
            if isinstance(v, np.ndarray):
                # tolist() converts arrays of any shape to nested Python lists/scalars
                return _make_serialisable(v.tolist())
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                fv = float(v)
                return None if (fv == float('inf') or fv == float('-inf') or fv != fv) else fv
            if isinstance(v, float):
                return None if (v == float('inf') or v == float('-inf') or v != v) else v
            return v

        json_results = _make_serialisable(results)

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

            print(f"ðŸ” OVERFITTING ASSESSMENT:")
            print(f"   Risk Level:         {risk}")
            print(f"   Generalization Gap: {gap:.4f}")
            ratio_str = "N/A (generative model)" if ratio == float('inf') else f"{ratio:.2f}"
            print(f"   Sample/Param Ratio: {ratio_str}")
            
            if risk == "HIGH":
                print(f"\n   ðŸš¨ ACTION REQUIRED:")
                for rec in results['overfitting']['recommendations'][:3]:
                    print(f"      â€¢ {rec}")
            elif risk == "MEDIUM":
                print(f"\n   âš ï¸  MONITOR CAREFULLY")
            else:
                print(f"\n   âœ“ Model appears healthy")
        
        # Feature importance
        if 'feature_importance' in results:
            print(f"\nðŸ“Š TOP 6 DISCRIMINATIVE FEATURES:")
            sorted_feats = sorted(
                results['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:6]
            for i, (feat, score) in enumerate(sorted_feats, 1):
                print(f"   {i}. {feat}: {score:.4f}")
        
        # Gradient health
        if 'gradients' in results:
            print(f"\nâš¡ GRADIENT HEALTH:")
            grad_data = results['gradients']
            if 'error' in grad_data:
                # Skipped section â€” error string stored under 'error' key
                print(f"   âš ï¸  Skipped ({grad_data['error']})")
            else:
                # grad_data is {layer_name: {'mean': ..., ...}, ...}
                layer_stats = [v for v in grad_data.values() if isinstance(v, dict) and 'mean' in v]
                if layer_stats:
                    grad_mean = layer_stats[0]['mean']
                    if grad_mean < 1e-7:
                        print(f"   ðŸš¨ VANISHING gradients detected!")
                    elif grad_mean > 10:
                        print(f"   ðŸš¨ EXPLODING gradients detected!")
                    else:
                        print(f"   âœ“ Gradients healthy (mean={grad_mean:.6f})")
                else:
                    print(f"   âš ï¸  No per-layer gradient data available")

        # Activation health
        if 'activations' in results:
            print(f"\nðŸ§  ACTIVATION HEALTH:")
            act_data = results['activations']
            if 'error' in act_data:
                print(f"   âš ï¸  Skipped ({act_data['error']})")
            else:
                sparsity_vals = [
                    v['sparsity'] for v in act_data.values()
                    if isinstance(v, dict) and 'sparsity' in v
                ]
                if sparsity_vals:
                    total_sparsity = np.mean(sparsity_vals)
                    if total_sparsity > 0.7:
                        print(f"   âš ï¸  High sparsity ({total_sparsity:.1%}) - many dead neurons")
                    else:
                        print(f"   âœ“ Good activation diversity (sparsity={total_sparsity:.1%})")
                else:
                    print(f"   âš ï¸  No per-layer sparsity data available")
        
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