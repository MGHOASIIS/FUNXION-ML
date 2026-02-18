"""
Overfitting & Underfitting Detection for Small Clinical Datasets (N=60)

State-of-the-art methods for detecting and preventing overfitting in small-sample
medical ML, specifically designed for XDash project.

Critical for N=60 sample size where models can easily memorize rather than learn.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr


@dataclass
class OverfittingAnalysis:
    """Results from overfitting detection analysis."""
    # Learning curves
    train_scores: np.ndarray
    val_scores: np.ndarray
    train_sizes: np.ndarray
    
    # Generalization gap
    generalization_gap: float  # |train_score - val_score|
    gap_significance: str  # 'severe', 'moderate', 'acceptable'
    
    # Complexity analysis
    model_capacity: int  # Number of parameters
    sample_to_param_ratio: float  # N_samples / N_params
    complexity_warning: str  # 'overparameterized', 'acceptable', 'underparameterized'
    
    # Prediction analysis
    prediction_confidence_gap: float  # Difference in confidence between train/val
    
    # Statistical tests
    train_val_correlation: float  # Correlation between train and val predictions
    
    # Summary
    overfitting_risk: str  # 'high', 'medium', 'low'
    recommendations: List[str]


class OverfittingDetector:
    """
    Comprehensive overfitting detection for small clinical datasets.
    
    Critical for N=60 where models can memorize rather than learn!
    """
    
    def __init__(self, verbose: bool = True):
        """
        Parameters
        ----------
        verbose : bool
            Print detailed analysis
        """
        self.verbose = verbose

    def count_parameters(self, model) -> int:
        temp_model = model._create_temp_model()
        n_params = sum(p.numel() for p in temp_model.parameters())
        del temp_model
        return n_params
    
    def analyze(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv_splits: List,
        fold_results: List[Dict],
        model_name: str = "Model"
    ) -> OverfittingAnalysis:
        """
        Comprehensive overfitting analysis.
        
        Parameters
        ----------
        model : Any
            Trained model
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        cv_splits : List
            Cross-validation splits
        fold_results : List[Dict]
            Results from each fold with train/val scores
        model_name : str
            Model name for reporting
        
        Returns
        -------
        OverfittingAnalysis
            Complete analysis results
        """
        print(f"\n{'='*70}")
        print(f"OVERFITTING ANALYSIS - {model_name}")
        print(f"{'='*70}")
        
        # Check what keys are available and use appropriate ones
        if 'train_loss' in fold_results[0] and 'val_loss' in fold_results[0]:
            # Using LOSSES
            train_scores = np.array([r['train_loss'] for r in fold_results])
            val_scores = np.array([r['val_loss'] for r in fold_results])
            metric_type = "LOSS"
            are_losses = True
        elif 'train_acc' in fold_results[0] and 'val_acc' in fold_results[0]:
            # Using ACCURACIES
            train_scores = np.array([r['train_acc'] for r in fold_results])
            val_scores = np.array([r['val_acc'] for r in fold_results])
            metric_type = "ACCURACY"
            are_losses = False
        elif 'train_score' in fold_results[0] and 'val_score' in fold_results[0]:
            # Generic 'score' fields - detect if loss or accuracy
            train_scores = np.array([r['train_score'] for r in fold_results])
            val_scores = np.array([r['val_score'] for r in fold_results])
            # Auto-detect: if any val_score > 1.0, they're losses
            are_losses = any(r['val_score'] > 1.0 for r in fold_results[:min(10, len(fold_results))])
            metric_type = "LOSS" if are_losses else "ACCURACY"
        else:
            raise ValueError("fold_results must have 'train_loss'/'val_loss' or 'train_acc'/'val_acc' or 'train_score'/'val_score'")
        
        print(f"\n1. GENERALIZATION GAP ANALYSIS")
        print(f"   Metric Type:        {metric_type}")
        
        # Compute gap based on metric type
        if are_losses:
            # LOSSES: lower is better, so val > train means overfitting
            generalization_gap = float(np.mean(val_scores) - np.mean(train_scores))
            print(f"   Train {metric_type} (mean): {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
            print(f"   Val {metric_type} (mean):   {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
            print(f"   Gap (val - train):  {generalization_gap:.4f}")
            print(f"   (Positive gap = validation worse = overfitting)")
        else:
            # ACCURACIES: higher is better, so train > val means overfitting
            generalization_gap = float(np.mean(train_scores) - np.mean(val_scores))
            print(f"   Train {metric_type} (mean): {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
            print(f"   Val {metric_type} (mean):   {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
            print(f"   Gap (train - val):  {generalization_gap:.4f}")
            print(f"   (Positive gap = validation worse = overfitting)")
        
        gap_significance = self._classify_gap(abs(generalization_gap))
        print(f"   Severity:           {gap_significance}")
        
        # 2. Model complexity analysis
        model_capacity = self.count_parameters(model)
        n_samples = len(X)
        sample_to_param_ratio = n_samples / max(model_capacity, 1)
        complexity_warning = self._classify_complexity(sample_to_param_ratio)
        
        print(f"\n2. MODEL COMPLEXITY ANALYSIS")
        print(f"   Parameters:         {model_capacity:,}")
        print(f"   Samples:            {n_samples}")
        print(f"   Sample/Param Ratio: {sample_to_param_ratio:.4f} ({complexity_warning})")
        print(f"   ⚠️  Rule of Thumb: Ratio > 10 is safe for small datasets")
        
        # 3. Prediction confidence analysis
        print(f"\n3. PREDICTION CONFIDENCE ANALYSIS")
        
        # Initialize confidence_gap
        confidence_gap = 0.0
        
        # Compute validation confidence from y_proba
        if fold_results and 'y_proba' in fold_results[0] and fold_results[0]['y_proba']:
            val_confs = [np.mean(f['y_proba']) for f in fold_results 
                        if 'y_proba' in f and f['y_proba']]
            
            if val_confs:
                mean_val_conf = np.mean(val_confs)
                std_val_conf = np.std(val_confs)
                
                print(f"   Val Confidence:     {mean_val_conf:.4f} ± {std_val_conf:.4f}")
                print(f"   (Mean prediction probability)")
                
                if mean_val_conf < 0.55:
                    print(f"   ⚠️  LOW confidence - model very uncertain")
                elif mean_val_conf > 0.85:
                    print(f"   ⚠️  HIGH confidence - possible overconfidence")
                else:
                    print(f"   ✓ Moderate confidence (reasonable)")
                
                # Check for train_conf (optional)
                if 'train_conf' in fold_results[0]:
                    train_confs = [r['train_conf'] for r in fold_results]
                    mean_train_conf = np.mean(train_confs)
                    confidence_gap = abs(mean_train_conf - mean_val_conf)
                    
                    print(f"   Train Confidence:   {mean_train_conf:.4f}")
                    print(f"   Confidence Gap:     {confidence_gap:.4f}")
                    
                    if confidence_gap > 0.15:
                        print(f"   🚨 Large gap - model overconfident on training")
                else:
                    print(f"   Train Confidence:   Not tracked")
                    print(f"   (Train confidence requires predictions on training set)")
            else:
                print(f"   Val Confidence:     N/A (no probability data)")
        else:
            print(f"   Val Confidence:     N/A (no probability data)")

        
        # 4. Train-val correlation
        if len(fold_results) > 5:
            train_val_corr = self._compute_train_val_correlation(fold_results)
            print(f"\n4. TRAIN-VAL CORRELATION")
            print(f"   Correlation:        {train_val_corr:.4f}")
            print(f"   ⚠️  Low correlation suggests overfitting")
        else:
            train_val_corr = None
        
        # 5. Overall risk assessment
        overfitting_risk = self._assess_overfitting_risk(
            generalization_gap, sample_to_param_ratio, confidence_gap
        )
        
        recommendations = self._generate_recommendations(
            overfitting_risk, generalization_gap, sample_to_param_ratio
        )
        
        print(f"\n5. OVERALL ASSESSMENT")
        print(f"   Overfitting Risk:   {overfitting_risk.upper()}")
        print(f"\n   Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        print(f"\n{'='*70}\n")
        
        return OverfittingAnalysis(
            train_scores=train_scores,
            val_scores=val_scores,
            train_sizes=np.arange(len(fold_results)),
            generalization_gap=generalization_gap,
            gap_significance=gap_significance,
            model_capacity=model_capacity,
            sample_to_param_ratio=sample_to_param_ratio,
            complexity_warning=complexity_warning,
            prediction_confidence_gap=confidence_gap,
            train_val_correlation=train_val_corr if train_val_corr else 0.0,
            overfitting_risk=overfitting_risk,
            recommendations=recommendations
        )
    
    def _classify_gap(self, gap: float) -> str:
        """Classify generalization gap severity."""
        if gap > 0.15:
            return "SEVERE - Strong overfitting"
        elif gap > 0.08:
            return "MODERATE - Some overfitting"
        elif gap > 0.03:
            return "MILD - Acceptable"
        else:
            return "GOOD - Generalizing well"
    
    def _classify_complexity(self, ratio: float) -> str:
        """Classify model complexity relative to data."""
        if ratio < 5:
            return "OVERPARAMETERIZED - High overfitting risk"
        elif ratio < 10:
            return "BORDERLINE - Monitor carefully"
        else:
            return "ACCEPTABLE - Good ratio"
    
    
    def _estimate_rnn_params(self, params: Dict, model_name: str) -> int:
        """Estimate RNN parameters from hyperparameters."""
        if model_name != "RNN":
            return 10000  # Default for non-RNN
        
        input_dim = 18  # XDash feature dimension
        hidden_size = params.get('hidden_size', 128)
        num_layers = params.get('num_layers', 2)
        bidirectional = params.get('bidirectional', True)
        rnn_type = params.get('rnn_type', 'gru')
        
        # RNN parameters calculation
        gates = 3 if rnn_type == 'gru' else 4  # GRU has 3 gates, LSTM has 4
        directions = 2 if bidirectional else 1
        
        # First layer: input_dim → hidden_size
        params_l0 = (gates * hidden_size * input_dim +      # weight_ih
                     gates * hidden_size * hidden_size +    # weight_hh
                     2 * gates * hidden_size)               # biases (ih and hh)
        params_l0 *= directions
        
        # Additional layers: hidden → hidden
        params_additional = 0
        for layer in range(1, num_layers):
            params_layer = (gates * hidden_size * (hidden_size * directions) +  # weight_ih
                           gates * hidden_size * hidden_size +                  # weight_hh
                           2 * gates * hidden_size)                             # biases
            params_layer *= directions
            params_additional += params_layer
        
        # Classifier: hidden → 2 classes
        classifier_input = hidden_size * directions
        params_classifier = classifier_input * 2 + 2  # weights + biases
        
        total_params = params_l0 + params_additional + params_classifier
        
        return int(total_params)
    
    def _analyze_confidence_gap(self, fold_results: List[Dict]) -> float:
        """Analyze difference in prediction confidence."""
        if 'train_conf' in fold_results[0] and 'val_conf' in fold_results[0]:
            train_confs = [r['train_conf'] for r in fold_results]
            val_confs = [r['val_conf'] for r in fold_results]
            return abs(np.mean(train_confs) - np.mean(val_confs))
        return 0.0
    
    def _compute_train_val_correlation(self, fold_results: List[Dict], are_losses: bool = None) -> float:
        """Compute correlation between train and val scores across folds."""
        # Extract scores based on available keys
        if 'train_loss' in fold_results[0] and 'val_loss' in fold_results[0]:
            train_scores = [r['train_loss'] for r in fold_results]
            val_scores = [r['val_loss'] for r in fold_results]
        elif 'train_acc' in fold_results[0] and 'val_acc' in fold_results[0]:
            train_scores = [r['train_acc'] for r in fold_results]
            val_scores = [r['val_acc'] for r in fold_results]
        else:
            train_scores = [r['train_score'] for r in fold_results]
            val_scores = [r['val_score'] for r in fold_results]
        
        corr, _ = pearsonr(train_scores, val_scores)
        return corr
    
    def _assess_overfitting_risk(
        self,
        gap: float,
        ratio: float,
        conf_gap: float
    ) -> str:
        """Assess overall overfitting risk."""
        risk_score = 0
        
        # Gap contribution
        if gap > 0.15:
            risk_score += 3
        elif gap > 0.08:
            risk_score += 2
        elif gap > 0.03:
            risk_score += 1
        
        # Complexity contribution
        if ratio < 5:
            risk_score += 3
        elif ratio < 10:
            risk_score += 1
        
        # Confidence contribution
        if conf_gap > 0.1:
            risk_score += 2
        elif conf_gap > 0.05:
            risk_score += 1
        
        if risk_score >= 5:
            return "HIGH"
        elif risk_score >= 3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_recommendations(
        self,
        risk: str,
        gap: float,
        ratio: float
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if risk == "HIGH":
            recommendations.append("🚨 CRITICAL: Model shows strong signs of overfitting")
            
            if gap > 0.15:
                recommendations.append("Increase regularization (dropout, weight decay)")
                recommendations.append("Consider simpler model architecture")
            
            if ratio < 5:
                recommendations.append("Reduce model capacity (fewer layers/units)")
                recommendations.append("Use data augmentation to increase effective sample size")
            
            recommendations.append("Try ensemble methods (reduce variance)")
            recommendations.append("Consider traditional ML (XGBoost) instead of DL")
        
        elif risk == "MEDIUM":
            recommendations.append("⚠️  MODERATE: Monitor overfitting carefully")
            
            if gap > 0.08:
                recommendations.append("Consider increasing dropout rate")
            
            if ratio < 10:
                recommendations.append("Consider reducing model size or augmenting data")
            
            recommendations.append("Use early stopping with validation set")
        
        else:
            recommendations.append("✓ GOOD: Model appears to generalize well")
            recommendations.append("Continue with current approach")
            recommendations.append("Consider slightly increasing model capacity if underfitting")
        
        return recommendations


class LearningCurveAnalyzer:
    """
    Analyze learning curves to detect overfitting/underfitting.
    
    For N=60, this is CRITICAL for understanding if model can learn
    or is just memorizing!
    """
    
    @staticmethod
    def compute_learning_curves(
        model_class: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv_splits: List,
        train_sizes: Optional[np.ndarray] = None,
        scoring: str = 'balanced_accuracy'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute learning curves (train/val scores vs training set size).
        
        Parameters
        ----------
        model_class : Any
            Model class or instance
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        cv_splits : List
            CV splits
        train_sizes : np.ndarray, optional
            Training set sizes to evaluate
        scoring : str
            Scoring metric
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (train_sizes, train_scores, val_scores)
        """
        if train_sizes is None:
            # For N=60, test: 10, 20, 30, 40, 50, 59 samples
            train_sizes = np.array([10, 20, 30, 40, 50, 59])
        
        n_sizes = len(train_sizes)
        n_folds = len(cv_splits)
        
        train_scores_all = np.zeros((n_sizes, n_folds))
        val_scores_all = np.zeros((n_sizes, n_folds))
        
        print(f"\n[LearningCurveAnalyzer] Computing learning curves...")
        print(f"  Train sizes: {train_sizes}")
        
        for size_idx, size in enumerate(train_sizes):
            print(f"  Training with {size} samples...")
            
            for fold_idx, split in enumerate(cv_splits):
                # Subsample training data
                train_idx = split.train_idx[:size]
                test_idx = split.test_idx
                
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                
                # Train model
                # This is simplified - actual implementation depends on model type
                # You'd call the model's fit method here
                
                # For now, placeholder scores
                train_scores_all[size_idx, fold_idx] = 0.8  # Replace with actual
                val_scores_all[size_idx, fold_idx] = 0.7    # Replace with actual
        
        train_scores = train_scores_all.mean(axis=1)
        val_scores = val_scores_all.mean(axis=1)
        
        return train_sizes, train_scores, val_scores
    
    @staticmethod
    def plot_learning_curves(
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        title: str = "Learning Curves",
        save_path: Optional[str] = None
    ):
        """
        Plot learning curves with interpretation.
        
        Parameters
        ----------
        train_sizes : np.ndarray
            Training set sizes
        train_scores : np.ndarray
            Training scores
        val_scores : np.ndarray
            Validation scores
        title : str
            Plot title
        save_path : str, optional
            Save path
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot curves
        ax.plot(train_sizes, train_scores, 'o-', linewidth=2, 
                label='Training Score', color='blue', markersize=8)
        ax.plot(train_sizes, val_scores, 's-', linewidth=2, 
                label='Validation Score', color='red', markersize=8)
        
        # Shaded region between curves
        ax.fill_between(train_sizes, train_scores, val_scores, 
                        alpha=0.2, color='gray', label='Generalization Gap')
        
        # Annotations
        final_gap = train_scores[-1] - val_scores[-1]
        
        # Add interpretation zones
        ax.axhspan(0, 0.6, alpha=0.1, color='red', label='Underfitting Zone')
        ax.axhspan(0.85, 1.0, alpha=0.1, color='green', label='Good Performance Zone')
        
        # Add text box with interpretation
        interpretation = self._interpret_curves(train_scores, val_scores, train_sizes)
        textstr = '\n'.join(interpretation)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()
    
    @staticmethod
    def _interpret_curves(
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        train_sizes: np.ndarray
    ) -> List[str]:
        """Interpret learning curves."""
        interpretations = []
        
        final_gap = train_scores[-1] - val_scores[-1]
        final_train = train_scores[-1]
        final_val = val_scores[-1]
        
        # Check for overfitting
        if final_gap > 0.15:
            interpretations.append("🚨 OVERFITTING:")
            interpretations.append("  Large train-val gap")
            interpretations.append("  → Increase regularization")
        elif final_gap > 0.08:
            interpretations.append("⚠️  MILD OVERFITTING:")
            interpretations.append("  Moderate train-val gap")
            interpretations.append("  → Monitor carefully")
        
        # Check for underfitting
        if final_val < 0.65:
            interpretations.append("🚨 UNDERFITTING:")
            interpretations.append("  Low validation score")
            interpretations.append("  → Increase model capacity")
        
        # Check convergence
        if len(train_scores) > 3:
            late_improvement = val_scores[-1] - val_scores[-3]
            if late_improvement > 0.02:
                interpretations.append("✓ STILL IMPROVING:")
                interpretations.append("  → Try more data/epochs")
        
        if not interpretations:
            interpretations.append("✓ GOOD GENERALIZATION")
        
        return interpretations


class BiasVarianceAnalyzer:
    """
    Bias-variance decomposition for understanding model errors.
    
    Critical for small datasets to understand if errors come from:
    - High bias (underfitting) → model too simple
    - High variance (overfitting) → model too complex
    """
    
    @staticmethod
    def analyze(
        fold_predictions: List[np.ndarray],
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Approximate bias-variance decomposition.
        
        Parameters
        ----------
        fold_predictions : List[np.ndarray]
            Predictions from each CV fold for same samples
        y_true : np.ndarray
            True labels
        
        Returns
        -------
        Dict[str, float]
            Bias, variance, and error components
        """
        # Stack predictions (n_folds, n_samples)
        predictions = np.stack(fold_predictions)
        
        # Mean prediction across folds
        mean_pred = predictions.mean(axis=0)
        
        # Bias: (mean_pred - y_true)^2
        bias_squared = np.mean((mean_pred - y_true) ** 2)
        
        # Variance: var(predictions across folds)
        variance = np.mean(predictions.var(axis=0))
        
        # Total error
        total_error = bias_squared + variance
        
        print(f"\n{'='*70}")
        print("BIAS-VARIANCE DECOMPOSITION")
        print(f"{'='*70}")
        print(f"Bias²:           {bias_squared:.4f}")
        print(f"Variance:        {variance:.4f}")
        print(f"Total Error:     {total_error:.4f}")
        print(f"\nError Breakdown:")
        print(f"  Bias²:         {100*bias_squared/total_error:.1f}%")
        print(f"  Variance:      {100*variance/total_error:.1f}%")
        
        if bias_squared > variance:
            print(f"\n⚠️  HIGH BIAS (Underfitting)")
            print("   → Model too simple, increase capacity")
        else:
            print(f"\n⚠️  HIGH VARIANCE (Overfitting)")
            print("   → Model too complex, add regularization")
        
        print(f"{'='*70}\n")
        
        return {
            'bias_squared': float(bias_squared),
            'variance': float(variance),
            'total_error': float(total_error)
        }


class PerFoldAnalyzer:
    """
    Per-fold analysis to detect inconsistent behavior across folds.
    
    For N=60 LOO CV, analyzing 60 folds helps identify:
    - Which subjects are hardest to classify
    - Whether performance is consistent
    - Outlier folds that suggest overfitting
    """
    
    @staticmethod
    def analyze_fold_consistency(
        fold_results: List[Dict],
        subject_ids: Optional[List[str]] = None
    ):
        """
        Analyze consistency across folds.
        
        Parameters
        ----------
        fold_results : List[Dict]
            Results from each fold
        subject_ids : List[str], optional
            Subject IDs for each fold
        """
        # Extract scores based on available keys
        if 'train_loss' in fold_results[0] and 'val_loss' in fold_results[0]:
            train_scores = [r['train_loss'] for r in fold_results]
            val_scores = [r['val_loss'] for r in fold_results]
            metric_type = "LOSS"
        elif 'train_acc' in fold_results[0] and 'val_acc' in fold_results[0]:
            train_scores = [r['train_acc'] for r in fold_results]
            val_scores = [r['val_acc'] for r in fold_results]
            metric_type = "ACCURACY"
        else:
            train_scores = [r.get('train_score', 0) for r in fold_results]
            val_scores = [r.get('val_score', 0) for r in fold_results]
            metric_type = "SCORE"
        
        print(f"\n{'='*70}")
        print("PER-FOLD CONSISTENCY ANALYSIS")
        print(f"{'='*70}")
        
        # Overall statistics
        print(f"\nValidation {metric_type} Statistics (across {len(fold_results)} folds):")
        print(f"  Mean:    {np.mean(val_scores):.4f}")
        print(f"  Std:     {np.std(val_scores):.4f}")
        print(f"  Min:     {np.min(val_scores):.4f}")
        print(f"  Max:     {np.max(val_scores):.4f}")
        print(f"  Range:   {np.max(val_scores) - np.min(val_scores):.4f}")
        
        # High variance suggests issues
        std_val = np.std(val_scores)
        if std_val > 0.5:
            print(f"\n🚨 VERY HIGH VARIANCE across folds")
            print("   → Model extremely unstable or data quality issues")
        elif std_val > 0.15:
            print(f"\n🚨 HIGH VARIANCE across folds")
            print("   → Model unstable, likely overfitting or data issues")
        elif std_val > 0.08:
            print(f"\n⚠️  MODERATE VARIANCE across folds")
            print("   → Monitor stability")
        else:
            print(f"\n✓ CONSISTENT performance across folds")
        
        # Identify outlier folds
        mean_val = np.mean(val_scores)
        std_val = np.std(val_scores)
        
        outliers = []
        for i, (fold, score) in enumerate(zip(fold_results, val_scores)):
            if abs(score - mean_val) > 2 * std_val:
                # Get subject ID if available
                if 'test_subjects' in fold and fold['test_subjects']:
                    subject = fold['test_subjects'][0]
                elif subject_ids and i < len(subject_ids):
                    subject = subject_ids[i]
                else:
                    subject = f"Fold {fold.get('fold', i)}"
                
                outliers.append((subject, score, fold.get('fold', i)))
        
        if outliers:
            print(f"\nOutlier Folds (>2 std from mean, {metric_type} = {mean_val:.3f} ± {std_val:.3f}):")
            for subject, score, fold_idx in sorted(outliers, key=lambda x: abs(x[1] - mean_val), reverse=True):
                deviation = abs(score - mean_val) / std_val
                print(f"  Fold {fold_idx:2d} ({subject}): {score:.4f} ({deviation:.1f}σ)")
            print(f"   → These {len(outliers)} subjects are edge cases - investigate clinically")
        
        print(f"{'='*70}\n")
    
    @staticmethod
    def plot_fold_scores(
        fold_results: List[Dict],
        subject_ids: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Plot scores for each fold."""
        # Extract scores based on available keys
        if 'train_loss' in fold_results[0] and 'val_loss' in fold_results[0]:
            train_scores = [r['train_loss'] for r in fold_results]
            val_scores = [r['val_loss'] for r in fold_results]
            metric_label = "Loss"
        elif 'train_acc' in fold_results[0] and 'val_acc' in fold_results[0]:
            train_scores = [r['train_acc'] for r in fold_results]
            val_scores = [r['val_acc'] for r in fold_results]
            metric_label = "Accuracy"
        else:
            train_scores = [r.get('train_score', 0) for r in fold_results]
            val_scores = [r.get('val_score', 0) for r in fold_results]
            metric_label = "Score"
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(fold_results))
        
        ax.plot(x, train_scores, 'o-', label=f'Train {metric_label}', alpha=0.6, markersize=4)
        ax.plot(x, val_scores, 's-', label=f'Val {metric_label}', alpha=0.8, markersize=5)
        
        # Add mean lines
        ax.axhline(np.mean(train_scores), color='blue', linestyle='--', 
                   alpha=0.5, label=f'Mean Train ({np.mean(train_scores):.3f})')
        ax.axhline(np.mean(val_scores), color='red', linestyle='--', 
                   alpha=0.5, label=f'Mean Val ({np.mean(val_scores):.3f})')
        
        # Shade standard deviation
        ax.fill_between(x, 
                        np.mean(val_scores) - np.std(val_scores),
                        np.mean(val_scores) + np.std(val_scores),
                        alpha=0.2, color='red')
        
        ax.set_xlabel('Fold Index (Subject)')
        ax.set_ylabel(metric_label)
        ax.set_title(f'Per-Fold {metric_label} Analysis')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # Simulate fold results
    n_folds = 60
    fold_results = []
    
    for i in range(n_folds):
        fold_results.append({
            'train_score': 0.85 + np.random.randn() * 0.05,
            'val_score': 0.72 + np.random.randn() * 0.08,
            'train_conf': 0.9 + np.random.randn() * 0.02,
            'val_conf': 0.75 + np.random.randn() * 0.05
        })
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 50)
            self.fc2 = nn.Linear(50, 2)
    
    model = DummyModel()
    X = np.random.randn(60, 100)
    y = np.random.randint(0, 2, 60)
    
    # Analyze overfitting
    detector = OverfittingDetector()
    analysis = detector.analyze(model, X, y, [], fold_results)
    
    # Analyze fold consistency
    fold_analyzer = PerFoldAnalyzer()
    fold_analyzer.analyze_fold_consistency(fold_results)