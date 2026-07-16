"""
Model evaluation utilities.

Provides comprehensive evaluation metrics, visualization, and analysis tools.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    balanced_accuracy_score, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score,
    average_precision_score
)
from dataclasses import dataclass
import pandas as pd


@dataclass
class EvaluationResults:
    """Comprehensive evaluation results."""
    # Core metrics
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Probability metrics
    auc_roc: float
    auc_roc_ci: Tuple[float, float]
    auc_pr: float
    
    # Confusion matrix
    confusion_matrix: np.ndarray
    
    # Per-class metrics
    class_metrics: Dict[int, Dict[str, float]]
    
    # Predictions
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    
    # Optional
    subject_ids: Optional[np.ndarray] = None
    classification_report: Optional[str] = None


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, pos_label: int = 1):
        """
        Parameters
        ----------
        pos_label : int
            Label of positive class
        """
        self.pos_label = pos_label
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> EvaluationResults:
        """
        Comprehensive evaluation.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_proba : np.ndarray
            Predicted probabilities for positive class
        subject_ids : np.ndarray, optional
            Subject identifiers
        
        Returns
        -------
        EvaluationResults
            Complete evaluation results
        """
        # Ensure arrays are 1D
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        y_proba = np.asarray(y_proba).ravel()
        
        # Core metrics
        acc = accuracy_score(y_true, y_pred)
        ba = balanced_accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0)
        
        # Probability metrics
        auc_roc = roc_auc_score(y_true, y_proba)
        auc_roc_ci = self._compute_auc_ci(y_true, y_proba)
        auc_pr = average_precision_score(y_true, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        class_metrics = self._compute_class_metrics(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, zero_division=0)
        
        return EvaluationResults(
            accuracy=acc,
            balanced_accuracy=ba,
            precision=prec,
            recall=rec,
            f1_score=f1,
            auc_roc=auc_roc,
            auc_roc_ci=auc_roc_ci,
            auc_pr=auc_pr,
            confusion_matrix=cm,
            class_metrics=class_metrics,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            subject_ids=subject_ids,
            classification_report=report
        )
    
    def _compute_auc_ci(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bootstraps: int = 1000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute AUC confidence interval via bootstrapping.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_proba : np.ndarray
            Predicted probabilities
        n_bootstraps : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level (e.g., 0.95 for 95% CI)
        
        Returns
        -------
        Tuple[float, float]
            (lower_bound, upper_bound)
        """
        rng = np.random.RandomState(42)
        aucs = []
        
        n = len(y_true)
        for _ in range(n_bootstraps):
            # Bootstrap sample
            indices = rng.randint(0, n, n)
            
            # Skip if only one class present
            if len(np.unique(y_true[indices])) < 2:
                continue
            
            try:
                auc_val = roc_auc_score(y_true[indices], y_proba[indices])
                aucs.append(auc_val)
            except:
                continue
        
        # Compute percentiles
        alpha = 1 - confidence_level
        lower = np.percentile(aucs, alpha/2 * 100)
        upper = np.percentile(aucs, (1 - alpha/2) * 100)
        
        return (lower, upper)
    
    def _compute_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute per-class metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        
        Returns
        -------
        Dict[int, Dict[str, float]]
            Metrics for each class
        """
        classes = np.unique(y_true)
        class_metrics = {}
        
        for cls in classes:
            # Binary classification for this class
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            
            # Metrics
            prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            # Support (number of true instances)
            support = np.sum(y_true == cls)
            
            class_metrics[int(cls)] = {
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'support': int(support)
            }
        
        return class_metrics
    
    def print_report(self, results: EvaluationResults):
        """
        Print comprehensive evaluation report.
        
        Parameters
        ----------
        results : EvaluationResults
            Evaluation results to print
        """
        print("\n" + "="*70)
        print("Model Evaluation Report")
        print("="*70)
        
        # Overall metrics
        print("\nOverall Metrics:")
        print(f"  Accuracy:          {results.accuracy:.4f}")
        print(f"  Balanced Accuracy: {results.balanced_accuracy:.4f}")
        print(f"  Precision:         {results.precision:.4f}")
        print(f"  Recall:            {results.recall:.4f}")
        print(f"  F1-Score:          {results.f1_score:.4f}")
        
        # Probability metrics
        print("\nProbability Metrics:")
        print(f"  AUC-ROC:           {results.auc_roc:.4f} "
              f"[{results.auc_roc_ci[0]:.4f}, {results.auc_roc_ci[1]:.4f}]")
        print(f"  AUC-PR:            {results.auc_pr:.4f}")
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print(results.confusion_matrix)
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        for cls, metrics in results.class_metrics.items():
            print(f"  Class {cls}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1_score']:.4f}")
            print(f"    Support:   {metrics['support']}")
        
        # Classification report
        if results.classification_report:
            print("\nClassification Report:")
            print(results.classification_report)
        
        # Data info
        print("\nData Information:")
        print(f"  Total Samples:   {len(results.y_true)}")
        print(f"  Class 0 Samples: {np.sum(results.y_true == 0)}")
        print(f"  Class 1 Samples: {np.sum(results.y_true == 1)}")
        
        if results.subject_ids is not None:
            n_subjects = len(np.unique(results.subject_ids))
            print(f"  Unique Subjects: {n_subjects}")
        
        print("="*70 + "\n")


class Visualizer:
    """Visualization utilities for evaluation results."""
    
    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = False,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Parameters
        ----------
        cm : np.ndarray
            Confusion matrix
        class_names : List[str], optional
            Names of classes
        normalize : bool
            Whether to normalize
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curve.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_proba : np.ndarray
            Predicted probabilities
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc_score = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, 
                label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_probability_distribution(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Probability Distribution",
        save_path: Optional[str] = None
    ):
        """
        Plot probability distributions by class.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_proba : np.ndarray
            Predicted probabilities
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Split by true class
        probs_0 = y_proba[y_true == 0]
        probs_1 = y_proba[y_true == 1]
        
        # Histograms
        ax.hist(probs_0, bins=30, range=(0, 1), density=True,
                alpha=0.5, label='Class 0 (True Negative)', color='blue')
        ax.hist(probs_1, bins=30, range=(0, 1), density=True,
                alpha=0.5, label='Class 1 (True Positive)', color='red')
        
        # Add KDE curves
        from scipy.stats import gaussian_kde
        
        if len(probs_0) > 1:
            kde_0 = gaussian_kde(probs_0)
            x = np.linspace(0, 1, 200)
            ax.plot(x, kde_0(x), 'b-', linewidth=2, label='Class 0 (KDE)')
        
        if len(probs_1) > 1:
            kde_1 = gaussian_kde(probs_1)
            x = np.linspace(0, 1, 200)
            ax.plot(x, kde_1(x), 'r-', linewidth=2, label='Class 1 (KDE)')
        
        ax.axvline(0.5, color='k', linestyle='--', linewidth=1, label='Threshold (0.5)')
        
        ax.set_xlabel('Predicted Probability (Class 1)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()
    


# Example usage
if __name__ == "__main__":
    # Create sample predictions
    np.random.seed(42)
    n_samples = 200
    
    y_true = np.random.randint(0, 2, n_samples)
    y_proba = np.random.beta(2, 5, n_samples)
    y_proba[y_true == 1] = np.random.beta(5, 2, np.sum(y_true == 1))
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Evaluate
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(y_true, y_pred, y_proba)
    evaluator.print_report(results)
    
    # Visualize
    viz = Visualizer()
    viz.plot_confusion_matrix(results.confusion_matrix, class_names=['Control', 'Patient'])
    viz.plot_roc_curve(y_true, y_proba)
    viz.plot_probability_distribution(y_true, y_proba)