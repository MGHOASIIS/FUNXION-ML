"""
Feature importance analysis utilities.

Provides multiple methods for computing and analyzing feature importance:
- Permutation importance
- Weight-based importance
- SHAP values
- Gradient-based importance
- Ablation studies
"""
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score
from dataclasses import dataclass
import pandas as pd


@dataclass
class ImportanceResults:
    """Results from feature importance analysis."""
    feature_names: List[str]
    importance_scores: np.ndarray
    method: str
    baseline_score: Optional[float] = None
    std_scores: Optional[np.ndarray] = None
    ranking: Optional[np.ndarray] = None


class PermutationImportance:
    """
    Permutation-based feature importance.
    
    Measures importance by randomly permuting each feature and observing
    the decrease in model performance.
    """
    
    def __init__(
        self,
        model: Any,
        metric: Callable = balanced_accuracy_score,
        n_repeats: int = 10,
        random_state: int = 42
    ):
        """
        Parameters
        ----------
        model : Any
            Trained model with predict method
        metric : Callable
            Metric function (y_true, y_pred) -> score
        n_repeats : int
            Number of permutation repeats
        random_state : int
            Random seed
        """
        self.model = model
        self.metric = metric
        self.n_repeats = n_repeats
        self.random_state = random_state
    
    def compute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> ImportanceResults:
        """
        Compute permutation importance.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, n_features) or (N, T, n_features)
        y : np.ndarray
            True labels
        feature_names : List[str], optional
            Feature names
        
        Returns
        -------
        ImportanceResults
            Importance analysis results
        """
        rng = np.random.RandomState(self.random_state)
        
        # Get baseline score
        y_pred = self._predict(X)
        baseline_score = self.metric(y, y_pred)
        
        # Determine number of features
        if X.ndim == 2:
            n_features = X.shape[1]
        elif X.ndim == 3:
            n_features = X.shape[2]
        else:
            raise ValueError(f"Unsupported X shape: {X.shape}")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Compute importance for each feature
        importances = np.zeros((n_features, self.n_repeats))
        
        for feat_idx in range(n_features):
            for repeat in range(self.n_repeats):
                # Permute feature
                X_permuted = self._permute_feature(X, feat_idx, rng)
                
                # Compute score
                y_pred_perm = self._predict(X_permuted)
                score_perm = self.metric(y, y_pred_perm)
                
                # Importance = drop in performance
                importances[feat_idx, repeat] = baseline_score - score_perm
        
        # Aggregate across repeats
        importance_scores = importances.mean(axis=1)
        std_scores = importances.std(axis=1)
        ranking = np.argsort(importance_scores)[::-1]
        
        return ImportanceResults(
            feature_names=feature_names,
            importance_scores=importance_scores,
            method="permutation",
            baseline_score=baseline_score,
            std_scores=std_scores,
            ranking=ranking
        )
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        elif hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)
        else:
            raise ValueError("Model must have predict or predict_proba method")
    
    def _permute_feature(
        self,
        X: np.ndarray,
        feature_idx: int,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """Permute a single feature."""
        X_permuted = X.copy()
        
        if X.ndim == 2:
            # (N, n_features)
            rng.shuffle(X_permuted[:, feature_idx])
        elif X.ndim == 3:
            # (N, T, n_features)
            # Permute across samples, keeping temporal structure
            indices = rng.permutation(X.shape[0])
            X_permuted[:, :, feature_idx] = X[indices, :, feature_idx]
        
        return X_permuted


class WeightBasedImportance:
    """
    Weight-based feature importance for neural networks.
    
    Computes importance from model weights (e.g., first layer weights).
    """
    
    @staticmethod
    def from_linear_weights(
        weights: np.ndarray,
        feature_names: Optional[List[str]] = None,
        aggregate: str = 'mean'
    ) -> ImportanceResults:
        """
        Compute importance from linear layer weights.
        
        Parameters
        ----------
        weights : np.ndarray
            Weight matrix (out_features, in_features)
        feature_names : List[str], optional
            Feature names
        aggregate : str
            Aggregation method: 'mean', 'sum', 'max'
        
        Returns
        -------
        ImportanceResults
            Importance results
        """
        n_features = weights.shape[1]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Compute importance per input feature
        if aggregate == 'mean':
            importance = np.abs(weights).mean(axis=0)
        elif aggregate == 'sum':
            importance = np.abs(weights).sum(axis=0)
        elif aggregate == 'max':
            importance = np.abs(weights).max(axis=0)
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")
        
        # Normalize
        importance = importance / (importance.sum() + 1e-12)
        
        ranking = np.argsort(importance)[::-1]
        
        return ImportanceResults(
            feature_names=feature_names,
            importance_scores=importance,
            method=f"weights_{aggregate}",
            ranking=ranking
        )
    
    @staticmethod
    def from_conv_weights(
        weights: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> ImportanceResults:
        """
        Compute importance from convolutional layer weights.
        
        Parameters
        ----------
        weights : np.ndarray
            Conv weights (out_channels, in_channels, kernel_size)
        feature_names : List[str], optional
            Channel names
        
        Returns
        -------
        ImportanceResults
            Importance results
        """
        n_channels = weights.shape[1]
        
        if feature_names is None:
            feature_names = [f"channel_{i}" for i in range(n_channels)]
        
        # Average absolute weights across output channels and kernel
        importance = np.abs(weights).mean(axis=(0, 2))
        
        # Normalize
        importance = importance / (importance.sum() + 1e-12)
        
        ranking = np.argsort(importance)[::-1]
        
        return ImportanceResults(
            feature_names=feature_names,
            importance_scores=importance,
            method="conv_weights",
            ranking=ranking
        )


class AblationImportance:
    """
    Ablation-based feature importance.
    
    Measures importance by removing (zeroing out) each feature and
    observing performance drop.
    """
    
    def __init__(
        self,
        model: Any,
        metric: Callable = balanced_accuracy_score
    ):
        """
        Parameters
        ----------
        model : Any
            Trained model
        metric : Callable
            Metric function
        """
        self.model = model
        self.metric = metric
    
    def compute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> ImportanceResults:
        """
        Compute ablation importance.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True labels
        feature_names : List[str], optional
            Feature names
        
        Returns
        -------
        ImportanceResults
            Importance results
        """
        # Get baseline score
        y_pred = self._predict(X)
        baseline_score = self.metric(y, y_pred)
        
        # Determine features
        if X.ndim == 2:
            n_features = X.shape[1]
        elif X.ndim == 3:
            n_features = X.shape[2]
        else:
            raise ValueError(f"Unsupported X shape: {X.shape}")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Compute importance for each feature
        importances = np.zeros(n_features)
        
        for feat_idx in range(n_features):
            # Zero out feature
            X_ablated = self._ablate_feature(X, feat_idx)
            
            # Compute score
            y_pred_abl = self._predict(X_ablated)
            score_abl = self.metric(y, y_pred_abl)
            
            # Importance = drop in performance
            importances[feat_idx] = baseline_score - score_abl
        
        ranking = np.argsort(importances)[::-1]
        
        return ImportanceResults(
            feature_names=feature_names,
            importance_scores=importances,
            method="ablation",
            baseline_score=baseline_score,
            ranking=ranking
        )
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        elif hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)
        else:
            raise ValueError("Model must have predict or predict_proba method")
    
    def _ablate_feature(self, X: np.ndarray, feature_idx: int) -> np.ndarray:
        """Zero out a feature."""
        X_ablated = X.copy()
        
        if X.ndim == 2:
            X_ablated[:, feature_idx] = 0
        elif X.ndim == 3:
            X_ablated[:, :, feature_idx] = 0
        
        return X_ablated


class ImportanceVisualizer:
    """Visualize feature importance results."""
    
    @staticmethod
    def plot_importance_bars(
        results: ImportanceResults,
        top_k: Optional[int] = None,
        title: str = "Feature Importance",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot feature importance as bar chart.
        
        Parameters
        ----------
        results : ImportanceResults
            Importance results
        top_k : int, optional
            Show only top k features
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        # Select top k
        if top_k is None:
            top_k = len(results.feature_names)
        
        top_indices = results.ranking[:top_k]
        top_features = [results.feature_names[i] for i in top_indices]
        top_scores = results.importance_scores[top_indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(top_features))
        
        bars = ax.barh(y_pos, top_scores, color='steelblue', alpha=0.8)
        
        # Add error bars if available
        if results.std_scores is not None:
            top_stds = results.std_scores[top_indices]
            ax.errorbar(
                top_scores, y_pos,
                xerr=top_stds,
                fmt='none',
                ecolor='black',
                capsize=3,
                alpha=0.6
            )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f"{title} ({results.method})", fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        # Add baseline score if available
        if results.baseline_score is not None:
            textstr = f"Baseline Score: {results.baseline_score:.3f}"
            ax.text(
                0.98, 0.02, textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_importance_heatmap(
        results_dict: Dict[str, ImportanceResults],
        title: str = "Feature Importance Comparison",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot heatmap comparing importance across multiple methods/models.
        
        Parameters
        ----------
        results_dict : Dict[str, ImportanceResults]
            {method_name: ImportanceResults}
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        # Create importance matrix
        feature_names = list(results_dict.values())[0].feature_names
        method_names = list(results_dict.keys())
        
        importance_matrix = np.zeros((len(method_names), len(feature_names)))
        
        for i, (method, results) in enumerate(results_dict.items()):
            importance_matrix[i] = results.importance_scores
        
        # Normalize across methods for better visualization
        importance_matrix = importance_matrix / (importance_matrix.max(axis=1, keepdims=True) + 1e-12)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            importance_matrix,
            xticklabels=feature_names,
            yticklabels=method_names,
            cmap='YlOrRd',
            annot=False,
            fmt='.2f',
            cbar_kws={'label': 'Normalized Importance'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_importance_ranking(
        results_dict: Dict[str, ImportanceResults],
        top_k: int = 10,
        title: str = "Feature Ranking Comparison",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot ranking comparison across methods.
        
        Parameters
        ----------
        results_dict : Dict[str, ImportanceResults]
            {method_name: ImportanceResults}
        top_k : int
            Number of top features to show
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        x_offset = 0
        bar_width = 0.8 / len(results_dict)
        
        for i, (method, results) in enumerate(results_dict.items()):
            # Get top k features
            top_indices = results.ranking[:top_k]
            top_features = [results.feature_names[idx] for idx in top_indices]
            ranks = np.arange(1, top_k + 1)
            
            x_pos = np.arange(len(top_features)) + x_offset
            
            ax.bar(
                x_pos, ranks,
                width=bar_width,
                label=method,
                alpha=0.8
            )
            
            x_offset += bar_width
        
        # Get feature names from first method for x-axis
        first_results = list(results_dict.values())[0]
        top_features = [first_results.feature_names[i] for i in first_results.ranking[:top_k]]
        
        ax.set_xticks(np.arange(top_k) + bar_width * (len(results_dict) - 1) / 2)
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.set_ylabel('Rank', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best')
        ax.invert_yaxis()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()


def print_importance_report(
    results: ImportanceResults,
    top_k: int = 10
):
    """
    Print importance analysis report.
    
    Parameters
    ----------
    results : ImportanceResults
        Importance results
    top_k : int
        Number of top features to show
    """
    print("\n" + "="*60)
    print(f"Feature Importance Analysis - {results.method.upper()}")
    print("="*60)
    
    if results.baseline_score is not None:
        print(f"Baseline Score: {results.baseline_score:.4f}")
    
    print(f"\nTop {top_k} Most Important Features:")
    print("-" * 60)
    
    for rank, idx in enumerate(results.ranking[:top_k], 1):
        feature = results.feature_names[idx]
        score = results.importance_scores[idx]
        
        if results.std_scores is not None:
            std = results.std_scores[idx]
            print(f"  {rank:2d}. {feature:20s}  {score:8.4f} ± {std:.4f}")
        else:
            print(f"  {rank:2d}. {feature:20s}  {score:8.4f}")
    
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Create dummy data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # Dummy model
    class DummyModel:
        def predict(self, X):
            return (X[:, 0] > 0).astype(int)
    
    model = DummyModel()
    
    # Compute permutation importance
    perm_imp = PermutationImportance(model, n_repeats=5)
    results = perm_imp.compute(X, y)
    
    # Print report
    print_importance_report(results)
    
    # Visualize
    viz = ImportanceVisualizer()
    viz.plot_importance_bars(results, top_k=10)