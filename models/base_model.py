"""
Base model interface for all classification models.

This provides a unified API for training, evaluation, and feature importance.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelResults:
    """Structured results from model training/evaluation."""
    metrics: Dict[str, Any]
    best_params: Dict[str, Any]
    feature_importance: Dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    X_shape: Tuple
    subject_ids: Optional[np.ndarray] = None
    per_fold_results: Optional[Dict[str, Any]] = None


class BaseModel(ABC):
    """Abstract base class for all classification models."""
    
    def __init__(self, model_name: str, checkpoints_dir=None, patience=None, min_delta=None,
                 task=None, paradigm=None, channel_names=None,
                 multilabel: bool = False, label_names: Optional[List[str]] = None):
        """
        Parameters
        ----------
        model_name : str
            Name of the model (e.g., 'HMM', 'CNN', 'RNN')
        channel_names : list of str, optional
            Names of the input channels/features, e.g. dataset_config["channels"].
            Used for feature-importance labelling. If not provided, generic
            "ch_{i}" labels are used (resolved lazily once the channel count
            is known from the data — see resolve_channel_names()).
        multilabel : bool
            If True, this model is trained on a multi-label paradigm (N
            independent yes/no labels per sample) rather than the default
            binary paradigm.
        label_names : list of str, optional
            Names of the labels, in column order matching y's shape
            (N, len(label_names)). Required when multilabel=True.
        """
        self.model_name = model_name
        self.best_params = None
        self.feature_importance = None
        self.checkpoint_dir = checkpoints_dir
        self.patience = patience
        self.min_delta = min_delta
        self.task = task
        self.paradigm = paradigm
        self.channel_names = channel_names
        self.n_channels = len(channel_names) if channel_names is not None else None
        self.multilabel = multilabel
        self.label_names = label_names
        self.n_labels = len(label_names) if label_names is not None else None

    def resolve_channel_names(self, n_channels: int) -> list:
        """
        Return self.channel_names if set (and matching n_channels), else
        generic fallback labels ["ch_0", "ch_1", ...].
        """
        if self.channel_names is not None and len(self.channel_names) == n_channels:
            return self.channel_names
        return [f"ch_{i}" for i in range(n_channels)]
    
    @abstractmethod
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        param_grid: Optional[Dict] = None,
    ) -> ModelResults:
        """
        Train model with hyperparameter search and evaluate using LOO CV.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        subject_ids : np.ndarray, optional
            Subject identifiers for proper cross-validation
        param_grid : Dict, optional
            Hyperparameter grid for search
        
        Returns
        -------
        ModelResults
            Structured results including metrics and predictions
        """
        import random
        import numpy as np
        import torch
        import os
        
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(SEED)

        pass
    
    @abstractmethod
    def compute_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute feature importance scores.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        **kwargs
            Model-specific parameters
        
        Returns
        -------
        Dict[str, float]
            Feature names mapped to importance scores
        """
        pass
    
    def fit(
        self,
        g1: Dict,
        g0: Dict,
        preprocessor: 'BasePreprocessor',
        paradigm: int,
        param_grid: Optional[Dict] = None
    ) -> ModelResults:
        """
        High-level fit method that handles preprocessing and training.
        
        Parameters
        ----------
        g1 : Dict
            Group 1 data
        g0 : Dict
            Group 0 data
        preprocessor : BasePreprocessor
            Preprocessor instance
        paradigm : int
            Classification paradigm
        param_grid : Dict, optional
            Hyperparameter grid
        
        Returns
        -------
        ModelResults
            Complete training results
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name} - Paradigm {paradigm}")
        print(f"{'='*60}")
        
        # Preprocess data
        X, y, subject_ids = preprocessor.prepare_data(g1, g0)
        
        # Train and evaluate
        results = self.train_and_evaluate(
            X=X,
            y=y,
            subject_ids=subject_ids,
            param_grid=param_grid,
        )
        
        # Store best parameters
        self.best_params = results.best_params
        self.feature_importance = results.feature_importance
        self.per_fold_results = results.metrics.get('per_fold_results', None)
        
        print(f"\n{self.model_name} Training Complete")
        print(f"Best Balanced Accuracy: {results.metrics['ba']:.3f}")
        print(f"{'='*60}\n")

        return results

    def fit_multilabel(
        self,
        groups: Dict[str, Dict],
        label_names: List[str],
        preprocessor: 'BasePreprocessor',
        paradigm: int,
        param_grid: Optional[Dict] = None
    ) -> ModelResults:
        """
        Multi-label counterpart of fit() — same flow, but groups is a dict
        of N named label-groups (see ParadigmSelector.select_labels()) and
        preprocessing produces an (N_samples, len(label_names)) y matrix
        instead of a binary 0/1 vector.

        Parameters
        ----------
        groups : Dict[str, Dict]
            One subject-data dict per label name.
        label_names : List[str]
            Ordered label names, matching y's column order.
        preprocessor : BasePreprocessor
            Preprocessor instance.
        paradigm : int
            Classification paradigm.
        param_grid : Dict, optional
            Hyperparameter grid.

        Returns
        -------
        ModelResults
            Complete training results.
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name} - Paradigm {paradigm} (multilabel)")
        print(f"{'='*60}")

        X, y, subject_ids = preprocessor.prepare_data_multilabel(groups, label_names)

        results = self.train_and_evaluate(
            X=X,
            y=y,
            subject_ids=subject_ids,
            param_grid=param_grid,
        )

        self.best_params = results.best_params
        self.feature_importance = results.feature_importance
        self.per_fold_results = results.metrics.get('per_fold_results', None)

        print(f"\n{self.model_name} Training Complete")
        if "subset_accuracy" in results.metrics:
            print(f"Subset Accuracy: {results.metrics['subset_accuracy']:.3f}")
        if "macro_f1" in results.metrics:
            print(f"Macro F1:        {results.metrics['macro_f1']:.3f}")
        print(f"{'='*60}\n")

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of model configuration and performance."""
        return {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "feature_importance": self.feature_importance
        }


class PyTorchModelMixin:
    """Mixin for PyTorch-based models with common utilities."""
    
    @staticmethod
    def train_epoch(model, loader, optimizer, criterion, device, multilabel: bool = False):
        """Train one epoch.

        multilabel=False (default): yb is a scalar class index per sample,
        accuracy is standard argmax accuracy.
        multilabel=True: yb is a multi-hot (batch, n_labels) float tensor,
        accuracy is subset accuracy (all labels correct for a sample).
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * yb.size(0)
            if multilabel:
                import torch
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == yb).all(dim=1).sum().item()
            else:
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
            total += yb.size(0)

        return total_loss / total, correct / total
    
    # @staticmethod
    # def evaluate(model, X_test, device):
    #     """Evaluate on test data."""
    #     model.eval()
    #     import torch
    #     import torch.nn.functional as F
        
    #     with torch.no_grad():
    #         logits = model(X_test.to(device, non_blocking=True))
    #         probs = F.softmax(logits, dim=1)
            
    #         if len(probs.shape) > 1:
    #             prob = probs[0, 1].item()
    #         else:
    #             prob = probs[1].item()
            
    #         pred = int(prob >= 0.5)
        
    #     return pred, prob