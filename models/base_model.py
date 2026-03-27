"""
Base model interface for all classification models.

This provides a unified API for training, evaluation, and feature importance.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
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
    
    def __init__(self, model_name: str, checkpoints_dir=None, patience=None, min_delta=None, task=None, paradigm=None):
        """
        Parameters
        ----------
        model_name : str
            Name of the model (e.g., 'HMM', 'CNN', 'RNN')
        """
        self.model_name = model_name
        self.best_params = None
        self.feature_importance = None
        self.checkpoint_dir = checkpoints_dir
        self.patience = patience
        self.min_delta = min_delta
        self.task = task
        self.paradigm = paradigm
    
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
    def train_epoch(model, loader, optimizer, criterion, device):
        """Train one epoch."""
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