"""
Convolutional Neural Network for shoulder pathology classification.

Implements 1D CNN with:
- Configurable conv layers
- Batch normalization
- Global average pooling
- Hyperparameter search with LOO CV
- Feature importance via first-layer weights
"""
from typing import Dict, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from sklearn.metrics import balanced_accuracy_score
from joblib import parallel_backend, Parallel, delayed

from models.base_model import BaseModel, ModelResults, PyTorchModelMixin
from config.constants import CHAN_NAME, DEVICE
from utils.metrics import compute_metrics


# ============================================================================
# CNN Architecture
# ============================================================================

class CNNClassifier(nn.Module):
    """
    1D CNN for multivariate time-series classification.
    
    Architecture:
        Input: (batch, C_in, T)
        Conv blocks: (Conv1d → BatchNorm → ReLU) × K layers
        Global pooling: AdaptiveAvgPool1d
        Classifier: Dropout → Linear
    """
    
    def __init__(
        self,
        in_channels: int,
        conv_channels: List[int],
        kernel_sizes: List[int],
        dropout_fc: float = 0.3,
        n_classes: int = 2
    ):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels (e.g., 18 for XDash)
        conv_channels : List[int]
            Output channels for each conv layer
        kernel_sizes : List[int]
            Kernel size for each conv layer
        dropout_fc : float
            Dropout rate before final linear layer
        n_classes : int
            Number of output classes
        """
        super().__init__()
        
        assert len(conv_channels) == len(kernel_sizes), \
            "conv_channels and kernel_sizes must have same length"
        
        # Build feature extractor
        layers = []
        prev_channels = in_channels
        
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(
                    prev_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ])
            prev_channels = out_channels
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Global average pooling + classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_fc) if dropout_fc > 0 else nn.Identity(),
            nn.Linear(prev_channels, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, C_in, T)
        
        Returns
        -------
        torch.Tensor
            Logits (batch, n_classes)
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representations.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, C_in, T)
        
        Returns
        -------
        torch.Tensor
            Feature vectors (batch, feature_dim)
        """
        features = self.feature_extractor(x)
        pooled = F.adaptive_avg_pool1d(features, 1)
        return pooled.flatten(1)


# ============================================================================
# CNN Model Wrapper
# ============================================================================

class CNNModel(BaseModel, PyTorchModelMixin):
    """CNN model wrapper with LOO CV and hyperparameter search."""
    
    def __init__(self):
        super().__init__(model_name="CNN")
    
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        param_grid: Optional[Dict] = None
    ) -> ModelResults:
        """
        Train CNN with hyperparameter search using LOO CV.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, C, T) - channels-first format
        y : np.ndarray
            Labels
        subject_ids : np.ndarray, optional
            Subject identifiers for proper CV
        param_grid : Dict, optional
            Hyperparameter grid for search
        
        Returns
        -------
        ModelResults
            Complete results including metrics and predictions
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Handle subject-level CV if we have subject IDs
        if subject_ids is not None:
            unique_subjects = np.unique(subject_ids)
            loo = LeaveOneOut()
            cv_splits = list(loo.split(unique_subjects))
            
            print(f"\n[CNN] Subject-level LOO CV: {len(unique_subjects)} subjects")
            
            # Map subject IDs to indices
            def get_indices_for_subject(subject_id):
                return np.where(subject_ids == subject_id)[0]
        else:
            loo = LeaveOneOut()
            cv_splits = list(loo.split(range(len(X))))
            unique_subjects = None
            print(f"\n[CNN] Sample-level LOO CV: {len(X)} samples")
        
        grid = list(ParameterGrid(param_grid))
        print(f"[CNN] Evaluating {len(grid)} hyperparameter combinations...")
        
        # Parallel hyperparameter search
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=-1, verbose=10)(
                delayed(self._loo_score)(
                    params, X_tensor, y_tensor, cv_splits,
                    subject_ids, unique_subjects
                )
                for params in grid
            )
        
        # Select best configuration
        best_result = max(results, key=lambda t: t[0])
        best_score, best_params, y_true, y_pred, y_proba, first_layer_weights = best_result
        
        print(f"\n[CNN] Best params: {best_params}")
        print(f"[CNN] Best balanced accuracy: {best_score:.4f}")
        
        # Compute feature importance from first layer weights
        feature_imp = self._compute_channel_importance(first_layer_weights)
        
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, y_proba)
        
        return ModelResults(
            metrics=metrics,
            best_params=best_params,
            feature_importance=feature_imp,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            X_shape=X.shape
        )
    
    def _loo_score(
        self,
        cfg: Dict,
        X: torch.Tensor,
        y: torch.Tensor,
        cv_splits: List,
        subject_ids: Optional[np.ndarray],
        unique_subjects: Optional[np.ndarray]
    ):
        """
        Compute LOO CV score for given hyperparameters.
        
        Parameters
        ----------
        cfg : Dict
            Hyperparameter configuration
        X : torch.Tensor
            Feature tensor (N, C, T)
        y : torch.Tensor
            Labels
        cv_splits : List
            CV split indices
        subject_ids : np.ndarray or None
            Subject identifiers
        unique_subjects : np.ndarray or None
            Unique subject IDs for subject-level CV
        
        Returns
        -------
        tuple
            (balanced_accuracy, config, y_true, y_pred, y_proba, first_layer_weights)
        """
        y_true, y_pred, y_proba = [], [], []
        g = torch.Generator().manual_seed(42)
        
        for train_idx, test_idx in cv_splits:
            # Handle subject-level splits
            if subject_ids is not None:
                train_subjects = unique_subjects[train_idx]
                test_subjects = unique_subjects[test_idx]
                
                train_mask = np.isin(subject_ids, train_subjects)
                test_mask = np.isin(subject_ids, test_subjects)
                
                train_sample_idx = np.where(train_mask)[0]
                test_sample_idx = np.where(test_mask)[0]
            else:
                train_sample_idx = train_idx
                test_sample_idx = test_idx
            
            # Split data
            X_train = X[train_sample_idx]
            y_train = y[train_sample_idx]
            X_test = X[test_sample_idx]
            y_test_list = y[test_sample_idx].tolist()
            
            # Create model
            model = CNNClassifier(
                in_channels=X.shape[1],  # C dimension
                n_classes=2,
                conv_channels=cfg["conv_channels"],
                kernel_sizes=cfg["kernel_sizes"],
                dropout_fc=cfg["dropout_fc"]
            ).to(DEVICE)
            
            # Create data loader
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg["batch_size"],
                shuffle=True,
                generator=g
            )
            
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"]
            )
            criterion = nn.CrossEntropyLoss()
            
            # Train with warmup (freeze feature extractor)
            for p in model.feature_extractor.parameters():
                p.requires_grad = False
            for p in model.classifier.parameters():
                p.requires_grad = True
            
            for _ in range(cfg["warmup_epochs"]):
                self.train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            
            # Fine-tune (unfreeze all)
            for p in model.parameters():
                p.requires_grad = True
            
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"]
            )
            
            for _ in range(cfg["finetune_epochs"]):
                self.train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            
            # Predict on test set
            model.eval()
            with torch.no_grad():
                logits = model(X_test.to(DEVICE))
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= 0.5).astype(int)
            
            y_true.extend(y_test_list)
            y_pred.extend(preds.tolist())
            y_proba.extend(probs.tolist())
            
            # Save first layer weights from last fold
            first_layer = model.feature_extractor[0]
        
        # Compute balanced accuracy
        ba = balanced_accuracy_score(y_true, y_pred)
        
        # Extract first layer weights
        first_layer_weights = first_layer.weight.detach().cpu()
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return (
            ba,
            cfg,
            np.array(y_true),
            np.array(y_pred),
            np.array(y_proba),
            first_layer_weights
        )
    
    def _compute_channel_importance(self, first_layer_weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute channel importance from first convolutional layer weights.
        
        Parameters
        ----------
        first_layer_weights : torch.Tensor
            Weights from first conv layer (out_channels, in_channels, kernel_size)
        
        Returns
        -------
        Dict[str, float]
            Channel names mapped to importance scores
        """
        # Average absolute weights across output channels and kernel dimension
        # Shape: (out_channels, in_channels, kernel_size) → (in_channels,)
        importance = first_layer_weights.abs().mean(dim=(0, 2)).numpy()
        
        # Create dictionary
        feature_imp = {
            CHAN_NAME[i]: float(importance[i])
            for i in np.argsort(importance)[::-1]
        }
        
        print("\n[CNN] Top 5 Important Channels:")
        for i, (feat, imp) in enumerate(list(feature_imp.items())[:5]):
            print(f"  {i+1}. {feat}: {imp:.4f}")
        
        return feature_imp
    
    def compute_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """
        Feature importance already computed during training.
        This method is here for interface consistency.
        """
        if self.feature_importance is not None:
            return self.feature_importance
        
        raise RuntimeError(
            "Feature importance not available. "
            "Run train_and_evaluate() first."
        )
    
    def _get_default_param_grid(self) -> Dict:
        """Get default hyperparameter grid for CNN."""
        from config.hyperparameter import CNN_PARAM_GRID
        return CNN_PARAM_GRID


# ============================================================================
# Convenience Functions
# ============================================================================

def create_cnn_model(
    input_channels: int = 18,
    conv_channels: List[int] = [64, 128, 128],
    kernel_sizes: List[int] = [7, 5, 3],
    dropout_fc: float = 0.3,
    n_classes: int = 2
) -> CNNClassifier:
    """
    Convenience function to create a CNN model.
    
    Parameters
    ----------
    input_channels : int
        Number of input channels
    conv_channels : List[int]
        Channels for each conv layer
    kernel_sizes : List[int]
        Kernel sizes for each conv layer
    dropout_fc : float
        Dropout rate
    n_classes : int
        Number of classes
    
    Returns
    -------
    CNNClassifier
        Initialized model
    """
    return CNNClassifier(
        in_channels=input_channels,
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        dropout_fc=dropout_fc,
        n_classes=n_classes
    )


# Example usage
if __name__ == "__main__":
    # Test CNN architecture
    model = create_cnn_model()
    print(model)
    
    # Test forward pass
    batch_size = 4
    channels = 18
    time_steps = 200
    
    x = torch.randn(batch_size, channels, time_steps)
    logits = model(x)
    features = model.get_features(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")