"""
Convolutional Neural Network for shoulder pathology classification.

Implements 1D CNN with:
- Configurable conv layers
- Batch normalization
- Global average pooling
- Early stopping (matching RNN)
- Hyperparameter search with LOO CV
- Feature importance via averaged first-layer weights across all folds
- Full per-fold diagnostic tracking for overfitting monitoring (Phase 2)
"""
from typing import Dict, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import balanced_accuracy_score
from joblib import parallel_backend, Parallel, delayed

from models.base_model import BaseModel, ModelResults, PyTorchModelMixin
from config.constants import CHAN_NAME, DEVICE
from utils.metrics import compute_metrics
from utils.training import (
    EarlyStopping, build_loo_splits, resolve_fold_masks,
    build_fold_record, print_best,
)


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
        Extract feature representations before the classifier head.

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
    """CNN model wrapper with LOO CV, early stopping, and hyperparameter search."""

    def __init__(self, checkpoints_dir=None, patience=10, min_delta=1e-4, task=None, paradigm=None):
        """
        Parameters
        ----------
        checkpoints_dir : Path or None
            Directory to save model checkpoints
        patience : int
            Early stopping patience (epochs)
        min_delta : float
            Minimum loss improvement to reset early stopping counter
        task : str or None
            Task name (e.g. 'jar_opening') — stored for downstream tracking
        paradigm : int or None
            Classification paradigm index — stored for downstream tracking
        """
        super().__init__(
            model_name="CNN",
            checkpoints_dir=checkpoints_dir,
            patience=patience,
            min_delta=min_delta,
            task=task,
            paradigm=paradigm
        )

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
            Feature matrix (N, C, T) — channels-first format
        y : np.ndarray
            Labels
        subject_ids : np.ndarray, optional
            Subject identifiers for subject-level CV splits
        param_grid : Dict, optional
            Hyperparameter grid for search

        Returns
        -------
        ModelResults
            Complete results including metrics, per-fold diagnostics, and predictions
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Set up LOO CV splits at the subject or sample level
        cv_splits, unique_subjects = build_loo_splits(len(X), subject_ids, "CNN")

        grid = list(ParameterGrid(param_grid))
        print(f"[CNN] Evaluating {len(grid)} hyperparameter combinations...")

        # Sequential search (mirrors RNN — avoids Parallel/loky issues with CUDA)
        results = []
        for params in grid:
            score = self._loo_score(
                params, X_tensor, y_tensor,
                cv_splits, subject_ids, unique_subjects
            )
            results.append(score)

        # Select best configuration by balanced accuracy
        best_result = max(results, key=lambda t: t[0])
        best_score, best_params, y_true, y_pred, y_proba, avg_first_layer_weights, per_fold_results = best_result

        print_best("CNN", best_params, best_score)

        # Compute feature importance from averaged first-layer weights
        feature_imp = self._compute_channel_importance(avg_first_layer_weights)

        # Compute aggregate metrics
        metrics = compute_metrics(y_true, y_pred, y_proba)

        # Save best model checkpoint
        if self.checkpoint_dir:
            from datetime import datetime
            best_path = self.checkpoint_dir / f"best_model_BA{best_score:.4f}.pt"

            torch.save({
                'model_name': 'CNN',
                'hyperparameters': best_params,
                'metrics': {'balanced_accuracy': best_score, **metrics},
                'feature_importance': feature_imp,
                'input_shape': list(X.shape),
                'predictions': {
                    'y_true': y_true.tolist(),
                    'y_pred': y_pred.tolist(),
                    'y_proba': y_proba.tolist()
                },
                'timestamp': datetime.now().isoformat()
            }, best_path)

            print(f"\n BEST MODEL SAVED: {best_path}")

        return ModelResults(
            metrics=metrics,
            best_params=best_params,
            feature_importance=feature_imp,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            X_shape=X.shape,
            subject_ids=subject_ids,
            per_fold_results=per_fold_results
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
        Run one full LOO CV pass for a given hyperparameter configuration.

        Parameters
        ----------
        cfg : Dict
            Hyperparameter configuration
        X : torch.Tensor
            Feature tensor (N, C, T)
        y : torch.Tensor
            Labels
        cv_splits : List
            Precomputed LOO split indices
        subject_ids : np.ndarray or None
            Subject identifiers (for subject-level splits)
        unique_subjects : np.ndarray or None
            Unique subject IDs

        Returns
        -------
        tuple
            (balanced_accuracy, config, y_true, y_pred, y_proba,
             avg_first_layer_weights, per_fold_results)
        """
        y_true, y_pred, y_proba = [], [], []
        all_first_layer_weights = []   # accumulate across folds, not just last
        per_fold_results = []          # full per-fold diagnostics

        g = torch.Generator().manual_seed(42)

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):

            # Resolve sample indices from subject-level or sample-level splits
            train_sample_idx, test_sample_idx, test_subjects = resolve_fold_masks(
                subject_ids, unique_subjects, train_idx, test_idx, fold_idx
            )

            # Split data
            X_train = X[train_sample_idx]
            y_train = y[train_sample_idx]
            X_test = X[test_sample_idx]
            y_test_list = y[test_sample_idx].tolist()

            # Instantiate model fresh for each fold
            model = CNNClassifier(
                in_channels=X.shape[1],
                n_classes=2,
                conv_channels=cfg["conv_channels"],
                kernel_sizes=cfg["kernel_sizes"],
                dropout_fc=cfg["dropout_fc"]
            ).to(DEVICE)

            # Data loader
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

            # Early stopping (mirrors RNN)
            early_stopper = EarlyStopping(
                patience=self.patience,
                min_delta=self.min_delta,
                mode="min"
            )

            best_train_loss = float('inf')
            best_train_acc = 0.0

            # Training loop with early stopping
            for epoch in range(cfg["epochs"]):
                train_loss, train_acc = self.train_epoch(
                    model, train_loader, optimizer, criterion, DEVICE
                )

                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_train_acc = train_acc

                print(f"[CNN] fold {fold_idx+1} | epoch {epoch+1} | "
                      f"loss: {train_loss:.4f} | acc: {train_acc:.4f}")

                if early_stopper.step(train_loss, model):
                    break

            print(f"{'='*30}")
            print(f"[CNN] Fold {fold_idx + 1} complete")
            print(f"{'='*30}")

            # Restore best weights before evaluation
            model = early_stopper.load_best(model)

            # Checkpoint every 10 folds (mirrors RNN)
            if self.checkpoint_dir:
                should_save = (fold_idx % 10 == 0) or (fold_idx == len(cv_splits) - 1)
                if should_save:
                    torch.save({
                        'fold': fold_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': cfg,
                        'metrics': {'train_loss': early_stopper.best_score}
                    }, self.checkpoint_dir / f"fold_{fold_idx:03d}.pt")

            # Evaluate on held-out subject
            model.eval()
            with torch.no_grad():
                logits = model(X_test.to(DEVICE))
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                # Compute held-out loss for generalization curve (overfitting monitoring)
                y_test_tensor = torch.tensor(y_test_list, dtype=torch.long).to(DEVICE)
                val_loss = criterion(logits, y_test_tensor).item()

            y_true.extend(y_test_list)
            y_pred.extend(preds.tolist())
            y_proba.extend(probs.tolist())

            fold_ba = balanced_accuracy_score(y_test_list, preds)

            per_fold_results.append(build_fold_record(
                fold_idx, test_subjects, subject_ids,
                y_test_list, preds, probs, fold_ba,
                train_loss=float(early_stopper.best_score),
                val_loss=float(val_loss),
                train_acc=float(best_train_acc),
                epochs_trained=epoch + 1,
                early_stopped=early_stopper.early_stop,
            ))

            # FIX: accumulate first-layer weights from every fold
            first_layer_weights = model.feature_extractor[0].weight.detach().cpu().clone()
            all_first_layer_weights.append(first_layer_weights)

            # Clean up
            del model
            torch.cuda.empty_cache()

        # Aggregate
        ba = balanced_accuracy_score(y_true, y_pred)

        # FIX: average weights across all folds (not just last fold)
        avg_first_layer_weights = torch.stack(all_first_layer_weights).mean(dim=0)

        return (
            ba,
            cfg,
            np.array(y_true),
            np.array(y_pred),
            np.array(y_proba),
            avg_first_layer_weights,
            per_fold_results
        )

    def _compute_channel_importance(self, first_layer_weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute channel importance from averaged first convolutional layer weights.

        Averaging across folds before computing importance gives a more stable
        estimate than using a single (potentially atypical) fold.

        Parameters
        ----------
        first_layer_weights : torch.Tensor
            Averaged weights from first conv layer (out_channels, in_channels, kernel_size)

        Returns
        -------
        Dict[str, float]
            Channel names mapped to importance scores, sorted descending
        """
        # Reduce: abs mean over output channels and kernel dimension → (in_channels,)
        importance = first_layer_weights.abs().mean(dim=(0, 2)).numpy()

        # Normalize (mirrors RNN)
        importance = importance / (importance.sum() + 1e-12)

        feature_imp = {
            CHAN_NAME[i]: float(importance[i])
            for i in np.argsort(importance)[::-1]
        }

        print("\n[CNN] Channel Importance:")
        for i, (feat, imp) in enumerate(list(feature_imp.items())):
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
        This method exists for interface consistency with BaseModel.
        """
        if self.feature_importance is not None:
            return self.feature_importance

        raise RuntimeError(
            "Feature importance not available. "
            "Run train_and_evaluate() first."
        )

    def _create_temp_model(self) -> Optional[nn.Module]:
        """
        Reconstruct a CNNClassifier from best_params for post-hoc analysis
        (e.g. parameter counting, architecture inspection, weight visualisation).

        Mirrors the same pattern used in RNNModel._create_temp_model(), but
        uses CNNClassifier and CNN-specific hyperparameter keys.

        Note: the returned model is uninitialised (random weights) — it
        reflects the architecture of the best configuration, not the trained
        weights. Use saved checkpoints to restore trained weights.

        Returns
        -------
        nn.Module or None
            CNNClassifier instance if best_params are available, else None
        """
        if not hasattr(self, 'best_params') or self.best_params is None:
            return None

        try:
            from config.constants import DOFS  # DOFS = 18 input channels

            temp_model = CNNClassifier(
                in_channels=DOFS,
                conv_channels=self.best_params["conv_channels"],
                kernel_sizes=self.best_params["kernel_sizes"],
                dropout_fc=self.best_params["dropout_fc"],
                n_classes=2
            )

            return temp_model

        except Exception as e:
            print(f"⚠️  Could not create temp CNN model: {e}")
            return None

    def _get_default_param_grid(self) -> Dict:
        """Load default hyperparameter grid for CNN."""
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
    Convenience function to create a CNNClassifier.

    Parameters
    ----------
    input_channels : int
        Number of input channels
    conv_channels : List[int]
        Channels for each conv layer
    kernel_sizes : List[int]
        Kernel sizes for each conv layer
    dropout_fc : float
        Dropout rate before classifier head
    n_classes : int
        Number of output classes

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
    model = create_cnn_model()
    print(model)

    batch_size = 4
    channels = 18
    time_steps = 200

    x = torch.randn(batch_size, channels, time_steps)
    logits = model(x)
    features = model.get_features(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")