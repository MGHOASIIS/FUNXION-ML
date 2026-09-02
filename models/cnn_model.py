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
from sklearn.metrics import balanced_accuracy_score, f1_score
from joblib import parallel_backend, Parallel, delayed

from models.base_model import BaseModel, ModelResults, PyTorchModelMixin
from config.constants import DEVICE
from utils.metrics import compute_metrics, compute_multilabel_metrics
from utils.training import (
    EarlyStopping, build_loo_splits, resolve_fold_masks,
    build_fold_record, print_best,
    fold_truncate_and_scale, build_inner_kfold_splits,
    most_common_config,
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

    def __init__(self, checkpoints_dir=None, patience=10, min_delta=1e-4, task=None,
                 paradigm=None, channel_names=None, multilabel=False, label_names=None):
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
        channel_names : list of str, optional
            Input channel names, e.g. dataset_config["channels"].
        multilabel : bool
            If True, train N independent per-label sigmoid outputs instead
            of a single binary softmax output.
        label_names : list of str, optional
            Required when multilabel=True.
        """
        super().__init__(
            model_name="CNN",
            checkpoints_dir=checkpoints_dir,
            patience=patience,
            min_delta=min_delta,
            task=task,
            paradigm=paradigm,
            channel_names=channel_names,
            multilabel=multilabel,
            label_names=label_names,
        )

    def train_and_evaluate(
        self,
        X: List[np.ndarray],
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        param_grid: Optional[Dict] = None
    ) -> ModelResults:
        """
        Train CNN with nested LOO CV: for each outer held-out subject,
        hyperparameters and the early-stopping epoch are chosen via a
        5-fold inner CV over that fold's training subjects only, then one
        model is retrained on all of that fold's training subjects before
        scoring the held-out subject.

        Parameters
        ----------
        X : list of np.ndarray
            Raw per-subject sequences, each shape (T_i, C) — variable
            length. Truncation length and z-score normalization are fit
            per LOSO fold (training subjects only) inside _nested_loo(),
            not once over the whole dataset — see
            utils.training.fold_truncate_and_scale().
        y : np.ndarray
            Labels
        subject_ids : np.ndarray, optional
            Subject identifiers for subject-level CV splits
        param_grid : Dict, optional
            Hyperparameter grid to select from via inner CV

        Returns
        -------
        ModelResults
            Complete results including metrics, per-fold diagnostics, and predictions
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()

        y_tensor = torch.tensor(y, dtype=torch.float32 if self.multilabel else torch.long)
        n_channels = X[0].shape[1]

        # Set up LOO CV splits at the subject or sample level
        cv_splits, unique_subjects = build_loo_splits(len(X), subject_ids, "CNN")

        grid = list(ParameterGrid(param_grid))
        print(f"[CNN] {len(grid)} candidate configs — selected per outer "
              f"fold via 5-fold inner CV, not by a single global search")

        (score, chosen_configs, y_true, y_pred, y_proba, avg_first_layer_weights,
         per_fold_results, subject_order) = self._nested_loo(
            grid, X, y_tensor, cv_splits, subject_ids, unique_subjects, n_channels
        )

        # Outer folds may each have picked a different config via inner CV
        # — that's expected in nested CV, not a bug. best_params here is
        # only a representative summary (the most frequently chosen
        # config); see per_fold_results[i]['hyperparameters'] for the
        # config actually used to score each fold's held-out subject.
        best_params = most_common_config(chosen_configs)

        print_best("CNN", best_params, score)

        # Compute feature importance from averaged first-layer weights
        feature_imp = self._compute_channel_importance(avg_first_layer_weights)

        # Compute aggregate metrics
        if self.multilabel:
            metrics = compute_multilabel_metrics(y_true, y_pred, y_proba, self.label_names)
            checkpoint_metrics = {'macro_f1': score, **metrics}
        else:
            metrics = compute_metrics(y_true, y_pred, y_proba)
            checkpoint_metrics = {'balanced_accuracy': score, **metrics}

        # Save best model checkpoint
        if self.checkpoint_dir:
            from datetime import datetime
            best_path = self.checkpoint_dir / f"best_model_BA{score:.4f}.pt"

            torch.save({
                'model_name': 'CNN',
                'hyperparameters': best_params,
                'metrics': checkpoint_metrics,
                'feature_importance': feature_imp,
                'input_shape': [len(X), n_channels],
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
            X_shape=(len(X), n_channels),
            subject_ids=subject_order if subject_ids is not None else subject_ids,
            per_fold_results=per_fold_results
        )

    def _train_one_model(
        self,
        cfg: Dict,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        n_channels: int,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        fixed_epochs: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Train one CNNClassifier instance.

        If X_val/y_val are given: train with early stopping monitored on
        val loss, return (model, best_epoch, best_val_loss).
        Otherwise (fixed_epochs given instead): train for exactly that
        many epochs with no early stopping — used for the final retrain on
        all of a fold's training subjects, once inner CV has already
        picked both the config and the epoch count.
        """
        g = torch.Generator().manual_seed(seed)
        model = CNNClassifier(
            in_channels=n_channels,
            n_classes=self.n_labels if self.multilabel else 2,
            conv_channels=cfg["conv_channels"],
            kernel_sizes=cfg["kernel_sizes"],
            dropout_fc=cfg["dropout_fc"],
        ).to(DEVICE)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=cfg["batch_size"], shuffle=True, generator=g
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
        criterion = nn.BCEWithLogitsLoss() if self.multilabel else nn.CrossEntropyLoss()

        if X_val is not None:
            early_stopper = EarlyStopping(
                patience=self.patience, min_delta=self.min_delta, mode="min"
            )
            X_val_dev = X_val.to(DEVICE)
            y_val_dev = y_val.to(DEVICE).float() if self.multilabel else y_val.to(DEVICE).long()

            for epoch in range(cfg["epochs"]):
                self.train_epoch(
                    model, train_loader, optimizer, criterion, DEVICE,
                    multilabel=self.multilabel,
                )
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_val_dev), y_val_dev).item()
                model.train()
                if early_stopper.step(val_loss, model):
                    break

            model = early_stopper.load_best(model)
            return model, epoch + 1, float(early_stopper.best_score)
        else:
            n_epochs = fixed_epochs if fixed_epochs is not None else cfg["epochs"]
            last_train_loss = None
            for _ in range(n_epochs):
                last_train_loss, _ = self.train_epoch(
                    model, train_loader, optimizer, criterion, DEVICE,
                    multilabel=self.multilabel,
                )
            return model, n_epochs, last_train_loss

    def _nested_loo(
        self,
        grid: List[Dict],
        X: List[np.ndarray],
        y: torch.Tensor,
        cv_splits: List,
        subject_ids: Optional[np.ndarray],
        unique_subjects: Optional[np.ndarray],
        n_channels: int,
    ):
        """
        Nested LOO CV: for each outer fold, run a 5-fold inner CV over
        every candidate config to pick that fold's hyperparameters and
        stopping epoch, retrain once on the full outer-training set, then
        score the outer held-out subject.

        Returns
        -------
        tuple
            (balanced_accuracy, chosen_configs, y_true, y_pred, y_proba,
             avg_first_layer_weights, per_fold_results, subject_order)
            chosen_configs[i] is the hyperparameter config picked for the
            i-th outer fold (in fold-iteration order).
        """
        y_true, y_pred, y_proba = [], [], []
        all_first_layer_weights = []
        per_fold_results = []
        subject_order = []
        chosen_configs = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):

            train_sample_idx, test_sample_idx, test_subjects = resolve_fold_masks(
                subject_ids, unique_subjects, train_idx, test_idx, fold_idx
            )

            # Truncation length and z-score scaler are fit on this fold's
            # training subjects only — the held-out subject never
            # contributes to either statistic.
            train_signals = [X[i] for i in train_sample_idx]
            test_signals  = [X[i] for i in test_sample_idx]
            X_train_np, X_test_np = fold_truncate_and_scale(
                train_signals, test_signals, output_format="channels_first"
            )
            X_train_full = torch.tensor(X_train_np, dtype=torch.float32)
            X_test = torch.tensor(X_test_np, dtype=torch.float32)
            y_train_full = y[train_sample_idx]
            y_test_list = y[test_sample_idx].tolist()

            # 5-fold inner CV over this fold's training subjects only —
            # used for BOTH hyperparameter selection and early stopping.
            # The outer held-out subject above is never part of it.
            inner_splits = build_inner_kfold_splits(
                train_sample_idx, subject_ids, n_inner_folds=5, seed=42 + fold_idx
            )

            config_scores = []
            for cfg in grid:
                inner_losses, inner_epochs = [], []
                for inner_train_pos, inner_val_pos in inner_splits:
                    _, best_epoch, best_val_loss = self._train_one_model(
                        cfg,
                        X_train_full[inner_train_pos], y_train_full[inner_train_pos],
                        n_channels,
                        X_val=X_train_full[inner_val_pos], y_val=y_train_full[inner_val_pos],
                        seed=42 + fold_idx,
                    )
                    inner_losses.append(best_val_loss)
                    inner_epochs.append(best_epoch)
                config_scores.append((
                    float(np.mean(inner_losses)),
                    int(round(np.mean(inner_epochs))),
                    cfg,
                ))

            # This fold's winner: lowest mean inner-CV validation loss —
            # never the outer held-out subject's outcome.
            best_mean_loss, best_epochs, best_cfg = min(config_scores, key=lambda t: t[0])
            chosen_configs.append(best_cfg)

            print(f"[CNN] fold {fold_idx+1}/{len(cv_splits)} — chosen config "
                  f"(mean inner-CV loss {best_mean_loss:.4f}, {best_epochs} epochs): {best_cfg}")

            # Final retrain on ALL of this fold's training subjects, for
            # the epoch count inner CV already decided — no further early
            # stopping needed, and no further validation split required.
            final_model, epochs_trained, final_train_loss = self._train_one_model(
                best_cfg, X_train_full, y_train_full, n_channels, fixed_epochs=best_epochs
            )

            if self.checkpoint_dir:
                should_save = (fold_idx % 10 == 0) or (fold_idx == len(cv_splits) - 1)
                if should_save:
                    torch.save({
                        'fold': fold_idx,
                        'model_state_dict': final_model.state_dict(),
                        'config': best_cfg,
                        'metrics': {'inner_cv_val_loss': best_mean_loss},
                    }, self.checkpoint_dir / f"fold_{fold_idx:03d}.pt")

            # Evaluate on held-out subject
            final_model.eval()
            criterion = nn.BCEWithLogitsLoss() if self.multilabel else nn.CrossEntropyLoss()
            with torch.no_grad():
                logits = final_model(X_test.to(DEVICE))
                if self.multilabel:
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs >= 0.5).astype(int)
                    y_test_tensor = torch.tensor(y_test_list, dtype=torch.float32).to(DEVICE)
                else:
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    preds = (probs >= 0.5).astype(int)
                    y_test_tensor = torch.tensor(y_test_list, dtype=torch.long).to(DEVICE)

                # Held-out loss for generalization-gap diagnostics only —
                # never used to make any training or selection decision.
                val_loss = criterion(logits, y_test_tensor).item()

            y_true.extend(y_test_list)
            y_pred.extend(preds.tolist())
            y_proba.extend(probs.tolist())
            subject_order.extend(list(test_subjects))

            if self.multilabel:
                fold_ba = f1_score(y_test_list, preds, average="macro", zero_division=0)
            else:
                fold_ba = balanced_accuracy_score(y_test_list, preds)

            per_fold_results.append(build_fold_record(
                fold_idx, test_subjects, subject_ids,
                y_test_list, preds, probs, fold_ba,
                train_loss=float(final_train_loss) if final_train_loss is not None else None,
                val_loss=float(val_loss),
                inner_val_loss=float(best_mean_loss),
                epochs_trained=epochs_trained,
                early_stopped=False,
                hyperparameters=best_cfg,
            ))

            # Reduce to a per-channel importance vector *before*
            # accumulating — different folds can pick different
            # conv_channels/kernel_sizes via inner CV, so the raw weight
            # tensors aren't stackable across folds, but the reduced
            # (in_channels,) vector always is.
            first_layer_weights = final_model.feature_extractor[0].weight.detach().cpu()
            fold_importance = first_layer_weights.abs().mean(dim=(0, 2))  # (in_channels,)
            all_first_layer_weights.append(fold_importance)

            del final_model
            torch.cuda.empty_cache()

        if self.multilabel:
            ba = f1_score(y_true, y_pred, average="macro", zero_division=0)
        else:
            ba = balanced_accuracy_score(y_true, y_pred)

        avg_first_layer_weights = torch.stack(all_first_layer_weights).mean(dim=0)

        return (
            ba,
            chosen_configs,
            np.array(y_true),
            np.array(y_pred),
            np.array(y_proba),
            avg_first_layer_weights,
            per_fold_results,
            np.array(subject_order),
        )

    def _compute_channel_importance(self, importance_vector: torch.Tensor) -> Dict[str, float]:
        """
        Compute channel importance from the per-fold importance vectors,
        already reduced and averaged across folds.

        Reduction (abs mean over output channels and kernel dimension) now
        happens per fold in _nested_loo(), before averaging — necessary
        because different folds can pick different conv_channels/
        kernel_sizes via inner CV, so raw weight tensors aren't stackable
        across folds the way the reduced (in_channels,) vectors are.

        Parameters
        ----------
        importance_vector : torch.Tensor
            Per-channel importance, averaged across folds, shape (in_channels,)

        Returns
        -------
        Dict[str, float]
            Channel names mapped to importance scores, sorted descending
        """
        importance = importance_vector.numpy()

        # Normalize (mirrors RNN)
        importance = importance / (importance.sum() + 1e-12)

        ch_names = self.resolve_channel_names(len(importance))
        feature_imp = {
            ch_names[i]: float(importance[i])
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
            temp_model = CNNClassifier(
                in_channels=self.n_channels,
                conv_channels=self.best_params["conv_channels"],
                kernel_sizes=self.best_params["kernel_sizes"],
                dropout_fc=self.best_params["dropout_fc"],
                n_classes=self.n_labels if self.multilabel else 2
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