"""
Vanilla Transformer for shoulder pathology classification.

Implements a standard Transformer encoder with:
- Positional encoding (sinusoidal)
- Multi-head self-attention
- Feed-forward network
- CLS token pooling (like BERT)
- Early stopping (matching RNN/CNN)
- Hyperparameter search with LOO CV
- Feature importance via input projection weights
- Full per-fold diagnostic tracking for overfitting monitoring (Phase 2)
"""
from typing import Dict, Optional, List
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import balanced_accuracy_score, f1_score

from models.base_model import BaseModel, ModelResults, PyTorchModelMixin
from config.constants import DEVICE
from utils.metrics import compute_metrics, compute_multilabel_metrics
from utils.training import (
    EarlyStopping, build_loo_splits, resolve_fold_masks,
    build_fold_record, print_best,
    fold_truncate_and_scale, build_inner_validation_split,
)


# ============================================================================
# Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Adds position information to the input embeddings so the Transformer
    can distinguish temporal order of time-series steps.

    Shape: (batch, T, d_model) -> (batch, T, d_model)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute sinusoidal encoding matrix (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but moves with .to(device))
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, T, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ============================================================================
# Transformer Architecture
# ============================================================================

class TransformerClassifier(nn.Module):
    """
    Vanilla Transformer encoder for multivariate time-series classification.

    Architecture:
        Input:  (batch, T, C=18)
        Linear projection: C -> d_model
        Prepend CLS token
        Positional encoding
        N x TransformerEncoderLayer (multi-head attention + FFN)
        CLS token output -> Dropout -> Linear -> logits

    The CLS token is a learnable vector prepended to the sequence.
    After encoding, its output representation is used for classification
    (same approach as BERT). This avoids the need to choose a pooling strategy.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        dropout_fc: float = 0.3,
        n_classes: int = 2,
        max_len: int = 5000,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features per timestep (18 for XDash)
        d_model : int
            Internal embedding dimension. Must be divisible by nhead.
        nhead : int
            Number of attention heads
        num_layers : int
            Number of TransformerEncoderLayer blocks
        dim_feedforward : int
            Hidden size of the FFN sublayer inside each encoder block
        dropout : float
            Dropout inside attention and FFN layers
        dropout_fc : float
            Dropout before the final classification head
        n_classes : int
            Number of output classes
        max_len : int
            Maximum sequence length for positional encoding
        """
        super().__init__()

        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )

        # --- Input projection: C -> d_model ---
        self.input_projection = nn.Linear(input_dim, d_model)

        # --- CLS token (learnable) ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # --- Positional encoding ---
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # expects (batch, seq, feature)
            norm_first=True,    # Pre-LN: more stable training on small data
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_fc),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, T, C)

        Returns
        -------
        torch.Tensor
            Logits (batch, n_classes)
        """
        batch_size = x.size(0)

        # Project to d_model
        x = self.input_projection(x)                        # (B, T, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)               # (B, T+1, d_model)

        # Positional encoding
        x = self.pos_encoder(x)                             # (B, T+1, d_model)

        # Transformer encoder
        x = self.transformer_encoder(x)                     # (B, T+1, d_model)

        # Extract CLS token output for classification
        cls_output = x[:, 0, :]                             # (B, d_model)

        # Classify
        logits = self.classifier(cls_output)                # (B, n_classes)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS-token feature representation (before classifier head).

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, T, C)

        Returns
        -------
        torch.Tensor
            Feature vectors (batch, d_model)
        """
        batch_size = x.size(0)
        x = self.input_projection(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x[:, 0, :]


# ============================================================================
# Transformer Model Wrapper
# ============================================================================

class TransformerModel(BaseModel, PyTorchModelMixin):
    """
    Transformer model wrapper with LOO CV, early stopping, and hyperparameter search.

    Follows the exact same interface as RNNModel and CNNModel.
    """

    def __init__(
        self,
        checkpoints_dir=None,
        patience: int = 15,
        min_delta: float = 1e-4,
        task=None,
        paradigm=None,
        channel_names=None,
        multilabel=False,
        label_names=None,
    ):
        super().__init__(
            model_name="Transformer",
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
        param_grid: Optional[Dict] = None,
    ) -> ModelResults:
        """
        Train Transformer with hyperparameter search using LOO CV.

        Parameters
        ----------
        X : list of np.ndarray
            Raw per-subject sequences, each shape (T_i, C) — variable
            length. Truncation length and z-score normalization are fit
            per LOSO fold (training subjects only) inside _loo_score(),
            not once over the whole dataset — see
            utils.training.fold_truncate_and_scale().
        y : np.ndarray
            Labels
        subject_ids : np.ndarray, optional
            Subject identifiers for subject-level CV splits
        param_grid : Dict, optional
            Hyperparameter grid for search

        Returns
        -------
        ModelResults
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()

        y_tensor = torch.tensor(y, dtype=torch.float32 if self.multilabel else torch.long)
        n_channels = X[0].shape[1]

        # Subject-level or sample-level LOO CV
        cv_splits, unique_subjects = build_loo_splits(len(X), subject_ids, "Transformer")

        grid = list(ParameterGrid(param_grid))
        print(f"[Transformer] Evaluating {len(grid)} hyperparameter combinations...")

        results = []
        for params in grid:
            score = self._loo_score(
                params, X, y_tensor,
                cv_splits, subject_ids, unique_subjects
            )
            results.append(score)

        best_result = max(results, key=lambda t: t[0])
        (
            best_score, best_params, y_true, y_pred,
            y_proba, avg_proj_weights, per_fold_results, subject_order
        ) = best_result

        print_best("Transformer", best_params, best_score)

        # Feature importance from input projection weights
        feature_imp = self._compute_channel_importance(avg_proj_weights)

        if self.multilabel:
            metrics = compute_multilabel_metrics(y_true, y_pred, y_proba, self.label_names)
            checkpoint_metrics = {"macro_f1": best_score, **metrics}
        else:
            metrics = compute_metrics(y_true, y_pred, y_proba)
            checkpoint_metrics = {"balanced_accuracy": best_score, **metrics}

        # Save checkpoint
        if self.checkpoint_dir:
            from datetime import datetime
            best_path = self.checkpoint_dir / f"best_model_BA{best_score:.4f}.pt"
            torch.save(
                {
                    "model_name": "Transformer",
                    "hyperparameters": best_params,
                    "metrics": checkpoint_metrics,
                    "feature_importance": feature_imp,
                    "input_shape": [len(X), n_channels],
                    "predictions": {
                        "y_true": y_true.tolist(),
                        "y_pred": y_pred.tolist(),
                        "y_proba": y_proba.tolist(),
                    },
                    "timestamp": datetime.now().isoformat(),
                },
                best_path,
            )
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
            per_fold_results=per_fold_results,
        )

    def _loo_score(
        self,
        cfg: Dict,
        X: List[np.ndarray],
        y: torch.Tensor,
        cv_splits: List,
        subject_ids: Optional[np.ndarray],
        unique_subjects: Optional[np.ndarray],
    ):
        """
        Run one full LOO CV pass for a given hyperparameter configuration.

        Returns
        -------
        tuple
            (balanced_accuracy, config, y_true, y_pred, y_proba,
             avg_proj_weights, per_fold_results)
        """
        y_true, y_pred, y_proba = [], [], []
        all_proj_weights = []
        per_fold_results = []
        subject_order = []

        g = torch.Generator().manual_seed(42)
        input_dim = X[0].shape[1]  # C dimension

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):

            # Resolve subject-level vs sample-level splits
            train_sample_idx, test_sample_idx, test_subjects = resolve_fold_masks(
                subject_ids, unique_subjects, train_idx, test_idx, fold_idx
            )

            # Truncation length and z-score scaler are fit on this fold's
            # training subjects only — the held-out subject never
            # contributes to either statistic.
            train_signals = [X[i] for i in train_sample_idx]
            test_signals  = [X[i] for i in test_sample_idx]
            X_train_np, X_test_np = fold_truncate_and_scale(
                train_signals, test_signals, output_format="3d"
            )
            X_train_full = torch.tensor(X_train_np, dtype=torch.float32)
            X_test = torch.tensor(X_test_np, dtype=torch.float32)
            y_train_full = y[train_sample_idx]
            y_test_list = y[test_sample_idx].tolist()

            # Inner validation split carved from this fold's training
            # subjects only (never the outer held-out subject above) —
            # this is what actually drives early stopping below.
            inner_train_pos, inner_val_pos = build_inner_validation_split(
                train_sample_idx, subject_ids, val_fraction=0.15, seed=42 + fold_idx
            )
            X_train = X_train_full[inner_train_pos]
            y_train = y_train_full[inner_train_pos]
            X_val   = X_train_full[inner_val_pos]
            y_val   = y_train_full[inner_val_pos]

            # Instantiate model fresh for each fold
            model = TransformerClassifier(
                input_dim=input_dim,
                d_model=cfg["d_model"],
                nhead=cfg["nhead"],
                num_layers=cfg["num_layers"],
                dim_feedforward=cfg["dim_feedforward"],
                dropout=cfg["dropout"],
                dropout_fc=cfg["dropout_fc"],
                n_classes=self.n_labels if self.multilabel else 2,
            ).to(DEVICE)

            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg["batch_size"],
                shuffle=True,
                generator=g,
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg["lr"],
                weight_decay=cfg.get("weight_decay", 1e-4),
            )
            criterion = nn.BCEWithLogitsLoss() if self.multilabel else nn.CrossEntropyLoss()

            # Early stopping — driven by the inner validation slice, not
            # training loss and not the outer test subject.
            early_stopper = EarlyStopping(
                patience=self.patience,
                min_delta=self.min_delta,
                mode="min",
            )

            best_train_loss = float("inf")
            best_train_acc = 0.0
            X_val_dev = X_val.to(DEVICE)
            y_val_dev = y_val.to(DEVICE).float() if self.multilabel else y_val.to(DEVICE).long()

            for epoch in range(cfg["epochs"]):
                train_loss, train_acc = self.train_epoch(
                    model, train_loader, optimizer, criterion, DEVICE,
                    multilabel=self.multilabel,
                )

                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_train_acc = train_acc

                model.eval()
                with torch.no_grad():
                    inner_val_loss = criterion(model(X_val_dev), y_val_dev).item()
                model.train()

                print(
                    f"[Transformer] fold {fold_idx+1} | epoch {epoch+1} | "
                    f"loss: {train_loss:.4f} | acc: {train_acc:.4f} | "
                    f"inner_val_loss: {inner_val_loss:.4f}"
                )

                if early_stopper.step(inner_val_loss, model):
                    break

            print("=" * 30)
            print(f"[Transformer] Fold {fold_idx + 1} complete")
            print("=" * 30)

            model = early_stopper.load_best(model)

            # Checkpoint every 10 folds
            if self.checkpoint_dir:
                should_save = (fold_idx % 10 == 0) or (fold_idx == len(cv_splits) - 1)
                if should_save:
                    torch.save(
                        {
                            "fold": fold_idx,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": cfg,
                            "metrics": {"inner_val_loss": early_stopper.best_score},
                        },
                        self.checkpoint_dir / f"fold_{fold_idx:03d}.pt",
                    )

            # Evaluate on held-out subject
            model.eval()
            with torch.no_grad():
                logits = model(X_test.to(DEVICE))
                if self.multilabel:
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs >= 0.5).astype(int)
                    y_test_tensor = torch.tensor(y_test_list, dtype=torch.float32).to(DEVICE)
                else:
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    preds = (probs >= 0.5).astype(int)
                    y_test_tensor = torch.tensor(y_test_list, dtype=torch.long).to(DEVICE)

                # Held-out loss for generalization-gap diagnostics only —
                # never used to make any training or stopping decision.
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
                train_loss=float(best_train_loss),
                val_loss=float(val_loss),
                inner_val_loss=float(early_stopper.best_score),
                train_acc=float(best_train_acc),
                epochs_trained=epoch + 1,
                early_stopped=early_stopper.early_stop,
            ))

            # Collect input projection weights for feature importance
            proj_weights = model.input_projection.weight.detach().cpu().clone()
            all_proj_weights.append(proj_weights)

            del model
            torch.cuda.empty_cache()

        if self.multilabel:
            ba = f1_score(y_true, y_pred, average="macro", zero_division=0)
        else:
            ba = balanced_accuracy_score(y_true, y_pred)
        avg_proj_weights = torch.stack(all_proj_weights).mean(dim=0)

        return (
            ba,
            cfg,
            np.array(y_true),
            np.array(y_pred),
            np.array(y_proba),
            avg_proj_weights,
            per_fold_results,
            np.array(subject_order),
        )

    def _compute_channel_importance(
        self, proj_weights: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute channel importance from averaged input projection weights.

        The input projection is a Linear(input_dim=18, d_model) layer.
        Weight shape: (d_model, input_dim).
        We average absolute values across the d_model dimension to get
        one importance score per input channel.

        Parameters
        ----------
        proj_weights : torch.Tensor
            Averaged projection weights (d_model, input_dim)

        Returns
        -------
        Dict[str, float]
            Channel names mapped to importance scores, sorted descending
        """
        # Average absolute weights across output (d_model) dimension -> (input_dim,)
        importance = proj_weights.abs().mean(dim=0).numpy()

        # Normalize
        importance = importance / (importance.sum() + 1e-12)

        ch_names = self.resolve_channel_names(len(importance))
        feature_imp = {
            ch_names[i]: float(importance[i])
            for i in np.argsort(importance)[::-1]
        }

        print("\n[Transformer] Channel Importance:")
        for i, (feat, imp) in enumerate(list(feature_imp.items())):
            print(f"  {i+1}. {feat}: {imp:.4f}")

        return feature_imp

    def compute_feature_importance(
        self, X: np.ndarray, y: np.ndarray, **kwargs
    ) -> Dict[str, float]:
        """Feature importance already computed during training."""
        if self.feature_importance is not None:
            return self.feature_importance
        raise RuntimeError(
            "Feature importance not available. Run train_and_evaluate() first."
        )

    def _create_temp_model(self) -> Optional[nn.Module]:
        """
        Reconstruct a TransformerClassifier from best_params for post-hoc analysis.
        Returns None if best_params not available.
        """
        if not hasattr(self, "best_params") or self.best_params is None:
            return None

        try:
            temp_model = TransformerClassifier(
                input_dim=self.n_channels,
                d_model=self.best_params["d_model"],
                nhead=self.best_params["nhead"],
                num_layers=self.best_params["num_layers"],
                dim_feedforward=self.best_params["dim_feedforward"],
                dropout=self.best_params["dropout"],
                dropout_fc=self.best_params["dropout_fc"],
                n_classes=self.n_labels if self.multilabel else 2,
            )
            return temp_model

        except Exception as e:
            print(f"⚠️  Could not create temp Transformer model: {e}")
            return None

    def _get_default_param_grid(self) -> Dict:
        """Load default hyperparameter grid for Transformer."""
        from config.hyperparameter import TRANSFORMER_PARAM_GRID
        return TRANSFORMER_PARAM_GRID


# ============================================================================
# Convenience Function
# ============================================================================

def create_transformer_model(
    input_dim: int = 18,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    dropout_fc: float = 0.3,
    n_classes: int = 2,
) -> TransformerClassifier:
    """Convenience function to create a TransformerClassifier."""
    return TransformerClassifier(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        dropout_fc=dropout_fc,
        n_classes=n_classes,
    )


# Example usage
if __name__ == "__main__":
    model = create_transformer_model()
    print(model)

    batch_size = 4
    time_steps = 200
    features = 18

    x = torch.randn(batch_size, time_steps, features)
    logits = model(x)
    feature_vec = model.get_features(x)

    print(f"\nInput shape:    {x.shape}")
    print(f"Logits shape:   {logits.shape}")
    print(f"Features shape: {feature_vec.shape}")