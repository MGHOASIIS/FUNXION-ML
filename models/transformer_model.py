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
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from sklearn.metrics import balanced_accuracy_score

from models.base_model import BaseModel, ModelResults, PyTorchModelMixin
from config.constants import CHAN_NAME, DEVICE
from utils.metrics import compute_metrics


# ============================================================================
# Early Stopping (identical pattern to RNN/CNN)
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False

    def step(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def load_best(self, model: nn.Module) -> nn.Module:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model


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
    ):
        super().__init__(
            model_name="Transformer",
            checkpoints_dir=checkpoints_dir,
            patience=patience,
            min_delta=min_delta,
            task=task,
            paradigm=paradigm,
        )

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        param_grid: Optional[Dict] = None,
    ) -> ModelResults:
        """
        Train Transformer with hyperparameter search using LOO CV.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, T, C) — batch-first, same as RNN input format
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

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Subject-level or sample-level LOO CV
        if subject_ids is not None:
            unique_subjects = np.unique(subject_ids)
            loo = LeaveOneOut()
            cv_splits = list(loo.split(unique_subjects))
            print(f"\n[Transformer] Subject-level LOO CV: {len(unique_subjects)} subjects")
        else:
            unique_subjects = None
            loo = LeaveOneOut()
            cv_splits = list(loo.split(range(len(X))))
            print(f"\n[Transformer] Sample-level LOO CV: {len(X)} samples")

        grid = list(ParameterGrid(param_grid))
        print(f"[Transformer] Evaluating {len(grid)} hyperparameter combinations...")

        results = []
        for params in grid:
            score = self._loo_score(
                params, X_tensor, y_tensor,
                cv_splits, subject_ids, unique_subjects
            )
            results.append(score)

        best_result = max(results, key=lambda t: t[0])
        (
            best_score, best_params, y_true, y_pred,
            y_proba, avg_proj_weights, per_fold_results
        ) = best_result

        print(f"\n[Transformer] Best params: {best_params}")
        print(f"[Transformer] Best balanced accuracy: {best_score:.4f}")

        # Feature importance from input projection weights
        feature_imp = self._compute_channel_importance(avg_proj_weights)

        metrics = compute_metrics(y_true, y_pred, y_proba)

        # Save checkpoint
        if self.checkpoint_dir:
            from datetime import datetime
            best_path = self.checkpoint_dir / f"best_model_BA{best_score:.4f}.pt"
            torch.save(
                {
                    "model_name": "Transformer",
                    "hyperparameters": best_params,
                    "metrics": {"balanced_accuracy": best_score, **metrics},
                    "feature_importance": feature_imp,
                    "input_shape": list(X.shape),
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
            X_shape=X.shape,
            subject_ids=subject_ids,
            per_fold_results=per_fold_results,
        )

    def _loo_score(
        self,
        cfg: Dict,
        X: torch.Tensor,
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

        g = torch.Generator().manual_seed(42)
        input_dim = X.shape[2]  # C dimension

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):

            # Resolve subject-level vs sample-level splits
            if subject_ids is not None:
                train_subjects = unique_subjects[train_idx]
                test_subjects = unique_subjects[test_idx]
                train_mask = np.isin(subject_ids, train_subjects)
                test_mask = np.isin(subject_ids, test_subjects)
                train_sample_idx = np.where(train_mask)[0]
                test_sample_idx = np.where(test_mask)[0]
            else:
                test_subjects = np.array([fold_idx])
                train_sample_idx = train_idx
                test_sample_idx = test_idx

            X_train = X[train_sample_idx]
            y_train = y[train_sample_idx]
            X_test = X[test_sample_idx]
            y_test_list = y[test_sample_idx].tolist()

            # Instantiate model fresh for each fold
            model = TransformerClassifier(
                input_dim=input_dim,
                d_model=cfg["d_model"],
                nhead=cfg["nhead"],
                num_layers=cfg["num_layers"],
                dim_feedforward=cfg["dim_feedforward"],
                dropout=cfg["dropout"],
                dropout_fc=cfg["dropout_fc"],
                n_classes=2,
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
            criterion = nn.CrossEntropyLoss()

            early_stopper = EarlyStopping(
                patience=self.patience,
                min_delta=self.min_delta,
                mode="min",
            )

            best_train_loss = float("inf")
            best_train_acc = 0.0

            for epoch in range(cfg["epochs"]):
                train_loss, train_acc = self.train_epoch(
                    model, train_loader, optimizer, criterion, DEVICE
                )

                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_train_acc = train_acc

                print(
                    f"[Transformer] fold {fold_idx+1} | epoch {epoch+1} | "
                    f"loss: {train_loss:.4f} | acc: {train_acc:.4f}"
                )

                if early_stopper.step(train_loss, model):
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
                            "metrics": {"train_loss": early_stopper.best_score},
                        },
                        self.checkpoint_dir / f"fold_{fold_idx:03d}.pt",
                    )

            # Evaluate on held-out subject
            model.eval()
            with torch.no_grad():
                logits = model(X_test.to(DEVICE))
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                y_test_tensor = torch.tensor(y_test_list, dtype=torch.long).to(DEVICE)
                val_loss = criterion(logits, y_test_tensor).item()

            y_true.extend(y_test_list)
            y_pred.extend(preds.tolist())
            y_proba.extend(probs.tolist())

            fold_ba = balanced_accuracy_score(y_test_list, preds)

            per_fold_results.append(
                {
                    "fold": fold_idx,
                    "test_subjects": (
                        test_subjects.tolist()
                        if subject_ids is not None
                        else [fold_idx]
                    ),
                    # Losses — generalization curve
                    "train_loss": float(early_stopper.best_score),
                    "val_loss": float(val_loss),
                    # Accuracies
                    "train_acc": float(best_train_acc),
                    "val_acc": float(fold_ba),
                    # Predictions
                    "y_true": y_test_list,
                    "y_pred": preds.tolist(),
                    "y_proba": probs.tolist(),
                    # Training diagnostics
                    "epochs_trained": epoch + 1,
                    "early_stopped": early_stopper.early_stop,
                }
            )

            # Collect input projection weights for feature importance
            proj_weights = model.input_projection.weight.detach().cpu().clone()
            all_proj_weights.append(proj_weights)

            del model
            torch.cuda.empty_cache()

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

        feature_imp = {
            CHAN_NAME[i]: float(importance[i])
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
            from config.constants import DOFS

            temp_model = TransformerClassifier(
                input_dim=DOFS,
                d_model=self.best_params["d_model"],
                nhead=self.best_params["nhead"],
                num_layers=self.best_params["num_layers"],
                dim_feedforward=self.best_params["dim_feedforward"],
                dropout=self.best_params["dropout"],
                dropout_fc=self.best_params["dropout_fc"],
                n_classes=2,
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