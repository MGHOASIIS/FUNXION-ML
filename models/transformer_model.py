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
    fold_truncate_and_scale, build_inner_kfold_splits,
    most_common_config,
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
        Train Transformer with nested LOO CV: for each outer held-out
        subject, hyperparameters and the early-stopping epoch are chosen
        via a 5-fold inner CV over that fold's training subjects only,
        then one model is retrained on all of that fold's training
        subjects before scoring the held-out subject.

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
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()

        y_tensor = torch.tensor(y, dtype=torch.float32 if self.multilabel else torch.long)
        n_channels = X[0].shape[1]

        # Subject-level or sample-level LOO CV
        cv_splits, unique_subjects = build_loo_splits(len(X), subject_ids, "Transformer")

        grid = list(ParameterGrid(param_grid))
        print(f"[Transformer] {len(grid)} candidate configs — selected per "
              f"outer fold via 5-fold inner CV, not by a single global search")

        (score, chosen_configs, y_true, y_pred, y_proba, avg_proj_weights,
         per_fold_results, subject_order) = self._nested_loo(
            grid, X, y_tensor, cv_splits, subject_ids, unique_subjects, n_channels
        )

        # Outer folds may each have picked a different config via inner CV
        # — that's expected in nested CV, not a bug. best_params here is
        # only a representative summary (the most frequently chosen
        # config); see per_fold_results[i]['hyperparameters'] for the
        # config actually used to score each fold's held-out subject.
        best_params = most_common_config(chosen_configs)

        print_best("Transformer", best_params, score)

        # Feature importance from input projection weights
        feature_imp = self._compute_channel_importance(avg_proj_weights)

        if self.multilabel:
            metrics = compute_multilabel_metrics(y_true, y_pred, y_proba, self.label_names)
            checkpoint_metrics = {"macro_f1": score, **metrics}
        else:
            metrics = compute_metrics(y_true, y_pred, y_proba)
            checkpoint_metrics = {"balanced_accuracy": score, **metrics}

        # Save checkpoint
        if self.checkpoint_dir:
            from datetime import datetime
            best_path = self.checkpoint_dir / f"best_model_BA{score:.4f}.pt"
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

    def _train_one_model(
        self,
        cfg: Dict,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        input_dim: int,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        fixed_epochs: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Train one TransformerClassifier instance.

        If X_val/y_val are given: train with early stopping monitored on
        val loss, return (model, best_epoch, best_val_loss).
        Otherwise (fixed_epochs given instead): train for exactly that
        many epochs with no early stopping — used for the final retrain on
        all of a fold's training subjects, once inner CV has already
        picked both the config and the epoch count.
        """
        g = torch.Generator().manual_seed(seed)
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
            train_dataset, batch_size=cfg["batch_size"], shuffle=True, generator=g,
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4),
        )
        criterion = nn.BCEWithLogitsLoss() if self.multilabel else nn.CrossEntropyLoss()

        if X_val is not None:
            early_stopper = EarlyStopping(
                patience=self.patience, min_delta=self.min_delta, mode="min",
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
        input_dim: int,
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
             avg_proj_weights, per_fold_results, subject_order)
        """
        y_true, y_pred, y_proba = [], [], []
        all_proj_weights = []
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
                train_signals, test_signals, output_format="3d"
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
                        input_dim,
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

            print(f"[Transformer] fold {fold_idx+1}/{len(cv_splits)} — chosen config "
                  f"(mean inner-CV loss {best_mean_loss:.4f}, {best_epochs} epochs): {best_cfg}")

            # Final retrain on ALL of this fold's training subjects, for
            # the epoch count inner CV already decided.
            final_model, epochs_trained, final_train_loss = self._train_one_model(
                best_cfg, X_train_full, y_train_full, input_dim, fixed_epochs=best_epochs
            )

            if self.checkpoint_dir:
                should_save = (fold_idx % 10 == 0) or (fold_idx == len(cv_splits) - 1)
                if should_save:
                    torch.save(
                        {
                            "fold": fold_idx,
                            "model_state_dict": final_model.state_dict(),
                            "config": best_cfg,
                            "metrics": {"inner_cv_val_loss": best_mean_loss},
                        },
                        self.checkpoint_dir / f"fold_{fold_idx:03d}.pt",
                    )

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
            # accumulating — different folds can pick different d_model
            # via inner CV, so raw projection weight tensors aren't
            # stackable across folds, but the reduced (input_dim,) vector
            # always is.
            proj_weights = final_model.input_projection.weight.detach().cpu()
            fold_importance = proj_weights.abs().mean(dim=0)  # (input_dim,)
            all_proj_weights.append(fold_importance)

            del final_model
            torch.cuda.empty_cache()

        if self.multilabel:
            ba = f1_score(y_true, y_pred, average="macro", zero_division=0)
        else:
            ba = balanced_accuracy_score(y_true, y_pred)
        avg_proj_weights = torch.stack(all_proj_weights).mean(dim=0)

        return (
            ba,
            chosen_configs,
            np.array(y_true),
            np.array(y_pred),
            np.array(y_proba),
            avg_proj_weights,
            per_fold_results,
            np.array(subject_order),
        )

    def _compute_channel_importance(
        self, importance_vector: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute channel importance from the per-fold importance vectors,
        already reduced (abs mean over the d_model dimension) and
        averaged across folds in _nested_loo() — necessary because
        different folds can pick different d_model via inner CV, so raw
        projection weight tensors aren't stackable across folds the way
        the reduced (input_dim,) vectors are.

        Parameters
        ----------
        importance_vector : torch.Tensor
            Per-channel importance, averaged across folds, shape (input_dim,)

        Returns
        -------
        Dict[str, float]
            Channel names mapped to importance scores, sorted descending
        """
        importance = importance_vector.numpy()

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