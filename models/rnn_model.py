"""
Recurrent Neural Network (GRU/LSTM) for shoulder pathology classification.

Implements RNN with:
- GRU or LSTM cells
- Bidirectional support
- Multiple pooling strategies (last, mean, max)
- Early stopping
- Hyperparameter search with LOO CV
- Feature importance via input-to-hidden weights
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
# RNN Architecture
# ============================================================================

class RNNClassifier(nn.Module):
    """
    Generic GRU/LSTM classifier for multivariate time-series.
    
    Architecture:
        Input: (batch, T, C)
        RNN: GRU or LSTM with optional bidirection
        Pooling: last, mean, or max
        Classifier: Dropout → Linear
    """
    
    def __init__(
        self,
        input_dim: int,
        rnn_type: str = "gru",
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout_rnn: float = 0.2,
        dropout_fc: float = 0.3,
        num_classes: int = 2,
        pooling: str = "last"
    ):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features per timestep
        rnn_type : str
            'gru' or 'lstm'
        hidden_size : int
            Hidden state size
        num_layers : int
            Number of RNN layers
        bidirectional : bool
            Whether to use bidirectional RNN
        dropout_rnn : float
            Dropout between RNN layers
        dropout_fc : float
            Dropout before final linear layer
        num_classes : int
            Number of output classes
        pooling : str
            Pooling strategy: 'last', 'mean', or 'max'
        """
        super().__init__()
        
        assert rnn_type in {"gru", "lstm"}, "rnn_type must be 'gru' or 'lstm'"
        assert pooling in {"last", "mean", "max"}, "pooling must be 'last', 'mean', or 'max'"
        
        self.pooling = pooling.lower()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.rnn_type = rnn_type
        
        # RNN layer
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rnn if num_layers > 1 else 0.0
        )
        
        # Classifier
        effective_dim = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_fc),
            nn.Linear(effective_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, T, C)
        lengths : torch.Tensor, optional
            Sequence lengths for packed sequences
        
        Returns
        -------
        torch.Tensor
            Logits (batch, num_classes)
        """
        # RNN forward
        if lengths is not None:
            # Handle variable-length sequences
            lengths, perm_idx = lengths.sort(descending=True)
            x = x[perm_idx]
            
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True
            )
            out_packed, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
            
            # Restore original order
            _, unperm_idx = perm_idx.sort()
            out = out[unperm_idx]
            lengths = lengths[unperm_idx]
        else:
            out, _ = self.rnn(x)
        
        # Apply pooling
        if self.pooling == "last":
            if lengths is None:
                features = out[:, -1]
            else:
                # Get last actual timestep for each sequence
                batch_idx = torch.arange(out.size(0), device=out.device)
                features = out[batch_idx, lengths - 1]
        
        elif self.pooling == "mean":
            if lengths is None:
                features = out.mean(dim=1)
            else:
                # Masked mean (only over valid timesteps)
                mask = (
                    torch.arange(out.size(1), device=out.device)[None, :]
                    < lengths[:, None]
                )
                masked_out = out * mask.unsqueeze(-1)
                features = masked_out.sum(1) / lengths.unsqueeze(-1).float()
        
        else:  # 'max'
            if lengths is not None:
                # Mask out padding before max pooling
                mask = (
                    torch.arange(out.size(1), device=out.device)[None, :]
                    < lengths[:, None]
                )
                out = out.clone()
                out[~mask] = float('-inf')
            
            features = out.max(dim=1).values
        
        # Classifier
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract feature representations.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, T, C)
        lengths : torch.Tensor, optional
            Sequence lengths
        
        Returns
        -------
        torch.Tensor
            Feature vectors (batch, feature_dim)
        """
        # Same as forward but stop before classifier
        if lengths is not None:
            lengths, perm_idx = lengths.sort(descending=True)
            x = x[perm_idx]
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
            out_packed, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
            _, unperm_idx = perm_idx.sort()
            out = out[unperm_idx]
            lengths = lengths[unperm_idx]
        else:
            out, _ = self.rnn(x)
        
        # Apply pooling
        if self.pooling == "last":
            if lengths is None:
                features = out[:, -1]
            else:
                batch_idx = torch.arange(out.size(0), device=out.device)
                features = out[batch_idx, lengths - 1]
        elif self.pooling == "mean":
            if lengths is None:
                features = out.mean(dim=1)
            else:
                mask = (torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None])
                features = (out * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1).float()
        else:  # max
            if lengths is not None:
                mask = (torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None])
                out[~mask] = float('-inf')
            features = out.max(dim=1).values
        
        return features


# ============================================================================
# RNN Model Wrapper
# ============================================================================

class RNNModel(BaseModel, PyTorchModelMixin):
    """RNN model wrapper with LOO CV and hyperparameter search."""
    
    def __init__(self, checkpoints_dir=None, patience=15, min_delta=1e-4, task=None,
                 paradigm=None, channel_names=None, multilabel=False, label_names=None):
        super().__init__(model_name="RNN",
                         checkpoints_dir=checkpoints_dir,
                         patience=patience,
                         min_delta=min_delta,
                         task=task,
                         paradigm=paradigm,
                         channel_names=channel_names,
                         multilabel=multilabel,
                         label_names=label_names)
        
    def train_and_evaluate(
        self,
        X: List[np.ndarray],
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        param_grid: Optional[Dict] = None,
    ) -> ModelResults:
        """
        Train RNN with nested LOO CV: for each outer held-out subject,
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
            Subject identifiers for proper CV
        param_grid : Dict, optional
            Hyperparameter grid to select from via inner CV

        Returns
        -------
        ModelResults
            Complete results including metrics and predictions
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()

        y_tensor = torch.tensor(y, dtype=torch.float32 if self.multilabel else torch.long)
        n_channels = X[0].shape[1]

        # Handle subject-level CV if we have subject IDs
        cv_splits, unique_subjects = build_loo_splits(len(X), subject_ids, "RNN")

        grid = list(ParameterGrid(param_grid))
        print(f"[RNN] {len(grid)} candidate configs — selected per outer "
              f"fold via 5-fold inner CV, not by a single global search")

        (score, chosen_configs, y_true, y_pred, y_proba, avg_ih_weight,
         per_fold_results, subject_order) = self._nested_loo(
            grid, X, y_tensor, cv_splits, subject_ids, unique_subjects, n_channels
        )

        # Outer folds may each have picked a different config via inner CV
        # — that's expected in nested CV, not a bug. best_params here is
        # only a representative summary (the most frequently chosen
        # config); see per_fold_results[i]['hyperparameters'] for the
        # config actually used to score each fold's held-out subject.
        best_params = most_common_config(chosen_configs)

        print_best("RNN", best_params, score)

        # Compute feature importance from input-to-hidden weights
        feature_imp = self._compute_channel_importance(avg_ih_weight)

        # Compute metrics
        if self.multilabel:
            metrics = compute_multilabel_metrics(y_true, y_pred, y_proba, self.label_names)
            checkpoint_metrics = {'macro_f1': score, **metrics}
        else:
            metrics = compute_metrics(y_true, y_pred, y_proba)
            checkpoint_metrics = {'balanced_accuracy': score, **metrics}

        # Checkpoint best_model with all relevant info
        if self.checkpoint_dir:
            from datetime import datetime
            best_path = self.checkpoint_dir / f"best_model_BA{score:.4f}.pt"

            torch.save({
                'model_name': 'RNN',
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
        input_dim: int,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        fixed_epochs: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Train one RNNClassifier instance.

        If X_val/y_val are given: train with early stopping monitored on
        val loss, return (model, best_epoch, best_val_loss).
        Otherwise (fixed_epochs given instead): train for exactly that
        many epochs with no early stopping — used for the final retrain on
        all of a fold's training subjects, once inner CV has already
        picked both the config and the epoch count.
        """
        g = torch.Generator().manual_seed(seed)
        model = RNNClassifier(
            input_dim=input_dim,
            rnn_type=cfg["rnn_type"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            bidirectional=cfg["bidirectional"],
            dropout_rnn=cfg["dropout_rnn"],
            dropout_fc=cfg["dropout_fc"],
            pooling=cfg["pooling"],
            num_classes=self.n_labels if self.multilabel else 2,
        ).to(DEVICE)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=cfg["batch_size"], shuffle=True, generator=g
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        criterion = nn.BCEWithLogitsLoss() if self.multilabel else nn.CrossEntropyLoss()

        if X_val is not None:
            early_stopper = EarlyStopping(patience=self.patience, min_delta=self.min_delta, mode="min")
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
             avg_ih_weight, per_fold_results, subject_order)
        """
        y_true, y_pred, y_proba = [], [], []
        all_ih_weights = []
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

            print(f"[RNN] fold {fold_idx+1}/{len(cv_splits)} — chosen config "
                  f"(mean inner-CV loss {best_mean_loss:.4f}, {best_epochs} epochs): {best_cfg}")

            # Final retrain on ALL of this fold's training subjects, for
            # the epoch count inner CV already decided.
            final_model, epochs_trained, final_train_loss = self._train_one_model(
                best_cfg, X_train_full, y_train_full, input_dim, fixed_epochs=best_epochs
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

            # Predict on test set
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

            # Extract input-to-hidden weights and reduce to a per-channel
            # importance vector *before* accumulating — different folds
            # can pick different hidden_size/bidirectional via inner CV,
            # so raw weight tensors aren't stackable across folds, but the
            # reduced (C,) vector always is.
            w_fwd = final_model.rnn.weight_ih_l0.detach().cpu()
            if best_cfg["bidirectional"]:
                w_rev = final_model.rnn.weight_ih_l0_reverse.detach().cpu()
                w_combined = torch.cat([w_fwd, w_rev], dim=0)
            else:
                w_combined = w_fwd

            fold_importance = w_combined.abs().sum(dim=0)  # (C,)
            all_ih_weights.append(fold_importance)

            del final_model
            torch.cuda.empty_cache()

        if self.multilabel:
            ba = f1_score(y_true, y_pred, average="macro", zero_division=0)
        else:
            ba = balanced_accuracy_score(y_true, y_pred)

        avg_ih_weight = torch.stack(all_ih_weights).mean(dim=0)

        return (
            ba,
            chosen_configs,
            np.array(y_true),
            np.array(y_pred),
            np.array(y_proba),
            avg_ih_weight,
            per_fold_results,
            np.array(subject_order)
        )
    
    def _compute_channel_importance(self, importance_vector: torch.Tensor) -> Dict[str, float]:
        """
        Compute channel importance from the per-fold importance vectors,
        already reduced (abs sum across gates/directions) and averaged
        across folds in _nested_loo() — necessary because different folds
        can pick different hidden_size/bidirectional via inner CV, so raw
        weight tensors aren't stackable across folds the way the reduced
        (C,) vectors are.

        Parameters
        ----------
        importance_vector : torch.Tensor
            Per-channel importance, averaged across folds, shape (C,)

        Returns
        -------
        Dict[str, float]
            Channel names mapped to importance scores
        """
        importance = importance_vector.numpy()

        # Normalize
        importance = importance / (importance.sum() + 1e-12)

        # Create dictionary
        ch_names = self.resolve_channel_names(len(importance))
        feature_imp = {
            ch_names[i]: float(importance[i])
            for i in np.argsort(importance)[::-1]
        }
        
        print("\n[RNN] Channel Importance:")
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
        This method is here for interface consistency.
        """
        if self.feature_importance is not None:
            return self.feature_importance
        
        raise RuntimeError(
            "Feature importance not available. "
            "Run train_and_evaluate() first."
        )
    
    
    def _create_temp_model(self) -> Optional[nn.Module]:
        """
        Create temporary model from best_params for analysis.
        Same pattern as count_parameters() but returns the model.
        
        Returns
        -------
        nn.Module or None
            Temporary model if best_params available
        """
        if not hasattr(self, 'best_params') or self.best_params is None:
            return None
        
        try:
            from models.rnn_model import RNNClassifier

            temp_model = RNNClassifier(
                input_dim=self.n_channels,
                rnn_type=self.best_params["rnn_type"],
                hidden_size=self.best_params["hidden_size"],
                num_layers=self.best_params["num_layers"],
                bidirectional=self.best_params["bidirectional"],
                dropout_rnn=self.best_params["dropout_rnn"],
                dropout_fc=self.best_params["dropout_fc"],
                pooling=self.best_params["pooling"],
                num_classes=self.n_labels if self.multilabel else 2,
            )
            
            return temp_model
            
        except Exception as e:
            print(f"⚠️  Could not create temp model: {e}")
            return None
    
    def _get_default_param_grid(self) -> Dict:
        """Get default hyperparameter grid for RNN."""
        from config.hyperparameter import RNN_PARAM_GRID
        return RNN_PARAM_GRID


# ============================================================================
# Convenience Functions
# ============================================================================

def create_rnn_model(
    input_dim: int = 18,
    rnn_type: str = "gru",
    hidden_size: int = 128,
    num_layers: int = 2,
    bidirectional: bool = True,
    dropout_rnn: float = 0.2,
    dropout_fc: float = 0.3,
    pooling: str = "last",
    n_classes: int = 2
) -> RNNClassifier:
    """
    Convenience function to create an RNN model.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    rnn_type : str
        'gru' or 'lstm'
    hidden_size : int
        Hidden state size
    num_layers : int
        Number of RNN layers
    bidirectional : bool
        Whether to use bidirectional RNN
    dropout_rnn : float
        Dropout between RNN layers
    dropout_fc : float
        Dropout before classifier
    pooling : str
        Pooling strategy
    n_classes : int
        Number of classes
    
    Returns
    -------
    RNNClassifier
        Initialized model
    """
    return RNNClassifier(
        input_dim=input_dim,
        rnn_type=rnn_type,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout_rnn=dropout_rnn,
        dropout_fc=dropout_fc,
        num_classes=n_classes,
        pooling=pooling
    )


# Example usage
if __name__ == "__main__":
    # Test RNN architecture
    model = create_rnn_model()
    print(model)
    
    # Test forward pass
    batch_size = 4
    time_steps = 200
    features = 18
    
    x = torch.randn(batch_size, time_steps, features)
    logits = model(x)
    feature_vec = model.get_features(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {feature_vec.shape}")