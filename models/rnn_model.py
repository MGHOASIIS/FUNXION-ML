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
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from sklearn.metrics import balanced_accuracy_score
from joblib import parallel_backend, Parallel, delayed

from models.base_model import BaseModel, ModelResults, PyTorchModelMixin
from config.constants import CHAN_NAME, DEVICE
from utils.metrics import compute_metrics


# ============================================================================
# Early Stopping
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        """
        Parameters
        ----------
        patience : int
            Number of epochs to wait before stopping
        min_delta : float
            Minimum change to qualify as improvement
        mode : str
            'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False
    
    def step(self, score: float, model: nn.Module) -> bool:
        """
        Check if should stop training.
        
        Parameters
        ----------
        score : float
            Current metric value
        model : nn.Module
            Model to save state from
        
        Returns
        -------
        bool
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
            return False
        
        # Check for improvement
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
        """Load best model state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model


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
    
    def __init__(self, checkpoints_dir=None, patience=15, min_delta=1e-4, task=None, paradigm=None):
        super().__init__(model_name="RNN", 
                         checkpoints_dir=checkpoints_dir,
                         patience=patience,
                         min_delta=min_delta,
                         task=task,
                         paradigm=paradigm)
        
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        param_grid: Optional[Dict] = None,
    ) -> ModelResults:
        """
        Train RNN with hyperparameter search using LOO CV.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, T, C) - batch-first format
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
            
            print(f"\n[RNN] Subject-level LOO CV: {len(unique_subjects)} subjects")
        else:
            loo = LeaveOneOut()
            cv_splits = list(loo.split(range(len(X))))
            unique_subjects = None
            print(f"\n[RNN] Sample-level LOO CV: {len(X)} samples")
        
        grid = list(ParameterGrid(param_grid))
        print(f"[RNN] Evaluating {len(grid)} hyperparameter combinations...")
        
        # Parallel hyperparameter search
        results = []

        for params in grid:
            score = self._loo_score(
                params, X_tensor, y_tensor,
                cv_splits, subject_ids, unique_subjects
            )
            results.append(score)

        # Select best configuration
        best_result = max(results, key=lambda t: t[0])
        best_score, best_params, y_true, y_pred, y_proba, avg_ih_weight, per_fold_results = best_result
        
        print(f"\n[RNN] Best params: {best_params}")
        print(f"[RNN] Best balanced accuracy: {best_score:.4f}")
        
        # Compute feature importance from input-to-hidden weights
        feature_imp = self._compute_channel_importance(avg_ih_weight)
        
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, y_proba)

        # Checkpoint best_model with all relevant info
        if self.checkpoint_dir:
            from datetime import datetime
            best_path = self.checkpoint_dir / f"best_model_BA{best_score:.4f}.pt"
            
            torch.save({
                'model_name': 'RNN',
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
        Compute LOO CV score with early stopping.
        
        Parameters
        ----------
        cfg : Dict
            Hyperparameter configuration
        X : torch.Tensor
            Feature tensor (N, T, C)
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
            (balanced_accuracy, config, y_true, y_pred, y_proba, avg_ih_weights)
        """
        y_true, y_pred, y_proba = [], [], []
        all_ih_weights = []
        per_fold_results = []

        g = torch.Generator().manual_seed(42)
        
        input_dim = X.shape[2]  # C dimension
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
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
            model = RNNClassifier(
                input_dim=input_dim,
                rnn_type=cfg["rnn_type"],
                hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"],
                bidirectional=cfg["bidirectional"],
                dropout_rnn=cfg["dropout_rnn"],
                dropout_fc=cfg["dropout_fc"],
                pooling=cfg["pooling"]
            ).to(DEVICE)
            
            # Create data loader
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg["batch_size"],
                shuffle=True,
                generator=g
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            criterion = nn.CrossEntropyLoss()
            
            # Early stopping
            early_stopper = EarlyStopping(patience=self.patience, min_delta=self.min_delta, mode="min")

            best_train_acc = 0.0
            best_train_loss = float('inf')
            
            # Training loop with early stopping
            for epoch in range(cfg["epochs"]):
                train_loss, train_acc = self.train_epoch(
                    model, train_loader, optimizer, criterion, DEVICE
                )
                
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_train_acc = train_acc

                print(f"[RNN] fold {fold_idx+1} | epoch {epoch+1} | "
                      f"loss: {train_loss:.4f} | acc: {train_acc:.4f}")
                
                # Check early stopping
                if early_stopper.step(train_loss, model):
                    break
            
            print("==============================")
            print("fold_idx: ", fold_idx+1)
            print("==============================")
            
            # Load best weights
            model = early_stopper.load_best(model)

            # Checkpoint model after every 10 folds
            if self.checkpoint_dir:
                should_save = (fold_idx % 10 == 0) or \
                            (fold_idx == len(cv_splits) - 1)
                if should_save:
                    torch.save({
                        'fold': fold_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': cfg,
                        'metrics': {'train_loss': early_stopper.best_score}
                    }, self.checkpoint_dir / f"fold_{fold_idx:03d}.pt")

            # Predict on test set
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

            
            # TODO:
            # How to calculate proba for train for the last epoch, check early_stopping best params - pred and proba

            from sklearn.metrics import balanced_accuracy_score
            fold_ba = balanced_accuracy_score(y_test_list, preds)
            # fold_acc = accuracy_score(y_test_list, preds)
            
            per_fold_results.append({
                'fold': fold_idx,
                'test_subjects': test_subjects.tolist() if subject_ids is not None else [fold_idx],
                
                # losses
                'train_loss': float(early_stopper.best_score),
                'val_loss': float(val_loss),
                
                # accuracies
                'train_acc': float(best_train_acc),
                'val_acc': float(fold_ba),

                # Predictions
                'y_true': y_test_list,
                'y_pred': preds.tolist(),
                'y_proba': probs.tolist(),
                
                # Training info
                'epochs_trained': epoch + 1,
                'early_stopped': early_stopper.early_stop
            })
            
            # Extract input-to-hidden weights
            w_fwd = model.rnn.weight_ih_l0.detach().cpu().clone()
            if cfg["bidirectional"]:
                w_rev = model.rnn.weight_ih_l0_reverse.detach().cpu().clone()
                w_combined = torch.cat([w_fwd, w_rev], dim=0)
            else:
                w_combined = w_fwd
            
            all_ih_weights.append(w_combined)
            
            # Clean up
            del model
            torch.cuda.empty_cache()
        
        # Compute balanced accuracy
        ba = balanced_accuracy_score(y_true, y_pred)
        
        # Average weights across folds
        avg_ih_weight = torch.stack(all_ih_weights).mean(dim=0)
        
        return (
            ba,
            cfg,
            np.array(y_true),
            np.array(y_pred),
            np.array(y_proba),
            avg_ih_weight, 
            per_fold_results
        )
    
    def _compute_channel_importance(self, weight_ih: torch.Tensor) -> Dict[str, float]:
        """
        Compute channel importance from input-to-hidden weights.
        
        Parameters
        ----------
        weight_ih : torch.Tensor
            Averaged input-to-hidden weights (gates*dirs*H, C)
        
        Returns
        -------
        Dict[str, float]
            Channel names mapped to importance scores
        """
        # Average absolute weights across all gates/directions
        importance = weight_ih.abs().sum(dim=0).numpy()
        
        # Normalize
        importance = importance / (importance.sum() + 1e-12)
        
        # Create dictionary
        feature_imp = {
            CHAN_NAME[i]: float(importance[i])
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
            from config.constants import DOFS
            
            temp_model = RNNClassifier(
                input_dim=DOFS,
                rnn_type=self.best_params["rnn_type"],
                hidden_size=self.best_params["hidden_size"],
                num_layers=self.best_params["num_layers"],
                bidirectional=self.best_params["bidirectional"],
                dropout_rnn=self.best_params["dropout_rnn"],
                dropout_fc=self.best_params["dropout_fc"],
                pooling=self.best_params["pooling"]
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