"""
Shared training utilities for all model classes.

Extracted to eliminate duplicated code across cnn_model, rnn_model,
transformer_model, hmm_model, and hsmm_model.
"""
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneOut


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
        """Return True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def load_best(self, model: nn.Module) -> nn.Module:
        """Restore model to the best observed state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model


def build_loo_splits(
    X_len: int,
    subject_ids: Optional[np.ndarray],
    model_name: str,
) -> Tuple[List, Optional[np.ndarray]]:
    """
    Build subject-level or sample-level LOO CV splits.

    Parameters
    ----------
    X_len : int
        Number of samples (len(X))
    subject_ids : np.ndarray or None
        Subject identifiers — if provided, splits are at subject level
    model_name : str
        Used in the printed log line (e.g. "CNN", "HMM")

    Returns
    -------
    cv_splits : list of (train_idx, test_idx) tuples
    unique_subjects : np.ndarray or None
    """
    loo = LeaveOneOut()
    if subject_ids is not None:
        unique_subjects = np.unique(subject_ids)
        cv_splits = list(loo.split(unique_subjects))
        print(f"\n[{model_name}] Subject-level LOO CV: {len(unique_subjects)} subjects")
    else:
        unique_subjects = None
        cv_splits = list(loo.split(range(X_len)))
        print(f"\n[{model_name}] Sample-level LOO CV: {X_len} samples")
    return cv_splits, unique_subjects


def resolve_fold_masks(
    subject_ids: Optional[np.ndarray],
    unique_subjects: Optional[np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert LOO split indices to sample-level masks.

    Parameters
    ----------
    subject_ids : np.ndarray or None
    unique_subjects : np.ndarray or None
    train_idx : np.ndarray
        Indices into unique_subjects (subject-level) or X (sample-level)
    test_idx : np.ndarray
    fold_idx : int
        Current fold number — used as a fallback subject label when
        subject_ids is None

    Returns
    -------
    train_sample_idx : np.ndarray
    test_sample_idx  : np.ndarray
    test_subjects    : np.ndarray
    """
    if subject_ids is not None:
        train_subjects = unique_subjects[train_idx]
        test_subjects  = unique_subjects[test_idx]
        train_mask     = np.isin(subject_ids, train_subjects)
        test_mask      = np.isin(subject_ids, test_subjects)
        train_sample_idx = np.where(train_mask)[0]
        test_sample_idx  = np.where(test_mask)[0]
    else:
        test_subjects    = np.array([fold_idx])
        train_sample_idx = train_idx
        test_sample_idx  = test_idx
    return train_sample_idx, test_sample_idx, test_subjects


def build_fold_record(
    fold_idx: int,
    test_subjects: np.ndarray,
    subject_ids: Optional[np.ndarray],
    y_test: list,
    preds,
    probs,
    fold_ba: float,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    train_acc: Optional[float] = None,
    epochs_trained: Optional[int] = None,
    early_stopped: bool = False,
) -> Dict:
    """
    Build the per-fold diagnostics dict stored in per_fold_results.

    Parameters
    ----------
    fold_idx : int
    test_subjects : np.ndarray
    subject_ids : np.ndarray or None
        If None, test_subjects entry is replaced with [fold_idx]
    y_test : list
        Ground-truth labels for this fold
    preds : list or np.ndarray
    probs : list or np.ndarray
    fold_ba : float
        Balanced accuracy for this fold
    train_loss, val_loss, train_acc : float or None
        Set for NN models; left None for HMM/HSMM
    epochs_trained : int or None
    early_stopped : bool

    Returns
    -------
    dict
    """
    return {
        "fold":          fold_idx,
        "test_subjects": test_subjects.tolist() if subject_ids is not None else [fold_idx],
        "train_loss":    train_loss,
        "val_loss":      val_loss,
        "train_acc":     train_acc,
        "val_acc":       float(fold_ba),
        "y_true":        y_test if isinstance(y_test, list) else y_test.tolist(),
        "y_pred":        preds if isinstance(preds, list) else preds.tolist(),
        "y_proba":       probs if isinstance(probs, list) else probs.tolist(),
        "epochs_trained": epochs_trained,
        "early_stopped": early_stopped,
    }


def print_best(model_name: str, best_params: Dict, best_score: float) -> None:
    """Print best hyperparameters and balanced accuracy."""
    print(f"\n[{model_name}] Best params: {best_params}")
    print(f"[{model_name}] Best balanced accuracy: {best_score:.4f}")
