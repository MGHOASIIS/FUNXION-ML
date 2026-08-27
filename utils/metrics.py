"""
Evaluation metrics and visualization utilities.
"""
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    hamming_loss,
)
from typing import Dict, Any, List


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, Any]:
    """
    Compute classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    
    # Ensure binary classification
    y_pred = np.clip(y_pred, 0, 1)
    labels = [0, 1]
    
    # Compute metrics
    ba = balanced_accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, labels=labels, zero_division=0)
    precision = precision_score(y_true, y_pred, labels=labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, zero_division=0)
    
    # AUC with bootstrap CI
    mean_auc, (ci_low, ci_high) = auc_ci_bootstrap(y_true, y_proba)
    
    return {
        "ba": round(ba, 3),
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "f1": round(f1, 3),
        "auc": round(mean_auc, 3),
        "auc_ci_low": round(ci_low, 3),
        "auc_ci_high": round(ci_high, 3)
    }



def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_names: List[str],
) -> Dict[str, Any]:
    """
    Multi-label counterpart of compute_metrics(). y_true/y_pred/y_proba are
    (N, len(label_names)) — a subject can be 1 in more than one column.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Multi-hot label matrices, shape (N, n_labels)
    y_proba : np.ndarray
        Per-label predicted probabilities, shape (N, n_labels)
    label_names : List[str]
        Names for each column, in order

    Returns
    -------
    Dict[str, Any]
        Aggregate metrics (subset_accuracy, hamming_loss, macro/micro F1,
        macro balanced accuracy) plus a "per_label" breakdown.
    """
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), 0, 1)
    y_proba = np.asarray(y_proba)

    n_labels = y_true.shape[1]
    per_label: Dict[str, Any] = {}
    label_bas = []

    for i, name in enumerate(label_names[:n_labels]):
        col_true, col_pred, col_proba = y_true[:, i], y_pred[:, i], y_proba[:, i]
        if len(np.unique(col_true)) > 1:
            ba = balanced_accuracy_score(col_true, col_pred)
            mean_auc, (ci_low, ci_high) = auc_ci_bootstrap(col_true, col_proba)
        else:
            # Only one class present for this label in this split — BA/AUC undefined.
            ba, mean_auc, ci_low, ci_high = (float("nan"),) * 4

        if not np.isnan(ba):
            label_bas.append(ba)

        per_label[name] = {
            "ba": None if np.isnan(ba) else round(ba, 3),
            "auc": None if np.isnan(mean_auc) else round(mean_auc, 3),
            "auc_ci_low": None if np.isnan(ci_low) else round(ci_low, 3),
            "auc_ci_high": None if np.isnan(ci_high) else round(ci_high, 3),
        }

    macro_ba = float(np.mean(label_bas)) if label_bas else float("nan")

    return {
        "subset_accuracy": round(accuracy_score(y_true, y_pred), 3),
        "hamming_loss": round(hamming_loss(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3),
        "micro_f1": round(f1_score(y_true, y_pred, average="micro", zero_division=0), 3),
        "macro_balanced_accuracy": None if np.isnan(macro_ba) else round(macro_ba, 3),
        "per_label": per_label,
    }


def auc_ci_bootstrap(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95
) -> tuple:
    """
    Compute AUC with bootstrap confidence interval.
    
    Returns
    -------
    mean_auc : float
    (ci_low, ci_high) : tuple
    """
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # skip draws where only one class is present
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))

    if not aucs:
        nan = float("nan")
        return nan, (nan, nan)

    lower = np.percentile(aucs, (1 - ci) * 50)
    upper = np.percentile(aucs, 100 - (1 - ci) * 50)

    return np.mean(aucs), (lower, upper)