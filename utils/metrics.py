"""
Evaluation metrics and visualization utilities.
"""
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)
from typing import Dict, Any


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