"""
Evaluation metrics and visualization utilities.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from typing import Dict, Any, Optional


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


def print_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    paradigm: Optional[int] = None,
    task: Optional[int] = None,
    save_figure: bool = False
) -> Dict[str, Any]:
    """
    Print and optionally visualize metrics.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    y_true, y_pred, y_proba : np.ndarray
        True labels, predictions, and probabilities
    paradigm : int, optional
        Paradigm number for figure naming
    task : int, optional
        Task number for figure naming
    save_figure : bool
        Whether to save probability density figure
    
    Returns
    -------
    Dict[str, Any]
        Computed metrics
    """
    metrics = compute_metrics(y_true, y_pred, y_proba)
    
    print(f"\n{'='*60}")
    print(f"Metrics for {model_name}")
    print(f"{'='*60}")
    print(f"Balanced Accuracy:  {metrics['ba']:.3f}")
    print(f"Recall (Sensitivity): {metrics['recall']:.3f}")
    print(f"Precision:          {metrics['precision']:.3f}")
    print(f"F1 Score:           {metrics['f1']:.3f}")
    print(f"AUC:                {metrics['auc']:.3f} "
          f"[{metrics['auc_ci_low']:.3f} - {metrics['auc_ci_high']:.3f}]")
    print(f"{'='*60}\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=[0, 1], zero_division=0))
    
    # Optionally save probability density curve
    if save_figure and paradigm is not None and task is not None:
        plot_probability_density(
            y_true=y_true,
            y_proba=y_proba,
            model_name=model_name,
            paradigm=paradigm,
            task=task
        )
    
    return metrics


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
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
    
    lower = np.percentile(aucs, (1 - ci) * 50)
    upper = np.percentile(aucs, 100 - (1 - ci) * 50)
    
    return np.mean(aucs), (lower, upper)


def kde_vals(samples: np.ndarray, x: np.ndarray, jitter_scale: float = 1e-3):
    """
    Compute KDE values with numerical stability.
    
    Parameters
    ----------
    samples : np.ndarray
        Sample values
    x : np.ndarray
        Points at which to evaluate KDE
    jitter_scale : float
        Jitter for numerical stability
    
    Returns
    -------
    np.ndarray or None
        KDE values, or None if computation fails
    """
    samples = np.asarray(samples, dtype=float).ravel()
    
    if samples.size < 2:
        return None
    
    s = samples.std()
    
    if s < 1e-8:
        # Near-constant; draw narrow normal
        mu = samples.mean()
        width = max(jitter_scale, 0.02)
        return norm.pdf(x, loc=mu, scale=width)
    
    try:
        kde = gaussian_kde(samples)
        return kde(x)
    except np.linalg.LinAlgError:
        # Add jitter and retry
        eps = max(jitter_scale, 1e-3 * s)
        kde = gaussian_kde(samples + np.random.normal(0, eps, size=samples.shape))
        return kde(x)


def plot_probability_density(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    paradigm: int,
    task: int
):
    """
    Plot probability density by true label.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    model_name : str
        Model name for title
    paradigm : int
        Paradigm number
    task : int
        Task number
    """
    probs_0 = y_proba[y_true == 0]
    probs_1 = y_proba[y_true == 1]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Histograms
    ax.hist(probs_0, bins=30, range=(0, 1), density=True, 
            alpha=0.5, label="Class 0 (Controls)", color='blue')
    ax.hist(probs_1, bins=30, range=(0, 1), density=True, 
            alpha=0.5, label="Class 1 (Patients)", color='red')
    
    # KDE curves
    x = np.linspace(0, 1, 300)
    y0 = kde_vals(probs_0, x)
    y1 = kde_vals(probs_1, x)
    
    if y0 is not None:
        ax.plot(x, y0, linewidth=2, label="Class 0 - KDE", color='darkblue')
    if y1 is not None:
        ax.plot(x, y1, linewidth=2, label="Class 1 - KDE", color='darkred')
    
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted Probability (Class 1)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"{model_name} - Task {task} - Paradigm {paradigm}\n"
        f"Probability Density by True Label",
        fontsize=13
    )
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    # filename = f"prob_density_{model_name}_P{paradigm}.png"
    # filepath = get_figure_path(task, filename)
    # fig.savefig(filepath, dpi=300, bbox_inches="tight")
    # print(f"[Figure saved] {filepath}")
    # plt.close(fig)