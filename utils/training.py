"""
Shared training utilities for all model classes.

Extracted to eliminate duplicated code across cnn_model, rnn_model,
transformer_model, hmm_model, and hsmm_model.
"""
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler


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
    inner_val_loss: Optional[float] = None,
    hyperparameters: Optional[Dict] = None,
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
    train_loss : float or None
        Best training loss reached this fold (informational only).
    val_loss : float or None
        Loss on the outer held-out subject — never used to make any
        decision, kept purely for post-hoc generalization-gap diagnostics
        (utils/overfitting_detection.py). Set for NN models; left None for
        HMM/HSMM.
    train_acc : float or None
    epochs_trained : int or None
    early_stopped : bool
    inner_val_loss : float or None
        Mean inner-CV validation loss achieved by this fold's chosen
        hyperparameter config — this is what actually drives both early
        stopping and hyperparameter selection. Distinct from val_loss (the
        outer test subject), which the held-out subject never influences.
    hyperparameters : Dict or None
        The hyperparameter config chosen for this fold via inner CV. Folds
        are allowed to pick different configs from each other — this is
        expected in nested CV, not a bug — so there's no longer a single
        global "best_params" the way there was with the flat grid search.

    Returns
    -------
    dict
    """
    return {
        "fold":          fold_idx,
        "test_subjects": test_subjects.tolist() if subject_ids is not None else [fold_idx],
        "train_loss":    train_loss,
        "val_loss":      val_loss,
        "inner_val_loss": inner_val_loss,
        "train_acc":     train_acc,
        "val_acc":       float(fold_ba),
        "y_true":        y_test if isinstance(y_test, list) else y_test.tolist(),
        "y_pred":        preds if isinstance(preds, list) else preds.tolist(),
        "y_proba":       probs if isinstance(probs, list) else probs.tolist(),
        "epochs_trained": epochs_trained,
        "early_stopped": early_stopped,
        "hyperparameters": hyperparameters,
    }


def fold_truncate_and_scale(
    train_signals: List[np.ndarray],
    test_signals: List[np.ndarray],
    output_format: str = "3d",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fold-safe replacement for TruncatePreprocessor's global fit: computes
    the truncation length (T_min) and the z-score scaler from
    ``train_signals`` only, then applies both to ``test_signals`` — so the
    held-out fold never contributes to either statistic.

    Parameters
    ----------
    train_signals : list of (T_i, C) arrays
        This fold's training subjects (raw, unscaled).
    test_signals : list of (T_i, C) arrays
        This fold's held-out subject(s) (raw, unscaled).
    output_format : str
        '3d' for (N, T, C) (RNN/Transformer) or 'channels_first' for
        (N, C, T) (CNN).

    Returns
    -------
    X_train, X_test : np.ndarray
        Truncated, z-scored arrays in the requested output format.
    """
    T_min = min(s.shape[0] for s in train_signals)

    def _clip(sig: np.ndarray) -> np.ndarray:
        # Keep the last T_min timesteps, mirroring TruncatePreprocessor.
        # A held-out subject shorter than T_min can only happen on the test
        # side (T_min is a training-only statistic) — zero-pad at the front
        # instead of truncating, since there's nothing left to cut.
        if sig.shape[0] >= T_min:
            return sig[-T_min:]
        pad = np.zeros((T_min - sig.shape[0], sig.shape[1]), dtype=sig.dtype)
        return np.concatenate([pad, sig], axis=0)

    X_train = np.stack([_clip(s) for s in train_signals], axis=0)  # (N_train, T_min, C)
    X_test  = np.stack([_clip(s) for s in test_signals], axis=0)   # (N_test,  T_min, C)

    N_train, T, C = X_train.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(N_train * T, C)).reshape(N_train, T, C)

    N_test = X_test.shape[0]
    X_test = scaler.transform(X_test.reshape(N_test * T, C)).reshape(N_test, T, C)

    if output_format == "channels_first":
        X_train = X_train.transpose(0, 2, 1)
        X_test  = X_test.transpose(0, 2, 1)

    return X_train, X_test


def scale_sequences_global(signals: List[np.ndarray]) -> List[np.ndarray]:
    """
    Fit one StandardScaler on all frames of all sequences concatenated,
    then transform each sequence independently.

    Used for whole-dataset descriptive/interpretability fits (e.g. HMM's
    fit_for_analysis(), which is explicitly documented as a full-data,
    no-CV fit for Phase-2 analysis, not a performance estimate) — there is
    no held-out subject to protect there. For LOSO evaluation, use
    fold_scale_variable_length() below instead.

    Parameters
    ----------
    signals : list of (T_i, C) arrays

    Returns
    -------
    list of (T_i, C) arrays, z-scored
    """
    scaler = StandardScaler()
    scaler.fit(np.vstack(signals))
    return [scaler.transform(s) for s in signals]


def fold_scale_variable_length(
    train_signals: List[np.ndarray],
    test_signals: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Fold-safe z-score normalization for variable-length sequences (HMM/
    HSMM): fits the scaler on train_signals only, then applies it to both
    train and test — so the held-out fold never contributes to the
    statistic.

    Unlike fold_truncate_and_scale() (CNN/RNN/Transformer), there's no
    truncation step here — each sequence keeps its own length, matching
    hmmlearn's native variable-length support via the ``lengths`` parameter.

    Parameters
    ----------
    train_signals : list of (T_i, C) arrays — this fold's training subjects
    test_signals : list of (T_i, C) arrays — this fold's held-out subject(s)

    Returns
    -------
    X_train, X_test : list of (T_i, C) arrays, z-scored
    """
    scaler = StandardScaler()
    scaler.fit(np.vstack(train_signals))
    X_train = [scaler.transform(s) for s in train_signals]
    X_test  = [scaler.transform(s) for s in test_signals]
    return X_train, X_test


def build_inner_validation_split(
    train_sample_idx: np.ndarray,
    subject_ids: Optional[np.ndarray],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split one outer LOSO fold's training samples into an inner-train /
    inner-validation partition, used only to drive early stopping. The
    outer held-out subject is never part of ``train_sample_idx`` to begin
    with, so it can't leak in here either.

    Splits by subject (not by raw sample index) when subject_ids is
    available, so a subject with multiple samples can't end up on both
    sides of the split.

    Parameters
    ----------
    train_sample_idx : np.ndarray
        Global sample indices belonging to this fold's training set.
    subject_ids : np.ndarray or None
    val_fraction : float
        Fraction of training subjects held back for the validation slice.
    seed : int
        Varies by fold (caller should pass e.g. ``42 + fold_idx``) so the
        inner split isn't identical across every fold.

    Returns
    -------
    inner_train_pos, inner_val_pos : np.ndarray
        *Positions* into train_sample_idx (i.e. row indices into any array
        built by indexing with train_sample_idx) — not global sample ids.
    """
    n = len(train_sample_idx)
    rng = np.random.RandomState(seed)

    if subject_ids is not None:
        fold_subjects = subject_ids[train_sample_idx]
        unique_fold_subjects = np.unique(fold_subjects)
        shuffled = unique_fold_subjects.copy()
        rng.shuffle(shuffled)
        n_val_subjects = max(1, int(round(len(shuffled) * val_fraction)))
        val_subjects = set(shuffled[:n_val_subjects].tolist())
        val_mask = np.array([s in val_subjects for s in fold_subjects])
    else:
        positions = np.arange(n)
        rng.shuffle(positions)
        n_val = max(1, int(round(n * val_fraction)))
        val_mask = np.zeros(n, dtype=bool)
        val_mask[positions[:n_val]] = True

    inner_val_pos   = np.where(val_mask)[0]
    inner_train_pos = np.where(~val_mask)[0]
    return inner_train_pos, inner_val_pos


def build_inner_kfold_splits(
    train_sample_idx: np.ndarray,
    subject_ids: Optional[np.ndarray],
    n_inner_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split one outer LOSO fold's training samples into ``n_inner_folds``
    inner-train/inner-validation partitions — used for BOTH hyperparameter
    selection and early stopping, replacing the single-split version above.
    The outer held-out subject is never part of ``train_sample_idx`` to
    begin with, so it can't leak into this either.

    Splits by subject (not by raw sample index) when subject_ids is
    available, so a subject with multiple samples can't end up on both
    sides of any one inner fold.

    Parameters
    ----------
    train_sample_idx : np.ndarray
        Global sample indices belonging to this outer fold's training set.
    subject_ids : np.ndarray or None
    n_inner_folds : int
        Number of inner folds. Reduced automatically if there are fewer
        training subjects than this.
    seed : int
        Varies by outer fold (caller should pass e.g. ``42 + fold_idx``).

    Returns
    -------
    list of (inner_train_pos, inner_val_pos)
        One pair per inner fold. Both are *positions* into
        train_sample_idx (row indices into any array built by indexing
        with train_sample_idx) — not global sample ids.
    """
    if subject_ids is not None:
        fold_subjects = subject_ids[train_sample_idx]
        unique_fold_subjects = np.unique(fold_subjects)
        n_splits = min(n_inner_folds, len(unique_fold_subjects))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        splits = []
        for inner_train_subj_pos, inner_val_subj_pos in kf.split(unique_fold_subjects):
            inner_train_subjects = set(unique_fold_subjects[inner_train_subj_pos].tolist())
            inner_val_subjects   = set(unique_fold_subjects[inner_val_subj_pos].tolist())
            inner_train_pos = np.where(np.isin(fold_subjects, list(inner_train_subjects)))[0]
            inner_val_pos   = np.where(np.isin(fold_subjects, list(inner_val_subjects)))[0]
            splits.append((inner_train_pos, inner_val_pos))
        return splits
    else:
        n = len(train_sample_idx)
        n_splits = min(n_inner_folds, n)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(np.arange(n)))


def print_best(model_name: str, best_params: Dict, best_score: float) -> None:
    """Print best hyperparameters and balanced accuracy."""
    print(f"\n[{model_name}] Best params: {best_params}")
    print(f"[{model_name}] Best balanced accuracy: {best_score:.4f}")


def most_common_config(chosen_configs: List[Dict]) -> Dict:
    """
    Return the most frequently chosen config across outer folds.

    With nested CV, each outer fold picks its own hyperparameters via
    inner CV — they don't have to agree. This is purely a representative
    summary for logging/checkpointing; the authoritative record is each
    fold's own choice in per_fold_results[i]['hyperparameters'].

    Parameters
    ----------
    chosen_configs : list of Dict
        One hyperparameter dict per outer fold (fold-iteration order).

    Returns
    -------
    Dict
        The most common config (first-seen on ties).
    """
    key = lambda cfg: str(sorted(cfg.items()))
    counts: Dict[str, int] = {}
    for cfg in chosen_configs:
        counts[key(cfg)] = counts.get(key(cfg), 0) + 1
    best_key = max(counts, key=counts.get)
    return next(cfg for cfg in chosen_configs if key(cfg) == best_key)
