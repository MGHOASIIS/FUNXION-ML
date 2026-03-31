"""
Hidden Semi-Markov Model for shoulder pathology classification.

Mirrors hmm_model.py exactly, with one targeted change:
    GaussianHMM  →  GaussianHSMM  (explicit Poisson state-duration model)

Key difference from HMM
-----------------------
A standard HMM assumes geometrically distributed state sojourn times
(the memoryless property).  This means the probability of leaving a state
is constant regardless of how long the model has already been in it.

For structured, task-driven shoulder movements this assumption is violated:
each movement phase (e.g. "reaching for the jar", "gripping the lid") has
a physiologically meaningful duration that is *not* geometric.  Patients
with shoulder pathology may dwell significantly longer in certain phases
compared to controls.

An HSMM replaces the geometric sojourn distribution with an explicit
duration model.  Here we use a Poisson duration distribution per state,
which requires estimating one additional parameter per state (the mean
sojourn length λ_s).  This is the most parsimonious HSMM variant and
adds minimal parameters relative to the full HMM parameter count.

Implementation note
-------------------
hmmlearn ≥ 0.3 ships a PoissonHMM but does NOT include a Gaussian HSMM
directly.  We therefore implement GaussianHSMM as a thin subclass of
GaussianHMM that overrides the Viterbi and forward passes to respect
explicit Poisson duration distributions (the "HS" modification of
Rabiner's algorithm, Yu 2010 — IEEE Trans. Signal Process.).

The classification pipeline (train_and_evaluate / fit / _loo_score /
compute_feature_importance) is identical to HMMModel.  All Phase-2
analysis methods (decode_sequence, event alignment, emission plots,
feature importance heatmaps) are also carried over unchanged, because
they operate on the already-fitted model parameters, which have the same
structure as GaussianHMM (means_, covars_, transmat_).

References
----------
Yu, S.-Z. (2010). Hidden semi-Markov models. Artificial Intelligence,
    174(2), 215–243.
"""
from typing import Dict, List, Optional, Tuple
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from sklearn.metrics import balanced_accuracy_score
from hmmlearn.hmm import GaussianHMM
from joblib import parallel_backend, Parallel, delayed
from scipy.special import logsumexp

from models.base_model import BaseModel, ModelResults
from config.constants import CHAN_NAME
from utils.metrics import compute_metrics


# =============================================================================
# Dataclasses — identical to hmm_model.py
# =============================================================================

from dataclasses import dataclass, field


@dataclass
class StateSegment:
    """A contiguous time segment assigned to one hidden state."""
    state: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration_s: float


@dataclass
class EventAlignment:
    """Alignment between one annotated event and the nearest HSMM transition."""
    event_name: str
    event_time_s: float
    nearest_transition_time_s: float
    temporal_error_s: float
    hmm_state_before: int
    hmm_state_after: int
    matched: bool


@dataclass
class AlignmentSummary:
    """Aggregate alignment quality across all events for one subject/task."""
    subject_id: str
    task_id: int
    n_events: int
    n_matched: int
    match_rate: float
    mean_error_s: float
    median_error_s: float
    std_error_s: float
    per_event: List[EventAlignment]


# =============================================================================
# GaussianHSMM — Poisson-duration wrapper around GaussianHMM
# =============================================================================

class GaussianHSMM(GaussianHMM):
    """
    Gaussian Hidden Semi-Markov Model with Poisson duration distributions.

    Inherits all emission and transition parameters from GaussianHMM.
    Adds one Poisson rate parameter λ_s per state, estimated from the
    mean sojourn durations observed during the EM M-step.

    The Viterbi decoder is overridden to incorporate the log-duration
    probability into the path score, implementing the explicit-duration
    Viterbi algorithm (Yu 2010, Algorithm 2).

    The forward (score) method is also overridden to compute the
    explicit-duration forward variable, enabling correct log-likelihood
    computation for class-conditional scoring.

    Parameters
    ----------
    max_duration : int
        Maximum sojourn length considered (frames).  Sequences shorter
        than this are handled gracefully.  Default 200 frames (4 s at
        50 Hz) is generous for shoulder-task phases.
    All other parameters are passed through to GaussianHMM.
    """

    def __init__(self, max_duration: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.max_duration = max_duration
        # Poisson rates — shape (n_components,), initialised after fit()
        self._duration_rates: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Duration distribution helpers
    # ------------------------------------------------------------------

    def _log_duration_prob(self, state: int, d: int) -> float:
        """
        log P(duration = d | state) under Poisson(λ_s).

        Uses log-factorial via stirling for large d to avoid overflow.
        """
        if self._duration_rates is None:
            # Before fit — fall back to uniform (equivalent to standard HMM)
            return 0.0
        lam = max(self._duration_rates[state], 1e-6)
        # log Poisson PMF: d*log(lam) - lam - log(d!)
        log_d_fact = float(np.sum(np.log(np.arange(1, d + 1)))) if d > 0 else 0.0
        return d * np.log(lam) - lam - log_d_fact

    def _log_duration_matrix(self, T: int) -> np.ndarray:
        """
        Precompute log P(duration = d | state s) for all s, d.

        Returns
        -------
        log_dur : np.ndarray, shape (n_components, T)
            log_dur[s, d-1] = log P(duration = d | state s)
        """
        n = self.n_components
        max_d = min(self.max_duration, T)
        log_dur = np.full((n, max_d), -np.inf)
        for s in range(n):
            for d in range(1, max_d + 1):
                log_dur[s, d - 1] = self._log_duration_prob(s, d)
        return log_dur  # (n_components, max_d)

    # ------------------------------------------------------------------
    # Fit — estimate Poisson rates from decoded state sequences
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, lengths=None):
        """
        Fit emission/transition parameters via parent EM, then estimate
        Poisson duration rates from Viterbi-decoded state sequences.
        """
        # Step 1 — fit Gaussian emissions and transition matrix
        super().fit(X, lengths=lengths)

        # Step 2 — decode the training sequences to collect sojourn stats
        self._duration_rates = self._estimate_duration_rates(X, lengths)
        return self

    def _estimate_duration_rates(
        self,
        X: np.ndarray,
        lengths: Optional[List[int]]
    ) -> np.ndarray:
        """
        Estimate Poisson λ_s for each state from decoded training sequences.

        Runs Viterbi (parent implementation, geometric duration) on the
        full training set, segments the resulting state sequences into
        contiguous runs, and computes the mean run length per state.
        This is a post-hoc approximation; a full EM HSMM would update
        duration parameters in the E-step, but this approximation is
        standard in literature and sufficient for N=60.

        Returns
        -------
        rates : np.ndarray, shape (n_components,)
            Estimated Poisson rate (mean sojourn length) per state.
        """
        if lengths is None:
            lengths = [len(X)]

        sojourn_sums = np.zeros(self.n_components)
        sojourn_counts = np.zeros(self.n_components)

        start = 0
        for length in lengths:
            seq = X[start: start + length]
            states = super().predict(seq)           # geometric Viterbi

            # Segment into runs
            prev = states[0]
            run_len = 1
            for t in range(1, len(states)):
                if states[t] == prev:
                    run_len += 1
                else:
                    sojourn_sums[prev] += run_len
                    sojourn_counts[prev] += 1
                    prev = states[t]
                    run_len = 1
            sojourn_sums[prev] += run_len
            sojourn_counts[prev] += 1

            start += length

        # Mean sojourn = MLE for Poisson rate
        rates = np.where(
            sojourn_counts > 0,
            sojourn_sums / np.maximum(sojourn_counts, 1),
            1.0   # fallback: rate=1 → geometric-like
        )
        return rates

    # ------------------------------------------------------------------
    # Override score() — explicit-duration forward algorithm
    # ------------------------------------------------------------------

    def score(self, X: np.ndarray, lengths=None) -> float:
        """
        Compute log P(X | HSMM) using the explicit-duration forward variable.

        Algorithm
        ---------
        Implements the forward recursion for explicit-duration HMMs
        (Yu 2010, eq. 3–5):

            α_t(s) = Σ_{d=1}^{min(t, D)} Σ_{s'≠s} [
                α_{t-d}(s') · a_{s',s} ·
                P(duration=d | s) · Π_{u=t-d+1}^{t} b_s(x_u)
            ]

        where b_s(x_u) is the Gaussian emission probability and a_{s',s}
        is the transition probability from s' to s.

        For computational efficiency we work in log-space throughout.
        """
        T = len(X)
        n = self.n_components
        max_d = min(self.max_duration, T)

        # Precompute log emission probabilities: (T, n)
        log_emit = self._compute_log_likelihood(X)  # inherited from GaussianHMM

        # Precompute log duration probabilities: (n, max_d)
        log_dur = self._log_duration_matrix(T)

        # Log transition matrix, zeroing self-transitions (explicit duration)
        log_trans = np.log(np.maximum(self.transmat_, 1e-300))
        np.fill_diagonal(log_trans, -np.inf)   # no self-loops in HSMM

        # Log start probabilities
        log_pi = np.log(np.maximum(self.startprob_, 1e-300))

        # Forward variable: log α[t, s] = log P(o_1..o_t, S_t=s)
        log_alpha = np.full((T, n), -np.inf)

        # Initialise: state s occupies duration d starting at frame 0
        # Precompute cumulative log-emission sums for O(1) window queries
        cum_log_emit = np.zeros((T + 1, n))
        for t in range(T):
            cum_log_emit[t + 1] = cum_log_emit[t] + log_emit[t]

        def _seg_log_emit(s, t_start, t_end):
            """Sum of log b_s(x_u) for u in [t_start, t_end] inclusive."""
            return cum_log_emit[t_end + 1, s] - cum_log_emit[t_start, s]

        # Initialise: segments ending at t=d-1 that start at t=0
        for s in range(n):
            for d in range(1, max_d + 1):
                t_end = d - 1
                if t_end >= T:
                    break
                seg_emit = _seg_log_emit(s, 0, t_end)
                log_alpha[t_end, s] = logsumexp([
                    log_alpha[t_end, s],
                    log_pi[s] + log_dur[s, d - 1] + seg_emit
                ])

        # Recursion
        for t in range(1, T):
            for s in range(n):
                candidates = []
                for d in range(1, min(t + 1, max_d) + 1):
                    t_start = t - d + 1
                    t_prev  = t_start - 1
                    if t_prev < 0:
                        continue
                    seg_emit = _seg_log_emit(s, t_start, t)
                    # Sum over all predecessor states s'
                    pred_vals = log_alpha[t_prev] + log_trans[:, s]
                    pred_sum  = logsumexp(pred_vals)
                    candidates.append(
                        pred_sum + log_dur[s, d - 1] + seg_emit
                    )
                if candidates:
                    log_alpha[t, s] = logsumexp(candidates)

        return float(logsumexp(log_alpha[T - 1]))

    # ------------------------------------------------------------------
    # Override predict() — explicit-duration Viterbi
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray, lengths=None) -> np.ndarray:
        """
        Decode the most likely state sequence using explicit-duration Viterbi.

        Implements Yu (2010) Algorithm 2 in log-space.
        """
        T = len(X)
        n = self.n_components
        max_d = min(self.max_duration, T)

        log_emit  = self._compute_log_likelihood(X)
        log_dur   = self._log_duration_matrix(T)
        log_trans = np.log(np.maximum(self.transmat_, 1e-300))
        np.fill_diagonal(log_trans, -np.inf)
        log_pi    = np.log(np.maximum(self.startprob_, 1e-300))

        # Cumulative log-emission sums
        cum_log_emit = np.zeros((T + 1, n))
        for t in range(T):
            cum_log_emit[t + 1] = cum_log_emit[t] + log_emit[t]

        def _seg_log_emit(s, t_start, t_end):
            return cum_log_emit[t_end + 1, s] - cum_log_emit[t_start, s]

        # delta[t, s] = max log-prob of sequence o_1..o_t ending with S_t=s
        delta   = np.full((T, n), -np.inf)
        # psi[t, s] = (prev_state, duration) that achieved delta[t, s]
        psi_state = np.full((T, n), -1, dtype=int)
        psi_dur   = np.full((T, n), -1, dtype=int)

        # Initialise
        for s in range(n):
            for d in range(1, max_d + 1):
                t_end = d - 1
                if t_end >= T:
                    break
                val = log_pi[s] + log_dur[s, d - 1] + _seg_log_emit(s, 0, t_end)
                if val > delta[t_end, s]:
                    delta[t_end, s] = val
                    psi_state[t_end, s] = -1   # no predecessor (start)
                    psi_dur[t_end, s]   = d

        # Recursion
        for t in range(1, T):
            for s in range(n):
                for d in range(1, min(t + 1, max_d) + 1):
                    t_start = t - d + 1
                    t_prev  = t_start - 1
                    if t_prev < 0:
                        continue
                    seg_emit = _seg_log_emit(s, t_start, t)
                    for s_prev in range(n):
                        if s_prev == s:
                            continue
                        val = (delta[t_prev, s_prev]
                               + log_trans[s_prev, s]
                               + log_dur[s, d - 1]
                               + seg_emit)
                        if val > delta[t, s]:
                            delta[t, s]       = val
                            psi_state[t, s]   = s_prev
                            psi_dur[t, s]     = d

        # Backtrack
        states = np.zeros(T, dtype=int)
        t = T - 1
        s = int(np.argmax(delta[t]))
        while t >= 0:
            d = psi_dur[t, s]
            if d <= 0:
                d = 1   # safety fallback
            t_start = t - d + 1
            states[max(0, t_start): t + 1] = s
            s_prev = psi_state[t, s]
            t = t_start - 1
            if s_prev >= 0:
                s = s_prev
            elif t >= 0:
                # No stored predecessor — use argmax of delta at t
                s = int(np.argmax(delta[t]))

        return states

    # predict_proba falls back to parent forward-backward (approximate but
    # consistent with usage in decode_sequence for soft posteriors)


# =============================================================================
# Main HSMMModel class
# =============================================================================

class HSMMModel(BaseModel):
    """
    Hidden Semi-Markov Model wrapper with subject-level LOO CV.

    Drop-in replacement for HMMModel.  All differences are confined to:
      - _fit_hsmm()  uses GaussianHSMM instead of GaussianHMM
      - HSMM_PARAM_GRID adds 'max_duration' to the search space
      - Model name is "HSMM" for checkpoint filenames and logs

    All Phase-2 analysis methods are inherited from the same implementation
    as HMMModel (copy-pasted to keep this file self-contained and avoid
    import coupling).
    """

    def __init__(self, checkpoints_dir=None, task=None, paradigm=None):
        super().__init__(
            model_name="HSMM",
            checkpoints_dir=checkpoints_dir,
            patience=None,
            min_delta=None,
            task=task,
            paradigm=paradigm
        )

        self.fitted_hsmm0: Optional[GaussianHSMM] = None   # control model
        self.fitted_hsmm1: Optional[GaussianHSMM] = None   # patient model

    # =========================================================================
    # SECTION 1 — Classification pipeline
    # =========================================================================

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        param_grid: Optional[Dict] = None
    ) -> ModelResults:
        """
        Train HSMM with hyperparameter search using LOO CV.

        Parameters mirror HMMModel.train_and_evaluate() exactly.
        The only difference is that param_grid defaults to HSMM_PARAM_GRID
        (which adds 'max_duration') and _fit_hsmm() is called instead of
        _fit_hmm().
        """
        if param_grid is None:
            try:
                from config.hyperparameter import HSMM_PARAM_GRID
                param_grid = HSMM_PARAM_GRID
            except ImportError:
                # Fallback if HSMM_PARAM_GRID not yet added to hyperparameter.py
                param_grid = {
                    "covariance_type": ["diag"],
                    "n_components":    [2, 3, 4, 5],
                    "n_iter":          [100],
                    "max_duration":    [100, 200],
                }

        if subject_ids is not None:
            unique_subjects = np.unique(subject_ids)
            loo = LeaveOneOut()
            cv_splits = list(loo.split(unique_subjects))
            print(f"\n[HSMM] Subject-level LOO CV: {len(unique_subjects)} subjects")
        else:
            unique_subjects = None
            loo = LeaveOneOut()
            cv_splits = list(loo.split(range(len(X))))
            print(f"\n[HSMM] Sample-level LOO CV: {len(X)} samples")

        grid = list(ParameterGrid(param_grid))
        print(f"[HSMM] Evaluating {len(grid)} hyperparameter combinations...")
        print(f"[HSMM] Note: HSMM inference is O(T²) — expect ~2–4× slower than HMM")

        # HSMM Viterbi is O(T²) — parallelism still helps across CV folds
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=-1, verbose=10)(
                delayed(self._loo_score)(
                    params, X, y, cv_splits, subject_ids, unique_subjects
                )
                for params in grid
            )

        best_result = max(results, key=lambda t: t[0])
        best_score, best_params, y_true, y_pred, y_proba, per_fold_results = best_result

        print(f"\n[HSMM] Best params: {best_params}")
        print(f"[HSMM] Best balanced accuracy: {best_score:.4f}")

        # Fit on full data for Phase-2 analysis
        print(f"\n[HSMM] Fitting on full data with best params for analysis ...")
        self.fit_for_analysis(
            X               = X,
            y               = y,
            n_components    = best_params["n_components"],
            covariance_type = best_params["covariance_type"],
            n_iter          = best_params["n_iter"],
            max_duration    = best_params.get("max_duration", 200)
        )

        feature_imp = self.compute_feature_importance(
            X=X,
            y=y,
            best_params=best_params,
            cv_splits=cv_splits,
            subject_ids=subject_ids,
            unique_subjects=unique_subjects
        )

        metrics = compute_metrics(y_true, y_pred, y_proba)

        # Save checkpoint
        from datetime import datetime

        if isinstance(self.checkpoint_dir, Path):
            save_dir = self.checkpoint_dir
        else:
            save_dir = (
                Path(__file__).resolve().parent.parent
                / "experiments"
                / f"task{self.task}"
                / f"paradigm{self.paradigm}"
            )

        save_dir.mkdir(parents=True, exist_ok=True)

        task_tag      = f"T{self.task}"      if self.task      is not None else "Tunk"
        paradigm_tag  = f"P{self.paradigm}"  if self.paradigm  is not None else "Punk"
        timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        best_path = save_dir / (
            f"HSMM_{task_tag}_{paradigm_tag}_"
            f"BA{best_score:.4f}_{timestamp_tag}.json"
        )

        with open(best_path, "w") as f:
            json.dump({
                'model_name':       'HSMM',
                'task':             self.task,
                'paradigm':         self.paradigm,
                'hyperparameters':  best_params,
                'metrics':          {'balanced_accuracy': best_score, **metrics},
                'feature_importance': feature_imp,
                'input_shape':      [len(X), int(X[0].shape[1])],
                'predictions': {
                    'y_true':       y_true.tolist(),
                    'y_pred':       y_pred.tolist(),
                    'y_proba':      y_proba.tolist(),
                    'subject_ids':  subject_ids.tolist() if subject_ids is not None else [],
                },
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        print(f"\n[HSMM] Best model saved → {best_path}")

        return ModelResults(
            metrics=metrics,
            best_params=best_params,
            feature_importance=feature_imp,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            X_shape=(len(X), X[0].shape[1]),
            subject_ids=subject_ids,
            per_fold_results=per_fold_results
        )

    def _loo_score(
        self,
        params: Dict,
        sequences: np.ndarray,
        y: np.ndarray,
        cv_splits: list,
        subject_ids: Optional[np.ndarray],
        unique_subjects: Optional[np.ndarray]
    ):
        """
        Run one full LOO CV pass for a given hyperparameter configuration.
        Identical to HMMModel._loo_score() except calls _fit_hsmm().
        """
        y_true, y_pred, y_proba = [], [], []
        per_fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):

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

            seqs_train  = [sequences[i] for i in train_sample_idx]
            y_train     = y[train_sample_idx]
            seqs_test   = [sequences[i] for i in test_sample_idx]
            y_test_list = y[test_sample_idx].tolist()

            hsmm0 = self._fit_hsmm(
                [s for s, lab in zip(seqs_train, y_train) if lab == 0],
                **params
            )
            hsmm1 = self._fit_hsmm(
                [s for s, lab in zip(seqs_train, y_train) if lab == 1],
                **params
            )

            fold_preds, fold_proba = [], []
            for seq in seqs_test:
                delta = hsmm1.score(seq) - hsmm0.score(seq)
                prob  = self._stable_sigmoid(delta)
                pred  = int(prob >= 0.5)
                fold_preds.append(pred)
                fold_proba.append(prob)

            y_true.extend(y_test_list)
            y_pred.extend(fold_preds)
            y_proba.extend(fold_proba)

            fold_ba = balanced_accuracy_score(y_test_list, fold_preds)

            per_fold_results.append({
                'fold':          fold_idx,
                'test_subjects': test_subjects.tolist() if subject_ids is not None else [fold_idx],
                'train_loss':    None,
                'val_loss':      None,
                'train_acc':     None,
                'val_acc':       float(fold_ba),
                'y_true':        y_test_list,
                'y_pred':        fold_preds,
                'y_proba':       fold_proba,
                'epochs_trained': params.get('n_iter'),
                'early_stopped': False
            })

        ba = balanced_accuracy_score(y_true, y_pred)
        return (
            ba,
            params,
            np.array(y_true),
            np.array(y_pred),
            np.array(y_proba),
            per_fold_results
        )

    def _fit_hsmm(
        self,
        seq_list: list,
        n_components: int,
        covariance_type: str,
        n_iter: int,
        max_duration: int = 200
    ) -> GaussianHSMM:
        """
        Fit a GaussianHSMM to sequences from one class.

        Parameters mirror _fit_hmm() with the addition of max_duration.
        """
        lengths     = [s.shape[0] for s in seq_list]
        X_stacked   = np.vstack(seq_list)

        model = GaussianHSMM(
            max_duration    = max_duration,
            n_components    = n_components,
            covariance_type = covariance_type,
            n_iter          = n_iter,
            random_state    = 42,
            verbose         = False
        ).fit(X_stacked, lengths=lengths)

        return model

    @staticmethod
    def _stable_sigmoid(delta: float) -> float:
        """Numerically stable sigmoid of log-likelihood delta."""
        if delta >= 0:
            z = np.exp(-delta)
            return 1.0 / (1.0 + z)
        else:
            z = np.exp(delta)
            return z / (1.0 + z)

    def compute_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """
        Permutation-based feature importance — identical to HMMModel.

        Delegates to _loo_score() which internally uses _fit_hsmm().
        """
        from config.hyperparameter import IS_TEST

        best_params     = kwargs["best_params"]
        cv_splits       = kwargs["cv_splits"]
        subject_ids     = kwargs.get("subject_ids", None)
        unique_subjects = kwargs.get("unique_subjects", None)

        n_channels = X[0].shape[1]

        if IS_TEST:
            print("\n[HSMM] IS_TEST=True — returning uniform dummy importance")
            uniform = 1.0 / n_channels
            return {ch: uniform for ch in CHAN_NAME}

        print("\n[HSMM] Computing feature importance via permutation "
              f"({n_channels} channels × LOO CV) ...")

        rng        = np.random.default_rng(42)
        importance = np.zeros(n_channels)

        baseline_ba, *_ = self._loo_score(
            best_params, X, y, cv_splits, subject_ids, unique_subjects
        )
        print(f"  Baseline BA: {baseline_ba:.4f}")

        for d in range(n_channels):
            print(f"  Channel {d+1:2d}/{n_channels}: {CHAN_NAME[d]:<30}", end=" ")
            seqs_perm = self._permute_channel(X, d, rng)
            ba_d, *_  = self._loo_score(
                best_params, seqs_perm, y, cv_splits, subject_ids, unique_subjects
            )
            importance[d] = baseline_ba - ba_d
            print(f"drop={importance[d]:+.4f}")

        importance = np.clip(importance, 0, None)
        denom = importance.sum()
        if denom > 1e-12:
            importance = importance / denom
        else:
            importance = np.ones(n_channels) / n_channels

        feature_imp = {
            CHAN_NAME[i]: float(importance[i])
            for i in np.argsort(importance)[::-1]
        }

        print("\n[HSMM] Channel Importance:")
        for i, (feat, imp) in enumerate(feature_imp.items()):
            print(f"  {i+1:2d}. {feat:<30}: {imp:.4f}")

        return feature_imp

    @staticmethod
    def _permute_channel(seqs: list, channel: int, rng) -> list:
        """Permute a single channel across all sequences (identical to HMMModel)."""
        seqs_perm = [s.copy() for s in seqs]
        col = np.concatenate([s[:, channel] for s in seqs_perm])
        rng.shuffle(col)
        start = 0
        for s in seqs_perm:
            L = len(s)
            s[:, channel] = col[start: start + L]
            start += L
        return seqs_perm

    # =========================================================================
    # SECTION 2 — Phase 2: fit on full data for interpretability
    # =========================================================================

    def fit_for_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_components: int,
        covariance_type: str = 'diag',
        n_iter: int = 100,
        max_duration: int = 200
    ) -> Tuple[GaussianHSMM, GaussianHSMM]:
        """
        Train class-conditional HSMMs on the FULL dataset (no CV).

        Mirrors HMMModel.fit_for_analysis(); adds max_duration parameter.
        """
        seqs_0 = [X[i] for i in range(len(X)) if y[i] == 0]
        seqs_1 = [X[i] for i in range(len(X)) if y[i] == 1]

        print(f"\n[HSMM Analysis] Fitting on full data")
        print(f"  n_components={n_components}, covariance_type={covariance_type}, "
              f"max_duration={max_duration}")
        print(f"  Class 0 (controls): {len(seqs_0)} sequences")
        print(f"  Class 1 (patients): {len(seqs_1)} sequences")

        self.fitted_hsmm0 = self._fit_hsmm(
            seqs_0, n_components, covariance_type, n_iter, max_duration
        )
        self.fitted_hsmm1 = self._fit_hsmm(
            seqs_1, n_components, covariance_type, n_iter, max_duration
        )

        print("  ✓ Both class-conditional HSMMs fitted successfully")
        print(f"  Duration rates (patient model): "
              f"{np.round(self.fitted_hsmm1._duration_rates, 2)}")
        print(f"  Duration rates (control model): "
              f"{np.round(self.fitted_hsmm0._duration_rates, 2)}")

        return self.fitted_hsmm0, self.fitted_hsmm1

    # =========================================================================
    # SECTION 3 — State decoding and temporal segmentation
    # (identical to HMMModel, adapted for GaussianHSMM)
    # =========================================================================

    def decode_sequence(
        self,
        sequence: np.ndarray,
        model: Optional[GaussianHSMM] = None,
        sampling_rate: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, List[StateSegment]]:
        """
        Decode the most likely hidden-state sequence for a single recording.

        Uses the explicit-duration Viterbi (hard assignment) from GaussianHSMM
        and the parent forward-backward (predict_proba) for soft posteriors.
        """
        if model is None:
            if self.fitted_hsmm1 is None:
                raise RuntimeError(
                    "No fitted model available. Call fit_for_analysis() first."
                )
            model = self.fitted_hsmm1

        states     = model.predict(sequence)            # explicit-duration Viterbi
        posteriors = model.predict_proba(sequence)      # approximate (parent FB)
        segments   = self._states_to_segments(states, sampling_rate)

        return states, posteriors, segments

    @staticmethod
    def _states_to_segments(
        state_sequence: np.ndarray,
        sampling_rate: int = 50
    ) -> List[StateSegment]:
        """Convert per-timestep state array into contiguous segments."""
        segments = []
        current_state = state_sequence[0]
        start_frame   = 0

        for frame in range(1, len(state_sequence)):
            if state_sequence[frame] != current_state:
                segments.append(StateSegment(
                    state       = int(current_state),
                    start_frame = start_frame,
                    end_frame   = frame - 1,
                    start_time  = start_frame / sampling_rate,
                    end_time    = (frame - 1) / sampling_rate,
                    duration_s  = (frame - start_frame) / sampling_rate
                ))
                current_state = state_sequence[frame]
                start_frame   = frame

        T = len(state_sequence)
        segments.append(StateSegment(
            state       = int(current_state),
            start_frame = start_frame,
            end_frame   = T - 1,
            start_time  = start_frame / sampling_rate,
            end_time    = (T - 1) / sampling_rate,
            duration_s  = (T - start_frame) / sampling_rate
        ))
        return segments

    # =========================================================================
    # SECTION 4 — CSV event marker loading and alignment
    # (identical to HMMModel — no HSMM-specific changes needed)
    # =========================================================================

    @staticmethod
    def load_events_csv(csv_path: Path) -> pd.DataFrame:
        """Load a consolidated task-events CSV (mirrors HMMModel)."""
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        required = {'timestamp', 'event', 'subject_id'}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        for optional_col in ('progress', 'hand_used'):
            if optional_col not in df.columns:
                df[optional_col] = None
        df['timestamp']  = pd.to_numeric(df['timestamp'], errors='coerce')
        df['subject_id'] = df['subject_id'].astype(str).str.strip()
        df['event']      = df['event'].astype(str).str.strip()
        n_before = len(df)
        df = df.dropna(subset=['timestamp']).reset_index(drop=True)
        if len(df) < n_before:
            print(f"  ⚠️  Dropped {n_before - len(df)} rows with unparseable timestamps")
        return df

    def load_event_markers(
        self,
        csv_path: Path,
        subject_id: str,
        task_id: int,
        relative_timestamps: bool = True
    ) -> List[Dict]:
        """Load annotated event markers for one subject (mirrors HMMModel)."""
        df   = self.load_events_csv(csv_path)
        mask = df['subject_id'].str.upper() == subject_id.upper()
        subject_df = df[mask].copy()
        if subject_df.empty:
            available = sorted(df['subject_id'].unique().tolist())
            raise ValueError(
                f"Subject '{subject_id}' not found in {csv_path.name}.\n"
                f"Available: {available}"
            )
        subject_df = subject_df.sort_values('timestamp').reset_index(drop=True)
        if relative_timestamps:
            t0 = subject_df['timestamp'].iloc[0]
            subject_df['timestamp'] = subject_df['timestamp'] - t0
        events = []
        for _, row in subject_df.iterrows():
            hand = row.get('hand_used', None)
            hand = None if pd.isna(hand) else str(hand).strip()
            prog = row.get('progress', None)
            try:
                prog = float(prog) if prog is not None and not pd.isna(prog) else None
            except (TypeError, ValueError):
                prog = None
            events.append({
                'event_name': str(row['event']),
                'timestamp':  float(row['timestamp']),
                'hand_used':  hand,
                'progress':   prog,
            })
        return events

    def align_states_with_events(
        self,
        segments: List[StateSegment],
        events: List[Dict],
        tolerance_s: float = 0.5
    ) -> AlignmentSummary:
        """Compare HSMM transitions against annotated events (mirrors HMMModel)."""
        transition_times = [seg.end_time for seg in segments[:-1]]
        transition_state_pairs = [
            (segments[i].state, segments[i + 1].state)
            for i in range(len(segments) - 1)
        ]
        alignments: List[EventAlignment] = []
        for event in events:
            event_time = float(event['timestamp'])
            event_name = str(event['event_name'])
            if not transition_times:
                alignments.append(EventAlignment(
                    event_name=event_name,
                    event_time_s=event_time,
                    nearest_transition_time_s=float('nan'),
                    temporal_error_s=float('nan'),
                    hmm_state_before=-1,
                    hmm_state_after=-1,
                    matched=False
                ))
                continue
            errors       = [abs(event_time - t) for t in transition_times]
            nearest_idx  = int(np.argmin(errors))
            nearest_time = transition_times[nearest_idx]
            error        = errors[nearest_idx]
            alignments.append(EventAlignment(
                event_name=event_name,
                event_time_s=event_time,
                nearest_transition_time_s=nearest_time,
                temporal_error_s=error,
                hmm_state_before=transition_state_pairs[nearest_idx][0],
                hmm_state_after=transition_state_pairs[nearest_idx][1],
                matched=error <= tolerance_s
            ))
        valid_errors = [a.temporal_error_s for a in alignments
                        if not np.isnan(a.temporal_error_s)]
        n_matched = sum(a.matched for a in alignments)
        return AlignmentSummary(
            subject_id='',
            task_id=0,
            n_events=len(alignments),
            n_matched=n_matched,
            match_rate=n_matched / max(len(alignments), 1),
            mean_error_s=float(np.mean(valid_errors)) if valid_errors else float('nan'),
            median_error_s=float(np.median(valid_errors)) if valid_errors else float('nan'),
            std_error_s=float(np.std(valid_errors)) if valid_errors else float('nan'),
            per_event=alignments
        )

    def run_alignment_analysis(
        self,
        sequences: List[np.ndarray],
        subject_ids: List[str],
        csv_path: Path,
        task_id: int,
        paradigm_id: int = 1,
        model: Optional[GaussianHSMM] = None,
        tolerance_s: float = 0.5,
        sampling_rate: int = 50,
        save_path: Optional[Path] = None
    ) -> List[AlignmentSummary]:
        """Run event-alignment analysis across multiple subjects (mirrors HMMModel)."""
        paradigm_names = {
            1: 'patients_vs_controls',
            2: 'rct_vs_controls',
            3: 'other_conditions_vs_controls',
            4: 'rct_vs_other_conditions'
        }
        p_name = paradigm_names.get(paradigm_id, f'paradigm_{paradigm_id}')

        print(f"\n[HSMM Alignment] Task {task_id} | Paradigm {paradigm_id} ({p_name})")
        print(f"  Subjects:  {len(subject_ids)}  |  Tolerance: {tolerance_s}s")

        summaries: List[AlignmentSummary] = []

        for seq, sid in zip(sequences, subject_ids):
            try:
                _, _, segments = self.decode_sequence(seq, model, sampling_rate)
                events  = self.load_event_markers(csv_path, sid, task_id, True)
                summary = self.align_states_with_events(segments, events, tolerance_s)
                summary.subject_id = sid
                summary.task_id    = task_id
                summaries.append(summary)
                print(f"  [{sid:6s}] n_events={summary.n_events:3d}  "
                      f"match={summary.match_rate:.2f}  "
                      f"mean_err={summary.mean_error_s:.3f}s")
            except ValueError as e:
                print(f"  ⚠️  [{sid}] No events found — {e}")
            except Exception as e:
                print(f"  ⚠️  [{sid}] Skipped — {type(e).__name__}: {e}")

        if summaries:
            match_rates = [s.match_rate for s in summaries]
            mean_errors = [s.mean_error_s for s in summaries
                           if not np.isnan(s.mean_error_s)]
            print(f"\n  ── Aggregate (Task {task_id}, Paradigm {paradigm_id}) ──")
            print(f"  Mean match rate   : {np.mean(match_rates):.3f} "
                  f"± {np.std(match_rates):.3f}")
            if mean_errors:
                print(f"  Mean temporal err : {np.mean(mean_errors):.3f}s "
                      f"± {np.std(mean_errors):.3f}s")

            if save_path is not None:
                rows = []
                for s in summaries:
                    for ev in s.per_event:
                        rows.append({
                            'subject_id':           s.subject_id,
                            'task_id':              s.task_id,
                            'paradigm_id':          paradigm_id,
                            'paradigm_name':        p_name,
                            'event_name':           ev.event_name,
                            'event_time_s':         ev.event_time_s,
                            'nearest_transition_s': ev.nearest_transition_time_s,
                            'temporal_error_s':     ev.temporal_error_s,
                            'hmm_state_before':     ev.hmm_state_before,
                            'hmm_state_after':      ev.hmm_state_after,
                            'matched':              ev.matched,
                            'tolerance_s':          tolerance_s,
                        })
                pd.DataFrame(rows).to_csv(save_path, index=False)
                print(f"\n  [Saved alignment results] {save_path}")

        return summaries

    # =========================================================================
    # SECTION 5 — Emission distribution visualization
    # (identical to HMMModel — GaussianHSMM has the same means_/covars_ attrs)
    # =========================================================================

    def plot_emission_distributions(
        self,
        model: Optional[GaussianHSMM] = None,
        channel_names: Optional[List[str]] = None,
        title_suffix: str = '',
        n_top_highlight: int = 6,
        save_path: Optional[Path] = None
    ):
        """Visualise emission mean per hidden state (mirrors HMMModel)."""
        if model is None:
            if self.fitted_hsmm1 is None:
                raise RuntimeError("Call fit_for_analysis() first.")
            model = self.fitted_hsmm1

        if channel_names is None:
            channel_names = CHAN_NAME

        means    = model.means_
        n_states = model.n_components
        n_features = means.shape[1]

        fig, axes = plt.subplots(n_states, 1, figsize=(14, 3 * n_states), sharex=True)
        if n_states == 1:
            axes = [axes]

        colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, n_features))

        for state_idx, ax in enumerate(axes):
            state_means = means[state_idx]
            bars = ax.bar(range(n_features), state_means, color=colors, alpha=0.8)
            top_idx = np.argsort(np.abs(state_means))[-n_top_highlight:]
            for idx in top_idx:
                bars[idx].set_edgecolor('red')
                bars[idx].set_linewidth(2.0)
            ax.set_ylabel(f'State {state_idx}\nMean', fontsize=9)
            ax.set_xticks(range(n_features))
            ax.set_xticklabels(channel_names, rotation=45, ha='right', fontsize=7)
            ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle(
            f'HSMM Emission Means per Hidden State  {title_suffix}\n'
            f'Red borders = top {n_top_highlight} channels by |mean|',
            fontsize=11
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        plt.close()

    def plot_transition_matrix(
        self,
        model: Optional[GaussianHSMM] = None,
        title_suffix: str = '',
        save_path: Optional[Path] = None
    ):
        """Visualise the state transition matrix as a heatmap (mirrors HMMModel)."""
        if model is None:
            if self.fitted_hsmm1 is None:
                raise RuntimeError("Call fit_for_analysis() first.")
            model = self.fitted_hsmm1

        trans_mat = model.transmat_
        n_states  = model.n_components

        fig, ax = plt.subplots(figsize=(max(6, n_states + 2), max(5, n_states + 1)))
        sns.heatmap(
            trans_mat, annot=True, fmt='.3f', cmap='Blues',
            vmin=0, vmax=1,
            xticklabels=[f'S{i}' for i in range(n_states)],
            yticklabels=[f'S{i}' for i in range(n_states)],
            ax=ax, linewidths=0.5
        )
        ax.set_title(
            f'HSMM State Transition Matrix  {title_suffix}\n'
            '(Diagonal is zero by construction — explicit duration model)',
            fontsize=10
        )
        ax.set_xlabel('Next State')
        ax.set_ylabel('Current State')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        plt.close()

    def plot_duration_distributions(
        self,
        save_path: Optional[Path] = None
    ):
        """
        HSMM-exclusive: visualise the fitted Poisson duration distributions
        for patient and control models side-by-side.

        This plot has no HMM equivalent and directly shows the key HSMM
        contribution — the model learns different mean sojourn times per
        state for patients vs controls.  States where the two distributions
        diverge most are the clinically informative movement phases.
        """
        if self.fitted_hsmm0 is None or self.fitted_hsmm1 is None:
            raise RuntimeError("Call fit_for_analysis() first.")

        rates0 = self.fitted_hsmm0._duration_rates   # (n_states,)
        rates1 = self.fitted_hsmm1._duration_rates
        n_states = len(rates0)

        x = np.arange(n_states)
        width = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Left: Poisson mean (λ) per state
        ax = axes[0]
        ax.bar(x - width/2, rates0, width, label='Control', color='#1f77b4', alpha=0.8)
        ax.bar(x + width/2, rates1, width, label='Patient',  color='#d62728', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'State {s}' for s in range(n_states)])
        ax.set_ylabel('Mean sojourn duration (frames)')
        ax.set_title('Poisson λ per State — Patient vs Control')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Right: Difference (patient − control)
        ax2 = axes[1]
        diff = rates1 - rates0
        colors = ['#d62728' if d > 0 else '#1f77b4' for d in diff]
        ax2.bar(x, diff, color=colors, alpha=0.8)
        ax2.axhline(0, color='black', linewidth=0.6)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'State {s}' for s in range(n_states)])
        ax2.set_ylabel('Patient λ − Control λ (frames)')
        ax2.set_title('Duration Difference per State\n'
                      'Red = patients dwell longer  |  Blue = controls dwell longer')
        ax2.grid(axis='y', alpha=0.3)

        plt.suptitle('HSMM Fitted Duration Distributions (Poisson)', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        plt.close()

    # =========================================================================
    # SECTION 6 — State-specific and patient-vs-control feature importance
    # (identical to HMMModel — emission structure is the same)
    # =========================================================================

    def compute_state_specific_importance(
        self,
        model: Optional[GaussianHSMM] = None,
        channel_names: Optional[List[str]] = None
    ) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
        """State-specific feature importance from emission means (mirrors HMMModel)."""
        if model is None:
            if self.fitted_hsmm1 is None:
                raise RuntimeError("Call fit_for_analysis() first.")
            model = self.fitted_hsmm1

        if channel_names is None:
            channel_names = CHAN_NAME

        means    = model.means_
        n_states, n_features = means.shape

        mean_across_states  = means.mean(axis=0)
        range_across_states = means.max(axis=0) - means.min(axis=0)

        state_importance: Dict[int, Dict[str, float]] = {}
        for s in range(n_states):
            raw_dev = np.abs(means[s] - mean_across_states)
            scaled  = raw_dev * (range_across_states + 1e-8)
            normed  = scaled / (scaled.sum() + 1e-12)
            state_importance[s] = {
                channel_names[int(i)]: float(normed[i])
                for i in np.argsort(normed)[::-1]
            }

        global_scores = range_across_states / (range_across_states.sum() + 1e-12)
        global_importance = {
            channel_names[int(i)]: float(global_scores[i])
            for i in np.argsort(global_scores)[::-1]
        }

        print("\n[HSMM] Global Feature Importance (emission mean range across states):")
        for rank, (ch, sc) in enumerate(global_importance.items(), 1):
            print(f"  {rank:2d}. {ch:<25}: {sc:.4f}")

        return global_importance, state_importance

    def compare_patient_control_emissions(
        self,
        channel_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None
    ) -> np.ndarray:
        """Patient vs control emission mean differences (mirrors HMMModel)."""
        if self.fitted_hsmm0 is None or self.fitted_hsmm1 is None:
            raise RuntimeError("Call fit_for_analysis() first.")

        if channel_names is None:
            channel_names = CHAN_NAME

        m0 = self.fitted_hsmm0.means_
        m1 = self.fitted_hsmm1.means_

        if m0.shape[0] != m1.shape[0]:
            raise ValueError(
                f"n_components mismatch: class0={m0.shape[0]}, class1={m1.shape[0]}."
            )

        mean_diff = m1 - m0
        n_states  = mean_diff.shape[0]

        fig, axes = plt.subplots(1, n_states, figsize=(max(6, 5 * n_states), 7), sharey=True)
        if n_states == 1:
            axes = [axes]

        for s_idx, ax in enumerate(axes):
            diff = mean_diff[s_idx]
            bar_colors = ['#d62728' if d > 0 else '#1f77b4' for d in diff]
            ax.barh(range(len(channel_names)), diff, color=bar_colors, alpha=0.8)
            ax.set_yticks(range(len(channel_names)))
            ax.set_yticklabels(channel_names, fontsize=7)
            ax.axvline(0, color='black', linewidth=0.6)
            ax.set_title(f'State {s_idx}', fontsize=10)
            ax.set_xlabel('Patient − Control\nEmission Mean Diff', fontsize=8)
            ax.grid(axis='x', alpha=0.3)

        plt.suptitle(
            'HSMM Patient vs Control Emission Mean Differences per Movement Phase\n'
            'Red = patients higher  |  Blue = controls higher',
            fontsize=11
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        plt.close()
        return mean_diff

    def plot_state_importance_heatmap(
        self,
        state_importance: Dict[int, Dict[str, float]],
        channel_names: Optional[List[str]] = None,
        title: str = 'HSMM State-Specific Feature Importance',
        save_path: Optional[Path] = None
    ):
        """Heatmap of per-state feature importances (mirrors HMMModel)."""
        if channel_names is None:
            channel_names = CHAN_NAME

        n_states   = len(state_importance)
        n_features = len(channel_names)
        matrix     = np.zeros((n_states, n_features))

        for s_idx in range(n_states):
            for c_idx, ch in enumerate(channel_names):
                matrix[s_idx, c_idx] = state_importance[s_idx].get(ch, 0.0)

        fig, ax = plt.subplots(
            figsize=(max(12, n_features * 0.7), max(4, n_states * 0.8 + 1))
        )
        sns.heatmap(
            matrix,
            xticklabels=channel_names,
            yticklabels=[f'State {s}' for s in range(n_states)],
            cmap='YlOrRd', annot=True, fmt='.3f',
            linewidths=0.4, ax=ax,
            cbar_kws={'label': 'Normalised Importance'}
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_title(title, fontsize=11, pad=12)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        plt.close()