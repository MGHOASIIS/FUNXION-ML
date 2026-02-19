"""
Hidden Markov Model for shoulder pathology classification.

Implements class-conditional Gaussian HMMs with:
- Generative scoring (log-likelihood delta → sigmoid → probability)
- Permutation-based feature importance (channel-level)
- Subject-level LOO CV (consistent with RNN/CNN)
- Per-fold diagnostic tracking
- Checkpoint saving (JSON — no PyTorch state dict)

Phase 2 Analysis Extensions:
- Hidden state decoding and temporal segmentation
- CSV event marker alignment (does HMM rediscover annotated phases?)
  CSV schema: timestamp, event, progress, hand_used, subject_id
  One consolidated CSV per task (task1–task6).
  Timestamps are absolute wall-clock times — converted to relative
  task time by subtracting each subject's first event timestamp.
  Missing progress / hand_used columns handled gracefully.
- Emission distribution visualization per state
- State-specific feature importance (per movement phase)
- Patient vs control emission difference analysis
- BIC/AIC-based optimal n_components selection
- Paradigm-aware analysis: hidden states differ across paradigms
  because each paradigm presents the HMM with a different contrast
  (all patients vs controls / RCT vs controls / RCT vs other conditions)
"""
from typing import Dict, List, Optional, Tuple
import json
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

from models.base_model import BaseModel, ModelResults
from config.constants import CHAN_NAME
from utils.metrics import compute_metrics


# =============================================================================
# Dataclasses for structured analysis outputs
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
    """Alignment between one annotated event and the nearest HMM transition."""
    event_name: str
    event_time_s: float
    nearest_transition_time_s: float
    temporal_error_s: float
    hmm_state_before: int
    hmm_state_after: int
    matched: bool                  # True if error <= tolerance


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
# Main HMMModel class — classification pipeline unchanged
# =============================================================================

class HMMModel(BaseModel):
    """
    Hidden Markov Model wrapper with subject-level LOO CV.

    Classification pipeline (train_and_evaluate / fit) is identical to
    the original implementation.  All new methods are additive and can be
    called independently after training to perform Phase-2 analysis.
    """

    def __init__(self, checkpoints_dir=None, task=None, paradigm=None):
        """
        Parameters
        ----------
        checkpoints_dir : Path or None
            Directory to save best-model JSON checkpoint
        task : str or None
            Task name — stored for downstream tracking
        paradigm : int or None
            Classification paradigm index — stored for downstream tracking

        Note: patience and min_delta are not used by the HMM (no iterative
        training loop), but are accepted by BaseModel and left as None.
        """
        super().__init__(
            model_name="HMM",
            checkpoints_dir=checkpoints_dir,
            patience=None,
            min_delta=None,
            task=task,
            paradigm=paradigm
        )

        # Stored after fit_for_analysis() — used by all analysis methods
        self.fitted_hmm0: Optional[GaussianHMM] = None   # control model
        self.fitted_hmm1: Optional[GaussianHMM] = None   # patient/condition model

    # =========================================================================
    # SECTION 1 — Classification pipeline (UNCHANGED from original)
    # =========================================================================

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        param_grid: Optional[Dict] = None
    ) -> ModelResults:
        """
        Train HMM with hyperparameter search using LOO CV.

        Parameters
        ----------
        X : np.ndarray
            Array of sequences, shape (N,) where each element is (T_i, C)
        y : np.ndarray
            Labels, shape (N,)
        subject_ids : np.ndarray, optional
            Subject identifiers for subject-level CV splits.
            If None, falls back to sample-level LOO (one sequence = one fold).
        param_grid : Dict, optional
            Hyperparameter grid for search

        Returns
        -------
        ModelResults
            Complete results including metrics, per-fold diagnostics, predictions
        """
        if param_grid is None:
            from config.hyperparameter import HMM_PARAM_GRID
            param_grid = HMM_PARAM_GRID

        # Build CV splits — subject-level if IDs provided, else sample-level
        if subject_ids is not None:
            unique_subjects = np.unique(subject_ids)
            loo = LeaveOneOut()
            cv_splits = list(loo.split(unique_subjects))
            print(f"\n[HMM] Subject-level LOO CV: {len(unique_subjects)} subjects")
        else:
            unique_subjects = None
            loo = LeaveOneOut()
            cv_splits = list(loo.split(range(len(X))))
            print(f"\n[HMM] Sample-level LOO CV: {len(X)} samples")

        grid = list(ParameterGrid(param_grid))
        print(f"[HMM] Evaluating {len(grid)} hyperparameter combinations...")

        # Parallel search — hmmlearn is CPU-only and process-safe
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=-1, verbose=10)(
                delayed(self._loo_score)(
                    params, X, y, cv_splits, subject_ids, unique_subjects
                )
                for params in grid
            )

        # Select best configuration
        best_result = max(results, key=lambda t: t[0])
        best_score, best_params, y_true, y_pred, y_proba, per_fold_results = best_result

        print(f"\n[HMM] Best params: {best_params}")
        print(f"[HMM] Best balanced accuracy: {best_score:.4f}")

        # Compute permutation feature importance using best params
        feature_imp = self.compute_feature_importance(
            X=X,
            y=y,
            best_params=best_params,
            cv_splits=cv_splits,
            subject_ids=subject_ids,
            unique_subjects=unique_subjects
        )

        # Compute aggregate metrics
        metrics = compute_metrics(y_true, y_pred, y_proba)

        # ── Save best model checkpoint ────────────────────────────────────────
        # Always save — if checkpoint_dir was not explicitly provided (i.e.
        # --save-checkpoints flag was not passed), fall back to a 'checkpoints'
        # subdirectory next to wherever the module lives so the file is never lost.
        from datetime import datetime

        if isinstance(self.checkpoint_dir, Path):
            save_dir = self.checkpoint_dir
        else:
            save_dir = Path(__file__).resolve().parent / "checkpoints"

        save_dir.mkdir(parents=True, exist_ok=True)

        task_tag    = f"T{self.task}"    if self.task     is not None else "Tunk"
        paradigm_tag = f"P{self.paradigm}" if self.paradigm is not None else "Punk"
        timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        best_path = save_dir / (
            f"HMM_{task_tag}_{paradigm_tag}_"
            f"BA{best_score:.4f}_{timestamp_tag}.json"
        )

        with open(best_path, "w") as f:
            json.dump({
                'model_name':       'HMM',
                'task':             self.task,
                'paradigm':         self.paradigm,
                'hyperparameters':  best_params,
                'metrics':          {'balanced_accuracy': best_score, **metrics},
                'feature_importance': feature_imp,
                'input_shape':      [len(X), int(X[0].shape[1])],
                'predictions': {
                    'y_true':  y_true.tolist(),
                    'y_pred':  y_pred.tolist(),
                    'y_proba': y_proba.tolist()
                },
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        print(f"\n[HMM] Best model saved → {best_path}")

        return ModelResults(
            metrics=metrics,
            best_params=best_params,
            feature_importance=feature_imp,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            X_shape=X.shape,
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

        Parameters
        ----------
        params : Dict
            HMM hyperparameters (n_components, covariance_type, n_iter)
        sequences : np.ndarray
            Array of (T_i, C) sequences
        y : np.ndarray
            Labels
        cv_splits : list
            Precomputed LOO split indices
        subject_ids : np.ndarray or None
            Subject identifiers
        unique_subjects : np.ndarray or None
            Unique subject IDs for subject-level splits

        Returns
        -------
        tuple
            (balanced_accuracy, params, y_true, y_pred, y_proba, per_fold_results)
        """
        y_true, y_pred, y_proba = [], [], []
        per_fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):

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

            seqs_train = [sequences[i] for i in train_sample_idx]
            y_train = y[train_sample_idx]

            seqs_test = [sequences[i] for i in test_sample_idx]
            y_test_list = y[test_sample_idx].tolist()

            # Train class-conditional HMMs
            hmm0 = self._fit_hmm(
                [s for s, lab in zip(seqs_train, y_train) if lab == 0],
                **params
            )
            hmm1 = self._fit_hmm(
                [s for s, lab in zip(seqs_train, y_train) if lab == 1],
                **params
            )

            # Score each test sequence
            fold_preds, fold_proba = [], []
            for seq in seqs_test:
                delta = hmm1.score(seq) - hmm0.score(seq)
                prob = self._stable_sigmoid(delta)
                pred = int(prob >= 0.5)
                fold_preds.append(pred)
                fold_proba.append(prob)

            y_true.extend(y_test_list)
            y_pred.extend(fold_preds)
            y_proba.extend(fold_proba)

            fold_ba = balanced_accuracy_score(y_test_list, fold_preds)

            per_fold_results.append({
                'fold': fold_idx,
                'test_subjects': test_subjects.tolist() if subject_ids is not None else [fold_idx],
                'train_loss': None,
                'val_loss': None,
                'train_acc': None,
                'val_acc': float(fold_ba),
                'y_true': y_test_list,
                'y_pred': fold_preds,
                'y_proba': fold_proba,
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

    def _fit_hmm(
        self,
        seq_list: list,
        n_components: int,
        covariance_type: str,
        n_iter: int
    ) -> GaussianHMM:
        """
        Fit a Gaussian HMM to a list of sequences from one class.

        Parameters
        ----------
        seq_list : list of np.ndarray
            Sequences of shape (T_i, C) for one class
        n_components : int
            Number of hidden states
        covariance_type : str
            Covariance structure ('diag', 'full', 'tied', 'spherical')
        n_iter : int
            Maximum EM iterations

        Returns
        -------
        GaussianHMM
            Fitted model
        """
        lengths = [s.shape[0] for s in seq_list]
        X_stacked = np.vstack(seq_list)

        model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=42,
            verbose=False
        ).fit(X_stacked, lengths=lengths)

        return model

    @staticmethod
    def _stable_sigmoid(delta: float) -> float:
        """
        Numerically stable sigmoid of log-likelihood delta.

        Parameters
        ----------
        delta : float
            log P(seq | HMM1) - log P(seq | HMM0)

        Returns
        -------
        float
            Probability of class 1
        """
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
        Compute permutation-based feature importance.

        Permutes each channel independently across all sequences and measures
        the drop in balanced accuracy relative to the baseline. A larger drop
        means the channel is more discriminative.

        Parameters
        ----------
        X : np.ndarray
            Array of (T_i, C) sequences
        y : np.ndarray
            Labels
        **kwargs
            Must include:
            - best_params (Dict): HMM hyperparameters to use
            - cv_splits (list): Precomputed LOO splits
            - subject_ids (np.ndarray or None)
            - unique_subjects (np.ndarray or None)

        Returns
        -------
        Dict[str, float]
            Channel names mapped to importance scores, sorted descending
        """
        best_params = kwargs["best_params"]
        cv_splits = kwargs["cv_splits"]
        subject_ids = kwargs.get("subject_ids", None)
        unique_subjects = kwargs.get("unique_subjects", None)

        print("\n[HMM] Computing feature importance via permutation...")

        rng = np.random.default_rng(42)
        n_channels = X[0].shape[1]
        importance = np.zeros(n_channels)

        # Baseline score
        baseline_ba, *_ = self._loo_score(
            best_params, X, y, cv_splits, subject_ids, unique_subjects
        )

        # Permute each channel and measure accuracy drop
        for d in range(n_channels):
            print(f"  Channel {d+1}/{n_channels}: {CHAN_NAME[d]}")
            seqs_perm = self._permute_channel(X, d, rng)
            ba_d, *_ = self._loo_score(
                best_params, seqs_perm, y, cv_splits, subject_ids, unique_subjects
            )
            importance[d] = baseline_ba - ba_d

        denom = importance.sum()
        if denom > 1e-12:
            importance = importance / denom

        feature_imp = {
            CHAN_NAME[i]: float(importance[i])
            for i in np.argsort(importance)[::-1]
        }

        print("\n[HMM] Channel Importance:")
        for i, (feat, imp) in enumerate(list(feature_imp.items())):
            print(f"  {i+1}. {feat}: {imp:.4f}")

        return feature_imp

    def _create_temp_model(
        self,
        n_components: Optional[int] = None,
        covariance_type: Optional[str] = None,
        n_iter: Optional[int] = None
    ) -> Optional[GaussianHMM]:
        """
        Instantiate a GaussianHMM from best_params for post-hoc inspection.

        The returned model is unfitted (architecture only, no trained weights).

        Parameters
        ----------
        n_components : int, optional
            Override number of hidden states (uses best_params if None)
        covariance_type : str, optional
            Override covariance type (uses best_params if None)
        n_iter : int, optional
            Override EM iterations (uses best_params if None)

        Returns
        -------
        GaussianHMM or None
        """
        if not hasattr(self, 'best_params') or self.best_params is None:
            return None

        try:
            model = GaussianHMM(
                n_components=n_components or self.best_params["n_components"],
                covariance_type=covariance_type or self.best_params["covariance_type"],
                n_iter=n_iter or self.best_params["n_iter"],
                random_state=42,
                verbose=False
            )
            return model

        except Exception as e:
            print(f"⚠️  Could not create temp HMM model: {e}")
            return None

    @staticmethod
    def _permute_channel(seqs: list, channel: int, rng) -> list:
        """
        Permute a single channel across all sequences.

        Parameters
        ----------
        seqs : list of np.ndarray
            Sequences of shape (T_i, C)
        channel : int
            Channel index to permute
        rng : np.random.Generator

        Returns
        -------
        list of np.ndarray
            Sequences with the specified channel permuted
        """
        seqs_perm = [s.copy() for s in seqs]

        col = np.concatenate([s[:, channel] for s in seqs_perm])
        rng.shuffle(col)

        start = 0
        for s in seqs_perm:
            L = len(s)
            s[:, channel] = col[start:start + L]
            start += L

        return seqs_perm

    # =========================================================================
    # SECTION 2 — Phase 2 Analysis: fit on full data for interpretability
    # =========================================================================

    def fit_for_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_components: int,
        covariance_type: str = 'diag',
        n_iter: int = 100
    ) -> Tuple[GaussianHMM, GaussianHMM]:
        """
        Train class-conditional HMMs on the FULL dataset (no CV).

        This is used exclusively for Phase-2 interpretability analysis:
        state decoding, event alignment, emission visualization, and
        state-specific feature importance.  It is NOT used for classification
        performance — use train_and_evaluate() for that.

        The fitted models are stored as self.fitted_hmm0 (controls / class 0)
        and self.fitted_hmm1 (patients / class 1) for downstream methods.

        Parameters
        ----------
        X : np.ndarray
            Array of (T_i, C) sequences, one per subject
        y : np.ndarray
            Labels (0 = control/class-0, 1 = patient/class-1)
        n_components : int
            Number of hidden states — use select_optimal_n_components() first
        covariance_type : str
            'diag' recommended for N=60 to avoid ill-conditioned covariances
        n_iter : int
            Maximum EM iterations

        Returns
        -------
        hmm0, hmm1 : Tuple[GaussianHMM, GaussianHMM]
            Fitted class-conditional models
        """
        seqs_0 = [X[i] for i in range(len(X)) if y[i] == 0]
        seqs_1 = [X[i] for i in range(len(X)) if y[i] == 1]

        print(f"\n[HMM Analysis] Fitting on full data")
        print(f"  n_components={n_components}, covariance_type={covariance_type}")
        print(f"  Class 0 (controls): {len(seqs_0)} sequences")
        print(f"  Class 1 (patients): {len(seqs_1)} sequences")

        self.fitted_hmm0 = self._fit_hmm(seqs_0, n_components, covariance_type, n_iter)
        self.fitted_hmm1 = self._fit_hmm(seqs_1, n_components, covariance_type, n_iter)

        print("  ✓ Both class-conditional HMMs fitted successfully")
        return self.fitted_hmm0, self.fitted_hmm1

    # =========================================================================
    # SECTION 3 — Model selection: BIC / AIC
    # =========================================================================

    def select_optimal_n_components(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_range: range = range(2, 8),
        covariance_type: str = 'diag',
        n_iter: int = 100,
        save_path: Optional[Path] = None
    ) -> Dict[str, list]:
        """
        Select the optimal number of hidden states via BIC and AIC.

        Trains separate class-conditional HMMs for each n_components value
        and scores them independently.  The combined BIC/AIC across both
        classes is used for model selection.

        BIC penalises complexity more strongly and is recommended for N=60.
        Lower BIC/AIC = better model.

        Parameters
        ----------
        X : np.ndarray
            Array of (T_i, C) sequences
        y : np.ndarray
            Labels
        n_range : range
            Candidate numbers of hidden states to evaluate
        covariance_type : str
            Covariance type (keep consistent with classification run)
        n_iter : int
            EM iterations per model fit
        save_path : Path, optional
            If provided, saves the selection plot here

        Returns
        -------
        results : dict
            Keys: 'n_components', 'bic_class0', 'bic_class1', 'bic_total',
                  'aic_class0', 'aic_class1', 'aic_total', 'log_ll_class0',
                  'log_ll_class1', 'optimal_bic', 'optimal_aic'
        """
        seqs_0 = [X[i] for i in range(len(X)) if y[i] == 0]
        seqs_1 = [X[i] for i in range(len(X)) if y[i] == 1]
        n_features = X[0].shape[1]

        results: Dict[str, list] = {
            'n_components': [],
            'bic_class0': [], 'bic_class1': [], 'bic_total': [],
            'aic_class0': [], 'aic_class1': [], 'aic_total': [],
            'log_ll_class0': [], 'log_ll_class1': [],
        }

        print(f"\n[HMM] Model selection: evaluating n_components = {list(n_range)}")
        print(f"  covariance_type = {covariance_type}")

        for n in n_range:
            h0 = self._fit_hmm(seqs_0, n, covariance_type, n_iter)
            h1 = self._fit_hmm(seqs_1, n, covariance_type, n_iter)

            # Stack each class for scoring
            X0 = np.vstack(seqs_0)
            X1 = np.vstack(seqs_1)
            len0 = [s.shape[0] for s in seqs_0]
            len1 = [s.shape[0] for s in seqs_1]

            ll0 = h0.score(X0, lengths=len0)
            ll1 = h1.score(X1, lengths=len1)

            # Free parameters per class (diag covariance)
            # Transition matrix: n*(n-1), initial probs: n-1,
            # means: n*d, diag variances: n*d
            n_params = (n * (n - 1)) + (n - 1) + 2 * (n * n_features)

            n0 = X0.shape[0]
            n1 = X1.shape[0]

            bic0 = -2 * ll0 * n0 + n_params * np.log(n0)
            bic1 = -2 * ll1 * n1 + n_params * np.log(n1)
            aic0 = -2 * ll0 * n0 + 2 * n_params
            aic1 = -2 * ll1 * n1 + 2 * n_params

            results['n_components'].append(n)
            results['log_ll_class0'].append(ll0)
            results['log_ll_class1'].append(ll1)
            results['bic_class0'].append(bic0)
            results['bic_class1'].append(bic1)
            results['bic_total'].append(bic0 + bic1)
            results['aic_class0'].append(aic0)
            results['aic_class1'].append(aic1)
            results['aic_total'].append(aic0 + aic1)

            print(f"  n={n}: BIC_total={bic0+bic1:.1f}  AIC_total={aic0+aic1:.1f}")

        # Identify optima
        opt_bic = results['n_components'][int(np.argmin(results['bic_total']))]
        opt_aic = results['n_components'][int(np.argmin(results['aic_total']))]
        results['optimal_bic'] = opt_bic
        results['optimal_aic'] = opt_aic

        print(f"\n  ✓ Optimal n_components by BIC: {opt_bic}")
        print(f"  ✓ Optimal n_components by AIC: {opt_aic}")

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        ax.plot(results['n_components'], results['log_ll_class0'],
                'o-', label='Class 0 (controls)', linewidth=2)
        ax.plot(results['n_components'], results['log_ll_class1'],
                's-', label='Class 1 (patients)', linewidth=2)
        ax.set_xlabel('Number of Hidden States')
        ax.set_ylabel('Log-Likelihood (higher = better)')
        ax.set_title('Log-Likelihood per Class')
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(results['n_components'], results['bic_total'],
                'o-', color='crimson', linewidth=2, label='BIC total')
        ax.axvline(opt_bic, color='crimson', linestyle='--',
                   label=f'Optimal BIC: {opt_bic}')
        ax.set_xlabel('Number of Hidden States')
        ax.set_ylabel('BIC (lower = better)')
        ax.set_title('BIC — Model Selection')
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[2]
        ax.plot(results['n_components'], results['aic_total'],
                'o-', color='steelblue', linewidth=2, label='AIC total')
        ax.axvline(opt_aic, color='steelblue', linestyle='--',
                   label=f'Optimal AIC: {opt_aic}')
        ax.set_xlabel('Number of Hidden States')
        ax.set_ylabel('AIC (lower = better)')
        ax.set_title('AIC — Model Selection')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.suptitle('HMM Model Selection: Optimal Number of Hidden States',
                     fontsize=13)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  [Saved] {save_path}")
        plt.close()

        return results

    # =========================================================================
    # SECTION 4 — State decoding and temporal segmentation
    # =========================================================================

    def decode_sequence(
        self,
        sequence: np.ndarray,
        model: Optional[GaussianHMM] = None,
        sampling_rate: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, List[StateSegment]]:
        """
        Decode the most likely hidden-state sequence for a single recording.

        Uses the Viterbi algorithm (hard assignment) for segmentation and
        the forward-backward algorithm (predict_proba) for soft posteriors.

        Parameters
        ----------
        sequence : np.ndarray
            Single subject recording, shape (T, 18)
        model : GaussianHMM, optional
            Which fitted model to use for decoding.
            Defaults to self.fitted_hmm1 (patient model).
            Pass self.fitted_hmm0 to decode with the control model.
        sampling_rate : int
            Hz — used to convert frame indices to seconds

        Returns
        -------
        states : np.ndarray, shape (T,)
            Hard state assignment per timestep (Viterbi)
        posteriors : np.ndarray, shape (T, n_components)
            Soft posterior probabilities per state
        segments : List[StateSegment]
            Contiguous time segments, one entry per state run
        """
        if model is None:
            if self.fitted_hmm1 is None:
                raise RuntimeError(
                    "No fitted model available. "
                    "Call fit_for_analysis() first."
                )
            model = self.fitted_hmm1

        states = model.predict(sequence)           # Viterbi
        posteriors = model.predict_proba(sequence) # Forward-backward

        segments = self._states_to_segments(states, sampling_rate)

        return states, posteriors, segments

    @staticmethod
    def _states_to_segments(
        state_sequence: np.ndarray,
        sampling_rate: int = 50
    ) -> List[StateSegment]:
        """
        Convert a per-timestep state array into contiguous segments.

        Parameters
        ----------
        state_sequence : np.ndarray, shape (T,)
        sampling_rate : int

        Returns
        -------
        List[StateSegment]
        """
        segments = []
        current_state = state_sequence[0]
        start_frame = 0

        for frame in range(1, len(state_sequence)):
            if state_sequence[frame] != current_state:
                segments.append(StateSegment(
                    state=int(current_state),
                    start_frame=start_frame,
                    end_frame=frame - 1,
                    start_time=start_frame / sampling_rate,
                    end_time=(frame - 1) / sampling_rate,
                    duration_s=(frame - start_frame) / sampling_rate
                ))
                current_state = state_sequence[frame]
                start_frame = frame

        # Final segment
        T = len(state_sequence)
        segments.append(StateSegment(
            state=int(current_state),
            start_frame=start_frame,
            end_frame=T - 1,
            start_time=start_frame / sampling_rate,
            end_time=(T - 1) / sampling_rate,
            duration_s=(T - start_frame) / sampling_rate
        ))

        return segments

    # =========================================================================
    # SECTION 5 — CSV event marker loading and alignment
    # =========================================================================

    # CSV schema (one file per task):
    #   timestamp  : float  — absolute wall-clock time in seconds
    #   event      : str    — event label e.g. "Jar picked up", "Lid grabbed"
    #   progress   : float  — may be empty/NaN for controls — handled gracefully
    #   hand_used  : str    — may be empty for controls — handled gracefully
    #   subject_id : str    — e.g. "fx01" (control) or "PX01" (patient)
    #
    # Key design decision — relative timestamps:
    #   Raw timestamps are absolute wall-clock times and differ across subjects.
    #   The HMM sequences start at frame 0 = the beginning of the recording.
    #   We therefore subtract each subject's FIRST event timestamp so that
    #   time 0 in the events matches time 0 in the decoded state sequence.
    #   This relative offset is stored in EventAlignment.event_time_s.

    @staticmethod
    def load_events_csv(csv_path: Path) -> pd.DataFrame:
        """
        Load a consolidated task-events CSV into a clean DataFrame.

        Handles missing progress / hand_used columns gracefully and
        normalises subject_id to upper-case for consistent matching.

        Parameters
        ----------
        csv_path : Path
            Path to consolidated_task{N}.csv

        Returns
        -------
        pd.DataFrame
            Columns: timestamp (float), event (str), progress (float, nullable),
                     hand_used (str, nullable), subject_id (str)
        """
        df = pd.read_csv(csv_path)

        # Normalise column names — strip whitespace, lower-case
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

        # Ensure required columns exist
        required = {'timestamp', 'event', 'subject_id'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}\n"
                f"Found columns: {list(df.columns)}"
            )

        # Fill optional columns with None if absent
        for optional_col in ('progress', 'hand_used'):
            if optional_col not in df.columns:
                df[optional_col] = None

        # Clean types
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['subject_id'] = df['subject_id'].astype(str).str.strip()
        df['event'] = df['event'].astype(str).str.strip()

        # Drop rows with unparseable timestamps
        n_before = len(df)
        df = df.dropna(subset=['timestamp']).reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"  ⚠️  Dropped {n_dropped} rows with unparseable timestamps")

        return df

    def load_event_markers(
        self,
        csv_path: Path,
        subject_id: str,
        task_id: int,
        relative_timestamps: bool = True
    ) -> List[Dict]:
        """
        Load annotated event markers for one subject from a consolidated CSV.

        CSV schema expected (one file per task):
            timestamp, event, progress, hand_used, subject_id

        The progress and hand_used columns may be empty/NaN — this is handled
        gracefully and the information is preserved when present.

        Parameters
        ----------
        csv_path : Path
            Path to consolidated_task{task_id}.csv
        subject_id : str
            Subject identifier (e.g. 'fx01', 'PX01').
            Matching is case-insensitive.
        task_id : int
            Task number — used only for logging; the CSV already contains
            only one task's data.
        relative_timestamps : bool
            If True (default), subtract the subject's first-event timestamp
            so time 0 aligns with the start of the HMM sequence.
            If False, return raw wall-clock timestamps as-is.

        Returns
        -------
        events : List[Dict]
            Sorted by timestamp. Each dict has:
              - 'event_name'   : str
              - 'timestamp'    : float  (seconds, relative if relative_timestamps=True)
              - 'hand_used'    : str or None
              - 'progress'     : float or None
        """
        df = self.load_events_csv(csv_path)

        # Case-insensitive subject match
        mask = df['subject_id'].str.upper() == subject_id.upper()
        subject_df = df[mask].copy()

        if subject_df.empty:
            available = sorted(df['subject_id'].unique().tolist())
            raise ValueError(
                f"Subject '{subject_id}' not found in {csv_path.name}.\n"
                f"Available subjects: {available}"
            )

        # Sort by timestamp
        subject_df = subject_df.sort_values('timestamp').reset_index(drop=True)

        # Make timestamps relative to task start (first event for this subject)
        if relative_timestamps:
            t0 = subject_df['timestamp'].iloc[0]
            subject_df['timestamp'] = subject_df['timestamp'] - t0

        # Build list of event dicts
        events = []
        for _, row in subject_df.iterrows():
            # hand_used and progress may be NaN — convert to None
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
        """
        Compare HMM-discovered state transitions against annotated events.

        For each annotated event boundary, this finds the nearest HMM state
        transition and records the temporal error.  A small mean error and
        high match rate indicate that the HMM is independently rediscovering
        the same movement phase boundaries that human annotators marked.

        Parameters
        ----------
        segments : List[StateSegment]
            From decode_sequence()
        events : List[Dict]
            From load_event_markers() — each must have 'event_name' and
            'timestamp' (in seconds)
        tolerance_s : float
            Maximum temporal error (seconds) to count as 'matched'.
            Default 0.5 s = 25 frames at 50 Hz.

        Returns
        -------
        AlignmentSummary
        """
        # Extract transition times from segments (boundary between consecutive segments)
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
                # Edge case: only one state, no transitions
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

            errors = [abs(event_time - t) for t in transition_times]
            nearest_idx = int(np.argmin(errors))
            nearest_time = transition_times[nearest_idx]
            error = errors[nearest_idx]

            alignments.append(EventAlignment(
                event_name=event_name,
                event_time_s=event_time,
                nearest_transition_time_s=nearest_time,
                temporal_error_s=error,
                hmm_state_before=transition_state_pairs[nearest_idx][0],
                hmm_state_after=transition_state_pairs[nearest_idx][1],
                matched=error <= tolerance_s
            ))

        # Aggregate metrics — skip nan entries
        valid_errors = [a.temporal_error_s for a in alignments
                        if not np.isnan(a.temporal_error_s)]
        n_matched = sum(a.matched for a in alignments)

        summary = AlignmentSummary(
            subject_id='',          # caller should fill in
            task_id=0,              # caller should fill in
            n_events=len(alignments),
            n_matched=n_matched,
            match_rate=n_matched / max(len(alignments), 1),
            mean_error_s=float(np.mean(valid_errors)) if valid_errors else float('nan'),
            median_error_s=float(np.median(valid_errors)) if valid_errors else float('nan'),
            std_error_s=float(np.std(valid_errors)) if valid_errors else float('nan'),
            per_event=alignments
        )

        return summary

    def run_alignment_analysis(
        self,
        sequences: List[np.ndarray],
        subject_ids: List[str],
        csv_path: Path,
        task_id: int,
        paradigm_id: int = 1,
        model: Optional[GaussianHMM] = None,
        tolerance_s: float = 0.5,
        sampling_rate: int = 50,
        save_path: Optional[Path] = None
    ) -> List[AlignmentSummary]:
        """
        Run event-alignment analysis across multiple subjects for one task.

        Loops over subjects, decodes their state sequences using the
        fitted HMM, loads their event markers from the task CSV, and
        computes alignment quality (match rate, temporal error).

        Paradigm-awareness note
        -----------------------
        The fitted_hmm0 / fitted_hmm1 models stored on self were trained
        under a specific paradigm (e.g. paradigm 1 = patients vs controls).
        Pass the paradigm_id for logging so results can be compared across
        paradigms. The actual model used for decoding is selected via
        the 'model' parameter — pass self.fitted_hmm0 or self.fitted_hmm1
        explicitly to decode with the control or patient model respectively.

        Parameters
        ----------
        sequences : List[np.ndarray]
            One (T_i, 18) array per subject — must be z-score normalised
            with the same scaler used during classification training
        subject_ids : List[str]
            Subject identifiers matching subject_id column in the CSV
        csv_path : Path
            Path to consolidated_task{task_id}.csv
        task_id : int
            Task number (1-6) — used for logging and saved reports
        paradigm_id : int
            Paradigm index (1-4) — stored in summary for cross-paradigm
            comparison. Paradigm affects which sequences are included and
            therefore which HMMs were fitted.
              1 = patients vs controls
              2 = RCT vs controls
              3 = other conditions vs controls
              4 = RCT vs other conditions
        model : GaussianHMM, optional
            Model to decode with (defaults to self.fitted_hmm1)
        tolerance_s : float
            Temporal tolerance in seconds for a transition to count as
            'matched' to an annotated event. Default 0.5 s = 25 frames
            at 50 Hz.
        sampling_rate : int
        save_path : Path, optional
            If provided, saves a per-subject alignment summary CSV here

        Returns
        -------
        List[AlignmentSummary]
            One AlignmentSummary per subject. Subjects for whom events
            cannot be found in the CSV are skipped with a warning.
        """
        paradigm_names = {
            1: 'patients_vs_controls',
            2: 'rct_vs_controls',
            3: 'other_conditions_vs_controls',
            4: 'rct_vs_other_conditions'
        }
        p_name = paradigm_names.get(paradigm_id, f'paradigm_{paradigm_id}')

        print(f"\n[HMM Alignment] Task {task_id} | Paradigm {paradigm_id} ({p_name})")
        print(f"  CSV:         {csv_path}")
        print(f"  Subjects:    {len(subject_ids)}")
        print(f"  Tolerance:   {tolerance_s}s ({int(tolerance_s * sampling_rate)} frames)")

        summaries: List[AlignmentSummary] = []

        for seq, sid in zip(sequences, subject_ids):
            try:
                _, _, segments = self.decode_sequence(seq, model, sampling_rate)
                events = self.load_event_markers(
                    csv_path, sid, task_id,
                    relative_timestamps=True
                )
                summary = self.align_states_with_events(segments, events, tolerance_s)
                summary.subject_id = sid
                summary.task_id = task_id
                summaries.append(summary)

                print(f"  [{sid:6s}] n_events={summary.n_events:3d}  "
                      f"match={summary.match_rate:.2f}  "
                      f"mean_err={summary.mean_error_s:.3f}s")

            except ValueError as e:
                # Subject not found in CSV — skip silently with warning
                print(f"  ⚠️  [{sid}] No events found — {e}")
            except Exception as e:
                print(f"  ⚠️  [{sid}] Skipped — {type(e).__name__}: {e}")

        # ── Aggregate summary ─────────────────────────────────────────────────
        if summaries:
            match_rates  = [s.match_rate for s in summaries]
            mean_errors  = [s.mean_error_s for s in summaries
                            if not np.isnan(s.mean_error_s)]

            print(f"\n  ── Aggregate (Task {task_id}, Paradigm {paradigm_id}) ──")
            print(f"  Subjects analysed : {len(summaries)}")
            print(f"  Mean match rate   : {np.mean(match_rates):.3f} "
                  f"± {np.std(match_rates):.3f}")
            if mean_errors:
                print(f"  Mean temporal err : {np.mean(mean_errors):.3f}s "
                      f"± {np.std(mean_errors):.3f}s")

            # Per-event-type breakdown
            all_events: List[EventAlignment] = []
            for s in summaries:
                all_events.extend(s.per_event)

            event_types = sorted(set(e.event_name for e in all_events))
            print(f"\n  Per-event-type match rates:")
            for ev_type in event_types:
                ev_subset = [e for e in all_events if e.event_name == ev_type]
                ev_match  = np.mean([e.matched for e in ev_subset])
                ev_err    = np.mean([e.temporal_error_s for e in ev_subset
                                     if not np.isnan(e.temporal_error_s)])
                print(f"    {ev_type:30s}  "
                      f"match={ev_match:.2f}  mean_err={ev_err:.3f}s  "
                      f"(n={len(ev_subset)})")

            # ── Optional CSV export ───────────────────────────────────────────
            if save_path is not None:
                rows = []
                for s in summaries:
                    for ev in s.per_event:
                        rows.append({
                            'subject_id':              s.subject_id,
                            'task_id':                 s.task_id,
                            'paradigm_id':             paradigm_id,
                            'paradigm_name':           p_name,
                            'event_name':              ev.event_name,
                            'event_time_s':            ev.event_time_s,
                            'nearest_transition_s':    ev.nearest_transition_time_s,
                            'temporal_error_s':        ev.temporal_error_s,
                            'hmm_state_before':        ev.hmm_state_before,
                            'hmm_state_after':         ev.hmm_state_after,
                            'matched':                 ev.matched,
                            'tolerance_s':             tolerance_s,
                        })
                results_df = pd.DataFrame(rows)
                results_df.to_csv(save_path, index=False)
                print(f"\n  [Saved alignment results] {save_path}")

        return summaries


    # =========================================================================
    # SECTION 6 — Emission distribution visualization
    # =========================================================================

    def plot_emission_distributions(
        self,
        model: Optional[GaussianHMM] = None,
        channel_names: Optional[List[str]] = None,
        title_suffix: str = '',
        n_top_highlight: int = 6,
        save_path: Optional[Path] = None
    ):
        """
        Visualise the emission mean per hidden state as a bar chart.

        Each hidden state has a Gaussian emission distribution.  The mean
        encodes the average sensor reading while in that state.  Channels
        with large absolute means are the primary kinematic signatures of
        that movement phase.

        Red-bordered bars = top channels by absolute magnitude for that state.

        Parameters
        ----------
        model : GaussianHMM, optional
            Fitted model to inspect (defaults to self.fitted_hmm1)
        channel_names : List[str], optional
            18 channel names (defaults to CHAN_NAME from constants)
        title_suffix : str
            Appended to the figure title (e.g. 'Patient Model — Task 1')
        n_top_highlight : int
            How many top channels to highlight with red borders
        save_path : Path, optional
        """
        if model is None:
            if self.fitted_hmm1 is None:
                raise RuntimeError("Call fit_for_analysis() first.")
            model = self.fitted_hmm1

        if channel_names is None:
            channel_names = CHAN_NAME

        means = model.means_           # (n_states, n_features)
        n_states = model.n_components
        n_features = means.shape[1]

        fig, axes = plt.subplots(
            n_states, 1,
            figsize=(14, 3 * n_states),
            sharex=True
        )
        if n_states == 1:
            axes = [axes]

        colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, n_features))

        for state_idx, ax in enumerate(axes):
            state_means = means[state_idx]
            bars = ax.bar(range(n_features), state_means, color=colors, alpha=0.8)

            # Highlight top channels by absolute magnitude
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
            f'Emission Means per Hidden State  {title_suffix}\n'
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
        model: Optional[GaussianHMM] = None,
        title_suffix: str = '',
        save_path: Optional[Path] = None
    ):
        """
        Visualise the state transition probability matrix as a heatmap.

        High diagonal values indicate that states are self-sustaining
        (stable movement phases).  High off-diagonal values indicate
        frequent transitions (variable or fragmented movement).

        Parameters
        ----------
        model : GaussianHMM, optional
            Defaults to self.fitted_hmm1
        title_suffix : str
        save_path : Path, optional
        """
        if model is None:
            if self.fitted_hmm1 is None:
                raise RuntimeError("Call fit_for_analysis() first.")
            model = self.fitted_hmm1

        trans_mat = model.transmat_
        n_states = model.n_components

        fig, ax = plt.subplots(figsize=(max(6, n_states + 2),
                                        max(5, n_states + 1)))
        sns.heatmap(
            trans_mat,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            vmin=0,
            vmax=1,
            xticklabels=[f'S{i}' for i in range(n_states)],
            yticklabels=[f'S{i}' for i in range(n_states)],
            ax=ax,
            linewidths=0.5
        )
        ax.set_title(
            f'State Transition Matrix  {title_suffix}\n'
            '(High diagonal = stable phases | '
            'High off-diagonal = variable movement)',
            fontsize=10
        )
        ax.set_xlabel('Next State')
        ax.set_ylabel('Current State')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        plt.close()

    def plot_state_sequence_over_time(
        self,
        sequence: np.ndarray,
        state_sequence: np.ndarray,
        channel_idx: int = 0,
        events: Optional[List[Dict]] = None,
        sampling_rate: int = 50,
        title: str = 'State Sequence over Time',
        save_path: Optional[Path] = None
    ):
        """
        Plot HMM state assignments overlaid on a raw signal channel.

        The top panel shows the raw signal with coloured background regions
        for each HMM state.  Red dashed lines mark annotated event boundaries
        if provided.  The bottom panel shows the discrete state sequence.

        Parameters
        ----------
        sequence : np.ndarray, shape (T, 18)
            Raw (z-scored) signal for one subject
        state_sequence : np.ndarray, shape (T,)
            From decode_sequence()
        channel_idx : int
            Which of the 18 channels to display in the top panel
        events : List[Dict], optional
            Annotated events with 'event_name' and 'timestamp' (seconds)
        sampling_rate : int
        title : str
        save_path : Path, optional
        """
        T = len(state_sequence)
        time = np.arange(T) / sampling_rate
        unique_states = np.unique(state_sequence)
        n_states = len(unique_states)

        state_colors = {
            s: plt.cm.Set2(i / max(n_states - 1, 1))
            for i, s in enumerate(sorted(unique_states))
        }

        fig, (ax_sig, ax_state) = plt.subplots(
            2, 1, figsize=(14, 7), sharex=True,
            gridspec_kw={'height_ratios': [2, 1]}
        )

        # ── Top panel: raw signal + state background ──────────────────────────
        channel_label = CHAN_NAME[channel_idx] if channel_idx < len(CHAN_NAME) \
            else f'channel_{channel_idx}'
        ax_sig.plot(time, sequence[:, channel_idx], 'k-',
                    linewidth=0.9, label=channel_label)

        # Shade background by state
        prev_state = state_sequence[0]
        seg_start = 0.0
        for t in range(1, T):
            if state_sequence[t] != prev_state or t == T - 1:
                ax_sig.axvspan(
                    seg_start, time[t],
                    alpha=0.25,
                    color=state_colors[prev_state]
                )
                ax_state.axvspan(
                    seg_start, time[t],
                    alpha=0.35,
                    color=state_colors[prev_state]
                )
                prev_state = state_sequence[t]
                seg_start = time[t]

        # Annotated event markers
        if events:
            for event in events:
                t_ev = float(event['timestamp'])
                ax_sig.axvline(t_ev, color='red', linestyle='--',
                               linewidth=1.2, alpha=0.85)
                ax_sig.text(
                    t_ev, ax_sig.get_ylim()[1] * 0.88,
                    event['event_name'],
                    rotation=90, fontsize=6, color='red', va='top'
                )

        ax_sig.set_ylabel(f'Amplitude (z-score)\n[{channel_label}]', fontsize=9)
        ax_sig.set_title(title, fontsize=11)
        ax_sig.grid(alpha=0.3)

        # Legend for states
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=state_colors[s], alpha=0.6)
            for s in sorted(unique_states)
        ]
        ax_sig.legend(handles, [f'State {s}' for s in sorted(unique_states)],
                      loc='upper right', fontsize=8)

        # ── Bottom panel: discrete state sequence ─────────────────────────────
        ax_state.step(time, state_sequence, where='post',
                      linewidth=1.5, color='navy')
        ax_state.set_yticks(sorted(unique_states))
        ax_state.set_yticklabels([f'S{s}' for s in sorted(unique_states)])
        ax_state.set_xlabel('Time (seconds)', fontsize=9)
        ax_state.set_ylabel('Hidden State', fontsize=9)
        ax_state.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        plt.close()

    # =========================================================================
    # SECTION 7 — State-specific and patient-vs-control feature importance
    # =========================================================================

    def compute_state_specific_importance(
        self,
        model: Optional[GaussianHMM] = None,
        channel_names: Optional[List[str]] = None
    ) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
        """
        Compute feature importance at two granularities:

        1. Global importance — which channels vary most across all states?
           A channel with high variance in its emission mean across states
           is a strong discriminator of movement phase.

        2. State-specific importance — for each movement phase (state),
           which channels have the largest absolute emission mean?
           This tells you what kinematic signature defines that phase.

        These are complementary to the classification-level permutation
        importance from compute_feature_importance() and serve as the
        emission-based baseline importance for the paper.

        Parameters
        ----------
        model : GaussianHMM, optional
            Defaults to self.fitted_hmm1
        channel_names : List[str], optional
            Defaults to CHAN_NAME

        Returns
        -------
        global_importance : Dict[str, float]
            Sorted descending — channel → normalised importance
        state_importance : Dict[int, Dict[str, float]]
            {state_index → {channel → normalised importance}}
        """
        if model is None:
            if self.fitted_hmm1 is None:
                raise RuntimeError("Call fit_for_analysis() first.")
            model = self.fitted_hmm1

        if channel_names is None:
            channel_names = CHAN_NAME

        means = model.means_           # (n_states, n_features)
        n_states, n_features = means.shape

        # ── Per-state importance: absolute emission mean, normalised ──────────
        state_importance: Dict[int, Dict[str, float]] = {}

        for s in range(n_states):
            abs_means = np.abs(means[s])
            normed = abs_means / (abs_means.sum() + 1e-12)
            state_importance[s] = {
                channel_names[i]: float(normed[i])
                for i in np.argsort(normed)[::-1]
            }

        # ── Global importance: std of emission means across states ────────────
        # High std → channel changes a lot between phases → highly informative
        global_scores = means.std(axis=0)
        global_scores = global_scores / (global_scores.sum() + 1e-12)
        global_importance = {
            channel_names[i]: float(global_scores[i])
            for i in np.argsort(global_scores)[::-1]
        }

        # Print report
        print("\n[HMM] Global State-Based Feature Importance "
              "(emission std across states):")
        for rank, (ch, sc) in enumerate(global_importance.items(), 1):
            print(f"  {rank:2d}. {ch}: {sc:.4f}")

        return global_importance, state_importance

    def compare_patient_control_emissions(
        self,
        channel_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Visualise per-state emission mean differences between patient and
        control class-conditional HMMs.

        A positive difference (patient − control) for a channel in a given
        state means patients show a higher mean value in that movement phase.
        This directly identifies compensatory movement signatures.

        ⚠️  Requires both models to have the same n_components.  If they
        differ, use Hungarian matching (not implemented here) or re-train
        with a fixed n_components.

        Parameters
        ----------
        channel_names : List[str], optional
        save_path : Path, optional

        Returns
        -------
        mean_diff : np.ndarray, shape (n_states, n_features)
            patient_means − control_means per state and channel
        """
        if self.fitted_hmm0 is None or self.fitted_hmm1 is None:
            raise RuntimeError("Call fit_for_analysis() first.")

        if channel_names is None:
            channel_names = CHAN_NAME

        m0 = self.fitted_hmm0.means_   # (n_states, n_features)
        m1 = self.fitted_hmm1.means_

        if m0.shape[0] != m1.shape[0]:
            raise ValueError(
                f"n_components mismatch: class0={m0.shape[0]}, "
                f"class1={m1.shape[0]}. Re-train both with the same n_components."
            )

        mean_diff = m1 - m0           # positive = patients higher
        n_states = mean_diff.shape[0]

        fig, axes = plt.subplots(
            1, n_states,
            figsize=(max(6, 5 * n_states), 7),
            sharey=True
        )
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
            'Patient vs Control Emission Mean Differences per Movement Phase\n'
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
        title: str = 'State-Specific Feature Importance',
        save_path: Optional[Path] = None
    ):
        """
        Heatmap of per-state feature importances.

        Rows = hidden states (movement phases).
        Columns = 18 sensor channels.
        Colour = normalised importance within each state.

        Parameters
        ----------
        state_importance : Dict[int, Dict[str, float]]
            From compute_state_specific_importance()
        channel_names : List[str], optional
        title : str
        save_path : Path, optional
        """
        if channel_names is None:
            channel_names = CHAN_NAME

        n_states = len(state_importance)
        n_features = len(channel_names)

        # Build matrix: rows = states, cols = channels
        matrix = np.zeros((n_states, n_features))
        for s_idx in range(n_states):
            for c_idx, ch in enumerate(channel_names):
                matrix[s_idx, c_idx] = state_importance[s_idx].get(ch, 0.0)

        fig, ax = plt.subplots(figsize=(max(12, n_features * 0.7),
                                        max(4, n_states * 0.8 + 1)))
        sns.heatmap(
            matrix,
            xticklabels=channel_names,
            yticklabels=[f'State {s}' for s in range(n_states)],
            cmap='YlOrRd',
            annot=True,
            fmt='.3f',
            linewidths=0.4,
            ax=ax,
            cbar_kws={'label': 'Normalised Importance'}
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_title(title, fontsize=11, pad=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        plt.close()