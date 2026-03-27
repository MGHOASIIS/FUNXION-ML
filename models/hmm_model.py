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

        # ── Auto-fit on full data using best LOO CV params ────────────────────
        # This populates self.fitted_hmm0 and self.fitted_hmm1 so downstream
        # analysis methods (decode_sequence, plot_emission_distributions etc.)
        # work immediately after train_and_evaluate() without any extra call.
        # We use the best params found by LOO CV — not BIC/AIC — so the
        # interpretation reflects the actual best-performing configuration.
        print(f"\n[HMM] Fitting on full data with best params for analysis ...")
        self.fit_for_analysis(
            X               = X,
            y               = y,
            n_components    = best_params["n_components"],
            covariance_type = best_params["covariance_type"],
            n_iter          = best_params["n_iter"]
        )

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
            # Fallback: save alongside this source file
            save_dir = Path(__file__).resolve().parent.parent / "experiments" / f"task{self.task}" / f"paradigm{self.paradigm}"

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

        When IS_TEST=True in hyperparameter.py, skips the permutation loop
        entirely and returns uniform dummy importance so the pipeline can be
        tested end-to-end quickly. Set IS_TEST=False for real runs.

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
        from config.hyperparameter import IS_TEST

        best_params     = kwargs["best_params"]
        cv_splits       = kwargs["cv_splits"]
        subject_ids     = kwargs.get("subject_ids", None)
        unique_subjects = kwargs.get("unique_subjects", None)

        n_channels = X[0].shape[1]

        # ── Dummy mode: skip permutation loop when IS_TEST=True ───────────────
        if IS_TEST:
            print("\n[HMM] IS_TEST=True — returning uniform dummy importance "
                  "(set IS_TEST=False for real permutation importance)")
            uniform = 1.0 / n_channels
            feature_imp = {ch: uniform for ch in CHAN_NAME}
            return feature_imp

        # ── Real permutation importance ───────────────────────────────────────
        print("\n[HMM] Computing feature importance via permutation "
              f"({n_channels} channels × LOO CV) ...")

        rng        = np.random.default_rng(42)
        importance = np.zeros(n_channels)

        # Baseline balanced accuracy with all channels intact
        baseline_ba, *_ = self._loo_score(
            best_params, X, y, cv_splits, subject_ids, unique_subjects
        )
        print(f"  Baseline BA: {baseline_ba:.4f}")

        # Permute each channel and measure accuracy drop
        for d in range(n_channels):
            print(f"  Channel {d+1:2d}/{n_channels}: {CHAN_NAME[d]:<30}", end=" ")
            seqs_perm = self._permute_channel(X, d, rng)
            ba_d, *_ = self._loo_score(
                best_params, seqs_perm, y, cv_splits, subject_ids, unique_subjects
            )
            importance[d] = baseline_ba - ba_d
            print(f"drop={importance[d]:+.4f}")

        # Clip negatives to zero (channel had no effect or helped by chance)
        importance = np.clip(importance, 0, None)

        denom = importance.sum()
        if denom > 1e-12:
            importance = importance / denom
        else:
            # All channels equally unimportant — return uniform
            importance = np.ones(n_channels) / n_channels

        feature_imp = {
            CHAN_NAME[i]: float(importance[i])
            for i in np.argsort(importance)[::-1]
        }

        print("\n[HMM] Channel Importance:")
        for i, (feat, imp) in enumerate(list(feature_imp.items())):
            print(f"  {i+1:2d}. {feat:<30}: {imp:.4f}")

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
    # SECTION 3 — State decoding and temporal segmentation
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
    # SECTION 4 — CSV event marker loading and alignment
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
    # SECTION 5 — Emission distribution visualization
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
        events: Optional[List[Dict]] = None,
        sampling_rate: int = 50,
        title: str = 'State Sequence over Time',
        save_path: Optional[Path] = None
    ):
        """
        Comprehensive state-sequence visualisation showing ALL 18 sensor
        channels, the decoded state assignment on a proper time axis, and
        annotated event markers with state-alignment annotations.

        Layout (top to bottom)
        ----------------------
        Row 0          : Head position  (3 channels — pos x/y/z)
        Row 1          : Head rotation  (3 channels — rot x/y/z)
        Row 2          : Right hand pos (3 channels)
        Row 3          : Right hand rot (3 channels)
        Row 4          : Left hand pos  (3 channels)
        Row 5          : Left hand rot  (3 channels)
        Row 6 (bottom) : State timeline — discrete state index vs time (seconds)
                         with event markers, state-colour bands, and a
                         state-to-event alignment summary table printed to stdout.

        Parameters
        ----------
        sequence : np.ndarray, shape (T, 18)
            Z-scored sensor data for one subject
        state_sequence : np.ndarray, shape (T,)
            Viterbi-decoded state per frame, from decode_sequence()
        events : List[Dict], optional
            Event markers from load_event_markers().
            Each dict must have 'event_name' (str) and 'timestamp' (float, seconds).
        sampling_rate : int
            Sampling rate in Hz (default 50) — used to build the time axis.
        title : str
            Figure suptitle
        save_path : Path, optional
            If provided, saves the figure here.
        """
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
        import matplotlib.patches as mpatches

        T            = len(state_sequence)
        time         = np.arange(T) / sampling_rate          # seconds
        unique_states = np.unique(state_sequence)
        n_states      = len(unique_states)

        # ── Colour palette — one distinct colour per state ────────────────────
        cmap = plt.cm.get_cmap("tab10", max(n_states, 2))
        state_colors = {int(s): cmap(i) for i, s in enumerate(sorted(unique_states))}

        # ── Sensor groups: (label, channel_indices) ───────────────────────────
        sensor_groups = [
            ("Head Position",      list(range(0, 3))),
            ("Head Rotation",      list(range(3, 6))),
            ("Left Hand Pos",     list(range(6, 9))),
            ("Left Hand Rot",     list(range(9, 12))),
            ("Right Hand Pos",      list(range(12, 15))),
            ("Right Hand Rot",      list(range(15, 18))),
        ]
        n_groups = len(sensor_groups)

        # ── Build figure: n_groups signal rows + 1 state row ─────────────────
        fig = plt.figure(figsize=(20, 3 * n_groups + 3))
        gs  = GridSpec(
            n_groups + 1, 1,
            figure=fig,
            height_ratios=[2] * n_groups + [2],
            hspace=0.08
        )

        axes_signal = [fig.add_subplot(gs[i]) for i in range(n_groups)]
        ax_state    = fig.add_subplot(gs[n_groups])

        # Share x axis across all panels
        for ax in axes_signal:
            ax.sharex(ax_state)

        # ── Helper: shade state background on any axis ────────────────────────
        def _shade_states(ax, alpha=0.18):
            prev  = state_sequence[0]
            start = time[0]
            for t in range(1, T):
                if state_sequence[t] != prev or t == T - 1:
                    end = time[t] if t < T - 1 else time[-1]
                    ax.axvspan(start, end,
                               alpha=alpha,
                               color=state_colors[int(prev)],
                               linewidth=0)
                    prev  = state_sequence[t]
                    start = time[t]

        # ── Helper: draw event lines on any axis ──────────────────────────────
        def _draw_events(ax, label_y_frac=None, fontsize=6):
            if not events:
                return
            ylim = ax.get_ylim()
            yrange = ylim[1] - ylim[0]
            label_y = ylim[1] - 0.02 * yrange if label_y_frac is None \
                      else ylim[0] + label_y_frac * yrange
            for ev in events:
                t_ev = float(ev["timestamp"])
                ax.axvline(t_ev, color="black", linestyle=":",
                           linewidth=1.0, alpha=0.7, zorder=5)
                ax.text(t_ev + 0.1, label_y,
                        ev["event_name"],
                        rotation=90, fontsize=fontsize,
                        color="black", va="top", ha="left",
                        zorder=6,
                        bbox=dict(boxstyle="round,pad=0.1",
                                  fc="white", ec="none", alpha=0.6))

        # ── Signal panels: one per sensor group ───────────────────────────────
        line_colors = ["#2166ac", "#d6604d", "#4dac26"]  # blue, red, green

        for row_idx, (group_label, ch_indices) in enumerate(sensor_groups):
            ax = axes_signal[row_idx]
            _shade_states(ax, alpha=0.18)

            for ci, ch_idx in enumerate(ch_indices):
                ch_name   = CHAN_NAME[ch_idx].split("_")[-1]   # x / y / z
                ax.plot(time, sequence[:, ch_idx],
                        color=line_colors[ci],
                        linewidth=0.7, alpha=0.85,
                        label=ch_name)

            ax.set_ylabel(group_label, fontsize=8, labelpad=4)
            ax.tick_params(axis="y", labelsize=7)
            ax.tick_params(axis="x", labelbottom=False)
            ax.legend(loc="upper right", fontsize=6, ncol=3,
                      framealpha=0.7, handlelength=1.2)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.set_xlim(time[0], time[-1])

            # Draw event lines after plot so ylim is set
            _draw_events(ax, fontsize=6)

        # ── State timeline panel ───────────────────────────────────────────────
        _shade_states(ax_state, alpha=0.45)

        # Draw state as a step function
        ax_state.step(time, state_sequence, where="post",
                      color="black", linewidth=1.2, zorder=4)

        ax_state.set_yticks(sorted(unique_states))
        ax_state.set_yticklabels(
            [f"State {s}" for s in sorted(unique_states)], fontsize=8
        )
        ax_state.set_xlabel("Time (seconds)", fontsize=9)
        ax_state.set_ylabel("Hidden State", fontsize=9)
        ax_state.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax_state.set_xlim(time[0], time[-1])

        # Draw event markers with labels on the state panel
        if events:
            y_mid  = (sorted(unique_states)[0] + sorted(unique_states)[-1]) / 2
            y_top  = sorted(unique_states)[-1]
            for ev in events:
                t_ev = float(ev["timestamp"])
                ax_state.axvline(t_ev, color="black", linestyle=":",
                                 linewidth=1.2, alpha=0.85, zorder=5)
                # Place label just inside the top of the axis so it's never clipped
                ax_state.text(
                    t_ev + 0.05,
                    sorted(unique_states)[-1] - 0.05,
                    ev["event_name"],
                    rotation=45, fontsize=7,
                    color="black", va="top", ha="left",
                    zorder=6, clip_on=False,
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="lightyellow", ec="gray",
                              alpha=0.85, linewidth=0.5)
                )

        # ── Time-axis ticks — MUST be set before tight_layout ───────────────
        # tight_layout measures label sizes to allocate space; if ticks are
        # set after, the labels get clipped or the axis bottom is misaligned.
        max_t    = time[-1]
        tick_gap = 5.0 if max_t > 30 else (2.0 if max_t > 10 else 1.0)
        xticks   = np.arange(0, max_t + tick_gap, tick_gap)
        ax_state.set_xticks(xticks)
        ax_state.set_xticklabels([f"{t:.0f}s" for t in xticks],
                                  fontsize=8, rotation=0)
        ax_state.tick_params(axis="x", labelbottom=True)

        # ── State legend in state panel ───────────────────────────────────────
        state_patches = [
            mpatches.Patch(color=state_colors[int(s)], alpha=0.7,
                           label=f"State {s}")
            for s in sorted(unique_states)
        ]
        ax_state.legend(handles=state_patches,
                        loc="lower right", fontsize=8,
                        ncol=min(n_states, 4), framealpha=0.8)

        # ── Suptitle ──────────────────────────────────────────────────────────
        fig.suptitle(title, fontsize=12, fontweight="bold")

        # tight_layout AFTER all axis labels are set
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"[Saved] {save_path}")
        plt.close()

        # ── State-to-event alignment summary (printed to stdout) ──────────────
        if events:
            self._print_state_event_alignment(
                state_sequence=state_sequence,
                events=events,
                sampling_rate=sampling_rate,
                unique_states=unique_states
            )

    def _print_state_event_alignment(
        self,
        state_sequence: np.ndarray,
        events: List[Dict],
        sampling_rate: int,
        unique_states: np.ndarray
    ):
        """
        Print a human-readable table showing which hidden state was active
        at each annotated event boundary.

        For each event:
          - The state assigned at that exact frame
          - The dominant state in the 1-second window around the event
          - Whether a state transition occurred near the event

        This directly answers: "Is State 1 the jar-picking-up state?"

        Parameters
        ----------
        state_sequence : np.ndarray (T,)
        events : List[Dict]  — each has 'event_name' and 'timestamp' (seconds)
        sampling_rate : int
        unique_states : np.ndarray
        """
        T = len(state_sequence)

        print("\n" + "="*70)
        print("STATE-TO-EVENT ALIGNMENT SUMMARY")
        print("="*70)
        print(f"  {'Event':<30} {'Time(s)':>7}  "
              f"{'State@event':>11}  "
              f"{'Dom.state±1s':>13}  "
              f"{'Transition?':>11}")
        print("-"*70)

        for ev in events:
            t_ev    = float(ev["timestamp"])
            ev_name = str(ev["event_name"])
            frame   = int(t_ev * sampling_rate)
            frame   = max(0, min(frame, T - 1))

            # State exactly at the event frame
            state_at = int(state_sequence[frame])

            # Dominant state in ±1s window around event
            win_start = max(0, frame - sampling_rate)
            win_end   = min(T, frame + sampling_rate)
            window    = state_sequence[win_start:win_end]
            counts    = {int(s): int(np.sum(window == s)) for s in unique_states}
            dom_state = max(counts, key=counts.get)
            dom_pct   = 100 * counts[dom_state] / len(window)

            # Was there a state transition within ±0.5s?
            half = sampling_rate // 2
            near_start = max(0, frame - half)
            near_end   = min(T, frame + half)
            near_window = state_sequence[near_start:near_end]
            has_transition = len(np.unique(near_window)) > 1

            print(f"  {ev_name:<30} {t_ev:>7.2f}  "
                  f"State {state_at:>2}      "
                  f"State {dom_state:>2} ({dom_pct:>4.0f}%)  "
                  f"{'YES ← transition' if has_transition else 'no'}")

        # Summary: for each state, list which events most commonly occur in it
        print("\n" + "-"*70)
        print("STATE PROFILES (which events are most associated with each state):")
        print("-"*70)

        for s in sorted(unique_states):
            s = int(s)
            events_in_state = []
            for ev in events:
                t_ev  = float(ev["timestamp"])
                frame = int(t_ev * sampling_rate)
                frame = max(0, min(frame, T - 1))
                # Check if the dominant state in ±0.5s window is this state
                half  = sampling_rate // 2
                win   = state_sequence[max(0, frame-half):min(T, frame+half)]
                if len(win) > 0 and np.bincount(win).argmax() == s:
                    events_in_state.append(ev["event_name"])

            if events_in_state:
                print(f"  State {s}: {', '.join(events_in_state)}")
            else:
                print(f"  State {s}: (no annotated events fall predominantly here)")

        print("="*70 + "\n")

    # =========================================================================
    # SECTION 6 — State-specific and patient-vs-control feature importance
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

        means    = model.means_     # (n_states, n_features)
        n_states, n_features = means.shape

        # ── Why previous approaches failed for position channels ──────────────
        #
        # Attempt 1 — abs(emission mean):
        #   After z-scoring, position channel means ≈ 0 for ALL states.
        #   Only rotation channels (which have non-zero resting angles after
        #   normalisation) got non-zero scores. Positions were invisible.
        #
        # Attempt 2 — abs(mean) + emission variance:
        #   z-scoring also makes emission VARIANCES ≈ 1.0 for all channels,
        #   so variance scores are nearly uniform. Still biased.
        #
        # Root cause: both metrics operate on the EMISSION DISTRIBUTION
        # parameters alone. Position channels encode their discriminative
        # information in HOW MUCH THE MEAN SHIFTS BETWEEN STATES, not in
        # the absolute value of the mean in any single state.
        #
        # ── Correct approach: cross-state mean shift ──────────────────────────
        #
        # Per-state importance = how much does this channel's emission mean
        # differ from the AVERAGE emission mean across all states?
        #   score[s, d] = |means[s, d] - mean_over_states[d]|
        #
        # This directly measures "is this channel behaving differently in
        # this state compared to other states?" — which is what we want.
        # Positions that shift from low in State 0 to high in State 2 will
        # score highly; rotations that barely move across states will score low.
        #
        # Global importance = std of means across states per channel.
        # High std → channel sweeps a wide range across phases → highly
        # informative for distinguishing movement phases.
        # This is the same metric but aggregated across states.

        mean_across_states = means.mean(axis=0)   # (n_features,) — average level

        # ── Use range (max-min) across states instead of std ─────────────────
        # std and mean-deviation both collapse to near-zero for z-scored
        # position channels because the HMM may assign similar mean values
        # to positions across states. Range (max - min) is more sensitive:
        # even a small shift from state 0 to state 2 gets captured.
        range_across_states = means.max(axis=0) - means.min(axis=0)  # (n_features,)

        state_importance: Dict[int, Dict[str, float]] = {}

        for s in range(n_states):
            # Deviation of this state from cross-state mean, scaled by the
            # channel's overall range — channels with small range get less
            # credit even if their deviation looks large relative to their mean
            raw_dev = np.abs(means[s] - mean_across_states)
            # Scale by range so channels that barely move get appropriately low scores
            scaled  = raw_dev * (range_across_states + 1e-8)
            normed  = scaled / (scaled.sum() + 1e-12)

            state_importance[s] = {
                channel_names[int(i)]: float(normed[i])
                for i in np.argsort(normed)[::-1]
            }

        # ── Global importance: range of emission means across states ──────────
        # Range = max_state_mean - min_state_mean per channel.
        # A channel whose mean shifts from -0.5 in State 0 to +0.8 in State 2
        # has range 1.3 — much more sensitive than std for small N_states.
        global_scores = range_across_states
        global_scores = global_scores / (global_scores.sum() + 1e-12)
        global_importance = {
            channel_names[int(i)]: float(global_scores[i])
            for i in np.argsort(global_scores)[::-1]
        }

        # ── Print reports ─────────────────────────────────────────────────────
        print("\n[HMM] Global Feature Importance (emission mean range across states):")
        for rank, (ch, sc) in enumerate(global_importance.items(), 1):
            marker = "← position" if "pos" in ch else ""
            print(f"  {rank:2d}. {ch:<25}: {sc:.4f}  {marker}")

        # Abbreviation map — unique across all 18 channels
        _abbrev = {
            "head_pos_x": "hd_px", "head_pos_y": "hd_py", "head_pos_z": "hd_pz",
            "head_rot_x": "hd_rx", "head_rot_y": "hd_ry", "head_rot_z": "hd_rz",
            "left_hand_pos_x":  "lh_px", "left_hand_pos_y":  "lh_py",
            "left_hand_pos_z":  "lh_pz", "left_hand_rot_x":  "lh_rx",
            "left_hand_rot_y":  "lh_ry", "left_hand_rot_z":  "lh_rz",
            "right_hand_pos_x": "rh_px", "right_hand_pos_y": "rh_py",
            "right_hand_pos_z": "rh_pz", "right_hand_rot_x": "rh_rx",
            "right_hand_rot_y": "rh_ry", "right_hand_rot_z": "rh_rz",
        }
        print("\n[HMM] Per-State Importance (mean shift from cross-state average, top 6):")
        for s in range(n_states):
            top6 = list(state_importance[s].items())[:6]
            top6_str = "  ".join(
                f"{_abbrev.get(ch, ch)}:{sc:.3f}" for ch, sc in top6
            )
            print(f"  State {s}: {top6_str}")

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