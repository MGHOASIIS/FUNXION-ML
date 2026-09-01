"""
Hidden Markov Model for shoulder pathology classification.

Implements class-conditional Gaussian HMMs with:
- Generative scoring (log-likelihood delta → sigmoid → probability)
- Permutation-based feature importance (channel-level)
- Subject-level LOO CV (consistent with RNN/CNN)
- Per-fold diagnostic tracking
- Checkpoint saving (JSON — no PyTorch state dict)

Phase 2 analysis (hidden state decoding, CSV event-marker alignment, emission/
transition plotting, state-specific feature importance, patient vs control
comparison) is shared with HSMMModel via StateSequenceAnalysisMixin in
models/state_sequence_analysis.py — see that module for details.
"""
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import balanced_accuracy_score, f1_score
from hmmlearn.hmm import GaussianHMM
from joblib import parallel_backend, Parallel, delayed

from models.base_model import BaseModel, ModelResults
from models.state_sequence_analysis import StateSequenceAnalysisMixin
from utils.metrics import compute_metrics, compute_multilabel_metrics
from utils.training import (
    build_loo_splits, resolve_fold_masks, build_fold_record, print_best,
    scale_sequences_global, fold_scale_variable_length,
)


# =============================================================================
# Main HMMModel class — classification pipeline unchanged
# =============================================================================

class HMMModel(StateSequenceAnalysisMixin, BaseModel):
    """
    Hidden Markov Model wrapper with subject-level LOO CV.

    Classification pipeline (train_and_evaluate / fit) is identical to
    the original implementation.  All Phase-2 analysis methods (decoding,
    event alignment, plotting, importance) are inherited from
    StateSequenceAnalysisMixin and can be called independently after
    training.
    """

    def __init__(self, checkpoints_dir=None, task=None, paradigm=None, channel_names=None,
                 multilabel=False, label_names=None):
        """
        Parameters
        ----------
        checkpoints_dir : Path or None
            Directory to save best-model JSON checkpoint
        task : str or None
            Task name — stored for downstream tracking
        paradigm : int or None
            Classification paradigm index — stored for downstream tracking
        channel_names : list of str, optional
            Input channel names, e.g. dataset_config["channels"]. Used for
            feature-importance labelling.
        multilabel : bool
            If True, fit N independent one-vs-rest HMM pairs (one per label)
            instead of a single class-0-vs-class-1 pair.
        label_names : list of str, optional
            Required when multilabel=True.

        Note: patience and min_delta are not used by the HMM (no iterative
        training loop), but are accepted by BaseModel and left as None.
        """
        super().__init__(
            model_name="HMM",
            checkpoints_dir=checkpoints_dir,
            patience=None,
            min_delta=None,
            task=task,
            paradigm=paradigm,
            channel_names=channel_names,
            multilabel=multilabel,
            label_names=label_names,
        )

        # Stored after fit_for_analysis() — used by all analysis methods
        self.fitted_hmm0: Optional[GaussianHMM] = None   # control model (binary only)
        self.fitted_hmm1: Optional[GaussianHMM] = None   # patient/condition model (binary only)

        # Stored after fit_for_analysis_multilabel() — one-vs-rest pair per label
        self.fitted_hmms: Dict[str, Tuple[GaussianHMM, GaussianHMM]] = {}

    def _get_fitted(self, cls_idx: int):
        return self.fitted_hmm0 if cls_idx == 0 else self.fitted_hmm1

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
        cv_splits, unique_subjects = build_loo_splits(len(X), subject_ids, "HMM")

        grid = list(ParameterGrid(param_grid))
        print(f"[HMM] Evaluating {len(grid)} hyperparameter combinations...")

        # Parallel search — hmmlearn is CPU-only and process-safe
        score_fn = self._loo_score_multilabel if self.multilabel else self._loo_score
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=-1, verbose=10)(
                delayed(score_fn)(
                    params, X, y, cv_splits, subject_ids, unique_subjects
                )
                for params in grid
            )

        # Select best configuration
        best_result = max(results, key=lambda t: t[0])
        (best_score, best_params, y_true, y_pred, y_proba,
         per_fold_results, subject_order) = best_result

        print_best("HMM", best_params, best_score)

        # Sanity check: every prediction's subject ID must agree with its own
        # y_true label (g1_ tag → label 1, g0_ tag → label 0). This is the
        # invariant that a patients-first/controls-first ordering mismatch
        # between IDs and predictions would violate. Binary-only — multilabel
        # subject_ids have no g1_/g0_ prefix (see _collect_sequences_multilabel).
        if not self.multilabel and subject_ids is not None:
            assert len(subject_order) == len(y_true), (
                f"subject_order length {len(subject_order)} != y_true length {len(y_true)}"
            )
            for sid, yt in zip(subject_order, y_true):
                expected = 1 if str(sid).startswith("g1_") else 0
                assert expected == int(yt), (
                    f"subject_id/y_true mismatch: {sid} implies label {expected}, "
                    f"got y_true={yt}"
                )

        # ── Auto-fit on full data using best LOO CV params ────────────────────
        # This populates self.fitted_hmm0/self.fitted_hmm1 (binary) or
        # self.fitted_hmms (multilabel) so downstream analysis methods work
        # immediately after train_and_evaluate() without any extra call.
        # We use the best params found by LOO CV — not BIC/AIC — so the
        # interpretation reflects the actual best-performing configuration.
        #
        # This is an explicit full-dataset, no-CV fit for Phase-2
        # interpretability (not a performance estimate — see
        # fit_for_analysis()'s docstring), so there's no held-out subject to
        # protect here. It gets its own globally-scaled copy of X, fit once
        # over everyone — distinct from the raw X above, which _loo_score()
        # scales per fold (training subjects only) for the actual LOSO
        # evaluation.
        X_analysis = scale_sequences_global(list(X))

        if self.multilabel:
            print(f"\n[HMM] Fitting on full data with best params for analysis (multilabel) ...")
            self.fit_for_analysis_multilabel(
                X               = X_analysis,
                y               = y,
                label_names     = self.label_names,
                n_components    = best_params["n_components"],
                covariance_type = best_params["covariance_type"],
                n_iter          = best_params["n_iter"]
            )
        else:
            print(f"\n[HMM] Fitting on full data with best params for analysis ...")
            self.fit_for_analysis(
                X               = X_analysis,
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
        if self.multilabel:
            metrics = compute_multilabel_metrics(y_true, y_pred, y_proba, self.label_names)
            checkpoint_metrics = {'macro_f1': best_score, **metrics}
        else:
            metrics = compute_metrics(y_true, y_pred, y_proba)
            checkpoint_metrics = {'balanced_accuracy': best_score, **metrics}

        # ── Save best model checkpoint ────────────────────────────────────────
        # Always save — if checkpoint_dir was not explicitly provided (i.e.
        # --save-checkpoints flag was not passed), fall back to a 'checkpoints'
        # subdirectory next to wherever the module lives so the file is never lost.
        from datetime import datetime

        if isinstance(self.checkpoint_dir, Path):
            save_dir = self.checkpoint_dir
        else:
            # Fallback: save under storage/, never the repo root
            from config.paths import STORAGE_DIR
            save_dir = STORAGE_DIR / "experiments" / f"task{self.task}" / f"paradigm{self.paradigm}"

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
                'metrics':          checkpoint_metrics,
                'feature_importance': feature_imp,
                'input_shape':      [len(X), int(X[0].shape[1])],
                'predictions': {
                    'y_true':       y_true.tolist(),
                    'y_pred':       y_pred.tolist(),
                    'y_proba':      y_proba.tolist(),
                    'subject_ids':  subject_order.tolist() if subject_ids is not None else [],
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
            subject_ids=subject_order if subject_ids is not None else subject_ids,
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
            (balanced_accuracy, params, y_true, y_pred, y_proba, per_fold_results,
             subject_order)
            subject_order[i] is the subject ID that produced y_true[i]/y_pred[i]/
            y_proba[i] — the fold-iteration order (sorted by np.unique(subject_ids)
            inside build_loo_splits), NOT the original input subject_ids order.
            Callers must use this array, not the raw input, when attaching IDs to
            these predictions (see train_and_evaluate).
        """
        y_true, y_pred, y_proba, subject_order = [], [], [], []
        per_fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):

            train_sample_idx, test_sample_idx, test_subjects = resolve_fold_masks(
                subject_ids, unique_subjects, train_idx, test_idx, fold_idx
            )

            seqs_train = [sequences[i] for i in train_sample_idx]
            y_train = y[train_sample_idx]

            seqs_test = [sequences[i] for i in test_sample_idx]
            y_test_list = y[test_sample_idx].tolist()

            # Fold-safe z-score normalization: fit on training subjects
            # only — the held-out subject never contributes to the
            # statistic used to normalize the training data.
            seqs_train, seqs_test = fold_scale_variable_length(seqs_train, seqs_test)

            # Train class-conditional HMMs
            hmm0 = self._fit_hmm(
                [s for s, lab in zip(seqs_train, y_train) if lab == 0],
                **params
            )
            hmm1 = self._fit_hmm(
                [s for s, lab in zip(seqs_train, y_train) if lab == 1],
                **params
            )

            # Score each test sequence. The raw log-likelihood delta scales with
            # sequence length (thousands of nats over a full 50Hz trial), which
            # saturates sigmoid(delta) to exactly 0 or 1 and makes trials of
            # different duration incomparable. Normalising by T_i (per-frame
            # average log-likelihood delta) keeps the score graded and duration-
            # independent.
            fold_preds, fold_proba = [], []
            for seq in seqs_test:
                delta = (hmm1.score(seq) - hmm0.score(seq)) / len(seq)
                prob = self._stable_sigmoid(delta)
                pred = int(prob >= 0.5)
                fold_preds.append(pred)
                fold_proba.append(prob)

            y_true.extend(y_test_list)
            y_pred.extend(fold_preds)
            y_proba.extend(fold_proba)
            subject_order.extend(list(test_subjects))

            fold_ba = balanced_accuracy_score(y_test_list, fold_preds)

            per_fold_results.append(build_fold_record(
                fold_idx, test_subjects, subject_ids,
                y_test_list, fold_preds, fold_proba, fold_ba,
                epochs_trained=params.get("n_iter"),
            ))

        ba = balanced_accuracy_score(y_true, y_pred)
        return (
            ba,
            params,
            np.array(y_true),
            np.array(y_pred),
            np.array(y_proba),
            per_fold_results,
            np.array(subject_order)
        )

    def _loo_score_multilabel(
        self,
        params: Dict,
        sequences: np.ndarray,
        y: np.ndarray,
        cv_splits: list,
        subject_ids: Optional[np.ndarray],
        unique_subjects: Optional[np.ndarray]
    ):
        """
        Multi-label counterpart of _loo_score(). Instead of one class-0-vs-
        class-1 HMM pair, fits N independent one-vs-rest HMM pairs — one per
        label in self.label_names — per fold. Each test sequence gets N
        independent probabilities (does it have label k, yes/no), reusing
        the exact same LLR → sigmoid scoring as the binary path, applied
        once per label.

        Note: this is N times the HMM fits per fold compared to the binary
        path (2*n_labels vs 2) — an inherent cost of one-vs-rest multi-label
        with generative per-class models, not something optimized away here.

        Returns
        -------
        tuple
            (macro_f1, params, y_true, y_pred, y_proba, per_fold_results,
             subject_order) — y_true/y_pred/y_proba are (N, n_labels).
        """
        label_names = self.label_names
        n_labels = len(label_names)

        y_true_rows, y_pred_rows, y_proba_rows = [], [], []
        subject_order = []
        per_fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):

            train_sample_idx, test_sample_idx, test_subjects = resolve_fold_masks(
                subject_ids, unique_subjects, train_idx, test_idx, fold_idx
            )

            seqs_train = [sequences[i] for i in train_sample_idx]
            y_train = y[train_sample_idx]          # (n_train, n_labels)

            seqs_test = [sequences[i] for i in test_sample_idx]
            y_test_rows = y[test_sample_idx]        # (n_test, n_labels)

            # Fold-safe z-score normalization: fit on training subjects
            # only — the held-out subject never contributes to the
            # statistic used to normalize the training data.
            seqs_train, seqs_test = fold_scale_variable_length(seqs_train, seqs_test)

            # Fit one one-vs-rest HMM pair per label
            label_hmms = {}
            for li, label_name in enumerate(label_names):
                pos_seqs = [s for s, row in zip(seqs_train, y_train) if row[li] == 1]
                neg_seqs = [s for s, row in zip(seqs_train, y_train) if row[li] == 0]
                label_hmms[label_name] = (
                    self._fit_hmm(pos_seqs, **params),
                    self._fit_hmm(neg_seqs, **params),
                )

            n_test = len(seqs_test)
            fold_preds = np.zeros((n_test, n_labels), dtype=int)
            fold_proba = np.zeros((n_test, n_labels), dtype=float)

            for si, seq in enumerate(seqs_test):
                for li, label_name in enumerate(label_names):
                    hmm_pos, hmm_neg = label_hmms[label_name]
                    delta = (hmm_pos.score(seq) - hmm_neg.score(seq)) / len(seq)
                    prob = self._stable_sigmoid(delta)
                    fold_proba[si, li] = prob
                    fold_preds[si, li] = int(prob >= 0.5)

            y_true_rows.append(y_test_rows)
            y_pred_rows.append(fold_preds)
            y_proba_rows.append(fold_proba)
            subject_order.extend(list(test_subjects))

            fold_score = f1_score(y_test_rows, fold_preds, average="macro", zero_division=0)

            per_fold_results.append(build_fold_record(
                fold_idx, test_subjects, subject_ids,
                y_test_rows.tolist(), fold_preds.tolist(), fold_proba.tolist(), fold_score,
                epochs_trained=params.get("n_iter"),
            ))

        y_true_all = np.vstack(y_true_rows)
        y_pred_all = np.vstack(y_pred_rows)
        y_proba_all = np.vstack(y_proba_rows)
        score = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)

        return (
            score,
            params,
            y_true_all,
            y_pred_all,
            y_proba_all,
            per_fold_results,
            np.array(subject_order)
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
        ch_names = self.resolve_channel_names(n_channels)

        # ── Dummy mode: skip permutation loop when IS_TEST=True ───────────────
        if IS_TEST:
            print("\n[HMM] IS_TEST=True — returning uniform dummy importance "
                  "(set IS_TEST=False for real permutation importance)")
            uniform = 1.0 / n_channels
            feature_imp = {ch: uniform for ch in ch_names}
            return feature_imp

        # ── Real permutation importance ───────────────────────────────────────
        print("\n[HMM] Computing feature importance via permutation "
              f"({n_channels} channels × LOO CV) ...")

        score_fn = self._loo_score_multilabel if self.multilabel else self._loo_score

        rng        = np.random.default_rng(42)
        importance = np.zeros(n_channels)

        # Baseline score with all channels intact (macro-F1 if multilabel, else BA)
        baseline_ba, *_ = score_fn(
            best_params, X, y, cv_splits, subject_ids, unique_subjects
        )
        print(f"  Baseline score: {baseline_ba:.4f}")

        # Permute each channel and measure score drop
        for d in range(n_channels):
            print(f"  Channel {d+1:2d}/{n_channels}: {ch_names[d]:<30}", end=" ")
            seqs_perm = self._permute_channel(X, d, rng)
            ba_d, *_ = score_fn(
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
            ch_names[i]: float(importance[i])
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

    def fit_for_analysis_multilabel(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_names: List[str],
        n_components: int,
        covariance_type: str = 'diag',
        n_iter: int = 100
    ) -> Dict[str, Tuple[GaussianHMM, GaussianHMM]]:
        """
        Multi-label counterpart of fit_for_analysis() — fits one one-vs-rest
        HMM pair per label on the FULL dataset (no CV), storing the result
        in self.fitted_hmms[label_name] = (hmm_pos, hmm_neg).

        Parameters
        ----------
        X : np.ndarray
            Array of (T_i, C) sequences, one per sample
        y : np.ndarray
            Multi-hot label matrix, shape (N, len(label_names))
        label_names : List[str]
            Names for each column of y, in order
        n_components, covariance_type, n_iter
            Same as fit_for_analysis()

        Returns
        -------
        Dict[str, Tuple[GaussianHMM, GaussianHMM]]
            self.fitted_hmms, for convenience
        """
        print(f"\n[HMM Analysis] Fitting on full data (multilabel)")
        print(f"  n_components={n_components}, covariance_type={covariance_type}")

        self.fitted_hmms = {}
        for li, label_name in enumerate(label_names):
            pos_seqs = [X[i] for i in range(len(X)) if y[i, li] == 1]
            neg_seqs = [X[i] for i in range(len(X)) if y[i, li] == 0]
            print(f"  Label '{label_name}': {len(pos_seqs)} positive, {len(neg_seqs)} negative")
            self.fitted_hmms[label_name] = (
                self._fit_hmm(pos_seqs, n_components, covariance_type, n_iter),
                self._fit_hmm(neg_seqs, n_components, covariance_type, n_iter),
            )

        print(f"  ✓ {len(label_names)} one-vs-rest HMM pairs fitted successfully")
        return self.fitted_hmms
