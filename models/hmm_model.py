"""
Hidden Markov Model for shoulder pathology classification.

Implements class-conditional Gaussian HMMs with:
- Generative scoring (log-likelihood delta → sigmoid → probability)
- Permutation-based feature importance (channel-level)
- Subject-level LOO CV (consistent with RNN/CNN)
- Per-fold diagnostic tracking
- Checkpoint saving (JSON — no PyTorch state dict)
"""
from typing import Dict, Optional
import numpy as np
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from sklearn.metrics import balanced_accuracy_score
from hmmlearn.hmm import GaussianHMM
from joblib import parallel_backend, Parallel, delayed

from models.base_model import BaseModel, ModelResults
from config.constants import CHAN_NAME
from utils.metrics import compute_metrics


class HMMModel(BaseModel):
    """Hidden Markov Model wrapper with subject-level LOO CV."""

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

        # Save checkpoint (JSON — no PyTorch state dict for HMM)
        if self.checkpoint_dir:
            import json
            from datetime import datetime

            best_path = self.checkpoint_dir / f"best_model_BA{best_score:.4f}.json"
            with open(best_path, "w") as f:
                json.dump({
                    'model_name': 'HMM',
                    'hyperparameters': best_params,
                    'metrics': {'balanced_accuracy': best_score, **metrics},
                    'feature_importance': feature_imp,
                    'input_shape': [len(X), int(X[0].shape[1])],
                    'predictions': {
                        'y_true': y_true.tolist(),
                        'y_pred': y_pred.tolist(),
                        'y_proba': y_proba.tolist()
                    },
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)

            print(f"\n BEST MODEL SAVED: {best_path}")

        return ModelResults(
            metrics=metrics,
            best_params=best_params,
            feature_importance=feature_imp,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            X_shape=X.shape,
            per_fold_results=per_fold_results   # populated — not None like before
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

            # HMM supports multiple test sequences per fold (e.g. windowed data)
            # but for XDash subject-level LOO this will always be a single sequence
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

                # HMM has no loss curve — log-likelihood delta is the closest proxy
                'train_loss': None,     # not applicable
                'val_loss': None,       # not applicable

                # Accuracies
                'train_acc': None,      # not computed (generative model, no train acc)
                'val_acc': float(fold_ba),

                # Predictions
                'y_true': y_test_list,
                'y_pred': fold_preds,
                'y_proba': fold_proba,

                # Training info
                'epochs_trained': params.get('n_iter'),   # EM iterations
                'early_stopped': False  # hmmlearn runs fixed n_iter EM
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

        Unlike the CNN/RNN equivalent, this returns a single HMM (not a
        pair of class-conditional models) since GaussianHMM has no concept
        of class — it is always fitted on a single class's sequences.
        The returned model is unfitted (no trained parameters).

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
            Uninitialised model with best architecture, or None if unavailable
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

        Concatenates the channel values from all sequences, shuffles them
        globally, then scatters them back — preserving sequence lengths but
        destroying the channel's temporal and cross-sequence structure.

        Parameters
        ----------
        seqs : list of np.ndarray
            Sequences of shape (T_i, C)
        channel : int
            Channel index to permute
        rng : np.random.Generator
            Seeded random generator for reproducibility

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