"""
Hidden Markov Model for shoulder pathology classification.
"""
from typing import Dict, Optional
import numpy as np
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from sklearn.metrics import balanced_accuracy_score
from hmmlearn.hmm import GaussianHMM
from joblib import parallel_backend, Parallel, delayed

from models.base_model import BaseModel, ModelResults
from config.constants import CHAN_NAME


class HMMModel(BaseModel):
    """Hidden Markov Model wrapper."""
    
    def __init__(self):
        super().__init__(model_name="HMM")
    
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        param_grid: Optional[Dict] = None
    ) -> ModelResults:
        """Train HMM with hyperparameter search using LOO CV."""
        
        if param_grid is None:
            from config.hyperparameter import HMM_PARAM_GRID
            param_grid = HMM_PARAM_GRID
        
        loo = LeaveOneOut()
        grid = list(ParameterGrid(param_grid))
        
        print(f"[HMM] Evaluating {len(grid)} hyperparameter combinations...")
        
        # Parallel hyperparameter search
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=-1, verbose=10)(
                delayed(self._loo_score)(params, X, y, loo) 
                for params in grid
            )
        
        # Select best configuration
        best_score, best_params, y_true, y_pred, y_proba = max(
            results, key=lambda t: t[0]
        )
        
        print(f"\n[HMM] Best params: {best_params}")
        print(f"[HMM] Best balanced accuracy: {best_score:.4f}")
        
        # Compute feature importance
        feature_imp = self.compute_feature_importance(
            X=X,
            y=y,
            best_params=best_params,
            loo=loo
        )
        
        # Compute metrics
        from utils.metrics import compute_metrics
        metrics = compute_metrics(y_true, y_pred, y_proba)
        
        return ModelResults(
            metrics=metrics,
            best_params=best_params,
            feature_importance=feature_imp,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            X_shape=X.shape
        )
    
    def _loo_score(
        self,
        params: Dict,
        sequences: np.ndarray,
        y: np.ndarray,
        loo: LeaveOneOut
    ):
        """Compute LOO score for given hyperparameters."""
        y_true, y_pred, y_proba = [], [], []
        
        for train_idx, test_idx in loo.split(range(len(sequences))):
            # Split data
            seqs_train = [sequences[i] for i in train_idx]
            seq_test = sequences[test_idx[0]]
            y_train = y[train_idx]
            y_test = y[test_idx][0]
            
            # Train class-conditional HMMs
            hmm0 = self._fit_hmm(
                [s for s, lab in zip(seqs_train, y_train) if lab == 0],
                **params
            )
            hmm1 = self._fit_hmm(
                [s for s, lab in zip(seqs_train, y_train) if lab == 1],
                **params
            )
            
            # Predict
            delta = hmm1.score(seq_test) - hmm0.score(seq_test)
            prob = self._stable_sigmoid(delta)
            pred = int(prob >= 0.5)
            
            y_true.append(int(y_test))
            y_pred.append(pred)
            y_proba.append(prob)
        
        ba = balanced_accuracy_score(y_true, y_pred)
        return ba, params, np.array(y_true), np.array(y_pred), np.array(y_proba)
    
    def _fit_hmm(
        self,
        seq_list: list,
        n_components: int,
        covariance_type: str,
        n_iter: int
    ) -> GaussianHMM:
        """Fit a Gaussian HMM."""
        lengths = [s.shape[0] for s in seq_list]
        X = np.vstack(seq_list)
        
        model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=42,
            verbose=False
        ).fit(X, lengths=lengths)
        
        return model
    
    @staticmethod
    def _stable_sigmoid(delta: float) -> float:
        """Numerically stable sigmoid."""
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
        best_params: Dict,
        loo: LeaveOneOut
    ) -> Dict[str, float]:
        """Compute permutation-based feature importance."""
        
        print("\n[HMM] Computing feature importance via permutation...")
        
        rng = np.random.default_rng(42)
        n_channels = X[0].shape[1]
        importance = np.zeros(n_channels)
        
        # Baseline score
        baseline_ba, *_ = self._loo_score(best_params, X, y, loo)
        
        # Permute each channel
        for d in range(n_channels):
            print(f"  Channel {d+1}/{n_channels}: {CHAN_NAME[d]}")
            seqs_perm = self._permute_channel(X, d, rng)
            ba_d, *_ = self._loo_score(best_params, seqs_perm, y, loo)
            importance[d] = baseline_ba - ba_d
        
        # Create importance dictionary
        feature_imp = {
            CHAN_NAME[i]: float(importance[i])
            for i in np.argsort(importance)[::-1]
        }
        
        print("\n[HMM] Top 5 features:")
        for i, (feat, imp) in enumerate(list(feature_imp.items())[:5]):
            print(f"  {i+1}. {feat}: {imp:.4f}")
        
        return feature_imp
    
    @staticmethod
    def _permute_channel(seqs: list, channel: int, rng) -> list:
        """Permute a single channel across all sequences."""
        seqs_perm = [s.copy() for s in seqs]
        
        # Stack channel, shuffle, scatter back
        col = np.concatenate([s[:, channel] for s in seqs_perm])
        rng.shuffle(col)
        
        start = 0
        for s in seqs_perm:
            L = len(s)
            s[:, channel] = col[start:start + L]
            start += L
        
        return seqs_perm