"""
Cross-validation strategies for XDash models.

Provides various CV strategies with proper handling of:
- Subject-level splits (no data leakage)
- Windowed/augmented data
- Stratification
- Time-series aware splits
"""
from typing import List, Tuple, Optional, Generator, Dict, Any
import numpy as np
from sklearn.model_selection import (
    LeaveOneOut, KFold, StratifiedKFold, GroupKFold
)
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class CVSplit:
    """Represents a single train/test split."""
    train_idx: np.ndarray
    test_idx: np.ndarray
    fold: int
    train_subjects: Optional[List[str]] = None
    test_subjects: Optional[List[str]] = None


@dataclass
class CVResults:
    """Aggregated results from cross-validation."""
    fold_results: List[Dict[str, Any]]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    subject_ids: Optional[np.ndarray] = None
    mean_metrics: Optional[Dict[str, float]] = None
    std_metrics: Optional[Dict[str, float]] = None


class BaseCrossValidator:
    """Base class for cross-validators."""
    
    def __init__(self, verbose: bool = True):
        """
        Parameters
        ----------
        verbose : bool
            Whether to print progress
        """
        self.verbose = verbose
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> Generator[CVSplit, None, None]:
        """
        Generate train/test splits.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        subject_ids : np.ndarray, optional
            Subject identifiers for grouped splits
        
        Yields
        ------
        CVSplit
            Train/test split with metadata
        """
        raise NotImplementedError
    
    def get_n_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> int:
        """Get number of splits."""
        raise NotImplementedError
    
    def _print(self, message: str):
        """Print if verbose."""
        if self.verbose:
            print(message)


class SubjectLevelLOOCV(BaseCrossValidator):
    """
    Leave-One-Subject-Out Cross-Validation.
    
    Critical for windowed/augmented data to prevent data leakage.
    Each fold leaves out all samples from one subject.
    """
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> Generator[CVSplit, None, None]:
        """
        Generate leave-one-subject-out splits.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, ...)
        y : np.ndarray
            Labels (N,)
        subject_ids : np.ndarray
            Subject identifiers (N,) - REQUIRED
        
        Yields
        ------
        CVSplit
            Train/test split
        """
        if subject_ids is None:
            raise ValueError("subject_ids is required for SubjectLevelLOOCV")
        
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)
        
        self._print(f"\n[SubjectLevelLOOCV] {n_subjects} subjects")
        
        for fold, test_subject in enumerate(unique_subjects):
            # Find all samples belonging to test subject
            test_mask = subject_ids == test_subject
            train_mask = ~test_mask
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            train_subjects = list(np.unique(subject_ids[train_idx]))
            test_subjects = [test_subject]
            
            self._print(
                f"  Fold {fold+1}/{n_subjects}: "
                f"Train={len(train_idx)} samples ({len(train_subjects)} subjects), "
                f"Test={len(test_idx)} samples (1 subject: {test_subject})"
            )
            
            yield CVSplit(
                train_idx=train_idx,
                test_idx=test_idx,
                fold=fold,
                train_subjects=train_subjects,
                test_subjects=test_subjects
            )
    
    def get_n_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> int:
        """Get number of splits (= number of unique subjects)."""
        if subject_ids is None:
            raise ValueError("subject_ids is required")
        return len(np.unique(subject_ids))


class SampleLevelLOOCV(BaseCrossValidator):
    """
    Leave-One-Out Cross-Validation at sample level.
    
    WARNING: Only use when you have one sample per subject!
    For windowed data, use SubjectLevelLOOCV instead.
    """
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> Generator[CVSplit, None, None]:
        """
        Generate leave-one-sample-out splits.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, ...)
        y : np.ndarray
            Labels (N,)
        subject_ids : np.ndarray, optional
            Not used, but kept for API consistency
        
        Yields
        ------
        CVSplit
            Train/test split
        """
        n_samples = len(X)
        self._print(f"\n[SampleLevelLOOCV] {n_samples} samples")
        
        loo = LeaveOneOut()
        for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
            self._print(
                f"  Fold {fold+1}/{n_samples}: "
                f"Train={len(train_idx)}, Test={len(test_idx)}"
            )
            
            yield CVSplit(
                train_idx=train_idx,
                test_idx=test_idx,
                fold=fold
            )
    
    def get_n_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> int:
        """Get number of splits (= number of samples)."""
        return len(X)


class StratifiedKFoldCV(BaseCrossValidator):
    """
    Stratified K-Fold Cross-Validation.
    
    Maintains class proportions in each fold.
    Useful when you can't do LOO (e.g., very large datasets).
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42, verbose: bool = True):
        """
        Parameters
        ----------
        n_splits : int
            Number of folds
        shuffle : bool
            Whether to shuffle before splitting
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Whether to print progress
        """
        super().__init__(verbose)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> Generator[CVSplit, None, None]:
        """
        Generate stratified k-fold splits.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        subject_ids : np.ndarray, optional
            Not used, but kept for API consistency
        
        Yields
        ------
        CVSplit
            Train/test split
        """
        self._print(f"\n[StratifiedKFoldCV] {self.n_splits}-fold")
        
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            # Compute class distribution
            train_dist = np.bincount(y[train_idx])
            test_dist = np.bincount(y[test_idx])
            
            self._print(
                f"  Fold {fold+1}/{self.n_splits}: "
                f"Train={len(train_idx)} (class dist: {train_dist}), "
                f"Test={len(test_idx)} (class dist: {test_dist})"
            )
            
            yield CVSplit(
                train_idx=train_idx,
                test_idx=test_idx,
                fold=fold
            )
    
    def get_n_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> int:
        """Get number of splits."""
        return self.n_splits


class GroupKFoldCV(BaseCrossValidator):
    """
    Group K-Fold Cross-Validation.
    
    Ensures samples from same subject never appear in both train and test.
    Alternative to SubjectLevelLOOCV when LOO is too expensive.
    """
    
    def __init__(self, n_splits: int = 5, verbose: bool = True):
        """
        Parameters
        ----------
        n_splits : int
            Number of folds
        verbose : bool
            Whether to print progress
        """
        super().__init__(verbose)
        self.n_splits = n_splits
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> Generator[CVSplit, None, None]:
        """
        Generate group k-fold splits.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        subject_ids : np.ndarray
            Subject identifiers - REQUIRED
        
        Yields
        ------
        CVSplit
            Train/test split
        """
        if subject_ids is None:
            raise ValueError("subject_ids is required for GroupKFoldCV")
        
        self._print(f"\n[GroupKFoldCV] {self.n_splits}-fold")
        
        gkf = GroupKFold(n_splits=self.n_splits)
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=subject_ids)):
            train_subjects = list(np.unique(subject_ids[train_idx]))
            test_subjects = list(np.unique(subject_ids[test_idx]))
            
            self._print(
                f"  Fold {fold+1}/{self.n_splits}: "
                f"Train={len(train_idx)} samples ({len(train_subjects)} subjects), "
                f"Test={len(test_idx)} samples ({len(test_subjects)} subjects)"
            )
            
            yield CVSplit(
                train_idx=train_idx,
                test_idx=test_idx,
                fold=fold,
                train_subjects=train_subjects,
                test_subjects=test_subjects
            )
    
    def get_n_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> int:
        """Get number of splits."""
        return self.n_splits


class TimeSeriesSplitCV(BaseCrossValidator):
    """
    Time Series Split for temporal validation.
    
    Ensures training always precedes testing (no future information leakage).
    Useful when you have temporal ordering in your data.
    """
    
    def __init__(self, n_splits: int = 5, verbose: bool = True):
        """
        Parameters
        ----------
        n_splits : int
            Number of folds
        verbose : bool
            Whether to print progress
        """
        super().__init__(verbose)
        self.n_splits = n_splits
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> Generator[CVSplit, None, None]:
        """
        Generate time series splits.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (assumed to be temporally ordered)
        y : np.ndarray
            Labels
        subject_ids : np.ndarray, optional
            Not used, but kept for API consistency
        
        Yields
        ------
        CVSplit
            Train/test split
        """
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)
        
        self._print(f"\n[TimeSeriesSplitCV] {self.n_splits}-fold (expanding window)")
        
        for fold in range(self.n_splits):
            # Expanding training window
            train_end = (fold + 1) * test_size
            test_start = train_end
            test_end = test_start + test_size
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, min(test_end, n_samples))
            
            self._print(
                f"  Fold {fold+1}/{self.n_splits}: "
                f"Train=[0:{train_end}] ({len(train_idx)} samples), "
                f"Test=[{test_start}:{test_end}] ({len(test_idx)} samples)"
            )
            
            yield CVSplit(
                train_idx=train_idx,
                test_idx=test_idx,
                fold=fold
            )
    
    def get_n_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ) -> int:
        """Get number of splits."""
        return self.n_splits


# ============================================================================
# Cross-Validator Factory
# ============================================================================

class CrossValidatorFactory:
    """Factory for creating cross-validators."""
    
    @staticmethod
    def create(
        strategy: str,
        subject_ids: Optional[np.ndarray] = None,
        **kwargs
    ) -> BaseCrossValidator:
        """
        Create a cross-validator.
        
        Parameters
        ----------
        strategy : str
            CV strategy: 'subject_loo', 'sample_loo', 'stratified_kfold',
            'group_kfold', 'timeseries'
        subject_ids : np.ndarray, optional
            Subject identifiers (required for subject-level strategies)
        **kwargs
            Additional parameters for the validator
        
        Returns
        -------
        BaseCrossValidator
            Configured cross-validator
        """
        strategy = strategy.lower()
        
        if strategy == "subject_loo":
            if subject_ids is None:
                raise ValueError("subject_ids required for subject_loo")
            return SubjectLevelLOOCV(**kwargs)
        
        elif strategy == "sample_loo":
            return SampleLevelLOOCV(**kwargs)
        
        elif strategy == "stratified_kfold":
            return StratifiedKFoldCV(**kwargs)
        
        elif strategy == "group_kfold":
            if subject_ids is None:
                raise ValueError("subject_ids required for group_kfold")
            return GroupKFoldCV(**kwargs)
        
        elif strategy == "timeseries":
            return TimeSeriesSplitCV(**kwargs)
        
        else:
            raise ValueError(f"Unknown CV strategy: {strategy}")
    
    @staticmethod
    def auto_select(
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        **kwargs
    ) -> BaseCrossValidator:
        """
        Automatically select appropriate CV strategy.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        subject_ids : np.ndarray, optional
            Subject identifiers
        **kwargs
            Additional parameters
        
        Returns
        -------
        BaseCrossValidator
            Appropriate cross-validator
        """
        n_samples = len(X)
        
        # If we have subject IDs
        if subject_ids is not None:
            unique_subjects = np.unique(subject_ids)
            n_subjects = len(unique_subjects)
            
            # If each sample is from different subject → sample LOO
            if n_subjects == n_samples:
                print("[Auto CV] Detected: 1 sample per subject → SampleLevelLOOCV")
                return SampleLevelLOOCV(**kwargs)
            
            # If we have windowed/augmented data → subject LOO
            else:
                print(f"[Auto CV] Detected: {n_samples} samples from {n_subjects} subjects → SubjectLevelLOOCV")
                return SubjectLevelLOOCV(**kwargs)
        
        # No subject IDs provided
        else:
            # Small dataset → LOO
            if n_samples <= 100:
                print(f"[Auto CV] Small dataset ({n_samples} samples) → SampleLevelLOOCV")
                return SampleLevelLOOCV(**kwargs)
            
            # Large dataset → 5-fold
            else:
                print(f"[Auto CV] Large dataset ({n_samples} samples) → StratifiedKFoldCV")
                return StratifiedKFoldCV(n_splits=5, **kwargs)


# ============================================================================
# Utilities
# ============================================================================

def compute_cv_statistics(fold_results: List[Dict[str, Any]]) -> Tuple[Dict, Dict]:
    """
    Compute mean and std across folds.
    
    Parameters
    ----------
    fold_results : List[Dict]
        Results from each fold
    
    Returns
    -------
    mean_metrics : Dict
        Mean of each metric
    std_metrics : Dict
        Standard deviation of each metric
    """
    if not fold_results:
        return {}, {}
    
    # Collect metrics across folds
    metrics_by_fold = defaultdict(list)
    for fold_result in fold_results:
        for metric, value in fold_result.items():
            if isinstance(value, (int, float)):
                metrics_by_fold[metric].append(value)
    
    # Compute statistics
    mean_metrics = {
        metric: float(np.mean(values))
        for metric, values in metrics_by_fold.items()
    }
    
    std_metrics = {
        metric: float(np.std(values))
        for metric, values in metrics_by_fold.items()
    }
    
    return mean_metrics, std_metrics


def print_cv_summary(cv_results: CVResults):
    """
    Print summary of CV results.
    
    Parameters
    ----------
    cv_results : CVResults
        Cross-validation results
    """
    print("\n" + "="*60)
    print("Cross-Validation Summary")
    print("="*60)
    
    if cv_results.mean_metrics:
        print("\nMean Metrics (across folds):")
        for metric, value in cv_results.mean_metrics.items():
            std = cv_results.std_metrics.get(metric, 0)
            print(f"  {metric}: {value:.4f} ± {std:.4f}")
    
    print(f"\nTotal Predictions: {len(cv_results.y_true)}")
    print(f"Class 0: {np.sum(cv_results.y_true == 0)}")
    print(f"Class 1: {np.sum(cv_results.y_true == 1)}")
    
    if cv_results.subject_ids is not None:
        n_subjects = len(np.unique(cv_results.subject_ids))
        print(f"Unique Subjects: {n_subjects}")
    
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_subjects = 10
    windows_per_subject = 5
    n_samples = n_subjects * windows_per_subject
    
    X = np.random.randn(n_samples, 100, 18)
    y = np.random.randint(0, 2, n_samples)
    subject_ids = np.repeat([f"subj_{i}" for i in range(n_subjects)], windows_per_subject)
    
    print("Sample Data:")
    print(f"  Samples: {n_samples}")
    print(f"  Subjects: {n_subjects}")
    print(f"  Windows per subject: {windows_per_subject}")
    
    # Test different CV strategies
    print("\n" + "="*60)
    print("Testing CV Strategies")
    print("="*60)
    
    # 1. Subject-level LOO
    cv = SubjectLevelLOOCV(verbose=True)
    n_splits = cv.get_n_splits(X, y, subject_ids)
    print(f"Number of splits: {n_splits}")
    
    # 2. Auto-select
    cv = CrossValidatorFactory.auto_select(X, y, subject_ids)
    
    # 3. Iterate through splits
    for i, split in enumerate(cv.split(X, y, subject_ids)):
        if i >= 2:  # Only show first 2 splits
            break
        print(f"\nSplit {i+1}:")
        print(f"  Train subjects: {split.train_subjects[:3]}...")
        print(f"  Test subjects: {split.test_subjects}")