"""
datasets/xdash/outlier_detection.py
=====================================
Outlier detection for XDash trial durations.

OutlierDetector supports four methods:
  iqr           — Interquartile Range (robust, recommended)
  zscore        — Z-score (assumes normality)
  percentile    — Simple threshold-based
  task_specific — Domain-knowledge duration caps per task
"""

import numpy as np
from typing import Dict, List, Tuple


# Default thresholds (can be overridden per-call)
IQR_MULTIPLIER = 3.0       # 1.5 = mild outliers, 3.0 = extreme only
ZSCORE_THRESHOLD = 3.0
PERCENTILE_LOWER = 1
PERCENTILE_UPPER = 100


class OutlierDetector:
    """Detect and filter outlier trials based on duration."""

    @staticmethod
    def detect_iqr_outliers(
        durations: np.ndarray,
        multiplier: float = IQR_MULTIPLIER,
    ) -> Tuple[np.ndarray, Dict]:
        """IQR-based outlier detection."""
        q1, q3 = np.percentile(durations, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        mask = (durations < lo) | (durations > hi)
        return mask, {
            "method": "IQR",
            "Q1": q1, "Q3": q3, "IQR": iqr,
            "lower_bound": lo, "upper_bound": hi,
            "n_outliers": int(mask.sum()),
            "n_lower_outliers": int((durations < lo).sum()),
            "n_upper_outliers": int((durations > hi).sum()),
        }

    @staticmethod
    def detect_zscore_outliers(
        durations: np.ndarray,
        threshold: float = ZSCORE_THRESHOLD,
    ) -> Tuple[np.ndarray, Dict]:
        """Z-score outlier detection."""
        mean, std = np.mean(durations), np.std(durations)
        z = np.abs((durations - mean) / std) if std > 0 else np.zeros_like(durations)
        mask = z > threshold
        return mask, {
            "method": "Z-score",
            "mean": float(mean), "std": float(std),
            "threshold": threshold,
            "n_outliers": int(mask.sum()),
            "max_zscore": float(z.max()),
        }

    @staticmethod
    def detect_percentile_outliers(
        durations: np.ndarray,
        lower_percentile: float = PERCENTILE_LOWER,
        upper_percentile: float = PERCENTILE_UPPER,
    ) -> Tuple[np.ndarray, Dict]:
        """Percentile threshold outlier detection."""
        lo = np.percentile(durations, lower_percentile)
        hi = np.percentile(durations, upper_percentile)
        mask = (durations < lo) | (durations > hi)
        return mask, {
            "method": "Percentile",
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "lower_bound": float(lo),
            "upper_bound": float(hi),
            "n_outliers": int(mask.sum()),
        }

    @staticmethod
    def detect_task_specific_outliers(
        durations: np.ndarray,
        task_numbers: np.ndarray,
        task_max_duration: Dict[int, float | None],
    ) -> Tuple[np.ndarray, Dict]:
        """Flag trials that exceed per-task maximum durations.

        task_max_duration: {task_num: max_seconds | None}  (None = no cap)
        """
        mask = np.zeros(len(durations), dtype=bool)
        per_task = {}

        for task_num, max_dur in task_max_duration.items():
            task_mask = task_numbers == task_num
            if max_dur is not None:
                outliers = (durations > max_dur) & task_mask
                mask |= outliers
            else:
                outliers = np.zeros(len(durations), dtype=bool)

            per_task[task_num] = {
                "max_allowed": max_dur,
                "n_trials": int(task_mask.sum()),
                "n_outliers": int(outliers.sum()),
                "actual_max": float(durations[task_mask].max()) if task_mask.any() else 0.0,
            }

        return mask, {"method": "Task-specific limits", "per_task": per_task, "total_outliers": int(mask.sum())}

    @classmethod
    def detect_outliers(
        cls,
        trials_data: List[Dict],
        method: str = "iqr",
        task_max_duration: Dict[int, float | None] | None = None,
    ) -> Tuple[List[bool], Dict]:
        """Detect outliers across a list of trial dicts.

        Each trial dict must have 'duration_seconds' and 'task_number'.
        Returns (is_outlier list, stats dict).
        """
        durations = np.array([t["duration_seconds"] for t in trials_data])
        task_numbers = np.array([t["task_number"] for t in trials_data])

        if method == "iqr":
            mask, stats = cls.detect_iqr_outliers(durations)
        elif method == "zscore":
            mask, stats = cls.detect_zscore_outliers(durations)
        elif method == "percentile":
            mask, stats = cls.detect_percentile_outliers(durations)
        elif method == "task_specific":
            if task_max_duration is None:
                raise ValueError("task_max_duration required for method='task_specific'")
            mask, stats = cls.detect_task_specific_outliers(durations, task_numbers, task_max_duration)
        else:
            raise ValueError(f"Unknown outlier detection method: {method!r}")

        return mask.tolist(), stats

    @staticmethod
    def print_outlier_report(
        trials_data: List[Dict],
        is_outlier: List[bool],
        stats: Dict,
        task_names: Dict[int, str] | None = None,
    ) -> None:
        """Print a human-readable outlier detection report."""
        outliers = [t for t, flag in zip(trials_data, is_outlier) if flag]

        print(f"\n{'='*80}")
        print("OUTLIER DETECTION REPORT")
        print(f"{'='*80}")
        print(f"  Method   : {stats['method']}")
        print(f"  Total    : {len(trials_data)}")
        print(f"  Outliers : {sum(is_outlier)} ({sum(is_outlier)/len(trials_data)*100:.1f}%)")

        if stats["method"] == "IQR":
            print(f"\n  IQR bounds: [{stats['lower_bound']:.2f}s, {stats['upper_bound']:.2f}s]")
            print(f"  Q1={stats['Q1']:.2f}s  Q3={stats['Q3']:.2f}s  IQR={stats['IQR']:.2f}s")
            print(f"  Below lower: {stats['n_lower_outliers']}  Above upper: {stats['n_upper_outliers']}")

        elif stats["method"] == "Task-specific limits":
            for task_num, ts in stats["per_task"].items():
                name = (task_names or {}).get(task_num, f"Task {task_num}")
                print(f"  {name}: max={ts['max_allowed']}s  "
                      f"actual_max={ts['actual_max']:.2f}s  "
                      f"outliers={ts['n_outliers']}/{ts['n_trials']}")

        if outliers:
            print(f"\nOutlier trials (first 20 of {len(outliers)}):")
            print(f"  {'ID':<30} {'Subject':<10} {'Task':<22} {'Duration':<12} Reason")
            print("  " + "-" * 90)
            for t in outliers[:20]:
                if stats["method"] == "IQR":
                    reason = (
                        f"< {stats['lower_bound']:.2f}s"
                        if t["duration_seconds"] < stats["lower_bound"]
                        else f"> {stats['upper_bound']:.2f}s"
                    )
                else:
                    reason = "statistical outlier"
                print(f"  {t.get('window_id','?'):<30} {t.get('subject_id','?'):<10} "
                      f"{t.get('task_type','?'):<22} {t['duration_seconds']:.2f}s{'':<8} {reason}")
            if len(outliers) > 20:
                print(f"\n  … and {len(outliers) - 20} more")

        print(f"\n{'='*80}\n")
