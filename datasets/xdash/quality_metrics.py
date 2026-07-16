"""
datasets/xdash/quality_metrics.py
==================================
Quality assessment for XDash trial data.

QualityMetrics  — movement and temporal quality checks
DataValidator   — array-level integrity validation
"""

import numpy as np
import pandas as pd

# Thresholds applied by passes_quality_thresholds()
MINIMUM_FRAMES = 25
MINIMUM_QUALITY_SCORE = 100.0   # percent (0–100); set < 100 to allow partial NaN
MINIMUM_DURATION = 0.1          # seconds


class QualityMetrics:
    """Movement and temporal quality checks for a single trial."""

    @staticmethod
    def calculate_movement_quality(movement_df: pd.DataFrame) -> dict:
        """Return quality metrics for each sensor channel plus an overall score."""
        sensor_cols = [c for c in movement_df.columns if c != "TimeElapsed"]
        n = len(movement_df)

        per_channel = {}
        for col in sensor_cols:
            missing_pct = movement_df[col].isna().sum() / n * 100
            per_channel[col] = 100.0 - missing_pct

        overall = float(np.mean(list(per_channel.values()))) if per_channel else 0.0

        return {
            "total_frames": n,
            "missing_frames_pct": 0.0,
            "sensor_quality_scores": per_channel,
            "overall_quality_score": overall,
        }

    @staticmethod
    def calculate_temporal_quality(time_array: np.ndarray) -> dict:
        """Return temporal consistency metrics from a 1-D time array."""
        if len(time_array) < 2:
            return {
                "sampling_rate": None,
                "mean_frame_interval": None,
                "frame_interval_std": None,
                "duration": 0.0,
                "frame_interval_consistency": 0.0,
            }

        intervals = np.diff(time_array)
        duration = float(time_array[-1] - time_array[0])
        mean_interval = float(np.mean(intervals))

        return {
            "sampling_rate": len(time_array) / duration if duration > 0 else None,
            "mean_frame_interval": mean_interval,
            "frame_interval_std": float(np.std(intervals)),
            "duration": duration,
            "frame_interval_consistency": (
                1.0 - float(np.std(intervals)) / mean_interval
                if mean_interval > 0 else 0.0
            ),
        }

    @staticmethod
    def passes_quality_thresholds(quality: dict, temporal: dict) -> tuple[bool, list[str]]:
        """Check whether a trial clears all quality thresholds.

        Returns (passes, reasons_failed).
        """
        reasons: list[str] = []

        if quality["overall_quality_score"] < MINIMUM_QUALITY_SCORE:
            reasons.append(
                f"quality {quality['overall_quality_score']:.1f}% < {MINIMUM_QUALITY_SCORE}%"
            )
        if quality["total_frames"] < MINIMUM_FRAMES:
            reasons.append(
                f"{quality['total_frames']} frames < {MINIMUM_FRAMES} minimum"
            )
        if temporal.get("duration", 0.0) < MINIMUM_DURATION:
            reasons.append(
                f"duration {temporal.get('duration', 0.0):.3f}s < {MINIMUM_DURATION}s"
            )

        return len(reasons) == 0, reasons

    @staticmethod
    def generate_quality_report(
        movement_df: pd.DataFrame,
        time_array: np.ndarray | None = None,
    ) -> dict:
        """Return a complete quality report for one trial."""
        movement_quality = QualityMetrics.calculate_movement_quality(movement_df)
        temporal_quality = (
            QualityMetrics.calculate_temporal_quality(time_array)
            if time_array is not None
            else {}
        )
        passes, reasons = QualityMetrics.passes_quality_thresholds(
            movement_quality, temporal_quality
        )
        return {
            "movement_quality": movement_quality,
            "temporal_quality": temporal_quality,
            "passes_quality_check": passes,
            "failure_reasons": reasons,
        }


class DataValidator:
    """Array-level integrity checks for a single trial."""

    @staticmethod
    def validate_trial_data(
        movement_array: np.ndarray,
        time_array: np.ndarray,
    ) -> tuple[bool, list[str]]:
        """Validate movement array + time array for common issues.

        Returns (is_valid, issues).
        """
        issues: list[str] = []

        if movement_array.shape[0] != len(time_array):
            issues.append(
                f"shape mismatch: movement rows {movement_array.shape[0]} "
                f"!= time length {len(time_array)}"
            )

        nan_count = int(np.isnan(movement_array).sum())
        if nan_count:
            issues.append(f"{nan_count / movement_array.size * 100:.1f}% NaN in movement")

        inf_count = int(np.isinf(movement_array).sum())
        if inf_count:
            issues.append(f"{inf_count} inf values in movement")

        if not np.all(np.diff(time_array) >= 0):
            issues.append("time array not monotonically increasing")

        return len(issues) == 0, issues
