"""
Signal transformation utilities for time-series preprocessing.

Includes normalization, filtering, augmentation, and feature extraction methods.
"""
import numpy as np
import torch
from typing import Union, List, Tuple, Optional
from scipy.signal import butter, filtfilt, resample
from scipy.interpolate import interp1d
from scipy.stats import linregress


# ============================================================================
# Type Definitions
# ============================================================================

TensorLike = Union[np.ndarray, torch.Tensor]


# ============================================================================
# Conversion Utilities
# ============================================================================

def to_numpy(data: TensorLike) -> np.ndarray:
    """Convert tensor-like data to numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def to_torch(data: TensorLike) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    if isinstance(data, torch.Tensor):
        return data
    return torch.from_numpy(np.asarray(data)).float()


# ============================================================================
# Normalization Transforms
# ============================================================================

class ZScoreNormalize:
    """Z-score normalization (mean=0, std=1)."""
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Parameters
        ----------
        epsilon : float
            Small constant for numerical stability
        """
        self.epsilon = epsilon
        self.mean = None
        self.std = None
    
    def fit(self, data: TensorLike):
        """Fit normalization parameters."""
        data_np = to_numpy(data)
        self.mean = np.mean(data_np, axis=0, keepdims=True)
        self.std = np.std(data_np, axis=0, keepdims=True)
        self.std = np.maximum(self.std, self.epsilon)
        return self
    
    def transform(self, data: TensorLike) -> np.ndarray:
        """Apply normalization."""
        if self.mean is None:
            raise RuntimeError("Must call fit() before transform()")
        data_np = to_numpy(data)
        return (data_np - self.mean) / self.std
    
    def fit_transform(self, data: TensorLike) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: TensorLike) -> np.ndarray:
        """Reverse normalization."""
        if self.mean is None:
            raise RuntimeError("Must call fit() before inverse_transform()")
        data_np = to_numpy(data)
        return data_np * self.std + self.mean


class MinMaxNormalize:
    """Min-max normalization to [0, 1] range."""
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        """
        Parameters
        ----------
        feature_range : tuple
            Target range (min, max)
        """
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None
    
    def fit(self, data: TensorLike):
        """Fit normalization parameters."""
        data_np = to_numpy(data)
        self.data_min = np.min(data_np, axis=0, keepdims=True)
        self.data_max = np.max(data_np, axis=0, keepdims=True)
        return self
    
    def transform(self, data: TensorLike) -> np.ndarray:
        """Apply normalization."""
        if self.data_min is None:
            raise RuntimeError("Must call fit() before transform()")
        
        data_np = to_numpy(data)
        data_range = self.data_max - self.data_min
        data_range = np.maximum(data_range, 1e-8)
        
        # Scale to [0, 1]
        normalized = (data_np - self.data_min) / data_range
        
        # Scale to target range
        min_val, max_val = self.feature_range
        return normalized * (max_val - min_val) + min_val
    
    def fit_transform(self, data: TensorLike) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)


class RobustScaler:
    """Robust scaling using median and IQR (resistant to outliers)."""
    
    def __init__(self):
        self.median = None
        self.iqr = None
    
    def fit(self, data: TensorLike):
        """Fit normalization parameters."""
        data_np = to_numpy(data)
        self.median = np.median(data_np, axis=0, keepdims=True)
        q75 = np.percentile(data_np, 75, axis=0, keepdims=True)
        q25 = np.percentile(data_np, 25, axis=0, keepdims=True)
        self.iqr = np.maximum(q75 - q25, 1e-8)
        return self
    
    def transform(self, data: TensorLike) -> np.ndarray:
        """Apply normalization."""
        if self.median is None:
            raise RuntimeError("Must call fit() before transform()")
        data_np = to_numpy(data)
        return (data_np - self.median) / self.iqr
    
    def fit_transform(self, data: TensorLike) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)


# ============================================================================
# Filtering Transforms
# ============================================================================

class LowPassFilter:
    """Butterworth low-pass filter."""
    
    def __init__(self, cutoff_freq: float, sampling_rate: float, order: int = 4):
        """
        Parameters
        ----------
        cutoff_freq : float
            Cutoff frequency in Hz
        sampling_rate : float
            Sampling rate in Hz
        order : int
            Filter order
        """
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.order = order
        
        # Design filter
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)
    
    def transform(self, data: TensorLike) -> np.ndarray:
        """Apply filter to data."""
        data_np = to_numpy(data)
        
        # Apply filter along time axis
        if data_np.ndim == 1:
            return filtfilt(self.b, self.a, data_np)
        else:
            # Filter each channel separately
            filtered = np.zeros_like(data_np)
            for i in range(data_np.shape[1]):
                filtered[:, i] = filtfilt(self.b, self.a, data_np[:, i])
            return filtered


class HighPassFilter:
    """Butterworth high-pass filter."""
    
    def __init__(self, cutoff_freq: float, sampling_rate: float, order: int = 4):
        """
        Parameters
        ----------
        cutoff_freq : float
            Cutoff frequency in Hz
        sampling_rate : float
            Sampling rate in Hz
        order : int
            Filter order
        """
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.order = order
        
        # Design filter
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        self.b, self.a = butter(order, normal_cutoff, btype='high', analog=False)
    
    def transform(self, data: TensorLike) -> np.ndarray:
        """Apply filter to data."""
        data_np = to_numpy(data)
        
        if data_np.ndim == 1:
            return filtfilt(self.b, self.a, data_np)
        else:
            filtered = np.zeros_like(data_np)
            for i in range(data_np.shape[1]):
                filtered[:, i] = filtfilt(self.b, self.a, data_np[:, i])
            return filtered


class BandPassFilter:
    """Butterworth band-pass filter."""
    
    def __init__(
        self,
        low_freq: float,
        high_freq: float,
        sampling_rate: float,
        order: int = 4
    ):
        """
        Parameters
        ----------
        low_freq : float
            Low cutoff frequency in Hz
        high_freq : float
            High cutoff frequency in Hz
        sampling_rate : float
            Sampling rate in Hz
        order : int
            Filter order
        """
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.sampling_rate = sampling_rate
        self.order = order
        
        # Design filter
        nyquist = 0.5 * sampling_rate
        low = low_freq / nyquist
        high = high_freq / nyquist
        self.b, self.a = butter(order, [low, high], btype='band', analog=False)
    
    def transform(self, data: TensorLike) -> np.ndarray:
        """Apply filter to data."""
        data_np = to_numpy(data)
        
        if data_np.ndim == 1:
            return filtfilt(self.b, self.a, data_np)
        else:
            filtered = np.zeros_like(data_np)
            for i in range(data_np.shape[1]):
                filtered[:, i] = filtfilt(self.b, self.a, data_np[:, i])
            return filtered


# ============================================================================
# Resampling Transforms
# ============================================================================

class Downsample:
    """Downsample time-series to lower sampling rate."""
    
    def __init__(self, target_rate: int, original_rate: int):
        """
        Parameters
        ----------
        target_rate : int
            Target sampling rate in Hz
        original_rate : int
            Original sampling rate in Hz
        """
        self.target_rate = target_rate
        self.original_rate = original_rate
        self.ratio = original_rate / target_rate
    
    def transform(self, data: TensorLike) -> np.ndarray:
        """Downsample data."""
        data_np = to_numpy(data)
        T_original = data_np.shape[0]
        T_new = int(T_original / self.ratio)
        
        # Use scipy's resample for anti-aliasing
        if data_np.ndim == 1:
            return resample(data_np, T_new)
        else:
            resampled = np.zeros((T_new, data_np.shape[1]))
            for i in range(data_np.shape[1]):
                resampled[:, i] = resample(data_np[:, i], T_new)
            return resampled


class Interpolate:
    """Interpolate time-series to new length."""
    
    def __init__(self, target_length: int, kind: str = 'linear'):
        """
        Parameters
        ----------
        target_length : int
            Target sequence length
        kind : str
            Interpolation type ('linear', 'cubic', 'quadratic')
        """
        self.target_length = target_length
        self.kind = kind
    
    def transform(self, data: TensorLike) -> np.ndarray:
        """Interpolate data."""
        data_np = to_numpy(data)
        T_original = data_np.shape[0]
        
        # Original time points
        t_old = np.linspace(0, 1, T_original)
        # New time points
        t_new = np.linspace(0, 1, self.target_length)
        
        if data_np.ndim == 1:
            f = interp1d(t_old, data_np, kind=self.kind)
            return f(t_new)
        else:
            interpolated = np.zeros((self.target_length, data_np.shape[1]))
            for i in range(data_np.shape[1]):
                f = interp1d(t_old, data_np[:, i], kind=self.kind)
                interpolated[:, i] = f(t_new)
            return interpolated


# ============================================================================
# Data Augmentation Transforms
# ============================================================================

class TimeJitter:
    """Add random jitter to signal values."""
    
    def __init__(self, sigma: float = 0.01):
        """
        Parameters
        ----------
        sigma : float
            Standard deviation of Gaussian noise
        """
        self.sigma = sigma
    
    def transform(self, data: TensorLike, seed: Optional[int] = None) -> np.ndarray:
        """Add jitter to data."""
        if seed is not None:
            np.random.seed(seed)
        
        data_np = to_numpy(data)
        noise = np.random.normal(0, self.sigma, data_np.shape)
        return data_np + noise


class TimeWarping:
    """Smooth time warping augmentation."""
    
    def __init__(self, sigma: float = 0.2, knot: int = 4):
        """
        Parameters
        ----------
        sigma : float
            Standard deviation of warping
        knot : int
            Number of knot points for warping
        """
        self.sigma = sigma
        self.knot = knot
    
    def transform(self, data: TensorLike, seed: Optional[int] = None) -> np.ndarray:
        """Apply time warping to data."""
        if seed is not None:
            np.random.seed(seed)
        
        data_np = to_numpy(data)
        T = data_np.shape[0]
        
        # Generate smooth warping curve
        orig_steps = np.arange(T)
        random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=(self.knot + 2,))
        
        # Ensure positive warps
        random_warps = np.maximum(random_warps, 0.1)
        
        # Create warp points
        warp_steps = np.linspace(0, T - 1, num=self.knot + 2)
        
        # Interpolate to get smooth warping
        f = interp1d(warp_steps, random_warps, kind='cubic')
        time_warp = f(orig_steps)
        
        # Cumulative sum for time indices
        time_warp = np.cumsum(time_warp)
        time_warp = (T - 1) * (time_warp - time_warp[0]) / (time_warp[-1] - time_warp[0])
        
        # Apply warping
        if data_np.ndim == 1:
            return np.interp(orig_steps, time_warp, data_np)
        else:
            warped = np.zeros_like(data_np)
            for i in range(data_np.shape[1]):
                warped[:, i] = np.interp(orig_steps, time_warp, data_np[:, i])
            return warped


class MagnitudeWarping:
    """Warp signal magnitudes."""
    
    def __init__(self, sigma: float = 0.2, knot: int = 4):
        """
        Parameters
        ----------
        sigma : float
            Standard deviation of magnitude warping
        knot : int
            Number of knot points
        """
        self.sigma = sigma
        self.knot = knot
    
    def transform(self, data: TensorLike, seed: Optional[int] = None) -> np.ndarray:
        """Apply magnitude warping."""
        if seed is not None:
            np.random.seed(seed)
        
        data_np = to_numpy(data)
        T = data_np.shape[0]
        
        # Generate smooth magnitude curve
        orig_steps = np.arange(T)
        random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=(self.knot + 2,))
        
        warp_steps = np.linspace(0, T - 1, num=self.knot + 2)
        f = interp1d(warp_steps, random_warps, kind='cubic')
        magnitude_warp = f(orig_steps)
        
        # Apply warping
        if data_np.ndim == 1:
            return data_np * magnitude_warp
        else:
            return data_np * magnitude_warp[:, np.newaxis]


class RandomSlice:
    """Extract random temporal slice."""
    
    def __init__(self, min_length: int):
        """
        Parameters
        ----------
        min_length : int
            Minimum slice length
        """
        self.min_length = min_length
    
    def transform(self, data: TensorLike, seed: Optional[int] = None) -> np.ndarray:
        """Extract random slice."""
        if seed is not None:
            np.random.seed(seed)
        
        data_np = to_numpy(data)
        T = data_np.shape[0]
        
        if T <= self.min_length:
            return data_np
        
        # Random slice length
        slice_length = np.random.randint(self.min_length, T + 1)
        
        # Random start position
        max_start = T - slice_length
        start_idx = np.random.randint(0, max_start + 1)
        
        return data_np[start_idx:start_idx + slice_length]


# ============================================================================
# Feature Extraction
# ============================================================================

class WindowedFeatureExtractor:
    """Extract statistical features from sliding windows."""
    
    def __init__(self, window_size: int):
        """
        Parameters
        ----------
        window_size : int
            Size of sliding window
        """
        self.window_size = window_size
    
    def transform(self, data: TensorLike) -> np.ndarray:
        """
        Extract features from data.
        
        Returns feature vector with statistics per window and channel:
        [mean, std, slope, max, min] for each window and each channel
        """
        data_np = to_numpy(data)
        T, C = data_np.shape if data_np.ndim == 2 else (data_np.shape[0], 1)
        
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)
        
        features = []
        
        for c in range(C):
            signal = data_np[:, c]
            
            for start in range(0, T, self.window_size):
                window = signal[start:start + self.window_size]
                
                if window.size == 0:
                    continue
                
                # Compute slope
                if window.size >= 2:
                    slope = linregress(np.arange(window.size), window).slope
                else:
                    slope = 0.0
                
                # Extract features
                features.extend([
                    float(np.mean(window)),
                    float(np.std(window)),
                    float(slope),
                    float(np.max(window)),
                    float(np.min(window))
                ])
        
        return np.array(features, dtype=np.float32)


# ============================================================================
# Composite Transforms
# ============================================================================

class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List):
        """
        Parameters
        ----------
        transforms : List
            List of transform objects
        """
        self.transforms = transforms
    
    def transform(self, data: TensorLike) -> np.ndarray:
        """Apply all transforms sequentially."""
        result = data
        for t in self.transforms:
            result = t.transform(result)
        return result


# Example usage
if __name__ == "__main__":
    # Test transforms
    test_data = np.random.randn(1000, 18)
    
    # Normalization
    scaler = ZScoreNormalize()
    normalized = scaler.fit_transform(test_data)
    print(f"Normalized shape: {normalized.shape}")
    print(f"Mean: {normalized.mean(axis=0)[:3]}")
    print(f"Std: {normalized.std(axis=0)[:3]}")
    
    # Filtering
    lpf = LowPassFilter(cutoff_freq=10, sampling_rate=50)
    filtered = lpf.transform(test_data)
    print(f"\nFiltered shape: {filtered.shape}")
    
    # Augmentation
    jitter = TimeJitter(sigma=0.01)
    jittered = jitter.transform(test_data)
    print(f"\nJittered shape: {jittered.shape}")
    
    # Composite
    pipeline = Compose([
        LowPassFilter(cutoff_freq=10, sampling_rate=50),
        ZScoreNormalize(),
        TimeJitter(sigma=0.005)
    ])
    transformed = pipeline.transform(test_data)
    print(f"\nPipeline output shape: {transformed.shape}")