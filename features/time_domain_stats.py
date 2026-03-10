"""
Time Domain Statistical Features
Comprehensive statistical characterization of signals
"""

from scipy import stats
import numpy as np


def extract_time_domain_features(signal):
    """
    Extract comprehensive statistical features from signal.
    
    Args:
        signal: (T,) - single channel time series
    
    Returns:
        dict: 19 statistical features
    
    Features extracted:
        - Central tendency (mean, median)
        - Dispersion (std, variance, range, IQR, MAD)
        - Extremes (min, max, percentiles)
        - Shape (kurtosis, skewness)
        - Energy (RMS, energy, SMA)
        - Zero crossing rate
        - Autocorrelation
    """
    features = {
        # ===== CENTRAL TENDENCY =====
        'mean': np.mean(signal),
        'median': np.median(signal),
        
        # ===== DISPERSION =====
        'std': np.std(signal),
        'variance': np.var(signal),
        'range': np.ptp(signal),  # peak-to-peak
        'iqr': stats.iqr(signal),
        'mad': np.mean(np.abs(signal - np.mean(signal))),  # mean absolute deviation
        
        # ===== EXTREMES =====
        'min': np.min(signal),
        'max': np.max(signal),
        'percentile_25': np.percentile(signal, 25),
        'percentile_75': np.percentile(signal, 75),
        'percentile_95': np.percentile(signal, 95),
        
        # ===== SHAPE =====
        'kurtosis': stats.kurtosis(signal),
        'skewness': stats.skew(signal),
        
        # ===== SIGNAL ENERGY =====
        'rms': np.sqrt(np.mean(signal**2)),
        'energy': np.sum(signal**2),
        'sma': np.sum(np.abs(signal)),  # signal magnitude area
        
        # ===== ZERO CROSSINGS =====
        'zero_crossing_rate': np.sum(np.diff(np.sign(signal)) != 0) / len(signal),
        
        # ===== AUTOCORRELATION =====
        'autocorr_lag1': np.corrcoef(signal[:-1], signal[1:])[0, 1] if len(signal) > 1 else 0
    }
    
    return features


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    print("="*70)
    print("Testing Time Domain Statistical Features")
    print("="*70)
    
    # Create test signal
    T = 500
    signal = np.random.randn(T) + np.sin(np.linspace(0, 10*np.pi, T))
    
    print(f"\nTest signal:")
    print(f"  Length: {T} samples")
    print(f"  Mean: {np.mean(signal):.4f}")
    print(f"  Std: {np.std(signal):.4f}")
    
    # Extract features
    print("\nExtracting time domain features...")
    features = extract_time_domain_features(signal)
    
    print(f"\n✓ Extracted {len(features)} features")
    
    # Display by category
    print("\n" + "="*70)
    print("CENTRAL TENDENCY")
    print("="*70)
    print(f"  Mean:   {features['mean']:.6f}")
    print(f"  Median: {features['median']:.6f}")
    
    print("\n" + "="*70)
    print("DISPERSION")
    print("="*70)
    print(f"  Std:      {features['std']:.6f}")
    print(f"  Variance: {features['variance']:.6f}")
    print(f"  Range:    {features['range']:.6f}")
    print(f"  IQR:      {features['iqr']:.6f}")
    print(f"  MAD:      {features['mad']:.6f}")
    
    print("\n" + "="*70)
    print("EXTREMES")
    print("="*70)
    print(f"  Min:          {features['min']:.6f}")
    print(f"  Max:          {features['max']:.6f}")
    print(f"  Percentile 25: {features['percentile_25']:.6f}")
    print(f"  Percentile 75: {features['percentile_75']:.6f}")
    print(f"  Percentile 95: {features['percentile_95']:.6f}")
    
    print("\n" + "="*70)
    print("SHAPE")
    print("="*70)
    print(f"  Kurtosis: {features['kurtosis']:.6f}")
    print(f"  Skewness: {features['skewness']:.6f}")
    
    print("\n" + "="*70)
    print("ENERGY")
    print("="*70)
    print(f"  RMS:    {features['rms']:.6f}")
    print(f"  Energy: {features['energy']:.6f}")
    print(f"  SMA:    {features['sma']:.6f}")
    
    print("\n" + "="*70)
    print("OTHER")
    print("="*70)
    print(f"  Zero crossing rate: {features['zero_crossing_rate']:.6f}")
    print(f"  Autocorr lag1:      {features['autocorr_lag1']:.6f}")
    
    print("\n" + "="*70)
    print("✓ Test complete!")
    print("="*70)
