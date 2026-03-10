"""
Frequency Domain Feature Extraction
For single-channel time series signals
"""

from scipy import signal, stats
import numpy as np


def extract_frequency_features(signal_data, fs=100):
    """
    Extract frequency domain features from signal.
    
    Args:
        signal_data: (T,) - single channel time series
        fs: sampling rate in Hz
    
    Returns:
        dict: Frequency domain features
    
    Features extracted:
        - Power Spectral Density (Welch method)
        - Dominant frequency
        - Frequency band energies (5 bands)
        - Spectral moments (mean, median, spread)
        - Spectral entropy
        - Spectral centroid
        - Spectral rolloff
        - Spectral shape (kurtosis, skewness)
        - FFT-based features
    """
    features = {}
    
    # Validate signal
    if len(signal_data) < 10 or np.std(signal_data) < 1e-10:
        return _get_nan_features()
    
    # Clean signal (remove NaN/inf)
    signal_clean = signal_data[~(np.isnan(signal_data) | np.isinf(signal_data))]
    if len(signal_clean) < 10:
        return _get_nan_features()
    
    # ===== POWER SPECTRAL DENSITY (Welch method) =====
    try:
        f, psd = signal.welch(signal_clean, fs=fs, nperseg=min(256, len(signal_clean)))
        
        # Total spectral energy
        features['spectral_energy_total'] = np.sum(psd)
        
        # ===== DOMINANT FREQUENCY =====
        features['dominant_frequency'] = f[np.argmax(psd)]
        features['dominant_frequency_magnitude'] = np.max(psd)
        
        # ===== FREQUENCY BANDS ENERGY =====
        # Define bands relevant to human movement
        bands = {
            'very_low': (0, 0.5),
            'low': (0.5, 2),
            'mid': (2, 5),
            'high': (5, 10),
            'very_high': (10, 20)
        }
        
        for band_name, (f_low, f_high) in bands.items():
            idx_band = np.logical_and(f >= f_low, f <= f_high)
            features[f'spectral_energy_{band_name}'] = np.sum(psd[idx_band])
        
        # ===== SPECTRAL MOMENTS =====
        # Weighted average frequency
        features['mean_frequency'] = np.sum(f * psd) / (np.sum(psd) + 1e-10)
        
        # Median frequency
        cumsum_psd = np.cumsum(psd)
        median_idx = np.where(cumsum_psd >= np.sum(psd)/2)[0]
        if len(median_idx) > 0:
            features['median_frequency'] = f[median_idx[0]]
        else:
            features['median_frequency'] = np.nan
        
        # Spectral spread (variance in frequency)
        features['spectral_spread'] = np.sqrt(
            np.sum(((f - features['mean_frequency'])**2) * psd) / (np.sum(psd) + 1e-10)
        )
        
        # ===== SPECTRAL ENTROPY =====
        psd_normalized = psd / (np.sum(psd) + 1e-10)
        features['spectral_entropy'] = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-10))
        
        # ===== SPECTRAL CENTROID =====
        features['spectral_centroid'] = np.sum(f * psd) / (np.sum(psd) + 1e-10)
        
        # ===== SPECTRAL ROLLOFF (95% energy) =====
        rolloff_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
        if len(rolloff_idx) > 0:
            features['spectral_rolloff'] = f[rolloff_idx[0]]
        else:
            features['spectral_rolloff'] = np.nan
        
        # ===== SPECTRAL KURTOSIS & SKEWNESS =====
        features['spectral_kurtosis'] = stats.kurtosis(psd)
        features['spectral_skewness'] = stats.skew(psd)
        
    except Exception as e:
        print(f"      Welch PSD failed: {e}")
        return _get_nan_features()
    
    # ===== FFT-BASED (alternative to Welch) =====
    try:
        fft_vals = np.fft.fft(signal_clean)
        fft_magnitude = np.abs(fft_vals)[:len(signal_clean)//2]
        fft_freqs = np.fft.fftfreq(len(signal_clean), 1/fs)[:len(signal_clean)//2]
        
        features['fft_max_magnitude'] = np.max(fft_magnitude)
        features['fft_mean_magnitude'] = np.mean(fft_magnitude)
    except Exception as e:
        features['fft_max_magnitude'] = np.nan
        features['fft_mean_magnitude'] = np.nan
    
    return features


def _get_nan_features():
    """Return all features as NaN."""
    return {
        'spectral_energy_total': np.nan,
        'dominant_frequency': np.nan,
        'dominant_frequency_magnitude': np.nan,
        'spectral_energy_very_low': np.nan,
        'spectral_energy_low': np.nan,
        'spectral_energy_mid': np.nan,
        'spectral_energy_high': np.nan,
        'spectral_energy_very_high': np.nan,
        'mean_frequency': np.nan,
        'median_frequency': np.nan,
        'spectral_spread': np.nan,
        'spectral_entropy': np.nan,
        'spectral_centroid': np.nan,
        'spectral_rolloff': np.nan,
        'spectral_kurtosis': np.nan,
        'spectral_skewness': np.nan,
        'fft_max_magnitude': np.nan,
        'fft_mean_magnitude': np.nan
    }


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    print("="*70)
    print("Testing Frequency Domain Feature Extraction")
    print("="*70)
    
    # Create test signal
    T = 500
    fs = 100
    t = np.linspace(0, T/fs, T)
    
    # Signal with multiple frequency components
    signal_test = (
        np.sin(2 * np.pi * 1.5 * t) +      # 1.5 Hz
        0.5 * np.sin(2 * np.pi * 3.0 * t) + # 3.0 Hz
        0.2 * np.random.randn(T)            # Noise
    )
    
    print(f"\nTest signal:")
    print(f"  Length: {len(signal_test)} samples")
    print(f"  Duration: {T/fs:.2f} seconds")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Expected frequencies: 1.5 Hz, 3.0 Hz")
    
    # Extract features
    print("\nExtracting frequency features...")
    features = extract_frequency_features(signal_test, fs=fs)
    
    print(f"\n✓ Extracted {len(features)} features")
    
    # Display results
    print("\n" + "="*70)
    print("SPECTRAL FEATURES")
    print("="*70)
    
    print(f"\nDominant Frequency: {features['dominant_frequency']:.2f} Hz")
    print(f"Dominant Magnitude: {features['dominant_frequency_magnitude']:.6f}")
    print(f"Total Energy: {features['spectral_energy_total']:.6f}")
    
    print("\n" + "="*70)
    print("FREQUENCY BAND ENERGIES")
    print("="*70)
    
    bands = ['very_low', 'low', 'mid', 'high', 'very_high']
    band_ranges = ['0-0.5 Hz', '0.5-2 Hz', '2-5 Hz', '5-10 Hz', '10-20 Hz']
    
    for band, range_str in zip(bands, band_ranges):
        energy = features[f'spectral_energy_{band}']
        percent = (energy / features['spectral_energy_total']) * 100
        print(f"  {band:12s} ({range_str:11s}): {energy:10.6f} ({percent:5.2f}%)")
    
    print("\n" + "="*70)
    print("SPECTRAL MOMENTS")
    print("="*70)
    
    print(f"  Mean Frequency:   {features['mean_frequency']:.2f} Hz")
    print(f"  Median Frequency: {features['median_frequency']:.2f} Hz")
    print(f"  Spectral Spread:  {features['spectral_spread']:.2f} Hz")
    print(f"  Spectral Centroid: {features['spectral_centroid']:.2f} Hz")
    print(f"  Spectral Rolloff: {features['spectral_rolloff']:.2f} Hz")
    
    print("\n" + "="*70)
    print("SPECTRAL SHAPE")
    print("="*70)
    
    print(f"  Spectral Entropy:  {features['spectral_entropy']:.6f}")
    print(f"  Spectral Kurtosis: {features['spectral_kurtosis']:.6f}")
    print(f"  Spectral Skewness: {features['spectral_skewness']:.6f}")
    
    print("\n" + "="*70)
    print("FFT FEATURES")
    print("="*70)
    
    print(f"  FFT Max Magnitude:  {features['fft_max_magnitude']:.6f}")
    print(f"  FFT Mean Magnitude: {features['fft_mean_magnitude']:.6f}")
    
    print("\n" + "="*70)
    print("✓ Test complete!")
    print("="*70)