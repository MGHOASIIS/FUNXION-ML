"""
Temporal Feature Extraction
Peak detection, onset/offset, duration metrics
"""

from scipy.signal import find_peaks
import numpy as np


def extract_temporal_features(signal, fs=100):
    """
    Extract time-based features from signal.
    
    Args:
        signal: (T,) - single channel time series
        fs: sampling rate in Hz
    
    Returns:
        dict: Temporal features
    
    Features extracted:
        - Movement duration
        - Peak detection (count, heights)
        - Inter-peak intervals
        - Valley detection
        - Movement onset/offset times
        - Active movement duration
    """
    features = {}
    
    # ===== MOVEMENT DURATION =====
    features['duration'] = len(signal) / fs  # in seconds
    
    # ===== PEAK DETECTION =====
    peaks, properties = find_peaks(
        signal, 
        height=np.mean(signal),
        distance=int(fs//10)  # min 0.1s between peaks
    )
    
    features['num_peaks'] = len(peaks)
    features['mean_peak_height'] = np.mean(properties['peak_heights']) if len(peaks) > 0 else 0
    features['max_peak_height'] = np.max(properties['peak_heights']) if len(peaks) > 0 else 0
    features['std_peak_height'] = np.std(properties['peak_heights']) if len(peaks) > 0 else 0
    
    # ===== INTER-PEAK INTERVALS =====
    if len(peaks) > 1:
        inter_peak_intervals = np.diff(peaks) / fs
        features['mean_inter_peak_interval'] = np.mean(inter_peak_intervals)
        features['std_inter_peak_interval'] = np.std(inter_peak_intervals)
        features['cv_inter_peak_interval'] = np.std(inter_peak_intervals) / (np.mean(inter_peak_intervals) + 1e-10)
    else:
        features['mean_inter_peak_interval'] = 0
        features['std_inter_peak_interval'] = 0
        features['cv_inter_peak_interval'] = 0
    
    # ===== VALLEY DETECTION =====
    valleys, _ = find_peaks(
        -signal, 
        height=-np.mean(signal),
        distance=int(fs//10)
    )
    features['num_valleys'] = len(valleys)
    
    # ===== ONSET/OFFSET DETECTION =====
    # Detect movement start/end based on threshold
    threshold = np.mean(signal) + 2*np.std(signal)
    above_threshold = signal > threshold
    
    if np.any(above_threshold):
        onset = np.where(above_threshold)[0][0]
        offset = np.where(above_threshold)[0][-1]
        features['movement_onset_time'] = onset / fs
        features['movement_offset_time'] = offset / fs
        features['active_movement_duration'] = (offset - onset) / fs
    else:
        features['movement_onset_time'] = 0
        features['movement_offset_time'] = 0
        features['active_movement_duration'] = 0
    
    return features


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    print("="*70)
    print("Testing Temporal Feature Extraction")
    print("="*70)
    
    # Create test signal (reaching movement)
    T = 500
    fs = 100
    t = np.linspace(0, T/fs, T)
    
    # Bell-shaped movement with some peaks
    signal = (
        2.0 * np.exp(-((t - 1.5)**2) / 0.3) +    # Main movement peak
        0.5 * np.exp(-((t - 2.0)**2) / 0.1) +    # Secondary peak
        0.3 * np.exp(-((t - 3.0)**2) / 0.2) +    # Another peak
        0.1 * np.random.randn(T)                  # Noise
    )
    
    print(f"\nTest signal:")
    print(f"  Length: {T} samples ({T/fs:.1f} seconds)")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Mean: {np.mean(signal):.4f}")
    print(f"  Std: {np.std(signal):.4f}")
    
    # Extract features
    print("\nExtracting temporal features...")
    features = extract_temporal_features(signal, fs=fs)
    
    print(f"\n✓ Extracted {len(features)} features")
    
    # Display results
    print("\n" + "="*70)
    print("TEMPORAL FEATURES")
    print("="*70)
    
    print(f"\nDuration:")
    print(f"  Total duration: {features['duration']:.2f} seconds")
    print(f"  Active movement: {features['active_movement_duration']:.2f} seconds")
    
    print(f"\nPeaks:")
    print(f"  Number of peaks: {features['num_peaks']}")
    print(f"  Mean peak height: {features['mean_peak_height']:.4f}")
    print(f"  Max peak height: {features['max_peak_height']:.4f}")
    print(f"  Std peak height: {features['std_peak_height']:.4f}")
    
    print(f"\nTiming:")
    print(f"  Movement onset: {features['movement_onset_time']:.2f} s")
    print(f"  Movement offset: {features['movement_offset_time']:.2f} s")
    
    if features['num_peaks'] > 1:
        print(f"\nInter-peak intervals:")
        print(f"  Mean: {features['mean_inter_peak_interval']:.2f} s")
        print(f"  Std: {features['std_inter_peak_interval']:.2f} s")
        print(f"  CV: {features['cv_inter_peak_interval']:.2f}")
    
    print(f"\nValleys:")
    print(f"  Number of valleys: {features['num_valleys']}")
    
    # Clinical interpretation
    print("\n" + "="*70)
    print("CLINICAL INTERPRETATION")
    print("="*70)
    
    num_peaks = features['num_peaks']
    print(f"\nNumber of movement units: {num_peaks}")
    if num_peaks == 1:
        print("  → Single smooth movement (healthy) ✓")
    elif num_peaks == 2:
        print("  → Slightly fragmented movement")
    else:
        print("  → Highly fragmented movement (compensation/impairment)")
    
    print("\n" + "="*70)
    print("✓ Test complete!")
    print("="*70)
