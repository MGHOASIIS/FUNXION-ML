"""
Wavelet Feature Extraction
Discrete and continuous wavelet transform features
"""

import pywt
import numpy as np


def extract_wavelet_features(signal_data, wavelet='db4', level=5):
    """
    Extract wavelet decomposition features.
    
    Args:
        signal_data: (T,) - single channel time series
        wavelet: wavelet type ('db4', 'coif1', 'sym5', etc.)
        level: decomposition level
    
    Returns:
        dict: Wavelet features
    
    Features extracted:
        - Approximation coefficients (low frequency)
        - Detail coefficients (high frequency, multiple levels)
        - Energy per level
        - Energy ratios
        - Wavelet entropy per level
        - Continuous wavelet transform features
    """
    features = {}
    
    try:
        # ===== DISCRETE WAVELET TRANSFORM (DWT) =====
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        # coeffs[0] = approximation coefficients (low freq)
        # coeffs[1:] = detail coefficients (high freq)
        
        # ===== COEFFICIENT STATISTICS =====
        for i, coeff in enumerate(coeffs):
            if i == 0:
                prefix = 'approximation'
            else:
                prefix = f'detail_level{i}'
            
            features[f'{prefix}_mean'] = np.mean(coeff)
            features[f'{prefix}_std'] = np.std(coeff)
            features[f'{prefix}_energy'] = np.sum(coeff**2)
            features[f'{prefix}_max'] = np.max(np.abs(coeff))
        
        # ===== WAVELET ENERGY =====
        total_energy = sum([np.sum(c**2) for c in coeffs])
        
        for i, coeff in enumerate(coeffs):
            level_energy = np.sum(coeff**2)
            
            if i == 0:
                features['approximation_energy_ratio'] = level_energy / (total_energy + 1e-10)
            else:
                features[f'detail_level{i}_energy_ratio'] = level_energy / (total_energy + 1e-10)
        
        # ===== WAVELET ENTROPY =====
        for i, coeff in enumerate(coeffs):
            # Shannon entropy of wavelet coefficients
            coeff_normalized = np.abs(coeff) / (np.sum(np.abs(coeff)) + 1e-10)
            entropy = -np.sum(coeff_normalized * np.log2(coeff_normalized + 1e-10))
            
            if i == 0:
                features['approximation_entropy'] = entropy
            else:
                features[f'detail_level{i}_entropy'] = entropy
        
        # ===== CONTINUOUS WAVELET TRANSFORM (CWT) - Optional =====
        # More computationally expensive
        try:
            scales = np.arange(1, 128)
            cwt_coeffs, freqs = pywt.cwt(signal_data, scales, wavelet)
            features['cwt_mean_energy'] = np.mean(np.abs(cwt_coeffs)**2)
            features['cwt_max_energy'] = np.max(np.abs(cwt_coeffs)**2)
        except Exception as e:
            features['cwt_mean_energy'] = np.nan
            features['cwt_max_energy'] = np.nan
    
    except Exception as e:
        print(f"      Wavelet extraction failed: {e}")
        return _get_nan_features()
    
    return features


def extract_wavelet_packets(signal_data, wavelet='db4', level=3):
    """
    Wavelet packet decomposition for finer frequency resolution.
    
    Args:
        signal_data: (T,) - single channel
        wavelet: wavelet type
        level: decomposition level
    
    Returns:
        dict: Wavelet packet features
    """
    try:
        wp = pywt.WaveletPacket(data=signal_data, wavelet=wavelet, maxlevel=level)
        features = {}
        
        # Extract features from all nodes at max level
        for node in wp.get_level(level, 'freq'):
            node_data = node.data
            node_name = node.path
            features[f'wp_{node_name}_energy'] = np.sum(node_data**2)
            features[f'wp_{node_name}_mean'] = np.mean(node_data)
            features[f'wp_{node_name}_std'] = np.std(node_data)
        
        return features
    except Exception as e:
        print(f"      Wavelet packet extraction failed: {e}")
        return {}


def _get_nan_features():
    """Return all features as NaN."""
    features = {}
    
    # Approximation
    features['approximation_mean'] = np.nan
    features['approximation_std'] = np.nan
    features['approximation_energy'] = np.nan
    features['approximation_max'] = np.nan
    features['approximation_energy_ratio'] = np.nan
    features['approximation_entropy'] = np.nan
    
    # Details (5 levels)
    for i in range(1, 6):
        features[f'detail_level{i}_mean'] = np.nan
        features[f'detail_level{i}_std'] = np.nan
        features[f'detail_level{i}_energy'] = np.nan
        features[f'detail_level{i}_max'] = np.nan
        features[f'detail_level{i}_energy_ratio'] = np.nan
        features[f'detail_level{i}_entropy'] = np.nan
    
    # CWT
    features['cwt_mean_energy'] = np.nan
    features['cwt_max_energy'] = np.nan
    
    return features


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    print("="*70)
    print("Testing Wavelet Feature Extraction")
    print("="*70)
    
    # Create test signal
    T = 500
    t = np.linspace(0, 10, T)
    
    # Multi-frequency signal
    signal = (
        np.sin(2 * np.pi * 1.0 * t) +      # Low frequency
        0.5 * np.sin(2 * np.pi * 5.0 * t) + # Mid frequency
        0.2 * np.sin(2 * np.pi * 15.0 * t) + # High frequency
        0.1 * np.random.randn(T)            # Noise
    )
    
    print(f"\nTest signal:")
    print(f"  Length: {T} samples")
    print(f"  Contains: 1 Hz, 5 Hz, 15 Hz components")
    
    # Extract DWT features
    print("\n" + "="*70)
    print("DISCRETE WAVELET TRANSFORM")
    print("="*70)
    
    features = extract_wavelet_features(signal, wavelet='db4', level=5)
    
    print(f"\n✓ Extracted {len(features)} wavelet features")
    
    # Display energy distribution
    print("\nEnergy distribution across levels:")
    print(f"  Approximation: {features['approximation_energy_ratio']:.4f} ({features['approximation_energy_ratio']*100:.1f}%)")
    
    for i in range(1, 6):
        ratio = features[f'detail_level{i}_energy_ratio']
        print(f"  Detail level {i}: {ratio:.4f} ({ratio*100:.1f}%)")
    
    # Display entropy
    print("\nEntropy per level:")
    print(f"  Approximation: {features['approximation_entropy']:.6f}")
    for i in range(1, 6):
        print(f"  Detail level {i}: {features[f'detail_level{i}_entropy']:.6f}")
    
    # Display CWT features
    print("\nContinuous Wavelet Transform:")
    print(f"  Mean energy: {features['cwt_mean_energy']:.6f}")
    print(f"  Max energy:  {features['cwt_max_energy']:.6f}")
    
    # Test wavelet packets
    print("\n" + "="*70)
    print("WAVELET PACKET DECOMPOSITION")
    print("="*70)
    
    wp_features = extract_wavelet_packets(signal, wavelet='db4', level=3)
    
    print(f"\n✓ Extracted {len(wp_features)} wavelet packet features")
    
    # Show first few
    print("\nSample wavelet packet features:")
    for i, (key, val) in enumerate(list(wp_features.items())[:5]):
        print(f"  {key}: {val:.6f}")
    
    print("\n" + "="*70)
    print("CLINICAL INTERPRETATION")
    print("="*70)
    
    print("\nApproximation (low frequency):")
    print("  → Overall movement trend")
    print("  → Slow postural adjustments")
    
    print("\nDetail levels (high frequency):")
    print("  → Rapid corrections and adjustments")
    print("  → Tremor and fine motor control")
    print("  → Higher detail = higher frequency components")
    
    print("\nEnergy ratios:")
    print("  → Distribution of frequency content")
    print("  → High approximation ratio = smooth movement")
    print("  → High detail ratio = irregular/corrective movement")
    
    print("\n" + "="*70)
    print("✓ Test complete!")
    print("="*70)
