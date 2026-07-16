"""
Complexity and Entropy Feature Extraction
For single-channel time series signals
"""

import numpy as np
import antropy as ant
import neurokit2 as nk

from feature_config import SAMPLING_RATE


def extract_complexity_features(signal_data):
    """
    Extract comprehensive entropy and complexity measures.
    
    Args:
        signal_data: (T,) - single channel time series
    
    Returns:
        dict: Complexity and entropy features
    
    Features extracted:
        - Sample Entropy (regularity)
        - Approximate Entropy
        - Permutation Entropy
        - Spectral Entropy
        - SVD Entropy
        - Hjorth Parameters (activity, mobility, complexity)
        - Detrended Fluctuation Analysis (DFA)
        - Fractal Dimensions (Petrosian, Katz, Higuchi)
        - Lempel-Ziv Complexity
        - Fuzzy Entropy
        - Multiscale Entropy
    """
    features = {}
    
    
    # ===== SAMPLE ENTROPY =====
    # Measures regularity/predictability (lower = more regular)
    try:
        features['sample_entropy'] = ant.sample_entropy(signal_data, order=2, metric='chebyshev')
    except Exception as e:
        features['sample_entropy'] = np.nan
    
    # ===== APPROXIMATE ENTROPY =====
    try:
        features['approx_entropy'] = ant.app_entropy(signal_data, order=2, metric='chebyshev')
    except Exception as e:
        features['approx_entropy'] = np.nan
    
    # ===== PERMUTATION ENTROPY =====
    # Order-based entropy (rotation invariant)
    try:
        features['perm_entropy'] = ant.perm_entropy(signal_data, order=3, normalize=True)
    except Exception as e:
        features['perm_entropy'] = np.nan
    
    # ===== SPECTRAL ENTROPY =====
    # Entropy of power spectral density
    try:
        features['spectral_entropy'] = ant.spectral_entropy(
            signal_data, 
            sf=SAMPLING_RATE,
            method='welch', 
            normalize=True
        )
    except Exception as e:
        features['spectral_entropy'] = np.nan
    
    # ===== SVD ENTROPY =====
    # Singular value decomposition entropy
    try:
        features['svd_entropy'] = ant.svd_entropy(
            signal_data, 
            order=3, 
            delay=1, 
            normalize=True
        )
    except Exception as e:
        features['svd_entropy'] = np.nan
    
    # ===== HJORTH PARAMETERS =====
    # Activity, Mobility, Complexity
    try:
        hjorth = ant.hjorth_params(signal_data)
        features['hjorth_activity'] = hjorth[0]
        features['hjorth_mobility'] = hjorth[1]
        # features['hjorth_complexity'] = hjorth[2]
    except Exception as e:
        features['hjorth_activity'] = np.nan
        features['hjorth_mobility'] = np.nan
        # features['hjorth_complexity'] = np.nan
    
    # ===== DETRENDED FLUCTUATION ANALYSIS =====
    try:
        features['dfa'] = ant.detrended_fluctuation(signal_data)
    except Exception as e:
        features['dfa'] = np.nan
    
    # ===== PETROSIAN FRACTAL DIMENSION =====
    try:
        features['petrosian_fd'] = ant.petrosian_fd(signal_data)
    except Exception as e:
        features['petrosian_fd'] = np.nan
    
    # ===== KATZ FRACTAL DIMENSION =====
    try:
        features['katz_fd'] = ant.katz_fd(signal_data)
    except Exception as e:
        features['katz_fd'] = np.nan
    
    # ===== HIGUCHI FRACTAL DIMENSION =====
    try:
        features['higuchi_fd'] = ant.higuchi_fd(signal_data, kmax=10)
    except Exception as e:
        features['higuchi_fd'] = np.nan
    
    # ===== LEMPEL-ZIV COMPLEXITY =====
    try:
        # Binarize signal first
        binary_signal = (signal_data > np.median(signal_data)).astype(int)
        features['lempel_ziv'] = ant.lziv_complexity(binary_signal, normalize=True)
    except Exception as e:
        features['lempel_ziv'] = np.nan
    
    # ===== FUZZY ENTROPY (via NeuroKit2) =====
    try:
        features['fuzzy_entropy'] = nk.entropy_fuzzy(signal_data)[0]
    except Exception as e:
        features['fuzzy_entropy'] = np.nan
    
    # ===== MULTISCALE ENTROPY =====
    # try:
    #     mse = multiscale_entropy(signal_data, maxscale=10)
    #     features['mse_mean'] = np.mean(mse)
    #     features['mse_std'] = np.std(mse)
    # except Exception as e:
    #     features['mse_mean'] = np.nan
    #     features['mse_std'] = np.nan
    
    return features


# =====  HELPER FUNCTIONS =====
def calculate_lyapunov(signal_data):
    """
    Lyapunov exponent - movement stability.
    Higher values = more chaotic/less stable.
    
    Args:
        signal_data: (T,) single channel
    
    Returns:
        float: Lyapunov exponent
    """
    try:
        lyap = nk.complexity_lyapunov(signal_data, delay=1, dimension=3)
        return lyap
    except:
        return np.nan


# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    print("="*70)
    print("Testing Complexity & Entropy Feature Extraction")
    print("="*70)
    
    # Create test signal
    T = 1000
    t = np.linspace(0, 10, T)
    
    # Synthetic signal (sine wave + noise)
    signal = np.sin(2 * np.pi * 1.5 * t) + 0.1 * np.random.randn(T)
    
    print(f"\nTest signal shape: {signal.shape}")
    print(f"Signal range: [{np.min(signal):.3f}, {np.max(signal):.3f}]")
    
    # Extract features
    print("\nExtracting complexity features...")
    features = extract_complexity_features(signal)
    
    print(f"\n✓ Extracted {len(features)} features")
    
    # Display results
    print("\n" + "="*70)
    print("ENTROPY MEASURES")
    print("="*70)
    
    entropy_features = {
        'Sample Entropy': features.get('sample_entropy'),
        'Approximate Entropy': features.get('approx_entropy'),
        'Permutation Entropy': features.get('perm_entropy'),
        'Spectral Entropy': features.get('spectral_entropy'),
        'SVD Entropy': features.get('svd_entropy'),
        'Fuzzy Entropy': features.get('fuzzy_entropy')
    }
    
    for name, value in entropy_features.items():
        if not np.isnan(value):
            print(f"  {name:25s}: {value:.6f}")
        else:
            print(f"  {name:25s}: N/A")
    
    print("\n" + "="*70)
    print("HJORTH PARAMETERS")
    print("="*70)
    
    hjorth_features = {
        'Activity (power)': features.get('hjorth_activity'),
        'Mobility (mean freq)': features.get('hjorth_mobility'),
        'Complexity (freq spread)': features.get('hjorth_complexity')
    }
    
    # for name, value in hjorth_features.items():
    #     if not np.isnan(value):
    #         print(f"  {name:25s}: {value:.6f}")
    #     else:
    #         print(f"  {name:25s}: N/A")
    
    print("\n" + "="*70)
    print("FRACTAL DIMENSIONS")
    print("="*70)
    
    fractal_features = {
        'DFA': features.get('dfa'),
        'Petrosian FD': features.get('petrosian_fd'),
        'Katz FD': features.get('katz_fd'),
        'Higuchi FD': features.get('higuchi_fd')
    }
    
    for name, value in fractal_features.items():
        if not np.isnan(value):
            print(f"  {name:25s}: {value:.6f}")
        else:
            print(f"  {name:25s}: N/A")
    
    print("\n" + "="*70)
    print("OTHER COMPLEXITY MEASURES")
    print("="*70)
    
    other_features = {
        'Lempel-Ziv Complexity': features.get('lempel_ziv'),
        'Multiscale Entropy (mean)': features.get('mse_mean'),
        'Multiscale Entropy (std)': features.get('mse_std')
    }
    
    for name, value in other_features.items():
        if value is not None and not np.isnan(value):
            print(f"  {name:25s}: {value:.6f}")
        else:
            print(f"  {name:25s}: N/A")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    se = features.get('sample_entropy', np.nan)
    if not np.isnan(se):
        print(f"\nSample Entropy: {se:.4f}")
        if se < 0.5:
            print("  → Very regular/predictable movement")
        elif se < 1.5:
            print("  → Normal variability")
        else:
            print("  → High variability/irregularity")
    
    dfa_val = features.get('dfa', np.nan)
    if not np.isnan(dfa_val):
        print(f"\nDFA: {dfa_val:.4f}")
        if dfa_val < 0.5:
            print("  → Anti-correlated (pink noise)")
        elif dfa_val < 1.5:
            print("  → Long-range correlations")
        else:
            print("  → Random walk behavior")
    
    print("\n" + "="*70)
    print("✓ Test complete!")
    print("="*70)