"""
Movement Quality and Smoothness Features
Jerk-based and spectral smoothness metrics
"""

import numpy as np
from scipy import signal


def calculate_smoothness_jerk(velocity_data, dt=0.01):
    """
    Jerk-based smoothness metric.
    
    Args:
        velocity_data: (T,) - velocity time series
        dt: time step (1/sampling_rate)
    
    Returns:
        dict: Jerk metrics
    """
    # Calculate jerk (derivative of acceleration)
    jerk = np.diff(velocity_data) / dt
    
    # Normalized jerk
    duration = len(velocity_data) * dt
    peak_velocity = np.max(np.abs(velocity_data))
    
    if peak_velocity > 0:
        normalized_jerk = np.sqrt(np.mean(jerk**2)) * (duration**3) / (peak_velocity**2)
    else:
        normalized_jerk = 0
    
    return {
        'jerk_metric': normalized_jerk,
        'jerk_rms': np.sqrt(np.mean(jerk**2))
    }


def calculate_sparc(velocity_data, fs=100):
    """
    Spectral Arc Length (SPARC) - movement smoothness.
    Lower values = smoother movement.
    
    Args:
        velocity_data: (T,) - velocity time series
        fs: sampling rate in Hz
    
    Returns:
        float: SPARC value (negative, lower = smoother)
    """
    try:
        # Compute power spectral density
        f, psd = signal.welch(velocity_data, fs=fs, nperseg=min(256, len(velocity_data)))
        
        # Normalize PSD
        psd_normalized = psd / (np.sum(psd) + 1e-10)
        
        # Calculate arc length
        arc_length = -np.sum(np.sqrt(1 + np.diff(psd_normalized)**2))
        
        return arc_length
    except Exception as e:
        print(f"      SPARC calculation failed: {e}")
        return 0.0


# Note: calculate_entropy_features and calculate_lyapunov are duplicates
# of functions in complexity_entropy.py - they're not needed here
# The master integrator will call complexity_entropy.py for those features


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    print("="*70)
    print("Testing Movement Quality Features")
    print("="*70)
    
    # Create test velocity signal
    T = 500
    t = np.linspace(0, 5, T)
    
    # Simulate reaching movement (bell-shaped velocity profile)
    velocity = 2.0 * np.exp(-((t - 2.5)**2) / 0.5) + 0.05 * np.random.randn(T)
    
    print(f"\nTest velocity signal:")
    print(f"  Length: {T} samples")
    print(f"  Peak velocity: {np.max(velocity):.4f} m/s")
    print(f"  Mean velocity: {np.mean(np.abs(velocity)):.4f} m/s")
    
    # Test jerk calculation
    print("\n" + "="*70)
    print("TEST 1: Jerk Metric")
    print("="*70)
    
    jerk_features = calculate_smoothness_jerk(velocity, dt=0.01)
    
    print(f"\nJerk features:")
    for key, val in jerk_features.items():
        print(f"  {key}: {val:.6f}")
    
    # Interpretation
    njerk = jerk_features['jerk_metric']
    print(f"\nInterpretation:")
    if njerk < 50:
        print(f"  {njerk:.2f} → Smooth movement ✓")
    elif njerk < 100:
        print(f"  {njerk:.2f} → Moderate smoothness")
    else:
        print(f"  {njerk:.2f} → Jerky movement")
    
    # Test SPARC calculation
    print("\n" + "="*70)
    print("TEST 2: SPARC (Spectral Arc Length)")
    print("="*70)
    
    sparc_value = calculate_sparc(velocity, fs=100)
    
    print(f"\nSPARC value: {sparc_value:.6f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if sparc_value < -2:
        print(f"  {sparc_value:.2f} → Very smooth movement ✓")
    elif sparc_value < -1:
        print(f"  {sparc_value:.2f} → Smooth movement")
    else:
        print(f"  {sparc_value:.2f} → Jerky movement")
    
    print("\n" + "="*70)
    print("CLINICAL INTERPRETATION")
    print("="*70)
    
    print("\nJerk Metric:")
    print("  • Lower = Smoother movement")
    print("  • Healthy: < 50")
    print("  • Impaired: > 100")
    print("  • Indicates motor control quality")
    
    print("\nSPARC:")
    print("  • Lower (more negative) = Smoother")
    print("  • Healthy: < -2")
    print("  • Impaired: > -1")
    print("  • Sensitive to sub-movements and corrections")
    
    print("\n" + "="*70)
    print("✓ Test complete!")
    print("="*70)
