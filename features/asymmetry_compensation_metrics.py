"""
Asymmetry and Compensation Metrics
For Position + Rotation data

Your data format: [PosX, PosY, PosZ, RotX, RotY, RotZ] per sensor
"""

import numpy as np


def calculate_asymmetry(left_sensor_data, right_sensor_data):
    """
    Calculate bilateral asymmetry between left and right hand sensors.
    
    Args:
        left_sensor_data: (T, 6) - [LPosX, LPosY, LPosZ, LRotX, LRotY, LRotZ]
        right_sensor_data: (T, 6) - [RPosX, RPosY, RPosZ, RRotX, RRotY, RRotZ]
    
    Returns:
        dict: Asymmetry features
    
    Note: Your data is POSITION + ROTATION (not Accel + Gyro)
    """
    features = {}
    
    # ===== MAGNITUDE-BASED ASYMMETRY =====
    # Overall spatial + rotational asymmetry
    left_mag = np.linalg.norm(left_sensor_data, axis=1)
    right_mag = np.linalg.norm(right_sensor_data, axis=1)
    
    features['lr_ratio'] = np.mean(left_mag) / (np.mean(right_mag) + 1e-10)
    features['symmetry_index'] = (np.mean(left_mag) - np.mean(right_mag)) / \
                                  (np.mean(left_mag) + np.mean(right_mag) + 1e-10)
    
    # ===== CORRELATION-BASED =====
    # How similarly do left and right hands move/rotate?
    for axis in range(6):
        corr = np.corrcoef(left_sensor_data[:, axis], right_sensor_data[:, axis])[0, 1]
        features[f'lr_correlation_axis{axis}'] = corr
    
    # ===== CROSS-CORRELATION (temporal alignment) =====
    # Position magnitude synchronization
    left_pos_mag = np.linalg.norm(left_sensor_data[:, :3], axis=1)
    right_pos_mag = np.linalg.norm(right_sensor_data[:, :3], axis=1)
    
    cross_corr = np.correlate(left_pos_mag - np.mean(left_pos_mag),
                             right_pos_mag - np.mean(right_pos_mag),
                             mode='same')
    features['max_cross_correlation'] = np.max(cross_corr)
    
    # ===== VERTICAL POSITION ASYMMETRY =====
    # Z-axis position difference (vertical asymmetry)
    left_vertical = left_sensor_data[:, 2]  # Z-axis POSITION
    right_vertical = right_sensor_data[:, 2]
    features['vertical_asymmetry'] = np.mean(left_vertical) - np.mean(right_vertical)
    
    return features


# ===== ADDITIONAL USEFUL FUNCTIONS =====

def calculate_position_asymmetry(left_positions, right_positions):
    """
    Calculate asymmetry from position data only.
    
    Args:
        left_positions: (T, 3) - [X, Y, Z]
        right_positions: (T, 3) - [X, Y, Z]
    
    Returns:
        dict: Position-specific asymmetry features
    """
    features = {}
    
    # Per-axis position differences
    for i, axis in enumerate(['x', 'y', 'z']):
        features[f'mean_pos_diff_{axis}'] = np.mean(left_positions[:, i]) - np.mean(right_positions[:, i])
        features[f'std_pos_diff_{axis}'] = np.std(left_positions[:, i] - right_positions[:, i])
    
    # Range of motion asymmetry
    for i, axis in enumerate(['x', 'y', 'z']):
        left_rom = np.ptp(left_positions[:, i])
        right_rom = np.ptp(right_positions[:, i])
        features[f'rom_asymmetry_{axis}'] = left_rom - right_rom
        features[f'rom_ratio_{axis}'] = left_rom / (right_rom + 1e-10)
    
    # Hand separation metrics
    hand_separation = np.linalg.norm(left_positions - right_positions, axis=1)
    features['mean_hand_separation'] = np.mean(hand_separation)
    features['std_hand_separation'] = np.std(hand_separation)
    features['min_hand_separation'] = np.min(hand_separation)
    features['max_hand_separation'] = np.max(hand_separation)
    
    # Path length asymmetry
    left_path_length = np.sum(np.linalg.norm(np.diff(left_positions, axis=0), axis=1))
    right_path_length = np.sum(np.linalg.norm(np.diff(right_positions, axis=0), axis=1))
    features['path_length_ratio'] = left_path_length / (right_path_length + 1e-10)
    features['path_length_asymmetry'] = left_path_length - right_path_length
    
    return features


def calculate_rotation_asymmetry(left_rotations, right_rotations):
    """
    Calculate asymmetry from rotation data only.
    
    Args:
        left_rotations: (T, 3) - [RotX, RotY, RotZ] (Euler angles)
        right_rotations: (T, 3) - [RotX, RotY, RotZ]
    
    Returns:
        dict: Rotation-specific asymmetry features
    """
    features = {}
    
    # Per-axis rotation differences
    for i, axis in enumerate(['roll', 'pitch', 'yaw']):
        features[f'mean_rot_diff_{axis}'] = np.mean(left_rotations[:, i]) - np.mean(right_rotations[:, i])
        features[f'std_rot_diff_{axis}'] = np.std(left_rotations[:, i] - right_rotations[:, i])
    
    # Rotation range asymmetry
    for i, axis in enumerate(['roll', 'pitch', 'yaw']):
        left_rot_rom = np.ptp(left_rotations[:, i])
        right_rot_rom = np.ptp(right_rotations[:, i])
        features[f'rot_rom_asymmetry_{axis}'] = left_rot_rom - right_rot_rom
    
    # Overall rotation magnitude asymmetry
    left_rot_mag = np.linalg.norm(left_rotations, axis=1)
    right_rot_mag = np.linalg.norm(right_rotations, axis=1)
    features['rotation_magnitude_asymmetry'] = np.mean(left_rot_mag) - np.mean(right_rot_mag)
    
    return features


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    print("="*70)
    print("Testing Asymmetry Feature Extraction")
    print("Data format: Position + Rotation")
    print("="*70)
    
    # Create dummy data matching your format
    T = 500
    
    # LeftHand: channels 6-11 [LPosX, LPosY, LPosZ, LRotX, LRotY, LRotZ]
    left_data = np.random.randn(T, 6)
    left_data[:, 2] += 0.1  # Left hand slightly higher
    
    # RightHand: channels 12-17 [RPosX, RPosY, RPosZ, RRotX, RRotY, RRotZ]
    right_data = np.random.randn(T, 6)
    
    print(f"\nTest data shapes:")
    print(f"  Left sensor: {left_data.shape}")
    print(f"  Right sensor: {right_data.shape}")
    
    # Calculate asymmetry
    print("\nExtracting asymmetry features...")
    features = calculate_asymmetry(left_data, right_data)
    
    print(f"\n✓ Extracted {len(features)} asymmetry features")
    
    # Display results
    print("\n" + "="*70)
    print("ASYMMETRY RESULTS")
    print("="*70)
    
    print("\nMagnitude-based:")
    print(f"  LR Ratio: {features['lr_ratio']:.4f}")
    print(f"  Symmetry Index: {features['symmetry_index']:.4f}")
    
    print("\nSpatial asymmetry:")
    print(f"  Vertical Asymmetry: {features['vertical_asymmetry']:.4f}")
    print(f"  Max Cross-Correlation: {features['max_cross_correlation']:.4f}")
    
    print("\nCorrelations per axis:")
    axis_names = ['PosX', 'PosY', 'PosZ', 'RotX', 'RotY', 'RotZ']
    for axis in range(6):
        corr = features[f'lr_correlation_axis{axis}']
        print(f"  {axis_names[axis]:6s}: {corr:+.4f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Interpret symmetry index
    si = features['symmetry_index']
    print(f"\nSymmetry Index: {si:.4f}")
    if abs(si) < 0.1:
        print("  → Symmetric movement ✓")
    elif abs(si) < 0.2:
        print("  → Mild asymmetry")
    else:
        if si > 0:
            print("  → Left hand dominance (significant asymmetry)")
        else:
            print("  → Right hand dominance (significant asymmetry)")
    
    # Interpret vertical asymmetry
    va = features['vertical_asymmetry']
    print(f"\nVertical Asymmetry: {va:.4f}")
    if abs(va) < 0.05:
        print("  → Hands at similar height ✓")
    elif va > 0:
        print(f"  → Left hand is {va:.2f}m higher")
    else:
        print(f"  → Right hand is {abs(va):.2f}m higher")
    
    # Interpret correlations
    pos_corr = np.mean([features[f'lr_correlation_axis{i}'] for i in range(3)])
    print(f"\nAverage Position Correlation: {pos_corr:.4f}")
    if pos_corr > 0.7:
        print("  → Good bilateral coordination ✓")
    elif pos_corr > 0.4:
        print("  → Moderate coordination")
    else:
        print("  → Poor bilateral coordination")
    
    print("\n" + "="*70)
    print("✓ Test complete!")
    print("="*70)