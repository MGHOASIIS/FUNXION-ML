"""
Joint Kinematics Feature Extraction
CORRECTED FOR POSITION + ROTATION DATA

Your data: Position + Rotation (Euler angles already provided!)
NOT: Accelerometer + Gyroscope (doesn't need sensor fusion)
"""

import numpy as np
from feature_config import SAMPLING_RATE


def get_joint_angles(position_data, rotation_data):
    """
    Extract joint angles from position and rotation data.
    
    Args:
        position_data: (T, 3) - [PosX, PosY, PosZ] in meters
        rotation_data: (T, 3) - [RotX, RotY, RotZ] in degrees (Euler angles)
    
    Returns:
        (T, 3) - Euler angles [roll, pitch, yaw] in degrees
    
    Note: Your data ALREADY contains rotation (Euler angles)!
          No sensor fusion needed - just return the rotation data.
    """
    # Your data already has Euler angles in rotation_data!
    # rotation_data is already [RotX, RotY, RotZ] = [roll, pitch, yaw]
    euler_angles = rotation_data  # Already in degrees
    
    return euler_angles


def calculate_rom(angle_timeseries):
    """
    Calculate Range of Motion from angle time series.
    
    Args:
        angle_timeseries: (T,) - single joint angle over time (degrees)
    
    Returns:
        dict: ROM features
    """
    return {
        'rom': np.max(angle_timeseries) - np.min(angle_timeseries),
        'max_angle': np.max(angle_timeseries),
        'min_angle': np.min(angle_timeseries),
        'mean_angle': np.mean(angle_timeseries),
        'std_angle': np.std(angle_timeseries)
    }


def calculate_kinematics(angle_timeseries, dt=None):
    """
    Calculate angular velocity and acceleration.
    
    Args:
        angle_timeseries: (T,) - single joint angle over time (degrees)
        dt: time step (if None, uses 1/SAMPLING_RATE)
    
    Returns:
        dict: Angular velocity and acceleration features
    """
    if dt is None:
        dt = 1.0 / SAMPLING_RATE
    
    # Angular velocity (degrees/second)
    velocity = np.gradient(angle_timeseries, dt)
    
    # Angular acceleration (degrees/second²)
    acceleration = np.gradient(velocity, dt)
    
    return {
        'peak_angular_velocity': np.max(np.abs(velocity)),
        'mean_angular_velocity': np.mean(np.abs(velocity)),
        'std_angular_velocity': np.std(velocity),
        'peak_angular_acceleration': np.max(np.abs(acceleration)),
        'mean_angular_acceleration': np.mean(np.abs(acceleration))
    }


def extract_rotation_features(rotation_data, dt=None):
    """
    Extract comprehensive features from rotation data.
    
    Args:
        rotation_data: (T, 3) - [RotX, RotY, RotZ] in degrees
        dt: time step (if None, uses 1/SAMPLING_RATE)
    
    Returns:
        dict: All rotation features
    """
    if dt is None:
        dt = 1.0 / SAMPLING_RATE
    
    features = {}
    
    # For each axis (roll, pitch, yaw)
    axis_names = ['roll', 'pitch', 'yaw']
    
    for i, axis in enumerate(axis_names):
        angle_series = rotation_data[:, i]
        
        # ROM features
        rom_features = calculate_rom(angle_series)
        for key, val in rom_features.items():
            features[f'{axis}_{key}'] = val
        
        # Kinematics features
        kin_features = calculate_kinematics(angle_series, dt)
        for key, val in kin_features.items():
            features[f'{axis}_{key}'] = val
    
    # Overall rotation magnitude
    rot_magnitude = np.linalg.norm(rotation_data, axis=1)
    features['rotation_magnitude_mean'] = np.mean(rot_magnitude)
    features['rotation_magnitude_std'] = np.std(rot_magnitude)
    features['rotation_magnitude_max'] = np.max(rot_magnitude)
    
    return features


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    print("="*70)
    print("Testing Joint Kinematics - Position + Rotation Data")
    print("="*70)
    
    # Create test data matching your format
    T = 500
    
    # Position data (not used in this module, but showing format)
    position_data = np.random.randn(T, 3) * 0.5  # [PosX, PosY, PosZ]
    
    # Rotation data (Euler angles in degrees)
    # Simulating hand rotation during reaching
    t = np.linspace(0, 10, T)
    rotation_data = np.zeros((T, 3))
    rotation_data[:, 0] = 10 * np.sin(2 * np.pi * 0.5 * t)  # Roll
    rotation_data[:, 1] = 20 * np.sin(2 * np.pi * 0.3 * t)  # Pitch
    rotation_data[:, 2] = 5 * np.sin(2 * np.pi * 0.8 * t)   # Yaw
    
    print(f"\nTest data shapes:")
    print(f"  Position: {position_data.shape}")
    print(f"  Rotation: {rotation_data.shape}")
    
    # Test get_joint_angles
    print("\n" + "="*70)
    print("TEST 1: get_joint_angles()")
    print("="*70)
    
    euler_angles = get_joint_angles(position_data, rotation_data)
    
    print(f"\nEuler angles shape: {euler_angles.shape}")
    print(f"Euler angles are same as rotation_data: {np.allclose(euler_angles, rotation_data)}")
    print("✓ For Position+Rotation data, rotation IS the Euler angles!")
    
    # Test ROM calculation
    print("\n" + "="*70)
    print("TEST 2: calculate_rom()")
    print("="*70)
    
    roll_rom = calculate_rom(rotation_data[:, 0])
    
    print(f"\nRoll ROM features:")
    for key, val in roll_rom.items():
        print(f"  {key}: {val:.2f}°")
    
    # Test kinematics
    print("\n" + "="*70)
    print("TEST 3: calculate_kinematics()")
    print("="*70)
    
    roll_kin = calculate_kinematics(rotation_data[:, 0])
    
    print(f"\nRoll kinematics:")
    for key, val in roll_kin.items():
        print(f"  {key}: {val:.2f}°/s or °/s²")
    
    # Test comprehensive extraction
    print("\n" + "="*70)
    print("TEST 4: extract_rotation_features()")
    print("="*70)
    
    all_features = extract_rotation_features(rotation_data)
    
    print(f"\n✓ Extracted {len(all_features)} rotation features")
    
    print("\nFeatures by axis:")
    for axis in ['roll', 'pitch', 'yaw']:
        axis_features = [k for k in all_features.keys() if axis in k]
        print(f"  {axis}: {len(axis_features)} features")
    
    print("\nSample features:")
    for i, (key, val) in enumerate(list(all_features.items())[:10]):
        print(f"  {key}: {val:.2f}")
    
    print("\n" + "="*70)
    print("✓ Test complete!")
    print("="*70)
    
    print("\nIMPORTANT:")
    print("  Your data ALREADY contains Euler angles (RotX, RotY, RotZ)")
    print("  No sensor fusion needed!")
    print("  Just extract features directly from rotation channels")