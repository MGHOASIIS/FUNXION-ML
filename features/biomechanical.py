"""
Biomechanical Feature Extraction
For 3-sensor setup: Head, LeftHand, RightHand
Data format: Position + Rotation
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple
from config import N_SENSORS


class BiomechanicalFeatures:
    """Extract clinically-relevant biomechanical features."""
    
    def __init__(self, sensor_names: List[str]):
        self.sensor_names = sensor_names
        
        # Define body segment relationships based on N_SENSORS
        if N_SENSORS == 3:
            # 3-sensor: Functional relationships
            self.joint_pairs = {
                'left_reach': ('Head', 'LeftHand'),
                'right_reach': ('Head', 'RightHand'),
                'bilateral': ('LeftHand', 'RightHand')
            }
            self.use_anatomical_names = False
        else:
            # 8-sensor: Anatomical joints
            self.joint_pairs = {
                'spine_flexion': ('Back', 'Head'),
                'shoulder_left': ('Back', 'LeftHand'),
                'shoulder_right': ('Back', 'RightHand'),
                'hip_left': ('Back', 'LeftHip'),
                'hip_right': ('Back', 'RightHip'),
                'knee_left': ('LeftHip', 'LeftKnee'),
                'knee_right': ('RightHip', 'RightKnee')
            }
            self.use_anatomical_names = True
    
    def get_sensor_index(self, name: str) -> int:
        """Get index of sensor by name."""
        return self.sensor_names.index(name)
    
    def compute_joint_angles(self, positions: np.ndarray, 
                            rotations: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute joint angles (anatomical or functional based on sensor count).
        
        Args:
            positions: (T, N_SENSORS, 3) - XYZ positions for each sensor
            rotations: (T, N_SENSORS, 3) - Euler angles (degrees) for each sensor
            
        Returns:
            Dictionary of joint angle time series
        """
        if self.use_anatomical_names:
            return self._compute_anatomical_angles(positions, rotations)
        else:
            return self._compute_functional_angles(positions, rotations)
    
    def _compute_anatomical_angles(self, positions: np.ndarray,
                                   rotations: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute anatomical joint angles (8-sensor)."""
        angles = {}
        
        for joint_name, (proximal, distal) in self.joint_pairs.items():
            prox_idx = self.get_sensor_index(proximal)
            dist_idx = self.get_sensor_index(distal)
            
            prox_pos = positions[:, prox_idx, :]
            dist_pos = positions[:, dist_idx, :]
            segment_vector = dist_pos - prox_pos
            
            # Anatomical angles
            flexion = np.arctan2(segment_vector[:, 2], segment_vector[:, 1])
            flexion_deg = np.degrees(flexion)
            
            abduction = np.arctan2(segment_vector[:, 0], segment_vector[:, 2])
            abduction_deg = np.degrees(abduction)
            
            rotation_diff = rotations[:, dist_idx, 2] - rotations[:, prox_idx, 2]
            
            angles[f'{joint_name}_flexion'] = flexion_deg
            angles[f'{joint_name}_abduction'] = abduction_deg
            angles[f'{joint_name}_rotation'] = rotation_diff
        
        return angles
    
    def _compute_functional_angles(self, positions: np.ndarray,
                                   rotations: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute functional movement angles (3-sensor)."""
        angles = {}
        
        for joint_name, (reference, target) in self.joint_pairs.items():
            ref_idx = self.get_sensor_index(reference)
            targ_idx = self.get_sensor_index(target)
            
            ref_pos = positions[:, ref_idx, :]
            targ_pos = positions[:, targ_idx, :]
            movement_vector = targ_pos - ref_pos
            
            # Functional angles with proper naming
            # ELEVATION: Angle above/below horizontal
            horizontal_distance = np.sqrt(movement_vector[:, 0]**2 + movement_vector[:, 1]**2)
            elevation = np.arctan2(movement_vector[:, 2], horizontal_distance)
            elevation_deg = np.degrees(elevation)
            
            # AZIMUTH: Direction in horizontal plane
            azimuth = np.arctan2(movement_vector[:, 1], movement_vector[:, 0])
            azimuth_deg = np.degrees(azimuth)
            
            # DISTANCE: Magnitude of separation
            distance = np.linalg.norm(movement_vector, axis=1)
            
            # ORIENTATION DIFFERENCE
            orientation_diff = rotations[:, targ_idx, 2] - rotations[:, ref_idx, 2]
            
            angles[f'{joint_name}_elevation'] = elevation_deg
            angles[f'{joint_name}_azimuth'] = azimuth_deg
            angles[f'{joint_name}_distance'] = distance
            angles[f'{joint_name}_orientation'] = orientation_diff
        
        return angles
    
    def compute_velocities(self, positions: np.ndarray, 
                          fs: float = None) -> np.ndarray:
        """
        Compute velocities for all sensors.
        
        Args:
            positions: (T, N_SENSORS, 3)
            fs: Sampling frequency
            
        Returns:
            velocities: (T-1, N_SENSORS, 3)
        """
        if fs is None:
            from config import SAMPLING_RATE
            fs = SAMPLING_RATE
        
        dt = 1.0 / fs
        velocities = np.diff(positions, axis=0) / dt
        return velocities
    
    def compute_accelerations(self, positions: np.ndarray,
                             fs: float = None) -> np.ndarray:
        """
        Compute accelerations for all sensors.
        
        Args:
            positions: (T, N_SENSORS, 3)
            fs: Sampling frequency
            
        Returns:
            accelerations: (T-2, N_SENSORS, 3)
        """
        if fs is None:
            from config import SAMPLING_RATE
            fs = SAMPLING_RATE
        
        dt = 1.0 / fs
        velocities = self.compute_velocities(positions, fs)
        accelerations = np.diff(velocities, axis=0) / dt
        return accelerations
    
    def compute_rom(self, joint_angles: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute Range of Motion for each joint.
        
        Args:
            joint_angles: Dictionary from compute_joint_angles()
            
        Returns:
            Dictionary of ROM values (max - min)
        """
        rom = {}
        for joint, angles in joint_angles.items():
            rom[f'{joint}_rom'] = np.ptp(angles)  # peak-to-peak
            rom[f'{joint}_mean'] = np.mean(angles)
            rom[f'{joint}_std'] = np.std(angles)
            rom[f'{joint}_max'] = np.max(angles)
            rom[f'{joint}_min'] = np.min(angles)
        return rom
    
    def compute_peak_velocities(self, velocities: np.ndarray) -> Dict[str, float]:
        """Compute peak velocity for each sensor."""
        peak_vels = {}
        for i, sensor in enumerate(self.sensor_names):
            vel_mag = np.linalg.norm(velocities[:, i, :], axis=1)
            peak_vels[f'{sensor}_peak_vel'] = np.max(vel_mag)
            peak_vels[f'{sensor}_mean_vel'] = np.mean(vel_mag)
        return peak_vels
    
    def extract_biomechanical_features(self, movement_data: np.ndarray, 
                                      time_elapsed: np.ndarray = None) -> Dict[str, float]:
        """
        Extract all biomechanical features from movement data.
        
        Args:
            movement_data: (T, 18) array 
                          [HPosX,HPosY,HPosZ,HRotX,HRotY,HRotZ,
                           LPosX,LPosY,LPosZ,LRotX,LRotY,LRotZ,
                           RPosX,RPosY,RPosZ,RRotX,RRotY,RRotZ]
            time_elapsed: (T,) array of timestamps (optional)
        
        Returns:
            Dictionary of biomechanical features
        """
        # Extract positions and rotations from movement data
        # Reshape to (T, N_SENSORS, 3) for positions and rotations
        T = len(movement_data)
        
        positions = np.zeros((T, N_SENSORS, 3))
        rotations = np.zeros((T, N_SENSORS, 3))
        
        # Extract for each sensor
        # Head: channels 0-5
        positions[:, 0, :] = movement_data[:, 0:3]   # HPosX, HPosY, HPosZ
        rotations[:, 0, :] = movement_data[:, 3:6]   # HRotX, HRotY, HRotZ
        
        # LeftHand: channels 6-11
        positions[:, 1, :] = movement_data[:, 6:9]   # LPosX, LPosY, LPosZ
        rotations[:, 1, :] = movement_data[:, 9:12]  # LRotX, LRotY, LRotZ
        
        # RightHand: channels 12-17
        positions[:, 2, :] = movement_data[:, 12:15] # RPosX, RPosY, RPosZ
        rotations[:, 2, :] = movement_data[:, 15:18] # RRotX, RRotY, RRotZ
        
        features = {}
        
        # 1. Joint angles
        joint_angles = self.compute_joint_angles(positions, rotations)
        
        # 2. ROM
        rom = self.compute_rom(joint_angles)
        features.update(rom)
        
        # 3. Velocities
        from config import SAMPLING_RATE
        velocities = self.compute_velocities(positions, fs=SAMPLING_RATE)
        peak_vels = self.compute_peak_velocities(velocities)
        features.update(peak_vels)
        
        # 4. Angular velocities
        for joint, angles in joint_angles.items():
            angular_vel = np.abs(np.diff(angles))
            features[f'{joint}_mean_ang_vel'] = np.mean(angular_vel)
            features[f'{joint}_max_ang_vel'] = np.max(angular_vel)
        
        # 5. Movement duration metrics
        if time_elapsed is not None:
            features['trial_duration'] = time_elapsed[-1] - time_elapsed[0]
        else:
            features['trial_duration'] = len(positions) / SAMPLING_RATE
        
        return features


# ===== STANDALONE USAGE EXAMPLE =====
if __name__ == "__main__":
    # Test with dummy data
    T = 500  # 5 seconds at 100 Hz
    
    # Create dummy movement data (T, 18)
    movement_data = np.random.randn(T, 18)
    
    print("="*70)
    print("Testing BiomechanicalFeatures with 3-sensor data")
    print("="*70)
    
    print(f"\nMovement data shape: {movement_data.shape}")
    print(f"Expected: ({T}, 18)")
    
    # Initialize
    sensor_names = ['Head', 'LeftHand', 'RightHand']
    extractor = BiomechanicalFeatures(sensor_names)
    
    print(f"\nSensor names: {extractor.sensor_names}")
    print(f"Joint pairs: {list(extractor.joint_pairs.keys())}")
    print(f"Using anatomical names: {extractor.use_anatomical_names}")
    
    # Extract features
    features = extractor.extract_biomechanical_features(movement_data)
    
    print(f"\n✓ Extracted {len(features)} features")
    
    print("\nSample features:")
    for i, (key, val) in enumerate(list(features.items())[:10]):
        print(f"  {key}: {val:.4f}")
    
    print("\n" + "="*70)
    print("Feature categories:")
    
    # Show feature categories
    elevation_features = [k for k in features.keys() if 'elevation' in k]
    azimuth_features = [k for k in features.keys() if 'azimuth' in k]
    distance_features = [k for k in features.keys() if 'distance' in k]
    velocity_features = [k for k in features.keys() if 'vel' in k]
    
    print(f"\nElevation features: {len(elevation_features)}")
    print(f"  {elevation_features}")
    
    print(f"\nAzimuth features: {len(azimuth_features)}")
    print(f"  {azimuth_features}")
    
    print(f"\nDistance features: {len(distance_features)}")
    print(f"  {distance_features}")
    
    print(f"\nVelocity features: {len(velocity_features)}")
    print(f"  {velocity_features[:5]}")
    
    print("\n" + "="*70)
    print("✓ Test complete!")
    print("="*70)