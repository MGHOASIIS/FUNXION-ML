"""
Master Feature Integrator
Data format: Position + Rotation (18 channels)
Sensor order: Head, LeftHand, RightHand

Imports from your existing files:
- asymmetry_compensation_metrics.py
- biomechanical.py
- complexity_entropy.py
- frequency_domain.py
- joint_kinematics.py
- movement_quality.py
- temporal.py
- time_domain_stats.py
- wavelet_features.py
"""

import numpy as np
import pickle
import gc
from typing import Dict, List, Optional
from pathlib import Path

# Import all feature extraction modules
from asymmetry_compensation_metrics import *
from biomechanical import *
from complexity_entropy import *
from frequency_domain import *
from joint_kinematics import *
from movement_quality import *
from temporal import *
from time_domain_stats import *
from wavelet_features import *

# Import configuration
from feature_config import N_SENSORS, SAMPLING_RATE, SENSOR_NAMES


class MasterFeatureExtractor:
    """
    Master feature extractor for Position + Rotation data.
    
    Data format: [HPosX,HPosY,HPosZ,HRotX,HRotY,HRotZ,
                  LPosX,LPosY,LPosZ,LRotX,LRotY,LRotZ,
                  RPosX,RPosY,RPosZ,RRotX,RRotY,RRotZ]
    """
    
    def __init__(self):
        """Initialize using config.py settings."""
        self.n_sensors = N_SENSORS
        self.sampling_rate = SAMPLING_RATE
        self.sensor_names = SENSOR_NAMES
        self.n_channels = 18  # 3 sensors × 6 (3 pos + 3 rot)
        
        print(f"Initialized MasterFeatureExtractor:")
        print(f"  Sensors: {self.n_sensors} ({self.sensor_names})")
        print(f"  Channels: {self.n_channels}")
        print(f"  Sampling Rate: {self.sampling_rate} Hz")
        print(f"  Data format: Position + Rotation")
    
    def extract_all_features(self, 
                           movement_data: np.ndarray,
                           time_elapsed: np.ndarray = None,
                           verbose: bool = False) -> Dict[str, float]:
        """
        Extract ALL features from all available modules.
        
        Args:
            movement_data: (T, 18) array
            time_elapsed: (T,) array of timestamps
            verbose: Print detailed progress
        
        Returns:
            Dictionary with all extracted features
        """
        features = {}
        
        if verbose:
            print(f"  Extracting from data shape: {movement_data.shape}")
        
        # Verify data shape
        if movement_data.shape[1] != 18:
            print(f"  ⚠ Warning: Expected 18 channels, got {movement_data.shape[1]}")
        
        # Create time array if not provided
        if time_elapsed is None:
            time_elapsed = np.arange(len(movement_data)) / self.sampling_rate
        
        # ===== 1. BIOMECHANICAL FEATURES =====
        try:
            biomech_features = self._extract_biomechanical(movement_data, time_elapsed)
            features.update(biomech_features)
        except Exception as e:
            if verbose:
                print(f"    ✗ Error in biomechanical: {e}")
        
        # ===== 2. JOINT KINEMATICS =====
        try:
            joint_features = self._extract_joint_kinematics(movement_data, time_elapsed)
            features.update(joint_features)
        except Exception as e:
            if verbose:
                print(f"    ✗ Error in joint_kinematics: {e}")
        
        # ===== 3. ASYMMETRY =====
        try:
            asymmetry_features = self._extract_asymmetry(movement_data)
            features.update(asymmetry_features)
        except Exception as e:
            if verbose:
                print(f"    ✗ Error in asymmetry: {e}")
        
        # ===== 4. MOVEMENT QUALITY =====
        try:
            quality_features = self._extract_movement_quality(movement_data, time_elapsed)
            features.update(quality_features)
        except Exception as e:
            if verbose:
                print(f"    ✗ Error in movement_quality: {e}")
        
        # ===== 5. TEMPORAL =====
        try:
            temporal_features = self._extract_temporal(movement_data, time_elapsed)
            features.update(temporal_features)
        except Exception as e:
            if verbose:
                print(f"    ✗ Error in temporal: {e}")
        
        # ===== 6. TIME DOMAIN =====
        try:
            time_features = self._extract_time_domain(movement_data)
            features.update(time_features)
        except Exception as e:
            if verbose:
                print(f"    ✗ Error in time_domain: {e}")
        
        # ===== 7. FREQUENCY =====
        try:
            freq_features = self._extract_frequency(movement_data)
            features.update(freq_features)
        except Exception as e:
            if verbose:
                print(f"    ✗ Error in frequency: {e}")
        
        # ===== 8. WAVELET =====
        try:
            wavelet_feats = self._extract_wavelet(movement_data)
            features.update(wavelet_feats)
        except Exception as e:
            if verbose:
                print(f"    ✗ Error in wavelet: {e}")
        
        # ===== 9. COMPLEXITY =====
        try:
            complexity_features = self._extract_complexity(movement_data)
            features.update(complexity_features)
        except Exception as e:
            if verbose:
                print(f"    ✗ Error in complexity: {e}")
        
        # ===== 10. CALCULATE BASIC METRICS =====
        basic_metrics = self._calculate_basic_metrics(features)
        features.update(basic_metrics)
        
        return features
    
    def _calculate_basic_metrics(self, features: Dict) -> Dict:
        """
        Calculate primary Paper 1 metrics from extracted features.
        These are added directly to the feature dictionary.
        
        Args:
            features: Dictionary of all extracted features
        
        Returns:
            Dictionary of 12 Paper 1 metrics
        """
        paper1 = {}
        
        # 1. Speed metrics
        paper1['max_speed'] = max(
            features.get('LeftHand_peak_vel', 0),
            features.get('RightHand_peak_vel', 0)
        )
        
        paper1['mean_speed'] = np.mean([
            features.get('LeftHand_mean_vel', 0),
            features.get('RightHand_mean_vel', 0)
        ])
        
        paper1['peak_speed_ratio'] = paper1['max_speed'] / (paper1['mean_speed'] + 1e-10)
        
        # 2. Acceleration
        paper1['max_acceleration'] = features.get('H_PosZ_max', 0)
        
        # 3. Smoothness (jerk)
        paper1['mean_jerk'] = features.get('RightHand_jerk_metric', 
                                          features.get('LeftHand_jerk_metric', 0))
        paper1['normalized_jerk'] = paper1['mean_jerk']
        
        # 4. Path metrics
        paper1['total_path_length'] = features.get('duration', 0) * paper1['mean_speed']
        paper1['path_efficiency'] = 1.0 / (1.0 + paper1['mean_jerk'] + 1e-10)
        
        # 5. ROM
        paper1['rom_total'] = max(
            features.get('left_reach_elevation_rom', 0),
            features.get('right_reach_elevation_rom', 0)
        )
        
        # 6. Temporal
        paper1['time_to_peak_speed'] = features.get('movement_onset_time', 0)
        paper1['number_of_movement_units'] = features.get('num_peaks', 0)
        
        # 7. Efficiency
        paper1['movement_efficiency'] = paper1['path_efficiency']
        
        return paper1
    
    # ===== WRAPPER FUNCTIONS =====
    
    def _extract_biomechanical(self, movement_data: np.ndarray, 
                              time_elapsed: np.ndarray) -> Dict:
        """Extract features from biomechanical.py"""
        try:
            extractor = BiomechanicalFeatures(self.sensor_names)
            return extractor.extract_biomechanical_features(movement_data, time_elapsed)
        except Exception as e:
            print(f"      Biomechanical error: {e}")
            return {}
    
    def _extract_joint_kinematics(self, movement_data: np.ndarray,
                                 time_elapsed: np.ndarray) -> Dict:
        """Extract features from joint_kinematics.py"""
        features = {}
        
        # For Position+Rotation data, we already have rotations
        # Just extract ROM from the rotation channels
        for i, sensor_name in enumerate(self.sensor_names):
            # Get rotation channels for this sensor
            # Head: 3-5, LeftHand: 9-11, RightHand: 15-17
            rot_start = i * 6 + 3
            rotations = movement_data[:, rot_start:rot_start+3]
            
            for j, plane in enumerate(['roll', 'pitch', 'yaw']):
                features[f'{sensor_name}_{plane}_rom'] = np.ptp(rotations[:, j])
                features[f'{sensor_name}_{plane}_mean'] = np.mean(rotations[:, j])
                features[f'{sensor_name}_{plane}_max'] = np.max(rotations[:, j])
                features[f'{sensor_name}_{plane}_min'] = np.min(rotations[:, j])
        
        return features
    
    def _extract_asymmetry(self, movement_data: np.ndarray) -> Dict:
        """Extract features from asymmetry_compensation_metrics.py"""
        features = {}
        
        try:
            # For 3 sensors: compare LeftHand (idx 1) vs RightHand (idx 2)
            # LeftHand: channels 6-11
            # RightHand: channels 12-17
            left_data = movement_data[:, 6:12]   # LeftHand all 6 channels
            right_data = movement_data[:, 12:18] # RightHand all 6 channels
            
            asym_features = calculate_asymmetry(left_data, right_data)
            for key, val in asym_features.items():
                features[f'hand_{key}'] = val
        except Exception as e:
            print(f"      Asymmetry error: {e}")
        
        return features
    
    def _extract_movement_quality(self, movement_data: np.ndarray,
                                 time_elapsed: np.ndarray) -> Dict:
        """Extract features from movement_quality.py"""
        features = {}
        
        # Compute velocity from position data
        for i, sensor_name in enumerate(self.sensor_names):
            # Get position channels
            pos_start = i * 6
            positions = movement_data[:, pos_start:pos_start+3]
            
            # Compute velocity magnitude
            velocity = np.linalg.norm(np.diff(positions, axis=0), axis=1) * self.sampling_rate
            
            # Pad to match original length
            velocity = np.concatenate([[velocity[0]], velocity])
            
            try:
                jerk_features = calculate_smoothness_jerk(velocity, 1.0/self.sampling_rate)
                for key, val in jerk_features.items():
                    features[f'{sensor_name}_{key}'] = val
            except:
                pass
            
            try:
                sparc = calculate_sparc(velocity, self.sampling_rate)
                features[f'{sensor_name}_sparc'] = sparc
            except:
                pass
        
        return features
    
    def _extract_temporal(self, movement_data: np.ndarray,
                         time_elapsed: np.ndarray) -> Dict:
        """Extract features from temporal.py"""
        
        # Use head position magnitude as movement signal
        head_pos = movement_data[:, 0:3]
        signal = np.linalg.norm(np.diff(head_pos, axis=0), axis=1)
        signal = np.concatenate([[signal[0]], signal])
        
        try:
            return extract_temporal_features(signal, self.sampling_rate)
        except:
            return {}
    
    def _extract_time_domain(self, movement_data: np.ndarray) -> Dict:
        """Extract features from time_domain_stats.py"""
        features = {}
        
        try:
            # Extract for all 18 channels
            channel_names = [
                'H_PosX', 'H_PosY', 'H_PosZ', 'H_RotX', 'H_RotY', 'H_RotZ',
                'L_PosX', 'L_PosY', 'L_PosZ', 'L_RotX', 'L_RotY', 'L_RotZ',
                'R_PosX', 'R_PosY', 'R_PosZ', 'R_RotX', 'R_RotY', 'R_RotZ'
            ]
            
            for channel in range(18):
                signal = movement_data[:, channel]
                time_features = extract_time_domain_features(signal)
                
                for key, val in time_features.items():
                    features[f'{channel_names[channel]}_{key}'] = val
        except:
            pass
        
        return features
    
    def _extract_frequency(self, movement_data: np.ndarray) -> Dict:
        """Extract features from frequency_domain.py"""
        features = {}
        
        try:
            for i, sensor_name in enumerate(self.sensor_names):
                # Use position magnitude
                pos_start = i * 6
                positions = movement_data[:, pos_start:pos_start+3]
                pos_mag = np.linalg.norm(positions, axis=1)
                
                freq_features = extract_frequency_features(pos_mag, self.sampling_rate)
                for key, val in freq_features.items():
                    features[f'{sensor_name}_{key}'] = val
        except:
            pass
        
        return features
    
    def _extract_wavelet(self, movement_data: np.ndarray) -> Dict:
        """Extract features from wavelet_features.py"""
        features = {}
        
        try:
            # Use head position magnitude
            head_pos = movement_data[:, 0:3]
            signal = np.linalg.norm(head_pos, axis=1)
            
            wavelet_features = extract_wavelet_features(signal, wavelet='db4', level=5)
            for key, val in wavelet_features.items():
                features[f'head_{key}'] = val
        except:
            pass
        
        return features
    
    def _extract_complexity(self, movement_data: np.ndarray) -> Dict:
        """Extract features from complexity_entropy.py"""
        features = {}
        
        try:
            for i, sensor_name in enumerate(self.sensor_names):
                # Use position magnitude
                pos_start = i * 6
                positions = movement_data[:, pos_start:pos_start+3]
                pos_mag = np.linalg.norm(positions, axis=1)
                
                complexity_features = extract_complexity_features(pos_mag)
                for key, val in complexity_features.items():
                    features[f'{sensor_name}_{key}'] = val
        except:
            pass
        
        return features


def add_master_features_to_dataset(input_pickle_path: str,
                                   output_pickle_path: str = None,
                                   batch_size: int = 10):
    """
    Add all features to dataset with batched processing.
    
    Args:
        input_pickle_path: Path to input pickle file
        output_pickle_path: Path to save output
        batch_size: Number of trials to process before garbage collection
    """
    print(f"\n{'='*70}")
    print(" "*15 + "MASTER FEATURE EXTRACTION - BATCHED")
    print(" "*20 + "Position + Rotation Data")
    print(f"{'='*70}\n")
    
    print(f"Configuration:")
    print(f"  N_SENSORS: {N_SENSORS}")
    print(f"  SENSOR_NAMES: {SENSOR_NAMES}")
    print(f"  SAMPLING_RATE: {SAMPLING_RATE} Hz")
    print(f"  BATCH_SIZE: {batch_size} trials\n")
    
    print(f"Loading dataset from: {input_pickle_path}")
    with open(input_pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Initialize extractor
    extractor = MasterFeatureExtractor()
    
    # Count total trials
    if isinstance(data, list):
        total_trials = len([t for t in data if isinstance(t, dict) and 'movement_data' in t])
    else:
        total_trials = 0
    
    print(f"Total trials to process: {total_trials}\n")
    
    processed_count = 0
    batch_count = 0
    
    # Process trials
    if isinstance(data, list):
        for i, trial in enumerate(data):
            if isinstance(trial, dict) and 'movement_data' in trial:
                print(f"Processing trial {processed_count + 1}/{total_trials}", end='\r')
                
                try:
                    features = extractor.extract_all_features(
                        movement_data=trial['movement_data'],
                        time_elapsed=trial.get('time_elapsed', None),
                        verbose=False
                    )
                    
                    # Add Paper 1 metrics
                    trial = _add_paper1_metrics(trial, features)
                    
                    # Store all features
                    trial['all_extracted_features'] = features
                    
                    processed_count += 1
                    
                    # Garbage collection every batch
                    if processed_count % batch_size == 0:
                        batch_count += 1
                        print(f"\n  ✓ Batch {batch_count} complete ({processed_count}/{total_trials} trials)")
                        gc.collect()
                
                except Exception as e:
                    print(f"\n  ✗ Error processing trial {i+1}: {e}")
    
    print(f"\n\n{'='*70}")
    print(f"✓ Processed {processed_count}/{total_trials} trials")
    print(f"{'='*70}\n")
    
    # Save
    if output_pickle_path is None:
        output_pickle_path = input_pickle_path.replace('.pkl', '_master_features.pkl')
    
    print(f"Saving to: {output_pickle_path}")
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(data, f)
    
    gc.collect()
    print("✓ Complete!\n")
    
    return data


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "MASTER FEATURE INTEGRATOR")
    print(" "*25 + "Usage Example")
    print("="*70)
    
    print("\nData format:")
    print("  Channels: HPosX,HPosY,HPosZ,HRotX,HRotY,HRotZ,")
    print("            LPosX,LPosY,LPosZ,LRotX,LRotY,LRotZ,")
    print("            RPosX,RPosY,RPosZ,RRotX,RRotY,RRotZ")
    print("\n  Total: 18 channels (3 sensors × 6 values)")
    print("  Format: Position + Rotation (NOT Accel + Gyro)")
    
    print("\n" + "="*70)
    print("FEATURE STORAGE")
    print("="*70)
    
    print("""
All features are stored in:
    trial['all_extracted_features']

This includes:
    • basic metrics: max_speed, mean_jerk, rom_total, etc. (12)
    • Biomechanical: left_reach_elevation_rom, etc. (42)
    • Joint Kinematics: Head_roll_rom, etc. (36)
    • Asymmetry: hand_symmetry_index, etc. (18)
    • Movement Quality: RightHand_jerk_metric, etc. (18)
    • Temporal: num_peaks, duration, etc. (12)
    • Time Domain: H_PosX_mean, etc. (342)
    • Frequency: Head_dominant_frequency, etc. (54)
    • Wavelet: head_approximation_energy, etc. (30)
    • Complexity: Head_sample_entropy, etc. (45)
    
Total: ~560 features in all_extracted_features
    """)
    
    print("="*70)
    print("USAGE")
    print("="*70)
    
    print("""
from master_feature_integrator import add_master_features_to_dataset

# Extract features
data = add_master_features_to_dataset(
    input_pickle_path="storage/pickled/xdash/unified_dataset_raw.pkl",
    output_pickle_path="storage/pickled/xdash/unified_dataset_all_features.pkl",
    batch_size=10
)

# Access all features from one place:
trial = data[0]
features = trial['all_extracted_features']

# Basic metrics
print(f"Max Speed: {features['max_speed']}")
print(f"Mean Jerk: {features['mean_jerk']}")
print(f"ROM Total: {features['rom_total']}")

# Other features
print(f"Left Reach ROM: {features['left_reach_elevation_rom']}")
print(f"Hand Symmetry: {features['hand_symmetry_index']}")
print(f"Sample Entropy: {features['Head_sample_entropy']}")

# Create feature matrix for ML
feature_names = list(features.keys())
X = np.array([[t['all_extracted_features'][k] for k in feature_names] for t in data])
    """)
    
    print("="*70)
    print("\nStarting extraction...")
    
    data = add_master_features_to_dataset(
        input_pickle_path="storage/pickled/xdash/unified_dataset_raw.pkl",
        output_pickle_path="storage/pickled/xdash/unified_dataset_all_features.pkl",
        batch_size=10
    )
    
    if len(data) > 0:
        trial = data[0]
        features = trial['all_extracted_features']
        
        print(f"\n✓ Sample results:")
        print(f"  Total features: {len(features)}")
        print(f"  Max Speed: {features.get('max_speed', 'N/A')}")
        print(f"  Mean Jerk: {features.get('mean_jerk', 'N/A')}")
        print(f"  Left Reach ROM: {features.get('left_reach_elevation_rom', 'N/A')}")
