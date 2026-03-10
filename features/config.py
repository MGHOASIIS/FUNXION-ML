"""
Configuration file for feature extraction system
For 3-sensor setup: Head, LeftHand, RightHand
Data format: Position + Rotation

IMPORTANT: Your data format is:
HPosX,HPosY,HPosZ,HRotX,HRotY,HRotZ,
LPosX,LPosY,LPosZ,LRotX,LRotY,LRotZ,
RPosX,RPosY,RPosZ,RRotX,RRotY,RRotZ

Sensor order in data: Head, LeftHand, RightHand (H, L, R)
"""

import numpy as np

# ===== SENSOR CONFIGURATION =====
N_SENSORS = 3
SENSOR_NAMES = ['Head', 'LeftHand', 'RightHand']

SENSOR_INDICES = {
    'Head': 0,       
    'LeftHand': 1,   
    'RightHand': 2   
}

# ===== SAMPLING RATE =====
SAMPLING_RATE = 50.0 

# ===== DATA FORMAT =====
# Your data format: HPosX,HPosY,HPosZ,HRotX,HRotY,HRotZ,LPosX,LPosY,LPosZ,LRotX,LRotY,LRotZ,RPosX,RPosY,RPosZ,RRotX,RRotY,RRotZ
# Total: 18 channels (3 sensors × 6 values per sensor)

# Each sensor has 6 values: PosX, PosY, PosZ, RotX, RotY, RotZ

# ===== CHANNEL INDICES =====
# Position indices for all 3 sensors
# Head positions: 0, 1, 2
# LeftHand positions: 6, 7, 8
# RightHand positions: 12, 13, 14
POS_INDICES = [0, 1, 2, 6, 7, 8, 12, 13, 14]

# Rotation indices for all 3 sensors
# Head rotations: 3, 4, 5
# LeftHand rotations: 9, 10, 11
# RightHand rotations: 15, 16, 17
ROT_INDICES = [3, 4, 5, 9, 10, 11, 15, 16, 17]

HEAD_POS_INDICES = [0, 1, 2]
HEAD_ROT_INDICES = [3, 4, 5]

LEFTHAND_POS_INDICES = [6, 7, 8]
LEFTHAND_ROT_INDICES = [9, 10, 11]

RIGHTHAND_POS_INDICES = [12, 13, 14]
RIGHTHAND_ROT_INDICES = [15, 16, 17]

# ===== JOINT PAIRS =====
# For 3-sensor functional relationships
JOINT_PAIRS = {
    'left_reach': ('Head', 'LeftHand'),      # Head → LeftHand
    'right_reach': ('Head', 'RightHand'),    # Head → RightHand
    'bilateral': ('LeftHand', 'RightHand')   # LeftHand ↔ RightHand
}

# ===== BILATERAL PAIRS FOR ASYMMETRY =====
BILATERAL_PAIRS = {
    'hand': ('LeftHand', 'RightHand', 1, 2)  # (left_name, right_name, left_idx, right_idx)
}

# ===== DATA STRUCTURE SUMMARY =====
"""
Channel Layout (18 channels total):
  0-2:   Head Position (X, Y, Z)
  3-5:   Head Rotation (X, Y, Z)
  6-8:   LeftHand Position (X, Y, Z)
  9-11:  LeftHand Rotation (X, Y, Z)
  12-14: RightHand Position (X, Y, Z)
  15-17: RightHand Rotation (X, Y, Z)

Sensor Order in Data:
  Sensor 0: Head      (channels 0-5)
  Sensor 1: LeftHand  (channels 6-11)
  Sensor 2: RightHand (channels 12-17)
"""

# ===== HELPER FUNCTION =====
def get_sensor_data(movement_data: np.ndarray, sensor_name: str) -> tuple:
    """
    Extract position and rotation data for a specific sensor.
    
    Args:
        movement_data: (T, 18) array
        sensor_name: 'Head', 'LeftHand', or 'RightHand'
    
    Returns:
        positions: (T, 3)
        rotations: (T, 3)
    """
    idx = SENSOR_INDICES[sensor_name]
    
    if sensor_name == 'Head':
        pos = movement_data[:, HEAD_POS_INDICES]
        rot = movement_data[:, HEAD_ROT_INDICES]
    elif sensor_name == 'LeftHand':
        pos = movement_data[:, LEFTHAND_POS_INDICES]
        rot = movement_data[:, LEFTHAND_ROT_INDICES]
    elif sensor_name == 'RightHand':
        pos = movement_data[:, RIGHTHAND_POS_INDICES]
        rot = movement_data[:, RIGHTHAND_ROT_INDICES]
    else:
        raise ValueError(f"Unknown sensor: {sensor_name}")
    
    return pos, rot