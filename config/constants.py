"""
Central configuration constants for the XDash project.
"""
import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Channel names (18 features: 3 sensors × 6 DoF)
CHAN_NAME = [
    "head_pos_x", "head_pos_y", "head_pos_z",
    "head_rot_x", "head_rot_y", "head_rot_z",
    "right_hand_pos_x", "right_hand_pos_y", "right_hand_pos_z",
    "right_hand_rot_x", "right_hand_rot_y", "right_hand_rot_z",
    "left_hand_pos_x", "left_hand_pos_y", "left_hand_pos_z",
    "left_hand_rot_x", "left_hand_rot_y", "left_hand_rot_z",
]

# Number of degrees of freedom
DOFS = 18

# Task names
TASK_NAMES = {
    1: "jar_opening",
    2: "key_turning",
    3: "cleaning",
    4: "back_washing",
    5: "cutting",
    6: "hammering"
}

# Paradigm names
PARADIGM_NAMES = {
    1: "patients_vs_controls",
    2: "rct_vs_controls",
    3: "other_conditions_vs_controls",
    4: "rct_vs_other_conditions"
}

# Model names
MODEL_NAMES = {
    3: "HMM",
    4: "CNN",
    5: "RNN"
}

# Feature filter names
FEATURE_FILTER_NAMES = {
    1: "padding",
    2: "truncating",
    3: "dtw_embedding",
    4: "sliding_window"
}