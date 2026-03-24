"""
Path configuration for the XDash project.
"""
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Data files
PATIENT_DETAILS = DATA_DIR / "xdash_px_details.xlsx"

def get_pickled_dataset_path(task: int, data_type: str) -> Path:
    """Get path to legacy pickled dataset (one array per subject)."""
    return DATA_DIR / "pickled_datasets" / f"{data_type}_data_task{task}.pkl"


def get_event_window_path(task: int, group: str) -> Path:
    """
    Get path to event-window pickled dataset (one array per window).

    Parameters
    ----------
    task  : int   — task number (1-6)
    group : str   — 'g0' (controls) or 'g1' (patients)
    """
    return DATA_DIR / "pickled_datasets" / "event_window" / f"{group}_data_task{task}.pkl"

# Create necessary directories
for directory in [DATA_DIR, EXPERIMENTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def get_experiment_dir(experiment_name: str) -> Path:
    """
    Get or create experiment directory.
    
    Parameters
    ----------
    experiment_name : str
        Name of experiment (will be timestamped)
    
    Returns
    -------
    Path
        Experiment directory path
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = EXPERIMENTS_DIR / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def get_latest_experiment(pattern: str = "*") -> Path:
    """
    Get most recent experiment directory matching pattern.
    
    Parameters
    ----------
    pattern : str
        Glob pattern (e.g., "RNN_T1_P1_*")
    
    Returns
    -------
    Path
        Latest experiment directory
    """
    experiments = sorted(EXPERIMENTS_DIR.glob(pattern))
    if not experiments:
        raise FileNotFoundError(f"No experiments found matching: {pattern}")
    return experiments[-1]