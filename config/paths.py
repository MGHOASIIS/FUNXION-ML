"""
Path configuration.

All heavy / private data (raw recordings, pickled datasets, results) lives
under STORAGE_DIR, which is gitignored.  Set the env var XDASH_STORAGE_DIR
to relocate storage to a different drive or HPC scratch space.

Code paths (dataset profiles, model source, scripts) remain under PROJECT_ROOT
and are tracked in git as normal.
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

# Override via env var for HPC / custom storage locations.
STORAGE_DIR = Path(os.environ.get("XDASH_STORAGE_DIR", str(PROJECT_ROOT / "storage")))


# ---------------------------------------------------------------------------
# Dataset-namespaced accessors
# ---------------------------------------------------------------------------

def get_dataset_config_path(dataset: str) -> Path:
    return DATASETS_DIR / dataset / "dataset.yaml"


def get_raw_dir(dataset: str) -> Path:
    return STORAGE_DIR / "raw" / dataset


def get_pickled_dir(dataset: str) -> Path:
    return STORAGE_DIR / "pickled" / dataset


def get_results_dir(dataset: str) -> Path:
    return STORAGE_DIR / "results" / dataset


def get_experiments_dir(dataset: str) -> Path:
    return get_results_dir(dataset) / "experiments"


def get_metadata_path(dataset: str, filename: str) -> Path:
    return get_raw_dir(dataset) / filename


def get_pickled_dataset_path(task: int, data_type: str, dataset: str = "xdash") -> Path:
    """Return path to a per-task pickled dataset file."""
    return get_pickled_dir(dataset) / f"{data_type}_data_task{task}.pkl"


def get_event_window_path(task: int, group: str, dataset: str = "xdash") -> Path:
    """Return path to an event-window pickled dataset file."""
    return get_pickled_dir(dataset) / "event_window" / f"{group}_data_task{task}.pkl"


def get_paper_dir(area: str) -> Path:
    """Output dir for generated paper figures/tables (see paper/{area}/ for
    the scripts that write here). Not dataset-namespaced — a paper figure
    may combine results across datasets/tasks."""
    return STORAGE_DIR / "paper" / area
