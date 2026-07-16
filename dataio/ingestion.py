"""
Generic ingestion driver: raw data → pickled datasets.

Usage
-----
Programmatic (from main.py --ingest):
    from dataio.ingestion import ingest, load_dataset_config
    cfg = load_dataset_config("xdash")
    ingest("xdash", tasks=[1, 2], dry_run=False)

Each dataset provides its own ingestion logic in datasets/{name}/ingest.py.
That module must expose a function with this signature:
    ingest(config: dict, raw_dir: Path, out_dir: Path,
           tasks: list | None, dry_run: bool) -> None
"""
import importlib.util
import yaml

from config.paths import (
    get_dataset_config_path,
    get_raw_dir,
    get_pickled_dir,
)


def load_dataset_config(dataset: str) -> dict:
    """Load and return the parsed dataset.yaml for `dataset`."""
    path = get_dataset_config_path(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset config found at {path}.\n"
            f"Create datasets/{dataset}/dataset.yaml first."
        )
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Normalise task keys to int
    if "tasks" in cfg:
        cfg["tasks"] = {int(k): v for k, v in cfg["tasks"].items()}
    if "paradigms" in cfg:
        cfg["paradigms"] = {int(k): v for k, v in cfg["paradigms"].items()}
    return cfg


def ingest(dataset: str, tasks: list = None, dry_run: bool = False) -> None:
    """
    Run raw → pickled ingestion for `dataset`.

    Delegates to datasets/{dataset}/ingest.py which contains the
    dataset-specific file-parsing logic.
    """
    config = load_dataset_config(dataset)
    raw_dir = get_raw_dir(dataset)
    out_dir = get_pickled_dir(dataset)

    ingest_module = _load_ingest_module(dataset)
    ingest_module.ingest(
        config=config,
        raw_dir=raw_dir,
        out_dir=out_dir,
        tasks=tasks,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _load_ingest_module(dataset: str):
    """Dynamically load datasets/{dataset}/ingest.py."""
    ingest_path = get_dataset_config_path(dataset).parent / "ingest.py"
    if not ingest_path.exists():
        raise FileNotFoundError(
            f"No ingest.py found for dataset '{dataset}' at {ingest_path}.\n"
            f"Create datasets/{dataset}/ingest.py with an ingest() function."
        )
    spec = importlib.util.spec_from_file_location(
        f"datasets.{dataset}.ingest", ingest_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
