"""
XDash classification experiments.

Usage
-----
# Run an experiment (dataset defaults to xdash)
python main.py --task 1 --paradigm 1 --model rnn --method truncate

# Use a different dataset
python main.py --dataset raw_data2 --task 1 --paradigm 1 --model rnn --method truncate

# Generate pickled datasets from raw data first
python main.py --dataset xdash --ingest
python main.py --dataset raw_data2 --ingest --tasks 1 2

# Full options
python main.py --task 1 --paradigm 1 --model rnn --method sliding_window \\
    --window-size 300 --overlap 0.3 --diagnostics --save-figures
"""
import argparse
import sys

from dataio.ingestion import load_dataset_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="XDash classification experiments"
    )

    p.add_argument("--dataset", default="xdash",
                   help="Dataset name (must match a folder in datasets/). Default: xdash")

    # Ingest mode — generate pickled datasets from raw data
    p.add_argument("--ingest", action="store_true",
                   help="Run raw -> pickled ingestion for the selected dataset, then exit")
    p.add_argument("--ingest-tasks", type=int, nargs="+", default=None,
                   help="Tasks to ingest (default: all tasks in dataset config)")
    p.add_argument("--dry-run", action="store_true",
                   help="Ingest dry-run: print info without writing files")

    # Experiment args (required unless --ingest)
    p.add_argument("-t", "--task", type=int, default=None)
    p.add_argument("-p", "--paradigm", type=int, default=None)
    p.add_argument("-m", "--model", type=str, default=None,
                   choices=["hmm", "hsmm", "cnn", "rnn", "transformer"])

    # Preprocessing
    p.add_argument("-pre", "--method", default="truncate",
                   choices=["truncate", "sliding_window", "padding", "dtw_embedding",
                            "downsample_truncate", "variable_length", "phase_shift"])
    p.add_argument("--window-size", type=int, default=300)
    p.add_argument("--overlap", type=float, default=0.30)
    p.add_argument("--target-rate", type=int, default=25)
    p.add_argument("--original-rate", type=int, default=50)
    p.add_argument("--n-components", type=int, default=10)
    p.add_argument("--dtw-method", default="mds", choices=["mds", "isomap", "tsne"])
    p.add_argument("--shift-fraction", type=float, default=0.1)
    p.add_argument("--freq", type=int, default=None,
                   help="Resample to this Hz before preprocessing "
                        "(default: dataset sampling_rate)")

    # Training
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=1e-4)

    # Augmentation
    p.add_argument("--augment", action="store_true")
    p.add_argument("--augment-methods", nargs="+",
                   default=["jitter", "time_warp", "magnitude_warp"])
    p.add_argument("--n-augmentations", type=int, default=2)

    # Diagnostics / output
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--diagnostics-dir", type=str, default=None)
    p.add_argument("--save-figures", action="store_true")
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument("--save-checkpoints", action="store_true", default=False)
    p.add_argument("--hmm-csv-dir", type=str, default=None)
    p.add_argument("--data-source", default="standard",
                   choices=["standard", "event_window"])

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load dataset config — validates the dataset name early
    dataset_config = load_dataset_config(args.dataset)

    # Default --freq to dataset sampling rate
    if args.freq is None:
        args.freq = dataset_config.get("sampling_rate", 50)

    # ── Ingest mode ───────────────────────────────────────────────────────────
    if args.ingest:
        from dataio.ingestion import ingest
        ingest(args.dataset, tasks=args.ingest_tasks, dry_run=args.dry_run)
        return

    # ── Experiment mode ───────────────────────────────────────────────────────
    if args.task is None or args.paradigm is None or args.model is None:
        parser.error("--task, --paradigm, and --model are required for experiment mode.")

    valid_tasks = sorted(dataset_config["tasks"].keys())
    valid_paradigms = sorted(dataset_config["paradigms"].keys())
    if args.task not in valid_tasks:
        parser.error(f"--task must be one of {valid_tasks} for dataset '{args.dataset}'.")
    if args.paradigm not in valid_paradigms:
        parser.error(f"--paradigm must be one of {valid_paradigms} for dataset '{args.dataset}'.")

    from pipeline.runner import run_experiment
    run_experiment(args, dataset_config)


if __name__ == "__main__":
    main()
