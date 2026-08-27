# FUNXION-ML: A Scalable ML Pipeline for Motion-Capture Classification

> A dataset-agnostic, model-agnostic pipeline for classifying pathology from
> motion-tracking time series, built to grow to new datasets and new model
> architectures without touching the core pipeline code.

## Project overview

This is a research ML pipeline for classifying movement-based pathology from
time-series sensor data (currently XR/motion-capture kinematics). The
architecture is split so that **adding a new dataset or a new model is a
matter of writing one small adapter, not modifying `dataio/` or
`pipeline/`**:

- **Datasets** are plugged in under `datasets/{name}/` — each one just
  declares its channels, tasks, and classification paradigms in a
  `dataset.yaml` and provides an `ingest()` function. `dataio/` and
  `pipeline/` consume any dataset identically through that config.
- **Models** are plugged in under `models/` by subclassing
  `models/base_model.py:BaseModel` and registering the class in
  `pipeline/runner.py:create_model()`. Every model gets the same CLI surface,
  the same preprocessing options, the same LOO-CV evaluation and diagnostics
  for free.

Today the pipeline ships with one dataset (**XDash**) and five models
(**HMM, HSMM, 1D-CNN, RNN, Transformer**) — see below for both, and see
[`DATA_SETUP.md`](DATA_SETUP.md) for the exact steps to add another dataset.

### Current dataset: XDash (N=60)

Subjects perform six standardized functional tasks while wearing an XR
headset with two hand controllers, producing 6-DoF kinematic data (18
channels: head + left hand + right hand, each with position + rotation, at
50Hz).

- **Population**: 40 patients (RCT, arthritis, bursitis, tendonitis) + 20 controls.
- **Tasks**: 1 jar opening, 2 key turning, 3 cleaning, 4 back washing, 5 cutting, 6 hammering.
- **Classification paradigms**: 1 patients vs. controls, 2 RCT vs. controls,
  3 other conditions vs. controls, 4 RCT vs. other conditions.

See `datasets/xdash/dataset.yaml` for the exact channel list, task/paradigm
definitions, and subject-filtering rules — and `datasets/xdash/` generally
as the reference example to copy when adding a new dataset.

## Project structure

```
FUNXION-ML/
├── config/            # constants.py, hyperparameter.py (per-model param grids), paths.py
├── dataio/            # dataset-agnostic ingestion/paradigm/preprocessing/transform logic
├── datasets/          # one adapter folder per dataset (dataset.yaml + ingest.py), e.g. datasets/xdash/
├── models/            # base_model.py + hmm/hsmm/cnn/rnn/transformer implementations
├── features/          # handcrafted feature extractors (biomechanical, spectral, entropy, ...)
├── pipeline/          # runner.py wires dataio -> model -> training/evaluation for one experiment
├── inference/         # run a trained checkpoint on new/held-out subjects
├── utils/             # metrics, diagnostics (overfitting/gradient/activation), importance, plotting
├── scripts/           # one-off analysis scripts, grouped by area (data_prep/, hmm/, nn/, results/)
├── paper/             # scripts that reproduce one specific figure/table for a paper (see paper/README.md)
├── hpc/               # SLURM setup/submission scripts
├── storage/           # gitignored: raw/pickled/results data (see DATA_SETUP.md)
├── main.py            # CLI entry point for ingestion + experiments
└── requirements.txt
```

`scripts/nn/`, `scripts/results/`, and `paper/nn_jmir_xr/` target an older
`experiments_from_hpc/` CNN/RNN/Transformer results tree from an earlier
project phase, separate from the current HMM/HSMM-centric `storage/results/`
layout used everywhere else — kept for reference, not part of the active
pipeline. Likewise `features/extract_features.py` and
`features/feature_config.py` are unused legacy files kept for reference; the
feature extractors actually in use are wired in through
`scripts/data_prep/extract_subject_features.py`.

See `all_cmds.txt` for a full list of runnable commands across the repo.

## Getting started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for CNN/RNN/Transformer training)

### Installation

```bash
git clone https://github.com/MGHOASIIS/FUNXION-ML.git
cd FUNXION-ML
pip install -r requirements.txt
python main.py --help
```

Get access to the raw data and place it under `storage/` — see
[`DATA_SETUP.md`](DATA_SETUP.md).

### Usage

```bash
# Generate pickled datasets from raw data (dataset defaults to xdash)
python main.py --dataset xdash --ingest
python main.py --dataset xdash --ingest --ingest-tasks 1 2

# Run a single experiment
python main.py --task 1 --paradigm 1 --model rnn --method truncate

# Sliding-window preprocessing with diagnostics and saved figures
python main.py --task 1 --paradigm 1 --model rnn --method sliding_window \
    --window-size 300 --overlap 0.3 --diagnostics --save-figures

# Save model checkpoints
python main.py --task 1 --paradigm 1 --model cnn --method truncate --save-checkpoints
```

Key flags (see `python main.py --help` for the full set):

| Flag | Purpose |
|---|---|
| `--dataset` | Dataset name, must match a folder under `datasets/` (default: `xdash`) |
| `-t/--task`, `-p/--paradigm` | Which task/paradigm to run, per `dataset.yaml` |
| `-m/--model` | `hmm`, `hsmm`, `cnn`, `rnn`, or `transformer` |
| `-pre/--method` | Preprocessing: `truncate`, `sliding_window`, `padding`, `dtw_embedding`, `downsample_truncate`, `variable_length`, `phase_shift` |
| `--diagnostics` | Run the comprehensive diagnostic suite (overfitting, gradients, activations, feature importance) |
| `--augment` | Enable data augmentation (`jitter`, `time_warp`, `magnitude_warp`, ...) |
| `--save-checkpoints` | Persist model checkpoints under `storage/results/{dataset}/experiments/` |

### HPC usage (SLURM)

```bash
bash hpc/setup_env.sh          # once, on the login node
bash hpc/submit_all.sh         # submit all task x paradigm x model combinations
bash hpc/submit_ablations.sh   # submit ablation runs
squeue -u <username>
```

### Inference on new subjects

```bash
python inference/prepare_test_data.py --data-dir storage/raw/xdash/test_data/ --subject-id PX_NEW
python inference/run_all_inference.py --test-subject-dir storage/raw/xdash/test_data/PX_NEW --subject-id PX_NEW
```

### Tests

```bash
pytest tests/
```

## Models

Every model implements the same `models/base_model.py:BaseModel` interface
(`train_and_evaluate`, `compute_feature_importance`), which is what lets
`pipeline/runner.py` run LOO-CV, diagnostics, checkpointing, and feature
importance identically regardless of which model is selected.

Currently supported (`-m/--model`):

- **HMM / HSMM**: Generative probabilistic models with Gaussian emissions
  (HSMM adds explicit state-duration modeling); see `models/hmm_model.py`,
  `models/hsmm_model.py`, `models/state_sequence_analysis.py`.
- **1D-CNN**: Convolutional trunk with GRU head (`models/cnn_model.py`).
- **RNN**: GRU/LSTM, optionally bidirectional, multiple pooling strategies (`models/rnn_model.py`).
- **Transformer**: `models/transformer_model.py`.

Hyperparameter grids per model live in `config/hyperparameter.py`.

### Adding a new model

1. Subclass `BaseModel` in a new `models/{name}_model.py`, implementing
   `train_and_evaluate()` and `compute_feature_importance()`.
2. Register it in `pipeline/runner.py:create_model()`.
3. Add the name to the `-m/--model` choices list in `main.py`.

No other file needs to change — the CLI, preprocessing, evaluation, and
diagnostics all operate on the `BaseModel` interface, not on any specific
model.

## Diagnostics

`--diagnostics` runs `utils/comprehensive_monitor.py`, which covers:

- Overfitting analysis (`utils/overfitting_detection.py`) — important given N=60.
- Gradient/activation analysis (`utils/model_diagnostics.py`).
- Feature importance (`utils/importance.py`).
- Plotting (`utils/visualization.py`).

## Further reading

- [`DATA_SETUP.md`](DATA_SETUP.md) — where data lives, how to add a new dataset.
- [`paper/README.md`](paper/README.md) — conventions for paper-specific figure/table scripts.
- `all_cmds.txt` — every runnable command in the repo, by subdirectory.
