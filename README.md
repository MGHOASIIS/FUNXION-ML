# FUNXION-ML: A Scalable ML Pipeline for Motion-Capture (Inertial now) Classification

> A dataset-agnostic, model-agnostic pipeline for classifying pathology from
> motion-tracking time series, built to grow to new datasets and new model
> architectures without touching the core pipeline code.

## Project overview

This is a research ML pipeline for classifying movement-based pathology from
time-series sensor data (currently XR motion-capture). The architecture is 
split into four independent extension points: dataset, preprocessing strategy, model and metric for classifiers so adding any one of them is a matter of writing one small adapter/
import/function:

- **Datasets** are plugged in under `datasets/{name}/`, each one just
  declares its channels, tasks, and classification paradigms in a
  `dataset.yaml` and provides an `ingest()` function. `dataio/` and
  `pipeline/` consume any dataset identically through that config.
- **Preprocessing strategies** are plugged in under `dataio/preprocessors.py`
  by subclassing `BasePreprocessor` and registering in `PreprocessorFactory.create()`. 
  Every strategy works with every model and every dataset for free.
- **Models** are plugged in under `models/` by subclassing
  `models/base_model.py:BaseModel` and registering the class in
  `pipeline/runner.py:create_model()`. Every model gets the same CLI surface,
  the same preprocessing options, the same LOO-CV evaluation and diagnostics
  for free.
- **Metrics** are computed in `utils/metrics.py` (`compute_metrics`/`compute_multilabel_metrics`).
  A selectable metrics registry (add/remove metrics without touching call sites) is planned.

Today the pipeline ships with one fully-implemented dataset (**xdash**), a
second in progress (**funxion**), seven preprocessing strategies, and five
models (**HMM, HSMM, 1D-CNN, RNN, Transformer**), see each section below,
and [`DATA_SETUP.md`](DATA_SETUP.md) for the exact steps to add a dataset.

## Datasets

### XDash (N=60) : implemented

- **Modality**: XR headset with two hand controllers producing 6-DoF kinematic data (18
channels: head + left hand + right hand, each with position + rotation, at 50Hz)
- **Population**: 40 patients (RCT, arthritis, bursitis, tendonitis) + 20 controls.
- **Tasks**: 1 jar opening, 2 key turning, 3 cleaning, 4 back washing, 5 cutting, 6 hammering.
- **Classification paradigms**: 1 patients vs. controls, 2 RCT vs. controls,
  3 other conditions vs. controls, 4 RCT vs. other conditions (all binary).

See `datasets/xdash/dataset.yaml` for the exact channel list, task/paradigm
definitions, and subject-filtering rules, and `datasets/xdash/` generally as
the reference example to copy when adding a new dataset.

### funxion (in progress)

The next dataset. Raw recordings already exist under `storage/raw/funxion/`,
but the `datasets/funxion/` adapter (`dataset.yaml` + `ingest.py`) hasn't
been built yet, that's a separate follow-up once its label metadata is
ready. Two things distinguish it from XDash:

- **Modality**: camera-based body-keypoint tracking 
  (34/38-point skeletons across two cameras), and XR headset + controllers + 6 sensors:
  the first real test of `dataset.yaml`'s channel abstraction across a different modality.
- **Population**: 300 patients (100 Spine-injured, 100 Shoulder-injured, 100 Knee-injured)
  + 100 controls. (Collection in progress)
- **Tasks**: 
- **Multi-label classification, not binary**: 3 injury types + a control
  label, and a subject can have more than one injury at once. XDash's
  paradigms are all `g1_filter`/`g0_filter` binary comparisons; funxion needs
  a `type: multilabel` paradigm: see "Classification paradigm types" below.

### Adding a new dataset

See [`DATA_SETUP.md`](DATA_SETUP.md) for the full walkthrough: write a
`dataset.yaml` (tasks, paradigms, channels, sampling rate) and an `ingest()`
function under `datasets/{name}/`. No changes to `dataio/`, `models/`, or
`pipeline/` are required — that's the whole point of the adapter pattern.

### Classification paradigm types

The pipeline isn't limited to binary classifiers — `dataset.yaml` supports
three paradigm shapes, chosen per paradigm via a `type` field:

- **`binary`** (default, unchanged) — the original `g1_filter`/`g0_filter`
  shape. Exactly two mutually exclusive groups (e.g. patients vs. controls).
  Every model uses a 2-way softmax + `CrossEntropyLoss`.
- **`multilabel`** — an arbitrary `labels:` map of named groups (reusing the
  same filter schema as `g1_filter`/`g0_filter`), where a subject can belong
  to more than one label at once. This is what funxion needs (3 injury types
  + control, non-exclusive):

  ```yaml
  paradigms:
    5:
      name: injury_multilabel
      type: multilabel
      labels:
        control:   {source: control, filter: all}
        injury_a:  {source: patient, filter: metadata, column: injury_a, values: [1]}
        injury_b:  {source: patient, filter: metadata, column: injury_b, values: [1]}
        injury_c:  {source: patient, filter: metadata, column: injury_c, values: [1]}
  ```

  **Status: implemented end-to-end**, with dedicated `pytest` coverage.
  `ParadigmSelector.select_labels()`, every preprocessor's
  `prepare_data_multilabel()`, all five models (CNN/RNN/Transformer via
  per-label sigmoid heads + `BCEWithLogitsLoss`; HMM/HSMM via one
  one-vs-rest model pair per label), `utils/metrics.py:compute_multilabel_metrics`
  (subset accuracy, Hamming loss, macro/micro F1, per-label balanced
  accuracy/AUC), `utils/evaluator.py:evaluate_multilabel` +
  `Visualizer.plot_multilabel_confusion_matrices`, `pipeline/io.py`'s
  wide-format predictions CSV, and `pipeline/runner.py`'s paradigm-type
  dispatch all work together — verified via `tests/test_paradigms.py`,
  `test_preprocessors.py`, `test_metrics.py`, `test_evaluator.py`,
  `test_e2e_training.py::TestMultilabelEndToEnd`, `test_e2e_io.py`, and full
  synthetic end-to-end pipeline runs. `--diagnostics` is not yet supported
  for multi-label paradigms — it's cleanly skipped with a message rather
  than half-working.

- **`multiclass`** *(planned)* — the same `labels:` map as `multilabel`, but
  mutually exclusive: a subject belongs to exactly one of N groups (e.g. one
  of 4 diagnosis categories, never more than one). This reuses the
  `multilabel` paradigm schema and the same softmax/`CrossEntropyLoss`
  machinery the `binary` path already uses for 2 classes — generalizing to
  N classes needs no new loss function, and sklearn's `balanced_accuracy_score`/
  `confusion_matrix` already handle N-way classification natively, so no new
  metrics functions are needed either (unlike `multilabel`, which required
  genuinely new metrics for partial/overlapping correctness).

**Why `binary` wasn't merged into `multiclass`/`multilabel`, even though a
2-class softmax is mathematically just a special case:** `binary` is the
production path every existing XDash paradigm, published result, and saved
checkpoint depends on, so it must keep behaving exactly as it does today.
It also has narrower conventions baked in everywhere that a generic N-class
path doesn't share by default — `y_proba` is a single scalar (probability
of class 1) rather than an `(N, n_classes)` matrix, checkpoints save a
`'balanced_accuracy'` key that `inference/inference.py` reads directly,
subject IDs are literally prefixed `g1_`/`g0_`, and the YAML shape is
`g1_filter`/`g0_filter` rather than a `labels:` map. Generalizing `binary`
away would mean either rewriting those downstream readers to expect the
generic shape, or special-casing N=2 inside the generic path to reproduce
the old conventions anyway — neither removes the special case, it just
relocates it. So `binary`, `multilabel`, and `multiclass` are kept as three
explicit, additive paradigm types rather than one collapsed abstraction.

## Project structure

```
FUNXION-ML/
├── config/            # constants.py, hyperparameter.py (per-model param grids), paths.py
├── dataio/            # dataset-agnostic ingestion/paradigm/preprocessing/transform logic
├── datasets/          # one adapter folder per dataset (dataset.yaml + ingest.py), e.g. datasets/xdash/
├── models/            # base_model.py + hmm/hsmm/cnn/rnn/transformer... implementations
├── features/          # handcrafted feature extractors (biomechanical, spectral, entropy, ...)
├── pipeline/          # runner.py wires dataio -> model -> training/evaluation for one experiment
├── inference/         # run a trained checkpoint on new/held-out subjects
├── utils/             # metrics, diagnostics (overfitting/gradient/activation), importance, plotting
├── scripts/           # one-off analysis scripts, grouped by area (data_prep/, hmm/, nn/, results/)
├── paper/             # scripts that reproduce results/figures/tables for a paper (see paper/README.md)
├── hpc/               # SLURM setup/submission scripts
├── storage/           # gitignored: raw/pickled/results data (see DATA_SETUP.md)
├── main.py            # CLI entry point for ingestion + experiments
└── requirements.txt
```

`scripts/nn/` and `paper/nn_jmir_xr/` target an older
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
- CUDA-capable GPU (recommended for deep networks training)

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

## Preprocessing strategies

Every strategy implements `BasePreprocessor._prepare_from_sequences()`
(`dataio/preprocessors.py`), which is what lets `pipeline/runner.py` apply
any strategy to any model and any dataset — including both binary
(`prepare_data`) and multi-label (`prepare_data_multilabel`) paradigms — with
no method-specific branching elsewhere in the pipeline.

Currently supported (`-pre/--method`):

- **`truncate`**: crop every sequence to the shortest one (`TruncatePreprocessor`).
- **`sliding_window`**: split each sequence into overlapping fixed-size windows (`SlidingWindowPreprocessor`).
- **`padding`**: zero-pad every sequence to the longest one (`PaddingPreprocessor`).
- **`variable_length`**: keep each subject's full-length recording, for models (HMM/HSMM) that natively handle variable-length sequences (`VariableLengthPreprocessor`).
- **`dtw_embedding`**: pairwise DTW distance matrix + MDS/Isomap/t-SNE embedding into a fixed-size vector per subject (`DTWEmbeddingPreprocessor`).
- **`phase_shift`**: circularly shift each recording before truncating, to test sensitivity to where the recording starts within the movement cycle (`PhaseShiftPreprocessor`).
- **`downsample_truncate`**: truncate combined with explicit resampling (see `--target-rate`/`--original-rate`).

Two wrappers compose with any of the above rather than being methods
themselves: `ResamplingWrapper` (downsamples before delegating to any inner
strategy — used automatically whenever `--freq` is below the dataset's
native rate) and `AugmentedPreprocessor` (`--augment`, applies
jitter/time-warp/magnitude-warp on top of any base strategy's output).

### Adding a new preprocessing strategy

1. Subclass `BasePreprocessor` in `dataio/preprocessors.py`, implementing
   `_prepare_from_sequences(all_tensors, y, subject_ids)` — this method
   never needs to know whether `y` is a 1-D binary vector or an
   `(N, n_labels)` multi-label matrix, since it only ever passes it through
   untouched; `prepare_data()`/`prepare_data_multilabel()` are handled for
   you by the base class.
2. Register the method name in `PreprocessorFactory.create()`.
3. Add the name to the `-pre/--method` choices list in `main.py`.

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

Every model trains on `binary` or `multilabel` paradigms transparently — a
`multilabel=True, label_names=[...]` constructor flag switches CNN/RNN/
Transformer to per-label sigmoid outputs + `BCEWithLogitsLoss`, and HMM/HSMM
to N one-vs-rest model pairs instead of a single class-0-vs-class-1 pair.
No separate multi-label model classes exist.

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
