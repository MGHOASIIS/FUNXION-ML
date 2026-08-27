# Data Setup

Raw data, pickled datasets, and experiment results are **not tracked in git**
(see `.gitignore`). This file explains where each piece lives and how to
reproduce it.

---

## Adding a new dataset

Convention: Choose a simple name for the dataset all small letters (spaces replaced by underscore)

Adding a dataset means writing an *adapter* — a small amount of code under
`datasets/{name}/` — without touching `dataio/`, `models/`, or `pipeline/`.

1. **Place raw data.** Put the raw files under `storage/raw/{name}/`, in
   whatever format the dataset actually comes in. No particular layout is
   required at this stage — only your `ingest.py`.

2. **Create the adapter folder.** Add `datasets/{name}/` (e.g. `datasets/funxion/`).

3. **Write `datasets/{name}/dataset.yaml`.** Same schema as
   `datasets/xdash/dataset.yaml`:
   - `name` — the same dataset name string.
   - `sampling_rate` — sampling rate in Hz.
   - `channels` — list of channel names. Any length, any naming — not tied
     to xdash's 18 channels.
   - `tasks` — `{id: name}` map of task IDs to task labels.
   - `paradigms` — `{id: {name, g1_filter, g0_filter}}`. Each filter selects
     a subset of subjects using one of four built-in filter types:
     `all`, `subject_prefix`, `metadata`, `metadata_exclude` (see
     `dataio/paradigms.py` for the exact semantics of each).
   - `metadata_file` — path to the subject-metadata file, relative to
     `storage/raw/{name}/`.
   - `exclude_subjects` — `{g1: [...], g0: [...]}` lists of subject IDs to
     drop.

4. **Write `datasets/{name}/ingest.py`.** It must expose one function:

   ```python
   def ingest(config: dict, raw_dir: Path, out_dir: Path,
              tasks: list | None, dry_run: bool) -> None:
       ...
   ```

   `config` is the parsed `dataset.yaml`. The function reads the dataset's
   raw files however it needs to, and writes pickles to `out_dir` shaped as
   `{subject_id: np.ndarray of shape (T, C)}` — that dict shape is the only
   contract `dataio` relies on. Any dataset-specific helper modules the
   adapter needs (metadata parsing, quality checks — see
   `datasets/xdash/clinical_metadata.py`, `outlier_detection.py`,
   `quality_metrics.py` for examples) are private to `datasets/{name}/` and
   are not part of the required contract.

5. **Ingest.**

   ```bash
   python main.py --dataset {name} --ingest
   ```

6. **Run any model.**

   ```bash
   python main.py --dataset {name} --task <id> --paradigm <id> \
       --model {hmm,hsmm,cnn,rnn,transformer}
   ```

   No changes to `dataio/`, `models/`, or `pipeline/` should be required.

**Known current limitation:** channel count isn't yet fully threaded from
`dataset.yaml` into `dataio/preprocessors.py` — it currently infers whether a
leading timestamp column is present via a check tuned to xdash's 18 channels.
A dataset with a very different channel layout may need extra care until
this is fixed. The steps above describe the intended contract; closing this
gap is tracked as follow-up work.

---

## Directory layout (gitignored paths, under storage/)

```
storage/
  raw/{dataset}/            ← raw CSV/xlsx exports as delivered
    events/                 ← consolidated_task{1-6}.csv (event-marker CSVs)
    test_data/               ← raw + pickled data for new test subjects
    xdash_px_details.xlsx    ← subject demographic sheet
  pickled/{dataset}/        ← patient_data_task{1-6}.pkl, control_data_task{1-6}.pkl
    event_window/            ← event-window pickled datasets
  results/{dataset}/        ← all generated outputs (checkpoints, figures, tables)
    experiments/             ← model checkpoints from training runs
  paper/{area}/              ← generated paper figures/tables (see paper/{area}/
                                 for the scripts that write here, e.g. paper/hmm/).
                                 Not dataset-namespaced.
```

---

## Getting the data

Contact the project lead for access to the private data repository or shared
drive. Once you have the data, place files under `storage/` as described
above (or point `XDASH_STORAGE_DIR` at wherever they already live).

---

## Regenerating pickled datasets

If you have the raw XDash CSV exports:

```bash
# Generate all pickled datasets from raw patient/control recordings
python main.py --dataset xdash --ingest

# Prepare a single new test subject
python inference/prepare_test_data.py \
    --data-dir storage/raw/xdash/test_data/PX_NEW/ \
    --subject-id PX_NEW
```

---

## Note on git history

Private data files committed in earlier git history (both the original
commits and a later leftover copy under `data/`) were removed from tracking
via `git rm --cached` and are now gitignored. They remain in the local
repository history but will not appear in new commits or pushes.

To fully purge them from git history (e.g. before making the repo public) use
[BFG Repo Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) or
`git filter-repo`.


## Similarly-named directories — what each one is

- **`datasets/`** — code only, tracked in git. One subfolder per dataset
  (e.g. `datasets/xdash/`) holding `dataset.yaml` (tasks, paradigms, sampling
  rate, channel list) plus that dataset's ingestion/quality-check scripts.
  Read via `dataio.ingestion.load_dataset_config(dataset)`.
- **`storage/`** — all heavy/private data, entirely gitignored. Structure:
  `storage/{raw,pickled,results}/{dataset}/`. Never hardcode these paths —
  always go through `config/paths.py` (`get_raw_dir`, `get_pickled_dir`,
  `get_results_dir`, `get_pickled_dataset_path`, `get_metadata_path`, ...).
  Set `XDASH_STORAGE_DIR` to relocate (e.g. HPC scratch space).
- **`dataio/`** — the Python package (`ingestion.py`, `paradigms.py`,
  `preprocessors.py`, `transforms.py`) that turns raw/pickled data into
  model-ready arrays. Dataset-agnostic — used identically for every dataset
  (`xdash`, and any future one). Formerly named `data/`; renamed because that
  name collided with `datasets/` and with a leftover data-artifact folder
  (see below).
- **`data/`** — no longer a package. Only still contains a few old data
  artifacts left over from before the `storage/` migration
  (`pickled_datasets/`, `events/`, `xdash_px_details.xlsx`) — these are
  gitignored and deprecated; use the `storage/` equivalents instead. Do not
  add new code or data here.