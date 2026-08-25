"""
Pipeline-level constants. Dataset-specific values (tasks, paradigms, channels)
live in datasets/{name}/dataset.yaml and are loaded at runtime via
dataio.ingestion.load_dataset_config(). Model classes receive channel_names
explicitly (from dataset_config["channels"]) rather than importing a fixed
default from here.
"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAMES = {
    3: "HMM",
    4: "CNN",
    5: "RNN",
    6: "Transformer",
}

FEATURE_FILTER_NAMES = {
    1: "padding",
    2: "truncating",
    3: "dtw_embedding",
    4: "sliding_window",
    5: "phase_shift",
}
