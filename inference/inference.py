"""
Inference script for XDash shoulder pathology classification models.

This script:
1. Loads training data → fits the same preprocessing pipeline used during training
2. Retrains a final model on ALL training data using the best hyperparameters
   from the checkpoint (since LOO-CV produces per-fold models, not a single final model)
3. Loads new test subjects and applies the fitted preprocessing
4. Outputs predictions + probabilities per subject

Usage
-----
    # Pickle dict of new subjects  {subject_id: array (T, 18)}
    python inference.py \\
        --checkpoint experiments/task1/paradigm1/CNN_T1_P1_.../model_checkpoints/best_model_BA0.85.pt \\
        --task 1 --paradigm 1 --model cnn --method truncate \\
        --test-data path/to/new_subjects.pkl

    # Directory of CSV files (one CSV per subject)
    python inference.py \\
        --checkpoint ... --task 1 --paradigm 1 --model rnn --method truncate \\
        --test-csv-dir path/to/csv_dir/

    # Single CSV file (one subject, rows=time, cols=18 DOFs or timestamp+18 DOFs)
    python inference.py \\
        --checkpoint ... --task 1 --paradigm 1 --model cnn --method truncate \\
        --test-csv path/to/subject.csv
"""
import argparse
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Allow running as `python inference/inference.py` from the project root —
# the script's own directory (inference/) is on sys.path by default, not
# the project root, so top-level packages (config, data, models) wouldn't
# otherwise be importable.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.constants import DOFS, DEVICE
from config.paths import get_pickled_dataset_path
from dataio.ingestion import load_dataset_config
from dataio.paradigms import ParadigmSelector
from dataio.preprocessors import PreprocessorFactory

# Set by main() before any helper function runs
_DATASET: str = "xdash"
_DATASET_CONFIG: dict = {}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(task: int) -> Tuple[Dict, Dict]:
    """Load patient and control pickles for a given task."""
    patient_path = get_pickled_dataset_path(task, "patient", dataset=_DATASET)
    control_path = get_pickled_dataset_path(task, "control", dataset=_DATASET)
    with open(patient_path, "rb") as f:
        patient_data = pickle.load(f)
    with open(control_path, "rb") as f:
        control_data = pickle.load(f)
    print(f"[Data] Task {task}: {len(patient_data)} patients, {len(control_data)} controls")
    return patient_data, control_data


def load_test_pkl(path: str) -> Dict:
    """Load a pickle dict of {subject_id: array (T, C)}."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[Test] {len(data)} subjects loaded from {path}")
    return data


def _has_no_header(path) -> bool:
    """Return True when the first CSV row is all numeric (no header)."""
    try:
        first_row = pd.read_csv(path, nrows=1, header=None).iloc[0]
        pd.to_numeric(first_row, errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def load_test_csv_dir(csv_dir: str) -> Dict:
    """Load one CSV per subject from a directory."""
    data = {}
    for csv_file in sorted(Path(csv_dir).glob("*.csv")):
        header = None if _has_no_header(csv_file) else 0
        df = pd.read_csv(csv_file, header=header)
        data[csv_file.stem] = df.values.astype(np.float32)
    print(f"[Test] {len(data)} subjects loaded from {csv_dir}/")
    return data


def load_test_csv(path: str) -> Dict:
    """Load a single CSV as one test subject."""
    header = None if _has_no_header(path) else 0
    df = pd.read_csv(path, header=header)
    return {Path(path).stem: df.values.astype(np.float32)}


# ---------------------------------------------------------------------------
# Preprocessing new test data with a FITTED scaler (no re-fitting)
# ---------------------------------------------------------------------------

def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float32)


def preprocess_test_truncate(
    test_data: Dict,
    scaler,
    T_seq: int,
    output_format: str,
) -> Tuple[np.ndarray, List[str]]:
    """Truncate (or zero-pad) each sequence to T_seq, then z-score."""
    arrays, subject_ids = [], []
    for sid, arr in test_data.items():
        arr = _to_numpy(arr)
        if arr.shape[1] > DOFS:
            arr = arr[:, 1:]                        # drop optional timestamp column
        T = arr.shape[0]
        if T >= T_seq:
            arr = arr[-T_seq:]                      # keep last T_seq frames (same as training)
        else:
            arr = np.pad(arr, ((T_seq - T, 0), (0, 0)))  # zero-pad at start
        arrays.append(arr)
        subject_ids.append(sid)

    X = np.stack(arrays)                            # (N, T_seq, 18)
    N, T, C = X.shape
    X_scaled = scaler.transform(X.reshape(N * T, C)).reshape(N, T, C)

    if output_format == "channels_first":
        X_scaled = X_scaled.transpose(0, 2, 1)     # (N, 18, T) for CNN
    return X_scaled, subject_ids


def preprocess_test_sliding_window(
    test_data: Dict,
    scaler,
    window_size: int,
    stride: int,
    output_format: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Extract sliding windows for each subject.
    Returns windows, a per-window subject list, and the unique subject list.
    Call aggregate_window_predictions() to collapse back to per-subject level.
    """
    windows, window_subjects, subject_ids = [], [], []
    for sid, arr in test_data.items():
        subject_ids.append(sid)
        arr = _to_numpy(arr)
        if arr.shape[1] > DOFS:
            arr = arr[:, 1:]
        T = arr.shape[0]
        for start in range(0, T - window_size + 1, stride):
            windows.append(arr[start : start + window_size])
            window_subjects.append(sid)

    if not windows:
        raise ValueError(
            "No windows extracted — all test sequences are shorter than window_size "
            f"({window_size})."
        )

    X = np.stack(windows)                          # (N_windows, W, 18)
    N, T, C = X.shape
    X_scaled = scaler.transform(X.reshape(N * T, C)).reshape(N, T, C)

    if output_format == "channels_first":
        X_scaled = X_scaled.transpose(0, 2, 1)

    return X_scaled, window_subjects, subject_ids


def aggregate_window_predictions(
    preds: np.ndarray,
    probs: np.ndarray,
    window_subjects: List[str],
    all_subjects: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse per-window predictions to per-subject via majority vote + mean probability."""
    subj_preds, subj_probs = [], []
    for sid in all_subjects:
        mask = np.array([w == sid for w in window_subjects])
        mean_prob = float(probs[mask].mean())
        subj_probs.append(mean_prob)
        subj_preds.append(int(mean_prob >= 0.5))
    return np.array(subj_preds, dtype=int), np.array(subj_probs)


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------

def build_model(model_name: str, params: Dict) -> nn.Module:
    name = model_name.upper()

    if name == "CNN":
        from models.cnn_model import CNNClassifier
        return CNNClassifier(
            in_channels=DOFS,
            conv_channels=params["conv_channels"],
            kernel_sizes=params["kernel_sizes"],
            dropout_fc=params["dropout_fc"],
            n_classes=2,
        ).to(DEVICE)

    if name == "RNN":
        from models.rnn_model import RNNClassifier
        return RNNClassifier(
            input_dim=DOFS,
            rnn_type=params["rnn_type"],
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            bidirectional=params["bidirectional"],
            dropout_rnn=params["dropout_rnn"],
            dropout_fc=params["dropout_fc"],
            pooling=params["pooling"],
            num_classes=2,
        ).to(DEVICE)

    if name == "TRANSFORMER":
        from models.transformer_model import TransformerClassifier
        return TransformerClassifier(
            input_dim=DOFS,
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"],
            dropout_fc=params["dropout_fc"],
            n_classes=2,
        ).to(DEVICE)

    raise ValueError(
        f"Model '{model_name}' not supported for inference. "
        "Supported: CNN, RNN, Transformer"
    )


# ---------------------------------------------------------------------------
# Final model training on ALL training data
# ---------------------------------------------------------------------------

def train_final_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    patience: int = 15,
) -> nn.Module:
    """
    Train a single model on the full training set using the best hyperparameters
    from the checkpoint.  Early stopping is applied on training loss (same as
    the per-fold LOO-CV runs).
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=params.get("batch_size", 32),
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.get("learning_rate", 1e-3),
        weight_decay=params.get("weight_decay", 1e-4),
    )
    criterion = nn.CrossEntropyLoss()
    epochs = params.get("epochs", 100)

    best_loss, best_state, no_improve = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        if avg < best_loss - 1e-4:
            best_loss = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [Early stop] epoch {epoch + 1}  loss={avg:.4f}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Inference forward pass
# ---------------------------------------------------------------------------

def run_inference(model: nn.Module, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return (probs >= 0.5).astype(int), probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run XDash model inference on new test subjects."
    )
    parser.add_argument("--dataset", default="xdash",
                        help="Dataset name (must match datasets/ folder). Default: xdash")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model_BA*.pt checkpoint")
    parser.add_argument("--task", type=int, required=True,
                        help="Task number")
    parser.add_argument("--paradigm", type=int, required=True,
                        help="Paradigm number")
    parser.add_argument("--model", required=True,
                        choices=["cnn", "rnn", "transformer"],
                        help="Model type (must match the checkpoint)")
    parser.add_argument("--method", default="truncate",
                        choices=["truncate", "sliding_window", "padding", "phase_shift"],
                        help="Preprocessing method used during training")
    parser.add_argument("--resample-rate", type=int, default=None,
                        help="Sampling rate used during training (default: dataset sampling_rate)")

    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument("--test-data", metavar="PKL",
                            help="Pickle dict {subject_id: array (T, C)}")
    test_group.add_argument("--test-csv-dir", metavar="DIR",
                            help="Directory of per-subject CSV files")
    test_group.add_argument("--test-csv", metavar="FILE",
                            help="Single CSV file (one subject)")

    parser.add_argument("--output", default=None,
                        help="Output path (default: inference_results.json)")
    args = parser.parse_args()

    # Load dataset config and expose as module globals so helpers above can use them
    global _DATASET, _DATASET_CONFIG
    _DATASET = args.dataset
    _DATASET_CONFIG = load_dataset_config(args.dataset)
    if args.resample_rate is None:
        args.resample_rate = _DATASET_CONFIG.get("sampling_rate", 50)

    # ------------------------------------------------------------------
    # 1. Load checkpoint
    # ------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", args.model.upper())
    best_params = ckpt["hyperparameters"]
    input_shape = ckpt.get("input_shape")

    print(f"\n{'='*60}")
    print(f"Checkpoint  : {ckpt_path.name}")
    print(f"Model       : {model_name}")
    print(f"Train BA    : {ckpt['metrics'].get('balanced_accuracy', '?'):.4f}")
    print(f"Input shape : {input_shape}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 2. Load + preprocess training data (to fit the scaler)
    # ------------------------------------------------------------------
    print("\n[1/4] Loading training data to fit preprocessing pipeline...")
    patient_data, control_data = load_training_data(args.task)

    selector = ParadigmSelector(_DATASET_CONFIG)
    g1, g0 = selector.select_paradigm(patient_data, control_data, paradigm=args.paradigm)
    print(f"  Paradigm {args.paradigm}: g1={len(g1)}, g0={len(g0)}")

    model_type = args.model.lower()
    preprocessor = PreprocessorFactory.create(
        method=args.method,
        model_type=model_type,
        resample_rate=args.resample_rate,
        original_rate=_DATASET_CONFIG.get("sampling_rate", 50),
    )
    X_train, y_train, _ = preprocessor.prepare_data(g1, g0)

    output_format = "channels_first" if model_type == "cnn" else "3d"
    if input_shape is not None:
        T_seq = input_shape[2] if output_format == "channels_first" else input_shape[1]
    else:
        T_seq = X_train.shape[2] if output_format == "channels_first" else X_train.shape[1]

    # Unwrap ResamplingWrapper to reach the inner scaler
    inner = getattr(preprocessor, "inner", preprocessor)
    scaler = inner.scaler

    print(f"  T_seq={T_seq}  format={output_format}  "
          f"scaler fitted on {X_train.shape[0]} training samples")

    # ------------------------------------------------------------------
    # 3. Build and train final model on ALL training data
    # ------------------------------------------------------------------
    print(f"\n[2/4] Training final {model_name} on {X_train.shape[0]} samples...")
    model = build_model(model_name, best_params)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model = train_final_model(model, X_train, y_train, best_params)

    # ------------------------------------------------------------------
    # 4. Load + preprocess test data
    # ------------------------------------------------------------------
    print("\n[3/4] Loading and preprocessing test data...")
    if args.test_data:
        test_data = load_test_pkl(args.test_data)
    elif args.test_csv_dir:
        test_data = load_test_csv_dir(args.test_csv_dir)
    else:
        test_data = load_test_csv(args.test_csv)

    window_subjects = None
    if args.method == "sliding_window":
        window_size = best_params.get("window_size", 300)
        overlap = best_params.get("overlap", 0.3)
        stride = int(window_size * (1 - overlap))
        X_test, window_subjects, subject_ids = preprocess_test_sliding_window(
            test_data, scaler, window_size, stride, output_format
        )
        print(f"  {len(X_test)} windows from {len(subject_ids)} subjects")
    else:
        X_test, subject_ids = preprocess_test_truncate(
            test_data, scaler, T_seq, output_format
        )
        print(f"  {len(X_test)} subjects  shape={X_test.shape}")

    # ------------------------------------------------------------------
    # 5. Inference
    # ------------------------------------------------------------------
    print("\n[4/4] Running inference...")
    preds_raw, probs_raw = run_inference(model, X_test)

    if window_subjects is not None:
        preds, probs = aggregate_window_predictions(
            preds_raw, probs_raw, window_subjects, subject_ids
        )
    else:
        preds, probs = preds_raw, probs_raw

    # ------------------------------------------------------------------
    # 6. Display + save results
    # ------------------------------------------------------------------
    label_map = {1: "patient/condition", 0: "control"}
    print(f"\n{'='*65}")
    print(f"{'Subject':<30} {'Prediction':<25} {'Prob(patient)':>12}")
    print("-" * 65)
    for sid, pred, prob in zip(subject_ids, preds, probs):
        print(f"{sid:<30} {label_map[int(pred)]:<25} {prob:>12.4f}")
    print("=" * 65)

    results = {
        "checkpoint": str(ckpt_path),
        "task": args.task,
        "task_name": _DATASET_CONFIG["tasks"].get(args.task, str(args.task)),
        "paradigm": args.paradigm,
        "paradigm_name": _DATASET_CONFIG["paradigms"].get(args.paradigm, {}).get("name", str(args.paradigm)),
        "model": model_name,
        "method": args.method,
        "training_balanced_accuracy": ckpt["metrics"].get("balanced_accuracy"),
        "timestamp": datetime.now().isoformat(),
        "n_test_subjects": len(subject_ids),
        "predictions": [
            {
                "subject_id": sid,
                "prediction": int(pred),
                "label": label_map[int(pred)],
                "prob_patient": round(float(prob), 6),
                "prob_control": round(float(1 - prob), 6),
            }
            for sid, pred, prob in zip(subject_ids, preds, probs)
        ],
    }

    out_path = Path(args.output) if args.output else Path("inference_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
