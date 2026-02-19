# XDash: ML Classification of Shoulder Pathology Using XR-Based Motion Capture

> **Machine Learning Classification of Shoulder Pathology Using Extended Reality (XR) Motion Tracking and Deep Learning**

## 🎯 Project Overview

This research project develops state-of-the-art AI models to classify shoulder pathology using Extended Reality (XR) motion-tracking data. We capture 6-degree-of-freedom (6-DoF) kinematic data from XR headsets and hand controllers during standardized functional assessment tasks, then apply deep learning to automate injury classification and identify discriminative movement patterns.

**Clinical Impact**: Bridging the gap between subjective manual assessments and objective, interpretable AI tools for orthopedic rehabilitation.

## 📊 Research Phases

### ✅ Phase 1: Baseline ML on Whole Time-Series (COMPLETED)
- **Models**: HMM, 1D-CNN, RNN (GRU/LSTM, bidirectional)
- **Evaluation**: Leave-One-Out Cross-Validation (LOO CV) with balanced accuracy, recall, AUC
- **Data**: N=60, 18 features per timepoint, 6 functional tasks, 4 classification paradigms
- **Results**: RNN achieved most consistent performance (BA: 0.60–0.84, recall: 0.67–0.95)

### 🔄 Phase 2: Enhanced ML with Validation (IN PROGRESS)  
- **Data Augmentation**: TimeGAN, jittering, magnitude warping
- **Validation Study**: Map DL sensor importance → XGBoost+SHAP traditional features
- **Downsampling**: 50Hz → 20-25Hz to reduce computational burden
- **Infrastructure**: Comprehensive diagnostic monitoring and overfitting detection

### 🚀 Phase 3: Self-Supervised Transformers (PLANNED)
- **Approach**: Event-based windowing (~3,600 samples) + self-supervised pretraining
- **Models**: Contrastive learning (SimCLR/MoCo) → Transformer fine-tuning
- **Goal**: Overcome N=60 limitation through representation learning

## 🏗️ Project Structure

```
X-DASH-Data-Analysis/
├── 📁 config/                    # Configuration files
│   ├── constants.py              # Project constants (tasks, paradigms, features)
│   ├── hyperparameter.py         # Model hyperparameter grids
│   └── paths.py                  # File paths and directory structure
├── 📁 models/                    # Model implementations
│   ├── base_model.py             # Abstract base class with reproducibility
│   ├── hmm_model.py              # Hidden Markov Model
│   ├── cnn_model.py              # 1D Convolutional Neural Network
│   └── rnn_model.py              # Recurrent Neural Network (GRU/LSTM)
├── 📁 training/                  # Training infrastructure
│   ├── trainer.py                # Main training orchestrator
│   ├── cross_validator.py        # LOO cross-validation
│   └── evaluator.py              # Model evaluation and metrics
├── 📁 utils/                     # Utilities and diagnostics
│   ├── comprehensive_monitor.py  # Complete model diagnostics
│   ├── model_diagnostics.py      # Gradient/activation analysis
│   ├── overfitting_detection.py  # Overfitting analysis for N=60
│   ├── importance.py             # Feature importance analysis
│   ├── visualization.py          # Plotting and visualization
│   ├── metrics.py                # Evaluation metrics
│   └── checkpointing.py          # Model checkpointing
├── 📁 preprocessing/              # Data preprocessing
│   ├── loaders.py                # Data loading utilities
│   ├── preprocessors.py          # Preprocessing pipelines
│   ├── transforms.py             # Data transformations
│   └── paradigms.py              # Classification paradigm setup
├── 📁 experiments/               # Experiment outputs
├── 📁 logs/                      # Training and diagnostic logs
├── main.py                       # Main experiment runner
└── environment.yml               # Conda environment
```

## 📈 Dataset & Tasks

### XDash Dataset (N=60)
- **Population**: 40 patients (RCT, arthritis, bursitis, tendonitis) + 20 controls
- **Sensors**: XR headset + 2 controllers (18 features: 3×6-DoF @ 50Hz)
- **Tasks**: 6 standardized functional assessments
  1. 🫙 **Jar Opening** 
  2. 🔑 **Key Turning** 
  3. 🧽 **Cleaning** 
  4. 🚿 **Back Washing**
  5. ✂️ **Cutting**
  6. 🔨 **Hammering**

### Classification Paradigms
1. **Patients vs Controls** - Primary diagnostic classification
2. **RCT vs Controls** - Rotator cuff tear specific
3. **Other Conditions vs Controls** - Non-RCT pathologies  
4. **RCT vs Other Conditions** - Differential diagnosis

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- conda/mamba package manager

### Installation

```bash
# Clone repository
git clone https://github.com/username/X-DASH-Data-Analysis.git
cd X-DASH-Data-Analysis

# Create conda environment
conda env create -f environment.yml
conda activate xdash

# Verify installation
python main.py --help
```

### Quick Start

```bash
# Run single experiment
python main.py --task 1 --paradigm 1 --model rnn --method truncate

# Run with comprehensive diagnostics
python main.py --task 1 --paradigm 1 --model rnn --diagnostics

# Hyperparameter search with early stopping
python main.py --task 2 --paradigm 2 --model cnn --patience 15 --min-delta 1e-4

# Save model checkpoints
python main.py --task 1 --paradigm 1 --model rnn --save-checkpoints
```

### HPC Usage (SLURM)

```bash
# Setup environment on HPC
bash setup_env.sh

# Submit single job
sbatch run_single.sh 1 1 rnn

# Submit all experiments (72 total)
bash submit_all.sh

# Monitor jobs
squeue -u username
```

## 🔬 Key Features

### 🎯 Model Architectures
- **HMM**: Generative probabilistic model with Gaussian emissions
- **1D-CNN**: Deep convolutional architecture with GRU head
- **RNN**: GRU/LSTM with bidirectional support and multiple pooling strategies

### 📊 Comprehensive Diagnostics
- **Overfitting Detection**: Critical for N=60 - generalization gap analysis, bias-variance decomposition
- **Gradient Analysis**: Vanishing/exploding gradient detection
- **Activation Analysis**: Dead neuron detection, activation diversity
- **Feature Importance**: Permutation importance + weight-based analysis
- **Clinical Interpretation**: Maps sensor importance to biomechanical understanding

### 🔧 Research Infrastructure
- **Reproducible**: All random seeds controlled for exact reproducibility
- **Modular**: Clean separation of preprocessing, models, training, evaluation
- **Extensible**: Easy to add new models, tasks, or datasets
- **Validated**: Comprehensive testing and error handling

## 🔧 Configuration

### Model Hyperparameters
```python
# HMM Configuration
HMM_PARAM_GRID = {
        "covariance_type": ["diag", "full"],
        "n_components":    [2, 3, 4, 5],
        "n_iter":          [50, 100],
    }

# CNN confiugration
  CNN_PARAM_GRID = {
      # CNN trunk
      "conv_channels": [
          [32, 64, 64],
          [64, 128, 128],
          [64, 128, 256],
      ],
      "kernel_sizes": [
          [7, 5, 3],
          [5, 5, 5],
      ],
      # Regularisation & optimiser
      "dropout_fc":    [0.2, 0.4],
      "learning_rate": [1e-3, 2e-4],
      "weight_decay":  [5e-4],
      # Training loop
      "batch_size":    [32, 64],
      "epochs":        [100],
  }

# RNN configuration
  RNN_PARAM_GRID = {
      # architecture
      "rnn_type":      ["lstm", "gru"],
      "hidden_size":   [32, 64, 128],
      "num_layers":    [1, 2],
      # regularisation
      "bidirectional": [True, False],
      "dropout_rnn":   [0.2],
      "dropout_fc":    [0.3],
      # optimisation
      "lr":            [1e-3],
      "epochs":        [100],
      "batch_size":    [32, 64],
      # pooling
      "pooling":       ["max", "mean", "last"],
  }
```

### Reproducibility
```python
# All models inherit automatic reproducibility
model = RNNModel(seed=42, task=1, paradigm=1)
# Sets torch, numpy, random, CUDA seeds automatically
```

## 📖 Usage Examples

### Basic Training
```python
from models.rnn_model import RNNModel
from preprocessing.loaders import load_xdash_data
from preprocessing.paradigms import ParadigmSelector

# Load and preprocess data
g1, g0 = load_xdash_data(task=1)
paradigm_selector = ParadigmSelector()
X, y, subject_ids = paradigm_selector.prepare_data(g1, g0, paradigm=1)

# Train model with reproducibility
model = RNNModel(seed=42, patience=15)
results = model.train_and_evaluate(X, y, subject_ids)

print(f"Balanced Accuracy: {results.metrics['ba']:.3f}")
print(f"Best Architecture: {results.best_params}")
```

### Comprehensive Diagnostics
```python
from utils.comprehensive_monitor import run_complete_monitoring

# Run full diagnostic suite
diagnostic_results = run_complete_monitoring(
    model=model,
    X=X, y=y, subject_ids=subject_ids,
    fold_results=results.per_fold_results,
    experiment_name="RNN_Task1_Paradigm1",
    save_dir=Path("diagnostics"),
    feature_names=CHAN_NAME
)

# Includes: overfitting analysis, gradient/activation analysis, 
# feature importance, clinical interpretation, bias-variance decomposition
```

## 📝 Publications & Citations

### Completed
- **"ML Classification of Shoulder Pathology from XR-Based Functional Assessments"** - Phase 1 results

### In Preparation
- **Phase 2**: Enhanced ML with validation study
- **Phase 3**: Self-supervised Transformers for clinical time-series

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black . && isort .

# Type checking
mypy models/ utils/
```

## 📧 Contact & Collaboration

**Research Team**: Orthopedic Lab, Massachusetts General Brigham
**Primary Contact**: [Your Name] - [email@institution.edu]
**Clinical Collaborator**: [Clinical Lead]
**Technical Lead**: [Your Name]

---

## 📄 License



## 🙏 Acknowledgments

- Massachusetts General Brigham Orthopedic Surgery Department
- Clinical research participants who made this work possible
- HPC resources provided by [Institution]
- Funding support from [Grant Numbers]

---

*This project aims to transform orthopedic assessment through interpretable AI and immersive technology, bridging the gap between research innovation and clinical impact.*
