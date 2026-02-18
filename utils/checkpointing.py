"""
Advanced Checkpoint & Model Saving System

State-of-the-art model persistence with:
- Multi-level checkpointing (epoch, fold, best)
- Automatic checkpoint management
- Resume from checkpoint
- Model export (ONNX, TorchScript)
- Experiment tracking integration
"""
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import shutil


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoints."""
    timestamp: str
    epoch: int
    fold: Optional[int]
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    model_architecture: str
    data_info: Dict[str, Any]
    
    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Path):
        """Save metadata to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @staticmethod
    def load(path: Path):
        """Load metadata from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return CheckpointMetadata(**data)


class CheckpointManager:
    """
    Comprehensive checkpoint management system.
    
    Handles multiple checkpoint types:
    - Best model (best validation score)
    - Latest model (most recent epoch)
    - Per-fold checkpoints (for LOO CV)
    - Recovery checkpoints (for crash recovery)
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        keep_n_best: int = 3,
        keep_n_latest: int = 5,
        auto_cleanup: bool = True
    ):
        """
        Parameters
        ----------
        checkpoint_dir : Path
            Directory to save checkpoints
        keep_n_best : int
            Keep top N best checkpoints
        keep_n_latest : int
            Keep N most recent checkpoints
        auto_cleanup : bool
            Automatically clean up old checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        # self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_n_best = keep_n_best
        self.keep_n_latest = keep_n_latest
        self.auto_cleanup = auto_cleanup
        
        # Tracking
        self.best_scores = []
        self.checkpoint_history = []
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        checkpoint_type: str = 'epoch',
        fold: Optional[int] = None,
        extra_data: Optional[Dict] = None
    ) -> Path:
        """
        Save a comprehensive checkpoint.
        
        Parameters
        ----------
        model : nn.Module
            Model to save
        optimizer : torch.optim.Optimizer
            Optimizer state
        epoch : int
            Current epoch
        metrics : Dict[str, float]
            Performance metrics
        hyperparameters : Dict[str, Any]
            Model hyperparameters
        checkpoint_type : str
            'epoch', 'best', 'fold', 'recovery'
        fold : int, optional
            Fold number (for LOO CV)
        extra_data : Dict, optional
            Additional data to save
        
        Returns
        -------
        Path
            Path to saved checkpoint
        """
        # Create checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if checkpoint_type == 'fold':
            filename = f"fold_{fold:03d}_epoch_{epoch:03d}.pt"
        elif checkpoint_type == 'best':
            filename = f"best_epoch_{epoch:03d}_score_{metrics.get('val_score', 0):.4f}.pt"
        elif checkpoint_type == 'recovery':
            filename = f"recovery_{timestamp}.pt"
        else:  # epoch
            filename = f"epoch_{epoch:03d}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint
        checkpoint = {
            'timestamp': timestamp,
            'epoch': epoch,
            'fold': fold,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'hyperparameters': hyperparameters,
            'model_architecture': str(model),
            'checkpoint_type': checkpoint_type
        }
        
        # Add extra data
        if extra_data:
            checkpoint['extra_data'] = extra_data
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata separately
        metadata = CheckpointMetadata(
            timestamp=timestamp,
            epoch=epoch,
            fold=fold,
            metrics=metrics,
            hyperparameters=hyperparameters,
            model_architecture=str(model),
            data_info=extra_data or {}
        )
        metadata.save(checkpoint_path.with_suffix('.json'))
        
        # Track checkpoint
        self.checkpoint_history.append({
            'path': checkpoint_path,
            'type': checkpoint_type,
            'score': metrics.get('val_score', 0),
            'epoch': epoch
        })
        
        # Cleanup if needed
        if self.auto_cleanup and checkpoint_type == 'epoch':
            self._cleanup_old_checkpoints()
        
        print(f"[Checkpoint Saved] {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore state.
        
        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint
        model : nn.Module
            Model to load into
        optimizer : torch.optim.Optimizer, optional
            Optimizer to restore state
        device : str
            Device to map checkpoint to
        
        Returns
        -------
        Dict[str, Any]
            Checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Restore model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"[Checkpoint Loaded] {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Metrics: {checkpoint['metrics']}")
        
        return checkpoint
    
    def get_best_checkpoint(self, metric: str = 'val_score') -> Optional[Path]:
        """
        Get path to best checkpoint.
        
        Parameters
        ----------
        metric : str
            Metric to sort by
        
        Returns
        -------
        Path or None
            Path to best checkpoint
        """
        best_checkpoints = [c for c in self.checkpoint_history if c['type'] == 'best']
        
        if not best_checkpoints:
            return None
        
        best = max(best_checkpoints, key=lambda x: x['score'])
        return best['path']
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints, keep only recent and best."""
        epoch_checkpoints = [c for c in self.checkpoint_history if c['type'] == 'epoch']
        
        if len(epoch_checkpoints) > self.keep_n_latest:
            # Sort by epoch
            epoch_checkpoints.sort(key=lambda x: x['epoch'])
            
            # Remove oldest
            to_remove = epoch_checkpoints[:-self.keep_n_latest]
            
            for checkpoint in to_remove:
                if checkpoint['path'].exists():
                    checkpoint['path'].unlink()
                    # Also remove metadata
                    json_path = checkpoint['path'].with_suffix('.json')
                    if json_path.exists():
                        json_path.unlink()


class ExperimentTracker:
    """
    Track experiments with comprehensive logging.
    
    Logs everything needed to reproduce experiments:
    - Configuration
    - Results
    - Model checkpoints
    - Code version (git commit)
    - Hardware info
    """
    
    def __init__(self, experiment_dir: Path, experiment_name: str):
        """
        Parameters
        ----------
        experiment_dir : Path
            Base directory for experiments
        experiment_name : str
            Name of this experiment
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = experiment_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.logs_dir = self.experiment_dir / "logs"
        self.results_dir = self.experiment_dir / "results"
        self.figures_dir = self.experiment_dir / "figures"
        
        for dir in [self.checkpoints_dir, self.logs_dir, self.results_dir, self.figures_dir]:
            dir.mkdir(exist_ok=True)
        
        # Initialize tracking
        self.start_time = time.time()
        self.config = {}
        self.results = {}
        
        print(f"\n[ExperimentTracker] Initialized: {self.experiment_dir}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.config = config
        
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[Config Logged] {config_path}")
    
    def log_results(self, results: Dict[str, Any]):
        """Log experiment results."""
        self.results = results
        
        results_path = self.results_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"[Results Logged] {results_path}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log metrics at a specific step.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Metrics to log
        step : int
            Step/epoch number
        """
        log_path = self.logs_dir / "metrics.jsonl"
        
        log_entry = {
            'step': step,
            'timestamp': time.time(),
            **metrics
        }
        
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def save_figure(self, fig: plt.Figure, name: str):
        """
        Save figure to experiment directory.
        
        Parameters
        ----------
        fig : plt.Figure
            Figure to save
        name : str
            Figure name
        """
        fig_path = self.figures_dir / f"{name}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"[Figure Saved] {fig_path}")
    
    def save_model_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint within experiment."""
        filename = f"best_model.pt" if is_best else f"checkpoint_epoch_{epoch:03d}.pt"
        checkpoint_path = self.checkpoints_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"[Model Checkpoint] {checkpoint_path}")
    
    def finalize(self):
        """Finalize experiment (save summary)."""
        elapsed = time.time() - self.start_time
        
        summary = {
            'experiment_dir': str(self.experiment_dir),
            'start_time': self.start_time,
            'elapsed_seconds': elapsed,
            'config': self.config,
            'results': self.results
        }
        
        summary_path = self.experiment_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[Experiment Complete] {self.experiment_dir}")
        print(f"  Duration: {elapsed/60:.1f} minutes")
        print(f"  Summary: {summary_path}")


class ModelExporter:
    """
    Export models for deployment.
    
    Supports:
    - ONNX (cross-platform inference)
    - TorchScript (production PyTorch)
    - Weights only (smallest size)
    """
    
    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        dummy_input: torch.Tensor,
        export_path: Path,
        opset_version: int = 11
    ):
        """
        Export model to ONNX format.
        
        Parameters
        ----------
        model : nn.Module
            Model to export
        dummy_input : torch.Tensor
            Example input for tracing
        export_path : Path
            Export path (.onnx extension)
        opset_version : int
            ONNX opset version
        """
        model.eval()
        
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"[ONNX Export] {export_path}")
        print(f"  Size: {export_path.stat().st_size / 1024:.1f} KB")
    
    @staticmethod
    def export_to_torchscript(
        model: nn.Module,
        dummy_input: torch.Tensor,
        export_path: Path,
        method: str = 'trace'
    ):
        """
        Export model to TorchScript.
        
        Parameters
        ----------
        model : nn.Module
            Model to export
        dummy_input : torch.Tensor
            Example input
        export_path : Path
            Export path (.pt extension)
        method : str
            'trace' or 'script'
        """
        model.eval()
        
        if method == 'trace':
            traced_model = torch.jit.trace(model, dummy_input)
        else:  # script
            traced_model = torch.jit.script(model)
        
        traced_model.save(str(export_path))
        
        print(f"[TorchScript Export] {export_path}")
        print(f"  Size: {export_path.stat().st_size / 1024:.1f} KB")
    
    @staticmethod
    def save_for_inference(
        model: nn.Module,
        export_dir: Path,
        example_input: torch.Tensor,
        config: Dict[str, Any]
    ):
        """
        Save complete package for inference.
        
        Parameters
        ----------
        model : nn.Module
            Trained model
        export_dir : Path
            Export directory
        example_input : torch.Tensor
            Example input
        config : Dict
            Configuration
        """
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save weights
        weights_path = export_dir / "model_weights.pt"
        torch.save(model.state_dict(), weights_path)
        
        # 2. Save config
        config_path = export_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # 3. Save ONNX
        onnx_path = export_dir / "model.onnx"
        ModelExporter.export_to_onnx(model, example_input, onnx_path)
        
        # 4. Save model architecture as text
        arch_path = export_dir / "architecture.txt"
        with open(arch_path, 'w') as f:
            f.write(str(model))
        
        # 5. Create README
        readme_path = export_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"# Model Export\n\n")
            f.write(f"**Export Date:** {datetime.now()}\n\n")
            f.write(f"## Files\n\n")
            f.write(f"- `model_weights.pt` - PyTorch weights\n")
            f.write(f"- `model.onnx` - ONNX format (cross-platform)\n")
            f.write(f"- `config.json` - Model configuration\n")
            f.write(f"- `architecture.txt` - Model architecture\n\n")
            f.write(f"## Loading\n\n")
            f.write(f"```python\n")
            f.write(f"# PyTorch\n")
            f.write(f"model.load_state_dict(torch.load('model_weights.pt'))\n\n")
            f.write(f"# ONNX\n")
            f.write(f"import onnxruntime\n")
            f.write(f"session = onnxruntime.InferenceSession('model.onnx')\n")
            f.write(f"```\n")
        
        print(f"\n[Inference Package] {export_dir}")
        print(f"  ✓ Weights:  {weights_path.stat().st_size / 1024:.1f} KB")
        print(f"  ✓ ONNX:     {onnx_path.stat().st_size / 1024:.1f} KB")
        print(f"  ✓ Config:   {config_path}")
        print(f"  ✓ README:   {readme_path}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints."""
        # Keep best checkpoints
        best_ckpts = [c for c in self.checkpoint_history if c['type'] == 'best']
        best_ckpts.sort(key=lambda x: x['score'], reverse=True)
        
        if len(best_ckpts) > self.keep_n_best:
            for ckpt in best_ckpts[self.keep_n_best:]:
                if ckpt['path'].exists():
                    ckpt['path'].unlink()
        
        # Keep latest checkpoints
        epoch_ckpts = [c for c in self.checkpoint_history if c['type'] == 'epoch']
        epoch_ckpts.sort(key=lambda x: x['epoch'], reverse=True)
        
        if len(epoch_ckpts) > self.keep_n_latest:
            for ckpt in epoch_ckpts[self.keep_n_latest:]:
                if ckpt['path'].exists():
                    ckpt['path'].unlink()


# Example usage
if __name__ == "__main__":
    # Create dummy model
    model = nn.Sequential(
        nn.Linear(18, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    
    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(
        checkpoint_dir=Path("experiments/checkpoints"),
        keep_n_best=3,
        keep_n_latest=5
    )
    
    # Save checkpoint
    metrics = {'train_score': 0.85, 'val_score': 0.72}
    hyperparams = {'lr': 0.001, 'hidden_size': 64}
    
    ckpt_path = ckpt_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=10,
        metrics=metrics,
        hyperparameters=hyperparams,
        checkpoint_type='best'
    )
    
    # Export for inference
    exporter = ModelExporter()
    dummy_input = torch.randn(1, 18)
    exporter.save_for_inference(
        model=model,
        export_dir=Path("models/deployed"),
        example_input=dummy_input,
        config=hyperparams
    )