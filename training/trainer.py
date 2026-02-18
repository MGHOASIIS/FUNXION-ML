"""
Training utilities for XDash models.

Provides generic training loops, callbacks, schedulers, and training orchestration.
"""
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from config.constants import DEVICE


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 30
    
    # Regularization
    dropout: float = 0.0
    gradient_clip: Optional[float] = None
    
    # Learning rate schedule
    lr_scheduler: Optional[str] = None  # 'step', 'cosine', 'plateau'
    lr_scheduler_params: Dict = field(default_factory=dict)
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing
    save_checkpoints: bool = False
    checkpoint_dir: Optional[Path] = None
    
    # Logging
    verbose: bool = True
    log_interval: int = 10
    
    # Device
    device: str = DEVICE


@dataclass
class TrainingHistory:
    """Records training history."""
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float('inf')
    
    def update(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        lr: Optional[float] = None,
        epoch_time: Optional[float] = None
    ):
        """Update history with new epoch results."""
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if val_acc is not None:
            self.val_acc.append(val_acc)
        if lr is not None:
            self.learning_rates.append(lr)
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)
        
        # Track best
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = len(self.train_loss) - 1


# ============================================================================
# Callbacks
# ============================================================================

class Callback(ABC):
    """Base callback class."""
    
    def on_train_begin(self, trainer: 'Trainer'):
        """Called at start of training."""
        pass
    
    def on_train_end(self, trainer: 'Trainer'):
        """Called at end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, trainer: 'Trainer'):
        """Called at start of epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, trainer: 'Trainer'):
        """Called at end of epoch."""
        pass
    
    def on_batch_begin(self, batch: int, trainer: 'Trainer'):
        """Called at start of batch."""
        pass
    
    def on_batch_end(self, batch: int, trainer: 'Trainer'):
        """Called at end of batch."""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping based on validation loss."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',
        restore_best: bool = True
    ):
        """
        Parameters
        ----------
        patience : int
            Number of epochs to wait
        min_delta : float
            Minimum change to qualify as improvement
        mode : str
            'min' for loss, 'max' for accuracy
        restore_best : bool
            Whether to restore best weights at end
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False
    
    def on_epoch_end(self, epoch: int, trainer: 'Trainer'):
        """Check if should stop."""
        score = trainer.history.val_loss[-1] if trainer.history.val_loss else None
        
        if score is None:
            return
        
        if self.best_score is None:
            self.best_score = score
            self.best_weights = trainer.model.state_dict()
        else:
            # Check for improvement
            if self.mode == 'min':
                improved = score < (self.best_score - self.min_delta)
            else:
                improved = score > (self.best_score + self.min_delta)
            
            if improved:
                self.best_score = score
                self.best_weights = trainer.model.state_dict()
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    if trainer.config.verbose:
                        print(f"\n[EarlyStopping] Stopped at epoch {epoch+1}")
    
    def on_train_end(self, trainer: 'Trainer'):
        """Restore best weights."""
        if self.restore_best and self.best_weights is not None:
            trainer.model.load_state_dict(self.best_weights)
            if trainer.config.verbose:
                print(f"[EarlyStopping] Restored best weights from epoch {trainer.history.best_epoch+1}")


class LRSchedulerCallback(Callback):
    """Learning rate scheduling."""
    
    def __init__(self, scheduler):
        """
        Parameters
        ----------
        scheduler : torch.optim.lr_scheduler
            Learning rate scheduler
        """
        self.scheduler = scheduler
    
    def on_epoch_end(self, epoch: int, trainer: 'Trainer'):
        """Step the scheduler."""
        # For ReduceLROnPlateau, need validation loss
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if trainer.history.val_loss:
                self.scheduler.step(trainer.history.val_loss[-1])
        else:
            self.scheduler.step()


class CheckpointCallback(Callback):
    """Save model checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        save_best_only: bool = True,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        """
        Parameters
        ----------
        checkpoint_dir : Path
            Directory to save checkpoints
        save_best_only : bool
            Only save when metric improves
        monitor : str
            Metric to monitor
        mode : str
            'min' or 'max'
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        # self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        self.best_score = None
    
    def on_epoch_end(self, epoch: int, trainer: 'Trainer'):
        """Save checkpoint if improved."""
        # Get monitored metric
        if self.monitor == 'val_loss' and trainer.history.val_loss:
            score = trainer.history.val_loss[-1]
        elif self.monitor == 'train_loss':
            score = trainer.history.train_loss[-1]
        else:
            return
        
        # Check if should save
        should_save = False
        if not self.save_best_only:
            should_save = True
        else:
            if self.best_score is None:
                should_save = True
            else:
                if self.mode == 'min':
                    should_save = score < self.best_score
                else:
                    should_save = score > self.best_score
        
        if should_save:
            self.best_score = score
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'history': trainer.history,
                'config': trainer.config
            }, path)
            
            if trainer.config.verbose:
                print(f"[Checkpoint] Saved to {path}")


class ProgressCallback(Callback):
    """Print training progress."""
    
    def on_epoch_end(self, epoch: int, trainer: 'Trainer'):
        """Print epoch summary."""
        if not trainer.config.verbose:
            return
        
        train_loss = trainer.history.train_loss[-1]
        train_acc = trainer.history.train_acc[-1]
        
        msg = f"Epoch {epoch+1}/{trainer.config.epochs}: "
        msg += f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
        
        if trainer.history.val_loss:
            val_loss = trainer.history.val_loss[-1]
            val_acc = trainer.history.val_acc[-1]
            msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        
        if trainer.history.learning_rates:
            lr = trainer.history.learning_rates[-1]
            msg += f", lr={lr:.6f}"
        
        if trainer.history.epoch_times:
            epoch_time = trainer.history.epoch_times[-1]
            msg += f", time={epoch_time:.2f}s"
        
        print(msg)


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """Generic trainer for PyTorch models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        callbacks: Optional[List[Callback]] = None
    ):
        """
        Parameters
        ----------
        model : nn.Module
            Model to train
        config : TrainingConfig
            Training configuration
        callbacks : List[Callback], optional
            List of callbacks
        """
        self.model = model
        self.config = config
        self.callbacks = callbacks or []
        
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        self.optimizer = None
        self.criterion = None
        self.history = TrainingHistory()
    
    def compile(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None
    ):
        """
        Setup optimizer and loss function.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer, optional
            Optimizer (default: Adam)
        criterion : nn.Module, optional
            Loss function (default: CrossEntropyLoss)
        """
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        # Setup LR scheduler if specified
        if self.config.lr_scheduler:
            scheduler = self._create_scheduler()
            self.callbacks.append(LRSchedulerCallback(scheduler))
        
        # Add early stopping if enabled
        if self.config.early_stopping:
            self.callbacks.append(EarlyStoppingCallback(
                patience=self.config.patience,
                min_delta=self.config.min_delta
            ))
        
        # Add checkpoint saving if enabled
        if self.config.save_checkpoints and self.config.checkpoint_dir:
            self.callbacks.append(CheckpointCallback(
                checkpoint_dir=self.config.checkpoint_dir
            ))
        
        # Add progress printing if verbose
        if self.config.verbose:
            self.callbacks.append(ProgressCallback())
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                **self.config.lr_scheduler_params
            )
        elif self.config.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                **self.config.lr_scheduler_params
            )
        elif self.config.lr_scheduler == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                **self.config.lr_scheduler_params
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.lr_scheduler}")
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> TrainingHistory:
        """
        Train the model.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        
        Returns
        -------
        TrainingHistory
            Training history
        """
        # Callbacks: train begin
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Callbacks: epoch begin
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, self)
            
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
            
            # Record
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]['lr']
            
            self.history.update(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=lr,
                epoch_time=epoch_time
            )
            
            # Callbacks: epoch end
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self)
            
            # Check early stopping
            early_stop = any(
                isinstance(cb, EarlyStoppingCallback) and cb.early_stop
                for cb in self.callbacks
            )
            if early_stop:
                break
        
        # Callbacks: train end
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        return self.history
    
    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(loader):
            # Callbacks: batch begin
            for callback in self.callbacks:
                callback.on_batch_begin(batch_idx, self)
            
            # Move to device
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            
            # Forward
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * y_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
            # Callbacks: batch end
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, self)
        
        avg_loss = total_loss / total
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def _validate(self, loader: DataLoader) -> Tuple[float, float]:
        """Validate."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                
                total_loss += loss.item() * y_batch.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        
        avg_loss = total_loss / total
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def predict(
        self,
        X: torch.Tensor,
        return_proba: bool = True
    ) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data
        return_proba : bool
            Whether to return probabilities
        
        Returns
        -------
        np.ndarray
            Predictions or probabilities
        """
        self.model.eval()
        
        with torch.no_grad():
            X = X.to(self.device)
            logits = self.model(X)
            
            if return_proba:
                probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy()
            else:
                preds = logits.argmax(dim=1)
                return preds.cpu().numpy()


# ============================================================================
# Helper Functions
# ============================================================================

def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch data loaders.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray, optional
        Validation features
    y_val : np.ndarray, optional
        Validation labels
    batch_size : int
        Batch size
    shuffle_train : bool
        Whether to shuffle training data
    num_workers : int
        Number of workers for data loading
    
    Returns
    -------
    Tuple[DataLoader, Optional[DataLoader]]
        (train_loader, val_loader)
    """
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if X_val is not None and y_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        
        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        
        def forward(self, x):
            return self.fc(x)
    
    # Create dummy data
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randn(20, 10)
    y_val = np.random.randint(0, 2, 20)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=16
    )
    
    # Setup trainer
    model = DummyModel()
    config = TrainingConfig(
        epochs=10,
        learning_rate=1e-3,
        early_stopping=True,
        patience=3
    )
    
    trainer = Trainer(model, config)
    trainer.compile()
    
    # Train
    history = trainer.fit(train_loader, val_loader)
    
    print(f"\nBest epoch: {history.best_epoch + 1}")
    print(f"Best val loss: {history.best_val_loss:.4f}")