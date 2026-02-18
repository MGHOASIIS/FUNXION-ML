"""
Comprehensive visualization utilities for XDash project.

Provides publication-ready plots for:
- Training history
- Model comparisons
- Time-series data
- Statistical distributions
- Feature analysis
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class TrainingHistoryVisualizer:
    """Visualize training history."""
    
    @staticmethod
    def plot_loss_curves(
        train_loss: List[float],
        val_loss: Optional[List[float]] = None,
        title: str = "Training & Validation Loss",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot training and validation loss curves.
        
        Parameters
        ----------
        train_loss : List[float]
            Training loss per epoch
        val_loss : List[float], optional
            Validation loss per epoch
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        epochs = np.arange(1, len(train_loss) + 1)
        
        ax.plot(epochs, train_loss, 'o-', label='Train Loss', linewidth=2, markersize=4)
        
        if val_loss is not None:
            ax.plot(epochs, val_loss, 's-', label='Val Loss', linewidth=2, markersize=4)
            
            # Mark best epoch
            best_epoch = np.argmin(val_loss)
            ax.axvline(best_epoch + 1, color='red', linestyle='--', 
                      alpha=0.5, label=f'Best Epoch ({best_epoch + 1})')
            ax.plot(best_epoch + 1, val_loss[best_epoch], 'r*', 
                   markersize=15, label=f'Best Val Loss ({val_loss[best_epoch]:.4f})')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_metric_curves(
        train_metrics: Dict[str, List[float]],
        val_metrics: Optional[Dict[str, List[float]]] = None,
        title: str = "Training Metrics",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot multiple metric curves in subplots.
        
        Parameters
        ----------
        train_metrics : Dict[str, List[float]]
            {metric_name: values_per_epoch}
        val_metrics : Dict[str, List[float]], optional
            {metric_name: values_per_epoch}
        title : str
            Main title
        save_path : str, optional
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        n_metrics = len(train_metrics)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, (metric_name, train_values) in enumerate(train_metrics.items()):
            ax = axes[idx]
            epochs = np.arange(1, len(train_values) + 1)
            
            ax.plot(epochs, train_values, 'o-', label=f'Train {metric_name}', 
                   linewidth=2, markersize=4)
            
            if val_metrics and metric_name in val_metrics:
                val_values = val_metrics[metric_name]
                ax.plot(epochs, val_values, 's-', label=f'Val {metric_name}',
                       linewidth=2, markersize=4)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(title, fontsize=14, y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_learning_rate_schedule(
        learning_rates: List[float],
        title: str = "Learning Rate Schedule",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 5)
    ):
        """Plot learning rate schedule over epochs."""
        fig, ax = plt.subplots(figsize=figsize)
        
        epochs = np.arange(1, len(learning_rates) + 1)
        ax.plot(epochs, learning_rates, 'o-', linewidth=2, markersize=4, color='orange')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(title)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()


class ModelComparisonVisualizer:
    """Visualize model comparisons."""
    
    @staticmethod
    def plot_metric_comparison(
        results_dict: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None,
        title: str = "Model Comparison",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Compare metrics across models.
        
        Parameters
        ----------
        results_dict : Dict[str, Dict[str, float]]
            {model_name: {metric: value}}
        metrics : List[str], optional
            Metrics to plot (if None, plot all)
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(results_dict).T
        
        if metrics is None:
            metrics = df.columns.tolist()
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(metrics))
        width = 0.8 / len(results_dict)
        
        for i, (model_name, model_results) in enumerate(results_dict.items()):
            values = [model_results.get(m, 0) for m in metrics]
            offset = (i - len(results_dict)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend(loc='best')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_performance_table(
        results_dict: Dict[str, Dict[str, float]],
        title: str = "Model Performance Summary",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Display performance as a table.
        
        Parameters
        ----------
        results_dict : Dict[str, Dict[str, float]]
            {model_name: {metric: value}}
        title : str
            Table title
        save_path : str, optional
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        df = pd.DataFrame(results_dict).T
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=df.round(3).values,
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style row labels
        for i in range(1, len(df) + 1):
            table[(i, -1)].set_facecolor('#40466e')
            table[(i, -1)].set_text_props(weight='bold', color='white')
        
        plt.title(title, fontsize=14, pad=20)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_radar_comparison(
        results_dict: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None,
        title: str = "Model Comparison - Radar Chart",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 10)
    ):
        """
        Radar chart for model comparison.
        
        Parameters
        ----------
        results_dict : Dict[str, Dict[str, float]]
            {model_name: {metric: value}}
        metrics : List[str], optional
            Metrics to include
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        if metrics is None:
            metrics = list(list(results_dict.values())[0].keys())
        
        # Number of variables
        num_vars = len(metrics)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        for model_name, model_results in results_dict.items():
            values = [model_results.get(m, 0) for m in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(title, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()


class TimeSeriesVisualizer:
    """Visualize time-series data."""
    
    @staticmethod
    def plot_signal(
        signal: np.ndarray,
        sampling_rate: int = 50,
        channel_names: Optional[List[str]] = None,
        title: str = "Signal Visualization",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Plot multi-channel time-series signal.
        
        Parameters
        ----------
        signal : np.ndarray
            Signal data (T, C) where T=time, C=channels
        sampling_rate : int
            Sampling rate in Hz
        channel_names : List[str], optional
            Channel names
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        T, C = signal.shape
        time = np.arange(T) / sampling_rate
        
        if channel_names is None:
            channel_names = [f"Channel {i+1}" for i in range(C)]
        
        # Plot each channel in a subplot
        fig, axes = plt.subplots(C, 1, figsize=figsize, sharex=True)
        
        if C == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            ax.plot(time, signal[:, i], linewidth=1)
            ax.set_ylabel(channel_names[i])
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        axes[0].set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_signal_comparison(
        signals: Dict[str, np.ndarray],
        channel_idx: int = 0,
        sampling_rate: int = 50,
        title: str = "Signal Comparison",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Compare multiple signals for a specific channel.
        
        Parameters
        ----------
        signals : Dict[str, np.ndarray]
            {label: signal_data}
        channel_idx : int
            Channel index to compare
        sampling_rate : int
            Sampling rate in Hz
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for label, signal in signals.items():
            T = signal.shape[0]
            time = np.arange(T) / sampling_rate
            ax.plot(time, signal[:, channel_idx], label=label, alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_spectrogram(
        signal: np.ndarray,
        sampling_rate: int = 50,
        channel_idx: int = 0,
        title: str = "Spectrogram",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot spectrogram of a signal.
        
        Parameters
        ----------
        signal : np.ndarray
            Signal data (T, C)
        sampling_rate : int
            Sampling rate in Hz
        channel_idx : int
            Channel to visualize
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.specgram(
            signal[:, channel_idx],
            Fs=sampling_rate,
            cmap='viridis',
            vmin=-40,
            vmax=40
        )
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title)
        
        plt.colorbar(ax.images[0], ax=ax, label='Power (dB)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()


class StatisticalVisualizer:
    """Visualize statistical distributions."""
    
    @staticmethod
    def plot_distributions(
        data_dict: Dict[str, np.ndarray],
        title: str = "Distribution Comparison",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        Plot distributions with histograms and KDE.
        
        Parameters
        ----------
        data_dict : Dict[str, np.ndarray]
            {label: data_array}
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for label, data in data_dict.items():
            ax.hist(data, bins=30, alpha=0.5, label=f'{label} (histogram)', density=True)
            
            # Add KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde(x_range), label=f'{label} (KDE)', linewidth=2)
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_boxplot_comparison(
        data_dict: Dict[str, np.ndarray],
        title: str = "Boxplot Comparison",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot boxplot comparison.
        
        Parameters
        ----------
        data_dict : Dict[str, np.ndarray]
            {label: data_array}
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        figsize : Tuple[int, int]
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        data_list = list(data_dict.values())
        labels = list(data_dict.keys())
        
        bp = ax.boxplot(data_list, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_list)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # Test training history visualization
    train_loss = np.exp(-np.linspace(0, 3, 30)) + np.random.randn(30) * 0.05
    val_loss = np.exp(-np.linspace(0, 2.5, 30)) + np.random.randn(30) * 0.08
    
    viz = TrainingHistoryVisualizer()
    viz.plot_loss_curves(train_loss.tolist(), val_loss.tolist())
    
    # Test model comparison
    results = {
        'HMM': {'ba': 0.72, 'auc': 0.79, 'recall': 0.68},
        'CNN': {'ba': 0.82, 'auc': 0.87, 'recall': 0.79},
        'RNN': {'ba': 0.85, 'auc': 0.89, 'recall': 0.82}
    }
    
    comp_viz = ModelComparisonVisualizer()
    comp_viz.plot_metric_comparison(results)