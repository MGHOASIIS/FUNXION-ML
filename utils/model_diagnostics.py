"""
Model Diagnostics & Interpretability for Clinical ML

State-of-the-art tools for understanding WHAT your model learned and HOW:
- Gradient analysis
- Activation analysis  
- Attention visualization (for Transformers)
- Saliency maps
- Layer-wise relevance propagation
- Clinical interpretation helpers
"""
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DiagnosticResults:
    """Results from model diagnostics."""
    gradient_norms: Dict[str, float]
    activation_stats: Dict[str, Dict[str, float]]
    weight_distributions: Dict[str, np.ndarray]
    dead_neurons: Dict[str, int]
    summary: List[str]


class GradientDiagnostics:
    """
    Analyze gradients to understand training dynamics.
    
    Critical for detecting:
    - Vanishing gradients (model not learning)
    - Exploding gradients (unstable training)
    - Dead neurons (ReLU neurons that never activate)
    """
    
    @staticmethod
    def analyze_gradients(
        model: Any,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: str = 'cuda'
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze gradients for all layers.
        
        Parameters
        ----------
        model : Any
            Model to analyze (nn.Module or wrapper with trained_model)
        dataloader : DataLoader
            Data for forward/backward pass
        criterion : nn.Module
            Loss function
        device : str
            Device
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            {layer_name: {'mean': ..., 'max': ..., 'norm': ...}}
        """
        # ⭐ Handle wrapper models
        if not isinstance(model, nn.Module):
            if hasattr(model, 'trained_model') and model.trained_model is not None:
                actual_model = model.trained_model
            else:
                print(f"\n⚠️  Gradient Analysis SKIPPED")
                print(f"   Model is a wrapper without trained nn.Module")
                print(f"   (This analysis requires access to actual PyTorch model)")
                return {}
        else:
            actual_model = model
        
        actual_model.train()
        actual_model.zero_grad()
        
        # Single batch for analysis
        X, y = next(iter(dataloader))
        X, y = X.to(device), y.to(device)
        
        # Forward + backward
        output = actual_model(X)
        loss = criterion(output, y)
        loss.backward()
        
        # Collect gradients
        gradient_stats = {}
        
        for name, param in actual_model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().cpu().numpy()
                
                gradient_stats[name] = {
                    'mean': float(np.abs(grad).mean()),
                    'max': float(np.abs(grad).max()),
                    'min': float(np.abs(grad).min()),
                    'std': float(grad.std()),
                    'norm': float(np.linalg.norm(grad))
                }
        
        # Print analysis
        print(f"\n{'='*70}")
        print("GRADIENT ANALYSIS")
        print(f"{'='*70}")
        
        for name, stats in gradient_stats.items():
            print(f"\n{name}:")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Max:  {stats['max']:.6f}")
            print(f"  Norm: {stats['norm']:.6f}")
            
            # Warnings
            if stats['mean'] < 1e-7:
                print(f"  ⚠️  VANISHING gradients detected!")
            if stats['max'] > 10:
                print(f"  ⚠️  EXPLODING gradients detected!")
        
        print(f"{'='*70}\n")
        
        return gradient_stats
    
    @staticmethod
    def plot_gradient_flow(
        gradient_stats: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """
        Visualize gradient flow through network.
        
        Parameters
        ----------
        gradient_stats : Dict
            Gradient statistics from analyze_gradients
        save_path : str, optional
            Save path
        """
        layer_names = list(gradient_stats.keys())
        mean_grads = [gradient_stats[name]['mean'] for name in layer_names]
        max_grads = [gradient_stats[name]['max'] for name in layer_names]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Mean gradients
        axes[0].bar(range(len(layer_names)), mean_grads, alpha=0.7)
        axes[0].set_xticks(range(len(layer_names)))
        axes[0].set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
        axes[0].set_ylabel('Mean Absolute Gradient')
        axes[0].set_title('Mean Gradient Magnitude per Layer')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(1e-7, color='red', linestyle='--', label='Vanishing threshold')
        axes[0].legend()
        
        # Max gradients
        axes[1].bar(range(len(layer_names)), max_grads, alpha=0.7, color='orange')
        axes[1].set_xticks(range(len(layer_names)))
        axes[1].set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel('Max Absolute Gradient')
        axes[1].set_title('Maximum Gradient per Layer')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(10, color='red', linestyle='--', label='Exploding threshold')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()


class ActivationAnalyzer:
    """
    Analyze neuron activations to understand what model learns.
    
    Helps identify:
    - Dead neurons (always zero)
    - Saturated neurons (always max)
    - Discriminative features
    """
    
    @staticmethod
    def extract_activations(
        model: Any,
        X: torch.Tensor,
        layer_names: Optional[List[str]] = None,
        device: str = 'cuda'
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations from specified layers.
        
        Parameters
        ----------
        model : Any
            Model (nn.Module or wrapper)
        X : torch.Tensor
            Input data
        layer_names : List[str], optional
            Layers to extract from (if None, extract all)
        device : str
            Device
        
        Returns
        -------
        Dict[str, torch.Tensor]
            {layer_name: activations}
        """        
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].detach()  # Handle RNN tuple outputs
                else:
                    activations[name] = output.detach()     # Handle regular tensor outputs
            return hook
        
        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if layer_names is None or name in layer_names:
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.GRU, nn.LSTM)):
                    handle = module.register_forward_hook(hook_fn(name))
                    handles.append(handle)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(X.to(device))
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return activations
    
    @staticmethod
    def analyze_activations(
        activations: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze activation statistics.
        
        Parameters
        ----------
        activations : Dict[str, torch.Tensor]
            Activations from extract_activations
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            {layer_name: statistics}
        """
        stats = {}
        
        print(f"\n{'='*70}")
        print("ACTIVATION ANALYSIS")
        print(f"{'='*70}")
        
        for name, activation in activations.items():
            act_np = activation.cpu().numpy()
            
            # Flatten to analyze all neurons
            act_flat = act_np.flatten()
            
            # Compute statistics
            layer_stats = {
                'mean': float(np.mean(act_flat)),
                'std': float(np.std(act_flat)),
                'max': float(np.max(act_flat)),
                'min': float(np.min(act_flat)),
                'sparsity': float(np.mean(act_flat == 0)),  # % of zeros
                'saturation': float(np.mean(np.abs(act_flat) > 0.99))  # % saturated
            }
            
            stats[name] = layer_stats
            
            print(f"\n{name}:")
            print(f"  Shape:      {activation.shape}")
            print(f"  Mean:       {layer_stats['mean']:.4f}")
            print(f"  Std:        {layer_stats['std']:.4f}")
            print(f"  Sparsity:   {layer_stats['sparsity']:.2%}")
            print(f"  Saturation: {layer_stats['saturation']:.2%}")
            
            # Warnings
            if layer_stats['sparsity'] > 0.7:
                print(f"  ⚠️  HIGH SPARSITY - Many dead neurons!")
            if layer_stats['saturation'] > 0.3:
                print(f"  ⚠️  HIGH SATURATION - Neurons saturating!")
        
        print(f"{'='*70}\n")
        
        return stats
    
    @staticmethod
    def plot_activation_distributions(
        activations: Dict[str, torch.Tensor],
        save_path: Optional[str] = None
    ):
        """Plot activation distributions for each layer."""
        n_layers = len(activations)
        n_cols = 3
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_layers > 1 else [axes]
        
        for idx, (name, activation) in enumerate(activations.items()):
            ax = axes[idx]
            
            act_flat = activation.cpu().numpy().flatten()
            
            ax.hist(act_flat, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{name}\n(Sparsity: {np.mean(act_flat==0):.1%})', fontsize=10)
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_layers, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Activation Distributions per Layer', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()


class SaliencyAnalyzer:
    """
    Compute saliency maps to understand which inputs are important.
    
    For time-series: Shows which timesteps and channels matter most.
    """
    
    @staticmethod
    def compute_saliency(
        model: nn.Module,
        X: torch.Tensor,
        target_class: int,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Compute saliency map using gradients.
        
        Parameters
        ----------
        model : nn.Module
            Model
        X : torch.Tensor
            Input (batch, ...)
        target_class : int
            Class to compute saliency for
        device : str
            Device
        
        Returns
        -------
        torch.Tensor
            Saliency map (same shape as X)
        """
        model.train()
        X = X.to(device).requires_grad_(True)
        
        # Forward
        output = model(X)
        
        # Backward w.r.t. target class
        model.zero_grad()
        output[0, target_class].backward()
        
        # Saliency = absolute gradient
        saliency = X.grad.abs()
        
        return saliency
    
    @staticmethod
    def plot_saliency_map(
        saliency: torch.Tensor,
        channel_names: Optional[List[str]] = None,
        title: str = "Saliency Map",
        save_path: Optional[str] = None
    ):
        """
        Visualize saliency map.
        
        Parameters
        ----------
        saliency : torch.Tensor
            Saliency map (T, C) or (1, T, C) or (1, C, T)
        channel_names : List[str], optional
            Channel names
        title : str
            Plot title
        save_path : str, optional
            Save path
        """
        # Handle different shapes
        if saliency.ndim == 3:
            saliency = saliency.squeeze(0)
        
        # Ensure (T, C) format
        if saliency.shape[0] < saliency.shape[1]:
            saliency = saliency.T
        
        saliency_np = saliency.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        im = ax.imshow(saliency_np.T, aspect='auto', cmap='hot', interpolation='nearest')
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Channel', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        if channel_names:
            ax.set_yticks(range(len(channel_names)))
            ax.set_yticklabels(channel_names, fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()
    
    @staticmethod
    def compute_integrated_gradients(
        model: nn.Module,
        X: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        target_class: int = 1,
        steps: int = 50,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Compute integrated gradients (more robust than vanilla gradients).
        
        Parameters
        ----------
        model : nn.Module
            Model
        X : torch.Tensor
            Input
        baseline : torch.Tensor, optional
            Baseline (if None, use zeros)
        target_class : int
            Target class
        steps : int
            Integration steps
        device : str
            Device
        
        Returns
        -------
        torch.Tensor
            Integrated gradients
        """
        if baseline is None:
            baseline = torch.zeros_like(X)
        
        # Create interpolation path
        alphas = torch.linspace(0, 1, steps, device=device)
        
        gradients = []
        
        for alpha in alphas:
            # Interpolated input
            X_interp = baseline + alpha * (X - baseline)
            X_interp = X_interp.to(device).requires_grad_(True)
            
            # Forward
            output = model(X_interp)
            
            # Backward
            model.zero_grad()
            output[0, target_class].backward()
            
            # Store gradient
            gradients.append(X_interp.grad.detach())
        
        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Integrated gradients = (X - baseline) * avg_gradients
        integrated_grads = (X - baseline) * avg_gradients
        
        return integrated_grads


class WeightDiagnostics:
    """
    Analyze model weights to understand learned representations.
    """
    
    @staticmethod
    def analyze_weight_distributions(
        model: nn.Module
    ) -> Dict[str, np.ndarray]:
        """
        Analyze weight distributions for each layer.
        
        Parameters
        ----------
        model : nn.Module
            Model to analyze
        
        Returns
        -------
        Dict[str, np.ndarray]
            {layer_name: weights}
        """
        weight_dists = {}
        
        print(f"\n{'='*70}")
        print("WEIGHT DISTRIBUTION ANALYSIS")
        print(f"{'='*70}")
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                weights = param.detach().cpu().numpy()
                weight_dists[name] = weights
                
                print(f"\n{name}:")
                print(f"  Shape:  {weights.shape}")
                print(f"  Mean:   {weights.mean():.6f}")
                print(f"  Std:    {weights.std():.6f}")
                print(f"  Min:    {weights.min():.6f}")
                print(f"  Max:    {weights.max():.6f}")
                
                # Check for issues
                if np.abs(weights.mean()) > 1:
                    print(f"  ⚠️  Large mean - may need better initialization")
                if weights.std() < 0.01:
                    print(f"  ⚠️  Low std - weights not learning?")
                if weights.std() > 2:
                    print(f"  ⚠️  High std - unstable weights")
        
        print(f"{'='*70}\n")
        
        return weight_dists
    
    @staticmethod
    def plot_weight_distributions(
        weight_dists: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ):
        """Plot weight distributions."""
        n_layers = len(weight_dists)
        n_cols = 3
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_layers > 1 else [axes]
        
        for idx, (name, weights) in enumerate(weight_dists.items()):
            ax = axes[idx]
            
            weights_flat = weights.flatten()
            
            ax.hist(weights_flat, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
            ax.set_title(f'{name}\n(μ={weights.mean():.3f}, σ={weights.std():.3f})', 
                        fontsize=10)
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_layers, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Weight Distributions per Layer', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        
        plt.show()


class ClinicalInterpreter:
    """
    Clinical interpretation helpers for XDash project.
    
    Maps DL insights to clinical understanding.
    """
    
    @staticmethod
    def interpret_sensor_importance(
        importance_dict: Dict[str, float],
        top_k: int = 10
    ) -> str:
        """
        Generate clinical interpretation of sensor importance.
        
        Parameters
        ----------
        importance_dict : Dict[str, float]
            {sensor_name: importance_score}
        top_k : int
            Top features to interpret
        
        Returns
        -------
        str
            Clinical interpretation
        """
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        interpretation = []
        interpretation.append(f"\nCLINICAL INTERPRETATION - Top {top_k} Features:\n")
        interpretation.append("="*60)
        
        # Group features by sensor
        head_features = []
        right_hand_features = []
        left_hand_features = []
        
        for feat, score in sorted_features[:top_k]:
            if 'head' in feat:
                head_features.append((feat, score))
            elif 'right' in feat:
                right_hand_features.append((feat, score))
            elif 'left' in feat:
                left_hand_features.append((feat, score))
        
        # Interpret by sensor group
        if head_features:
            interpretation.append("\n📍 HEAD MOVEMENTS:")
            for feat, score in head_features:
                interpretation.append(f"  • {feat}: {score:.4f}")
                if 'pos_x' in feat:
                    interpretation.append("    → Lateral head tilt/movement during task")
                elif 'pos_y' in feat:
                    interpretation.append("    → Vertical head movement (compensation?)")
                elif 'pos_z' in feat:
                    interpretation.append("    → Forward/backward head position")
        
        if right_hand_features:
            interpretation.append("\n✋ RIGHT HAND (Dominant for most patients):")
            for feat, score in right_hand_features:
                interpretation.append(f"  • {feat}: {score:.4f}")
                if 'pos' in feat:
                    interpretation.append("    → Hand position trajectory")
                elif 'rot' in feat:
                    interpretation.append("    → Wrist/hand rotation pattern")
        
        if left_hand_features:
            interpretation.append("\n✋ LEFT HAND:")
            for feat, score in left_hand_features:
                interpretation.append(f"  • {feat}: {score:.4f}")
        
        # Overall interpretation
        interpretation.append("\n💡 BIOMECHANICAL INSIGHTS:")
        
        if len(head_features) >= 3:
            interpretation.append("  • Patients show compensatory head movements")
        
        if len(right_hand_features) > len(left_hand_features):
            interpretation.append("  • Right hand dominant for classification")
            interpretation.append("    (expected for right-dominant population)")
        
        interpretation.append("\n" + "="*60)
        
        return '\n'.join(interpretation)


# Example usage
if __name__ == "__main__":
    # Create dummy model
    model = nn.Sequential(
        nn.Linear(18, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    
    # Dummy data
    X = torch.randn(10, 18)
    y = torch.randint(0, 2, (10,))
    
    from torch.utils.data import DataLoader, TensorDataset
    dataloader = DataLoader(TensorDataset(X, y), batch_size=10)
    
    # Analyze gradients
    grad_diag = GradientDiagnostics()
    gradient_stats = grad_diag.analyze_gradients(model, dataloader, nn.CrossEntropyLoss())
    grad_diag.plot_gradient_flow(gradient_stats)
    
    # Analyze activations
    act_analyzer = ActivationAnalyzer()
    activations = act_analyzer.extract_activations(model, X, device='cpu')
    act_stats = act_analyzer.analyze_activations(activations)