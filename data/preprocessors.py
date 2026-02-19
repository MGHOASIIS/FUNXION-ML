"""
Base preprocessor classes for feature extraction and data preparation.

This module provides model-agnostic preprocessing that can be extended
for specific model requirements.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import cdist_dtw
from sklearn.manifold import MDS, Isomap, TSNE

from data.transforms import (
    Downsample, TimeJitter, TimeWarping, MagnitudeWarping
)


class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False
    
    @abstractmethod
    def prepare_data(
        self,
        g1: Dict,
        g0: Dict
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data for model training.
        
        Parameters
        ----------
        g1 : Dict
            Group 1 data (e.g., patients) {id: tensor}
        g0 : Dict
            Group 0 data (e.g., controls) {id: tensor}
        
        Returns
        -------
        X : np.ndarray
            Prepared features
        y : np.ndarray
            Labels (1 for g1, 0 for g0)
        subject_ids : np.ndarray or None
            Subject identifiers for cross-validation
        """
        pass
    
    def _collect_sequences(
        self,
        g1: Dict,
        g0: Dict
    ) -> Tuple[List, np.ndarray, List[str]]:
        """
        Collect sequences from both groups.
        
        Returns
        -------
        all_tensors : List
            List of tensors (one per subject)
        labels : np.ndarray
            Labels for each subject
        subject_ids : List[str]
            Subject identifiers
        """
        all_tensors = []
        labels = []
        subject_ids = []
        
        # Group 1 (patients/condition of interest)
        for idx, (k, tensor_data) in enumerate(g1.items()):
            all_tensors.append(tensor_data)
            labels.append(1)
            subject_ids.append(f"g1_{idx}_{k}")
        
        # Group 0 (controls/comparison group)
        for idx, (k, tensor_data) in enumerate(g0.items()):
            all_tensors.append(tensor_data)
            labels.append(0)
            subject_ids.append(f"g0_{idx}_{k}")
        
        return all_tensors, np.array(labels, dtype=np.int32), subject_ids
    
    def _extract_signals(self, tensors: List) -> List:
        """Extract signal columns (remove timestamp if present)."""
        return [t[:, 1:] if t.shape[1] > 18 else t for t in tensors]
    
    def _to_numpy(self, tensor):
        """Convert tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)


class TruncatePreprocessor(BasePreprocessor):
    """Truncate sequences to minimum length."""
    
    def __init__(self, output_format: str = "3d"):
        """
        Parameters
        ----------
        output_format : str
            '3d' for (N, T, C) - RNN/HMM
            'channels_first' for (N, C, T) - CNN
        """
        super().__init__()
        self.output_format = output_format
    
    def prepare_data(
        self,
        g1: Dict,
        g0: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data by truncating to minimum length."""
        all_tensors, y, subject_ids = self._collect_sequences(g1, g0)
        signals = self._extract_signals(all_tensors)
        
        # Find minimum length
        T_min = min(t.shape[0] for t in signals)
        print(f"\n[TruncatePreprocessor] Truncating to T_min = {T_min}")
        
        # Truncate from the end (keep last T_min timesteps)
        truncated = [self._to_numpy(t[-T_min:]) for t in signals]
        X = np.stack(truncated, axis=0)  # (N, T_min, 18)
        
        # Apply scaling
        N, T, C = X.shape
        X_2d = X.reshape(N * T, C)
        X_scaled_2d = self.scaler.fit_transform(X_2d)
        X_scaled = X_scaled_2d.reshape(N, T, C)
        
        # Format conversion if needed
        if self.output_format == "channels_first":
            X_scaled = X_scaled.transpose(0, 2, 1)  # (N, C, T)
            print(f"[TruncatePreprocessor] Output shape (channels-first): {X_scaled.shape}")
        else:
            print(f"[TruncatePreprocessor] Output shape (3D): {X_scaled.shape}")
        
        return X_scaled, y, np.array(subject_ids)


class SlidingWindowPreprocessor(BasePreprocessor):
    """Create sliding windows from sequences."""
    
    def __init__(
        self,
        window_size: int = 300,
        overlap: float = 0.30,
        output_format: str = "3d"
    ):
        """
        Parameters
        ----------
        window_size : int
            Number of timesteps per window
        overlap : float
            Overlap percentage (0.0 to 1.0)
        output_format : str
            '3d' for (N_windows, T, C)
            'channels_first' for (N_windows, C, T)
        """
        super().__init__()
        self.window_size = window_size
        self.overlap = overlap
        self.stride = int(window_size * (1 - overlap))
        self.output_format = output_format
    
    def prepare_data(
        self,
        g1: Dict,
        g0: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data using sliding windows."""
        all_tensors, labels, subject_ids = self._collect_sequences(g1, g0)
        signals = self._extract_signals(all_tensors)
        
        all_windows = []
        all_window_labels = []
        all_window_subject_ids = []
        
        print(f"\n[SlidingWindowPreprocessor] Window size: {self.window_size}, "
              f"Stride: {self.stride}, Overlap: {self.overlap:.1%}")
        
        for signal, label, subject_id in zip(signals, labels, subject_ids):
            signal_np = self._to_numpy(signal)
            T = signal_np.shape[0]
            
            # Extract windows
            for start_idx in range(0, T - self.window_size + 1, self.stride):
                end_idx = start_idx + self.window_size
                window = signal_np[start_idx:end_idx]
                
                if window.shape[0] == self.window_size:
                    all_windows.append(window)
                    all_window_labels.append(label)
                    all_window_subject_ids.append(subject_id)
        
        print(f"[SlidingWindowPreprocessor] Total windows: {len(all_windows)}")
        print(f"[SlidingWindowPreprocessor] Unique subjects: {len(set(all_window_subject_ids))}")
        
        # Stack and scale
        X = np.stack(all_windows, axis=0)  # (N_windows, T, C)
        y = np.array(all_window_labels, dtype=np.int32)
        
        N, T, C = X.shape
        X_2d = X.reshape(N * T, C)
        X_scaled_2d = self.scaler.fit_transform(X_2d)
        X_scaled = X_scaled_2d.reshape(N, T, C)
        
        # Format conversion if needed
        if self.output_format == "channels_first":
            X_scaled = X_scaled.transpose(0, 2, 1)
            print(f"[SlidingWindowPreprocessor] Output shape (channels-first): {X_scaled.shape}")
        else:
            print(f"[SlidingWindowPreprocessor] Output shape (3D): {X_scaled.shape}")
        
        return X_scaled, y, np.array(all_window_subject_ids)



class VariableLengthPreprocessor(BasePreprocessor):
    """
    Preprocessor for models that natively handle variable-length sequences.

    Designed specifically for HMMs (and future sequence models) that use
    the hmmlearn 'lengths' parameter — no truncation, padding, or stacking
    is applied.  Each subject keeps their full recording.

    What it does
    ------------
    1. Extracts raw (T_i, 18) arrays from g1/g0 (drops timestamp col if present)
    2. Z-score normalises using a single StandardScaler fit on all data
       concatenated — same approach as TruncatePreprocessor, preserving
       inter-subject differences in absolute sensor values
    3. Returns X as a plain Python list of (T_i, 18) arrays — NOT stacked
       This is what hmmlearn expects: a list where each element can have
       a different T_i (sequence length)

    Why this matters for HMM
    ------------------------
    TruncatePreprocessor forces all sequences to T_min (shortest subject).
    For jar opening (Task 1), the shortest control recording may be ~20s
    while the longest patient recording is ~400s.  At 50 Hz that means
    patients lose up to 19,000 frames from the START of their recording —
    exactly where "Jar picked up" and early lid-grabbing events occur.
    The HMM then never sees the most clinically relevant part of the task.

    Returns
    -------
    X  : list of np.ndarray, each shape (T_i, 18)   ← variable length
    y  : np.ndarray shape (N,)
    subject_ids : np.ndarray shape (N,)
    """

    def prepare_data(
        self,
        g1: Dict,
        g0: Dict
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        """
        Extract variable-length z-scored sequences from g1 and g0.

        Parameters
        ----------
        g1 : Dict  {subject_id: tensor (T_i, C) or (T_i, C+1)}
        g0 : Dict  {subject_id: tensor (T_i, C) or (T_i, C+1)}

        Returns
        -------
        X           : list of np.ndarray (T_i, 18)
        y           : np.ndarray (N,)
        subject_ids : np.ndarray (N,)
        """
        all_tensors, y, subject_ids = self._collect_sequences(g1, g0)
        signals = self._extract_signals(all_tensors)
        signals = [self._to_numpy(s) for s in signals]

        # Lengths before normalisation — for logging
        lengths = [s.shape[0] for s in signals]
        print(f"\n[VariableLengthPreprocessor] {len(signals)} sequences")
        print(f"  Length min={min(lengths)}  max={max(lengths)}  "
              f"mean={int(np.mean(lengths))}  (frames @ 50 Hz)")
        print(f"  No truncation — each subject keeps their full recording")

        # Z-score normalise: fit scaler on all frames concatenated
        # then transform each sequence independently
        all_frames = np.vstack(signals)           # (sum(T_i), 18)
        self.scaler.fit(all_frames)
        X_scaled = [self.scaler.transform(s) for s in signals]

        print(f"  Z-score normalised (scaler fit on {all_frames.shape[0]} frames)")

        return X_scaled, y, np.array(subject_ids)


class PreprocessorFactory:
    """Factory for creating preprocessors."""
    
    @staticmethod
    def create(
        method: str,
        model_type: str,
        **kwargs
    ) -> BasePreprocessor:
        """
        Create a preprocessor.
        
        Parameters
        ----------
        method : str
            'truncate', 'sliding_window', 'padding', 'dtw_embedding','variable_length'
        model_type : str
            'hmm', 'cnn', or 'rnn'
        **kwargs
            Additional parameters for preprocessor
        
        Returns
        -------
        BasePreprocessor
            Configured preprocessor instance
        """
        # Determine output format based on model type
        if model_type.lower() == "cnn":
            output_format = "channels_first"
        else:
            output_format = "3d"
        
        if method == "truncate":
            return TruncatePreprocessor(output_format=output_format)
        elif method == "sliding_window":
            return SlidingWindowPreprocessor(
                output_format=output_format,
                **kwargs
            )
        elif method == "padding":
            return PaddingPreprocessor(output_format=output_format)
        elif method == "dtw_embedding":
            return DTWEmbeddingPreprocessor(**kwargs)
        elif method == "downsample_truncate":
            return DownsampleTruncatePreprocessor(
                output_format=output_format,
                **kwargs
            )
        elif method == "variable_length":
            return VariableLengthPreprocessor()
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")


class PaddingPreprocessor(BasePreprocessor):
    """Pad sequences to maximum length."""
    
    def __init__(self, output_format: str = "3d", pad_value: float = 0.0):
        """
        Parameters
        ----------
        output_format : str
            '3d' for (N, T_max, C) or 'channels_first' for (N, C, T_max)
        pad_value : float
            Value to use for padding
        """
        super().__init__()
        self.output_format = output_format
        self.pad_value = pad_value
    
    def prepare_data(
        self,
        g1: Dict,
        g0: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data by padding to maximum length."""
        all_tensors, y, subject_ids = self._collect_sequences(g1, g0)
        signals = self._extract_signals(all_tensors)
        
        # Find maximum length
        T_max = max(t.shape[0] for t in signals)
        print(f"\n[PaddingPreprocessor] Padding to T_max = {T_max}")
        
        # Pad sequences
        padded_list = []
        for t in signals:
            t_np = self._to_numpy(t)
            T, C = t_np.shape
            
            if T < T_max:
                # Pad with pad_value
                pad_width = ((0, T_max - T), (0, 0))
                padded = np.pad(t_np, pad_width, mode='constant', 
                               constant_values=self.pad_value)
            else:
                padded = t_np
            
            padded_list.append(padded)
        
        X = np.stack(padded_list, axis=0)  # (N, T_max, C)
        
        # Apply scaling
        N, T, C = X.shape
        X_2d = X.reshape(N * T, C)
        X_scaled_2d = self.scaler.fit_transform(X_2d)
        X_scaled = X_scaled_2d.reshape(N, T, C)
        
        # Format conversion if needed
        if self.output_format == "channels_first":
            X_scaled = X_scaled.transpose(0, 2, 1)  # (N, C, T)
            print(f"[PaddingPreprocessor] Output shape (channels-first): {X_scaled.shape}")
        else:
            print(f"[PaddingPreprocessor] Output shape (3D): {X_scaled.shape}")
        
        return X_scaled, y, np.array(subject_ids)


class DTWEmbeddingPreprocessor(BasePreprocessor):
    """Use DTW distance matrix with MDS for dimensionality reduction."""
    
    def __init__(
        self,
        n_components: int = 10,
        method: str = "mds"
    ):
        """
        Parameters
        ----------
        n_components : int
            Number of embedding dimensions
        method : str
            'mds', 'isomap', or 'tsne'
        """
        super().__init__()
        self.n_components = n_components
        self.method = method.lower()
    
    def prepare_data(
        self,
        g1: Dict,
        g0: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data using DTW + embedding."""
        all_tensors, y, subject_ids = self._collect_sequences(g1, g0)
        signals = self._extract_signals(all_tensors)
        
        print(f"\n[DTWEmbeddingPreprocessor] Computing DTW distance matrix...")
        
        # Convert to list of numpy arrays
        all_numpy = [self._to_numpy(s) for s in signals]
        
        # Compute DTW distance matrix
        D = cdist_dtw(all_numpy)
        print(f"[DTWEmbeddingPreprocessor] DTW matrix shape: {D.shape}")
        
        # Apply embedding
        print(f"[DTWEmbeddingPreprocessor] Applying {self.method.upper()} embedding...")
        
        if self.method == "mds":
            embedder = MDS(
                n_components=self.n_components,
                dissimilarity="precomputed",
                random_state=42
            )
        elif self.method == "isomap":
            embedder = Isomap(
                n_components=self.n_components,
                metric="precomputed",
                n_neighbors=min(5, len(all_numpy) - 1)
            )
        elif self.method == "tsne":
            embedder = TSNE(
                n_components=min(self.n_components, 3),
                metric="precomputed",
                random_state=42
            )
        else:
            raise ValueError(f"Unknown embedding method: {self.method}")
        
        X = embedder.fit_transform(D)
        print(f"[DTWEmbeddingPreprocessor] Embedded shape: {X.shape}")
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, np.array(subject_ids)


class DownsampleTruncatePreprocessor(BasePreprocessor):
    """Downsample then truncate sequences."""
    
    def __init__(
        self,
        target_rate: int = 25,
        original_rate: int = 50,
        output_format: str = "3d"
    ):
        """
        Parameters
        ----------
        target_rate : int
            Target sampling rate in Hz
        original_rate : int
            Original sampling rate in Hz
        output_format : str
            '3d' or 'channels_first'
        """
        super().__init__()
        self.target_rate = target_rate
        self.original_rate = original_rate
        self.output_format = output_format
        self.downsampler = Downsample(target_rate, original_rate)
    
    def prepare_data(
        self,
        g1: Dict,
        g0: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Downsample and truncate data."""
        all_tensors, y, subject_ids = self._collect_sequences(g1, g0)
        signals = self._extract_signals(all_tensors)
        
        print(f"\n[DownsampleTruncatePreprocessor] Downsampling "
              f"{self.original_rate}Hz → {self.target_rate}Hz")
        
        # Downsample all sequences
        downsampled = [self.downsampler.transform(self._to_numpy(s)) 
                      for s in signals]
        
        # Find minimum length after downsampling
        T_min = min(d.shape[0] for d in downsampled)
        print(f"[DownsampleTruncatePreprocessor] Truncating to T_min = {T_min}")
        
        # Truncate from the end
        truncated = [d[-T_min:] for d in downsampled]
        X = np.stack(truncated, axis=0)
        
        # Scale
        N, T, C = X.shape
        X_2d = X.reshape(N * T, C)
        X_scaled_2d = self.scaler.fit_transform(X_2d)
        X_scaled = X_scaled_2d.reshape(N, T, C)
        
        # Format conversion
        if self.output_format == "channels_first":
            X_scaled = X_scaled.transpose(0, 2, 1)
            print(f"[DownsampleTruncatePreprocessor] Output shape: {X_scaled.shape}")
        else:
            print(f"[DownsampleTruncatePreprocessor] Output shape: {X_scaled.shape}")
        
        return X_scaled, y, np.array(subject_ids)


class AugmentedPreprocessor(BasePreprocessor):
    """Preprocessor with data augmentation."""
    
    def __init__(
        self,
        base_preprocessor: BasePreprocessor,
        augmentations: List[str] = None,
        n_augmentations: int = 2
    ):
        """
        Parameters
        ----------
        base_preprocessor : BasePreprocessor
            Base preprocessing method
        augmentations : List[str]
            List of augmentation types: 'jitter', 'time_warp', 'magnitude_warp'
        n_augmentations : int
            Number of augmented samples per original sample
        """
        super().__init__()
        self.base_preprocessor = base_preprocessor
        self.n_augmentations = n_augmentations
        
        if augmentations is None:
            augmentations = ['jitter', 'time_warp']
        
        # Create augmentation objects
        self.augmenters = []
        for aug_type in augmentations:
            if aug_type == 'jitter':
                self.augmenters.append(TimeJitter(sigma=0.01))
            elif aug_type == 'time_warp':
                self.augmenters.append(TimeWarping(sigma=0.2, knot=4))
            elif aug_type == 'magnitude_warp':
                self.augmenters.append(MagnitudeWarping(sigma=0.2, knot=4))
    
    def prepare_data(
        self,
        g1: Dict,
        g0: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data with augmentation."""
        # Get base preprocessed data
        X_base, y_base, subject_ids_base = self.base_preprocessor.prepare_data(g1, g0)
        
        print(f"\n[AugmentedPreprocessor] Base data: {X_base.shape}")
        print(f"[AugmentedPreprocessor] Applying {len(self.augmenters)} augmentations, "
              f"{self.n_augmentations} times each")
        
        # Store augmented data
        X_list = [X_base]
        y_list = [y_base]
        subject_ids_list = [subject_ids_base]
        
        # Apply augmentations
        for aug_idx in range(self.n_augmentations):
            for aug in self.augmenters:
                X_aug = np.zeros_like(X_base)
                
                # Augment each sample
                for i in range(len(X_base)):
                    X_aug[i] = aug.transform(X_base[i], seed=42 + aug_idx + i)
                
                X_list.append(X_aug)
                y_list.append(y_base.copy())
                
                # Update subject IDs to reflect augmentation
                aug_subject_ids = np.array([
                    f"{sid}_aug{aug_idx}_{type(aug).__name__}" 
                    for sid in subject_ids_base
                ])
                subject_ids_list.append(aug_subject_ids)
        
        # Concatenate all data
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        subject_ids_all = np.concatenate(subject_ids_list, axis=0)
        
        print(f"[AugmentedPreprocessor] Final data: {X_all.shape}")
        print(f"[AugmentedPreprocessor] Original samples: {len(X_base)}")
        print(f"[AugmentedPreprocessor] Total samples: {len(X_all)}")
        
        return X_all, y_all, subject_ids_all
    

# TODO:
# class CustomAugmentedPreprocessor(BasePreprocessor):
#     def __init__(self, base_method="truncate", model_type="rnn"):
#         super().__init__()
#         self.base_method = base_method
#         self.model_type = model_type
        
#         # Your custom augmentations
#         from data.transforms import TimeJitter, TimeWarping
#         self.augmenters = [
#             TimeJitter(sigma=0.01),
#             TimeWarping(sigma=0.2, knot=4)
#         ]
    
#     def prepare_data(self, g1, g0):
#         # 1. Base preprocessing
#         all_tensors, y, subject_ids = self._collect_sequences(g1, g0)
#         signals = self._extract_signals(all_tensors)
        
#         # 2. Apply augmentation
#         aug_signals = []
#         aug_labels = []
#         aug_subjects = []
        
#         for signal, label, subject_id in zip(signals, y, subject_ids):
#             # Original
#             aug_signals.append(signal)
#             aug_labels.append(label)
#             aug_subjects.append(subject_id)
            
#             # Augmented versions
#             for i, augmenter in enumerate(self.augmenters):
#                 aug_signal = augmenter.transform(signal)
#                 aug_signals.append(aug_signal)
#                 aug_labels.append(label)
#                 aug_subjects.append(f"{subject_id}_aug{i}")
        
#         # 3. Continue with standard processing
#         # ... (truncate, scale, format)
        
#         return X, y, subject_ids