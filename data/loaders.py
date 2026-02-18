"""
Data loading utilities for XDash project.

Handles loading pickled datasets, JSON event markers, and patient metadata.
"""
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pandas as pd
import torch
import numpy as np

from config.paths import get_pickled_dataset_path, PATIENT_DETAILS, MATERIALS_DIR


class DataLoader:
    """Main data loader for XDash datasets."""
    
    def __init__(self, task: int):
        """
        Initialize data loader for a specific task.
        
        Parameters
        ----------
        task : int
            Task number (1-6)
        """
        self.task = task
        self.patient_data = None
        self.control_data = None
        self._loaded = False
    
    def load(self) -> Tuple[Dict, Dict]:
        """
        Load patient and control data.
        
        Returns
        -------
        patient_data : Dict
            {patient_id: tensor_data}
        control_data : Dict
            {control_id: tensor_data}
        """
        if self._loaded:
            return self.patient_data, self.control_data
        
        patient_path = get_pickled_dataset_path(self.task, "patient")
        control_path = get_pickled_dataset_path(self.task, "control")
        
        print(f"\n[DataLoader] Loading data for task {self.task}...")
        
        with open(patient_path, "rb") as f:
            self.patient_data = pickle.load(f)
        
        with open(control_path, "rb") as f:
            self.control_data = pickle.load(f)
        
        self._loaded = True
        
        print(f"[DataLoader] ✓ Loaded {len(self.patient_data)} patients")
        print(f"[DataLoader] ✓ Loaded {len(self.control_data)} controls")
        
        # Validate data
        self._validate_data()
        
        return self.patient_data, self.control_data
    
    def _validate_data(self):
        """Validate loaded data structure."""
        if not self.patient_data or not self.control_data:
            raise ValueError("No data loaded")
        
        # Check tensor shapes
        sample_patient = next(iter(self.patient_data.values()))
        sample_control = next(iter(self.control_data.values()))
        
        print(f"[DataLoader] Sample patient shape: {sample_patient.shape}")
        print(f"[DataLoader] Sample control shape: {sample_control.shape}")
        
        # Verify all tensors have consistent feature dimensions
        expected_features = sample_patient.shape[1]
        
        for pid, data in self.patient_data.items():
            if data.shape[1] != expected_features:
                raise ValueError(
                    f"Patient {pid} has inconsistent features: "
                    f"{data.shape[1]} vs expected {expected_features}"
                )
        
        for cid, data in self.control_data.items():
            if data.shape[1] != expected_features:
                raise ValueError(
                    f"Control {cid} has inconsistent features: "
                    f"{data.shape[1]} vs expected {expected_features}"
                )
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns
        -------
        Dict
            Statistics including sequence lengths, feature ranges, etc.
        """
        if not self._loaded:
            self.load()
        
        all_data = list(self.patient_data.values()) + list(self.control_data.values())
        
        # Sequence lengths
        lengths = [d.shape[0] for d in all_data]
        
        # Convert to numpy for statistics
        all_tensors_np = []
        for d in all_data:
            if isinstance(d, torch.Tensor):
                all_tensors_np.append(d.numpy())
            else:
                all_tensors_np.append(np.array(d))
        
        # Stack and compute statistics (excluding timestamp column if present)
        stacked = np.concatenate([d[:, 1:] if d.shape[1] > 18 else d 
                                  for d in all_tensors_np], axis=0)
        
        stats = {
            "n_patients": len(self.patient_data),
            "n_controls": len(self.control_data),
            "n_total": len(all_data),
            "sequence_length_min": int(np.min(lengths)),
            "sequence_length_max": int(np.max(lengths)),
            "sequence_length_mean": float(np.mean(lengths)),
            "sequence_length_std": float(np.std(lengths)),
            "feature_dim": stacked.shape[1],
            "feature_means": stacked.mean(axis=0).tolist(),
            "feature_stds": stacked.std(axis=0).tolist(),
            "feature_mins": stacked.min(axis=0).tolist(),
            "feature_maxs": stacked.max(axis=0).tolist()
        }
        
        return stats
    
    def print_summary(self):
        """Print a summary of the loaded data."""
        if not self._loaded:
            self.load()
        
        stats = self.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"Dataset Summary - Task {self.task}")
        print(f"{'='*60}")
        print(f"Total Subjects:  {stats['n_total']}")
        print(f"  Patients:      {stats['n_patients']}")
        print(f"  Controls:      {stats['n_controls']}")
        print(f"\nSequence Lengths:")
        print(f"  Min:           {stats['sequence_length_min']}")
        print(f"  Max:           {stats['sequence_length_max']}")
        print(f"  Mean:          {stats['sequence_length_mean']:.1f}")
        print(f"  Std:           {stats['sequence_length_std']:.1f}")
        print(f"\nFeatures:        {stats['feature_dim']}")
        print(f"{'='*60}\n")


class PatientMetadataLoader:
    """Load patient metadata and diagnosis information."""
    
    def __init__(self):
        """Initialize metadata loader."""
        self.df = None
        self._loaded = False
    
    def load(self) -> pd.DataFrame:
        """
        Load patient metadata from Excel file.
        
        Returns
        -------
        pd.DataFrame
            Patient metadata including diagnosis codes
        """
        if self._loaded:
            return self.df
        
        print(f"\n[PatientMetadataLoader] Loading metadata...")
        
        self.df = pd.read_excel(PATIENT_DETAILS, sheet_name='Sheet1', header=1)
        self._loaded = True
        
        print(f"[PatientMetadataLoader] ✓ Loaded metadata for {len(self.df)} patients")
        
        return self.df
    
    def get_diagnosis_groups(self) -> Dict[str, List]:
        """
        Get patient IDs grouped by diagnosis.
        
        Returns
        -------
        Dict[str, List]
            Diagnosis name mapped to list of patient IDs
        """
        if not self._loaded:
            self.load()
        
        # Diagnosis codes:
        # 1 = Rotator Cuff Tear (RCT)
        # 2 = Glenohumeral Arthritis (GA)
        # 3 = Shoulder Bursitis (SB)
        # 4 = Biceps Tendonitis (BT)
        
        diagnosis_mapping = {
            1: "rotator_cuff_tear",
            2: "glenohumeral_arthritis",
            3: "shoulder_bursitis",
            4: "biceps_tendonitis"
        }
        
        groups = {}
        for code, name in diagnosis_mapping.items():
            patient_ids = self.df[self.df['dia_code'] == code]['id'].tolist()
            groups[name] = patient_ids
            print(f"[PatientMetadataLoader] {name}: {len(patient_ids)} patients")
        
        return groups
    
    def get_laterality_info(self) -> Dict[str, List]:
        """
        Get patient IDs grouped by injury laterality.
        
        Returns
        -------
        Dict[str, List]
            Laterality (left/right) mapped to patient IDs
        """
        if not self._loaded:
            self.load()
        
        if 'laterality' not in self.df.columns:
            print("[PatientMetadataLoader] Warning: laterality column not found")
            return {}
        
        laterality = {
            "left": self.df[self.df['laterality'] == 'left']['id'].tolist(),
            "right": self.df[self.df['laterality'] == 'right']['id'].tolist()
        }
        
        print(f"[PatientMetadataLoader] Left injury: {len(laterality['left'])}")
        print(f"[PatientMetadataLoader] Right injury: {len(laterality['right'])}")
        
        return laterality


class EventMarkerLoader:
    """Load event markers for task segmentation."""
    
    def __init__(self, task: int):
        """
        Initialize event marker loader.
        
        Parameters
        ----------
        task : int
            Task number (1-6)
        """
        self.task = task
        self.events = None
        self._loaded = False
    
    def load(self, events_dir: Optional[Path] = None) -> Dict:
        """
        Load event markers from JSON files.
        
        Parameters
        ----------
        events_dir : Path, optional
            Directory containing event JSON files
            If None, uses MATERIALS_DIR / "events"
        
        Returns
        -------
        Dict
            {subject_id: {event_name: [start_frame, end_frame]}}
        """
        if self._loaded:
            return self.events
        
        if events_dir is None:
            events_dir = MATERIALS_DIR / "events" / f"task_{self.task}"
        
        if not events_dir.exists():
            print(f"[EventMarkerLoader] Warning: Event directory not found: {events_dir}")
            return {}
        
        print(f"\n[EventMarkerLoader] Loading events for task {self.task}...")
        
        self.events = {}
        json_files = list(events_dir.glob("*.json"))
        
        for json_file in json_files:
            subject_id = json_file.stem
            with open(json_file, 'r') as f:
                event_data = json.load(f)
                self.events[subject_id] = event_data
        
        self._loaded = True
        print(f"[EventMarkerLoader] ✓ Loaded events for {len(self.events)} subjects")
        
        return self.events
    
    def get_event_windows(
        self,
        subject_id: str,
        data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract event windows from full sequence.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier
        data : np.ndarray
            Full time-series data (T, C)
        
        Returns
        -------
        Dict[str, np.ndarray]
            {event_name: windowed_data}
        """
        if not self._loaded:
            raise RuntimeError("Event markers not loaded. Call load() first.")
        
        if subject_id not in self.events:
            print(f"[EventMarkerLoader] Warning: No events for subject {subject_id}")
            return {}
        
        subject_events = self.events[subject_id]
        windows = {}
        
        for event_name, (start_frame, end_frame) in subject_events.items():
            # Extract window
            window = data[start_frame:end_frame]
            windows[event_name] = window
        
        return windows
    
    def get_all_event_names(self) -> List[str]:
        """
        Get list of all unique event names across subjects.
        
        Returns
        -------
        List[str]
            Unique event names
        """
        if not self._loaded:
            self.load()
        
        all_events = set()
        for subject_events in self.events.values():
            all_events.update(subject_events.keys())
        
        return sorted(list(all_events))


class DataLoaderFactory:
    """Factory for creating data loaders."""
    
    @staticmethod
    def create_task_loader(task: int) -> DataLoader:
        """Create loader for a specific task."""
        return DataLoader(task=task)
    
    @staticmethod
    def create_metadata_loader() -> PatientMetadataLoader:
        """Create metadata loader."""
        return PatientMetadataLoader()
    
    @staticmethod
    def create_event_loader(task: int) -> EventMarkerLoader:
        """Create event marker loader."""
        return EventMarkerLoader(task=task)
    
    @staticmethod
    def load_all(task: int) -> Tuple[Dict, Dict, pd.DataFrame, Optional[Dict]]:
        """
        Load all data for a task (convenience method).
        
        Parameters
        ----------
        task : int
            Task number
        
        Returns
        -------
        patient_data : Dict
        control_data : Dict
        metadata : pd.DataFrame
        events : Dict or None
        """
        # Load task data
        task_loader = DataLoader(task=task)
        patient_data, control_data = task_loader.load()
        
        # Load metadata
        metadata_loader = PatientMetadataLoader()
        metadata = metadata_loader.load()
        
        # Try to load events (optional)
        event_loader = EventMarkerLoader(task=task)
        try:
            events = event_loader.load()
        except Exception as e:
            print(f"[DataLoaderFactory] Could not load events: {e}")
            events = None
        
        return patient_data, control_data, metadata, events


# Convenience function
def load_data(task: int) -> Tuple[Dict, Dict]:
    """
    Quick data loading function.
    
    Parameters
    ----------
    task : int
        Task number (1-6)
    
    Returns
    -------
    patient_data : Dict
    control_data : Dict
    """
    loader = DataLoader(task=task)
    return loader.load()


# Example usage
if __name__ == "__main__":
    # Test data loading
    loader = DataLoader(task=1)
    patient_data, control_data = loader.load()
    loader.print_summary()
    
    # Test metadata loading
    metadata_loader = PatientMetadataLoader()
    metadata = metadata_loader.load()
    diagnosis_groups = metadata_loader.get_diagnosis_groups()