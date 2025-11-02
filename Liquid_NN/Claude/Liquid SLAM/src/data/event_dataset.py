"""
src/data/event_dataset.py

Dataset-agnostic event camera data loaders.
Supports multiple event formats and datasets:
- DVS/DAVIS recordings
- Prophesee datasets
- Custom event formats

Event format: (x, y, t, p) where:
    x, y: spatial coordinates
    t: timestamp
    p: polarity (+1 or -1)
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For MacOS compatibility

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings


class BaseEventDataset(Dataset):
    """
    Base class for event camera datasets.
    
    Provides common functionality for event data handling:
    - Event binning/accumulation
    - Temporal windowing
    - Data augmentation
    
    Args:
        data_dir: Root directory of dataset
        split: 'train', 'val', or 'test'
        sequence_length: Number of time bins per sample
        time_window: Time window for event accumulation (seconds)
        spatial_size: (Height, Width) of output
        transform: Optional transform function
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'train',
        sequence_length: int = 10,
        time_window: float = 0.05,
        spatial_size: Tuple[int, int] = (128, 128),
        transform: Optional[Callable] = None,
        normalize: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.time_window = time_window
        self.spatial_size = spatial_size
        self.transform = transform
        self.normalize = normalize
        
        # Load data index
        self.samples = self._load_index()
        
    def _load_index(self) -> List[Dict]:
        """Load dataset index. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def accumulate_events(
        self,
        events: np.ndarray,
        time_bins: int = 1,
        polarity_mode: str = 'separate'
    ) -> np.ndarray:
        """
        Accumulate events into frames.
        
        Args:
            events: Event array (N, 4) with [x, y, t, p]
            time_bins: Number of temporal bins
            polarity_mode: 'separate', 'combined', or 'positive_only'
        
        Returns:
            frames: (time_bins, C, H, W) where C depends on polarity_mode
        """
        H, W = self.spatial_size
        
        if len(events) == 0:
            channels = 2 if polarity_mode == 'separate' else 1
            return np.zeros((time_bins, channels, H, W), dtype=np.float32)
        
        # Normalize timestamps to [0, 1]
        t_min, t_max = events[:, 2].min(), events[:, 2].max()
        if t_max > t_min:
            t_norm = (events[:, 2] - t_min) / (t_max - t_min)
        else:
            t_norm = np.zeros_like(events[:, 2])
        
        # Assign events to time bins
        bin_indices = np.clip(
            (t_norm * time_bins).astype(np.int32),
            0,
            time_bins - 1
        )
        
        # Accumulate
        if polarity_mode == 'separate':
            frames = np.zeros((time_bins, 2, H, W), dtype=np.float32)
            
            for i in range(time_bins):
                mask = bin_indices == i
                if mask.sum() == 0:
                    continue
                
                bin_events = events[mask]
                x = np.clip(bin_events[:, 0].astype(int), 0, W - 1)
                y = np.clip(bin_events[:, 1].astype(int), 0, H - 1)
                p = bin_events[:, 3]
                
                # Positive polarity
                pos_mask = p > 0
                np.add.at(frames[i, 0], (y[pos_mask], x[pos_mask]), 1)
                
                # Negative polarity
                neg_mask = p < 0
                np.add.at(frames[i, 1], (y[neg_mask], x[neg_mask]), 1)
        
        elif polarity_mode == 'combined':
            frames = np.zeros((time_bins, 1, H, W), dtype=np.float32)
            
            for i in range(time_bins):
                mask = bin_indices == i
                if mask.sum() == 0:
                    continue
                
                bin_events = events[mask]
                x = np.clip(bin_events[:, 0].astype(int), 0, W - 1)
                y = np.clip(bin_events[:, 1].astype(int), 0, H - 1)
                p = bin_events[:, 3]
                
                np.add.at(frames[i, 0], (y, x), p)
        
        else:  # positive_only
            frames = np.zeros((time_bins, 1, H, W), dtype=np.float32)
            
            for i in range(time_bins):
                mask = (bin_indices == i) & (events[:, 3] > 0)
                if mask.sum() == 0:
                    continue
                
                bin_events = events[mask]
                x = np.clip(bin_events[:, 0].astype(int), 0, W - 1)
                y = np.clip(bin_events[:, 1].astype(int), 0, H - 1)
                
                np.add.at(frames[i, 0], (y, x), 1)
        
        # Normalize
        if self.normalize:
            for i in range(time_bins):
                for c in range(frames.shape[1]):
                    if frames[i, c].max() > 0:
                        frames[i, c] /= frames[i, c].max()
        
        return frames
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample. Must be implemented by subclasses."""
        raise NotImplementedError


class HDF5EventDataset(BaseEventDataset):
    """
    Event dataset stored in HDF5 format.
    
    Expected HDF5 structure:
        /events/[sample_id]/
            - data: (N, 4) array of events
            - labels: optional labels
            - metadata: optional metadata
    
    Args:
        hdf5_path: Path to HDF5 file
        **kwargs: Arguments passed to BaseEventDataset
    """
    
    def __init__(
        self,
        hdf5_path: Union[str, Path],
        **kwargs
    ):
        self.hdf5_path = Path(hdf5_path)
        super().__init__(data_dir=self.hdf5_path.parent, **kwargs)
    
    def _load_index(self) -> List[Dict]:
        """Load sample indices from HDF5."""
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")
        
        samples = []
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'events' not in f:
                raise ValueError("HDF5 file must contain 'events' group")
            
            split_group = f['events'].get(self.split, f['events'])
            
            for sample_id in split_group.keys():
                samples.append({
                    'id': sample_id,
                    'path': self.hdf5_path,
                    'group': f'events/{self.split}/{sample_id}'
                })
        
        return samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample from HDF5.
        
        Returns:
            Dictionary with:
                - 'events': (seq_len, C, H, W) tensor
                - 'labels': optional labels
                - 'timestamps': time stamps for each frame
        """
        sample_info = self.samples[idx]
        
        with h5py.File(sample_info['path'], 'r') as f:
            group = f[sample_info['group']]
            
            # Load events
            events = group['data'][:]  # (N, 4) array
            
            # Accumulate into frames
            frames = self.accumulate_events(
                events,
                time_bins=self.sequence_length,
                polarity_mode='separate'
            )
            
            # Convert to tensor
            frames = torch.from_numpy(frames).float()
            
            # Get labels if available
            labels = None
            if 'labels' in group:
                labels = torch.from_numpy(group['labels'][:])
            
            # Get timestamps
            if len(events) > 0:
                t_min, t_max = events[:, 2].min(), events[:, 2].max()
                timestamps = torch.linspace(t_min, t_max, self.sequence_length)
            else:
                timestamps = torch.arange(self.sequence_length).float()
            
            result = {
                'events': frames,
                'timestamps': timestamps,
                'sample_id': sample_info['id']
            }
            
            if labels is not None:
                result['labels'] = labels
            
            if self.transform:
                result = self.transform(result)
            
            return result


class DVSGestureDataset(BaseEventDataset):
    """
    DVS Gesture dataset loader.
    
    DVS Gesture is a common benchmark for event-based gesture recognition.
    Format: .aedat or preprocessed .npy files
    
    Args:
        data_dir: Root directory containing DVS Gesture data
        **kwargs: Arguments passed to BaseEventDataset
    """
    
    def __init__(self, data_dir: Union[str, Path], **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)
        self.num_classes = 11  # DVS Gesture has 11 classes
    
    def _load_index(self) -> List[Dict]:
        """Load DVS Gesture sample index."""
        samples = []
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            warnings.warn(f"Split directory not found: {split_dir}")
            return samples
        
        # Find all .npy or .aedat files
        for file_path in split_dir.glob('**/*.npy'):
            # Parse class from filename or directory structure
            class_name = file_path.parent.name
            
            samples.append({
                'path': file_path,
                'class': class_name,
                'id': file_path.stem
            })
        
        return samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get DVS Gesture sample."""
        sample_info = self.samples[idx]
        
        # Load events
        events = np.load(sample_info['path'])
        
        # Accumulate
        frames = self.accumulate_events(
            events,
            time_bins=self.sequence_length,
            polarity_mode='separate'
        )
        
        frames = torch.from_numpy(frames).float()
        
        result = {
            'events': frames,
            'label': torch.tensor(int(sample_info.get('class', 0))),
            'sample_id': sample_info['id']
        }
        
        if self.transform:
            result = self.transform(result)
        
        return result


class SyntheticEventDataset(BaseEventDataset):
    """
    Synthetic event dataset for testing and debugging.
    
    Generates synthetic event streams with known patterns.
    Useful for:
        - Testing data pipeline
        - Debugging models
        - Prototyping
    
    Args:
        num_samples: Number of samples to generate
        pattern: 'moving_edge', 'rotation', 'random'
        **kwargs: Arguments passed to BaseEventDataset
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        pattern: str = 'moving_edge',
        **kwargs
    ):
        self.num_samples = num_samples
        self.pattern = pattern
        super().__init__(data_dir=Path('.'), **kwargs)
    
    def _load_index(self) -> List[Dict]:
        """Generate synthetic sample index."""
        return [{'id': i, 'pattern': self.pattern} for i in range(self.num_samples)]
    
    def _generate_moving_edge(self, H: int, W: int, num_events: int = 5000) -> np.ndarray:
        """Generate events from a moving edge."""
        events = []
        
        # Edge moves from left to right
        for i in range(num_events):
            t = i / num_events
            x = int(t * W)
            
            # Generate events along vertical edge
            num_edge_events = 10
            for _ in range(num_edge_events):
                y = np.random.randint(0, H)
                p = 1 if np.random.rand() > 0.5 else -1
                events.append([x, y, t, p])
        
        return np.array(events)
    
    def _generate_rotation(self, H: int, W: int, num_events: int = 5000) -> np.ndarray:
        """Generate events from rotating pattern."""
        events = []
        cx, cy = W // 2, H // 2
        radius = min(W, H) // 3
        
        for i in range(num_events):
            t = i / num_events
            angle = t * 2 * np.pi
            
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))
            
            if 0 <= x < W and 0 <= y < H:
                p = 1 if np.random.rand() > 0.5 else -1
                events.append([x, y, t, p])
        
        return np.array(events)
    
    def _generate_random(self, H: int, W: int, num_events: int = 5000) -> np.ndarray:
        """Generate random events."""
        x = np.random.randint(0, W, num_events)
        y = np.random.randint(0, H, num_events)
        t = np.sort(np.random.rand(num_events))
        p = np.random.choice([-1, 1], num_events)
        
        return np.column_stack([x, y, t, p])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate synthetic sample."""
        H, W = self.spatial_size
        
        # Generate events based on pattern
        if self.pattern == 'moving_edge':
            events = self._generate_moving_edge(H, W)
        elif self.pattern == 'rotation':
            events = self._generate_rotation(H, W)
        else:
            events = self._generate_random(H, W)
        
        # Accumulate
        frames = self.accumulate_events(
            events,
            time_bins=self.sequence_length,
            polarity_mode='separate'
        )
        
        frames = torch.from_numpy(frames).float()
        
        return {
            'events': frames,
            'label': torch.tensor(idx % 10),  # Dummy labels
            'sample_id': str(idx)
        }


def create_event_dataloader(
    dataset_type: str,
    dataset_config: Dict,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Factory function to create event dataloaders.
    
    Args:
        dataset_type: 'hdf5', 'dvs_gesture', or 'synthetic'
        dataset_config: Configuration dict for dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
    
    Returns:
        DataLoader instance
    """
    datasets = {
        'hdf5': HDF5EventDataset,
        'dvs_gesture': DVSGestureDataset,
        'synthetic': SyntheticEventDataset
    }
    
    if dataset_type not in datasets:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    dataset = datasets[dataset_type](**dataset_config)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )


if __name__ == "__main__":
    print("Testing Event Dataset Loaders")
    print("=" * 60)
    
    # Test synthetic dataset
    print("\n1. Testing SyntheticEventDataset")
    dataset = SyntheticEventDataset(
        num_samples=100,
        pattern='moving_edge',
        sequence_length=10,
        spatial_size=(64, 64)
    )
    
    sample = dataset[0]
    print(f"   Events shape: {sample['events'].shape}")
    print(f"   Label: {sample['label']}")
    print(f"   Sample ID: {sample['sample_id']}")
    
    # Test dataloader
    print("\n2. Testing DataLoader")
    dataloader = create_event_dataloader(
        'synthetic',
        {'num_samples': 100, 'sequence_length': 10, 'spatial_size': (64, 64)},
        batch_size=8,
        num_workers=0  # Use 0 for testing
    )
    
    batch = next(iter(dataloader))
    print(f"   Batch events shape: {batch['events'].shape}")
    print(f"   Batch labels shape: {batch['label'].shape}")
    
    print("\n✓ All dataset tests passed!")