"""
dataset.py -- PyTorch Dataset and DataLoader wrapper for LibriMix (Person A).

This module provides:
  1. LibriMixDataset: Loads mixtures and ground-truth sources from a LibriMix CSV manifest.
     Supports both fixed-length chunking (random/center crop) and loading full-length signals.
  2. pad_collate_fn: Collate function for batching variable-length signals by dynamic padding.
  3. get_dataloader: High-level wrapper to quickly get a train or validation DataLoader.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

class LibriMixDataset(Dataset):
    """
    PyTorch Dataset for LibriMix speech separation task.
    Loads mixtures and target source signals dynamically using soundfile.
    """
    def __init__(self, 
                 csv_path: str, 
                 data_root: str, 
                 n_src: int = 2, 
                 sample_rate: int = 8000, 
                 segment: float | None = 2.0, 
                 train: bool = True):
        """
        Args:
            csv_path: Path to the metadata CSV file (e.g. mixture_train_mix_clean.csv).
            data_root: Root directory containing the audio folders (mix_clean, s1, s2, etc.).
            n_src: Number of target speakers (sources) in the mixture.
            sample_rate: Standard sample rate (usually 8000 Hz).
            segment: Desired segment length in seconds. If None, load full length.
            train: If True, crops segments randomly. If False, crops from center.
        """
        self.csv_path = csv_path
        self.data_root = data_root
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment = segment
        self.train = train

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.segment_len = int(segment * sample_rate) if segment is not None else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mixture: [time] tensor.
            sources: [n_src, time] tensor containing the separate target sources.
        """
        row = self.df.iloc[idx]
        
        mix_path = os.path.join(self.data_root, row["mixture_path"])
        src_paths = [os.path.join(self.data_root, row[f"source_{k}_path"]) for k in range(1, self.n_src + 1)]

        # Determine segment frames and start position
        if self.segment_len is not None:
            info = sf.info(mix_path)
            file_len = info.frames
            
            if file_len > self.segment_len:
                if self.train:
                    # Random crop during training
                    start = np.random.randint(0, file_len - self.segment_len + 1)
                else:
                    # Center crop for validation
                    start = (file_len - self.segment_len) // 2
                frames = self.segment_len
            else:
                start = 0
                frames = file_len
            
            # Read segment directly from disk
            mix, _ = sf.read(mix_path, start=start, frames=frames, dtype="float32")
            srcs = []
            for src_path in src_paths:
                src, _ = sf.read(src_path, start=start, frames=frames, dtype="float32")
                srcs.append(src)
            
            # Pad if file is shorter than segment length
            if len(mix) < self.segment_len:
                pad_len = self.segment_len - len(mix)
                mix = np.pad(mix, (0, pad_len))
                srcs = [np.pad(s, (0, pad_len)) for s in srcs]
        else:
            # Read full files (variable lengths)
            mix, _ = sf.read(mix_path, dtype="float32")
            srcs = []
            for src_path in src_paths:
                src, _ = sf.read(src_path, dtype="float32")
                srcs.append(src)

        # Convert to Torch Tensors
        mix_tensor = torch.from_numpy(mix)
        srcs_tensor = torch.stack([torch.from_numpy(s) for s in srcs]) # Shape: [n_src, time]
        
        return mix_tensor, srcs_tensor


def pad_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to batch variable-length audio signals together.
    Pads all waveforms in the batch to the maximum length in this batch.
    
    Args:
        batch: List of tuples (mixture, sources)
    Returns:
        padded_mixtures: [batch_size, max_time] tensor.
        padded_sources: [batch_size, n_src, max_time] tensor.
    """
    mixtures, sources = zip(*batch)
    lengths = [mix.shape[0] for mix in mixtures]
    max_len = max(lengths)
    
    padded_mixtures = []
    padded_sources = []
    
    for mix, src in zip(mixtures, sources):
        pad_len = max_len - mix.shape[0]
        if pad_len > 0:
            padded_mixtures.append(torch.nn.functional.pad(mix, (0, pad_len)))
            padded_sources.append(torch.nn.functional.pad(src, (0, pad_len)))
        else:
            padded_mixtures.append(mix)
            padded_sources.append(src)
            
    return torch.stack(padded_mixtures), torch.stack(padded_sources)


def get_dataloader(csv_path: str, 
                   data_root: str, 
                   n_src: int = 2, 
                   sample_rate: int = 8000, 
                   segment: float | None = 2.0, 
                   train: bool = True, 
                   batch_size: int = 4, 
                   shuffle: bool = True, 
                   num_workers: int = 0) -> DataLoader:
    """
    Creates a PyTorch DataLoader for LibriMix.
    
    Args:
        csv_path: Path to the LibriMix metadata CSV file.
        data_root: Root directory of audio files.
        n_src: Number of speaker sources (2 or 3).
        sample_rate: Target sample rate (default: 8000 Hz).
        segment: Crop length in seconds (None for variable full length).
        train: Whether random cropping is active.
        batch_size: DataLoader batch size.
        shuffle: Whether to shuffle elements.
        num_workers: Number of DataLoader subprocesses.
    """
    dataset = LibriMixDataset(
        csv_path=csv_path,
        data_root=data_root,
        n_src=n_src,
        sample_rate=sample_rate,
        segment=segment,
        train=train
    )
    
    # Use pad_collate_fn if segment is None (variable length audio)
    collate_fn = pad_collate_fn if segment is None else None
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return loader
