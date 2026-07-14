"""
test_dataset.py -- Automated verification script for LibriMixDataset and get_dataloader (Person A).
Loads MiniLibriMix (2-speaker debugging data) and verifies data shapes and batching.
"""

import os
import sys
import torch

sys.path.insert(0, "src")
from dataset import LibriMixDataset, get_dataloader

CSV_PATH = "data/MiniLibriMix/metadata/mixture_train_mix_clean.csv"
DATA_ROOT = "data"
N_SRC = 2
SAMPLE_RATE = 8000

def test_fixed_length_item():
    print("\n--- Test 1: Load Fixed-Length Item (segment=2.0s) ---")
    dataset = LibriMixDataset(
        csv_path=CSV_PATH,
        data_root=DATA_ROOT,
        n_src=N_SRC,
        sample_rate=SAMPLE_RATE,
        segment=2.0,
        train=True
    )
    mix, sources = dataset[0]
    print(f"Dataset length: {len(dataset)}")
    print(f"Mixture shape: {mix.shape} (Expected: {int(2.0 * SAMPLE_RATE)})")
    print(f"Sources shape: {sources.shape} (Expected: ({N_SRC}, {int(2.0 * SAMPLE_RATE)}))")
    
    assert mix.shape == (16000,), f"Incorrect mixture shape: {mix.shape}"
    assert sources.shape == (N_SRC, 16000), f"Incorrect sources shape: {sources.shape}"
    print("Test 1 Passed!")

def test_variable_length_item():
    print("\n--- Test 2: Load Variable-Length Item (segment=None) ---")
    dataset = LibriMixDataset(
        csv_path=CSV_PATH,
        data_root=DATA_ROOT,
        n_src=N_SRC,
        sample_rate=SAMPLE_RATE,
        segment=None,
        train=False
    )
    mix, sources = dataset[0]
    print(f"Mixture shape: {mix.shape}")
    print(f"Sources shape: {sources.shape}")
    
    assert mix.ndim == 1, "Mixture should be 1D"
    assert sources.shape[0] == N_SRC, f"Expected {N_SRC} sources, got {sources.shape[0]}"
    assert sources.shape[1] == mix.shape[0], "Sources and mixture lengths must match"
    print("Test 2 Passed!")

def test_fixed_length_dataloader():
    print("\n--- Test 3: Fixed-Length DataLoader (segment=2.0s, batch_size=4) ---")
    loader = get_dataloader(
        csv_path=CSV_PATH,
        data_root=DATA_ROOT,
        n_src=N_SRC,
        sample_rate=SAMPLE_RATE,
        segment=2.0,
        train=True,
        batch_size=4,
        shuffle=True
    )
    
    # Grab one batch
    mix_batch, sources_batch = next(iter(loader))
    print(f"Mixture batch shape: {mix_batch.shape} (Expected: (4, 16000))")
    print(f"Sources batch shape: {sources_batch.shape} (Expected: (4, {N_SRC}, 16000))")
    
    assert mix_batch.shape == (4, 16000), f"Incorrect mixture batch shape: {mix_batch.shape}"
    assert sources_batch.shape == (4, N_SRC, 16000), f"Incorrect sources batch shape: {sources_batch.shape}"
    print("Test 3 Passed!")

def test_variable_length_dataloader():
    print("\n--- Test 4: Variable-Length DataLoader with Padding (segment=None, batch_size=4) ---")
    loader = get_dataloader(
        csv_path=CSV_PATH,
        data_root=DATA_ROOT,
        n_src=N_SRC,
        sample_rate=SAMPLE_RATE,
        segment=None,
        train=False,
        batch_size=4,
        shuffle=False
    )
    
    # Grab one batch
    mix_batch, sources_batch = next(iter(loader))
    print(f"Padded Mixture batch shape: {mix_batch.shape}")
    print(f"Padded Sources batch shape: {sources_batch.shape}")
    
    assert mix_batch.ndim == 2, "Mixture batch should be 2D [batch, time]"
    assert sources_batch.ndim == 3, "Sources batch should be 3D [batch, n_src, time]"
    assert mix_batch.shape[0] == 4, "Incorrect batch size"
    assert sources_batch.shape[0] == 4, "Incorrect batch size"
    assert sources_batch.shape[1] == N_SRC, "Incorrect source count"
    assert sources_batch.shape[2] == mix_batch.shape[1], "Source and mixture lengths must match in batch"
    print("Test 4 Passed!")

if __name__ == "__main__":
    try:
        test_fixed_length_item()
        test_variable_length_item()
        test_fixed_length_dataloader()
        test_variable_length_dataloader()
        print("\nAll Tests Passed Successfully!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
