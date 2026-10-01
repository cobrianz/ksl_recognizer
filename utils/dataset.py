"""
Dataset utilities for KSL landmark sequences.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import yaml


class KSLLandmarkDataset(Dataset):
    """
    Dataset of pre-extracted landmark sequences.
    Expects structure:
        data/landmarks/
            hello/
                sample_001.npy   # shape (seq_len, 1662)
                sample_002.npy
            thank_you/
                ...
    """

    def __init__(
        self,
        root_dir: str,
        signs: List[str],
        sequence_length: int = 30,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.signs = signs
        self.sequence_length = sequence_length
        self.transform = transform
        self.sign_to_idx = {s: i for i, s in enumerate(signs)}

        self.samples: List[Tuple[Path, int]] = []
        self._load_file_list()

    def _load_file_list(self):
        for sign in self.signs:
            sign_dir = self.root_dir / sign
            if not sign_dir.exists():
                continue
            for f in sorted(sign_dir.glob("*.npy")):
                self.samples.append((f, self.sign_to_idx[sign]))

        print(f"Loaded {len(self.samples)} samples from {len(self.signs)} signs.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        sequence = np.load(path).astype(np.float32)  # (T, D)

        # Pad or truncate to fixed length
        sequence = self._pad_or_truncate(sequence)

        if self.transform:
            sequence = self.transform(sequence)

        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)

    def _pad_or_truncate(self, seq: np.ndarray) -> np.ndarray:
        T, D = seq.shape
        if T > self.sequence_length:
            # Take middle portion or last frames
            start = (T - self.sequence_length) // 2
            return seq[start : start + self.sequence_length]
        elif T < self.sequence_length:
            pad = np.zeros((self.sequence_length - T, D), dtype=np.float32)
            return np.vstack([seq, pad])
        return seq


def get_dataloaders(
    config: dict,
    batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    data_cfg = config["data"]
    train_cfg = config["training"]
    signs = config["signs"]

    full_dataset = KSLLandmarkDataset(
        root_dir=config["paths"]["landmarks"],
        signs=signs,
        sequence_length=data_cfg["sequence_length"],
    )

    if len(full_dataset) == 0:
        raise RuntimeError(
            "No landmark data found. Run data collection first:\n"
            "  python scripts/collect_data.py"
        )

    # Simple random split
    n = len(full_dataset)
    n_train = int(n * train_cfg["train_split"])
    n_val = n - n_train

    generator = torch.Generator().manual_seed(train_cfg["seed"])
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    bs = batch_size or train_cfg["batch_size"]

    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, num_workers=0, drop_last=False
    )
    val_loader = DataLoader(
        val_set, batch_size=bs, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
