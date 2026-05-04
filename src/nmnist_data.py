"""N-MNIST data loading utilities for NR-NPU.

Handles loading pre-processed N-MNIST data from per-sample .npz files.
"""

import numpy as np
from pathlib import Path


class NMNISTDataset:
    """Lazy-loading N-MNIST dataset from pre-processed .npz files."""

    def __init__(self, data_dir, split="test", max_samples=0):
        self.data_dir = Path(data_dir) / f"nmnist_{split}"
        self.labels = np.load(str(Path(data_dir) / f"nmnist_{split}_labels.npy"))
        if max_samples > 0 and max_samples < len(self.labels):
            self.labels = self.labels[:max_samples]
        self.n = len(self.labels)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        frames = np.load(str(self.data_dir / f"{idx:05d}.npz"))["frames"]
        return frames, self.labels[idx]

    def get_labels(self):
        return self.labels.copy()


def load_nmnist_sample(data_dir, split, idx):
    """Load a single N-MNIST sample."""
    frames = np.load(str(Path(data_dir) / f"nmnist_{split}" / f"{idx:05d}.npz"))["frames"]
    return frames