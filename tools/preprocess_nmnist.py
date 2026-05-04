#!/usr/bin/env python3
"""Preprocess N-MNIST dataset: download and convert to numpy arrays.

Uses tonic to download N-MNIST and convert DVS events to frames.
Saves individual .npy files per sample to avoid OOM with 60k samples.

Must be run with Python 3.13+ (which has lzma module for tonic).

Usage:
    python3 tools/preprocess_nmnist.py
    python3 tools/preprocess_nmnist.py --time-window 3000
    python3 tools/preprocess_nmnist.py --max-samples 1000
"""
import numpy as np
import argparse
import os
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models" / "nmnist"


def main():
    parser = argparse.ArgumentParser(description="Preprocess N-MNIST dataset")
    parser.add_argument("--time-window", type=int, default=3000,
                        help="Time window in microseconds for frame conversion")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per split (0=all)")
    args = parser.parse_args()

    import tonic
    import tonic.transforms as transforms

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    sensor_size = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)
    print(f"N-MNIST sensor_size: {sensor_size}")

    frame_transform = transforms.ToFrame(
        sensor_size=sensor_size,
        time_window=args.time_window,
    )

    for split in ["train", "test"]:
        print(f"\nProcessing {split} set...")
        dataset = tonic.datasets.NMNIST(
            save_to=str(MODEL_DIR),
            train=(split == "train"),
            transform=frame_transform,
        )

        n = len(dataset) if args.max_samples == 0 else min(args.max_samples, len(dataset))
        print(f"  Total samples: {len(dataset)}, using: {n}")

        # Save per-sample files in a subdirectory
        frames_dir = MODEL_DIR / f"nmnist_{split}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        labels = []

        for i in range(n):
            frames, label = dataset[i]
            if not isinstance(frames, np.ndarray):
                frames = np.array(frames)
            labels.append(label)

            # Save individual frame as compressed npz
            np.savez_compressed(str(frames_dir / f"{i:05d}.npz"), frames=frames)

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{n}")
                # Force GC to free memory
                import gc
                gc.collect()

        # Save labels array
        labels_array = np.array(labels, dtype=np.int64)
        labels_path = str(MODEL_DIR / f"nmnist_{split}_labels.npy")
        np.save(labels_path, labels_array)
        print(f"  Labels: {labels_path}")
        print(f"  Saved {n} frame files to {frames_dir}/")

        # Save metadata
        meta = {
            'n_samples': n,
            'time_window': args.time_window,
            'sensor_size': sensor_size,
        }
        np.save(str(MODEL_DIR / f"nmnist_{split}_meta.npy"), meta)

        # Free memory
        import gc
        gc.collect()

    print("\nDone! Run 'python tools/train_nmnist.py' to train the model.")


if __name__ == "__main__":
    main()