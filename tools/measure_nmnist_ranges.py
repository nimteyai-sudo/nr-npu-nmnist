#!/usr/bin/env python3
"""Measure NPU activation ranges for SNN normalization on N-MNIST.

Computes 99th percentile of ReLU(conv_block_output) and ReLU(linear_block_output)
using training data. These values are used as output_scale in SNN inference to
normalize NPU outputs to [0, ~1] for LIF neurons.

Usage:
    python tools/measure_nmnist_ranges.py
    python tools/measure_nmnist_ranges.py --samples 500
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from pathlib import Path
from rknnlite.api import RKNNLite
from nmnist_data import NMNISTDataset

MODEL_DIR = Path(__file__).parent.parent / "models" / "nmnist"

INPUT_CHANNELS = 8    # padded from 2
CONV2_OUT = 16
IMG_SIZE = 34
FC1_IN = CONV2_OUT * IMG_SIZE * IMG_SIZE  # 18496
NUM_CLASSES = 10


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Measure N-MNIST activation ranges")
    parser.add_argument("--samples", type=int, default=0,
                        help="Max samples (0=all)")
    args = parser.parse_args()

    print("=" * 60)
    print("  N-MNIST Activation Range Measurement")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    try:
        dataset = NMNISTDataset(MODEL_DIR, split="train", max_samples=args.samples)
    except FileNotFoundError:
        print("  ERROR: Pre-processed data not found.")
        print("  Run 'python3.13 tools/preprocess_nmnist.py' first")
        return

    n = len(dataset)
    print(f"  Using {n} samples")

    # Load NPU models
    print("\nLoading NPU models...")
    rknn_conv = RKNNLite()
    rknn_conv.load_rknn(str(MODEL_DIR / "nmnist_conv_block.rknn"))
    rknn_conv.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_AUTO)

    rknn_linear = RKNNLite()
    rknn_linear.load_rknn(str(MODEL_DIR / "nmnist_linear_block.rknn"))
    rknn_linear.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_AUTO)

    # Warmup
    dummy = np.random.randn(1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE).astype(np.float32)
    rknn_conv.inference(inputs=[dummy])
    dummy_linear = np.random.randn(1, FC1_IN, 1, 1).astype(np.float32)
    rknn_linear.inference(inputs=[dummy_linear])

    # Measure activations
    conv_outputs = []
    linear_outputs = []

    indices = np.random.choice(n, min(n, 2000), replace=False) if n > 2000 else np.arange(n)

    print(f"\nMeasuring activations on {len(indices)} samples...")
    for i, idx in enumerate(indices):
        # Get conv block output
        frames, label = dataset[idx]
        frame = frames[0]  # [2, 34, 34] — use first frame
        if frame.ndim == 4:
            # Use first frame only
            frame = frame[0]  # [2, 34, 34]

        # Pad from 2→8 channels
        inp = np.zeros((1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        inp[0, :2] = frame
        out_conv = rknn_conv.inference(inputs=[inp])
        conv_out = out_conv[0].flatten()[:CONV2_OUT * IMG_SIZE * IMG_SIZE] if isinstance(out_conv, list) else out_conv.flatten()[:FC1_IN]
        conv_out_relu = np.maximum(conv_out, 0)

        # Get linear block output
        conv_out_padded = conv_out_relu.reshape(1, FC1_IN, 1, 1) if conv_out_relu.shape[0] == FC1_IN else np.pad(conv_out_relu, (0, FC1_IN - conv_out_relu.shape[0])).reshape(1, FC1_IN, 1, 1)
        out_linear = rknn_linear.inference(inputs=[conv_out_padded.astype(np.float32)])
        linear_out = out_linear[0].flatten()[:NUM_CLASSES] if isinstance(out_linear, list) else out_linear.flatten()[:NUM_CLASSES]
        linear_out_relu = np.maximum(linear_out, 0)

        conv_outputs.append(conv_out_relu)
        linear_outputs.append(linear_out_relu)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{n} samples")

    # Compute percentiles
    conv_outputs = np.array(conv_outputs)
    linear_outputs = np.array(linear_outputs)

    max_act_conv = float(np.percentile(conv_outputs, 99))
    max_act_linear = float(np.percentile(linear_outputs, 99))

    print(f"\n  Conv block (L1) output stats:")
    print(f"    Max: {conv_outputs.max():.4f}")
    print(f"    Mean: {conv_outputs.mean():.4f}")
    print(f"    99th percentile: {max_act_conv:.4f}")

    print(f"\n  Linear block (L2) output stats:")
    print(f"    Max: {linear_outputs.max():.4f}")
    print(f"    Mean: {linear_outputs.mean():.4f}")
    print(f"    99th percentile: {max_act_linear:.4f}")

    # Save
    np.save(str(MODEL_DIR / "nmnist_max_act_conv.npy"), np.array([max_act_conv]))
    np.save(str(MODEL_DIR / "nmnist_max_act_linear.npy"), np.array([max_act_linear]))
    print(f"\n  Saved nmnist_max_act_conv.npy ({max_act_conv:.4f})")
    print(f"  Saved nmnist_max_act_linear.npy ({max_act_linear:.4f})")

    rknn_conv.release()
    rknn_linear.release()


if __name__ == "__main__":
    main()