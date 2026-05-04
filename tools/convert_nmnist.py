#!/usr/bin/env python3
"""Convert N-MNIST ONNX models to RKNN for RK3588 NPU.

Takes ONNX models from train_nmnist.py and converts them to RKNN format
with INT8 quantization. Also computes calibration data from training set.

Two RKNN models:
  1. Conv block: Conv1(8→8, 3×3) + ReLU + Conv2(8→16, 3×3) + ReLU
     Input: [1, 8, 34, 34] (2 channels padded to 8)
  2. Linear block: FC1(18496→128) + ReLU + FC2(128→16, padded from 10)
     Input: [1, 18496, 1, 1] (spike-encoded conv features)

Usage:
    python tools/convert_nmnist.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path

from ann_to_snn import linear_to_conv2d

MODEL_DIR = Path(__file__).parent.parent / "models" / "nmnist"

INPUT_CHANNELS = 8       # padded from 2
CONV1_OUT = 8
CONV2_OUT = 16
IMG_SIZE = 34
FC1_IN = CONV2_OUT * IMG_SIZE * IMG_SIZE  # 16*34*34 = 18496
FC1_OUT = 128
NUM_CLASSES = 10
NUM_CLASSES_PADDED = 16  # padded to mult of 8


class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, CONV1_OUT, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(CONV1_OUT, CONV2_OUT, 3, padding=1, bias=False)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x


class LinearBlockConv(nn.Module):
    def __init__(self, fc1_state, fc2_state):
        super().__init__()
        self.fc1 = nn.Linear(FC1_IN, FC1_OUT, bias=False)
        self.fc2 = nn.Linear(FC1_OUT, NUM_CLASSES, bias=False)
        self.fc1.load_state_dict(fc1_state)
        self.fc2.load_state_dict(fc2_state)
        self.conv1 = linear_to_conv2d(self.fc1)
        self.conv2 = linear_to_conv2d(self.fc2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x


def load_nmnist_numpy(data_dir, split="train", max_samples=0):
    """Load pre-processed N-MNIST data from numpy files."""
    prefix = f"nmnist_{split}"
    frames = np.load(str(data_dir / f"{prefix}_frames.npy"), mmap_mode='r')
    labels = np.load(str(data_dir / f"{prefix}_labels.npy"))
    if max_samples > 0 and max_samples < len(labels):
        indices = np.random.choice(len(labels), max_samples, replace=False)
        frames = frames[indices]
        labels = labels[indices]
    return frames, labels


def generate_calibration_data(conv_block, linear_block_conv, n_samples=200):
    """Generate calibration data for RKNN INT8 quantization."""
    from rknn.api import RKNN

    print("Generating calibration data...")
    calib_dir = tempfile.mkdtemp(prefix="nmnist_calib_")

    # Conv block calibration: random input in [0, 1]
    # N-MNIST frames are binary (0/1) or small float values
    conv_inputs = []
    for i in range(n_samples):
        inp = np.random.rand(1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE).astype(np.float32)
        # Only first 2 channels have actual data, rest are zeros
        inp[:, 2:, :, :] = 0
        np.save(f"{calib_dir}/conv_{i}.npy", inp)
        conv_inputs.append(inp)

    # Linear block calibration: use conv block outputs + random spikes
    conv_block.eval()
    with torch.no_grad():
        for i in range(n_samples):
            inp = torch.from_numpy(conv_inputs[i])
            out = conv_block(inp).numpy()
            # ReLU output, positive values
            # For SNN calibration, also generate binary spike versions
            relu_out = np.maximum(out, 0)
            # Flatten for linear block: [1, 18496, 1, 1]
            flat = relu_out.flatten()[:FC1_IN]
            if flat.shape[0] < FC1_IN:
                flat = np.pad(flat, (0, FC1_IN - flat.shape[0]))
            linear_in = flat.reshape(1, FC1_IN, 1, 1).astype(np.float32)
            np.save(f"{calib_dir}/linear_relu_{i}.npy", linear_in)

            # Binary spike version (for SNN calibration)
            spike_in = (flat > 0.5).astype(np.float32)
            spike_in = spike_in.reshape(1, FC1_IN, 1, 1)
            np.save(f"{calib_dir}/linear_spike_{i}.npy", spike_in)

    # Write calibration lists
    with open(f"{calib_dir}/conv.txt", "w") as f:
        for i in range(n_samples):
            f.write(f"{calib_dir}/conv_{i}.npy\n")
    with open(f"{calib_dir}/linear_relu.txt", "w") as f:
        for i in range(n_samples):
            f.write(f"{calib_dir}/linear_relu_{i}.npy\n")
    with open(f"{calib_dir}/linear_spike.txt", "w") as f:
        for i in range(n_samples):
            f.write(f"{calib_dir}/linear_spike_{i}.npy\n")

    return calib_dir


def build_rknn(onnx_path, rknn_path, calib_txt, input_channels):
    """Build RKNN model from ONNX with INT8 quantization."""
    from rknn.api import RKNN

    rknn = RKNN()
    rknn.config(
        mean_values=[[0] * input_channels],
        std_values=[[1] * input_channels],
        target_platform="rk3588",
    )
    rknn.load_onnx(model=onnx_path)
    rknn.build(do_quantization=True, dataset=calib_txt)
    rknn.export_rknn(rknn_path)
    rknn.release()
    print(f"  Built: {rknn_path}")


def main():
    print("=" * 60)
    print("  N-MNIST ONNX → RKNN Conversion")
    print("=" * 60)

    # Load trained SNN weights
    ckpt_path = MODEL_DIR / "nmnist_snn.pt"
    if not ckpt_path.exists():
        print(f"\n  ERROR: {ckpt_path} not found.")
        print(f"  Run 'python tools/train_nmnist.py' first (on GPU machine)")
        return

    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # Build PyTorch sub-models
    print("\nBuilding NPU sub-models...")

    # Conv block
    conv_block = ConvBlock()
    conv_block.conv1.load_state_dict(ckpt['conv1'])
    conv_block.conv2.load_state_dict(ckpt['conv2'])
    conv_block.eval()

    # Linear block
    linear_block = LinearBlockConv(ckpt['fc1'], ckpt['fc2'])
    linear_block.eval()

    # Export to ONNX
    print("Exporting ONNX models...")
    onnx_conv = str(MODEL_DIR / "nmnist_conv_block.onnx")
    onnx_linear = str(MODEL_DIR / "nmnist_linear_block.onnx")

    # Conv block: input [1, 8, 34, 34]
    dummy_conv = torch.randn(1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        conv_block, dummy_conv, onnx_conv,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"  Exported: {onnx_conv}")

    # Linear block: input [1, 18496, 1, 1]
    dummy_linear = torch.randn(1, FC1_IN, 1, 1)
    torch.onnx.export(
        linear_block, dummy_linear, onnx_linear,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"  Exported: {onnx_linear}")

    # Generate calibration data and build RKNN
    calib_dir = generate_calibration_data(conv_block, linear_block, n_samples=200)

    print("\nBuilding RKNN models...")
    # Conv block RKNN - calibrate with random input (ReLU-like distribution)
    build_rknn(
        onnx_conv,
        str(MODEL_DIR / "nmnist_conv_block.rknn"),
        f"{calib_dir}/conv.txt",
        INPUT_CHANNELS,
    )

    # Linear block RKNN - calibrate with ReLU outputs (for ANN mode)
    # Note: SNN mode sends binary spikes, but we calibrate with ReLU outputs
    # because ANN mode also needs to work
    build_rknn(
        onnx_linear,
        str(MODEL_DIR / "nmnist_linear_block.rknn"),
        f"{calib_dir}/linear_relu.txt",
        FC1_IN,
    )

    shutil.rmtree(calib_dir)
    print(f"\n  Done! RKNN models saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()