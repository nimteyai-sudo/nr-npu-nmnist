#!/usr/bin/env python3
"""Convert N-MNIST SNN to 4 separate RKNN models for RK3588 NPU.

Each layer is a separate NPU call, with LIF neurons on CPU between them:

  Frame → conv1 (NPU) → LIF1 → conv2 (NPU) → LIF2 →
         flatten → fc1 (NPU) → LIF3 → fc2 (NPU) → LIF_out → argmax

This preserves the SNN dynamics (LIF between ALL layers) unlike the
2-block approach (conv_block + linear_block) which loses inter-layer LIF.

Architecture:
  conv1: Conv2d(8→8, 3×3, pad=1)   Input: [1, 8, 34, 34]  (2ch padded to 8)
  conv2: Conv2d(8→16, 3×3, pad=1)  Input: [1, 8, 34, 34]  (LIF1 spikes)
  fc1:   Conv2d(18496→128, 1×1)    Input: [1, 18496, 1, 1] (LIF2 spikes flattened)
  fc2:   Conv2d(128→16, 1×1)       Input: [1, 128, 1, 1]   (LIF3 spikes)

Usage:
    python tools/convert_nmnist_4layer.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models" / "nmnist"

# Architecture constants
INPUT_CHANNELS = 8       # padded from 2
CONV1_OUT = 8
CONV2_OUT = 16
IMG_SIZE = 34
FC1_IN = CONV2_OUT * IMG_SIZE * IMG_SIZE  # 18496
FC1_OUT = 128
NUM_CLASSES = 10
NUM_CLASSES_PADDED = 16  # padded to mult of 8


class Conv1Layer(nn.Module):
    """Conv1: 8→8 with padded weights from 2-channel trained model."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(INPUT_CHANNELS, CONV1_OUT, 3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Conv2Layer(nn.Module):
    """Conv2: 8→16."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(CONV1_OUT, CONV2_OUT, 3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class FC1Layer(nn.Module):
    """FC1 as Conv2d 1×1: 18496→128."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(FC1_IN, FC1_OUT, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


class FC2Layer(nn.Module):
    """FC2 as Conv2d 1×1: 128→16 (padded from 10)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(FC1_OUT, NUM_CLASSES_PADDED, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


def load_weights(ckpt):
    """Load trained weights into 4-layer models with proper padding."""
    # Conv1: original [8, 2, 3, 3] → pad to [8, 8, 3, 3]
    conv1 = Conv1Layer()
    with torch.no_grad():
        w = torch.zeros(CONV1_OUT, INPUT_CHANNELS, 3, 3)
        w[:, :2, :, :] = ckpt['conv1']['weight']
        conv1.conv.weight.copy_(w)
    conv1.eval()

    # Conv2: same shape [16, 8, 3, 3]
    conv2 = Conv2Layer()
    with torch.no_grad():
        conv2.conv.weight.copy_(ckpt['conv2']['weight'])
    conv2.eval()

    # FC1: [128, 18496] → Conv2d(18496, 128, 1, 1)
    fc1 = FC1Layer()
    with torch.no_grad():
        fc1.conv.weight[:, :, 0, 0] = ckpt['fc1']['weight']
    fc1.eval()

    # FC2: [10, 128] → Conv2d(128, 16, 1, 1) with zero padding for channels 10-15
    fc2 = FC2Layer()
    with torch.no_grad():
        fc2.conv.weight[:] = 0.0
        fc2.conv.weight[:NUM_CLASSES, :, 0, 0] = ckpt['fc2']['weight']
    fc2.eval()

    return conv1, conv2, fc1, fc2


def export_onnx(models, save_dir):
    """Export 4 ONNX models."""
    conv1, conv2, fc1, fc2 = models

    print("Exporting ONNX models...")

    # conv1: [1, 8, 34, 34] → [1, 8, 34, 34]
    torch.onnx.export(conv1, torch.randn(1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE),
                       str(save_dir / "nmnist_conv1.onnx"), opset_version=11,
                       input_names=["input"], output_names=["output"])
    print(f"  conv1: {save_dir / 'nmnist_conv1.onnx'}")

    # conv2: [1, 8, 34, 34] → [1, 16, 34, 34]
    torch.onnx.export(conv2, torch.randn(1, CONV1_OUT, IMG_SIZE, IMG_SIZE),
                       str(save_dir / "nmnist_conv2.onnx"), opset_version=11,
                       input_names=["input"], output_names=["output"])
    print(f"  conv2: {save_dir / 'nmnist_conv2.onnx'}")

    # fc1: [1, 18496, 1, 1] → [1, 128, 1, 1]
    torch.onnx.export(fc1, torch.randn(1, FC1_IN, 1, 1),
                       str(save_dir / "nmnist_fc1.onnx"), opset_version=11,
                       input_names=["input"], output_names=["output"])
    print(f"  fc1: {save_dir / 'nmnist_fc1.onnx'}")

    # fc2: [1, 128, 1, 1] → [1, 16, 1, 1]
    torch.onnx.export(fc2, torch.randn(1, FC1_OUT, 1, 1),
                       str(save_dir / "nmnist_fc2.onnx"), opset_version=11,
                       input_names=["input"], output_names=["output"])
    print(f"  fc2: {save_dir / 'nmnist_fc2.onnx'}")


def generate_calibration_data(conv1, conv2, fc1, fc2, n_samples=200):
    """Generate calibration data for INT8 quantization of all 4 layers."""
    print(f"\nGenerating calibration data ({n_samples} samples)...")

    calib_dir = tempfile.mkdtemp(prefix="nmnist_calib_")

    conv1_list = []
    conv2_list = []
    fc1_list = []
    fc2_list = []

    for i in range(n_samples):
        # Conv1 input: N-MNIST-like data (0/1 in first 2 channels)
        inp = np.zeros((1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        inp[0, :2] = np.random.rand(2, IMG_SIZE, IMG_SIZE).astype(np.float32)
        np.save(f"{calib_dir}/conv1_{i}.npy", inp)
        conv1_list.append(inp)

        # Propagate through conv1 to get conv2 input
        with torch.no_grad():
            c1_out = conv1(torch.from_numpy(inp)).numpy()

        # After ReLU (for ANN path) or after LIF spikes (for SNN path)
        # Calibrate with both positive continuous values and binary spikes
        c1_relu = np.maximum(c1_out, 0)
        # Binary spikes for SNN
        c1_spike = (c1_relu > 0.5).astype(np.float32)
        # Mix: use relu for calibration since NPU sees both
        np.save(f"{calib_dir}/conv2_relu_{i}.npy", c1_relu.astype(np.float32))
        np.save(f"{calib_dir}/conv2_spike_{i}.npy", c1_spike.astype(np.float32))

        # Conv2 → FC1 path (using relu for calibration)
        with torch.no_grad():
            c2_relu = conv2(torch.from_numpy(c1_relu)).numpy()
        c2_relu = np.maximum(c2_relu, 0)
        flat = c2_relu.flatten()[:FC1_IN]
        if flat.shape[0] < FC1_IN:
            flat = np.pad(flat, (0, FC1_IN - flat.shape[0]))
        fc1_in = flat.reshape(1, FC1_IN, 1, 1).astype(np.float32)
        np.save(f"{calib_dir}/fc1_relu_{i}.npy", fc1_in)

        # Binary spike version
        fc1_spike = (flat > 0.5).astype(np.float32)
        fc1_spike = fc1_spike.reshape(1, FC1_IN, 1, 1)
        np.save(f"{calib_dir}/fc1_spike_{i}.npy", fc1_spike)

        # FC1 → FC2 path
        with torch.no_grad():
            f1_out = fc1(torch.from_numpy(fc1_in)).numpy()
        f1_relu = np.maximum(f1_out, 0)

        fc2_in = f1_relu.flatten()[:FC1_OUT]
        if fc2_in.shape[0] < FC1_OUT:
            fc2_in = np.pad(fc2_in, (0, FC1_OUT - fc2_in.shape[0]))
        fc2_in = fc2_in.reshape(1, FC1_OUT, 1, 1).astype(np.float32)
        np.save(f"{calib_dir}/fc2_relu_{i}.npy", fc2_in)

        # Binary spike version
        fc2_spike = (f1_relu.flatten()[:FC1_OUT] > 0.5).astype(np.float32)
        fc2_spike = np.pad(fc2_spike, (0, FC1_OUT - fc2_spike.shape[0])).reshape(1, FC1_OUT, 1, 1)
        np.save(f"{calib_dir}/fc2_spike_{i}.npy", fc2_spike)

    # Write calibration lists (use spike data for SNN layers, mixed for conv1)
    with open(f"{calib_dir}/conv1.txt", "w") as f:
        for i in range(n_samples):
            f.write(f"{calib_dir}/conv1_{i}.npy\n")

    # conv2 sees LIF1 output (binary spikes or small positive)
    with open(f"{calib_dir}/conv2.txt", "w") as f:
        for i in range(n_samples):
            f.write(f"{calib_dir}/conv2_spike_{i}.npy\n")

    # fc1 sees LIF2 output (binary spikes)
    with open(f"{calib_dir}/fc1.txt", "w") as f:
        for i in range(n_samples):
            f.write(f"{calib_dir}/fc1_spike_{i}.npy\n")

    # fc2 sees LIF3 output (binary spikes)
    with open(f"{calib_dir}/fc2.txt", "w") as f:
        for i in range(n_samples):
            f.write(f"{calib_dir}/fc2_spike_{i}.npy\n")

    print(f"  Calibration data saved to {calib_dir}")
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
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"  ERROR: Failed to load {onnx_path}")
        rknn.release()
        return False

    ret = rknn.build(do_quantization=True, dataset=calib_txt)
    if ret != 0:
        print(f"  ERROR: Failed to build {rknn_path}")
        rknn.release()
        return False

    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"  ERROR: Failed to export {rknn_path}")
        rknn.release()
        return False

    # Print model size
    size_kb = os.path.getsize(rknn_path) / 1024
    print(f"  Built: {rknn_path} ({size_kb:.1f} KB)")

    # Verify on CPU (RKNN toolkit uses NHWC format)
    ret = rknn.init_runtime(target=None)
    if ret == 0:
        if "conv1" in rknn_path or "conv2" in rknn_path:
            # NHWC format for simulator
            test_in = np.random.randn(1, IMG_SIZE, IMG_SIZE, input_channels).astype(np.float32)
        else:
            test_in = np.random.randn(1, 1, 1, input_channels).astype(np.float32)
        try:
            out = rknn.inference(inputs=[test_in], data_format='nhwc')
            if isinstance(out, list):
                print(f"    Output shape: {out[0].shape}")
            else:
                print(f"    Output shape: {out.shape}")
        except Exception as e:
            print(f"    Verification skipped: {e}")
        rknn.release()

    return True


def main():
    print("=" * 60)
    print("  N-MNIST 4-Layer ONNX → RKNN Conversion")
    print("=" * 60)

    # Load trained SNN weights
    ckpt_path = MODEL_DIR / "nmnist_snn.pt"
    if not ckpt_path.exists():
        # Try weights file
        ckpt_path = MODEL_DIR / "nmnist_snn_weights.pt"
    if not ckpt_path.exists():
        print(f"\n  ERROR: No trained weights found in {MODEL_DIR}")
        print(f"  Run 'python tools/train_nmnist_cuda.py' first (on GPU machine)")
        return

    print(f"\nLoading weights from {ckpt_path}...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # Build 4-layer models
    print("Building 4-layer NPU models...")
    conv1, conv2, fc1, fc2 = load_weights(ckpt)

    # Quick sanity check
    with torch.no_grad():
        x = torch.randn(1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE)
        y1 = conv1(x)
        print(f"  conv1: {x.shape} → {y1.shape}")
        y2 = conv2(y1)
        print(f"  conv2: {y1.shape} → {y2.shape}")
        flat = y2.flatten(1).unsqueeze(-1).unsqueeze(-1)
        y3 = fc1(flat)
        print(f"  fc1:   {flat.shape} → {y3.shape}")
        y4 = fc2(y3)
        print(f"  fc2:   {y3.shape} → {y4.shape}")

    # Export ONNX
    export_onnx((conv1, conv2, fc1, fc2), MODEL_DIR)

    # Generate calibration data
    calib_dir = generate_calibration_data(conv1, conv2, fc1, fc2, n_samples=200)

    # Build RKNN models
    print("\nBuilding RKNN models (INT8 quantization)...")
    results = {}

    layers = [
        ("conv1", "nmnist_conv1.onnx", "nmnist_conv1.rknn", "conv1.txt", INPUT_CHANNELS),
        ("conv2", "nmnist_conv2.onnx", "nmnist_conv2.rknn", "conv2.txt", CONV1_OUT),
        ("fc1",   "nmnist_fc1.onnx",   "nmnist_fc1.rknn",   "fc1.txt",   FC1_IN),
        ("fc2",   "nmnist_fc2.onnx",    "nmnist_fc2.rknn",   "fc2.txt",   FC1_OUT),
    ]

    for name, onnx_file, rknn_file, calib_file, channels in layers:
        print(f"\n  Building {name}...")
        ok = build_rknn(
            str(MODEL_DIR / onnx_file),
            str(MODEL_DIR / rknn_file),
            f"{calib_dir}/{calib_file}",
            channels,
        )
        results[name] = ok

    # Cleanup
    shutil.rmtree(calib_dir)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  CONVERSION SUMMARY")
    print(f"{'=' * 60}")
    all_ok = True
    for name, onnx_file, rknn_file, _, _ in layers:
        rknn_path = MODEL_DIR / rknn_file
        onnx_path = MODEL_DIR / onnx_file
        if rknn_path.exists():
            size_kb = os.path.getsize(rknn_path) / 1024
            print(f"  {name}: {rknn_file} ({size_kb:.1f} KB)")
        else:
            print(f"  {name}: FAILED")
            all_ok = False

    if all_ok:
        print(f"\n  All 4 RKNN models built successfully!")
        print(f"\n  Next: Run SNN inference with 4 layers:")
        print(f"    python demo_nmnist.py -m snn")
        print(f"    python tools/benchmark_nmnist.py --mode snn")
    else:
        print(f"\n  Some models failed. Check errors above.")


if __name__ == "__main__":
    main()