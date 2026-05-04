#!/usr/bin/env python3
"""Measure per-layer NPU activation ranges for N-MNIST SNN normalization.

Measures each layer independently with representative inputs:
  1. conv1: feed N-MNIST frames → measure ReLU(conv1_output)
  2. conv2: feed random spikes → measure ReLU(conv2_output)
  3. fc1: feed random spikes → measure ReLU(fc1_output)
  4. fc2: feed random spikes → measure ReLU(fc2_output)

Computes 99th percentile of ReLU(output) for each layer.
These values normalize NPU outputs to [0, ~1] for LIF neurons.

Usage:
    python tools/measure_nmnist_ranges_4layer.py
    python tools/measure_nmnist_ranges_4layer.py --samples 500
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import argparse
from pathlib import Path
from rknnlite.api import RKNNLite
from nmnist_data import NMNISTDataset

MODEL_DIR = Path(__file__).parent.parent / "models" / "nmnist"

INPUT_CHANNELS = 8
CONV1_OUT = 8
CONV2_OUT = 16
IMG_SIZE = 34
FC1_IN = CONV2_OUT * IMG_SIZE * IMG_SIZE  # 18496
FC1_OUT = 128
NUM_CLASSES = 10
NUM_CLASSES_PADDED = 16


def npu_infer(rknn, inp):
    out = rknn.inference(inputs=[inp])
    if isinstance(out, list):
        return out[0].flatten()
    return out.flatten()


def main():
    parser = argparse.ArgumentParser(description="Measure N-MNIST per-layer activation ranges")
    parser.add_argument("--samples", type=int, default=0, help="Max samples (0=all)")
    args = parser.parse_args()

    print("=" * 60)
    print("  N-MNIST Per-Layer Activation Range Measurement (4-Layer)")
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

    # Load 4 NPU models
    print("\nLoading 4 NPU models...")
    models = {}
    for name in ["conv1", "conv2", "fc1", "fc2"]:
        rknn = RKNNLite()
        ret = rknn.load_rknn(str(MODEL_DIR / f"nmnist_{name}.rknn"))
        if ret != 0:
            print(f"  ERROR: Failed to load nmnist_{name}.rknn")
            return
        ret = rknn.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            print(f"  ERROR: Failed to init runtime for {name}")
            return
        models[name] = rknn

    # Warmup
    models["conv1"].inference(inputs=[np.random.randn(1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE).astype(np.float32)])
    models["conv2"].inference(inputs=[np.random.randn(1, CONV1_OUT, IMG_SIZE, IMG_SIZE).astype(np.float32)])
    models["fc1"].inference(inputs=[np.random.randn(1, FC1_IN, 1, 1).astype(np.float32)])
    models["fc2"].inference(inputs=[np.random.randn(1, FC1_OUT, 1, 1).astype(np.float32)])

    indices = np.random.choice(n, min(n, 2000), replace=False) if n > 2000 else np.arange(n)

    # === Conv1: measure with actual N-MNIST frames ===
    print(f"\nMeasuring conv1 activations on {len(indices)} frames...")
    conv1_outputs = []
    for i, idx in enumerate(indices):
        frames, label = dataset[idx]
        if frames.ndim == 4:
            # [T, 2, 34, 34]
            n_frames = min(frames.shape[0], 10)
        elif frames.ndim == 3:
            # [2, 34, 34]
            n_frames = 1
            frames = frames.reshape(1, 2, IMG_SIZE, IMG_SIZE)
        else:
            n_frames = 1
            frames = frames.reshape(1, 1, IMG_SIZE, IMG_SIZE)
        for t in range(n_frames):
            frame = frames[t]
            inp = np.zeros((1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            if frame.shape[0] <= 2:
                inp[0, :frame.shape[0]] = frame[:2]
            c1 = npu_infer(models["conv1"], inp)[:CONV1_OUT * IMG_SIZE * IMG_SIZE]
            conv1_outputs.append(np.maximum(c1, 0))
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(indices)} samples")

    conv1_outputs = np.array(conv1_outputs)
    max_act_conv1 = float(np.percentile(conv1_outputs[conv1_outputs > 0], 99))
    print(f"  Conv1: max={conv1_outputs.max():.4f}, "
          f"mean={conv1_outputs.mean():.4f}, "
          f"99th (positive)={max_act_conv1:.4f}")

    # === Conv2: feed random binary spikes (0/1) ===
    print(f"\nMeasuring conv2 activations with random spikes...")
    conv2_outputs = []
    for i in range(len(indices)):
        # Binary spike input (simulating LIF1 output)
        spike_in = np.random.choice([0.0, 1.0], size=(1, CONV1_OUT, IMG_SIZE, IMG_SIZE),
                                     p=[0.9, 0.1]).astype(np.float32)
        c2 = npu_infer(models["conv2"], spike_in)[:FC1_IN]
        conv2_outputs.append(np.maximum(c2, 0))
    conv2_outputs = np.array(conv2_outputs)
    max_act_conv2 = float(np.percentile(conv2_outputs[conv2_outputs > 0], 99))
    print(f"  Conv2: max={conv2_outputs.max():.4f}, "
          f"mean={conv2_outputs.mean():.4f}, "
          f"99th (positive)={max_act_conv2:.4f}")

    # === FC1: feed random binary spikes ===
    print(f"\nMeasuring fc1 activations with random spikes...")
    fc1_outputs = []
    for i in range(len(indices)):
        spike_in = np.random.choice([0.0, 1.0], size=(1, FC1_IN, 1, 1),
                                     p=[0.9, 0.1]).astype(np.float32)
        f1 = npu_infer(models["fc1"], spike_in)[:FC1_OUT]
        fc1_outputs.append(np.maximum(f1, 0))
    fc1_outputs = np.array(fc1_outputs)
    max_act_fc1 = float(np.percentile(fc1_outputs[fc1_outputs > 0], 99))
    print(f"  FC1: max={fc1_outputs.max():.4f}, "
          f"mean={fc1_outputs.mean():.4f}, "
          f"99th (positive)={max_act_fc1:.4f}")

    # === FC2: feed random binary spikes ===
    print(f"\nMeasuring fc2 activations with random spikes...")
    fc2_outputs = []
    for i in range(len(indices)):
        spike_in = np.random.choice([0.0, 1.0], size=(1, FC1_OUT, 1, 1),
                                     p=[0.9, 0.1]).astype(np.float32)
        f2 = npu_infer(models["fc2"], spike_in)[:NUM_CLASSES_PADDED]
        fc2_outputs.append(np.maximum(f2, 0))
    fc2_outputs = np.array(fc2_outputs)
    max_act_fc2 = float(np.percentile(fc2_outputs[fc2_outputs > 0], 99))
    print(f"  FC2: max={fc2_outputs.max():.4f}, "
          f"mean={fc2_outputs.mean():.4f}, "
          f"99th (positive)={max_act_fc2:.4f}")

    # Save
    np.save(str(MODEL_DIR / "nmnist_max_act_conv1.npy"), np.array([max_act_conv1]))
    np.save(str(MODEL_DIR / "nmnist_max_act_conv2.npy"), np.array([max_act_conv2]))
    np.save(str(MODEL_DIR / "nmnist_max_act_fc1.npy"), np.array([max_act_fc1]))
    np.save(str(MODEL_DIR / "nmnist_max_act_fc2.npy"), np.array([max_act_fc2]))

    print(f"\n  Saved:")
    print(f"    nmnist_max_act_conv1.npy ({max_act_conv1:.4f})")
    print(f"    nmnist_max_act_conv2.npy ({max_act_conv2:.4f})")
    print(f"    nmnist_max_act_fc1.npy   ({max_act_fc1:.4f})")
    print(f"    nmnist_max_act_fc2.npy   ({max_act_fc2:.4f})")

    for rknn in models.values():
        rknn.release()


if __name__ == "__main__":
    main()