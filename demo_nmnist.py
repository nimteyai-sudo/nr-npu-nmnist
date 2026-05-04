#!/usr/bin/env python3
"""Live demo: classify N-MNIST digits using SNN on RK3588 NPU (4-layer).

Uses 4 separate NPU calls per timestep with LIF neurons between each:
  Frame → conv1 (NPU) → LIF1 → conv2 (NPU) → LIF2 →
         flatten → fc1 (NPU) → LIF3 → fc2 (NPU) → LIF_out → argmax

IMPORTANT: RKNN NPU models require NHWC input format.
  Conv layers: (1, H, W, C) instead of (1, C, H, W)
  FC layers:   (1, 1, 1, C) instead of (1, C, 1, 1)

Usage:
    python demo_nmnist.py          # ANN mode (single pass)
    python demo_nmnist.py -m snn   # SNN mode (4-layer with LIF)
    python demo_nmnist.py -m snn -T 10 --threshold 0.5
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
from pathlib import Path
from rknnlite.api import RKNNLite
from lif_neuron import LIFNeuron
from nmnist_data import NMNISTDataset

MODEL_DIR = Path(__file__).parent / "models" / "nmnist"

INPUT_CHANNELS = 8
CONV1_OUT = 8
CONV2_OUT = 16
IMG_SIZE = 34
FC1_IN = CONV2_OUT * IMG_SIZE * IMG_SIZE  # 18496
FC1_OUT = 128
NUM_CLASSES = 10
NUM_CLASSES_PADDED = 16
CLASS_NAMES = [str(i) for i in range(10)]

# LIF alignment sizes
lif_conv1_size = ((CONV1_OUT * IMG_SIZE * IMG_SIZE + 7) // 8) * 8
lif_conv2_size = ((FC1_IN + 7) // 8) * 8
lif_fc1_size = ((FC1_OUT + 7) // 8) * 8
lif_out_size = ((NUM_CLASSES_PADDED + 7) // 8) * 8

# INT8 noise threshold
NOISE_THRESH = 0.001


def npu_infer_conv(rknn, inp_nchw):
    """Inference for conv layers. Input: NCHW (1,C,H,W), output: NCHW (1,C,H,W)."""
    inp_nhwc = np.transpose(inp_nchw, (0, 2, 3, 1))
    out = rknn.inference(inputs=[inp_nhwc], data_format='nhwc')
    out = out[0] if isinstance(out, list) else out
    if out.ndim == 4 and out.shape[-1] in (CONV1_OUT, CONV2_OUT):
        return np.transpose(out, (0, 3, 1, 2))
    return out


def npu_infer_fc(rknn, inp_nchw):
    """Inference for FC (1x1 conv) layers. Input: NCHW (1,C,1,1), output: NCHW (1,C,1,1)."""
    inp_nhwc = np.transpose(inp_nchw, (0, 2, 3, 1))
    out = rknn.inference(inputs=[inp_nhwc], data_format='nhwc')
    out = out[0] if isinstance(out, list) else out
    if out.ndim == 4 and out.shape[-1] in (FC1_OUT, NUM_CLASSES_PADDED):
        return np.transpose(out, (0, 3, 1, 2))
    return out


def load_nmnist_dataset(split="test", max_samples=0):
    return NMNISTDataset(MODEL_DIR, split=split, max_samples=max_samples)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="N-MNIST SNN Demo (4-Layer)")
    parser.add_argument("-m", "--mode", choices=["ann", "snn"], default="snn")
    parser.add_argument("-T", "--timesteps", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--leak", type=float, default=0.1)
    parser.add_argument("-n", "--num-samples", type=int, default=20)
    args = parser.parse_args()

    print(f"N-MNIST Demo ({args.mode.upper()} mode, 4-layer)")
    print(f"  T={args.timesteps}, threshold={args.threshold}, leak={args.leak}")
    print()

    # Load data
    dataset = load_nmnist_dataset("test", max_samples=args.num_samples)
    labels = dataset.get_labels()
    print(f"Loaded {len(dataset)} test samples")

    if args.mode == "ann":
        print("Loading 2-block NPU models (ANN mode)...")
        rknn_conv = RKNNLite()
        rknn_conv.load_rknn(str(MODEL_DIR / "nmnist_conv_block.rknn"))
        rknn_conv.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_AUTO)
        rknn_linear = RKNNLite()
        rknn_linear.load_rknn(str(MODEL_DIR / "nmnist_linear_block.rknn"))
        rknn_linear.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_AUTO)
        # Warmup
        npu_infer_conv(rknn_conv, np.random.randn(1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE).astype(np.float32))
        npu_infer_fc(rknn_linear, np.random.randn(1, FC1_IN, 1, 1).astype(np.float32))
    else:
        print("Loading 4-layer NPU models (SNN mode)...")
        models = {}
        for name in ["conv1", "conv2", "fc1", "fc2"]:
            rknn = RKNNLite()
            ret = rknn.load_rknn(str(MODEL_DIR / f"nmnist_{name}.rknn"))
            if ret != 0:
                print(f"  ERROR: Failed to load nmnist_{name}.rknn")
                return
            ret = rknn.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_AUTO)
            models[name] = rknn
        # Warmup
        npu_infer_conv(models["conv1"], np.random.randn(1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE).astype(np.float32))
        npu_infer_conv(models["conv2"], np.random.randn(1, CONV1_OUT, IMG_SIZE, IMG_SIZE).astype(np.float32))
        npu_infer_fc(models["fc1"], np.random.randn(1, FC1_IN, 1, 1).astype(np.float32))
        npu_infer_fc(models["fc2"], np.random.randn(1, FC1_OUT, 1, 1).astype(np.float32))

    correct = 0
    total = min(args.num_samples, len(labels))

    print(f"\nClassifying {total} samples from test set:\n")

    for idx in range(total):
        sample, label = dataset[idx]

        if args.mode == "ann":
            if sample.ndim == 3:
                frame_avg = sample.sum(axis=0)
            else:
                frame_avg = sample

            inp = np.zeros((1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            inp[0, :2] = frame_avg[:2]
            conv_out = npu_infer_conv(rknn_conv, inp).flatten()[:FC1_IN]
            conv_relu = np.maximum(conv_out, 0)
            if conv_relu.shape[0] < FC1_IN:
                conv_relu = np.pad(conv_relu, (0, FC1_IN - conv_relu.shape[0]))
            linear_in = conv_relu.reshape(1, FC1_IN, 1, 1).astype(np.float32)
            linear_out = npu_infer_fc(rknn_linear, linear_in).flatten()[:NUM_CLASSES]
            pred = np.argmax(linear_out)
        else:
            # SNN: 4-layer with LIF between each
            if sample.ndim == 2:
                sample = sample.reshape(1, 2, IMG_SIZE, IMG_SIZE)

            lif1 = LIFNeuron(CONV1_OUT * IMG_SIZE * IMG_SIZE, threshold=args.threshold, leak_rate=args.leak)
            lif2 = LIFNeuron(FC1_IN, threshold=args.threshold, leak_rate=args.leak)
            lif3 = LIFNeuron(FC1_OUT, threshold=args.threshold, leak_rate=args.leak)
            lif_out = LIFNeuron(NUM_CLASSES, threshold=args.threshold, leak_rate=args.leak)

            effective_t = min(args.timesteps, sample.shape[0])

            for t in range(effective_t):
                frame = sample[t]

                # Conv1 → LIF1 (no ReLU, just clip INT8 noise)
                inp = np.zeros((1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
                inp[0, :2] = frame[:2]
                c1 = npu_infer_conv(models["conv1"], inp).flatten()[:CONV1_OUT * IMG_SIZE * IMG_SIZE]
                c1[np.abs(c1) < NOISE_THRESH] = 0
                c1_padded = np.pad(c1, (0, lif_conv1_size - c1.shape[0])) if c1.shape[0] < lif_conv1_size else c1.copy()
                spikes1 = lif1.step(c1_padded)

                # Conv2: feed LIF1 spikes
                spk1_img = spikes1[:CONV1_OUT * IMG_SIZE * IMG_SIZE].reshape(1, CONV1_OUT, IMG_SIZE, IMG_SIZE).astype(np.float32)
                c2 = npu_infer_conv(models["conv2"], spk1_img).flatten()[:FC1_IN]
                c2[np.abs(c2) < NOISE_THRESH] = 0
                c2_padded = np.pad(c2, (0, lif_conv2_size - c2.shape[0])) if c2.shape[0] < lif_conv2_size else c2.copy()
                spikes2 = lif2.step(c2_padded)

                # FC1: feed LIF2 spikes
                spk2_flat = spikes2[:FC1_IN].reshape(1, FC1_IN, 1, 1).astype(np.float32)
                f1 = npu_infer_fc(models["fc1"], spk2_flat).flatten()[:FC1_OUT]
                f1[np.abs(f1) < NOISE_THRESH] = 0
                f1_padded = np.pad(f1, (0, lif_fc1_size - f1.shape[0])) if f1.shape[0] < lif_fc1_size else f1.copy()
                spikes3 = lif3.step(f1_padded)

                # FC2: feed LIF3 spikes
                spk3_flat = spikes3[:FC1_OUT].reshape(1, FC1_OUT, 1, 1).astype(np.float32)
                f2 = npu_infer_fc(models["fc2"], spk3_flat).flatten()[:NUM_CLASSES_PADDED]
                f2[np.abs(f2) < NOISE_THRESH] = 0
                f2_padded = np.pad(f2, (0, lif_out_size - f2.shape[0])) if f2.shape[0] < lif_out_size else f2.copy()
                lif_out.step(f2_padded)

            pred = np.argmax(lif_out.get_spike_rate()[:NUM_CLASSES])

        is_correct = pred == label
        correct += is_correct
        mark = "OK" if is_correct else "XX"
        print(f"  [{idx:3d}] True={CLASS_NAMES[label]} Pred={CLASS_NAMES[pred]} {mark}")

    print(f"\n  Accuracy: {correct}/{total} = {100.0*correct/total:.1f}%")

    if args.mode == "ann":
        rknn_conv.release()
        rknn_linear.release()
    else:
        for rknn in models.values():
            rknn.release()


if __name__ == "__main__":
    main()