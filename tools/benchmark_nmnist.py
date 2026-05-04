#!/usr/bin/env python3
"""Benchmark N-MNIST: ANN vs SNN on RK3588 NPU (4-layer).

SNN uses 4 separate NPU calls per timestep with LIF between each:
  Frame → conv1 (NPU) → LIF1 → conv2 (NPU) → LIF2 →
         flatten → fc1 (NPU) → LIF3 → fc2 (NPU) → LIF_out

ANN uses 2-block approach (conv_block + linear_block) for single-pass inference.

IMPORTANT: RKNN NPU models require NHWC input format.
  Conv layers: (1, H, W, C) instead of (1, C, H, W)
  FC layers:   (1, 1, 1, C) instead of (1, C, 1, 1)

Usage:
    python tools/benchmark_nmnist.py
    python tools/benchmark_nmnist.py --T 50 --threshold 0.3
    python tools/benchmark_nmnist.py --mode snn
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import time
import argparse
from pathlib import Path
from rknnlite.api import RKNNLite
from lif_neuron import LIFNeuron
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

lif_conv1_size = ((CONV1_OUT * IMG_SIZE * IMG_SIZE + 7) // 8) * 8
lif_conv2_size = ((FC1_IN + 7) // 8) * 8
lif_fc1_size = ((FC1_OUT + 7) // 8) * 8
lif_out_size = ((NUM_CLASSES_PADDED + 7) // 8) * 8

NOISE_THRESH = 0.001


def npu_infer_conv(rknn, inp_nchw):
    """Conv layer inference. Input: NCHW (1,C,H,W), output: NCHW (1,C,H,W)."""
    inp_nhwc = np.transpose(inp_nchw, (0, 2, 3, 1))
    out = rknn.inference(inputs=[inp_nhwc], data_format='nhwc')
    out = out[0] if isinstance(out, list) else out
    if out.ndim == 4 and out.shape[-1] in (CONV1_OUT, CONV2_OUT):
        return np.transpose(out, (0, 3, 1, 2))
    return out


def npu_infer_fc(rknn, inp_nchw):
    """FC (1x1 conv) layer inference. Input: NCHW (1,C,1,1), output: NCHW (1,C,1,1)."""
    inp_nhwc = np.transpose(inp_nchw, (0, 2, 3, 1))
    out = rknn.inference(inputs=[inp_nhwc], data_format='nhwc')
    out = out[0] if isinstance(out, list) else out
    if out.ndim == 4 and out.shape[-1] in (FC1_OUT, NUM_CLASSES_PADDED):
        return np.transpose(out, (0, 3, 1, 2))
    return out


def load_nmnist_dataset(split="test", max_samples=0):
    return NMNISTDataset(MODEL_DIR, split=split, max_samples=max_samples)


def main():
    parser = argparse.ArgumentParser(description="N-MNIST Benchmark on RK3588 NPU (4-Layer)")
    parser.add_argument("--mode", choices=["ann", "snn", "both"], default="snn")
    parser.add_argument("--T", type=int, default=50, help="SNN timesteps")
    parser.add_argument("--threshold", type=float, default=0.3, help="LIF threshold")
    parser.add_argument("--leak", type=float, default=0.1, help="LIF leak rate")
    parser.add_argument("--samples", type=int, default=0, help="Max samples (0=all)")
    args = parser.parse_args()

    print("=" * 65)
    print("  N-MNIST Benchmark on RK3588 NPU (4-Layer SNN)")
    print("=" * 65)

    # Load data
    print("\nLoading data...")
    try:
        dataset = load_nmnist_dataset("test", max_samples=args.samples)
    except FileNotFoundError:
        print("  ERROR: Pre-processed data not found.")
        print("  Run 'python3.13 tools/preprocess_nmnist.py' first")
        return

    labels = dataset.get_labels()
    n = len(dataset)
    print(f"  Samples: {n}")

    # === ANN Benchmark (2-block) ===
    ann_acc = ann_avg_ms = ann_energy_mj = 0
    if args.mode in ("ann", "both"):
        print("\n=== ANN on NPU (2-block) ===")

        rknn_conv = RKNNLite()
        rknn_conv.load_rknn(str(MODEL_DIR / "nmnist_conv_block.rknn"))
        rknn_conv.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_AUTO)

        rknn_linear = RKNNLite()
        rknn_linear.load_rknn(str(MODEL_DIR / "nmnist_linear_block.rknn"))
        rknn_linear.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_AUTO)

        # Warmup
        npu_infer_conv(rknn_conv, np.random.randn(1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE).astype(np.float32))
        npu_infer_fc(rknn_linear, np.random.randn(1, FC1_IN, 1, 1).astype(np.float32))

        ann_correct = 0
        ann_times = []
        t0 = time.perf_counter()

        for i in range(n):
            ts = time.perf_counter()
            sample, label = dataset[i]

            if sample.ndim == 4:
                frame_avg = sample.sum(axis=0)  # [2, 34, 34]
            elif sample.ndim == 3:
                frame_avg = sample
            else:
                frame_avg = sample.reshape(2, IMG_SIZE, IMG_SIZE)

            inp = np.zeros((1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            inp[0, :2] = frame_avg[:2]
            conv_out = npu_infer_conv(rknn_conv, inp).flatten()[:FC1_IN]
            conv_relu = np.maximum(conv_out, 0)
            if conv_relu.shape[0] < FC1_IN:
                conv_relu = np.pad(conv_relu, (0, FC1_IN - conv_relu.shape[0]))
            linear_in = conv_relu.reshape(1, FC1_IN, 1, 1).astype(np.float32)
            linear_out = npu_infer_fc(rknn_linear, linear_in).flatten()[:NUM_CLASSES]
            pred = np.argmax(linear_out)

            ann_times.append((time.perf_counter() - ts) * 1000)
            if pred == label:
                ann_correct += 1

            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{n} done, acc so far: {100.0*ann_correct/(i+1):.1f}%")

        ann_total_ms = (time.perf_counter() - t0) * 1000
        ann_acc = 100.0 * ann_correct / n
        ann_avg_ms = np.mean(ann_times)
        ann_energy_mj = ann_avg_ms / 1000.0 * 8.0 * 1000

        print(f"  Accuracy: {ann_acc:.1f}%")
        print(f"  Latency: {ann_avg_ms:.3f} ms/sample ({ann_total_ms:.0f} ms total)")
        print(f"  Energy: {ann_energy_mj:.3f} mJ/sample (est. 8W NPU)")

        rknn_conv.release()
        rknn_linear.release()

    # === SNN Benchmark (4-layer) ===
    snn_acc = snn_avg_ms = snn_energy_mj = 0
    if args.mode in ("snn", "both"):
        print(f"\n=== SNN on NPU (4-layer, T={args.T}, thresh={args.threshold}) ===")

        models = {}
        for name in ["conv1", "conv2", "fc1", "fc2"]:
            rknn = RKNNLite()
            ret = rknn.load_rknn(str(MODEL_DIR / f"nmnist_{name}.rknn"))
            if ret != 0:
                print(f"  ERROR: Failed to load nmnist_{name}.rknn")
                print(f"  Run 'python tools/convert_nmnist_4layer.py' first")
                return
            ret = rknn.init_runtime(target=None, core_mask=RKNNLite.NPU_CORE_AUTO)
            models[name] = rknn

        # Warmup
        npu_infer_conv(models["conv1"], np.random.randn(1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE).astype(np.float32))
        npu_infer_conv(models["conv2"], np.random.randn(1, CONV1_OUT, IMG_SIZE, IMG_SIZE).astype(np.float32))
        npu_infer_fc(models["fc1"], np.random.randn(1, FC1_IN, 1, 1).astype(np.float32))
        npu_infer_fc(models["fc2"], np.random.randn(1, FC1_OUT, 1, 1).astype(np.float32))

        snn_correct = 0
        snn_times = []

        t0 = time.perf_counter()
        for i in range(n):
            ts = time.perf_counter()
            sample, label = dataset[i]

            if sample.ndim == 2:
                sample = sample.reshape(1, 2, IMG_SIZE, IMG_SIZE)

            effective_t = min(args.T, sample.shape[0])

            lif1 = LIFNeuron(CONV1_OUT * IMG_SIZE * IMG_SIZE, threshold=args.threshold, leak_rate=args.leak)
            lif2 = LIFNeuron(FC1_IN, threshold=args.threshold, leak_rate=args.leak)
            lif3 = LIFNeuron(FC1_OUT, threshold=args.threshold, leak_rate=args.leak)
            lif_out = LIFNeuron(NUM_CLASSES, threshold=args.threshold, leak_rate=args.leak)

            for t in range(effective_t):
                frame = sample[t]

                inp = np.zeros((1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
                inp[0, :2] = frame[:2]
                c1 = npu_infer_conv(models["conv1"], inp).flatten()[:CONV1_OUT * IMG_SIZE * IMG_SIZE]
                c1[np.abs(c1) < NOISE_THRESH] = 0
                c1_padded = np.pad(c1, (0, lif_conv1_size - c1.shape[0])) if c1.shape[0] < lif_conv1_size else c1.copy()
                spikes1 = lif1.step(c1_padded)

                spk1_img = spikes1[:CONV1_OUT * IMG_SIZE * IMG_SIZE].reshape(1, CONV1_OUT, IMG_SIZE, IMG_SIZE).astype(np.float32)
                c2 = npu_infer_conv(models["conv2"], spk1_img).flatten()[:FC1_IN]
                c2[np.abs(c2) < NOISE_THRESH] = 0
                c2_padded = np.pad(c2, (0, lif_conv2_size - c2.shape[0])) if c2.shape[0] < lif_conv2_size else c2.copy()
                spikes2 = lif2.step(c2_padded)

                spk2_flat = spikes2[:FC1_IN].reshape(1, FC1_IN, 1, 1).astype(np.float32)
                f1 = npu_infer_fc(models["fc1"], spk2_flat).flatten()[:FC1_OUT]
                f1[np.abs(f1) < NOISE_THRESH] = 0
                f1_padded = np.pad(f1, (0, lif_fc1_size - f1.shape[0])) if f1.shape[0] < lif_fc1_size else f1.copy()
                spikes3 = lif3.step(f1_padded)

                spk3_flat = spikes3[:FC1_OUT].reshape(1, FC1_OUT, 1, 1).astype(np.float32)
                f2 = npu_infer_fc(models["fc2"], spk3_flat).flatten()[:NUM_CLASSES_PADDED]
                f2[np.abs(f2) < NOISE_THRESH] = 0
                f2_padded = np.pad(f2, (0, lif_out_size - f2.shape[0])) if f2.shape[0] < lif_out_size else f2.copy()
                lif_out.step(f2_padded)

            pred = np.argmax(lif_out.get_spike_rate()[:NUM_CLASSES])
            snn_times.append((time.perf_counter() - ts) * 1000)
            if pred == label:
                snn_correct += 1

            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{n} done, acc so far: {100.0*snn_correct/(i+1):.1f}%")

        snn_total_ms = (time.perf_counter() - t0) * 1000
        snn_acc = 100.0 * snn_correct / n
        snn_avg_ms = np.mean(snn_times)
        snn_energy_mj = snn_avg_ms / 1000.0 * 8.0 * 1000

        print(f"  Accuracy: {snn_acc:.1f}%")
        print(f"  Latency: {snn_avg_ms:.3f} ms/sample ({snn_total_ms:.0f} ms total)")
        print(f"  Energy: {snn_energy_mj:.3f} mJ/sample (est. 8W NPU)")

        for rknn in models.values():
            rknn.release()

    # Summary
    if args.mode == "both":
        print(f"\n{'='*65}")
        print(f"  BENCHMARK SUMMARY")
        print(f"{'='*65}")
        print(f"  Dataset: N-MNIST (10 classes, {n} test samples)")
        print(f"  Model: Conv(2→8→16) + FC(18496→128→10)")
        print(f"  {'Method':<30} {'Accuracy':>8} {'Latency':>10} {'Energy':>10}")
        print(f"  {'-'*60}")
        print(f"  {'ANN NPU (2-block)':<30} {ann_acc:>7.1f}% {ann_avg_ms:>9.3f}ms {ann_energy_mj:>9.3f}mJ")
        print(f"  {'SNN NPU (4-layer T='+str(args.T)+')':<30} {snn_acc:>7.1f}% {snn_avg_ms:>9.3f}ms {snn_energy_mj:>9.3f}mJ")
        print(f"  {'-'*60}")
        if ann_acc > 0:
            print(f"  SNN/ANN accuracy: {snn_acc/ann_acc*100:.1f}%")
            print(f"  SNN/ANN latency:  {snn_avg_ms/ann_avg_ms:.1f}x slower")
            print(f"  SNN/ANN energy:   {snn_energy_mj/ann_energy_mj:.1f}x more")


if __name__ == "__main__":
    main()