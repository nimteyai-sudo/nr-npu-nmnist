#!/usr/bin/env python3
"""Measure per-layer NPU latency and total SNN latency for N-MNIST on RK3588."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import time
from pathlib import Path
from rknnlite.api import RKNNLite
from lif_neuron import LIFNeuron

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
    inp_nhwc = np.transpose(inp_nchw, (0, 2, 3, 1))
    out = rknn.inference(inputs=[inp_nhwc], data_format='nhwc')
    out = out[0] if isinstance(out, list) else out
    if out.ndim == 4 and out.shape[-1] in (CONV1_OUT, CONV2_OUT):
        return np.transpose(out, (0, 3, 1, 2))
    return out


def npu_infer_fc(rknn, inp_nchw):
    inp_nhwc = np.transpose(inp_nchw, (0, 2, 3, 1))
    out = rknn.inference(inputs=[inp_nhwc], data_format='nhwc')
    out = out[0] if isinstance(out, list) else out
    if out.ndim == 4 and out.shape[-1] in (FC1_OUT, NUM_CLASSES_PADDED):
        return np.transpose(out, (0, 3, 1, 2))
    return out


def main():
    N = 1000  # iterations per measurement

    print("=" * 60)
    print("  N-MNIST Per-Layer NPU Latency Measurement")
    print("=" * 60)
    print(f"  NPU freq: {open('/sys/class/devfreq/fdab0000.npu/cur_freq').read().strip()} Hz")
    print(f"  Iterations per test: {N}")
    print()

    # Load models
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
    for _ in range(5):
        npu_infer_conv(models["conv1"], np.random.randn(1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE).astype(np.float32))
        npu_infer_conv(models["conv2"], np.random.randn(1, CONV1_OUT, IMG_SIZE, IMG_SIZE).astype(np.float32))
        npu_infer_fc(models["fc1"], np.random.randn(1, FC1_IN, 1, 1).astype(np.float32))
        npu_infer_fc(models["fc2"], np.random.randn(1, FC1_OUT, 1, 1).astype(np.float32))

    # Prepare test inputs
    inp_conv1 = np.random.rand(1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE).astype(np.float32)
    spike_conv2 = np.random.choice([0.0, 1.0], (1, CONV1_OUT, IMG_SIZE, IMG_SIZE), p=[0.9, 0.1]).astype(np.float32)
    spike_fc1 = np.random.choice([0.0, 1.0], (1, FC1_IN, 1, 1), p=[0.9, 0.1]).astype(np.float32)
    spike_fc2 = np.random.choice([0.0, 1.0], (1, FC1_OUT, 1, 1), p=[0.9, 0.1]).astype(np.float32)

    # Measure per-layer NPU latency
    results = {}

    print("Measuring conv1 NPU latency...")
    t0 = time.perf_counter()
    for _ in range(N):
        npu_infer_conv(models["conv1"], inp_conv1)
    results["conv1"] = (time.perf_counter() - t0) / N * 1000
    print(f"  conv1: {results['conv1']:.3f} ms/call")

    print("Measuring conv2 NPU latency...")
    t0 = time.perf_counter()
    for _ in range(N):
        npu_infer_conv(models["conv2"], spike_conv2)
    results["conv2"] = (time.perf_counter() - t0) / N * 1000
    print(f"  conv2: {results['conv2']:.3f} ms/call")

    print("Measuring fc1 NPU latency...")
    t0 = time.perf_counter()
    for _ in range(N):
        npu_infer_fc(models["fc1"], spike_fc1)
    results["fc1"] = (time.perf_counter() - t0) / N * 1000
    print(f"  fc1: {results['fc1']:.3f} ms/call")

    print("Measuring fc2 NPU latency...")
    t0 = time.perf_counter()
    for _ in range(N):
        npu_infer_fc(models["fc2"], spike_fc2)
    results["fc2"] = (time.perf_counter() - t0) / N * 1000
    print(f"  fc2: {results['fc2']:.3f} ms/call")

    # Measure CPU overhead per timestep
    print("\nMeasuring CPU overhead (LIF + noise clip + transpose)...")

    lif1 = LIFNeuron(CONV1_OUT * IMG_SIZE * IMG_SIZE, threshold=0.3, leak_rate=0.1)
    lif2 = LIFNeuron(FC1_IN, threshold=0.3, leak_rate=0.1)
    lif3 = LIFNeuron(FC1_OUT, threshold=0.3, leak_rate=0.1)
    lif_out = LIFNeuron(NUM_CLASSES, threshold=0.3, leak_rate=0.1)

    # Simulate CPU part of one timestep
    def cpu_step(c1, c2, f1, f2):
        c1[np.abs(c1) < NOISE_THRESH] = 0
        c1_p = np.pad(c1, (0, lif_conv1_size - c1.shape[0])) if c1.shape[0] < lif_conv1_size else c1.copy()
        s1 = lif1.step(c1_p)

        s1_img = s1[:CONV1_OUT * IMG_SIZE * IMG_SIZE].reshape(1, CONV1_OUT, IMG_SIZE, IMG_SIZE).astype(np.float32)
        c2[np.abs(c2) < NOISE_THRESH] = 0
        c2_p = np.pad(c2, (0, lif_conv2_size - c2.shape[0])) if c2.shape[0] < lif_conv2_size else c2.copy()
        s2 = lif2.step(c2_p)

        s2_flat = s2[:FC1_IN].reshape(1, FC1_IN, 1, 1).astype(np.float32)
        f1[np.abs(f1) < NOISE_THRESH] = 0
        f1_p = np.pad(f1, (0, lif_fc1_size - f1.shape[0])) if f1.shape[0] < lif_fc1_size else f1.copy()
        s3 = lif3.step(f1_p)

        s3_flat = s3[:FC1_OUT].reshape(1, FC1_OUT, 1, 1).astype(np.float32)
        f2[np.abs(f2) < NOISE_THRESH] = 0
        f2_p = np.pad(f2, (0, lif_out_size - f2.shape[0])) if f2.shape[0] < lif_out_size else f2.copy()
        lif_out.step(f2_p)

    # Pre-generate test data for CPU measurement
    c1_test = np.random.randn(CONV1_OUT * IMG_SIZE * IMG_SIZE).astype(np.float32)
    c2_test = np.random.randn(FC1_IN).astype(np.float32)
    f1_test = np.random.randn(FC1_OUT).astype(np.float32)
    f2_test = np.random.randn(NUM_CLASSES_PADDED).astype(np.float32)

    t0 = time.perf_counter()
    for _ in range(N):
        cpu_step(c1_test.copy(), c2_test.copy(), f1_test.copy(), f2_test.copy())
    cpu_time = (time.perf_counter() - t0) / N * 1000
    print(f"  CPU overhead per timestep: {cpu_time:.3f} ms")

    # Measure full SNN per-timestep latency
    print("\nMeasuring full SNN per-timestep latency (4 NPU + CPU)...")
    lif1 = LIFNeuron(CONV1_OUT * IMG_SIZE * IMG_SIZE, threshold=0.3, leak_rate=0.1)
    lif2 = LIFNeuron(FC1_IN, threshold=0.3, leak_rate=0.1)
    lif3 = LIFNeuron(FC1_OUT, threshold=0.3, leak_rate=0.1)
    lif_out = LIFNeuron(NUM_CLASSES, threshold=0.3, leak_rate=0.1)

    inp = np.random.rand(1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE).astype(np.float32)

    def snn_one_timestep(inp, lif1, lif2, lif3, lif_out, models):
        c1 = npu_infer_conv(models["conv1"], inp).flatten()[:CONV1_OUT * IMG_SIZE * IMG_SIZE]
        c1[np.abs(c1) < NOISE_THRESH] = 0
        c1_p = np.pad(c1, (0, lif_conv1_size - c1.shape[0])) if c1.shape[0] < lif_conv1_size else c1.copy()
        spikes1 = lif1.step(c1_p)

        spk1_img = spikes1[:CONV1_OUT * IMG_SIZE * IMG_SIZE].reshape(1, CONV1_OUT, IMG_SIZE, IMG_SIZE).astype(np.float32)
        c2 = npu_infer_conv(models["conv2"], spk1_img).flatten()[:FC1_IN]
        c2[np.abs(c2) < NOISE_THRESH] = 0
        c2_p = np.pad(c2, (0, lif_conv2_size - c2.shape[0])) if c2.shape[0] < lif_conv2_size else c2.copy()
        spikes2 = lif2.step(c2_p)

        spk2_flat = spikes2[:FC1_IN].reshape(1, FC1_IN, 1, 1).astype(np.float32)
        f1 = npu_infer_fc(models["fc1"], spk2_flat).flatten()[:FC1_OUT]
        f1[np.abs(f1) < NOISE_THRESH] = 0
        f1_p = np.pad(f1, (0, lif_fc1_size - f1.shape[0])) if f1.shape[0] < lif_fc1_size else f1.copy()
        spikes3 = lif3.step(f1_p)

        spk3_flat = spikes3[:FC1_OUT].reshape(1, FC1_OUT, 1, 1).astype(np.float32)
        f2 = npu_infer_fc(models["fc2"], spk3_flat).flatten()[:NUM_CLASSES_PADDED]
        f2[np.abs(f2) < NOISE_THRESH] = 0
        f2_p = np.pad(f2, (0, lif_out_size - f2.shape[0])) if f2.shape[0] < lif_out_size else f2.copy()
        lif_out.step(f2_p)

    # Warmup
    for _ in range(3):
        snn_one_timestep(inp, lif1, lif2, lif3, lif_out, models)

    # Reset LIF
    lif1 = LIFNeuron(CONV1_OUT * IMG_SIZE * IMG_SIZE, threshold=0.3, leak_rate=0.1)
    lif2 = LIFNeuron(FC1_IN, threshold=0.3, leak_rate=0.1)
    lif3 = LIFNeuron(FC1_OUT, threshold=0.3, leak_rate=0.1)
    lif_out = LIFNeuron(NUM_CLASSES, threshold=0.3, leak_rate=0.1)

    t0 = time.perf_counter()
    for _ in range(N):
        snn_one_timestep(inp, lif1, lif2, lif3, lif_out, models)
    snn_timestep_ms = (time.perf_counter() - t0) / N * 1000
    print(f"  Full SNN timestep: {snn_timestep_ms:.3f} ms")

    # Summary
    npu_total = results["conv1"] + results["conv2"] + results["fc1"] + results["fc2"]

    print(f"\n{'='*60}")
    print(f"  LATENCY SUMMARY")
    print(f"{'='*60}")
    print(f"  Per-layer NPU latency:")
    print(f"    conv1 (8→8,  34×34):  {results['conv1']:.3f} ms")
    print(f"    conv2 (8→16, 34×34):  {results['conv2']:.3f} ms")
    print(f"    fc1   (18496→128):    {results['fc1']:.3f} ms")
    print(f"    fc2   (128→16):       {results['fc2']:.3f} ms")
    print(f"    Total NPU/timestep:  {npu_total:.3f} ms")
    print(f"    CPU overhead/timestep: {cpu_time:.3f} ms")
    print(f"    Full SNN/timestep:    {snn_timestep_ms:.3f} ms")
    print()
    print(f"  SNN latency per sample (T=50):")
    T = 50
    snn_sample_ms = snn_timestep_ms * T
    print(f"    {snn_sample_ms:.1f} ms/sample ({T} timesteps)")
    print(f"    NPU portion: {npu_total * T:.1f} ms")
    print(f"    CPU portion: {cpu_time * T:.1f} ms")

    # Energy estimation
    # RK3588 NPU: ~5W typical at 1GHz (not 8W — that's max peak)
    # CPU cores: ~2-3W total for 4 A55 cores under load
    npu_power_w = 5.0
    cpu_power_w = 2.0
    npu_energy = npu_total * T / 1000 * npu_power_w * 1000  # mJ
    cpu_energy = cpu_time * T / 1000 * cpu_power_w * 1000  # mJ
    total_energy = npu_energy + cpu_energy

    print(f"\n  Energy estimation per sample (T={T}):")
    print(f"    NPU: {npu_energy:.1f} mJ  ({npu_total * T:.1f} ms × {npu_power_w}W)")
    print(f"    CPU: {cpu_energy:.1f} mJ  ({cpu_time * T:.1f} ms × {cpu_power_w}W)")
    print(f"    Total: {total_energy:.1f} mJ/sample")

    for rknn in models.values():
        rknn.release()


if __name__ == "__main__":
    main()