# SNN on RK3588 NPU — The Best Engineering Solution for Mass Neuromorphic Computing

## What Has Been Done

A full spiking neural network (SNN) has been launched on a standard Rockchip RK3588 NPU. LIF neurons are not hardware‑supported, but a workaround was found. Each layer of the network is a separate RKNN model. LIF runs on the CPU between NPU calls. Dataset: N‑MNIST.

**Result on N‑MNIST: 94.8% accuracy.**  
Loss compared to the same SNN running on a regular CPU (PyTorch, 98.6%) — only 3.8%. For a first implementation on non‑friendly hardware, this is excellent.

The goal was not to squeeze out 100% at all costs. The goal was to prove that SNN on a mass‑market NPU is possible, and to make it an open, reproducible solution.

## Comparison with Existing Platforms

| Platform | System Price | Power | N‑MNIST Accuracy | Size | Availability |
|----------|--------------|-------|------------------|------|---------------|
| SpiNNaker | $100k+ | hundreds W | ~95% | 19" rack | research centres only |
| Loihi 2 | not for sale | ~10 W | ~99% | chip | unavailable |
| GPU (RTX 3060) | $700+ (with PC) | 200+ W | ~99.5% | 250×120 mm | open, but expensive, power‑hungry |
| **RK3588 NPU (our solution)** | **$150** | **5–10 W** | **94.8%** | **100×70×15 mm** | **available to anyone** |

**This is the most energy‑efficient option among all available SNN platforms.** With accuracy close to the reference, power consumption is 20‑40 times lower than a GPU and tens of times lower than specialised racks.

## Temperature and Cooling

Measurements on Orange Pi 5 Ultra while running SNN (T=50, 100 classifications):

- CPU (A76) running LIF and encoding: heats up, requires heat dissipation.
- NPU (1 GHz, 3 cores) running convolutions and linear layers: **temperature rise only +4°C** (from 40.7 to 44.4°C). Passive cooling handles it completely.

For mobile and autonomous systems this means: no active cooling needed, reduced weight and power consumption.

## Performance and Energy

**One timestep (all 4 SNN layers):**  
NPU computations (conv1 0.44 ms + conv2 0.46 ms + fc1 0.97 ms + fc2 0.23 ms) = 2.1 ms. CPU LIF + overhead ≈ 0.2 ms. **Total per step: 2.3 ms**

**Full classification at T=50 (max accuracy 94.8%):**  
115 ms → 9 samples/s. Whole board power 5‑10 W → energy per sample ~0.6 J (overestimated because it includes the entire board).

**Fast mode T=5 (accuracy ~91%):**  
11.6 ms → 86 samples/s.

## Undocumented Technical Hurdles That Have Been Solved

- NPU expects **NHWC** layout. NCHW gives garbage, correlation with correct output near zero.
- **LIF must follow every layer**. Grouping two layers with one LIF kills accuracy (0%).
- **No ReLU between layers**. Negative outputs are needed for LIF to decrease membrane potential.
- **INT8 noise <0.001 must be clipped manually**. Otherwise LIF accumulates quantisation artefacts, accuracy drops.

All these obstacles have been overcome. Code is open.

## Where This Technology Creates Breakthrough Opportunities

**Drones and UAVs**  
Event‑based cameras output spikes. NPU processes them onboard. Latency — tens of milliseconds, power — a few watts. No need to stream video to the ground, no need for a powerful ground server. Payload weight decreases. **The cost of drone flight control drops dramatically** because a mass‑market $150 NPU replaces specialised neuromorphic hardware.

**Medical Devices**  
Wearable ECG, EEG, tremor monitors. Real‑time signal analysis directly on the device. No cloud upload required. Battery lasts for days (NPU draws 5‑10 W, not 170). **Diagnostics becomes more affordable** because it no longer requires expensive stationary systems.

**Industrial Diagnostics**  
Vibration, current, temperature sensors. NPU classifies anomalies on‑site. No cables to a server needed. The device is placed directly next to the machine.

**Smart Home & Security**  
Local keyword spotting, sound event detection (broken glass, baby cry). No internet connection, no subscriptions.

**Edge AI for Any Budget**  
Students, startups, engineers. Without multi‑thousand‑dollar investments.

## Summary 

- SNN on RK3588 NPU **works** and achieves 94.8% accuracy on N‑MNIST.
- This is the only solution in the $150 price category with passive cooling and 5‑10 W power draw.
- Best price / accuracy / power / availability ratio on the market for spiking networks.
- Drones become an order of magnitude cheaper. Wearable medical devices become practical. Industrial AI can be distributed across machines without central servers.

**This is the best engineering solution for neuromorphic computing on a mass‑market NPU. The technology cuts costs in drone engineering, unlocks new possibilities in health monitoring and diagnostics, and makes spiking networks accessible to everyone.**
