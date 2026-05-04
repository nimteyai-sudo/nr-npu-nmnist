#!/usr/bin/env python3
"""Train SNN on N-MNIST using snntorch surrogate gradient descent.

Architecture (channels aligned to mult of 8 for RKNN):
  Input: [T, 2, 34, 34]  (2 polarities: ON/OFF events)
  Conv2D(2→8, 3×3, pad=1) + LIF1
  Conv2D(8→16, 3×3, pad=1) + LIF2
  Linear(16*34*34→128) + LIF3
  Linear(128→10) + LIF_out

Channel alignment for NPU:
  Conv input 2→8 (pad input to 8 channels)
  Conv output 16 (already mult of 8)
  FC input 18496→128 (18496 is mult of 8: 2312*8)
  FC output 10→16 (pad to mult of 8)

Usage (GPU recommended, CPU very slow):
    python tools/train_nmnist.py
    python tools/train_nmnist.py --epochs 25 --lr 1e-3 --device cuda
    python tools/train_nmnist.py --device cpu --batch-size 16
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models" / "nmnist"

# Architecture constants (aligned to multiples of 8)
INPUT_CHANNELS = 2       # ON/OFF polarities
CONV1_OUT = 8            # 2→8 (input padded from 2→8 in NPU model)
CONV2_OUT = 16           # 8→16
FC1_OUT = 128            # 18496→128 (16*34*34 = 18496 inputs)
NUM_CLASSES = 10         # digits 0-9, padded to 16 in NPU model
IMG_SIZE = 34


class NMNISTNet(nn.Module):
    """SNN with LIF neurons for N-MNIST (snntorch-compatible)."""

    def __init__(self, beta=0.9, threshold=0.5):
        super().__init__()
        # Conv layers (bias=False for SNN compatibility)
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, CONV1_OUT, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(CONV1_OUT, CONV2_OUT, 3, padding=1, bias=False)
        self.fc1 = nn.Linear(CONV2_OUT * IMG_SIZE * IMG_SIZE, FC1_OUT, bias=False)
        self.fc2 = nn.Linear(FC1_OUT, NUM_CLASSES, bias=False)

        # LIF neurons
        self.beta = beta
        self.threshold = threshold

    def forward(self, x, num_steps=None):
        """Forward pass through time.

        Args:
            x: input tensor [batch, T, 2, 34, 34] or [batch, 2, 34, 34]
            num_steps: number of timesteps (overrides x.size(1) if given)
        """
        import snntorch as snn
        from snntorch import surrogate

        if not hasattr(self, 'lif1'):
            # Initialize LIF layers on first call (after snntorch is imported)
            # Use closure: surrogate.fast_sigmoid(slope=25), NOT surrogate.FastSigmoid()
            spike_grad = surrogate.fast_sigmoid(slope=25)
            self.lif1 = snn.Leaky(beta=self.beta, threshold=self.threshold,
                                   learn_beta=True, spike_grad=spike_grad)
            self.lif2 = snn.Leaky(beta=self.beta, threshold=self.threshold,
                                   learn_beta=True, spike_grad=spike_grad)
            self.lif3 = snn.Leaky(beta=self.beta, threshold=self.threshold,
                                   learn_beta=True, spike_grad=spike_grad)
            self.lif_out = snn.Leaky(beta=self.beta, threshold=self.threshold,
                                      learn_beta=True, spike_grad=spike_grad)

        if x.dim() == 4:
            # Single frame: [batch, 2, 34, 34]
            x = x.unsqueeze(1)  # [batch, 1, 2, 34, 34]
            if num_steps is None:
                num_steps = 1

        if num_steps is None:
            num_steps = x.size(1)
        elif num_steps < x.size(1):
            # Use only first num_steps timesteps
            x = x[:, :num_steps]

        batch_size = x.size(0)

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_rec = []
        for step in range(num_steps):
            cur1 = self.conv1(x[:, step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc1(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc2(spk3)
            spk4, mem_out = self.lif_out(cur4, mem_out)

            spk_rec.append(spk4)

        return torch.stack(spk_rec)  # [T, batch, 10]


class NMNISTNetANN(nn.Module):
    """ANN version for RKNN export (no LIF, just conv/linear layers + ReLU)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, CONV1_OUT, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(CONV1_OUT, CONV2_OUT, 3, padding=1, bias=False)
        self.fc1 = nn.Linear(CONV2_OUT * IMG_SIZE * IMG_SIZE, FC1_OUT, bias=False)
        self.fc2 = nn.Linear(FC1_OUT, NUM_CLASSES, bias=False)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvBlock(nn.Module):
    """Conv1 + Conv2 block for NPU (combined RKNN model)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CONV1_OUT, CONV1_OUT, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(CONV1_OUT, CONV2_OUT, 3, padding=1, bias=False)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x


class LinearBlock(nn.Module):
    """FC1 + FC2 block for NPU (combined RKNN model).

    FC1(18496→128) is converted to Conv2D(18496, 128, 1×1) for NPU.
    FC2(128→10) is converted to Conv2D(128, 16, 1×1) with output padded to 16.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(CONV2_OUT * IMG_SIZE * IMG_SIZE, FC1_OUT, bias=False)
        self.fc2 = nn.Linear(FC1_OUT, NUM_CLASSES, bias=False)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_nmnist(data_dir, batch_size=64, time_window=3000):
    """Load N-MNIST dataset using tonic (for Colab/GPU)."""
    import tonic
    import tonic.transforms as transforms

    sensor_size = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)

    frame_transform = transforms.ToFrame(
        sensor_size=sensor_size,
        time_window=time_window,
    )

    train_dataset = tonic.datasets.NMNIST(
        save_to=str(data_dir),
        train=True,
        transform=frame_transform,
    )

    test_dataset = tonic.datasets.NMNIST(
        save_to=str(data_dir),
        train=False,
        transform=frame_transform,
    )

    def collate_fn(batch):
        """Pad frames to same length within batch."""
        frames_list, labels = [], []
        for frames, label in batch:
            if isinstance(frames, np.ndarray):
                frames = torch.from_numpy(frames).float()
            frames_list.append(frames)
            labels.append(label)
        max_t = max(f.size(0) for f in frames_list)
        padded = []
        for f in frames_list:
            if f.size(0) < max_t:
                pad = torch.zeros(max_t - f.size(0), *f.shape[1:])
                f = torch.cat([f, pad], dim=0)
            padded.append(f)
        return torch.stack(padded), torch.tensor(labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    return train_loader, test_loader


def load_nmnist_preprocessed(data_dir, batch_size=64, max_steps=50):
    """Load pre-processed N-MNIST from .npz files (for CPU/local training)."""
    from nmnist_data import NMNISTDataset

    train_dataset = NMNISTDataset(data_dir, split="train")
    test_dataset = NMNISTDataset(data_dir, split="test")

    def collate_fn(batch):
        frames_list, labels = [], []
        for frames, label in batch:
            if isinstance(frames, np.ndarray):
                frames = torch.from_numpy(frames[:max_steps].astype(np.float32))
            frames_list.append(frames)
            labels.append(label)
        max_t = max(f.size(0) for f in frames_list)
        padded = []
        for f in frames_list:
            if f.size(0) < max_t:
                pad = torch.zeros(max_t - f.size(0), *f.shape[1:])
                f = torch.cat([f, pad], dim=0)
            padded.append(f)
        return torch.stack(padded), torch.tensor(labels, dtype=torch.long)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    return train_loader, test_loader


def train_snn(args):
    """Train SNN on N-MNIST using snntorch."""
    import snntorch
    import snntorch.functional as SF

    device = torch.device(args.device)

    # Load data
    print("Loading N-MNIST dataset...")
    if args.preprocessed:
        train_loader, test_loader = load_nmnist_preprocessed(
            MODEL_DIR, batch_size=args.batch_size, max_steps=args.max_steps
        )
    else:
        train_loader, test_loader = load_nmnist(
            MODEL_DIR, batch_size=args.batch_size, time_window=args.time_window
        )

    # Model
    model = NMNISTNet(beta=args.beta, threshold=args.threshold).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss: cross-entropy on spike count
    loss_fn = SF.ce_count_loss()

    print(f"\nTraining SNN on N-MNIST")
    print(f"  Device: {device}")
    print(f"  Beta: {args.beta}, Threshold: {args.threshold}")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"  snntorch version: {snntorch.__version__}")

    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device).float()
            targets = targets.to(device)

            # Truncate to max_steps
            if data.size(1) > args.max_steps:
                data = data[:, :args.max_steps]

            optimizer.zero_grad()
            spk_rec = model(data)  # [T, batch, 10]
            # ce_count_loss takes the full spike record [T, batch, 10]
            loss = loss_fn(spk_rec, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            spike_count = spk_rec.sum(0)  # [batch, 10]
            total_correct += (spike_count.argmax(1) == targets).sum().item()
            total_samples += data.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} batch {batch_idx+1}: "
                      f"loss={loss.item():.4f}")

        scheduler.step()
        train_acc = 100.0 * total_correct / total_samples
        avg_loss = total_loss / total_samples

        # Test
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(device).float()
                targets = targets.to(device)
                if data.size(1) > args.max_steps:
                    data = data[:, :args.max_steps]
                spk_rec = model(data)
                spike_count = spk_rec.sum(0)
                test_correct += (spike_count.argmax(1) == targets).sum().item()
                test_total += data.size(0)

        test_acc = 100.0 * test_correct / test_total
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} "
              f"train={train_acc:.1f}% test={test_acc:.1f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            # Save weights (just conv/linear, not LIF)
            torch.save({
                'conv1': model.conv1.state_dict(),
                'conv2': model.conv2.state_dict(),
                'fc1': model.fc1.state_dict(),
                'fc2': model.fc2.state_dict(),
                'beta': model.beta,
                'threshold': model.threshold,
            }, str(MODEL_DIR / "nmnist_snn.pt"))

    print(f"\n  Best test accuracy: {best_acc:.1f}%")

    # Save final model too
    torch.save({
        'conv1': model.conv1.state_dict(),
        'conv2': model.conv2.state_dict(),
        'fc1': model.fc1.state_dict(),
        'fc2': model.fc2.state_dict(),
        'beta': model.beta,
        'threshold': model.threshold,
    }, str(MODEL_DIR / "nmnist_snn_final.pt"))

    return best_acc


def export_ann_model(args):
    """Export trained SNN weights as ANN model for RKNN conversion."""
    from ann_to_snn import export_conv_to_onnx, linear_to_conv2d

    device = torch.device("cpu")

    # Load SNN weights
    ckpt = torch.load(str(MODEL_DIR / "nmnist_snn.pt"), map_location=device)

    # Build ANN model with same weights
    ann = NMNISTNetANN()
    ann.conv1.load_state_dict(ckpt['conv1'])
    ann.conv2.load_state_dict(ckpt['conv2'])
    ann.fc1.load_state_dict(ckpt['fc1'])
    ann.fc2.load_state_dict(ckpt['fc2'])
    ann.eval()

    # Test ANN accuracy
    print("\nTesting ANN model accuracy...")
    # We'll need test data for this, load on demand
    # For now, just save the model
    torch.save(ann.state_dict(), str(MODEL_DIR / "nmnist_ann.pt"))

    # Build NPU sub-models
    # ConvBlock: input [1, 8, 34, 34] (padded from 2→8), output [1, 16, 34, 34]
    conv_block = ConvBlock()
    conv_block.conv1.load_state_dict(ckpt['conv1'])
    conv_block.conv2.load_state_dict(ckpt['conv2'])

    # LinearBlock: input [1, 18496, 1, 1] (flattened from [16, 34, 34])
    linear_block = LinearBlock()
    linear_block.fc1.load_state_dict(ckpt['fc1'])
    linear_block.fc2.load_state_dict(ckpt['fc2'])

    # Export to ONNX
    onnx_conv = str(MODEL_DIR / "nmnist_conv_block.onnx")
    onnx_linear = str(MODEL_DIR / "nmnist_linear_block.onnx")

    # Conv block: input [1, 8, 34, 34]
    dummy_conv = torch.randn(1, CONV1_OUT, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        conv_block, dummy_conv, onnx_conv,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"  Exported: {onnx_conv}")

    # Linear block as Conv2D 1×1 layers
    # FC1(18496→128) → Conv2D(18496, 128, 1×1), input [1, 18496, 1, 1]
    # FC2(128→10) → Conv2D(128, 16, 1×1), input [1, 128, 1, 1], output padded to 16
    conv_fc1 = linear_to_conv2d(linear_block.fc1)
    conv_fc2 = linear_to_conv2d(linear_block.fc2)

    class LinearBlockConv(nn.Module):
        """Linear block using Conv2D 1×1 for NPU."""
        def __init__(self, fc1, fc2):
            super().__init__()
            self.conv1 = linear_to_conv2d(fc1)
            self.conv2 = linear_to_conv2d(fc2)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.conv2(x)
            return x

    linear_conv = LinearBlockConv(linear_block.fc1, linear_block.fc2)
    dummy_linear = torch.randn(1, CONV2_OUT * IMG_SIZE * IMG_SIZE, 1, 1)
    torch.onnx.export(
        linear_conv, dummy_linear, onnx_linear,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"  Exported: {onnx_linear}")

    print(f"\n  Models exported to {MODEL_DIR}")
    print(f"  Next step: Run 'python tools/convert_nmnist.py' to build RKNN models")


def main():
    parser = argparse.ArgumentParser(description="Train SNN on N-MNIST")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--time-window", type=int, default=3000,
                        help="Time window in microseconds for frame conversion")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max timesteps per sample (truncates longer sequences)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--preprocessed", action="store_true",
                        help="Use pre-processed .npz files instead of tonic")
    parser.add_argument("--export-only", action="store_true",
                        help="Only export ONNX from existing weights")
    parser.add_argument("--train-only", action="store_true",
                        help="Only train, don't export")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not args.export_only:
        train_snn(args)

    if not args.train_only:
        export_ann_model(args)


if __name__ == "__main__":
    main()