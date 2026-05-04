#!/usr/bin/env python3
"""Train SNN on N-MNIST using snntorch + CUDA.

Standalone script: runs on any machine with CUDA + tonic + snntorch.

Architecture (channels aligned to mult of 8 for RK3588 NPU):
  Input: [T, 2, 34, 34]  (2 polarities: ON/OFF)
  Conv2D(2→8, 3×3, pad=1, bias=False) + LIF1
  Conv2D(8→16, 3×3, pad=1, bias=False) + LIF2
  Linear(16*34*34→128, bias=False) + LIF3
  Linear(128→10, bias=False) + LIF_out

Usage:
    pip install tonic snntorch torch
    python train_nmnist_cuda.py
    python train_nmnist_cuda.py --epochs 25 --batch-size 64 --lr 1e-3
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
from pathlib import Path

# Architecture constants
INPUT_CHANNELS = 2
CONV1_OUT = 8
CONV2_OUT = 16
IMG_SIZE = 34
FC1_IN = CONV2_OUT * IMG_SIZE * IMG_SIZE  # 18496
FC1_OUT = 128
NUM_CLASSES = 10


class NMNISTNet(nn.Module):
    """SNN with LIF neurons for N-MNIST."""
    def __init__(self, beta=0.9, threshold=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, CONV1_OUT, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(CONV1_OUT, CONV2_OUT, 3, padding=1, bias=False)
        self.fc1 = nn.Linear(FC1_IN, FC1_OUT, bias=False)
        self.fc2 = nn.Linear(FC1_OUT, NUM_CLASSES, bias=False)
        self.beta = beta
        self.threshold = threshold
        self._lif_init = False

    def _init_lif(self):
        import snntorch as snn
        from snntorch import surrogate
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.lif1 = snn.Leaky(beta=self.beta, threshold=self.threshold,
                               learn_beta=True, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=self.beta, threshold=self.threshold,
                               learn_beta=True, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=self.beta, threshold=self.threshold,
                               learn_beta=True, spike_grad=spike_grad)
        self.lif_out = snn.Leaky(beta=self.beta, threshold=self.threshold,
                                  learn_beta=True, spike_grad=spike_grad)
        self._lif_init = True

    def forward(self, x):
        if not self._lif_init:
            self._init_lif()
            # Move LIF to same device as conv layers
            if x.is_cuda:
                self.lif1 = self.lif1.to(x.device)
                self.lif2 = self.lif2.to(x.device)
                self.lif3 = self.lif3.to(x.device)
                self.lif_out = self.lif_out.to(x.device)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_rec = []
        for step in range(x.size(1)):
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


class ConvBlock(nn.Module):
    """Conv1+Conv2 block for RKNN export."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CONV1_OUT, CONV1_OUT, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(CONV1_OUT, CONV2_OUT, 3, padding=1, bias=False)
    def forward(self, x):
        return torch.relu(self.conv2(torch.relu(self.conv1(x))))


class LinearBlockConv(nn.Module):
    """FC1+FC2 as Conv2D 1×1 for RKNN export."""
    def __init__(self, fc1, fc2):
        super().__init__()
        self.conv1 = self._linear_to_conv2d(fc1)
        self.conv2 = self._linear_to_conv2d(fc2)
    def forward(self, x):
        return self.conv2(torch.relu(self.conv1(x)))
    @staticmethod
    def _linear_to_conv2d(linear):
        in_f, out_f = linear.in_features, linear.out_features
        in_a = ((in_f + 7) // 8) * 8
        out_a = ((out_f + 7) // 8) * 8
        conv = nn.Conv2d(in_a, out_a, 1, bias=False)
        conv.weight[:] = 0.0
        conv.weight[:out_f, :in_f, 0, 0] = linear.weight.data
        return conv


def collate_fn(batch, max_steps=50):
    """Pad variable-length frames to same T, truncate to max_steps."""
    frames_list, labels = [], []
    for frames, label in batch:
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames[:max_steps].astype(np.float32))
        elif isinstance(frames, torch.Tensor):
            frames = frames[:max_steps].float()
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


def main():
    parser = argparse.ArgumentParser(description="Train SNN on N-MNIST (CUDA)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--time-window", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max timesteps per sample")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="models_nmnist",
                        help="Directory to save model weights")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    import tonic
    import tonic.transforms as transforms
    import snntorch
    import snntorch.functional as SF

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.ToFrame(sensor_size=sensor_size,
                                           time_window=args.time_window)

    print("Loading N-MNIST dataset...")
    train_ds = tonic.datasets.NMNIST(save_to=str(save_dir), train=True,
                                      transform=frame_transform)
    test_ds = tonic.datasets.NMNIST(save_to=str(save_dir), train=False,
                                     transform=frame_transform)
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

    collate = lambda b: collate_fn(b, max_steps=args.max_steps)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate, num_workers=args.num_workers,
                               pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=True)

    # Model
    model = NMNISTNet(beta=args.beta, threshold=args.threshold)
    # Force LIF init on correct device
    dummy = torch.randn(1, 1, 2, 34, 34, device=device)
    model.to(device)
    model(dummy)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = SF.ce_count_loss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")
    print(f"  Beta: {args.beta}, Threshold: {args.threshold}")
    print(f"  Max steps: {args.max_steps}, Time window: {args.time_window}us")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  snntorch: {snntorch.__version__}, torch: {torch.__version__}\n")

    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for data, targets in train_loader:
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            spk_rec = model(data)
            loss = loss_fn(spk_rec, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            spike_count = spk_rec.sum(0)
            correct += (spike_count.argmax(1) == targets).sum().item()
            total += data.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = total_loss / total
        elapsed = time.time() - t0

        # Test
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                spk_rec = model(data)
                spike_count = spk_rec.sum(0)
                test_correct += (spike_count.argmax(1) == targets).sum().item()
                test_total += data.size(0)
        test_acc = 100.0 * test_correct / test_total

        print(f"Epoch {epoch+1:2d}/{args.epochs}: "
              f"loss={avg_loss:.4f} train={train_acc:.1f}% "
              f"test={test_acc:.1f}% ({elapsed:.1f}s)")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'conv1': model.conv1.state_dict(),
                'conv2': model.conv2.state_dict(),
                'fc1': model.fc1.state_dict(),
                'fc2': model.fc2.state_dict(),
                'beta': model.beta,
                'threshold': model.threshold,
                'test_acc': test_acc,
                'epoch': epoch + 1,
            }, str(save_dir / "nmnist_snn.pt"))

    print(f"\nBest test accuracy: {best_acc:.1f}%")
    print(f"Model saved to: {save_dir / 'nmnist_snn.pt'}")

    # Export ONNX for RKNN conversion
    print("\nExporting ONNX models...")
    ckpt = torch.load(str(save_dir / "nmnist_snn.pt"), map_location="cpu")

    conv_block = ConvBlock()
    conv_block.conv1.load_state_dict(ckpt['conv1'])
    conv_block.conv2.load_state_dict(ckpt['conv2'])
    conv_block.eval()

    fc1 = nn.Linear(FC1_IN, FC1_OUT, bias=False)
    fc1.load_state_dict(ckpt['fc1'])
    fc2 = nn.Linear(FC1_OUT, NUM_CLASSES, bias=False)
    fc2.load_state_dict(ckpt['fc2'])
    linear_block = LinearBlockConv(fc1, fc2)
    linear_block.eval()

    # Conv block: [1, 8, 34, 34] input (2ch padded to 8)
    torch.onnx.export(conv_block, torch.randn(1, CONV1_OUT, IMG_SIZE, IMG_SIZE),
                       str(save_dir / "nmnist_conv_block.onnx"), opset_version=11,
                       input_names=["input"], output_names=["output"])

    # Linear block: [1, 18496, 1, 1] input
    fc1_in_aligned = ((FC1_IN + 7) // 8) * 8
    torch.onnx.export(linear_block, torch.randn(1, fc1_in_aligned, 1, 1),
                       str(save_dir / "nmnist_linear_block.onnx"), opset_version=11,
                       input_names=["input"], output_names=["output"])

    torch.save({
        'conv1': ckpt['conv1'], 'conv2': ckpt['conv2'],
        'fc1': ckpt['fc1'], 'fc2': ckpt['fc2'],
        'beta': ckpt['beta'], 'threshold': ckpt['threshold'],
        'test_acc': ckpt['test_acc'], 'epoch': ckpt['epoch'],
    }, str(save_dir / "nmnist_snn_weights.pt"))

    print(f"  ONNX: {save_dir / 'nmnist_conv_block.onnx'}")
    print(f"  ONNX: {save_dir / 'nmnist_linear_block.onnx'}")
    print(f"\nCopy these files to RK3588:")
    print(f"  nmnist_snn.pt, nmnist_conv_block.onnx, nmnist_linear_block.onnx")
    print(f"Then run: python tools/convert_nmnist.py")


if __name__ == "__main__":
    main()