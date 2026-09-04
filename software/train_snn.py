"""Train a hardware-aware SNN: 784 -> 128 -> 64 -> 10.

Features:
- LIF neurons with surrogate gradient training
- Rate encoding of static images
- QAT-style fake quantization (INT8-friendly)
- Spike sparsity and hardware-cost estimation
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from quantization import FakeQuantLinear


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SurrogateSpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.slope = slope
        return (input_tensor > 0.0).to(input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (input_tensor,) = ctx.saved_tensors
        slope = ctx.slope
        grad = grad_output / (1.0 + slope * input_tensor.abs()).pow(2)
        return grad, None


def surrogate_spike(x: torch.Tensor, slope: float) -> torch.Tensor:
    return SurrogateSpikeFn.apply(x, slope)


@dataclass
class BatchStats:
    input_spikes: float = 0.0
    layer1_spikes: float = 0.0
    layer2_spikes: float = 0.0
    dense_weight_accesses: float = 0.0
    event_weight_accesses: float = 0.0
    event_neuron_accesses: float = 0.0
    total_samples: int = 0
    total_steps: int = 0

    def update(self, other: "BatchStats") -> None:
        self.input_spikes += other.input_spikes
        self.layer1_spikes += other.layer1_spikes
        self.layer2_spikes += other.layer2_spikes
        self.dense_weight_accesses += other.dense_weight_accesses
        self.event_weight_accesses += other.event_weight_accesses
        self.event_neuron_accesses += other.event_neuron_accesses
        self.total_samples += other.total_samples
        self.total_steps += other.total_steps

    def to_metrics(self) -> Dict[str, float]:
        denom_input = max(1.0, float(self.total_samples * self.total_steps * 784))
        denom_l1 = max(1.0, float(self.total_samples * self.total_steps * 128))
        denom_l2 = max(1.0, float(self.total_samples * self.total_steps * 64))
        dense = max(1.0, self.dense_weight_accesses)
        return {
            "input_spike_density": self.input_spikes / denom_input,
            "layer1_spike_density": self.layer1_spikes / denom_l1,
            "layer2_spike_density": self.layer2_spikes / denom_l2,
            "input_spike_sparsity": 1.0 - (self.input_spikes / denom_input),
            "layer1_spike_sparsity": 1.0 - (self.layer1_spikes / denom_l1),
            "layer2_spike_sparsity": 1.0 - (self.layer2_spikes / denom_l2),
            "dense_weight_accesses": self.dense_weight_accesses,
            "event_weight_accesses": self.event_weight_accesses,
            "event_neuron_accesses": self.event_neuron_accesses,
            "weight_access_reduction_pct": 100.0 * (1.0 - (self.event_weight_accesses / dense)),
        }


class LIFLayer(nn.Module):
    def __init__(self, size: int, beta: float = 0.9, threshold: float = 1.0, surrogate_slope: float = 5.0) -> None:
        super().__init__()
        self.size = size
        self.beta = beta
        self.threshold = threshold
        self.surrogate_slope = surrogate_slope

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.size, device=device)

    def forward(self, current: torch.Tensor, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mem = self.beta * mem + current
        spk = surrogate_spike(mem - self.threshold, self.surrogate_slope)
        # Soft reset supports gradient flow and fixed-point style subtraction in hardware.
        mem = mem - spk * self.threshold
        return spk, mem


class HardwareAwareSNN(nn.Module):
    def __init__(
        self,
        time_steps: int = 16,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_slope: float = 5.0,
        weight_bits: int = 8,
        act_bits: int = 8,
    ) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.fc1 = FakeQuantLinear(784, 128, weight_bits=weight_bits, act_bits=act_bits)
        self.fc2 = FakeQuantLinear(128, 64, weight_bits=weight_bits, act_bits=act_bits)
        self.fc3 = FakeQuantLinear(64, 10, weight_bits=weight_bits, act_bits=act_bits)
        self.lif1 = LIFLayer(128, beta=beta, threshold=threshold, surrogate_slope=surrogate_slope)
        self.lif2 = LIFLayer(64, beta=beta, threshold=threshold, surrogate_slope=surrogate_slope)

    def rate_encode(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.rand_like(x) < x).to(x.dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, BatchStats]:
        x = x.view(x.size(0), -1).clamp(0.0, 1.0)
        batch_size = x.size(0)
        device = x.device

        mem1 = self.lif1.init_state(batch_size, device)
        mem2 = self.lif2.init_state(batch_size, device)
        out_acc = torch.zeros(batch_size, 10, device=device)

        stats = BatchStats(total_samples=batch_size, total_steps=self.time_steps)

        dense_per_step = batch_size * (784 * 128 + 128 * 64 + 64 * 10)

        for _ in range(self.time_steps):
            x_spk = self.rate_encode(x)
            cur1 = self.fc1(x_spk)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out_acc = out_acc + self.fc3(spk2)

            in_spk_count = float(x_spk.sum().item())
            l1_spk_count = float(spk1.sum().item())
            l2_spk_count = float(spk2.sum().item())

            stats.input_spikes += in_spk_count
            stats.layer1_spikes += l1_spk_count
            stats.layer2_spikes += l2_spk_count
            stats.dense_weight_accesses += dense_per_step

            # Event-driven estimate:
            # active_pre * fanout for each layer.
            event_w = in_spk_count * 128.0 + l1_spk_count * 64.0 + l2_spk_count * 10.0
            stats.event_weight_accesses += event_w
            stats.event_neuron_accesses += 2.0 * event_w  # one read + one write for membrane state.

        logits = out_acc / float(self.time_steps)
        return logits, stats


def get_loaders(data_dir: Path, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader


def run_epoch(
    model: HardwareAwareSNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    sparsity_lambda: float,
) -> Tuple[float, float, Dict[str, float]]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    agg_stats = BatchStats()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        logits, batch_stats = model(images)
        ce_loss = F.cross_entropy(logits, labels)

        spike_reg = (batch_stats.layer1_spikes + batch_stats.layer2_spikes) / max(
            1.0, float(batch_stats.total_samples * batch_stats.total_steps * (128 + 64))
        )
        loss = ce_loss + sparsity_lambda * spike_reg

        if train_mode:
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        total_seen += labels.numel()
        total_loss += float(loss.item()) * labels.size(0)
        agg_stats.update(batch_stats)

    avg_loss = total_loss / max(1, total_seen)
    acc = float(total_correct) / max(1, total_seen)
    return avg_loss, acc, agg_stats.to_metrics()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hardware-aware SNN (MNIST).")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--time-steps", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--surrogate-slope", type=float, default=5.0)
    p.add_argument("--sparsity-lambda", type=float, default=1e-3)
    p.add_argument("--weight-bits", type=int, default=8)
    p.add_argument("--act-bits", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-dir", type=Path, default=Path("./data"))
    p.add_argument("--output-dir", type=Path, default=Path("./artifacts"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    output_dir = args.output_dir
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.num_workers)

    model = HardwareAwareSNN(
        time_steps=args.time_steps,
        beta=args.beta,
        threshold=args.threshold,
        surrogate_slope=args.surrogate_slope,
        weight_bits=args.weight_bits,
        act_bits=args.act_bits,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_acc = 0.0
    best_path = ckpt_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_stats = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            sparsity_lambda=args.sparsity_lambda,
        )

        with torch.no_grad():
            val_loss, val_acc, val_stats = run_epoch(
                model=model,
                loader=test_loader,
                optimizer=None,
                device=device,
                sparsity_lambda=args.sparsity_lambda,
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_stats": train_stats,
            "val_stats": val_stats,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}% | "
            f"val_l1_sparsity={val_stats['layer1_spike_sparsity']*100:.2f}% "
            f"val_l2_sparsity={val_stats['layer2_spike_sparsity']*100:.2f}% "
            f"weight_reduction={val_stats['weight_access_reduction_pct']:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "best_val_acc": best_acc,
                    "history": history,
                },
                best_path,
            )

    final_path = ckpt_dir / "last_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "best_val_acc": best_acc,
            "history": history,
        },
        final_path,
    )

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    final_summary = {
        "best_val_acc": best_acc,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(final_path),
        "final_epoch": history[-1] if history else {},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_acc*100:.2f}%")
    print(f"Artifacts written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
