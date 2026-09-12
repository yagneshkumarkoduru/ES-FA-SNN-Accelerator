"""
snntorch Baseline on SHD - fair external comparison for ES-FA.

Identical dataset (SHD), identical architecture shape (700->256->128->20),
identical training hyperparameters. Uses snntorch's official Leaky neuron.
This is the EXTERNAL BASELINE that ES-FA results are compared against.

Multi-seed (same 3 seeds). Saves aggregate JSON that compare_sota.py reads.

Cite (snntorch):
    Eshraghian, J. et al. "Training Spiking Neural Networks Using Lessons
    From Deep Learning." Proc. IEEE, 2023.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import snntorch as snn
from snntorch import surrogate

# Re-use the SHD loader from benchmark_shd_esfa
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_shd_esfa import (
    set_seed, get_shd_loaders, aggregate_seeds
)


# ---------------------------------------------------------------------------
# snntorch baseline model - same shape as ES-FA (700->256->128->20)
# ---------------------------------------------------------------------------

class SNNTorchBaseline(nn.Module):
    """Leaky-integrate-and-fire SNN using snntorch, matched to ES-FA shape."""

    N_INPUT  = 700
    N_H1     = 256
    N_H2     = 128
    N_OUTPUT = 20

    def __init__(self, beta: float = 0.9, threshold: float = 1.0) -> None:
        super().__init__()
        grad = surrogate.fast_sigmoid(slope=25)

        self.fc1 = nn.Linear(self.N_INPUT, self.N_H1,  bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold,
                               spike_grad=grad, learn_beta=True)

        self.fc2 = nn.Linear(self.N_H1, self.N_H2, bias=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold,
                               spike_grad=grad, learn_beta=True)

        self.fc3 = nn.Linear(self.N_H2, self.N_OUTPUT, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, spikes_in: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, T, _ = spikes_in.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        out_acc = torch.zeros(B, self.N_OUTPUT, device=spikes_in.device)

        sop1 = sop2 = sop3 = 0.0
        spk1_total = spk2_total = 0.0

        for t in range(T):
            x_t = spikes_in[:, t, :]
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out_acc += self.fc3(spk2)

            n_in = float(x_t.sum().item())
            n_s1 = float(spk1.sum().item())
            n_s2 = float(spk2.sum().item())
            spk1_total += n_s1
            spk2_total += n_s2
            sop1 += n_in * self.N_H1
            sop2 += n_s1 * self.N_H2
            sop3 += n_s2 * self.N_OUTPUT

        logits = out_acc / float(T)
        total_h1 = float(B * T * self.N_H1)
        total_h2 = float(B * T * self.N_H2)
        dense_sop = float(T) * (
            self.N_INPUT * self.N_H1 + self.N_H1 * self.N_H2 +
            self.N_H2 * self.N_OUTPUT
        )

        stats: Dict[str, float] = {
            "hidden1_sparsity":  1.0 - spk1_total / max(1.0, total_h1),
            "hidden2_sparsity":  1.0 - spk2_total / max(1.0, total_h2),
            "sop_per_sample":    (sop1 + sop2 + sop3) / max(1.0, float(B)),
            "dense_sop_per_sample": dense_sop,
        }
        return logits, stats


# ---------------------------------------------------------------------------
# Training loop (identical to ES-FA)
# ---------------------------------------------------------------------------

def run_epoch(
    model: SNNTorchBaseline,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    sparsity_lambda: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = total_correct = total_seen = 0
    agg_h1 = agg_h2 = agg_sop = agg_dense = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits, stats = model(x)
        ce  = F.cross_entropy(logits, y)
        reg = ((1.0 - stats["hidden1_sparsity"]) +
               (1.0 - stats["hidden2_sparsity"])) / 2.0
        loss = ce + sparsity_lambda * reg

        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss    += float(loss.item()) * y.size(0)
        total_correct += int((logits.argmax(1) == y).sum().item())
        total_seen    += y.size(0)
        agg_h1    += stats["hidden1_sparsity"]
        agg_h2    += stats["hidden2_sparsity"]
        agg_sop   += stats["sop_per_sample"]
        agg_dense += stats["dense_sop_per_sample"]
        n_batches += 1

    nb = max(1, n_batches)
    sop = agg_sop / nb
    dense = agg_dense / nb
    return {
        "loss": total_loss / max(1, total_seen),
        "acc":  total_correct / max(1, total_seen),
        "hidden1_sparsity": agg_h1 / nb,
        "hidden2_sparsity": agg_h2 / nb,
        "sop_per_sample":   sop,
        "sop_reduction_pct": 100.0 * (1.0 - sop / max(1.0, dense)),
    }


def train_single_seed(
    seed: int, epochs: int, batch_size: int, lr: float,
    sparsity_lambda: float, train_loader: DataLoader,
    test_loader: DataLoader, device: torch.device,
) -> Dict:
    set_seed(seed)
    model = SNNTorchBaseline().to(device)
    optimizer = torch.optim.Adam([
        {"params": [p for n, p in model.named_parameters() if "beta" not in n], "lr": lr},
        {"params": [p for n, p in model.named_parameters() if "beta" in n], "lr": lr * 0.1},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.05
    )

    history = []
    best_acc = 0.0

    print(f"\n  snntorch seed {seed}:")
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr = run_epoch(model, train_loader, optimizer, device, sparsity_lambda)
        te = run_epoch(model, test_loader,  None, device, sparsity_lambda)
        scheduler.step()

        if te["acc"] > best_acc:
            best_acc = te["acc"]

        history.append({"epoch": ep, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"test_{k}": v for k, v in te.items()},
                        "elapsed_s": time.perf_counter() - t0})

        print(f"    ep{ep:02d} train={tr['acc']*100:.1f}% test={te['acc']*100:.2f}% "
              f"sop_red={te['sop_reduction_pct']:.1f}% [{time.perf_counter()-t0:.1f}s]")

    return {
        "seed": seed,
        "best_test_acc": best_acc,
        "best_epoch": max(range(len(history)),
                          key=lambda i: history[i]["test_acc"]) + 1,
        "final_test_acc": history[-1]["test_acc"],
        "final_hidden1_sparsity": history[-1]["test_hidden1_sparsity"],
        "final_hidden2_sparsity": history[-1]["test_hidden2_sparsity"],
        "final_sop_per_sample": history[-1]["test_sop_per_sample"],
        "final_sop_reduction_pct": history[-1]["test_sop_reduction_pct"],
        "history": history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="snntorch baseline on SHD")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--seeds",      type=str,   default="42,123,999")
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=5e-3)
    p.add_argument("--sparsity-lambda", type=float, default=5e-3)
    p.add_argument("--dt-ms",      type=float, default=2.0)
    p.add_argument("--t-max-ms",   type=float, default=100.0)
    p.add_argument("--data-dir",   type=Path,  default=Path("./data"))
    p.add_argument("--output-dir", type=Path,  default=Path("./results/shd_snntorch"))
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    seeds  = [int(s.strip()) for s in args.seeds.split(",")]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_shd_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        dt_ms=args.dt_ms,
        t_max_ms=args.t_max_ms,
    )

    print(f"\nsnntorch Baseline SHD Benchmark")
    print(f"  Model : 700 -> 256 -> 128 -> 20 snntorch Leaky (matched to ES-FA shape)")
    print(f"  Seeds : {seeds} | Device: {device}")

    seed_results = []
    for seed in seeds:
        r = train_single_seed(seed, args.epochs, args.batch_size,
                              args.lr, args.sparsity_lambda,
                              train_loader, test_loader, device)
        seed_results.append(r)
        with (args.output_dir / f"seed_{seed}.json").open("w") as f:
            json.dump(r, f, indent=2)

    agg = aggregate_seeds(seed_results)
    with (args.output_dir / "aggregate.json").open("w") as f:
        json.dump(agg, f, indent=2)

    print(f"\nsnntorch baseline: {agg['test_acc_pct_mean']:.2f}% "
          f"±{agg['test_acc_pct_std']:.2f}%")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
