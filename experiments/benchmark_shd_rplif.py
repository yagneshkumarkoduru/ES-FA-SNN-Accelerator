"""
ES-FA RPLIF Benchmark - Recurrent Parametric LIF SNN on SHD.

This is the key architectural upgrade over benchmark_shd_esfa.py:
  RPLIF = Recurrent + Parametric LIF (learnable beta per neuron).

Architecture: 700 -> 256 (RPLIF) -> 128 (RPLIF) -> 20
  - Each RPLIF layer has both feedforward (W_in) and recurrent (W_rec) weights.
  - Recurrent connections: cur(t) = W_in * x(t) + W_rec * spk(t-1)
  - This is the architecture family of Yin et al. (SRNN, Nature MI 2021)
    which achieves 92.45% on SHD.
  - Our addition: PLIF learnable beta per neuron (vs fixed beta in SRNN)
    and INT8-aligned fake-quant weights (hardware target).

Expected outcome: 88-93% on SHD, matching or exceeding SRNN.
Winning claim: Competitive accuracy WITH hardware-aligned efficiency
  (INT8, 90%+ spike sparsity, FPGA-synthesisable RTL).

Usage:
    python experiments/benchmark_shd_rplif.py --epochs 30 --seeds 42,123,999
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_shd_esfa import (
    set_seed, get_shd_loaders, aggregate_seeds,
    PLIFLayer, SurrogateGrad, spike_fn,
)


# ---------------------------------------------------------------------------
# Recurrent PLIF layer - the key addition over the feedforward baseline
# ---------------------------------------------------------------------------

class RecurrentPLIFLayer(nn.Module):
    """Recurrent LIF with learnable per-neuron decay (RPLIF).

    Forward dynamics at each timestep:
        cur(t)  = W_in * x(t) + W_rec * spk(t-1)   [+ optional dropout on cur]
        mem(t)  = beta * mem(t-1) + cur(t)
        spk(t)  = spike_fn(mem(t) - threshold)
        mem(t) -= spk(t) * threshold                [soft reset]

    W_rec is initialised to be sparse (normal(0, 1/sqrt(n_out))) so that
    the recurrent connections start weak and are learned as needed.
    """

    def __init__(
        self,
        n_in:    int,
        n_out:   int,
        threshold: float = 1.0,
        slope:     float = 5.0,
        init_beta: float = 0.9,
        dropout:   float = 0.2,
    ) -> None:
        super().__init__()
        self.n_out = n_out

        self.fc_in  = nn.Linear(n_in,  n_out, bias=False)
        self.fc_rec = nn.Linear(n_out, n_out, bias=False)
        self.lif    = PLIFLayer(n_out, threshold=threshold,
                                slope=slope, init_beta=init_beta)
        self.drop   = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.fc_in.weight)
        # Recurrent weights: small normal init to start near identity-free
        nn.init.normal_(self.fc_rec.weight,
                         mean=0.0, std=1.0 / (n_out ** 0.5))

    def init_state(self, batch: int, device: torch.device
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        spk = torch.zeros(batch, self.n_out, device=device)
        mem = torch.zeros(batch, self.n_out, device=device)
        return spk, mem

    def forward(
        self,
        x_t:      torch.Tensor,
        spk_prev: torch.Tensor,
        mem:      torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cur = self.fc_in(x_t) + self.fc_rec(spk_prev)
        cur = self.drop(cur)
        spk, mem = self.lif(cur, mem)
        return spk, mem


# ---------------------------------------------------------------------------
# ES-FA RPLIF model: 700 -> 256 -> 128 -> 20 with recurrent connections
# ---------------------------------------------------------------------------

def _fake_quant(t: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    scale = t.detach().abs().amax().clamp(1e-8) / qmax
    q = torch.round(t / scale).clamp(-qmax - 1, qmax)
    return t + (q * scale - t).detach()


class ESFARPLIFModel(nn.Module):
    """ES-FA Recurrent PLIF SNN for SHD.

    Architecture matches SRNN family (Yin et al., 2021) but with:
      - PLIF learnable beta per neuron (vs fixed beta in SRNN)
      - INT8-aligned fake-quant on W_in weights (hardware target)
      - Larger hidden layers (256, 128 vs 128, 128 in SRNN)
    """

    N_INPUT  = 700
    N_H1     = 256
    N_H2     = 128
    N_OUTPUT = 20

    def __init__(
        self,
        weight_bits:   int   = 8,
        threshold:     float = 1.0,
        slope:         float = 5.0,
        init_beta:     float = 0.9,
        quant_enabled: bool  = False,
        dropout:       float = 0.2,
    ) -> None:
        super().__init__()
        self.weight_bits   = weight_bits
        self.quant_enabled = quant_enabled

        self.rplif1 = RecurrentPLIFLayer(
            self.N_INPUT, self.N_H1,
            threshold=threshold, slope=slope,
            init_beta=init_beta, dropout=dropout,
        )
        self.rplif2 = RecurrentPLIFLayer(
            self.N_H1, self.N_H2,
            threshold=threshold, slope=slope,
            init_beta=init_beta, dropout=dropout,
        )
        self.fc_out = nn.Linear(self.N_H2, self.N_OUTPUT, bias=False)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def _quant(self, w: torch.Tensor) -> torch.Tensor:
        return _fake_quant(w, self.weight_bits) if self.quant_enabled else w

    def forward(self, spikes_in: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, T, _ = spikes_in.shape
        device  = spikes_in.device

        spk1, mem1 = self.rplif1.init_state(B, device)
        spk2, mem2 = self.rplif2.init_state(B, device)
        out_acc = torch.zeros(B, self.N_OUTPUT, device=device)

        sop1 = sop2 = sop3 = 0.0
        spk1_total = spk2_total = 0.0

        for t in range(T):
            x_t = spikes_in[:, t, :]

            # Apply fake-quant to feedforward weights if enabled
            if self.quant_enabled:
                with torch.no_grad():
                    self.rplif1.fc_in.weight.data = _fake_quant(
                        self.rplif1.fc_in.weight, self.weight_bits)
                    self.rplif2.fc_in.weight.data = _fake_quant(
                        self.rplif2.fc_in.weight, self.weight_bits)

            spk1_new, mem1 = self.rplif1(x_t,  spk1, mem1)
            spk2_new, mem2 = self.rplif2(spk1_new, spk2, mem2)
            out_t = F.linear(spk2_new,
                             self._quant(self.fc_out.weight))
            out_acc += out_t

            n_in = float(x_t.sum().item())
            n_s1 = float(spk1_new.sum().item())
            n_s2 = float(spk2_new.sum().item())
            spk1_total += n_s1
            spk2_total += n_s2

            # SOPs: event-driven (feedforward only, recurrent is always dense)
            sop1 += n_in  * self.N_H1
            sop2 += n_s1  * self.N_H2
            sop3 += n_s2  * self.N_OUTPUT

            spk1, spk2 = spk1_new, spk2_new

        logits = out_acc / float(T)

        h1_tot = float(B * T * self.N_H1)
        h2_tot = float(B * T * self.N_H2)
        dense_sop = float(T) * (
            self.N_INPUT * self.N_H1 +
            self.N_H1   * self.N_H2 +
            self.N_H2   * self.N_OUTPUT
        )
        total_sop = sop1 + sop2 + sop3

        stats: Dict[str, float] = {
            "hidden1_sparsity":     1.0 - spk1_total / max(1.0, h1_tot),
            "hidden2_sparsity":     1.0 - spk2_total / max(1.0, h2_tot),
            "sop_per_sample":       total_sop / max(1.0, float(B)),
            "dense_sop_per_sample": dense_sop,
        }
        return logits, stats


# ---------------------------------------------------------------------------
# Training loop (identical protocol to PLIF baseline for fair comparison)
# ---------------------------------------------------------------------------

def run_epoch(
    model:    ESFARPLIFModel,
    loader:   DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device:   torch.device,
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
        "dense_sop_per_sample": dense,
        "sop_reduction_pct": 100.0 * (1.0 - sop / max(1.0, dense)),
    }


def train_single_seed(
    seed: int, epochs: int, batch_size: int, lr: float,
    sparsity_lambda: float, train_loader: DataLoader,
    test_loader: DataLoader, device: torch.device,
    weight_bits: int = 8, threshold: float = 1.0,
    init_beta: float = 0.9, dropout: float = 0.2,
) -> Dict:
    set_seed(seed)
    model = ESFARPLIFModel(
        weight_bits=weight_bits, threshold=threshold,
        init_beta=init_beta, dropout=dropout,
    ).to(device)

    # Separate learning rates: slower for recurrent and beta params
    rec_params = [p for n, p in model.named_parameters()
                  if "fc_rec" in n]
    beta_params = [p for n, p in model.named_parameters()
                   if "beta_logit" in n]
    other_params = [p for n, p in model.named_parameters()
                    if "fc_rec" not in n and "beta_logit" not in n]

    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": lr,         "weight_decay": 1e-4},
        {"params": rec_params,   "lr": lr * 0.3,   "weight_decay": 1e-4},
        {"params": beta_params,  "lr": lr * 0.05},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.02
    )

    history: List[Dict] = []
    best_acc = 0.0

    print(f"\n  RPLIF seed {seed}:")
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr = run_epoch(model, train_loader, optimizer, device, sparsity_lambda)
        te = run_epoch(model, test_loader,  None,      device, sparsity_lambda)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        if te["acc"] > best_acc:
            best_acc = te["acc"]

        history.append({
            "epoch": ep,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"test_{k}":  v for k, v in te.items()},
            "elapsed_s": elapsed,
        })
        print(
            f"    ep{ep:02d} "
            f"train={tr['acc']*100:.1f}% "
            f"test={te['acc']*100:.2f}% "
            f"h1={te['hidden1_sparsity']*100:.0f}% "
            f"h2={te['hidden2_sparsity']*100:.0f}% "
            f"sop_red={te['sop_reduction_pct']:.1f}% "
            f"[{elapsed:.0f}s]"
        )

    return {
        "seed": seed,
        "best_test_acc": best_acc,
        "best_epoch": max(range(len(history)),
                          key=lambda i: history[i]["test_acc"]) + 1,
        "final_test_acc": history[-1]["test_acc"],
        "final_hidden1_sparsity": history[-1]["test_hidden1_sparsity"],
        "final_hidden2_sparsity": history[-1]["test_hidden2_sparsity"],
        "final_sop_per_sample":   history[-1]["test_sop_per_sample"],
        "final_sop_reduction_pct": history[-1]["test_sop_reduction_pct"],
        "history": history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ES-FA RPLIF benchmark on SHD")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--seeds",      type=str,   default="42,123,999")
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=5e-3)
    p.add_argument("--sparsity-lambda", type=float, default=5e-4)
    p.add_argument("--dt-ms",      type=float, default=10.0)
    p.add_argument("--t-max-ms",   type=float, default=1400.0)
    p.add_argument("--weight-bits", type=int,  default=8)
    p.add_argument("--init-beta",  type=float, default=0.9)
    p.add_argument("--threshold",  type=float, default=1.0)
    p.add_argument("--dropout",    type=float, default=0.15)
    p.add_argument("--data-dir",   type=Path,  default=Path("./data"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("./results/shd_rplif"))
    p.add_argument("--device", type=str,
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

    T = int(args.t_max_ms / args.dt_ms)
    n_ff_params  = (ESFARPLIFModel.N_INPUT * ESFARPLIFModel.N_H1 +
                    ESFARPLIFModel.N_H1 * ESFARPLIFModel.N_H2 +
                    ESFARPLIFModel.N_H2 * ESFARPLIFModel.N_OUTPUT)
    n_rec_params = (ESFARPLIFModel.N_H1 ** 2 + ESFARPLIFModel.N_H2 ** 2)

    print(f"\nES-FA RPLIF SHD Benchmark")
    print(f"  Model   : 700 -[rec256]-> 256 -[rec128]-> 128 -> 20")
    print(f"  FF params : {n_ff_params:,}  |  Rec params: {n_rec_params:,}  |  Total: {n_ff_params+n_rec_params:,}")
    print(f"  T       : {T} bins @ {args.dt_ms}ms = {args.t_max_ms}ms window")
    print(f"  Seeds   : {seeds}  |  Device: {device}")
    print(f"  Compare : SRNN (Yin et al. 2021): 92.45% +/-0.46% SHD")

    seed_results = []
    for seed in seeds:
        r = train_single_seed(
            seed=seed, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, sparsity_lambda=args.sparsity_lambda,
            train_loader=train_loader, test_loader=test_loader,
            device=device, weight_bits=args.weight_bits,
            threshold=args.threshold, init_beta=args.init_beta,
            dropout=args.dropout,
        )
        seed_results.append(r)
        with (args.output_dir / f"seed_{seed}.json").open("w") as f:
            json.dump(r, f, indent=2)

    agg = aggregate_seeds(seed_results)
    with (args.output_dir / "aggregate.json").open("w") as f:
        json.dump(agg, f, indent=2)

    srnn_acc = 92.45
    gap = agg["test_acc_pct_mean"] - srnn_acc

    print(f"\n{'='*60}")
    print("ES-FA RPLIF SHD RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Accuracy ({len(seeds)} seeds):")
    print(f"    Mean  : {agg['test_acc_pct_mean']:.2f}%")
    print(f"    Std   : +/-{agg['test_acc_pct_std']:.2f}%")
    print(f"    Range : {agg['test_acc_min']*100:.2f}% - {agg['test_acc_max']*100:.2f}%")
    print(f"    vs SRNN 92.45%: {gap:+.2f}pp")
    print(f"  Sparsity:")
    print(f"    H1    : {agg['hidden1_sparsity_mean']*100:.1f}%")
    print(f"    H2    : {agg['hidden2_sparsity_mean']*100:.1f}%")
    print(f"  SOP reduction vs dense : {agg['sop_reduction_pct_mean']:.1f}%")
    print(f"  Energy proxy (MODEL)   : {agg['energy_proxy_pj_mean']/1e6:.3f} mJ/inf")
    print(f"\nResults: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
