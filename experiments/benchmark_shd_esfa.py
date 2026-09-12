"""
ES-FA SHD Benchmark - Spiking Heidelberg Digits (SHD) training and evaluation.

Replaces the toy MNIST baseline with SHD, the standard temporal SNN benchmark.

Architecture: 700 -> 256 -> 128 -> 20 PLIF-SNN (Parametric LIF with learnable decay).
  - Learnable membrane decay (beta) per neuron - outperforms fixed beta
  - Sparsity regularisation matching ES-FA hardware ABI
  - INT8-aligned fake-quant weights
  - SOP counter for hardware energy proxy (4.43 pJ/SOP from C-engine model)

Multi-seed (3 seeds). Saves per-seed JSON + aggregate JSON.
Uses SHD T=50 @ 2ms bins (100ms window, matches cochlear filter timescale).

Usage:
    python experiments/benchmark_shd_esfa.py --epochs 20 --seeds 42,123,999

Cite (SHD dataset):
    Cramer, B. et al. "The Heidelberg Spiking Data Sets for the Systematic
    Evaluation of Spiking Neural Networks." IEEE TNNLS, 2022.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# SHD data loader (h5py, no expelliarmus needed)
# ---------------------------------------------------------------------------

def load_shd_h5(path: Path, n_input: int = 700, dt_ms: float = 2.0,
                t_max_ms: float = 100.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load SHD HDF5 file into dense spike tensors.

    Returns:
        spikes: (N, T, n_input) float32 spike raster
        labels: (N,) int64 class labels
    """
    import h5py

    T = int(t_max_ms / dt_ms)  # number of time bins

    with h5py.File(path, "r") as f:
        spike_times_list = list(f["spikes"]["times"])
        spike_units_list = list(f["spikes"]["units"])
        labels_raw = np.array(f["labels"], dtype=np.int64)

    n_samples = len(labels_raw)
    spikes = np.zeros((n_samples, T, n_input), dtype=np.float32)

    for i in range(n_samples):
        ts = np.array(spike_times_list[i], dtype=np.float32)
        us = np.array(spike_units_list[i], dtype=np.int64)
        # convert seconds -> ms -> bin index
        bin_idx = np.floor(ts * 1000.0 / dt_ms).astype(np.int64)
        mask = (bin_idx >= 0) & (bin_idx < T) & (us >= 0) & (us < n_input)
        bin_idx = bin_idx[mask]
        unit_idx = us[mask]
        spikes[i, bin_idx, unit_idx] = 1.0

    return torch.from_numpy(spikes), torch.from_numpy(labels_raw)


def get_shd_loaders(data_dir: Path, batch_size: int,
                    dt_ms: float = 2.0, t_max_ms: float = 100.0,
                    seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """Download (via tonic) and load SHD train/test sets."""
    import tonic
    import tonic.transforms as ttr

    data_dir.mkdir(parents=True, exist_ok=True)

    # Use tonic to download; it handles the URL and gz extraction.
    train_ds_raw = tonic.datasets.SHD(save_to=str(data_dir), train=True)
    test_ds_raw  = tonic.datasets.SHD(save_to=str(data_dir), train=False)

    # Extract HDF5 path that tonic downloaded
    # tonic stores files at data_dir/SHD/
    shd_dir = data_dir / "SHD"
    train_h5 = shd_dir / "shd_train.h5"
    test_h5  = shd_dir / "shd_test.h5"

    if not train_h5.exists():
        # Fallback: look for any .h5 files
        h5_files = list(shd_dir.glob("*.h5")) + list(shd_dir.glob("**/*.h5"))
        if len(h5_files) >= 2:
            h5_files = sorted(h5_files)
            train_h5 = h5_files[0]
            test_h5  = h5_files[1]
        else:
            raise FileNotFoundError(
                f"SHD HDF5 files not found in {shd_dir}. "
                "Run with --download-only first."
            )

    print(f"Loading SHD from {shd_dir} (T={int(t_max_ms/dt_ms)} bins @ {dt_ms}ms)...")
    t0 = time.perf_counter()
    x_train, y_train = load_shd_h5(train_h5, dt_ms=dt_ms, t_max_ms=t_max_ms)
    x_test,  y_test  = load_shd_h5(test_h5,  dt_ms=dt_ms, t_max_ms=t_max_ms)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s | "
          f"train={len(x_train)} test={len(x_test)}")

    from torch.utils.data import TensorDataset
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True,
        generator=g, drop_last=False
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Parametric LIF (PLIF) - learnable membrane decay
# ---------------------------------------------------------------------------

class SurrogateGrad(torch.autograd.Function):
    """Fast sigmoid surrogate gradient (ATan-like)."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x >= 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (x,) = ctx.saved_tensors
        return grad / (1.0 + ctx.slope * x.abs()).pow(2), None


def spike_fn(x: torch.Tensor, slope: float = 5.0) -> torch.Tensor:
    return SurrogateGrad.apply(x, slope)


class PLIFLayer(nn.Module):
    """Parametric LIF: beta = sigmoid(beta_logit) is learned per-neuron."""

    def __init__(self, n: int, threshold: float = 1.0, slope: float = 5.0,
                 init_beta: float = 0.9) -> None:
        super().__init__()
        self.n = n
        self.threshold = threshold
        self.slope = slope
        # inverse sigmoid: logit(0.9) ≈ 2.197
        init_logit = float(np.log(init_beta / (1.0 - init_beta)))
        self.beta_logit = nn.Parameter(torch.full((n,), init_logit))

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_logit)

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.n, device=device)

    def forward(self, cur: torch.Tensor, mem: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        mem = self.beta.unsqueeze(0) * mem + cur
        spk = spike_fn(mem - self.threshold, self.slope)
        mem = mem - spk * self.threshold   # soft reset
        return spk, mem


# ---------------------------------------------------------------------------
# ES-FA SHD model: 700 -> 256 -> 128 -> 20
# ---------------------------------------------------------------------------

def _fake_quant(t: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    scale = t.detach().abs().amax().clamp(1e-8) / qmax
    q = torch.round(t / scale).clamp(-qmax - 1, qmax)
    return t + (q * scale - t).detach()


class ESFASHDModel(nn.Module):
    """ES-FA SNN for SHD: PLIF hidden layers, INT8-aligned weights, dropout."""

    N_INPUT  = 700
    N_H1     = 256
    N_H2     = 128
    N_OUTPUT = 20

    def __init__(
        self,
        weight_bits: int = 8,
        threshold: float  = 1.0,
        slope: float      = 5.0,
        init_beta: float  = 0.9,
        quant_enabled: bool = False,
        dropout: float    = 0.2,
    ) -> None:
        super().__init__()
        self.weight_bits   = weight_bits
        self.quant_enabled = quant_enabled

        self.fc1 = nn.Linear(self.N_INPUT,  self.N_H1,     bias=False)
        self.fc2 = nn.Linear(self.N_H1,     self.N_H2,     bias=False)
        self.fc3 = nn.Linear(self.N_H2,     self.N_OUTPUT, bias=False)

        self.lif1 = PLIFLayer(self.N_H1, threshold=threshold,
                               slope=slope, init_beta=init_beta)
        self.lif2 = PLIFLayer(self.N_H2, threshold=threshold,
                               slope=slope, init_beta=init_beta)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def _linear(self, layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        if self.quant_enabled:
            w = _fake_quant(layer.weight, self.weight_bits)
            return F.linear(x, w)
        return layer(x)

    def forward(self, spikes_in: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            spikes_in: (B, T, 700) float32 spike raster
        Returns:
            logits: (B, 20)  – accumulated output, divided by T
            stats:  dict with SOP counts and sparsity for energy model
        """
        B, T, _ = spikes_in.shape
        device   = spikes_in.device

        mem1 = self.lif1.init_state(B, device)
        mem2 = self.lif2.init_state(B, device)
        out_acc = torch.zeros(B, self.N_OUTPUT, device=device)

        sop1 = sop2 = sop3 = 0.0
        spk_in_total = spk1_total = spk2_total = 0.0

        for t in range(T):
            x_t = spikes_in[:, t, :]
            cur1 = self._linear(self.fc1, x_t)
            cur1 = self.drop1(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self._linear(self.fc2, spk1)
            cur2 = self.drop2(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            out_acc += self._linear(self.fc3, spk2)

            n_in  = float(x_t.sum().item())
            n_s1  = float(spk1.sum().item())
            n_s2  = float(spk2.sum().item())
            spk_in_total += n_in
            spk1_total   += n_s1
            spk2_total   += n_s2

            # SOPs = active pre-synaptic neurons × fan-out
            sop1 += n_in  * self.N_H1
            sop2 += n_s1  * self.N_H2
            sop3 += n_s2  * self.N_OUTPUT

        logits = out_acc / float(T)

        total_possible_in = float(B * T * self.N_INPUT)
        total_possible_h1 = float(B * T * self.N_H1)
        total_possible_h2 = float(B * T * self.N_H2)

        stats: Dict[str, float] = {
            # Sparsity metrics
            "input_sparsity":  1.0 - spk_in_total / max(1.0, total_possible_in),
            "hidden1_sparsity": 1.0 - spk1_total  / max(1.0, total_possible_h1),
            "hidden2_sparsity": 1.0 - spk2_total  / max(1.0, total_possible_h2),
            # SOP counts (total over batch)
            "sop_total": sop1 + sop2 + sop3,
            "sop_per_sample": (sop1 + sop2 + sop3) / max(1.0, float(B)),
            # Dense equivalent (all neurons active every step)
            "dense_sop_per_sample": float(T) * (
                self.N_INPUT * self.N_H1 +
                self.N_H1   * self.N_H2 +
                self.N_H2   * self.N_OUTPUT
            ),
        }
        return logits, stats


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class EpochResult:
    loss: float
    acc: float
    input_sparsity: float
    hidden1_sparsity: float
    hidden2_sparsity: float
    sop_per_sample: float
    dense_sop_per_sample: float
    sop_reduction_pct: float


def run_epoch(
    model: ESFASHDModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    sparsity_lambda: float,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)

    total_loss = total_correct = total_seen = 0
    agg: Dict[str, float] = {
        "input_sparsity": 0.0, "hidden1_sparsity": 0.0,
        "hidden2_sparsity": 0.0, "sop_per_sample": 0.0,
        "dense_sop_per_sample": 0.0,
    }
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits, stats = model(x)
        ce = F.cross_entropy(logits, y)

        # Sparsity regularisation: penalise dense firing (encourage sparsity)
        spike_density = (1.0 - stats["hidden1_sparsity"] +
                         1.0 - stats["hidden2_sparsity"]) / 2.0
        loss = ce + sparsity_lambda * spike_density

        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss    += float(loss.item()) * y.size(0)
        total_correct += int((logits.argmax(1) == y).sum().item())
        total_seen    += y.size(0)

        for k in agg:
            agg[k] += stats.get(k, 0.0)
        n_batches += 1

    nb = max(1, n_batches)
    sop   = agg["sop_per_sample"]       / nb
    dense = agg["dense_sop_per_sample"] / nb

    return EpochResult(
        loss             = total_loss / max(1, total_seen),
        acc              = total_correct / max(1, total_seen),
        input_sparsity   = agg["input_sparsity"]   / nb,
        hidden1_sparsity = agg["hidden1_sparsity"]  / nb,
        hidden2_sparsity = agg["hidden2_sparsity"]  / nb,
        sop_per_sample   = sop,
        dense_sop_per_sample = dense,
        sop_reduction_pct   = 100.0 * (1.0 - sop / max(1.0, dense)),
    )


# ---------------------------------------------------------------------------
# Per-seed training + evaluation
# ---------------------------------------------------------------------------

def train_single_seed(
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    sparsity_lambda: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    weight_bits: int = 8,
    threshold: float = 1.0,
    init_beta: float = 0.9,
) -> Dict:
    set_seed(seed)

    model = ESFASHDModel(
        weight_bits=weight_bits,
        threshold=threshold,
        init_beta=init_beta,
        dropout=0.2,
    ).to(device)

    # Separate LR for decay parameters vs weights
    optimizer = torch.optim.Adam([
        {"params": [p for n, p in model.named_parameters()
                    if "beta_logit" not in n], "lr": lr, "weight_decay": 1e-4},
        {"params": [p for n, p in model.named_parameters()
                    if "beta_logit" in n],    "lr": lr * 0.1},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.05
    )

    history: List[Dict] = []
    best_acc  = 0.0
    best_epoch = 0

    print(f"\n  Seed {seed}:")
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr = run_epoch(model, train_loader, optimizer, device, sparsity_lambda)
        te = run_epoch(model, test_loader,  None,      device, sparsity_lambda)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        if te.acc > best_acc:
            best_acc   = te.acc
            best_epoch = ep

        row = {
            "epoch": ep,
            "train_loss": tr.loss, "train_acc": tr.acc,
            "test_loss":  te.loss, "test_acc":  te.acc,
            "input_sparsity":   te.input_sparsity,
            "hidden1_sparsity": te.hidden1_sparsity,
            "hidden2_sparsity": te.hidden2_sparsity,
            "sop_per_sample":      te.sop_per_sample,
            "dense_sop_per_sample": te.dense_sop_per_sample,
            "sop_reduction_pct":    te.sop_reduction_pct,
            "elapsed_s": elapsed,
        }
        history.append(row)

        print(
            f"    ep{ep:02d} "
            f"train={tr.acc*100:.1f}% "
            f"test={te.acc*100:.2f}% "
            f"h1_spar={te.hidden1_sparsity*100:.1f}% "
            f"h2_spar={te.hidden2_sparsity*100:.1f}% "
            f"sop_red={te.sop_reduction_pct:.1f}% "
            f"[{elapsed:.1f}s]"
        )

    return {
        "seed": seed,
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "final_test_acc": history[-1]["test_acc"],
        "final_hidden1_sparsity": history[-1]["hidden1_sparsity"],
        "final_hidden2_sparsity": history[-1]["hidden2_sparsity"],
        "final_sop_per_sample": history[-1]["sop_per_sample"],
        "final_sop_reduction_pct": history[-1]["sop_reduction_pct"],
        "history": history,
    }


# ---------------------------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------------------------

def aggregate_seeds(results: List[Dict]) -> Dict:
    accs = [r["best_test_acc"] for r in results]
    sops = [r["final_sop_per_sample"] for r in results]
    sop_reds = [r["final_sop_reduction_pct"] for r in results]
    h1_spar  = [r["final_hidden1_sparsity"] for r in results]
    h2_spar  = [r["final_hidden2_sparsity"] for r in results]

    # Energy proxy: 4.43 pJ/SOP (from ES-FA C-engine model, c_benchmark_results.json)
    PJ_PER_SOP = 4.43
    energy_pj   = [s * PJ_PER_SOP for s in sops]

    return {
        "n_seeds": len(results),
        "seeds": [r["seed"] for r in results],

        # Accuracy (primary metric)
        "test_acc_mean":   float(np.mean(accs)),
        "test_acc_std":    float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "test_acc_max":    float(np.max(accs)),
        "test_acc_min":    float(np.min(accs)),
        "test_acc_pct_mean": float(np.mean(accs)) * 100.0,
        "test_acc_pct_std":  float(np.std(accs, ddof=1)) * 100.0 if len(accs) > 1 else 0.0,

        # Energy / SOP (key hardware metric)
        "sop_per_sample_mean":  float(np.mean(sops)),
        "sop_per_sample_std":   float(np.std(sops, ddof=1)) if len(sops) > 1 else 0.0,
        "sop_reduction_pct_mean": float(np.mean(sop_reds)),
        "energy_proxy_pj_mean": float(np.mean(energy_pj)),   # MODEL, not board
        "energy_proxy_pj_std":  float(np.std(energy_pj, ddof=1)) if len(energy_pj) > 1 else 0.0,

        # Sparsity
        "hidden1_sparsity_mean": float(np.mean(h1_spar)),
        "hidden2_sparsity_mean": float(np.mean(h2_spar)),

        # Provenance
        "energy_model_note": (
            "Energy proxy = SOP_per_inference × 4.43 pJ/SOP "
            "(4.43 pJ/SOP from ES-FA C-engine event-mode model, "
            "c_engine/c_benchmark_results.json). "
            "NOT a board measurement. Label as MODEL in any paper."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ES-FA benchmark on SHD")
    p.add_argument("--epochs",    type=int,   default=20)
    p.add_argument("--seeds",     type=str,   default="42,123,999",
                   help="Comma-separated seed list")
    p.add_argument("--batch-size", type=int,  default=128)
    p.add_argument("--lr",         type=float, default=5e-3)
    p.add_argument("--sparsity-lambda", type=float, default=5e-3)
    p.add_argument("--dt-ms",      type=float, default=2.0,
                   help="Time bin width in ms (2ms = 50 bins per 100ms)")
    p.add_argument("--t-max-ms",   type=float, default=100.0,
                   help="Window length in ms")
    p.add_argument("--weight-bits", type=int, default=8)
    p.add_argument("--init-beta",   type=float, default=0.9)
    p.add_argument("--threshold",   type=float, default=1.0)
    p.add_argument("--data-dir",   type=Path, default=Path("./data"))
    p.add_argument("--output-dir", type=Path, default=Path("./results/shd_esfa"))
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--download-only", action="store_true",
                   help="Just download the dataset and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    seeds  = [int(s.strip()) for s in args.seeds.split(",")]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset ----
    train_loader, test_loader = get_shd_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        dt_ms=args.dt_ms,
        t_max_ms=args.t_max_ms,
    )

    if args.download_only:
        print("Dataset downloaded. Exiting.")
        sys.exit(0)

    T   = int(args.t_max_ms / args.dt_ms)
    print(f"\nES-FA SHD Benchmark")
    print(f"  Model : 700 -> {ESFASHDModel.N_H1} -> {ESFASHDModel.N_H2} -> 20 PLIF-SNN")
    print(f"  T     : {T} bins @ {args.dt_ms}ms = {args.t_max_ms}ms window")
    print(f"  Seeds : {seeds}")
    print(f"  Device: {device}")

    # ---- Multi-seed training ----
    seed_results = []
    for seed in seeds:
        r = train_single_seed(
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            sparsity_lambda=args.sparsity_lambda,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            weight_bits=args.weight_bits,
            threshold=args.threshold,
            init_beta=args.init_beta,
        )
        seed_results.append(r)

        per_seed_path = args.output_dir / f"seed_{seed}.json"
        with per_seed_path.open("w") as f:
            json.dump(r, f, indent=2)

    # ---- Aggregate ----
    agg = aggregate_seeds(seed_results)
    agg_path = args.output_dir / "aggregate.json"
    with agg_path.open("w") as f:
        json.dump(agg, f, indent=2)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("ES-FA SHD RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Accuracy (SHD test, {len(seeds)} seeds):")
    print(f"    Mean : {agg['test_acc_pct_mean']:.2f}%")
    print(f"    Std  : ±{agg['test_acc_pct_std']:.2f}%")
    print(f"    Range: {agg['test_acc_min']*100:.2f}% - {agg['test_acc_max']*100:.2f}%")
    print(f"  Sparsity (hidden layers):")
    print(f"    H1   : {agg['hidden1_sparsity_mean']*100:.1f}%")
    print(f"    H2   : {agg['hidden2_sparsity_mean']*100:.1f}%")
    print(f"  SOP reduction vs dense: {agg['sop_reduction_pct_mean']:.1f}%")
    print(f"  Energy proxy (MODEL, not board): "
          f"{agg['energy_proxy_pj_mean']/1e6:.3f} mJ/inference")
    print(f"\nResults written to: {args.output_dir.resolve()}")
    print(f"Aggregate: {agg_path}")


if __name__ == "__main__":
    main()
