"""ES-FA SHD v10: faithful SRNN-ALIF recipe (Yin, Corradi, Bohte 2021).

Why this script exists: v5-v9 used a custom recipe (hard reset, fixed
threshold, AdamW + cosine, label smoothing, dropout, event-drop, dual leaky
readout, optional rate governor) that plateaus around 80-81% on SHD. The
published adaptive-spiking recurrent network (SRNN) recipe reaches ~90.7%
with a much smaller network (2 x 128 neurons, 19 epochs) in the authors'
open-source implementation (MIT, github.com/byin-cwi/Efficient-spiking-networks,
SHD/SHD_2layer_ALIF.py). v10 ports that recipe faithfully so the ES-FA line
gets a comparable-accuracy training protocol; ES-FA additions (governor,
int16 deployment) can be layered on top once the recipe is validated.

Faithful elements (values from the reference code):
- ALIF neuron: soft reset by the dynamic threshold B, adaptation state
  b <- rho*b + (1-rho)*spike, B <- b0 + beta*b with b0=0.01, beta=1.8.
  alpha = exp(-1/tau_m), rho = exp(-1/tau_adp); tau_m, tau_adp learnable
  (linear parameterization), tau_m ~ N(20,5), tau_adp ~ N(150,10).
- Multi-Gaussian surrogate: g'(x) = gamma*[(1+h)N(0,sigma^2) - hN(sigma,(s*sigma)^2)
  - hN(-sigma,(s*sigma)^2)] with sigma=0.5, s=6, h=0.15, gamma=0.5.
- Two recurrent layers, 128 neurons each; readout is a non-spiking leaky
  integrator (learnable tau_m_o); logits are the running sum of per-step
  softmax of the readout membrane, starting at timestep 11.
- Init: recurrent orthogonal, feedforward/readout xavier-uniform, all biases
  zero, membrane potential U[0,1), adaptation state b0, spikes zero.
- Adam lr 1e-2, eps 1e-5, no weight decay; StepLR(step_size=10, gamma=0.5);
  plain cross-entropy; full BPTT; no dropout, no label smoothing, no
  augmentation, no rate regularization.
- SHD pipeline: 4 ms bins, T=250, 1.0 s window, binary 700-channel frames,
  official split (matches the reference and our existing data loader).

Reported reference result: 90.72% test, mean firing rate 0.110 (19 epochs).
An int16 weight-quantization evaluation is included at the end for the
ES-FA deployment story (the reference does not do this).

Usage:
    py -3 experiments/benchmark_shd_v10.py --epochs 19 --seeds 42,123,999 \
        --hidden 128 --layers 2 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_shd_esfa import load_shd_h5  # noqa: E402

N_INPUT = 700
N_CLASSES = 20


class MGSurrogateSpike(torch.autograd.Function):
    """Heaviside forward; Multi-Gaussian surrogate backward (Yin et al. 2021).

    g'(x) = gamma * [ (1+h) N(x | 0, sigma^2)
                      - h N(x | +sigma, (s*sigma)^2)
                      - h N(x | -sigma, (s*sigma)^2) ]
    with sigma=0.5, s=6, h=0.15, gamma=0.5 (reference code values).
    """

    sigma = 0.5
    s = 6.0
    h = 0.15
    gamma = 0.5

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        (x,) = ctx.saved_tensors
        sig, s, h, gam = (
            MGSurrogateSpike.sigma,
            MGSurrogateSpike.s,
            MGSurrogateSpike.h,
            MGSurrogateSpike.gamma,
        )

        def g(mu: float, sd: float) -> torch.Tensor:
            return torch.exp(-((x - mu) ** 2) / (2.0 * sd * sd)) / (math.sqrt(2.0 * math.pi) * sd)

        temp = (1.0 + h) * g(0.0, sig) - h * g(sig, s * sig) - h * g(-sig, s * sig)
        return grad_output * temp.float() * gam


def mg_spike_fn(x: torch.Tensor) -> torch.Tensor:
    return MGSurrogateSpike.apply(x)


class ALIFNeuron(nn.Module):
    """Adaptive-threshold LIF with soft reset (Yin et al. 2021).

        b     <- rho * b + (1 - rho) * spike          (adaptation state)
        B     <- b0 + beta * b                        (dynamic threshold)
        mem   <- alpha * mem + (1 - alpha) * I - B * spike
        spike <- Heaviside(mem - B)

    alpha = exp(-1 / tau_m), rho = exp(-1 / tau_adp); tau_m and tau_adp are
    learnable with no positivity constraint, matching the reference code.
    """

    def __init__(
        self,
        size: int,
        beta: float = 1.8,
        b0: float = 0.01,
        tau_m_mean: float = 20.0,
        tau_m_std: float = 5.0,
        tau_adp_mean: float = 150.0,
        tau_adp_std: float = 10.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.b0 = b0
        # Safety fix (post-run, 2026-09-18): clamp the initialization draws.
        # tau_m ~ N(20,5) has a 4-sigma tail (P ~ 3.2e-5 per neuron); a
        # negative draw makes alpha = exp(-1/tau_m) > 1 and the membrane
        # recursion diverges, poisoning the parameters with NaN. Seed 123 at
        # hidden=128 drew tau_m[5] = -1.84 and trained to chance silently.
        # Clamping only alters draws that would otherwise be fatal.
        self.tau_m = nn.Parameter(
            torch.normal(tau_m_mean, tau_m_std, size=(size,)).clamp(2.0, 100.0)
        )
        self.tau_adp = nn.Parameter(
            torch.normal(tau_adp_mean, tau_adp_std, size=(size,)).clamp(2.0, 1000.0)
        )

    def initial_state(
        self, batch: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mem = torch.rand(batch, self.tau_m.numel(), device=device)
        spike = torch.zeros(batch, self.tau_m.numel(), device=device)
        b = torch.full((batch, self.tau_m.numel()), self.b0, device=device)
        return mem, spike, b

    def forward(
        self,
        inputs: torch.Tensor,
        mem: torch.Tensor,
        spike: torch.Tensor,
        b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = torch.exp(-1.0 / self.tau_m)
        rho = torch.exp(-1.0 / self.tau_adp)
        b = rho * b + (1.0 - rho) * spike
        threshold = self.b0 + self.beta * b
        mem = mem * alpha + (1.0 - alpha) * inputs - threshold * spike
        spike_new = mg_spike_fn(mem - threshold)
        return mem, spike_new, b


class SRNNv10(nn.Module):
    """Two recurrent ALIF layers with a non-spiking leaky-integrator readout."""

    def __init__(
        self,
        n_hidden: int = 128,
        n_layers: int = 2,
        beta: float = 1.8,
        b0: float = 0.01,
        sum_start: int = 11,
    ) -> None:
        super().__init__()
        if n_layers not in (1, 2):
            raise ValueError(f"n_layers must be 1 or 2, got {n_layers}")
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.sum_start = sum_start

        self.i_2_h1 = nn.Linear(N_INPUT, n_hidden)
        self.h1_2_h1 = nn.Linear(n_hidden, n_hidden)
        self.alif1 = ALIFNeuron(n_hidden, beta=beta, b0=b0)
        if n_layers == 2:
            self.h1_2_h2 = nn.Linear(n_hidden, n_hidden)
            self.h2_2_h2 = nn.Linear(n_hidden, n_hidden)
            self.alif2 = ALIFNeuron(n_hidden, beta=beta, b0=b0)
        self.h2o = nn.Linear(n_hidden, N_CLASSES)
        self.tau_m_o = nn.Parameter(torch.normal(20.0, 5.0, size=(N_CLASSES,)))

        nn.init.xavier_uniform_(self.i_2_h1.weight)
        nn.init.orthogonal_(self.h1_2_h1.weight)
        if n_layers == 2:
            nn.init.xavier_uniform_(self.h1_2_h2.weight)
            nn.init.orthogonal_(self.h2_2_h2.weight)
        nn.init.xavier_uniform_(self.h2o.weight)
        for layer in (self.i_2_h1, self.h1_2_h1):
            nn.init.constant_(layer.bias, 0)
        if n_layers == 2:
            for layer in (self.h1_2_h2, self.h2_2_h2):
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.h2o.bias, 0)

        self.last_firing_rate: Optional[float] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, timesteps, _ = x.shape
        device = x.device

        mem1, spike1, b1 = self.alif1.initial_state(batch, device)
        if self.n_layers == 2:
            mem2, spike2, b2 = self.alif2.initial_state(batch, device)

        mem_o = torch.rand(batch, N_CLASSES, device=device)
        out = torch.zeros(batch, N_CLASSES, device=device)
        spike_units = 0.0
        spike_total = 0.0

        for t in range(timesteps):
            input_x = x[:, t, :]
            h1 = self.i_2_h1(input_x) + self.h1_2_h1(spike1)
            mem1, spike1, b1 = self.alif1(h1, mem1, spike1, b1)
            spike_units += float(spike1.numel())
            spike_total += float(spike1.sum().item())

            if self.n_layers == 2:
                h2 = self.h1_2_h2(spike1) + self.h2_2_h2(spike2)
                mem2, spike2, b2 = self.alif2(h2, mem2, spike2, b2)
                spike_units += float(spike2.numel())
                spike_total += float(spike2.sum().item())
                hidden_spikes = spike2
            else:
                hidden_spikes = spike1

            alpha_o = torch.exp(-1.0 / self.tau_m_o)
            mem_o = mem_o * alpha_o + (1.0 - alpha_o) * self.h2o(hidden_spikes)
            if t >= self.sum_start:
                out = out + F.softmax(mem_o, dim=1)

        self.last_firing_rate = spike_total / max(1.0, spike_units)
        return out


def make_loaders(data_dir: Path, batch_size: int, dt_ms: float, t_max_ms: float, seed: int) -> Tuple[DataLoader, DataLoader]:
    train_h5 = data_dir / "SHD" / "shd_train.h5"
    test_h5 = data_dir / "SHD" / "shd_test.h5"
    for path in (train_h5, test_h5):
        if not path.exists():
            raise FileNotFoundError(f"SHD file not found: {path}")
    x_train, y_train = load_shd_h5(train_h5, dt_ms=dt_ms, t_max_ms=t_max_ms)
    x_test, y_test = load_shd_h5(test_h5, dt_ms=dt_ms, t_max_ms=t_max_ms)
    x_train = x_train.to(torch.uint8)
    x_test = x_test.to(torch.uint8)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate(model: SRNNv10, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    fr_sum = 0.0
    batches = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += y.size(0)
            if model.last_firing_rate is not None:
                fr_sum += model.last_firing_rate
                batches += 1
    return total_loss / max(1, total), correct / max(1, total), fr_sum / max(1, batches)


def quantize_int16(model: SRNNv10, device: torch.device) -> SRNNv10:
    """Post-training int16 quantization of Linear weights on a fresh copy.

    ALIF/readout time-constant parameters stay fp32 (they are per-neuron
    scalars, cheap in hardware); all matrix weights are quantized to an
    int16 grid with a per-layer scale, matching the ES-FA deployment story.
    """
    quantized = type(model)(
        n_hidden=model.n_hidden,
        n_layers=model.n_layers,
    )
    quantized.load_state_dict(model.state_dict())
    quantized = quantized.to(device)
    quantized.eval()
    with torch.no_grad():
        linears = [quantized.i_2_h1, quantized.h1_2_h1, quantized.h2o]
        if model.n_layers == 2:
            linears.extend([quantized.h1_2_h2, quantized.h2_2_h2])
        for linear in linears:
            scale = linear.weight.abs().amax().clamp(1e-8) / 32767.0
            linear.weight.data = torch.round(linear.weight / scale).clamp(-32768, 32767) * scale
    return quantized


def train_seed(
    seed: int,
    args: argparse.Namespace,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = SRNNv10(
        n_hidden=args.hidden,
        n_layers=args.layers,
        beta=args.beta,
        b0=args.b0,
        sum_start=args.sum_start,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    history: List[Dict[str, float]] = []
    best_acc = 0.0
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x = x.float().to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * y.size(0)
            train_correct += int((logits.argmax(1) == y).sum().item())
            train_total += y.size(0)
        scheduler.step()

        test_loss, test_acc, test_fr = evaluate(model, test_loader, device)
        epoch_time = time.perf_counter() - started

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss / max(1, train_total), 4),
                "train_acc": round(train_correct / max(1, train_total), 4),
                "test_loss": round(test_loss, 4),
                "test_acc": round(test_acc, 4),
                "test_firing_rate": round(test_fr, 4),
                "lr": optimizer.param_groups[0]["lr"],
                "mean_tau_m_h1": round(float(model.alif1.tau_m.mean().item()), 3),
                "mean_tau_adp_h1": round(float(model.alif1.tau_adp.mean().item()), 3),
                "elapsed_s": round(epoch_time, 2),
            }
        )
        print(
            f"  seed {seed} epoch {epoch:3d}/{args.epochs} | "
            f"train {history[-1]['train_acc']*100:.1f}% | test {test_acc*100:.2f}% | "
            f"best {best_acc*100:.2f}% | FR {test_fr:.4f} | lr {optimizer.param_groups[0]['lr']:.2e} | "
            f"{epoch_time:.1f}s",
            flush=True,
        )

    assert best_state is not None
    model.load_state_dict(best_state)
    _, float_acc, float_fr = evaluate(model, test_loader, device)
    quantized = quantize_int16(model, device)
    _, int16_acc, int16_fr = evaluate(quantized, test_loader, device)

    return {
        "seed": seed,
        "n_params": n_params,
        "best_test_acc": round(best_acc, 4),
        "best_epoch": best_epoch,
        "final_test_acc": round(float_acc, 4),
        "int16_test_acc": round(int16_acc, 4),
        "final_firing_rate": round(float_fr, 4),
        "int16_firing_rate": round(int16_fr, 4),
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ES-FA SHD v10: faithful SRNN-ALIF recipe")
    parser.add_argument("--epochs", type=int, default=19)
    parser.add_argument("--seeds", type=str, default="42,123,999")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2, choices=[1, 2])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.8)
    parser.add_argument("--b0", type=float, default=0.01)
    parser.add_argument("--sum-start", type=int, default=11)
    parser.add_argument("--dt-ms", type=float, default=4.0)
    parser.add_argument("--t-max-ms", type=float, default=1000.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/shd_v10"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"ES-FA SHD v10 | SRNN-ALIF faithful port | hidden={args.hidden} | layers={args.layers} "
        f"| T={int(args.t_max_ms / args.dt_ms)} @ {args.dt_ms}ms | epochs={args.epochs} | seeds={seeds} "
        f"| batch={args.batch_size} | lr={args.lr} eps={args.eps} | step={args.step_size}x{args.gamma} "
        f"| beta={args.beta} b0={args.b0} | sum_start={args.sum_start} | device={device}"
    )
    train_loader, test_loader = make_loaders(args.data_dir, args.batch_size, args.dt_ms, args.t_max_ms, seeds[0])
    print(f"  train batches={len(train_loader)} test batches={len(test_loader)}")

    results = []
    for seed in seeds:
        print(f"seed {seed}:")
        result = train_seed(seed, args, train_loader, test_loader, device)
        results.append(result)
        (args.output_dir / f"seed_{seed}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )

    summary = {
        "positioning": "ES-FA v10: faithful port of the SRNN-ALIF recipe (Yin, Corradi, Bohte 2021, MIT reference code); protocol is theirs, int16 quantization evaluation is the ES-FA addition",
        "config": {
            "hidden": args.hidden,
            "layers": args.layers,
            "dt_ms": args.dt_ms,
            "t_max_ms": args.t_max_ms,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "eps": args.eps,
            "step_size": args.step_size,
            "gamma": args.gamma,
            "beta": args.beta,
            "b0": args.b0,
            "sum_start": args.sum_start,
        },
        "seeds": seeds,
        "best_acc_mean": round(float(np.mean([r["best_test_acc"] for r in results])), 4),
        "best_acc_max": round(float(np.max([r["best_test_acc"] for r in results])), 4),
        "int16_acc_mean": round(float(np.mean([r["int16_test_acc"] for r in results])), 4),
        "final_firing_rate_mean": round(float(np.mean([r["final_firing_rate"] for r in results])), 4),
        "results": results,
    }
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"\nv10 summary: best mean {summary['best_acc_mean']*100:.2f}% "
        f"max {summary['best_acc_max']*100:.2f}% | int16 mean {summary['int16_acc_mean']*100:.2f}% "
        f"| FR {summary['final_firing_rate_mean']:.4f}"
    )
    print(f"written: {args.output_dir / 'aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
