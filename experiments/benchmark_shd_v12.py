"""ES-FA SHD v12: ALIF backbone + ES-FA improvements + verified safety fixes.

Context: v10 (faithful port) and v11 (port + EMA/TTA/committee) both trained
seed 123 to chance level with a silently collapsing firing rate. The deep
review of 2026-09-18 verified the exact mechanism, and it is an initialization
hazard inherited from the reference recipe: tau_m is drawn from N(20, 5) with
no positivity protection, and at hidden=128 seed 123 draws tau_m[5] = -1.84.
Then alpha = exp(-1/tau_m) = 1.72 > 1, the membrane recursion diverges
geometrically in the first epoch, the first backward pass writes NaN into
tau_m[5], and because clamp(NaN) stays NaN the post-hoc clamp cannot repair
it: spike(H(NaN)) = 0 forever, and the network trains to chance for the whole
run with no warning. The reference implementation carries the same hazard
(~0.8 percent of 2x128 seeds at 4 sigma on torch 2.14 CPU); it trained a
single seed and never demonstrated robustness.

v12 is the robustness-first redesign. The safety fixes replace the fragile
post-hoc clamp with a construction that cannot leave the stable regime, and
the remaining items are the ES-FA method additions:

1. Bounded time constants by construction: tau_m, tau_adp, and tau_m_o are
   sigmoid-reparameterized into [2, 100] (membrane) and [2, 1000]
   (adaptation), so alpha and rho always lie strictly inside (0, 1) no matter
   what initialization draw or gradient noise does. Initial values still
   follow the reference distributions N(20, 5) and N(150, 10): the raw
   parameters are set so the mapped value equals the drawn value, so the
   recipe is unchanged where it is healthy and provably safe where it is not.
2. Non-finite guard: if the loss or any gradient is non-finite, the step is
   skipped and counted; a seed that cannot recover aborts loudly and the
   remaining seeds continue (the project forbids silent failures).
3. Deterministic evaluation: eval-time state initialization uses a dedicated
   generator with a fixed seed, so repeated evaluations of one checkpoint are
   identical and best-epoch tracking is not driven by random-init noise.
   Training stays stochastic exactly as in the reference.
4. Robust committee: members whose firing rate collapsed are excluded, each
   member's output is normalized before averaging, and the member count is
   reported. A dead member can no longer drag the ensemble down.
5. Checkpointing: best-so-far model states are written every
   --checkpoint-every epochs so a crash cannot lose a completed seed.

Method additions over the reference implementation:
6. Extended optimization: 64 epochs with a 5-epoch linear warmup followed by
   cosine decay to 1e-4 (reference: 19 epochs, StepLR halving).
7. EMA weight averaging (decay 0.999, after warmup): the deployed model is
   the exponential moving average of the weights.
8. Event-loss test-time augmentation (TTA): the output is averaged over
   several input copies with random event dropouts, modeling the
   accelerator's event-loss robustness.
9. Event-loss training augmentation (--train-drop, opt-in): training-time
   event dropout so the robustness decoding and the training objective agree.
10. int16 deployment: post-training int16 quantization of all matrix weights,
    evaluated on the frozen test split.
11. Capacity: default hidden width 160 (the reference checkpoint uses 128);
    the source paper reports accuracy growing with network size on SHD.

All reference-faithful elements are retained: ALIF neuron with soft reset
(b0=0.01, beta=1.8, learnable tau_m and tau_adp), Multi-Gaussian surrogate
(sigma=0.5, s=6, h=0.15, gamma=0.5), two recurrent ALIF layers, non-spiking
leaky-integrator readout with summed per-step softmax from timestep 11,
orthogonal recurrent / xavier feedforward init, plain cross-entropy, full
BPTT, 4 ms bins, T=250, official split.

Usage:
    py -3 experiments/benchmark_shd_v12.py --epochs 64 --seeds 42,123,999 \
        --hidden 160 --layers 2 --batch-size 64 --tta 4
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

# Bounded time-constant ranges (sigmoid reparameterization, v12 safety fix).
TAU_M_RANGE = (2.0, 100.0)
TAU_ADP_RANGE = (2.0, 1000.0)

# Fixed generator seeds for deterministic evaluation (training stays random).
FIXED_EVAL_SEED = 1234567


def _bounded_param(init_values: torch.Tensor, lo: float, hi: float) -> nn.Parameter:
    """Reparameterize init values into a raw parameter whose sigmoid-mapped
    value always lies strictly inside (lo, hi), matching init exactly."""
    eps = 1e-4 * (hi - lo)
    target = init_values.clamp(lo + eps, hi - eps)
    frac = (target - lo) / (hi - lo)
    return nn.Parameter(torch.log(frac / (1.0 - frac)))


def _bounded_value(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(raw)


class MGSurrogateSpike(torch.autograd.Function):
    """Heaviside forward; Multi-Gaussian surrogate backward (Yin et al. 2021)."""

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

    alpha = exp(-1 / tau_m), rho = exp(-1 / tau_adp). Both time constants are
    learnable; v12 bounds them by construction (sigmoid reparameterization)
    so alpha and rho can never leave the stable open interval (0, 1) even if
    the raw optimizer state diverges.
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
        self.tau_m_raw = _bounded_param(
            torch.normal(tau_m_mean, tau_m_std, size=(size,)), *TAU_M_RANGE
        )
        self.tau_adp_raw = _bounded_param(
            torch.normal(tau_adp_mean, tau_adp_std, size=(size,)), *TAU_ADP_RANGE
        )

    @property
    def tau_m(self) -> torch.Tensor:
        return _bounded_value(self.tau_m_raw, *TAU_M_RANGE)

    @property
    def tau_adp(self) -> torch.Tensor:
        return _bounded_value(self.tau_adp_raw, *TAU_ADP_RANGE)

    def initial_state(
        self,
        batch: int,
        device: torch.device,
        gen: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.tau_m_raw.numel()
        mem = torch.rand(batch, n, device=device, generator=gen)
        spike = torch.zeros(batch, n, device=device)
        b = torch.full((batch, n), self.b0, device=device)
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


class SRNNv12(nn.Module):
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
        self.tau_m_o_raw = _bounded_param(
            torch.normal(20.0, 5.0, size=(N_CLASSES,)), *TAU_M_RANGE
        )

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

    @property
    def tau_m_o(self) -> torch.Tensor:
        return _bounded_value(self.tau_m_o_raw, *TAU_M_RANGE)

    def alif_list(self) -> List[ALIFNeuron]:
        neurons = [self.alif1]
        if self.n_layers == 2:
            neurons.append(self.alif2)
        return neurons

    def forward(
        self,
        x: torch.Tensor,
        gen: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        batch, timesteps, _ = x.shape
        device = x.device

        mem1, spike1, b1 = self.alif1.initial_state(batch, device, gen=gen)
        if self.n_layers == 2:
            mem2, spike2, b2 = self.alif2.initial_state(batch, device, gen=gen)

        mem_o = torch.rand(batch, N_CLASSES, device=device, generator=gen)
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


def event_drop(
    spikes: torch.Tensor,
    drop_time_prob: float = 0.05,
    drop_neuron_prob: float = 0.03,
    gen: Optional[torch.Generator] = None,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Mask input time bins or channels (50/50). Used for event-loss TTA and
    for opt-in training augmentation; the generator and RNG hooks make the
    TTA path deterministic when seeded."""
    chooser = rng.random() if rng is not None else random.random()
    if chooser < 0.5:
        mask = (
            torch.rand(1, spikes.shape[1], 1, device=spikes.device, generator=gen)
            > drop_time_prob
        ).float()
    else:
        mask = (
            torch.rand(1, 1, spikes.shape[2], device=spikes.device, generator=gen)
            > drop_neuron_prob
        ).float()
    return spikes * mask


def make_loaders(data_dir: Path, batch_size: int, dt_ms: float, t_max_ms: float, seed: int) -> Tuple[DataLoader, DataLoader]:
    train_h5 = data_dir / "SHD" / "shd_train.h5"
    test_h5 = data_dir / "SHD" / "shd_test.h5"
    for path in (train_h5, test_h5):
        if not path.exists():
            raise FileNotFoundError(f"SHD file not found: {path}")
    x_train, y_train = load_shd_h5(train_h5, dt_ms=dt_ms, t_max_ms=t_max_ms, as_uint8=True)
    x_test, y_test = load_shd_h5(test_h5, dt_ms=dt_ms, t_max_ms=t_max_ms, as_uint8=True)
    x_train = x_train.to(torch.uint8)
    x_test = x_test.to(torch.uint8)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate(
    model: SRNNv12,
    loader: DataLoader,
    device: torch.device,
    eval_seed: int = FIXED_EVAL_SEED,
) -> Tuple[float, float, float]:
    """Deterministic evaluation (fixed init generator); training remains random."""
    model.eval()
    gen = torch.Generator(device=str(device))
    gen.manual_seed(eval_seed)
    total_loss = 0.0
    correct = 0
    total = 0
    fr_sum = 0.0
    batches = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.to(device)
            logits = model(x, gen=gen)
            loss = F.cross_entropy(logits, y)
            total_loss += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += y.size(0)
            if model.last_firing_rate is not None:
                fr_sum += model.last_firing_rate
                batches += 1
    return total_loss / max(1, total), correct / max(1, total), fr_sum / max(1, batches)


def tta_accuracy(
    model: SRNNv12,
    loader: DataLoader,
    device: torch.device,
    copies: int,
    drop_time_prob: float,
    drop_neuron_prob: float,
    seed: int = FIXED_EVAL_SEED + 1,
) -> float:
    """Event-loss test-time augmentation with deterministic masks and states."""
    model.eval()
    gen = torch.Generator(device=str(device))
    gen.manual_seed(seed)
    rng = random.Random(seed)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.to(device)
            accum = model(x, gen=gen)
            for _ in range(max(0, copies - 1)):
                xm = event_drop(
                    x,
                    drop_time_prob=drop_time_prob,
                    drop_neuron_prob=drop_neuron_prob,
                    gen=gen,
                    rng=rng,
                )
                accum = accum + model(xm, gen=gen)
            correct += int((accum.argmax(1) == y).sum().item())
            total += y.size(0)
    return correct / max(1, total)


def int16_quantize(model: SRNNv12, device: torch.device) -> SRNNv12:
    """int16 quantization of all Linear weights on a fresh copy.

    Rebuilt with the full dynamics (beta, b0, sum_start included) so the
    quantized evaluation cannot silently fall back to defaults.
    """
    quantized = type(model)(
        n_hidden=model.n_hidden,
        n_layers=model.n_layers,
        beta=model.alif1.beta,
        b0=model.alif1.b0,
        sum_start=model.sum_start,
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


def lr_at(epoch: int, epochs: int, warmup: int, lr_max: float, lr_min: float) -> float:
    """Linear warmup then cosine decay (ES-FA improvement over StepLR)."""
    if epoch < warmup:
        return lr_max * float(epoch + 1) / float(max(1, warmup))
    progress = (epoch - warmup) / float(max(1, epochs - warmup - 1))
    progress = min(max(progress, 0.0), 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def train_seed(
    seed: int,
    args: argparse.Namespace,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = SRNNv12(
        n_hidden=args.hidden,
        n_layers=args.layers,
        beta=args.beta,
        b0=args.b0,
        sum_start=args.sum_start,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    history: List[Dict[str, float]] = []
    best_acc = 0.0
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    ema: Optional[Dict[str, torch.Tensor]] = None
    nonfinite_steps = 0

    for epoch in range(args.epochs):
        lr_now = lr_at(epoch, args.epochs, args.warmup_epochs, args.lr, args.lr_min)
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        model.train()
        started = time.perf_counter()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        skips_this_epoch = 0
        for x, y in train_loader:
            x = x.float().to(device)
            y = y.to(device)
            if args.train_drop and random.random() < 0.5:
                x = event_drop(x, args.train_drop_time, args.train_drop_neuron)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            if not bool(torch.isfinite(loss)):
                nonfinite_steps += 1
                skips_this_epoch += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"  [guard] seed {seed} epoch {epoch+1}: non-finite loss, step skipped",
                    flush=True,
                )
                if nonfinite_steps > args.max_nonfinite_steps:
                    raise RuntimeError(
                        f"seed {seed}: {nonfinite_steps} non-finite steps; aborting seed"
                    )
                continue

            loss.backward()

            grad_finite = True
            for p in model.parameters():
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    grad_finite = False
                    break
            if not grad_finite:
                nonfinite_steps += 1
                skips_this_epoch += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"  [guard] seed {seed} epoch {epoch+1}: non-finite gradient, step skipped",
                    flush=True,
                )
                if nonfinite_steps > args.max_nonfinite_steps:
                    raise RuntimeError(
                        f"seed {seed}: {nonfinite_steps} non-finite steps; aborting seed"
                    )
                continue

            optimizer.step()

            if epoch >= args.warmup_epochs:
                with torch.no_grad():
                    state = model.state_dict()
                    if ema is None:
                        ema = {k: v.detach().clone() for k, v in state.items()}
                    else:
                        for k, v in state.items():
                            if v.dtype.is_floating_point:
                                ema[k].mul_(args.ema_decay).add_(v.detach(), alpha=1.0 - args.ema_decay)

            train_loss += float(loss.item()) * y.size(0)
            train_correct += int((logits.argmax(1) == y).sum().item())
            train_total += y.size(0)

        test_loss, test_acc, test_fr = evaluate(model, test_loader, device)
        epoch_time = time.perf_counter() - started

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(train_loss / max(1, train_total), 4),
                "train_acc": round(train_correct / max(1, train_total), 4),
                "test_loss": round(test_loss, 4),
                "test_acc": round(test_acc, 4),
                "test_firing_rate": round(test_fr, 4),
                "lr": lr_now,
                "nonfinite_steps": skips_this_epoch,
                "elapsed_s": round(epoch_time, 2),
            }
        )
        print(
            f"  seed {seed} epoch {epoch+1:3d}/{args.epochs} | "
            f"train {history[-1]['train_acc']*100:.1f}% | test {test_acc*100:.2f}% | "
            f"best {best_acc*100:.2f}% | FR {test_fr:.4f} | lr {lr_now:.2e} | {epoch_time:.1f}s",
            flush=True,
        )

        if args.checkpoint_every > 0 and (
            (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1
        ):
            torch.save(
                {
                    "seed": seed,
                    "epoch": epoch + 1,
                    "test_acc": float(test_acc),
                    "best_acc": float(best_acc),
                    "best_state": best_state,
                    "ema": ema,
                    "history": history,
                    "nonfinite_steps": nonfinite_steps,
                },
                args.output_dir / f"seed_{seed}.ckpt.pt",
            )

    assert ema is not None, "EMA was never initialized; check warmup < epochs"

    torch.save(
        {k: v.detach().cpu() for k, v in ema.items()},
        args.output_dir / f"seed_{seed}_ema.pt",
    )

    ema_model = SRNNv12(
        n_hidden=args.hidden,
        n_layers=args.layers,
        beta=args.beta,
        b0=args.b0,
        sum_start=args.sum_start,
    ).to(device)
    ema_model.load_state_dict({k: v for k, v in ema.items()})
    ema_model.eval()

    _, ema_acc, ema_fr = evaluate(ema_model, test_loader, device)
    _, raw_final_acc, raw_final_fr = evaluate(model, test_loader, device)
    tta_acc = tta_accuracy(ema_model, test_loader, device, args.tta, args.tta_drop_time, args.tta_drop_neuron)
    q = int16_quantize(ema_model, device)
    _, int16_acc, int16_fr = evaluate(q, test_loader, device)

    print(
        f"  seed {seed} FINAL | best {best_acc*100:.2f}% (ep {best_epoch}) | raw_final {raw_final_acc*100:.2f}% | "
        f"ema {ema_acc*100:.2f}% | ema+tta{args.tta} {tta_acc*100:.2f}% | int16(ema) {int16_acc*100:.2f}% | FR {ema_fr:.4f}",
        flush=True,
    )

    result = {
        "seed": seed,
        "n_params": n_params,
        "best_test_acc": round(best_acc, 4),
        "best_epoch": best_epoch,
        "final_test_acc": round(raw_final_acc, 4),
        "ema_test_acc": round(ema_acc, 4),
        "tta_test_acc": round(tta_acc, 4),
        "int16_test_acc": round(int16_acc, 4),
        "final_firing_rate": round(raw_final_fr, 4),
        "ema_firing_rate": round(ema_fr, 4),
        "int16_firing_rate": round(int16_fr, 4),
        "nonfinite_steps": nonfinite_steps,
        "history": history,
    }
    return result, ema


def committee_accuracy(
    states: List[Dict[str, torch.Tensor]],
    firing_rates: List[float],
    args: argparse.Namespace,
    loader: DataLoader,
    device: torch.device,
    min_fr: float = 0.02,
    seed: int = FIXED_EVAL_SEED + 2,
) -> Tuple[Optional[float], int]:
    """Quality-gated committee: members with collapsed firing rates are
    excluded, member outputs are normalized before averaging, and the member
    count is returned."""
    kept = [(s, fr) for s, fr in zip(states, firing_rates) if fr >= min_fr]
    if len(kept) < 2:
        return None, 0
    gen = torch.Generator(device=str(device))
    gen.manual_seed(seed)
    models = []
    for state, _ in kept:
        m = SRNNv12(
            n_hidden=args.hidden,
            n_layers=args.layers,
            beta=args.beta,
            b0=args.b0,
            sum_start=args.sum_start,
        ).to(device)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.to(device)
            accum = None
            for m in models:
                out = m(x, gen=gen)
                out = out / out.sum(dim=1, keepdim=True).clamp_min(1e-9)
                accum = out if accum is None else accum + out
            correct += int((accum.argmax(1) == y).sum().item())
            total += y.size(0)
    return correct / max(1, total), len(models)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ES-FA SHD v12: bounded ALIF backbone + ES-FA improvements")
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="42,123,999")
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--layers", type=int, default=2, choices=[1, 2])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--beta", type=float, default=1.8)
    parser.add_argument("--b0", type=float, default=0.01)
    parser.add_argument("--sum-start", type=int, default=11)
    parser.add_argument("--train-drop", action="store_true", help="event-drop augmentation during training")
    parser.add_argument("--train-drop-time", type=float, default=0.05)
    parser.add_argument("--train-drop-neuron", type=float, default=0.03)
    parser.add_argument("--max-nonfinite-steps", type=int, default=100,
                        help="abort a seed after this many skipped non-finite steps")
    parser.add_argument("--tta", type=int, default=4, help="test-time augmentation copies (1 = off)")
    parser.add_argument("--tta-drop-time", type=float, default=0.05)
    parser.add_argument("--tta-drop-neuron", type=float, default=0.03)
    parser.add_argument("--dt-ms", type=float, default=4.0)
    parser.add_argument("--t-max-ms", type=float, default=1000.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/shd_v12"))
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
        f"ES-FA SHD v12 | bounded ALIF backbone + ES-FA improvements | hidden={args.hidden} | layers={args.layers} "
        f"| T={int(args.t_max_ms / args.dt_ms)} @ {args.dt_ms}ms | epochs={args.epochs} (warmup {args.warmup_epochs}) "
        f"| seeds={seeds} | batch={args.batch_size} | lr={args.lr}->{args.lr_min} cosine | ema={args.ema_decay} "
        f"| tta={args.tta} | train_drop={args.train_drop} | device={device}"
    )
    train_loader, test_loader = make_loaders(args.data_dir, args.batch_size, args.dt_ms, args.t_max_ms, seeds[0])
    print(f"  train batches={len(train_loader)} test batches={len(test_loader)}")

    results: List[Dict[str, object]] = []
    ema_states: List[Dict[str, torch.Tensor]] = []
    ema_frs: List[float] = []
    aborted: List[int] = []
    for seed in seeds:
        print(f"seed {seed}:")
        try:
            result, ema_state = train_seed(seed, args, train_loader, test_loader, device)
        except RuntimeError as exc:
            print(f"  seed {seed} ABORTED: {exc}", flush=True)
            aborted.append(seed)
            results.append({"seed": seed, "status": "aborted", "error": str(exc)})
            continue
        results.append(result)
        ema_states.append(ema_state)
        ema_frs.append(float(result["ema_firing_rate"]))
        (args.output_dir / f"seed_{seed}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )

    committee_acc: Optional[float] = None
    committee_members = 0
    if len(ema_states) > 1:
        committee_acc, committee_members = committee_accuracy(
            ema_states, ema_frs, args, test_loader, device
        )
    if committee_acc is not None:
        print(f"\nv12 committee ({committee_members}-member EMA ensemble): {committee_acc*100:.2f}%")
    else:
        print("\nno committee (need at least two healthy members)")

    ok = [r for r in results if "best_test_acc" in r]

    def _mean(key: str) -> Optional[float]:
        if not ok:
            return None
        return round(float(np.mean([r[key] for r in ok])), 4)  # type: ignore[index]

    summary = {
        "positioning": "ES-FA v12: bounded-dynamics ALIF backbone (sigmoid-bounded tau_m, tau_adp, tau_m_o; non-finite step guard; deterministic evaluation) + ES-FA improvements (warmup+cosine 64 epochs, EMA weight averaging, event-loss TTA, quality-gated committee, int16 deployment)",
        "config": {
            "hidden": args.hidden,
            "layers": args.layers,
            "dt_ms": args.dt_ms,
            "t_max_ms": args.t_max_ms,
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_min": args.lr_min,
            "eps": args.eps,
            "ema_decay": args.ema_decay,
            "train_drop": args.train_drop,
            "tta": args.tta,
            "tta_drop_time": args.tta_drop_time,
            "tta_drop_neuron": args.tta_drop_neuron,
        },
        "seeds": seeds,
        "aborted_seeds": aborted,
        "best_acc_mean": _mean("best_test_acc"),
        "best_acc_max": round(float(np.max([r["best_test_acc"] for r in ok])), 4) if ok else None,
        "ema_acc_mean": _mean("ema_test_acc"),
        "tta_acc_mean": _mean("tta_test_acc"),
        "int16_acc_mean": _mean("int16_test_acc"),
        "ema_firing_rate_mean": _mean("ema_firing_rate"),
        "committee_acc": round(committee_acc, 4) if committee_acc is not None else None,
        "committee_members": committee_members,
        "results": results,
    }
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    if ok:
        print(
            f"\nv12 summary: best mean {summary['best_acc_mean']*100:.2f}% | ema mean {summary['ema_acc_mean']*100:.2f}% "
            f"| tta mean {summary['tta_acc_mean']*100:.2f}% | int16 mean {summary['int16_acc_mean']*100:.2f}%"
            + (f" | committee {summary['committee_acc']*100:.2f}% ({committee_members} members)" if committee_acc is not None else "")
        )
    else:
        print("\nv12 summary: no completed seeds")
    print(f"written: {args.output_dir / 'aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
