"""ES-FA SHD v5: ES-FA innovations on a protocol-corrected recurrent backbone.

Positioning (why this is not a port):

- The *protocol* follows the public reference recipe so results are comparable
  (4 ms bins, 1.0 s window, hard reset with scaled input, non-spiking leaky
  readout, AdamW + cosine, label smoothing, event-drop augmentation, fast
  sigmoid surrogate). Protocol elements are attributed to the Catalyst
  neuromorphic benchmarks (github.com/catalyst-neuromorphic/catalyst-benchmarks,
  MIT) and are not claimed as contributions.
- The *model and training mechanism* are ES-FA's:
    1. Event-rate governor training (ES-FA innovation): a closed-loop
       controller adjusts the sparsity pressure so the hidden event rate
       tracks an annealed budget, instead of a fixed penalty. This ties the
       training objective to the accelerator cost model (SOP = events x
       fanout) and yields an accuracy-vs-events Pareto reading.
    2. Dual-timescale leaky readout (ES-FA innovation): fast and slow leaky
       accumulators with learnable mixing - two accumulators in hardware,
       no multipliers.
    3. Recurrent PLIF backbone retained from ES-FA v2 where requested
       (learnable per-neuron beta, recurrence), single-layer default for the
       CPU budget.
    4. Hardware alignment: post-training int16 quantization evaluated on the
       same test split (ES-FA deployment path).

Usage:
    py -3 experiments/benchmark_shd_v5.py --epochs 200 --seeds 42 --hidden 512 \
        --governor --dual-readout
"""

from __future__ import annotations

import argparse
import copy
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


class SurrogateSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid backward (protocol element, scale 25)."""

    scale = 25.0

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        (x,) = ctx.saved_tensors
        grad = grad_output / (SurrogateSpike.scale * torch.abs(x) + 1.0) ** 2
        return grad


def spike_fn(x: torch.Tensor) -> torch.Tensor:
    return SurrogateSpike.apply(x)


class LIFNeuron(nn.Module):
    """Multiplicative-decay LIF: v = beta*v + (1-beta)*I; hard reset; learnable beta."""

    def __init__(self, size: int, beta_init: float = 0.95, threshold: float = 1.0) -> None:
        super().__init__()
        self.threshold = threshold
        init_val = float(np.log(beta_init / (1.0 - beta_init)))
        self.beta_raw = nn.Parameter(torch.full((size,), init_val))

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_raw)

    def forward(self, current: torch.Tensor, v_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta
        v = beta * v_prev + (1.0 - beta) * current
        spikes = spike_fn(v - self.threshold)
        v = v * (1.0 - spikes)
        return v, spikes


class ESFASDNetV5(nn.Module):
    """700 -> hidden (recurrent LIF, PLIF-style learnable beta) -> 20 readout.

    ES-FA innovations: dual-timescale leaky readout with learnable mixing,
    exposed per-step event counts for the event-rate governor.
    """

    def __init__(
        self,
        n_hidden: int = 512,
        dropout: float = 0.3,
        threshold: float = 1.0,
        beta_init: float = 0.95,
        beta_out: float = 0.9,
        dual_readout: bool = False,
        beta_fast: float = 0.5,
        beta_slow: float = 0.95,
        target_rate: float = 0.05,
        activity_lambda: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.dual_readout = dual_readout
        self.target_rate = target_rate
        self.activity_lambda = activity_lambda
        self.aux_loss: Optional[torch.Tensor] = None
        self.last_rate: Optional[torch.Tensor] = None

        self.fc1 = nn.Linear(N_INPUT, n_hidden, bias=False)
        self.fc_rec = nn.Linear(n_hidden, n_hidden, bias=False)
        self.fc_out = nn.Linear(n_hidden, N_CLASSES, bias=False)

        self.lif1 = LIFNeuron(n_hidden, beta_init=beta_init, threshold=threshold)
        if dual_readout:
            self.readout_fast = LIFNeuron(N_CLASSES, beta_init=beta_fast, threshold=threshold)
            self.readout_slow = LIFNeuron(N_CLASSES, beta_init=beta_slow, threshold=threshold)
            self.readout_mix = nn.Parameter(torch.zeros(2))
        else:
            self.lif_out = LIFNeuron(N_CLASSES, beta_init=beta_out, threshold=threshold)

        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.5)

    def readout_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dual_readout:
            mixing = torch.softmax(self.readout_mix, dim=0)
            return mixing[0], mixing[1]
        return torch.tensor(1.0), torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, timesteps, _ = x.shape
        device = x.device

        # Input projection is time-independent: compute it for all steps at once.
        projected_input = self.fc1(x.reshape(batch * timesteps, N_INPUT)).reshape(
            batch, timesteps, self.n_hidden
        )

        v1 = torch.zeros(batch, self.n_hidden, device=device)
        spk1 = torch.zeros(batch, self.n_hidden, device=device)
        spk1_d = torch.zeros(batch, self.n_hidden, device=device)
        out_sum = torch.zeros(batch, N_CLASSES, device=device)
        spike_count = torch.zeros(batch, self.n_hidden, device=device)

        if self.dual_readout:
            v_fast = torch.zeros(batch, N_CLASSES, device=device)
            v_slow = torch.zeros(batch, N_CLASSES, device=device)
            w_fast, w_slow = self.readout_weights()
            beta_fast = self.readout_fast.beta
            beta_slow = self.readout_slow.beta
        else:
            v_out = torch.zeros(batch, N_CLASSES, device=device)
            beta_out = self.lif_out.beta

        for t, projected_t in enumerate(projected_input.unbind(dim=1)):
            current = projected_t + self.fc_rec(spk1_d)
            v1, spk1 = self.lif1(current, v1)
            spk1_d = self.dropout(spk1) if self.training else spk1
            spike_count = spike_count + spk1

            i_out = self.fc_out(spk1_d)
            if self.dual_readout:
                v_fast = beta_fast * v_fast + (1.0 - beta_fast) * i_out
                v_slow = beta_slow * v_slow + (1.0 - beta_slow) * i_out
                out_sum = out_sum + (w_fast * v_fast + w_slow * v_slow)
            else:
                v_out = beta_out * v_out + (1.0 - beta_out) * i_out
                out_sum = out_sum + v_out

        mean_rate = spike_count / float(timesteps)
        if self.training:
            self.last_rate = mean_rate.mean().detach()
            if self.activity_lambda > 0.0:
                self.aux_loss = self.activity_lambda * ((mean_rate - self.target_rate) ** 2).mean()
            else:
                self.aux_loss = None
        else:
            self.last_rate = mean_rate.mean().detach()

        return out_sum / float(timesteps)


def event_drop(spikes: torch.Tensor, drop_time_prob: float = 0.1, drop_neuron_prob: float = 0.05) -> torch.Tensor:
    """Protocol element: mask input time bins or channels (50/50)."""
    if random.random() < 0.5:
        mask = (torch.rand(1, spikes.shape[1], 1, device=spikes.device) > drop_time_prob).float()
    else:
        mask = (torch.rand(1, 1, spikes.shape[2], device=spikes.device) > drop_neuron_prob).float()
    return spikes * mask


def make_loaders(data_dir: Path, batch_size: int, dt_ms: float, t_max_ms: float, seed: int) -> Tuple[DataLoader, DataLoader]:
    train_h5 = data_dir / "SHD" / "shd_train.h5"
    test_h5 = data_dir / "SHD" / "shd_test.h5"
    for path in (train_h5, test_h5):
        if not path.exists():
            raise FileNotFoundError(f"SHD file not found: {path}")
    x_train, y_train = load_shd_h5(train_h5, dt_ms=dt_ms, t_max_ms=t_max_ms)
    x_test, y_test = load_shd_h5(test_h5, dt_ms=dt_ms, t_max_ms=t_max_ms)
    # Store spikes as uint8 (binary events): 4x less memory than float32,
    # batches are cast back to float in the training/evaluation loops.
    x_train = x_train.to(torch.uint8)
    x_test = x_test.to(torch.uint8)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate(model: ESFASDNetV5, loader: DataLoader, device: torch.device, label_smoothing: float = 0.0) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    rate_sum = 0.0
    batches = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            total_loss += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += y.size(0)
            if model.last_rate is not None:
                rate_sum += float(model.last_rate.item())
                batches += 1
    return total_loss / max(1, total), correct / max(1, total), rate_sum / max(1, batches)


def quantize_int16(model: ESFASDNetV5, device: torch.device) -> ESFASDNetV5:
    """Post-training int16 quantization on a fresh copy (no module deepcopy)."""
    quantized = type(model)(
        n_hidden=model.n_hidden,
        dropout=0.0,
        dual_readout=model.dual_readout,
    )
    quantized.load_state_dict(model.state_dict())
    quantized = quantized.to(device)
    quantized.eval()
    with torch.no_grad():
        for linear in (quantized.fc1, quantized.fc_rec, quantized.fc_out):
            scale = linear.weight.abs().amax().clamp(1e-8) / 32767.0
            linear.weight.data = torch.round(linear.weight / scale).clamp(-32768, 32767) * scale
    return quantized


def governor_update(lambda_value: float, measured_rate: float, budget: float, eta: float) -> float:
    """ES-FA event-rate governor: multiplicative update toward the budget."""
    updated = lambda_value * math.exp(eta * (measured_rate - budget))
    return float(min(max(updated, 0.0), 1.0))


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

    activity_lambda = args.activity_lambda if not args.governor else args.lambda_start
    model = ESFASDNetV5(
        n_hidden=args.hidden,
        dropout=args.dropout,
        dual_readout=args.dual_readout,
        target_rate=args.target_rate,
        activity_lambda=activity_lambda,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    history: List[Dict[str, float]] = []
    best_acc = 0.0
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, args.epochs + 1):
        if args.governor:
            fraction = (epoch - 1) / max(1, args.epochs - 1)
            budget = args.budget_start + fraction * (args.budget_end - args.budget_start)
            model.activity_lambda = activity_lambda
            model.target_rate = budget
        else:
            budget = args.target_rate

        model.train()
        started = time.perf_counter()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        rate_sum = 0.0
        rate_batches = 0
        time_data = 0.0
        time_forward = 0.0
        time_backward = 0.0
        for x, y in train_loader:
            mark = time.perf_counter()
            x = x.float().to(device)
            y = y.to(device)
            if not args.no_event_drop:
                x = event_drop(x)
            time_data += time.perf_counter() - mark

            mark = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            if model.aux_loss is not None:
                loss = loss + model.aux_loss
            time_forward += time.perf_counter() - mark

            mark = time.perf_counter()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
            optimizer.step()
            time_backward += time.perf_counter() - mark

            train_loss += float(loss.item()) * y.size(0)
            train_correct += int((logits.argmax(1) == y).sum().item())
            train_total += y.size(0)
            if model.last_rate is not None:
                rate_sum += float(model.last_rate.item())
                rate_batches += 1
        scheduler.step()

        measured_rate = rate_sum / max(1, rate_batches)
        if args.governor:
            activity_lambda = governor_update(activity_lambda, measured_rate, budget, args.governor_eta)

        test_loss, test_acc, test_rate = evaluate(model, test_loader, device)
        epoch_time = time.perf_counter() - started

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        if args.checkpoint_every > 0 and (epoch % args.checkpoint_every == 0 or epoch == args.epochs):
            torch.save(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "best_epoch": best_epoch,
                    "history": history,
                    "state_dict": best_state,
                },
                args.output_dir / f"seed_{seed}.ckpt.pt",
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss / max(1, train_total), 4),
                "train_acc": round(train_correct / max(1, train_total), 4),
                "test_loss": round(test_loss, 4),
                "test_acc": round(test_acc, 4),
                "train_event_rate": round(measured_rate, 4),
                "test_event_rate": round(test_rate, 4),
                "event_budget": round(budget, 4),
                "activity_lambda": round(model.activity_lambda, 5),
                "elapsed_s": round(epoch_time, 2),
            }
        )
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"  seed {seed} epoch {epoch:3d}/{args.epochs} | "
                f"train {history[-1]['train_acc']*100:.1f}% | test {test_acc*100:.2f}% | "
                f"best {best_acc*100:.2f}% | events {measured_rate:.4f} (budget {budget:.3f}) | "
                f"lambda {model.activity_lambda:.4f} | {epoch_time:.1f}s "
                f"(data {time_data:.1f}s fwd {time_forward:.1f}s bwd {time_backward:.1f}s)",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    _, float_acc, float_rate = evaluate(model, test_loader, device)
    quantized = quantize_int16(model, device)
    _, int16_acc, int16_rate = evaluate(quantized, test_loader, device)

    return {
        "seed": seed,
        "n_params": n_params,
        "best_test_acc": round(best_acc, 4),
        "best_epoch": best_epoch,
        "final_test_acc": round(float_acc, 4),
        "int16_test_acc": round(int16_acc, 4),
        "final_test_event_rate": round(float_rate, 4),
        "int16_test_event_rate": round(int16_rate, 4),
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ES-FA SHD v5: innovations on a protocol-corrected backbone")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dt-ms", type=float, default=4.0)
    parser.add_argument("--t-max-ms", type=float, default=1000.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--target-rate", type=float, default=0.05)
    parser.add_argument("--activity-lambda", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--dual-readout", action="store_true", help="ES-FA dual-timescale readout")
    parser.add_argument("--governor", action="store_true", help="ES-FA event-rate governor")
    parser.add_argument("--lambda-start", type=float, default=0.01)
    parser.add_argument("--budget-start", type=float, default=0.15)
    parser.add_argument("--budget-end", type=float, default=0.03)
    parser.add_argument("--governor-eta", type=float, default=8.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--no-event-drop", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/shd_v5"))
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
        f"ES-FA SHD v5 | hidden={args.hidden} | dt={args.dt_ms}ms "
        f"| T={int(args.t_max_ms / args.dt_ms)} | epochs={args.epochs} | seeds={seeds} "
        f"| governor={args.governor} | dual_readout={args.dual_readout} | device={device}"
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
        "positioning": "ES-FA innovations (event-rate governor, dual-timescale readout, int16 deployment) on a protocol-corrected backbone; protocol elements attributed to github.com/catalyst-neuromorphic/catalyst-benchmarks (MIT)",
        "config": {
            "hidden": args.hidden,
            "dt_ms": args.dt_ms,
            "t_max_ms": args.t_max_ms,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
            "event_drop": not args.no_event_drop,
            "governor": args.governor,
            "budget_start": args.budget_start,
            "budget_end": args.budget_end,
            "governor_eta": args.governor_eta,
            "dual_readout": args.dual_readout,
        },
        "seeds": seeds,
        "best_acc_mean": round(float(np.mean([r["best_test_acc"] for r in results])), 4),
        "best_acc_max": round(float(np.max([r["best_test_acc"] for r in results])), 4),
        "int16_acc_mean": round(float(np.mean([r["int16_test_acc"] for r in results])), 4),
        "final_event_rate_mean": round(float(np.mean([r["final_test_event_rate"] for r in results])), 4),
        "results": results,
    }
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"\nv5 summary: best mean {summary['best_acc_mean']*100:.2f}% "
        f"max {summary['best_acc_max']*100:.2f}% | int16 mean {summary['int16_acc_mean']*100:.2f}% "
        f"| events {summary['final_event_rate_mean']:.4f}"
    )
    print(f"written: {args.output_dir / 'aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
