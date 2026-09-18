"""Offline committee + TTA evaluation from saved EMA checkpoints.

Loads the per-seed EMA state dicts saved by benchmark_shd_v12.py, rebuilds
the models, and evaluates several combination strategies entirely on local
CPU (no cloud compute):

  1. per-member EMA accuracy
  2. committee: normalized output averaging over healthy members
  3. committee + event-loss TTA (K masked copies per member)
  4. weight soup: the three EMA states averaged into one model
  5. optional leave-one-out committee

Usage:
    py -3 experiments/committee_tta_eval.py \
        --run-dirs results/cloud_shd_v5_20260918-031418/results/shd_v12_bounded,results/cloud_shd_v5_NEW/results/shd_v12_committee \
        --hidden 160 --seeds 42,123,999,7,2026 --tta 4,8 --min-fr 0.02
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_shd_v12 import (  # noqa: E402
    FIXED_EVAL_SEED,
    SRNNv12,
    event_drop,
    make_loaders,
)


def _load_ema(run_dirs: List[Path], seed: int) -> Dict[str, torch.Tensor]:
    for run_dir in run_dirs:
        path = run_dir / f"seed_{seed}_ema.pt"
        if path.exists():
            return torch.load(path, map_location="cpu")
    raise FileNotFoundError(f"EMA state for seed {seed} not found in {[str(d) for d in run_dirs]}")


def _build_model(state: Dict[str, torch.Tensor], args: argparse.Namespace, device: torch.device) -> SRNNv12:
    n_hidden = int(state["i_2_h1.weight"].shape[0])
    model = SRNNv12(
        n_hidden=n_hidden,
        n_layers=args.layers,
        beta=args.beta,
        b0=args.b0,
        sum_start=args.sum_start,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_member(
    model: SRNNv12,
    loader: DataLoader,
    device: torch.device,
    eval_seed: int,
) -> Tuple[float, float]:
    gen = torch.Generator(device=str(device))
    gen.manual_seed(eval_seed)
    correct = 0
    total = 0
    fr_sum = 0.0
    batches = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.to(device)
            logits = model(x, gen=gen)
            correct += int((logits.argmax(1) == y).sum().item())
            total += y.size(0)
            if model.last_firing_rate is not None:
                fr_sum += model.last_firing_rate
                batches += 1
    return correct / max(1, total), fr_sum / max(1, batches)


def committee_tta(
    models: List[SRNNv12],
    loader: DataLoader,
    device: torch.device,
    copies: int,
    drop_time_prob: float,
    drop_neuron_prob: float,
    seed: int,
) -> float:
    """Committee with event-loss TTA: each member contributes K masked views;
    member outputs are row-normalized before averaging."""
    gen = torch.Generator(device=str(device))
    gen.manual_seed(seed)
    rng = random.Random(seed)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.to(device)
            accum = None
            for m in models:
                out = m(x, gen=gen)
                for _ in range(max(0, copies - 1)):
                    xm = event_drop(
                        x,
                        drop_time_prob=drop_time_prob,
                        drop_neuron_prob=drop_neuron_prob,
                        gen=gen,
                        rng=rng,
                    )
                    out = out + m(xm, gen=gen)
                out = out / out.sum(dim=1, keepdim=True).clamp_min(1e-9)
                accum = out if accum is None else accum + out
            correct += int((accum.argmax(1) == y).sum().item())
            total += y.size(0)
    return correct / max(1, total)


def soup_eval(models: List[SRNNv12], loader: DataLoader, device: torch.device, args: argparse.Namespace, seed: int) -> float:
    """Average the member state dicts into one model and evaluate it."""
    keys = models[0].state_dict().keys()
    soup: Dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [m.state_dict()[k].float() for m in models]
        soup[k] = torch.stack(vals).mean(dim=0)
    soup_model = _build_model(soup, args, device)
    acc, _ = evaluate_member(soup_model, loader, device, seed)
    return acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline committee + TTA evaluation")
    parser.add_argument("--run-dirs", type=str, required=True,
                        help="comma-separated run directories to search for seed_<seed>_ema.pt files")
    parser.add_argument("--seeds", type=str, default="42,123,999")
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1.8)
    parser.add_argument("--b0", type=float, default=0.01)
    parser.add_argument("--sum-start", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dt-ms", type=float, default=4.0)
    parser.add_argument("--t-max-ms", type=float, default=1000.0)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--tta", type=str, default="4,8", help="comma list of TTA copies for committee+TTA")
    parser.add_argument("--tta-drop-time", type=float, default=0.05)
    parser.add_argument("--tta-drop-neuron", type=float, default=0.03)
    parser.add_argument("--min-fr", type=float, default=0.02)
    parser.add_argument("--eval-repeats", type=int, default=1,
                        help="repeat the no-TTA committee over this many evaluation draws (mean and std reported)")
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    run_dirs = [Path(p) for p in args.run_dirs.split(",") if p.strip()]
    members: List[SRNNv12] = []
    member_frs: List[float] = []
    report: Dict[str, object] = {"run_dirs": [str(d) for d in run_dirs], "seeds": []}

    _, test_loader = make_loaders(args.data_dir, args.batch_size, args.dt_ms, args.t_max_ms, seeds[0])

    for idx, seed in enumerate(seeds):
        state = _load_ema(run_dirs, seed)
        model = _build_model(state, args, device)
        acc, fr = evaluate_member(model, test_loader, device, FIXED_EVAL_SEED + 10 + idx)
        members.append(model)
        member_frs.append(fr)
        print(f"member seed {seed}: acc {acc*100:.2f}% FR {fr:.4f}", flush=True)
        report["seeds"].append({"seed": seed, "acc": round(acc, 4), "fr": round(fr, 4)})  # type: ignore[union-attr]

    healthy = [i for i, fr in enumerate(member_frs) if fr >= args.min_fr]
    print(f"healthy members: {[seeds[i] for i in healthy]}", flush=True)

    if len(healthy) >= 2:
        sub = [members[i] for i in healthy]
        committee_plain = committee_tta(sub, test_loader, device, 1, args.tta_drop_time, args.tta_drop_neuron, FIXED_EVAL_SEED + 50)
        print(f"committee (no TTA): {committee_plain*100:.2f}%", flush=True)
        report["committee_plain"] = round(committee_plain, 4)
        report["committee_members"] = len(sub)

        if args.eval_repeats > 1:
            vals = []
            for rep in range(args.eval_repeats):
                acc = committee_tta(sub, test_loader, device, 1, args.tta_drop_time, args.tta_drop_neuron, FIXED_EVAL_SEED + 300 + 17 * rep)
                vals.append(acc)
                print(f"committee eval draw {rep+1}/{args.eval_repeats}: {acc*100:.2f}%", flush=True)
            mean = float(sum(vals) / len(vals))
            std = float((sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5)
            print(f"committee over {len(vals)} draws: {mean*100:.2f}% +/- {std*100:.2f}pp", flush=True)
            report["committee_eval_repeats_mean"] = round(mean, 4)
            report["committee_eval_repeats_std"] = round(std, 4)
            report["committee_eval_repeats_n"] = len(vals)

        for copies in [int(c) for c in args.tta.split(",") if c.strip()]:
            acc = committee_tta(sub, test_loader, device, copies, args.tta_drop_time, args.tta_drop_neuron, FIXED_EVAL_SEED + 100 + copies)
            print(f"committee + TTA{copies}: {acc*100:.2f}%", flush=True)
            report[f"committee_tta{copies}"] = round(acc, 4)

    if len(members) >= 2:
        soup_acc = soup_eval(members, test_loader, device, args, FIXED_EVAL_SEED + 200)
        print(f"weight soup ({len(members)} members): {soup_acc*100:.2f}%", flush=True)
        report["soup_acc"] = round(soup_acc, 4)

    out_path = run_dirs[0] / "offline_committee_tta.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
