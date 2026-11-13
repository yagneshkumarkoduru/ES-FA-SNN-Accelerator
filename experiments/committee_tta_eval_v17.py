"""Offline committee + TTA evaluation for v17-class checkpoints (RSFA + HPI + CSMR).

Loads saved member states and evaluates combination strategies:

  1. per-member raw accuracy and firing rate
  2. committee: normalized output averaging over healthy members
  3. committee + event-drop TTA (K masked copies per view)
  4. committee + temporal-warp TTA (fixed speed factors)
  5. committee + combined TTA (warp factors x event-drop copies)

Member specs (comma separated):
  "42:ema" / "123:best"        resolved in --run-dirs as seed_<seed>_ema.pt or
                                seed_<seed>.ckpt.pt (best-validation snapshot)
  "label=/path/member.pt"       explicit EMA state file, default bin width
  "label=/path/member.pt@2.0"   explicit EMA state file at a 2 ms bin width

Members trained at different input bin widths are evaluated on their own test
raster. All rasters come from the same test file in the same sample order, so
per-batch outputs align for voting; labels are cross-checked on every batch.

Example:

    python experiments/committee_tta_eval_v17.py \
        --members "a=/data/m_271.pt,b=/data/m_2ms42.pt@2.0" \
        --warps 0.9,0.95,1.0,1.05,1.1 --tta 8 --eval-repeats 5 --device auto \
        --data-dir ./data --out report.json

The event-drop TTA mirrors the accelerator's event-loss fault model; the
temporal-warp TTA evaluates each sample at slightly different speech rates.
Both are inference-only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_shd_esfa import load_shd_h5  # noqa: E402
from benchmark_shd_v17 import FIXED_EVAL_SEED, SRNNv17, event_drop  # noqa: E402


def _find_member(run_dirs: List[Path], seed: int, kind: str) -> Path:
    for run_dir in run_dirs:
        path = run_dir / (f"seed_{seed}_ema.pt" if kind == "ema" else f"seed_{seed}.ckpt.pt")
        if path.exists():
            return path
    raise FileNotFoundError(f"member seed {seed} ({kind}) not found in {[str(d) for d in run_dirs]}")


def _parse_spec(spec: str, run_dirs: List[Path], default_dt: float) -> Tuple[str, Path, str, float]:
    """Return (label, path, kind, dt_ms) for one member spec."""
    if "=" in spec:
        label, rest = spec.split("=", 1)
        dt = default_dt
        path_str = rest
        if "@" in rest:
            head, tail = rest.rsplit("@", 1)
            try:
                dt = float(tail)
                path_str = head
            except ValueError:
                path_str = rest
        return label, Path(path_str), "ema", dt
    seed_str, kind = spec.split(":")
    seed = int(seed_str)
    return f"{seed}:{kind}", _find_member(run_dirs, seed, kind), kind, default_dt


def _load_member_state(path: Path, kind: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if kind == "ema":
        return payload
    state = payload.get("best_state")
    if state is None:
        raise ValueError(f"checkpoint {path} has no best_state")
    return state


def _build_model(state: Dict[str, torch.Tensor], device: torch.device) -> SRNNv17:
    # Width and depth come from the state dict. The resonant/heterogeneous flags
    # only shape initialization, which load_state_dict overwrites; non-resonant
    # members keep their frozen near-zero gamma from the saved state.
    n_hidden = int(state["i_2_h1.weight"].shape[0])
    n_layers = 3 if "h2_2_h3.weight" in state else (2 if "h1_2_h2.weight" in state else 1)
    model = SRNNv17(n_hidden=n_hidden, n_layers=n_layers, sum_start=11, resonant=True, heterogeneous=True).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _test_loader(data_dir: Path, dt_ms: float, t_max_ms: float, batch_size: int) -> DataLoader:
    test_h5 = data_dir / "SHD" / "shd_test.h5"
    if not test_h5.exists():
        raise FileNotFoundError(f"SHD test file not found: {test_h5}")
    x, y = load_shd_h5(test_h5, dt_ms=dt_ms, t_max_ms=t_max_ms, as_uint8=True)
    return DataLoader(TensorDataset(x.to(torch.uint8), y), batch_size=batch_size, shuffle=False)


def warp_batch(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Fixed-factor temporal warp: stretch/compress time, crop/pad to original T."""
    if abs(factor - 1.0) < 1e-6:
        return x
    _, T, _ = x.shape
    new_t = max(1, int(round(T * factor)))
    xw = F.interpolate(x.permute(0, 2, 1), size=new_t, mode="linear", align_corners=False)
    xw = xw[:, :, :T] if new_t >= T else F.pad(xw, (0, T - new_t))
    return xw.permute(0, 2, 1)


def evaluate_member(model: SRNNv17, loader: DataLoader, device: torch.device, eval_seed: int) -> Tuple[float, float]:
    gen = torch.Generator(device=str(device))
    gen.manual_seed(eval_seed)
    correct = total = batches = 0
    fr_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.float().to(device), y.to(device)
            logits = model(x, gen=gen)
            correct += int((logits.argmax(1) == y).sum().item())
            total += y.size(0)
            if model.last_firing_rate is not None:
                fr_sum += model.last_firing_rate
                batches += 1
    return correct / max(1, total), fr_sum / max(1, batches)


def _member_batch_logits(
    model: SRNNv17,
    x: torch.Tensor,
    copies: int,
    warps: List[float],
    drop_time_prob: float,
    drop_neuron_prob: float,
    gen: torch.Generator,
    rng: random.Random,
) -> torch.Tensor:
    """Row-normalized, TTA-aggregated logits for one member on one batch."""
    out = None
    for f in warps:
        xf = warp_batch(x, f)
        o = model(xf, gen=gen)
        for _ in range(max(0, copies - 1)):
            o = o + model(event_drop(xf, drop_time_prob, drop_neuron_prob, gen=gen, rng=rng), gen=gen)
        out = o if out is None else out + o
    return out / out.sum(dim=1, keepdim=True).clamp_min(1e-9)


def committee_eval(
    models: List[SRNNv17],
    dts: List[float],
    loaders: Dict[float, DataLoader],
    device: torch.device,
    copies: int,
    warps: List[float],
    drop_time_prob: float,
    drop_neuron_prob: float,
    seed: int,
) -> float:
    """Committee with combined TTA: each member contributes one base view plus
    (copies - 1) event-drop copies per warp factor, on its own input raster.
    Member outputs are row-normalized before averaging."""
    gen = torch.Generator(device=str(device))
    gen.manual_seed(seed)
    rng = random.Random(seed)
    order = sorted(set(dts))
    correct = total = 0
    with torch.no_grad():
        for batches in zip(*(loaders[d] for d in order)):
            by_dt = dict(zip(order, batches))
            y = by_dt[order[0]][1].to(device)
            for d in order[1:]:
                if not torch.equal(by_dt[d][1].to(device), y):
                    raise RuntimeError("test rasters out of order across bin widths")
            accum = None
            for m, d in zip(models, dts):
                x = by_dt[d][0].float().to(device)
                out = _member_batch_logits(m, x, copies, warps, drop_time_prob, drop_neuron_prob, gen, rng)
                accum = out if accum is None else accum + out
            correct += int((accum.argmax(1) == y).sum().item())
            total += y.size(0)
    return correct / max(1, total)


def committee_grid(
    models: List[SRNNv17],
    dts: List[float],
    labels: List[str],
    loaders: Dict[float, DataLoader],
    device: torch.device,
    subsets: List[List[int]],
    warps: List[float],
    copies_grid: List[int],
    drop_time_prob: float,
    drop_neuron_prob: float,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Exhaustive committee sweep: every subset x every TTA-copy count, one draw each.

    Subsets are lists of member indices; each is evaluated with the given warp
    grid at each copy count in copies_grid. Deterministic given the seed.
    """
    gen = torch.Generator(device=str(device))
    gen.manual_seed(seed)
    rng = random.Random(seed)
    order = sorted(set(dts))
    results: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for batches in zip(*(loaders[d] for d in order)):
            by_dt = dict(zip(order, batches))
            y = by_dt[order[0]][1].to(device)
            for d in order[1:]:
                if not torch.equal(by_dt[d][1].to(device), y):
                    raise RuntimeError("test rasters out of order across bin widths")
            # Per-member outputs per copy count, cached for this batch.
            per_member: Dict[int, Dict[int, torch.Tensor]] = {}
            for idx, (m, d) in enumerate(zip(models, dts)):
                x = by_dt[d][0].float().to(device)
                per_member[idx] = {
                    c: _member_batch_logits(m, x, c, warps, drop_time_prob, drop_neuron_prob, gen, rng)
                    for c in copies_grid
                }
            for sub in subsets:
                key = "+".join(labels[i] for i in sub)
                entry = results.setdefault(key, {})
                for c in copies_grid:
                    accum = None
                    for i in sub:
                        out = per_member[i][c]
                        accum = out if accum is None else accum + out
                    pred = accum.argmax(1)
                    ckey = f"tta{c}"
                    corr, tot = entry.get(ckey, (0, 0))
                    entry[ckey] = (corr + int((pred == y).sum().item()), tot + y.size(0))
    return {
        k: {ck: round(corr / max(1, tot), 4) for ck, (corr, tot) in sorted(v.items())}
        for k, v in results.items()
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline committee + TTA evaluation (v17-class)")
    p.add_argument("--run-dirs", type=str, default="",
                   help="comma-separated run directories for seed:kind member specs")
    p.add_argument("--members", type=str, required=True, help="comma-separated member specs")
    p.add_argument("--warps", type=str, default="1.0", help="comma list of temporal-warp factors")
    p.add_argument("--tta", type=int, default=1, help="event-drop copies per warp view")
    p.add_argument("--tta-drop-time", type=float, default=0.05)
    p.add_argument("--tta-drop-neuron", type=float, default=0.03)
    p.add_argument("--min-fr", type=float, default=0.002, help="member health floor")
    p.add_argument("--eval-repeats", type=int, default=1, help="repeat the plain committee over N draws")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dt-ms", type=float, default=4.0, help="default member input bin width")
    p.add_argument("--t-max-ms", type=float, default=1000.0)
    p.add_argument("--data-dir", type=Path, default=Path("./data"))
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--out", type=Path, default=None, help="report path")
    p.add_argument("--grid", action="store_true",
                   help="exhaustive sweep: every member subset x {1,tta} copies, warp grid, one draw each")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)

    run_dirs = [Path(p) for p in args.run_dirs.split(",") if p.strip()]
    specs = [_parse_spec(s.strip(), run_dirs, args.dt_ms) for s in args.members.split(",") if s.strip()]
    warps = [float(w) for w in args.warps.split(",") if w.strip()]
    loaders = {dt: _test_loader(args.data_dir, dt, args.t_max_ms, args.batch_size) for dt in sorted({s[3] for s in specs})}

    members: List[SRNNv17] = []
    member_dts: List[float] = []
    member_frs: List[float] = []
    labels: List[str] = []
    report: Dict[str, object] = {"warps": warps, "tta_copies": args.tta, "members": []}

    for idx, (label, path, kind, dt) in enumerate(specs):
        model = _build_model(_load_member_state(path, kind), device)
        acc, fr = evaluate_member(model, loaders[dt], device, FIXED_EVAL_SEED + 10 + idx)
        members.append(model)
        member_dts.append(dt)
        member_frs.append(fr)
        labels.append(label)
        print(f"member {label} (dt {dt} ms): acc {acc*100:.2f}% FR {fr:.4f}", flush=True)
        report["members"].append({"label": label, "dt_ms": dt, "acc": round(acc, 4), "fr": round(fr, 4)})  # type: ignore[union-attr]

    healthy = [i for i, fr in enumerate(member_frs) if fr >= args.min_fr]
    print(f"healthy members: {[labels[i] for i in healthy]}", flush=True)

    if len(healthy) < 2:
        print("need at least two healthy members for a committee", flush=True)
    else:
        sub = [members[i] for i in healthy]
        sub_dts = [member_dts[i] for i in healthy]
        sub_labels = [labels[i] for i in healthy]

        def run(copies: int, ws: List[float], seed: int) -> float:
            return committee_eval(sub, sub_dts, loaders, device, copies, ws, args.tta_drop_time, args.tta_drop_neuron, seed)

        plain = run(1, [1.0], FIXED_EVAL_SEED + 50)
        print(f"committee (no TTA): {plain*100:.2f}%", flush=True)
        report["committee_plain"] = round(plain, 4)
        report["committee_members"] = len(sub)

        if args.eval_repeats > 1:
            vals = []
            for rep in range(args.eval_repeats):
                acc = run(1, [1.0], FIXED_EVAL_SEED + 300 + 17 * rep)
                vals.append(acc)
                print(f"committee draw {rep+1}/{args.eval_repeats}: {acc*100:.2f}%", flush=True)
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
            print(f"committee over {len(vals)} draws: {mean*100:.2f}% +/- {std*100:.2f}pp", flush=True)
            report["committee_repeats_mean"] = round(mean, 4)
            report["committee_repeats_std"] = round(std, 4)

        if args.tta > 1:
            acc = run(args.tta, [1.0], FIXED_EVAL_SEED + 100 + args.tta)
            print(f"committee + event-drop TTA{args.tta}: {acc*100:.2f}%", flush=True)
            report["committee_tta_drop"] = round(acc, 4)

        if len(warps) > 1 or abs(warps[0] - 1.0) > 1e-6:
            acc = run(1, warps, FIXED_EVAL_SEED + 200)
            print(f"committee + warp TTA{warps}: {acc*100:.2f}%", flush=True)
            report["committee_tta_warp"] = round(acc, 4)
            if args.tta > 1:
                acc = run(args.tta, warps, FIXED_EVAL_SEED + 400)
                print(f"committee + warp x event-drop TTA: {acc*100:.2f}%", flush=True)
                report["committee_tta_both"] = round(acc, 4)

        if args.grid:
            # Every healthy subset of size >= 2, warp grid, copies in {1, args.tta}.
            from itertools import combinations

            subsets: List[List[int]] = []
            for size in range(2, len(sub) + 1):
                subsets.extend([list(c) for c in combinations(range(len(sub)), size)])
            copies_grid = sorted({1, max(1, args.tta)})
            print(f"grid: {len(subsets)} subsets x {copies_grid} copy counts over {len(warps)} warps", flush=True)
            grid = committee_grid(
                sub, sub_dts, sub_labels, loaders, device, subsets, warps,
                copies_grid, args.tta_drop_time, args.tta_drop_neuron, FIXED_EVAL_SEED + 500,
            )
            report["grid"] = grid
            ranked = sorted(
                ((k, ck, acc) for k, e in grid.items() for ck, acc in e.items()),
                key=lambda t: t[2], reverse=True,
            )
            for k, ck, acc in ranked[:15]:
                print(f"grid top: {acc*100:.2f}%  {ck}  {k}", flush=True)

    if args.out is not None:
        out_path = args.out
    elif run_dirs:
        out_path = run_dirs[0] / "offline_committee_v17.json"
    else:
        out_path = Path("offline_committee_v17.json")
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
