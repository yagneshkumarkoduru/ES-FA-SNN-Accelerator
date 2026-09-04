"""Export utilities for FPGA handoff (INT8 weights and spike stats)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


def _int_range(num_bits: int) -> Tuple[int, int]:
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    return qmin, qmax


def quantize_tensor(tensor: torch.Tensor, num_bits: int = 8) -> Tuple[torch.Tensor, float]:
    qmin, qmax = _int_range(num_bits)
    max_abs = tensor.detach().abs().amax().clamp(min=1e-8)
    scale = float((max_abs / float(qmax)).item())
    q = torch.round(tensor / scale).clamp(qmin, qmax).to(torch.int32)
    return q, scale


def export_int8_weights(model: torch.nn.Module, out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    meta = {"layers": []}

    packed = {}
    for key, tensor in state.items():
        if not key.endswith("linear.weight"):
            continue
        layer_name = key.split(".")[0]
        q, scale = quantize_tensor(tensor.detach().cpu(), num_bits=8)
        q_i8 = q.to(torch.int8).numpy()

        layer_dir = out_dir / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)
        np.save(layer_dir / "weight_int8.npy", q_i8)

        with (layer_dir / "weight_int8.mem").open("w", encoding="utf-8") as f:
            for v in q_i8.reshape(-1):
                f.write(f"{np.uint8(v):02x}\n")

        packed[f"{layer_name}_weight_int8"] = q_i8
        meta["layers"].append(
            {"layer": layer_name, "weight_shape": list(tensor.shape), "weight_scale": scale}
        )

    np.savez(out_dir / "weights_int8.npz", **packed)
    with (out_dir / "weights_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def export_spike_stats(spike_summary: Dict[str, float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "spike_stats.json").open("w", encoding="utf-8") as f:
        json.dump(spike_summary, f, indent=2)

