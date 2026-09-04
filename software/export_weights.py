"""Export trained SNN weights to INT8 files suitable for FPGA initialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from quantization import int8_tensor_to_hex_lines, quantize_tensor


def _layer_key_to_name(state_key: str) -> str:
    # Example: fc1.linear.weight -> fc1
    return state_key.split(".")[0]


def export_layer(
    layer_name: str,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out_dir: Path,
) -> Dict[str, object]:
    q_weight, w_scale = quantize_tensor(weight.detach().cpu(), num_bits=8)
    w_i8 = q_weight.to(torch.int8).numpy()

    layer_dir = out_dir / layer_name
    layer_dir.mkdir(parents=True, exist_ok=True)

    np.save(layer_dir / "weight_int8.npy", w_i8)
    with (layer_dir / "weight_int8.mem").open("w", encoding="utf-8") as f:
        f.write(int8_tensor_to_hex_lines(w_i8))

    meta: Dict[str, object] = {
        "layer": layer_name,
        "weight_shape": list(weight.shape),
        "weight_scale": float(w_scale.item()),
    }

    if bias is not None:
        # Bias is exported as int16 for extra dynamic range.
        b_scale = w_scale
        b_q = torch.round(bias.detach().cpu() / b_scale).clamp(-32768, 32767).to(torch.int16)
        b_i16 = b_q.numpy()
        np.save(layer_dir / "bias_int16.npy", b_i16)
        with (layer_dir / "bias_int16.mem").open("w", encoding="utf-8") as f:
            for v in b_i16.reshape(-1):
                f.write(f"{np.uint16(v):04x}\n")
        meta["bias_shape"] = list(bias.shape)
        meta["bias_scale"] = float(b_scale.item())

    return meta


def collect_layers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor | None]]:
    layers: Dict[str, Tuple[torch.Tensor, torch.Tensor | None]] = {}
    for key, tensor in state_dict.items():
        if key.endswith("weight") and ".linear." in key:
            lname = _layer_key_to_name(key)
            bkey = key.replace("weight", "bias")
            bias = state_dict.get(bkey)
            layers[lname] = (tensor, bias)
    return layers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export trained SNN weights to INT8.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint from train_snn.py")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./artifacts/export_int8"),
        help="Directory to write FPGA-ready quantized files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing `model_state_dict`.")
    state_dict = ckpt["model_state_dict"]

    layers = collect_layers(state_dict)
    if not layers:
        raise RuntimeError("No linear layer weights found. Expected keys like `fc1.linear.weight`.")

    export_meta = {
        "checkpoint": str(args.checkpoint.resolve()),
        "layers": [],
    }

    packed: Dict[str, np.ndarray] = {}
    for lname, (weight, bias) in layers.items():
        meta = export_layer(lname, weight, bias, args.output_dir)
        export_meta["layers"].append(meta)
        q, _ = quantize_tensor(weight.detach().cpu(), num_bits=8)
        packed[f"{lname}_weight_int8"] = q.to(torch.int8).numpy()

    np.savez(args.output_dir / "weights_int8.npz", **packed)
    with (args.output_dir / "export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2)

    print(f"Export complete. Files written to: {args.output_dir.resolve()}")
    for layer in export_meta["layers"]:
        print(
            f"  - {layer['layer']}: shape={layer['weight_shape']}, "
            f"scale={layer['weight_scale']:.8f}"
        )


if __name__ == "__main__":
    main()
