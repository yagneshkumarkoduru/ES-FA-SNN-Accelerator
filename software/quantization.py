"""Quantization utilities for hardware-aware SNN training and export."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _int_range(num_bits: int) -> Tuple[int, int]:
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    return qmin, qmax


def calculate_scale(tensor: torch.Tensor, num_bits: int = 8, eps: float = 1e-8) -> torch.Tensor:
    """Return a symmetric quantization scale for a tensor."""
    _, qmax = _int_range(num_bits)
    max_abs = tensor.detach().abs().amax().clamp(min=eps)
    return max_abs / float(qmax)


def quantize_tensor(tensor: torch.Tensor, num_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to signed integer range and return (q_tensor, scale)."""
    qmin, qmax = _int_range(num_bits)
    scale = calculate_scale(tensor, num_bits=num_bits)
    q = torch.round(tensor / scale).clamp(qmin, qmax)
    return q.to(torch.int32), scale


def dequantize_tensor(q_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q_tensor.to(torch.float32) * scale


def fake_quantize_tensor(tensor: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Fake-quantize tensor with straight-through gradient estimator."""
    qmin, qmax = _int_range(num_bits)
    scale = calculate_scale(tensor, num_bits=num_bits)
    q = torch.round(tensor / scale).clamp(qmin, qmax)
    dq = q * scale
    return tensor + (dq - tensor).detach()


class FakeQuantLinear(nn.Module):
    """Linear layer with activation/weight fake quantization for QAT."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_bits: int = 8,
        act_bits: int = 8,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_bits = weight_bits
        self.act_bits = act_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = fake_quantize_tensor(x, num_bits=self.act_bits)
        w_q = fake_quantize_tensor(self.linear.weight, num_bits=self.weight_bits)
        b_q = None
        if self.linear.bias is not None:
            b_q = fake_quantize_tensor(self.linear.bias, num_bits=16)
        return F.linear(x_q, w_q, b_q)


def quantize_state_dict_to_int8(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, np.ndarray]]:
    """Quantize floating-point weights from a state dict into int8 numpy arrays."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for key, tensor in state_dict.items():
        if not key.endswith("weight"):
            continue
        q, scale = quantize_tensor(tensor.detach().cpu(), num_bits=8)
        out[key] = {
            "int8": q.to(torch.int8).numpy(),
            "scale": np.array([scale.item()], dtype=np.float32),
        }
    return out


def int8_tensor_to_hex_lines(tensor: np.ndarray) -> str:
    """Convert an int8 tensor to line-wise two's complement hex text."""
    flat = tensor.astype(np.int8).reshape(-1)
    hex_lines = [f"{np.uint8(v):02x}" for v in flat]
    return "\n".join(hex_lines) + "\n"
