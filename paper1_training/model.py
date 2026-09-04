"""SNN model definition: 784 -> 128 -> 64 -> 10 with LIF neurons."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _int_range(num_bits: int) -> Tuple[int, int]:
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    return qmin, qmax


def _fake_quantize(tensor: torch.Tensor, num_bits: int = 8, eps: float = 1e-8) -> torch.Tensor:
    qmin, qmax = _int_range(num_bits)
    max_abs = tensor.detach().abs().amax().clamp(min=eps)
    scale = max_abs / float(qmax)
    q = torch.round(tensor / scale).clamp(qmin, qmax)
    dq = q * scale
    return tensor + (dq - tensor).detach()


class FakeQuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_bits: int = 8,
        act_bits: int = 8,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_bits = weight_bits
        self.act_bits = act_bits

    def forward(self, x: torch.Tensor, quant_enabled: bool = False) -> torch.Tensor:
        if quant_enabled:
            x = _fake_quantize(x, self.act_bits)
            w = _fake_quantize(self.linear.weight, self.weight_bits)
            b = None if self.linear.bias is None else _fake_quantize(self.linear.bias, 16)
            return F.linear(x, w, b)
        return self.linear(x)


class SurrogateSpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0.0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        slope = ctx.slope
        grad = grad_output / (1.0 + slope * x.abs()).pow(2)
        return grad, None


def surrogate_spike(x: torch.Tensor, slope: float) -> torch.Tensor:
    return SurrogateSpikeFn.apply(x, slope)


class LIFLayer(nn.Module):
    def __init__(self, size: int, beta: float = 0.9, threshold: float = 1.0, surrogate_slope: float = 5.0) -> None:
        super().__init__()
        self.size = size
        self.beta = beta
        self.threshold = threshold
        self.surrogate_slope = surrogate_slope

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.size, device=device)

    def forward(self, current: torch.Tensor, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mem = self.beta * mem + current
        spk = surrogate_spike(mem - self.threshold, self.surrogate_slope)
        mem = mem - spk * self.threshold
        return spk, mem


class ProposalAlignedSNN(nn.Module):
    """Proposal baseline SNN: 784 -> 128 -> 64 -> 10 with LIF hidden layers."""

    def __init__(
        self,
        time_steps: int = 16,
        encoder: str = "poisson",
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_slope: float = 5.0,
        quant_enabled: bool = False,
        weight_bits: int = 8,
        act_bits: int = 8,
    ) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.encoder = encoder
        self.quant_enabled = quant_enabled

        self.fc1 = FakeQuantLinear(784, 128, weight_bits=weight_bits, act_bits=act_bits)
        self.fc2 = FakeQuantLinear(128, 64, weight_bits=weight_bits, act_bits=act_bits)
        self.fc3 = FakeQuantLinear(64, 10, weight_bits=weight_bits, act_bits=act_bits)
        self.lif1 = LIFLayer(128, beta=beta, threshold=threshold, surrogate_slope=surrogate_slope)
        self.lif2 = LIFLayer(64, beta=beta, threshold=threshold, surrogate_slope=surrogate_slope)

    def _encode_step(self, x_flat: torch.Tensor, t: int) -> torch.Tensor:
        if self.encoder == "rate":
            # Deterministic rate coding: exactly floor(x*T) spikes spread across early timesteps.
            return ((x_flat * self.time_steps) > float(t)).to(x_flat.dtype)
        # Poisson/Bernoulli approximation.
        return (torch.rand_like(x_flat) < x_flat).to(x_flat.dtype)

    def forward(self, x: torch.Tensor, capture_raster: bool = False) -> Tuple[torch.Tensor, Dict[str, object]]:
        x_flat = x.view(x.size(0), -1).clamp(0.0, 1.0)
        batch = x_flat.size(0)
        device = x.device

        mem1 = self.lif1.init_state(batch, device)
        mem2 = self.lif2.init_state(batch, device)
        out_acc = torch.zeros(batch, 10, device=device)

        input_spikes = 0.0
        layer1_spikes = 0.0
        layer2_spikes = 0.0

        raster_l1 = []
        raster_l2 = []

        for t in range(self.time_steps):
            x_spk = self._encode_step(x_flat, t)
            cur1 = self.fc1(x_spk, quant_enabled=self.quant_enabled)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1, quant_enabled=self.quant_enabled)
            spk2, mem2 = self.lif2(cur2, mem2)
            out_acc = out_acc + self.fc3(spk2, quant_enabled=self.quant_enabled)

            input_spikes += float(x_spk.sum().item())
            layer1_spikes += float(spk1.sum().item())
            layer2_spikes += float(spk2.sum().item())

            if capture_raster:
                raster_l1.append(spk1[0].detach().cpu())
                raster_l2.append(spk2[0].detach().cpu())

        logits = out_acc / float(self.time_steps)
        stats: Dict[str, object] = {
            "spike_counts": {
                "input": input_spikes,
                "layer1": layer1_spikes,
                "layer2": layer2_spikes,
            },
            "possible_spikes": {
                "input": float(batch * self.time_steps * 784),
                "layer1": float(batch * self.time_steps * 128),
                "layer2": float(batch * self.time_steps * 64),
            },
        }

        if capture_raster:
            stats["raster"] = {
                "layer1": torch.stack(raster_l1, dim=0).numpy(),
                "layer2": torch.stack(raster_l2, dim=0).numpy(),
            }

        return logits, stats

