"""Software-level hardware estimators aligned with ES-FA proposal Paper-2 concepts."""

from __future__ import annotations

from typing import Dict


def estimate_spike_cost(spike_counts: Dict[str, float]) -> Dict[str, float]:
    """Estimate per-layer and total spike activity cost."""
    total_spikes = float(sum(spike_counts.values()))
    out = {f"{k}_spikes": float(v) for k, v in spike_counts.items()}
    out["total_spikes"] = total_spikes
    return out


def estimate_memory_access(
    spike_counts: Dict[str, float],
    batch_size: int,
    time_steps: int,
    dataflow_mode: str = "dense",
    density_threshold: float = 0.10,
    temporal_multiplex_factor: int = 1,
) -> Dict[str, float]:
    """Estimate memory read/write volume for dense, event, or adaptive dataflow.

    Model assumptions (proposal-scale MLP):
    - FC1: 784 -> 128
    - FC2: 128 -> 64
    - FC3: 64 -> 10
    """
    dense_weight_reads = float(batch_size * time_steps * ((784 * 128) + (128 * 64) + (64 * 10)))
    dense_state_rw = float(batch_size * time_steps * ((128 + 64) * 2))  # read + write

    input_spikes = float(spike_counts.get("input", 0.0))
    layer1_spikes = float(spike_counts.get("layer1", 0.0))
    layer2_spikes = float(spike_counts.get("layer2", 0.0))

    event_weight_reads = input_spikes * 128.0 + layer1_spikes * 64.0 + layer2_spikes * 10.0
    event_state_rw = 2.0 * (layer1_spikes + layer2_spikes)

    possible_input_spikes = max(1.0, float(batch_size * time_steps * 784))
    input_density = input_spikes / possible_input_spikes

    chosen_mode = dataflow_mode
    if dataflow_mode == "adaptive":
        chosen_mode = "event" if input_density <= density_threshold else "dense"

    if chosen_mode == "event":
        read_accesses = event_weight_reads + (event_state_rw / 2.0)
        write_accesses = event_state_rw / 2.0
    else:
        read_accesses = dense_weight_reads + (dense_state_rw / 2.0)
        write_accesses = dense_state_rw / 2.0

    # Temporal multiplexing increases effective reuse cycles and queue pressure proxy.
    temporal_factor = max(1, int(temporal_multiplex_factor))
    adjusted_total = (read_accesses + write_accesses) * temporal_factor

    return {
        "mode_used": 0.0 if chosen_mode == "dense" else 1.0,
        "input_density": input_density,
        "read_accesses": read_accesses,
        "write_accesses": write_accesses,
        "total_accesses": read_accesses + write_accesses,
        "adjusted_total_accesses": adjusted_total,
        "dense_total_accesses": dense_weight_reads + dense_state_rw,
        "event_total_accesses": event_weight_reads + event_state_rw,
    }


def estimate_energy_proxy(
    total_spikes: float,
    total_memory_accesses: float,
    a: float = 1.0,
    b: float = 0.2,
) -> float:
    """Energy proxy = a*(spikes) + b*(memory accesses)."""
    return float(a * total_spikes + b * total_memory_accesses)


def estimate_latency_proxy(
    spike_sparsity: float,
    temporal_multiplex_factor: int = 1,
    queue_pressure: float = 0.0,
    base_latency: float = 1.0,
) -> float:
    """Latency proxy based on activity and scheduling pressure.

    Higher sparsity should reduce latency in event-driven paths.
    Multiplexing and queue pressure increase effective latency.
    """
    sparsity = min(max(float(spike_sparsity), 0.0), 1.0)
    activity = 1.0 - sparsity
    tm_factor = float(max(1, int(temporal_multiplex_factor)))
    return float(base_latency * tm_factor * (0.15 + activity + max(0.0, queue_pressure)))

