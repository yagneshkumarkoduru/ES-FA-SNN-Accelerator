# Method

## SNN Model
We use a proposal-aligned feed-forward SNN with architecture `784 -> 128 -> 64 -> 10`.
The hidden layers are Leaky Integrate-and-Fire (LIF) neurons trained with surrogate gradients.
Inputs are encoded as Poisson/rate spikes over fixed simulation windows (`T` timesteps).

## Hardware-Aware Loss
Training optimizes cross-entropy with optional regularizers:
- energy-aware term from the software hardware estimator,
- spike sparsity control term,
- latency proxy regularization under temporal multiplexing assumptions.

## Software Hardware Estimator (Paper-2 Integration)
At each batch, we estimate:
1. `estimate_spike_cost()` for per-layer/total spikes
2. `estimate_memory_access()` for read/write access under dense/event/adaptive dataflow
3. `estimate_energy_proxy()` as `E = a * spikes + b * memory_accesses`
4. `estimate_latency_proxy()` from sparsity + multiplexing + queue pressure
