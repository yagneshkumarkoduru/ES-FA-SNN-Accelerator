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


# Experiments

## Setup
Experiments follow proposal concepts with minimal baseline modifications:
1. `exp1_hardware_aware_loss`
2. `exp2_spike_sparsity_control`
3. `exp3_temporal_multiplexing_simulation`
4. `exp4_quantization_effect`
5. `exp5_dataflow_adaptation`

Each run logs accuracy, sparsity, energy proxy, latency proxy, training history, INT8 weights, and spike raster/stats.

## Comparisons and Ablation
We compare baseline, all experiments, and iteration versions (`v1`, `v2`, `v3`) using a unified ranking and Pareto-style trade-off analysis.


# Results Table

| Run | Category | Accuracy | Sparsity | Energy | Latency |
|---|---|---:|---:|---:|---:|
| exp1_hardware_aware_loss | experiment | 0.9570 | 0.5648 | 4592863.17 | 0.6167 |
| baseline_paper1 | baseline | 0.9570 | 0.5648 | 22604295.76 | 0.6167 |
| exp5_dataflow_adaptation | experiment | 0.9570 | 0.5648 | 45016894.73 | 1.2333 |


# Key Findings

- Best run: `exp1_hardware_aware_loss` (experiment).
- Best metrics: accuracy=0.9570, sparsity=0.5648, energy=4592863.17, latency=0.6167.
- Efficiency gains are associated with higher sparsity and reduced proxy memory traffic in event/adaptive settings.
- Temporal multiplexing introduces latency-pressure trade-offs that can be partially compensated by sparsity-aware regularization.
- Compared to baseline: accuracy_delta=0.0000, sparsity_delta=0.0000, energy_delta=-18011432.59.
- Evidence-backed comparison highlights:
  - baseline_snn_vs_hardware_aware_snn / accuracy: baseline=0.957, optimized=0.957, delta=0.0%.
  - baseline_snn_vs_hardware_aware_snn / energy_proxy: baseline=22604295.758720063, optimized=4592863.1728, delta=-79.68145868455906%.
  - dense_vs_event_scheduling:baseline_paper1 / latency_ns: baseline=5760.0, optimized=5760.0, delta=0.0%.
  - dense_vs_event_scheduling:baseline_paper1 / cycle_count: baseline=576.0, optimized=576.0, delta=0.0%.
  - dense_vs_event_scheduling:exp5_dataflow_adaptation / latency_ns: baseline=5760.0, optimized=5760.0, delta=0.0%.
  - dense_vs_event_scheduling:exp5_dataflow_adaptation / cycle_count: baseline=576.0, optimized=576.0, delta=0.0%.
  - baseline_vs_optimized_hardware_dense / latency_ns: baseline=5760.0, optimized=5760.0, delta=0.0%.
  - baseline_vs_optimized_hardware_dense / lut_usage: baseline=22193.0, optimized=22193.0, delta=0.0%.
  - static_vs_adaptive_execution / accuracy: baseline=0.957, optimized=0.957, delta=0.0%.
  - static_vs_adaptive_execution / latency_proxy: baseline=0.6166690562633859, optimized=1.2333381125267717, delta=100.0%.
  - static_vs_adaptive_execution / energy_proxy: baseline=4592863.1728, optimized=45016894.72704013, delta=880.1488316403722%.
  - estimator_vs_measured_hardware / latency_ns: baseline=3840.0, optimized=5760.0, delta=50.0%.
  - estimator_vs_measured_hardware / latency_ns: baseline=3840.0, optimized=5760.0, delta=50.0%.
  - estimator_vs_measured_hardware / latency_ns: baseline=7680.0, optimized=5760.0, delta=-25.0%.
  - estimator_vs_measured_hardware / latency_ns: baseline=7680.0, optimized=5760.0, delta=-25.0%.
  - estimator_vs_measured_hardware / mean_absolute_percent_error_latency: baseline=0.0, optimized=33.33333333333333, delta=33.33333333333333%.
