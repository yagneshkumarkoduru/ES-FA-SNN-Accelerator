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
