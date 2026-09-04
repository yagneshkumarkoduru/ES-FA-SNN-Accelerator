# ES-FA Stage-1 + Stage-2 Architecture

## 1. Stage-1 Software SNN (Hardware-Aware)

### Network
- Input flatten: `28x28 -> 784`
- Fully-connected stack: `784 -> 128 -> 64 -> 10`
- Hidden neuron model: LIF (`lif1`, `lif2`)
- Temporal coding: rate encoding over `T` timesteps

### Training Features
- Surrogate gradient for spike function (fast-sigmoid derivative)
- Fake quantization (QAT-style) for weights/activations in all FC layers
- Spike sparsity regularization via `sparsity_lambda`

### Hardware-Aware Metrics
For every batch and timestep:
- Input spike count
- Hidden-layer spike counts
- Spike sparsity (density complement)
- Dense weight access estimate
- Event-driven weight access estimate:
  - `input_spikes * 128 + layer1_spikes * 64 + layer2_spikes * 10`
- Event-driven neuron state accesses:
  - `2 * event_weight_accesses` (read+write model)

Outputs from training:
- Accuracy (`train_acc`, `val_acc`)
- Sparsity statistics per layer
- Estimated access reduction percentage
- Checkpoints for export

## 2. Stage-2 FPGA Architecture

### Top Dataflow
`Input Events -> Spike Router -> Scheduler -> Memory Read -> LIF PE -> Neuron State Writeback -> Output Spikes`

### Modules

1. `memory/neuron_bram.v`
- Dual-port BRAM-inferred memory
- Port A: scheduled state read
- Port B: writeback from PE

2. `memory/weight_bram_bank.v`
- INT8 banked synaptic memory (2 banks)
- Two read request ports with simple arbitration
- Same-bank conflict: requester 0 priority

3. `compute/lif_neuron_pe.v`
- Fixed-point signed datapath (`STATE_WIDTH=16`)
- 4-stage pipeline:
  1. latch read state + syn input
  2. leak+integrate
  3. threshold check/reset and spike generation
  4. writeback payload output

4. `routing/spike_router.v`
- Dense mode: route all incoming events
- Sparse mode: route only active spikes

5. `scheduler/basic_scheduler.v`
- Round-robin neuron assignment
- One scheduled op per accepted input event

6. `scheduler/event_queue.v`
- BRAM-backed queue with FIFO core
- Approximate priority pop by timestamp comparison of first two entries
- Oldest-first behavior without full heap complexity

7. `scheduler/advanced_scheduler.v`
- Event-driven scheduling around `event_queue`
- Push active events, pop for PE when ready

8. `top/snn_top.v`
- Integrates router, both schedulers, queue-based advanced path, memories, and PE
- Runtime mode switch:
  - `mode_advanced=0` -> basic round-robin path
  - `mode_advanced=1` -> event-driven path
- Exposes operation counters:
  - `basic_op_count`, `advanced_op_count`, `pe_op_count`

## 3. Validation Strategy

Testbenches cover:
- Individual modules (`tb_*.v`)
- Full top-level side-by-side comparison (`tb_top.v`)

`tb_top.v` instantiates two tops with identical input stream:
- Basic mode instance
- Advanced mode instance

Success signal in simulation:
- `advanced_ops < basic_ops`
- Demonstrates operation reduction under sparse input activity

## 4. Local Hardware Validation Stack (KV260-Centric)

The evolved project adds a wrapper stack under `hardware_validation/kv260/`:

- `tb/tb_kv260_modes.v`: mode-specific dense/event simulation harness.
- `scripts/run_xsim_regression.py`: runs `xvlog/xelab/xsim` and extracts cycle/latency metrics.
- `vivado/build.tcl`: batch synthesis/implementation/timing flow.
- `scripts/parse_vivado_reports.py`: normalizes utilization/timing into JSON.
- `scripts/collect_hardware_metrics.py`: merges xsim + Vivado metrics with estimator references.

All outputs are persisted to:
`results/hardware_validation/<model>/<mode>/<run-id>/`

This keeps the original RTL untouched while enabling repeatable local hardware evidence generation.
