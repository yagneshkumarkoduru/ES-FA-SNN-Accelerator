# Known Bugs and Findings Register

Date: 2026-09-17
Method: three review passes over the full repository (one manual deep pass on
the cloud-critical path, two parallel agent sweeps over all remaining
directories including C, Verilog, and the Vivado/nextpnr tooling). Every
finding below cites the exact location. Fixed items are recorded too so the
trail is complete.

## Fixed 2026-09-17

| Location | Severity | Fix |
| :--- | :--- | :--- |
| `cloud/run_shd_train.py` | CRITICAL | Replaced all presigned-URL transfers (which expire with the aws-login session token in 10-20 minutes; measured) with instance-profile credentials via IMDS and `aws s3 cp`. Added 10-minute heartbeat uploading the log plus a partial results zip. Instance now stops instead of terminating so results survive on EBS. SSM Run Command enabled for live debugging. |
| `iterations/common.py:79` | CRITICAL | `_load_config` now merges `training_core.load_config` defaults. This fixes the missing-key crash in `train_from_config` (`KeyError: 'device'`) that aborted the initial and combined iteration stages, and the `quantization` `KeyError` in `_merge_for_combined`. |
| `hardware/compute/adaptive_leak_engine.v:49` | MODERATE | Zero-extend the 6-bit window count to 8 bits before the `<< RATE_FRAC_BITS` shift. Previously bits were discarded in 6-bit self-determined width, e.g. count 32 gave rate 0 instead of 128, which inverted the homeostatic adaptation direction for counts >= 16. |
| `hardware/compute/spike_driven_flash_attention.v` (start_attn block) | MODERATE | `start_attn` now clears `attn_map_acc` as well as `score_row_tag`. Previously a re-run whose first query row was token 0 accumulated onto the previous run's scores. |
| `hardware_validation/fpga_board/verilog/rtl_filelist.f` | LOW | Corrected all eight paths from four `../` (which overshoot the repo root) to three. |
| `analysis/compare.py:186` | LOW | `np.load` of the raster archive now uses a context manager so the file handle is released deterministically. |
| `experiments/benchmark_snntorch_baseline.py:144` | MODERATE | Documented that the sparsity penalty is derived from detached `.item()` floats and has zero gradient; the term affects reported loss only. |

Earlier same-day fixes (commit dbc60dc): documented the same detached-loss
pattern in `experiments/benchmark_shd_esfa.py`, `experiments/benchmark_shd_rplif.py`,
`software/train_snn.py`, `p1_training/training_core.py`; documented the latent
in-place `weight.data` mutation in `benchmark_shd_rplif.py`; added the
`--data-dir` argument to the cloud driver; replaced silent setup failures with
FATAL exits; pre-staged the SHD dataset to S3.

## Open (not fixed; need design work, simulation, or an evidence decision)

| Location | Severity | Finding |
| :--- | :--- | :--- |
| `hardware/top/snn_top.v:226` | CRITICAL | Back-to-back events for the same neuron silently drop a membrane update: the scheduler pops one event per cycle and `pe_ready` is tied high, while the PE write-back is visible only ~6 cycles later. `tb_top.v:138-140` spaces same-neuron events 32 cycles apart, so the testbenches never exercise the hazard. Fix requires a scheduler/PE interlock plus a new directed testbench. |
| `hardware_validation/fpga_board/vivado/constraints_fpga_board.xdc:3` + `parse_vivado_reports.py:83-86` | MODERATE | The XDC constrains a 5.000 ns (200 MHz) clock, but `run_hw_validation.py` passes `--clock-mhz 100` and the Fmax formula assumes the caller's period. The shipped `raw_hardware_metrics.csv` records `fmax_mhz_estimated=91.58`, whose true value from the same WNS is ~169 MHz. Requires either regenerating the evidence or amending the recorded derivation. |
| `hardware/tb/tb_snn_generic_top.v:138` | MODERATE | The bench prints PASS unconditionally and performs no comparisons, yet the open-CAD report counts it toward "9/9 testbenches pass". Either make it self-checking or re-label the 9/9 claim to distinguish self-checking from smoke benches. |
| `hardware/top/snn_accelerator_generic.v:338-342` | MODERATE | AXI4-Stream egress ignores `m_axis_spike_tready` (one-cycle pulse, event lost on backpressure) and `irq_overflow` is hardwired low, so dropped egress spikes are unreported. |
| `hardware_validation/fpga_board/scripts/run_model_matrix.py:26-41` | LOW | `quantization_mode` and `dataset_split` declared in `model_matrix.json` are never forwarded, so the matrix can disagree with the evidence it generates (example: exp5 declares int8, the CSV records fp32). |
| `scripts/run_comprehensive_benchmarks.py:98-106` | MODERATE | Plot x-labels truncate `sparsities` from the end (`[:len(...)]`) instead of aligning with successful runs; a failed middle run mislabels the figure. Track the sparsity value alongside each appended result. |
| `deployment/cli_interface/main.py:94-97` | LOW | The `--json` flag is a no-op: both branches print identical JSON. |
| `system_layer/modules/snn_module/module.py:64` | LOW | `execute()` checks `exists()` but not `is_file()`, so a directory argument raises an unhandled error instead of the structured failure payload. |
| `c_engine/snn_engine.c:65,190,301` | LOW | "Input Spikes Ingested" counts internally routed spikes (via `snn_enqueue_spike`), so the metric is inflated and not comparable between baseline and event modes. |
| `c_engine/snn_engine.c:238` + `main.c:18` | LOW | `--neurons 0` reaches `rand() % sys->neurons_per_core` and faults; `--cores 0` yields 0/0 NaN. Add CLI range validation. |
| `experiments/benchmark_snntorch_baseline.py:198` | MODERATE | Model selection uses the SHD test set directly (`best_test_acc` = max over epochs); the headline accuracy is optimistically biased. Needs a held-out validation split or honest labelling. |
| `experiments/benchmark_snntorch_baseline.py:89-91,140` | LOW | Three `.item()` calls per timestep force device syncs in the training loop; evaluation runs without `torch.no_grad()`. |
| `analysis/stdp_and_edp_benchmark.py:12-22` | LOW | Import-time side effects: mutates global matplotlib rcParams and creates `results/plots/` on import. |
| Training scripts (see Fixed table) | MODERATE | The sparsity/hardware auxiliary losses in the non-v5 scripts are inert (detached floats). The v5 governor path in `benchmark_shd_v5.py` is differentiable and unaffected. A proper fix requires models to return differentiable spike-rate tensors. |

## Observations from the live v5 run (not code bugs)

- The event-rate governor saturates (final, full-run view): seed 42 pinned
  `activity_lambda` at 1.0 with the event rate staying ~0.143-0.149 against an
  annealed budget down to 0.03; seed 123 started near 0.0, ramped to 1.0 by
  epoch 60, and stayed there with rate ~0.144-0.157. The aux loss magnitude is
  small relative to cross-entropy, so the governor cannot move the rate
  meaningfully. Candidate fixes for a v6 recipe: larger aux weight, lower eta,
  log-domain lambda, or rate measured only over hidden layer 1. Final ledgered
  pair: 80.80% mean accuracy at 0.168 mean event rate.
