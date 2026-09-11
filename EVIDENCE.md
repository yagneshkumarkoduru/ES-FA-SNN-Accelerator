# EVIDENCE.md - ES-FA-SNN-Accelerator

Verified: 2026-09-10. Every headline number below maps to an exact source file.
Classes: MEASURED = board/hardware run. MODEL = C engine / estimator output.
SIMULATION = Python training or replay simulation.

## Verified claims (safe to use)

| Claim | Value | Class | Source |
|---|---|---|---|
| Validation accuracy (exp1 best) | 0.957 (95.7%) | SIMULATION | `results/analysis_summary.json` best.best_metrics.accuracy |
| Spike sparsity (exp1 best) | 0.5648 (56.48%) | SIMULATION | `results/analysis_summary.json` best.best_metrics.spike_sparsity |
| Energy proxy exp1 vs baseline | 4,592,863 vs 22,604,295 = 79.68% reduction | MODEL (estimator, not board power) | `results/analysis_summary.json` ranking[0] vs ranking[1] energy_proxy |
| Memory accesses exp1 vs baseline | 17,604,665 vs 112,062,994 = 84.29% reduction | MODEL (estimator) | `results/analysis_summary.json` memory_accesses |
| C-engine event mode energy saving | 20.81%, EDP 1.25x | MODEL (C99 cycle model) | `c_engine/c_benchmark_results.json` es_fa_event |
| C-engine config | 4 cores, 128 neurons/core, 10000 timesteps, 85% sparsity, 250 MHz | CONFIG | `c_engine/c_benchmark_results.json` header |

## Additional characteristics (use only with stated scope)

| Claim | Value | Class | Source |
|---|---|---|---|
| 6.3x EDP reduction vs synchronous systolic arrays | 6.3x | MODEL (`stdp_and_edp_benchmark.py` analysis output) | `analysis/stdp_and_edp_benchmark.py` print; `docs/paper/RESEARCH_PAPER.md` |
| 576-cycle active-window latency | 576 cycles | RTL SIMULATION (identical across compared runs; characteristic, not a win) | `output/key_findings.md` cycle_count lines |
| Throughput figures | paper says 128.0 GSOP/s (withdrawn); C99 engine measures 59.9 GSOP/s (event mode, cycle model); 4.43 pJ/SOP event-mode | C99 engine values now consistent with `c_benchmark_results.json`; paper figure still withdrawn | `c_engine/c_benchmark_results.json` vs `docs/paper/RESEARCH_PAPER.md` |
| C# driver figures | RESOLVED by a verified driver run: 4.81 M packets/s, 208.0 ns mean inter-packet gap, 81.9 ns mean dispatch latency, model-labeled EDP 3.18e-19 J*s (host-dependent, re-measure on target hardware) | MEASURED (desktop host) | `implementations/v3_csharp_net9_hal_sd_flashattention/Program.cs` dotnet run output |

## Known issues (do NOT claim otherwise)

1. exp5 adaptive mode is WORSE than exp1: energy proxy 45,016,894 (880% higher) and latency proxy 2x. Source: `results/analysis_summary.json` ranking[2]. Do not present exp5 as an improvement.
2. Baseline and exp1 share identical accuracy, sparsity, and latency proxy. The 79.68% comes from the estimator dataflow assumption, not from learned behavior. Always label it "estimated energy proxy reduction (model, not board power)."
3. Estimator vs measured hardware latency error is 33.33% MAPE. Source: `output/key_findings.md` estimator_vs_measured_hardware lines. Estimator is not validated for absolute claims.
4. Cycle counts and LUT usage identical across compared runs (576 vs 576 cycles, 22193 vs 22193 LUT). No measured latency or area win may be claimed from these.
5. Dataset name, seeds, and error bars are missing from `output/paper_draft.md`. Do not claim SOTA or peer-review readiness until added.

## Next measurements required

- [ ] Real KV260 board power + latency for baseline vs exp1 (see `hardware_validation/kv260/README.md`)
- [ ] Multi-seed training runs with error bars
- [ ] Named published SOTA baseline comparison
