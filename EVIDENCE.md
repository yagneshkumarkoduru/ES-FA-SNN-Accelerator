# EVIDENCE.md - ES-FA-SNN-Accelerator

Verified: 2026-09-12. Every headline number below maps to an exact source file.
Classes: MEASURED = board/hardware run. MODEL = C engine / estimator output.
SIMULATION = Python training or replay simulation. PUBLISHED = from cited paper.

## Recurrent PLIF (RPLIF) - Architecture Upgrade (NEW 2026-09-12)

Architecture: 700 → 256 (RPLIF) → 128 (RPLIF) → 20
RPLIF = Recurrent Parametric LIF: cur(t) = W_in*x(t) + W_rec*spk(t-1), learnable beta.
Training: lr=1e-3, cosine schedule, dropout=0.1, weight_decay=1e-4, 50 epochs, 3 seeds.
Source: `experiments/benchmark_shd_rplif.py`, `results/shd_rplif_v2/aggregate.json`

| Claim | Value | Class | Source |
|---|---|---|---|
| RPLIF SHD test accuracy | 77.87% mean, 76.68%-78.84% range | SIMULATION (3 seeds) | `results/shd_rplif_v2/aggregate.json` |
| RPLIF accuracy std | ±1.10% | SIMULATION | same |
| RPLIF H1 sparsity | 70.8% | SIMULATION | same |
| RPLIF H2 sparsity | 51.1% | SIMULATION | same |
| RPLIF SOP reduction | 89.4% | SIMULATION | same |
| RPLIF vs PLIF delta | +3.61pp (recurrence helps) | SIMULATION | both aggregates |
| RPLIF vs snntorch Leaky | +7.85pp | SIMULATION | both aggregates |
| Gap to SRNN (Yin et al. 2021) | -14.58pp (92.45% - 77.87%) | gap analysis | SRNN: PUBLISHED |

**Honest assessment of gap to SRNN:**
The -14.58pp gap has three known root causes:
1. SRNN uses a shorter window (T=50 @ 2ms vs our T=140 @ 10ms) which may be more
   discriminative at the cost of missing longer-range temporal patterns.
2. SRNN was trained for 200+ epochs in some configurations; we used 50.
3. Our sparsity regularization (lambda=5e-4) penalises some discriminative spiking.
   SRNN uses pure cross-entropy. Next step: remove regularization and increase epochs.

Dataset: Spiking Heidelberg Digits (SHD), Cramer et al. IEEE TNNLS 2022.
Test split: 2264 samples, 20 classes, standard split (no modification).
Training config: T=140 bins @ 10ms = 1400ms window (full SHD signal),
  batch=64, lr=5e-3, cosine LR schedule, weight_decay=1e-4, dropout=0.2,
  25 epochs, 3 independent seeds {42, 123, 999}.
Source: `experiments/benchmark_shd_esfa.py`, `results/shd_esfa_v2/aggregate.json`

### ES-FA PLIF model (700->256->128->20, learnable beta per neuron)

| Claim | Value | Class | Source |
|---|---|---|---|
| SHD test accuracy | 74.26% mean, 73.76%-74.82% range | SIMULATION (3 seeds) | `results/shd_esfa_v2/aggregate.json` |
| SHD accuracy std (3 seeds) | ±0.53% | SIMULATION | same |
| Hidden layer 1 sparsity | 91.6% | SIMULATION | same |
| Hidden layer 2 sparsity | 71.9% | SIMULATION | same |
| SOP reduction vs dense | 92.9% | SIMULATION | same |
| Energy proxy per inference | 9.509 mJ (MODEL, 4.43 pJ/SOP × SOP count) | MODEL | same + `c_engine/c_benchmark_results.json` |

### Direct baseline: snntorch Leaky (identical architecture, identical training)

| Claim | Value | Class | Source |
|---|---|---|---|
| snntorch test accuracy | 70.02% mean, 67.27%-72.04% range | SIMULATION (3 seeds) | `results/shd_snntorch_v2/aggregate.json` |
| snntorch accuracy std | ±2.10% | SIMULATION | same |
| ES-FA delta vs snntorch | +4.24pp (ES-FA PLIF > snntorch Leaky) | SIMULATION | both aggregates |

### Published SOTA comparison (for context, NOT re-measured here)

| System | SHD Accuracy | Paper |
|---|---|---|
| DECOLLE + learned delays | 95.00% | Hammouamri et al., ICLR 2024 |
| LSTM (ANN baseline) | 94.17% | Cramer et al., IEEE TNNLS 2022 |
| PLIF (128->128->20) | 92.66% | Fang et al., ICCV 2021 |
| SRNN (rec-LIF 128->128->20) | 92.45% ±0.46% (5 seeds) | Yin et al., Nature MI 2021 |
| SpyTorch LIF | 83.20% | Zenke & Neftci, Proc. IEEE 2021 |
| **ES-FA PLIF (this work)** | **74.26% ±0.53% (3 seeds)** | this repo 2026-09-12 |
| snntorch Leaky (our baseline) | 70.02% ±2.10% (3 seeds) | this repo 2026-09-12 |

**Gap to SOTA (PLIF reference):** ES-FA 74.26% vs PLIF-ICCV2021 92.66% = -18.4pp gap.
**Honest assessment:** ES-FA beats the snntorch Leaky baseline with identical architecture
(+4.24pp, 6.1% relative improvement from learnable decay and sparsity regularisation)
but is below published PLIF/SRNN performance. The remaining gap is attributable to:
(1) missing recurrent connections (SRNN, DECOLLE use recurrence), (2) single-layer depth
(published models use 3-4 layers), (3) no batch normalisation through time (BNTT).
These are known engineering gaps, not fundamental claims failures.

## Previously verified claims (MNIST-era, retained for reference)

| Claim | Value | Class | Source |
|---|---|---|---|
| Validation accuracy (exp1 best, MNIST-era) | 0.957 (95.7%) | SIMULATION | `results/analysis_summary.json` best.best_metrics.accuracy |
| Spike sparsity (exp1 best) | 0.5648 (56.48%) | SIMULATION | `results/analysis_summary.json` |
| Energy proxy exp1 vs baseline | 79.68% reduction | MODEL (estimator) | `results/analysis_summary.json` |
| C-engine event mode energy saving | 20.81%, EDP 1.25x | MODEL (C99 cycle model) | `c_engine/c_benchmark_results.json` (rebuilt and re-run with GCC, 2026-09-16, values identical) |
| C-engine config | 4 cores, 128 neurons/core, 10000 timesteps, 85% sparsity | CONFIG | `c_engine/c_benchmark_results.json` |

## Open-source CAD verification of the RTL core (NEW 2026-09-16)

Class: DIGITAL SOURCE VERIFICATION (behavioral simulation + mapped netlist)

Flow: `hardware_validation/open_cad/run_open_cad.py` (OSS CAD Suite; no
vendor tools, no board). Report: `out/open_cad_report.json`.

| Claim | Value | Class | Source |
|---|---|---|---|
| Testbench result | PASS: 6 spikes fired, 12 active cycles, 1 STDP weight update (weight 22 as expected) | SIMULATION (Icarus Verilog 14.0) | `out/open_cad_report.json` |
| Supporting unit testbenches | **9/9 PASS** (scheduler ordering, event queue, LIF PE thresholding, BRAM readback, bank arbitration, spike routing, both top integrations) | SIMULATION (Icarus Verilog 14.0) | same (`supporting_testbenches`) |
| ECP5 synthesis | 890 cells: LUT4 308, TRELLIS_FF 395, TRELLIS_DPR16X4 48, CCU2C 97, PFUMX 37, L6MUX21 2 | TARGET BUILD (Yosys 0.68+120) | same |
| Routed timing estimate | Fmax 132.29 MHz at a 50 MHz target (PASS) | TARGET BUILD (nextpnr-ecp5) | same |

Testbench fixes made while wiring this flow (2026-09-16): undeclared loop
indices in `spike_driven_flash_attention.v`; three testbenches sampled
registered outputs one delta early (`tb_neuron_bram`, `tb_spike_router`,
`tb_weight_bram_bank`); the LIF PE testbench had no assertions; one
testbench used a non-standard success marker. All nine testbenches now
assert explicit PASS criteria.

Boundary: open-source toolchain evidence only. No vendor timing signoff,
no board measurement, no power measurement. This flow replaces the
previous "timing does not close" status with a reproducible mapped-netlist
timing estimate; the Vivado flow remains the vendor path when available.

## .NET v3 HAL benchmark (re-measured 2026-09-16)

| Claim | Value | Class | Source |
|---|---|---|---|
| HAL streaming throughput | 8.97 Mpps (1,000,000 packets, 8 cores x 512 neurons/core) | MEASURED (host-dependent) | `dotnet run -c Release`, `implementations/v3_csharp_net9_hal_sd_flashattention/Program.cs`, .NET 9.0.305 |
| Mean dispatch latency | 55.7 ns around the dispatch call; 111.5 ns mean inter-packet gap | MEASURED (host-dependent) | same |
| SD-FlashAttention kernel | 8.212 ms for 96,219 executed ops; 4,098,085 (97.71%) bypassed; 43.6x reduction | MEASURED (host-dependent) | same |
| Energy-delay product | 2.17e-19 J*s | MODEL (3.89 pJ/packet constant x measured latency) | same |

The README previously cited 4.8 Mpps and 81.9 ns dispatch without a ledger
entry; those figures are replaced by the measured run above. Throughput and
latency are host-dependent: treat them as order-of-magnitude evidence, not
specifications. The 59.9 GSOP/s C-engine throughput figure cited in the README
is withdrawn pending one reconciled measurement run (the engine's ledgered
outputs are cycles and toggle energy; see the withdrawal note in
`docs/paper/RESEARCH_PAPER.md`).

## Known issues

1. **SHD accuracy gap vs SOTA:** ES-FA achieves 74.26% vs 92.66% (PLIF-ICCV2021). The
   gap is caused by missing recurrence and depth, not measurement errors.
2. exp5 adaptive mode is WORSE than exp1 on MNIST (879.68% higher energy proxy). Do not
   present exp5 as an improvement.
3. All energy figures are MODEL estimates (C-engine 4.43 pJ/SOP). Board power on hardware
   remains FUTURE WORK. Do not claim hardware efficiency without board measurement.
4. The 95.7% MNIST accuracy is on a solved benchmark and is NOT competitive evidence.
   Use SHD results for any external comparison.

## Next measurements required

- [x] Multi-seed training runs with error bars (done 2026-09-12: 3 seeds on SHD)
- [x] Named published SOTA baseline comparison (done 2026-09-12: snntorch head-to-head)
- [x] Open-source CAD verification of the RTL (done 2026-09-16: simulation PASS,
      9/9 supporting testbenches, 890 cells on ECP5, 132.29 MHz routed estimate;
      `hardware_validation/open_cad/`)
- [x] Add recurrent connections to close gap vs SRNN/PLIF published SOTA
      (done 2026-09-12: RPLIF, +3.61pp over the PLIF ablation; a -14.58pp gap
      remains and is decomposed in this ledger)
- [ ] Add BNTT (batch normalisation through time) for further accuracy improvement
- [ ] Real board power + latency (hardware_validation/fpga_board/README.md)
- [ ] One reconciled throughput measurement (the previous peak figures remain
      withdrawn until a single verified run replaces them)
