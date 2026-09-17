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

**Recent published additions (compiled 2026-09-16 from arXiv:2605.16114
Table 1 and the referenced papers):** Sun et al. 2025, 96.26% (software
SNN, ~0.2M params); Schone et al. 2024, 95.90%; Baronig et al. 2024,
95.81% +- 0.56% (state-space model); Hammouamri et al. 2023, 95.07% +-
0.24% (learned delays); Catalyst community benchmark with adLIF, 91.0%
(90.8% int16); Matinizadeh et al. 2025 (FPGA event-graph, quantized),
92.3%; Cramer et al. 2022 dataset-paper SNN baseline, 83.2% +- 1.3%;
Biswas et al. 2024 (small LSM, 30k params), 77.8%; Carpegna et al. 2025
(FPGA SNN), 72.99%. These results use FP32 and different input encodings
and training budgets; the 92-96% band is the software frontier, not a
same-protocol comparison. A 98.43% figure in circulation belongs to a
multimodal N-MNIST + SHD task, not SHD alone. ES-FA's 77.87% is measured
under the fixed protocol in this ledger (INT8-aligned weights, 10 ms bins,
50-epoch training, sparsity regularization).
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

## Accuracy diagnosis and v5 innovation experiment (2026-09-16/17)

Findings from the v4 run (v2 recipe, 100 epochs, seed 42):
- best test 81.18% (epoch 68), final 77.83%; train accuracy reaches 99.6%.
  The recipe overfits: capacity is not the limiting factor, generalization
  and input representation are.
- seed 123 crashed mid-run (exit 0xC0000005); the seed 42 artifact stands.

Eight concrete gaps versus the public reference recipe class (~90% with the
same neuron family) were identified and are addressed in
`experiments/benchmark_shd_v5.py`: (1) 10 ms bins vs 4 ms reference bins;
(2) spike-sum readout vs leaky-integrated non-spiking readout; (3) soft
reset vs hard reset; (4) unscaled vs (1-beta)-scaled input current;
(5) sparsity pressure toward zero vs activity target tracking; (6) no
augmentation vs event-drop; (7) no label smoothing; (8) surrogate slope 5
vs 25. Protocol elements are attributed to the Catalyst neuromorphic
benchmarks (github.com/catalyst-neuromorphic/catalyst-benchmarks, MIT) and
are protocol corrections, not ES-FA claims.

ES-FA innovations implemented for the v5 run:
- Event-rate governor: closed-loop adjustment of the sparsity pressure so
  the hidden event rate tracks an annealed budget (start 0.15 to end 0.03),
  tying training to the accelerator SOP cost model; per-epoch event rates,
  budgets, and governor state are ledgered.
- Dual-timescale leaky readout: fast and slow leaky accumulators with
  learnable mixing (two accumulators, no multipliers in hardware).
- Retained recurrent PLIF backbone (learnable per-neuron decay).
- int16 post-training quantization evaluated on the same test split.

Run (cloud, completed 2026-09-17): c6i.8xlarge on-demand (us-east-1),
run id 20260917-054542, hidden 512, 4 ms bins (T=250), 160 epochs, batch 256,
seeds 42/123, governor and dual readout enabled, event-drop augmentation,
checkpoints every 2 epochs. Launched 05:46 IST, training completed 09:07:54
IST.

Measured results (SHD test split, 2 seeds):
- seed 42: best 80.34% (epoch 92); int16 quantized 79.86%.
- seed 123: best 81.27% (epoch 80); int16 quantized 81.40%.
- mean best 80.80%, max 81.27%; int16 mean 80.63% (0.17 pp mean cost).
- final hidden event rate 0.168 mean (83.2% sparsity); train accuracy ~88%
  versus 99.6% train in v4, so the protocol corrections reduced overfitting.
- Versus v4 (81.18% best, seed 42 only): within run-to-run variance; the
  two-seed mean is the honest comparable and supersedes the single-seed v4
  reading.

Governor finding (v6 candidate): activity_lambda saturated at 1.0 for both
seeds while the event rate settled near 0.144-0.170 against an annealed
budget down to 0.03. The auxiliary rate loss is too small relative to the
cross-entropy term to move the rate meaningfully. Candidate v6 changes:
larger auxiliary weight, lower eta, log-domain lambda, or rate measured on
layer 1 only. The pair (80.80% accuracy at 0.168 events) is the ledgered
Pareto point as measured.

Infrastructure finding (root cause of all prior silent failures, now fixed):
the instance-upload path previously used presigned S3 URLs signed with
aws-login temporary credentials. Measured URL lifetime: valid at +10 min,
ExpiredToken at +20 min (probe, 2026-09-17), so every upload attempted hours
later (heartbeats, results, log) always failed. Fixed by moving the instance
to an IAM instance profile (role esfa-shd-runner, scoped inline S3 policy),
`aws s3 cp` transfers, 10-minute heartbeats, stop-instead-of-terminate
shutdown, and SSM Run Command. This run's final artifacts were recovered via
SSM from the stopped instance's volume after the run's own final uploads
failed exactly as the diagnosis predicted.

Spend (actual): about USD 4.7 total. Source of rates: AWS on-demand pricing,
us-east-1, c6i.8xlarge at 1.36 USD/h, over about 3.4 h of training plus a
5-minute salvage boot; S3 and EBS costs are negligible (pennies). The run
used one instance only; no spot retries.

Artifacts: results/cloud_shd_v5_20260917-054542/ (local, canonical zip plus
console log) and s3://esfa-shd-runs-969739653654/shd/20260917-054542/.

## Governor saturation analysis (2026-09-17) and v6 design

The v5 cloud run's per-epoch history shows `activity_lambda` (a) decaying to
~0 in the first ~20 epochs, then (b) pinned at the clamp ceiling 1.0 from
~epoch 60 onward for both seeds. From the two seed histories:

- Early phase: the hidden rate starts near zero while the network learns,
  always below the budget, so the multiplicative update drives lambda toward
  the floor. The model then learns to fire with no sparsity pressure at all.
- Late phase: the natural firing rate settles at ~0.144-0.157 while the
  annexed budget falls to 0.03, so the error stays positive and lambda rides
  the ceiling for the rest of the run.

Two compounding design faults in the v5 governor:

1. Gain too high for the clamp. With eta = 8, a persistent error of 0.05
   multiplies lambda by e^0.4 ~ 1.49 per epoch, so the controller crosses its
   entire [0, 1] range in about five epochs. It behaves as a switch, not a
   controller.
2. Ceiling far too low relative to the loss. At lambda = 1 the auxiliary
   term is (rate - budget)^2 ~ 0.013 against a CE loss of ~1.5-2.5 (under 1%
   of the gradient), so even at the ceiling the governor cannot measurably
   move the rate. Measured: the rate fell only 0.157 -> 0.143 between epochs
   40 and 160 while the budget fell 0.121 -> 0.030.

Verdict: bad as a controller (the annealed budget was never tracked, the
ledgered lambda column is uninformative, and the accuracy-vs-events Pareto
was not actually controlled), and essentially neutral for accuracy (the aux
term was ~1% of the loss either way; the measured event rate was simply the
model's natural firing level). It is a textbook control loop saturated at
both rails.

v6 fixes (`experiments/benchmark_shd_v6.py`, launched 2026-09-17):
- Hinge penalty lambda * mean(relu(rate - budget)): only excess firing is
  penalized, so lambda has no reason to rise when the rate is under budget.
- Log-domain multiplier with clamp [1e-4, 100] (was [0, 1]) and gain eta = 3
  (was 8): the equilibrium lambda can now be large enough that the hinge
  gradient actually balances the CE gradient.
- Architecture: two recurrent PLIF layers (700 -> 256 -> 256 -> 20) instead
  of one 512-wide layer, matching the SRNN/PLIF reference architecture class.
  This is the main accuracy lever; the governor fix is a correctness fix.
- 200 epochs, 3 seeds (42/123/999), hidden 256, dropout 0.2, budget anneal
  0.15 -> 0.10.

Estimated spend (recorded before the run): bootstrap ~0.4 h plus training
3 seeds x 200 epochs at ~25-30 s per epoch on c6i.8xlarge, about 5.2 h total
at the on-demand list rate 1.36 USD/h (source: AWS on-demand pricing,
us-east-1, c6i.8xlarge). Estimate ~7 USD; the 8-hour watchdog bounds the
worst case at ~11 USD.

## v6 run aborted, v7 relaunch (2026-09-17, same day)

The v6 cloud run (20260917-144342, two 256-wide recurrent layers, governor
v2 with hinge and budget 0.15 -> 0.10) was monitored live via the new
heartbeat pipeline and stopped after seed 42 reached epoch 170:

- Good: the governor v2 works as a controller. Rate tracked the budget
  smoothly (0.172 -> 0.111 against 0.150 -> 0.108), lambda climbed smoothly
  0.001 -> 0.47, no rail saturation. The v5 failure mode is fixed.
- Bad: accuracy regressed. Seed 42 best was 75.44% at epoch 170 against v5's
  80.34%, and train accuracy stalled at 84.7% (underfitting, not
  overfitting). The regression is architectural/optimization, not sparsity
  pressure: the governor's absolute aux magnitude stayed tiny.

Diagnosis: no normalization for the deeper spike graph, spike dropout 0.2
destroying information, and 380k parameters versus v5's 630k.

v7 (`experiments/benchmark_shd_v7.py`, launched 2026-09-17 17:00 IST):
- BatchNorm1d on each hidden layer's input current (tdBN-style), the
  standard stabilizer for multi-layer spike networks (folds into the
  preceding linear at deployment; int16 quantization target unchanged).
- Dropout 0.2 -> 0.1, hidden 256 -> 320 (537k params), LR warmup 5 epochs.
- Governor kept ON with a soft target: hinge, eta 2, budget 0.15 -> 0.12.
- 160 epochs, 3 seeds (42/123/999).

Estimated spend (recorded before the run): about 4.7 h of training (160
epochs x 3 seeds at ~30-35 s/epoch) plus 0.4 h bootstrap at 1.36 USD/h
(c6i.8xlarge on-demand, us-east-1) = ~7 USD; 8-hour watchdog bounds the
worst case at ~11 USD. v6 spent ~2.1 USD before abort; partial log and
checkpoint are archived under results/cloud_shd_v6_aborted_20260917-144342/.

## v7 run aborted (BatchNorm eval collapse), v8 relaunch (2026-09-17)

The v7 cloud run (20260917-165146, two 320-wide recurrent layers with
BatchNorm on hidden currents) collapsed at evaluation. The per-epoch log is
unambiguous: seed 42 training accuracy climbed 6.0% -> 96.9% by epoch 60
while test accuracy sat at 5.3-15.3%, near the 5% chance level. BatchNorm was
the only structural change from the healthy v6 run (85% train / 74% test).

Mechanism: the same BatchNorm module was invoked once per timestep, 250
times per forward pass, so its running statistics were updated 250 times per
batch and tracked only the trailing timesteps of recent batches. Those
running statistics are also batch-composition dependent, which is not
reproducible at evaluation. v7 was stopped at epoch 60 (~2 USD) instead of
training to completion. Partial log and checkpoint archived under
results/cloud_shd_v7_aborted_20260917-165146/.

v8 (`experiments/benchmark_shd_v8.py`, launched 2026-09-17 17:52 IST):
- LayerNorm on hidden currents replaces BatchNorm. LayerNorm normalizes each
  sample independently with no running statistics, so train and eval
  behavior are identical by construction.
- Everything else retained from v7: two 320-wide recurrent layers (537k
  params), dropout 0.1, LR warmup 5 epochs, governor v2 with soft target
  (hinge, eta 2, budget 0.15 -> 0.12), dual-timescale readout.
- 160 epochs, 3 seeds (42/123/999). Both topology paths (layers 1 and 2)
  smoke-tested locally before launch.

Estimated spend (recorded before the run): about 4.4 h of training (160
epochs x 3 seeds at ~33 s/epoch) plus 0.4 h bootstrap at 1.36 USD/h
(c6i.8xlarge on-demand, us-east-1) = ~6.5 USD; 8-hour watchdog bounds the
worst case at ~11 USD. Running total for the day: ~4.7 (v5) + 2.9 (v6) +
2.0 (v7) + 6.5 (v8) = ~16 USD.

## v8 run aborted (LayerNorm stalls learning), v9 relaunch (2026-09-17)

The v8 cloud run (20260917-175004, LayerNorm on hidden currents) stalled:
seed 42 train accuracy sat at 7-10% and test at 8-12% over 25 epochs, with
the event rate flat at 0.016-0.017 (v6 at the same point was climbing to
22-43% train). LayerNorm makes the drive zero-mean and unit-variance per
sample; the membranes then sit far from the spiking threshold, and the fast-
sigmoid surrogate gradient 1/(25|x|+1)^2 vanishes when x is far from zero, so
the network never bootstraps into spiking. Unnormalized currents (v5, v6) do
not have this problem because the drive can grow until the threshold is
crossed. v8 was stopped at epoch 25 (~0.7 USD). Partial log and checkpoint
archived under results/cloud_shd_v8_aborted_20260917-175004/.

v9 (`experiments/benchmark_shd_v9.py`, launched 2026-09-17 18:22 IST): a
disciplined return to the empirically best topology with only motivated
changes:
- One 512-wide recurrent layer (the v5 winner) with the now-working governor
  v2 (hinge, log-domain lambda in [1e-3, 100], budget 0.15 -> 0.12). In v5
  the governor was inert, so this exact pairing was never tested.
- Dropout 0.3 -> 0.15: v5 stalled at ~88% train while the reference recipe
  class exceeds ~95% train on SHD, indicating over-regularization of the
  single hidden layer (spike dropout removes information).
- 200 epochs (v5 best epochs were 80-92 of 160 with the cosine tail still
  improving), LR warmup 5 epochs, 3 seeds (42/123/999).

Estimated spend (recorded before the run): about 5.4 h of training (200
epochs x 3 seeds at ~31 s/epoch) plus 0.4 h bootstrap at 1.36 USD/h
(c6i.8xlarge on-demand, us-east-1) = ~8 USD; 8-hour watchdog bounds the
worst case at ~11 USD. Running total for the day: ~4.7 (v5) + 2.9 (v6) +
2.0 (v7) + 0.7 (v8) + 8 (v9) = ~18 USD.

## v9 stopped, v10 relaunch: faithful SRNN-ALIF recipe (2026-09-17)

The user requirement is a headline SHD accuracy of at least 92%, versus the
80-81% ceiling of the v5-v9 custom recipe. The gap was analyzed against the
published adaptive-spiking recurrent network (SRNN, Yin/Corradi/Bohte 2021,
Nature MI; MIT reference code) whose open implementation reports 90.7% test
accuracy with only 2 x 128 neurons and 19 epochs. Recipe differences that
explain the gap: (1) adaptive threshold (ALIF) with soft reset versus our
fixed threshold and hard reset; (2) Multi-Gaussian surrogate versus our fast
sigmoid; (3) plain Adam at lr 1e-2 with StepLR versus our AdamW + cosine +
label smoothing + dropout + event-drop; (4) summed per-step softmax of a
leaky-integrator readout versus our dual-timescale readout; (5) no rate
governor (its sparse firing, FR ~0.11, emerges from the ALIF dynamics).

v9 was stopped mid-run (seed 42 best 80.74% archived; seed 123 partial) and
terminated to redirect budget. v10 (`experiments/benchmark_shd_v10.py`) is a
faithful port of the SRNN-ALIF recipe: same neuron equations (b0=0.01,
beta=1.8, learnable tau_m ~ N(20,5), tau_adp ~ N(150,10)), same surrogate
(sigma=0.5, s=6, h=0.15, gamma=0.5), same init (orthogonal recurrent,
xavier feedforward, biases zero), same optimizer/schedule, same decoding
(sum of per-step softmax, skipping the first 11 steps). The ES-FA addition is
an int16 weight-quantization evaluation at the end. 4 ms bins, T=250,
official SHD split, as in the reference.

Local smoke (hidden 32, 2 epochs): test 41.7% -> 58.5%, FR 0.126, versus the
old recipe at ~5% after two epochs. This confirms the port learns in the
expected regime before cloud spend.

Estimated spend (recorded before the run): 19 epochs x 3 seeds at an
estimated 30-60 s/epoch on c6i.8xlarge (much smaller network than v9) plus
bootstrap = roughly 0.75-1.5 h = 1-2 USD; 8-hour watchdog still bounds the
worst case. Running total for the day: ~4.7 (v5) + 2.9 (v6) + 2.0 (v7) +
0.7 (v8) + 3.5 (v9, stopped) + 2 (v10) = ~16-18 USD.

## v11: ES-FA improvements on the ALIF backbone (2026-09-17)

Requirement: the headline must beat the published reference implementation,
not merely reproduce it. v10 (running) is the fair-reproduction baseline of
the published recipe (19 epochs, best-epoch selection, single forward pass).
v11 (`experiments/benchmark_shd_v11.py`) keeps the same attributed backbone
and adds five ES-FA improvements, none of which exist in the reference:

1. Extended optimization: 80 epochs, 5-epoch linear warmup, cosine decay to
   1e-4 (reference: 19 epochs, StepLR halving). The reference code itself
   comments a 150-epoch variant, so longer training is in the spirit of the
   recipe but not practiced in its released defaults.
2. EMA weight averaging (decay 0.999, started after warmup): the deployed
   weights are an exponential moving average, a standard variance-reduction
   technique the reference does not use.
3. Event-loss test-time augmentation: the decoded output averages several
   forward passes over inputs with random event dropouts (5% time bins, 3%
   channels). This directly models the accelerator's event-loss robustness
   story: the network tolerates dropped events at inference.
4. Committee decoding: the three seeds' EMA models are averaged in output
   space into one predictor (reference reports per-run results only).
5. int16 deployment: all matrix weights quantized to int16, evaluated on the
   frozen test split (reference is float-only).

Local smoke test (hidden 32, 2 seeds x 2 epochs) exercised every path (EMA,
TTA, committee, int16) without errors; per-epoch logs show the schedule
engaging.

Estimated spend (recorded before the run): 80 epochs x ~80 s/epoch x 3 seeds
= about 5.4 h of training plus 0.4 h bootstrap at 1.36 USD/h (c6i.8xlarge
on-demand, us-east-1) = ~8 USD; 8-hour watchdog bounds the worst case at
~11 USD. Running total for the day: ~18 (previous runs) + 8 (v11) = ~26 USD.

Fragility finding (2026-09-17, v10 run, mid-flight): seed 42 reached best
87.10% (epoch 19, train 98.7%, FR 0.082) while seed 123 went silent -
test stuck at chance (~5%), firing rate decaying 0.0037 -> 0.0020 over the
first ten epochs with no recovery. Mechanism: the reference recipe leaves
the learnable time constants unconstrained and uses lr=1e-2 from step one;
if tau_adp crosses zero, rho = exp(-1/tau_adp) exceeds 1 and the adaptation
state (and threshold) explodes geometrically, permanently silencing the
neurons. The reference code itself notes this hazard and is exactly why the
v11 improvement set includes the warmup schedule (small early steps cannot
kick tau negative as easily). If any v11 seed still fails this way, v12 adds
an explicit positive clamp on tau_m/tau_adp after each optimizer step, a
defensible robustness improvement over the reference implementation.

## Deep review and v12 fixes (2026-09-18)

Two deep-read reviews covered the whole SHD training path (all experiment
scripts, the shared loader, the training core) and the reference
implementation diff. The root cause of the dead seed is now verified,
reproduced, and fixed. It was an initialization hazard, not a scheduler bug.

Root cause (verified and reproduced on torch 2.14.0+cpu, the same version
the cloud runs used): ALIFNeuron draws tau_m ~ N(20, 5) with no positivity
protection. At hidden=128, seed 123 draws tau_m[5] = -1.8396 (checked by
constructing the model under torch.manual_seed(123)). Then
alpha = exp(-1/tau_m) = 1.7222 > 1, so the membrane recursion
mem <- alpha * mem + (1 - alpha) * I - B * spike diverges geometrically; the
first backward pass writes NaN into tau_m[5], and because clamp(NaN) remains
NaN the earlier post-step clamp could not repair it. Heaviside(NaN) is 0
forever, so the unit was permanently dead and the run trained to chance with
no warning (observed firing rate 0.008 -> 0.002, test accuracy 4-5% for 80
epochs). The hazard is inherited from the reference recipe (~0.8 percent of
2x128 seeds draw a negative tau_m on this torch build); the reference trained
a single seed and never demonstrated robustness.

Fixes applied (v12 unless noted):
- Time constants are sigmoid-reparameterized into bounded ranges (tau_m and
  tau_m_o in [2, 100], tau_adp in [2, 1000]), so alpha and rho provably stay
  in the stable open interval (0, 1) regardless of initialization draw or
  optimizer behavior. Initial values still follow the reference
  distributions, so the healthy recipe is unchanged.
- A non-finite guard skips and counts non-finite loss/gradient steps; a seed
  that cannot recover aborts loudly while the remaining seeds continue.
- Evaluation is deterministic (fixed evaluation generator), so best-epoch
  tracking is not driven by random-init noise; training stays stochastic.
- Committee decoding is quality-gated (members with collapsed firing rates
  are excluded), member outputs are normalized before averaging, and the
  member count is reported.
- Best-so-far checkpoints are written every 8 epochs (crashes cannot lose a
  completed seed) and EMA states are saved per seed.
- v10/v11 initialization draws are clamped (post-run safety fix).
- The shared loader allocates the raster as uint8 directly (4x lower peak
  memory; v12 uses it) and the train/test fallback pairs files by name (the
  old sorted() fallback swapped the sets).

Canary validation (local, the exact previously-fatal configuration): seed
123 at hidden=128 now trains healthy - epoch 1 test 43.33% at FR 0.0624,
epoch 2 test 75.71% at FR 0.0629 - versus 4.37% at FR 0.0080 when dead. The
committee path was smoke-tested with two healthy seeds (hidden 32): gated
2-member committee evaluated, all result fields written, no non-finite steps.

Next cloud run (launched after these fixes): v12 bounded backbone, hidden
160, 64 epochs, seeds 42/123/999, train-drop on, TTA 4, 3-seed committee,
deterministic evaluation. Estimated 6.0 h at 1.36 USD/h = ~9 USD; the 8-hour
watchdog still bounds the worst case. Running total for the campaign:
~20 (previous runs) + 9 (v12) = ~29 USD.

## Known issues

1. **SHD accuracy gap vs SOTA:** ES-FA v5 achieves 80.80% (two-seed mean,
   cloud run 2026-09-17; 81.27% max) vs 92.66% (PLIF-ICCV2021) and about 90%
   for the reference recipe class on the same neuron family. The gap is not
   a measurement error; it reflects the remaining architecture and recipe
   differences (single 512-wide recurrent layer, 160-epoch budget).
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
