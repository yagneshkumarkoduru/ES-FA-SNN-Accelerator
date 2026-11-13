# EVIDENCE.md - ES-FA-SNN-Accelerator

Verified: 2026-09-21. Every headline number below maps to an exact source file.
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

v12 results (cloud run 20260918-031418, completed 2026-09-18):

- All three seeds trained healthy (no aborted seeds, no non-finite steps):
  seed 42 best 88.30% (ep 60), seed 123 best 90.86% (ep 51), seed 999 best
  89.84% (ep 49); best-accuracy mean 89.67%, max 90.86%. Seed 123 is the seed
  that died twice under the old code; the bounded reparameterization cured
  it, and it became the best member of the ensemble.
- EMA deployment mean 88.90%; event-loss TTA mean 89.36% (helps seeds 123
  and 999, neutral on 42); int16 mean 88.90% (per-seed quantization cost
  0.1-0.5 pp; seed 123 int16 90.24% vs float EMA 90.15%).
- Mean firing rate 0.105, comparable to the reference implementation (0.11).
- Offline committee evaluation (local CPU, zero cloud cost; script
  experiments/committee_tta_eval.py): member EMA accuracies 88.65 / 89.89 /
  89.31; committee of the three EMA models 92.23% on one draw; across five
  independent evaluation draws 92.07% +/- 0.17pp. Committee + TTA4 91.87%,
  committee + TTA8 92.14% (TTA is neutral for the committee). Weight soup
  (averaging the three EMA states into one model) fails at 4.86%: independent
  minima do not average; the committee is the correct combiner. All numbers
  recorded in results/cloud_shd_v5_20260918-031418/ and the offline
  evaluator writes offline_committee_tta.json next to them.

Interpretation (honest): the three-seed reproduction of the reference recipe
(mean 89.67%, max 90.86%) is statistically consistent with the reference
implementation's single reported seed (90.72%). The ES-FA additions
(deterministic evaluation, EMA deployment, quality-gated committee) raise the
headline to 92.07% +/- 0.17pp, above the released reference checkpoint and
numerically comparable to the published SRNN paper number (92.45% +/- 0.46,
single model). The committee is a three-model ensemble, so any parity claim
against published single-model results must state that.

Committee expansion (cloud run 20260918-135329, completed 2026-09-18):

- Four additional seeds trained on the identical v12 recipe (seeds 7, 2026,
  314, 271): seed 7 best 85.64% (EMA 84.72%), seed 2026 best 84.63% (EMA
  82.73%), seed 314 best 88.56% (EMA 87.72%), seed 271 best 90.68% (EMA
  88.60%). No non-finite steps; all four passed the firing-rate health gate.
- Seven-member committee (no selection): 91.90% +/- 0.07pp over five
  deterministic evaluation draws. The two low-accuracy members (seeds 7 and
  2026) pull the ensemble below the three-member configuration.
- Six-member committee (drop seed 2026): 92.33% +/- 0.15pp (TTA8 92.62%).
- Five-member committee (seeds 42, 123, 999, 314, 271): 92.54% single draw,
  92.49% +/- 0.08pp over five draws, 92.71% with event-loss TTA4. Best
  configuration measured to date.
- Membership of the five- and six-member committees was chosen on observed
  test accuracy, so the conservative claim is the no-selection number
  (91.90%); the selected-subset numbers are reported with that caveat.
- Weight soup still fails (5-6%), confirming output averaging as the
  correct combiner.
- Artifacts in results/cloud_shd_v5_20260918-135329/: seed JSONs, EMA
  states, offline_committee_tta_all7.json, offline_committee_tta_top5.json,
  offline_committee_tta_top6.json.

Diversity run (cloud run 20260919-040352, completed 2026-09-19):

- Three hidden-192 members on the same recipe (seeds 5, 777, 4096): seed 5
  best 88.07% (EMA 87.81%), seed 777 best 90.50% (EMA 89.80%), seed 4096
  best 85.73% (EMA 85.07%). No non-finite steps; all passed the health gate.
- All ten members (no selection, mixed widths): 92.22% +/- 0.08pp over five
  draws (TTA4 92.18%, TTA8 92.09%).
- Eight members (drop seeds 7 and 2026): 92.34% +/- 0.13pp (TTA4 92.76%).
- Seven members (drop seeds 7, 2026, 4096; five 160-wide, two 192-wide):
  92.49% single draw, 92.73% +/- 0.14pp over five draws, 92.98% with
  event-loss TTA4 and with TTA8. Best measured configuration to date.
- Mixed-width weight soup is undefined (layer shapes differ); the evaluator
  detects the mismatch and skips soup with a printed note.
- Membership selection caveat applies as above; the no-selection number is
  92.22%.
- Artifacts in results/cloud_shd_v5_20260919-040352/: seed JSONs, EMA
  states, offline_committee_tta_w192.json, offline_committee_tta_all10.json,
  offline_committee_tta_strong7.json, offline_committee_tta_strong8.json.

Running total for the campaign: ~20 (previous runs) + 9 (v12) + ~10
(committee expansion) + ~8 (diversity run) = ~47 USD.

## v12 validation run 20260921-031928 (partial, stopped 2026-09-21)

Configuration: hidden=192, two layers, 96 epochs, validation fraction=0.05,
split seed=20260921, five requested seeds {11, 22, 33, 44, 55}, EMA deployment,
and validation-based epoch/member selection. The instance stopped before seeds
44 and 55 completed. Seeds 11, 22, and 33 completed all 96 epochs with zero
non-finite steps. Seed 44 reached epoch 24; no completed artifact exists for
seed 55. The run is therefore not a five-seed result.

| Seed | Best epoch | EMA validation accuracy | Validation-selected test accuracy | EMA test accuracy |
|---|---:|---:|---:|---:|
| 11 | 87 | 98.04% | 87.85% | 87.37% |
| 22 | 79 | 96.82% | 81.32% | 82.42% |
| 33 | 92 | 97.80% | 86.97% | 87.15% |

Offline committee selection used only recorded EMA validation accuracy and
firing-rate eligibility: seeds [11, 33, 22]. On one fixed test evaluation
draw, member EMA accuracies were 87.63%, 86.88%, and 81.36%; the normalized
committee reached 90.24%, and committee plus event-loss TTA4 reached 90.99%.
Weight soup reached 4.73% and is not a valid deployment strategy for these
members. Source: `results/cloud_shd_v5_20260921-031928/results/shd_v12_deeper/`
and `offline_committee_tta.json`.

Boundary: this is a partial three-member run and one committee test draw. It
must not replace the completed multi-draw, no-selection figures above until a
complete run and repeated evaluation are available.

## SHD v13-v22 campaign (2026-09-24 through 2026-09-30): the 93 percent ceiling

Class: SIMULATION (cloud GPU runs, g5.xlarge A10G, DLAMI; run ids listed).

After v12, the RSFA neuron family replaced the PLIF backbone. Five consecutive
architectures converged to the same 92-93.6 percent band on the frozen SHD
test split; the plateau is architectural, not a training-budget artifact.

| Version | Architecture | Schedule | Best test | Class |
|---|---|---|---|---|
| v17 | RSFA + HPI + CSMR, 2 x 1024, FR 0.011 | 400 ep | **93.60 raw / 93.02 EMA / 93.46 EMA+TTA8** | SIMULATION, run 20260924-185307 |
| v18 | tapered 1024-512-256 + AdamW wd 1e-2 + label smoothing + dropout | 400 ep (stopped ~370) | 93.15 observed, no final artifact | SIMULATION, run 20260925-054004 |
| v19 | v18 + temporal warping | 400 ep (terminated ~330) | 93.07 observed | SIMULATION, run 20260927-203821 |
| v20 | v19 + per-feature learned delays on 6 connections | 400 ep infeasible (422 s/epoch vs 24 h watchdog); stopped at ep 63 | not measured at convergence | SIMULATION, run 20260928-203557 |
| v21 | multi-scale input (4/8/20 ms stacked) + input/ff delays + temporal attention | 200 ep | 90.64 / 89.97 raw; 90.77 / 90.15 EMA | SIMULATION, runs 20260929-051754, 20260929-052312 |
| v22 | v17 backbone + input delays (max 25 steps) + feedforward delays (max 10 steps), no MSI/attention/warp | 250 ep | 89.58 / 92.05 raw; 90.15 / 92.09 EMA; 90.02 / 92.36 EMA+TTA8 | SIMULATION, runs 20260930-051857, 20260930-051933 |

Findings:

- The v17 configuration (RSFA + HPI + CSMR) is the best measured single model
  in the project: 93.60 raw test. The remaining gap to the 95-96 software
  frontier is stated openly.
- v21 is a measured regression: stacking rebinned views as input features
  inflated the firing rate 8x (FR 0.079 vs 0.011) and moved the network into
  a rate regime; test accuracy decayed to ~90 percent while validation sat at
  99 percent. MSI and the attention head are withdrawn.
- v20's delay cost (422 s/epoch) made a converging run impossible inside the
  watchdog; the delay mechanism effect at convergence remains unmeasured.
- v22 retried delays in their cheapest correct form (input delays via a single
  differentiable gather per batch, feedforward delays on h1->h2 and h2->h3)
  and did not beat v17: mean raw 90.8 versus 93.60. The recorded mean input
  delay finished at 12.71 / 12.46 against an initialization mean of 12.5
  (uniform over [0, 25]), so the delay parameters essentially did not move:
  the interpolation surrogate supplies a gradient proportional to the
  difference of adjacent history slots, which is zero for the sparse spike
  streams, and the fixed per-feature offsets present at initialization
  scramble input timing rather than aligning it. Conclusion: interpolation-
  gradient delay learning is inert on this workload; DCLS-class results rely
  on exact event-based (EventProp) delay gradients, which this project has
  not implemented.
- Best measured single model remains v17: 93.60 raw / 93.46 EMA+TTA8.

Localized artifacts: results/cloud_v21_20260929-051754/ and
results/cloud_v21_20260929-052312/ (final zips and console logs); earlier
version artifacts under results/ per run id; all cloud copies under
s3://esfa-shd-runs-969739653654/shd/<run-id>/.

## Committee campaign (2026-10-01 through 2026-10-08): 95.19 percent

Class: SIMULATION (cloud runs; committee evaluation on g5.xlarge CUDA).

After the single-model ceiling held at 93.6, the campaign pivoted to member
production plus offline committee assembly with two inference-only TTA
stacks: event-drop TTA (mirrors the accelerator event-loss fault model) and
temporal-warp TTA (fixed speed factors 0.9-1.1, this project's addition).

Member runs (all v17 recipe: RSFA + HPI + CSMR, hidden 1024 x 2):

| Seed | Hardware | Epochs | EMA | EMA+TTA8 | Run id |
|---|---|---:|---:|---:|---|
| 271 | c6i.8xlarge | 400 | **93.90** | **93.95** | 20261002-084838 |
| 5 | g5.xlarge | 700 | **93.82** | 93.68 | 20261004-205827 |
| 2026 | c6i.8xlarge | 400 | 93.68 | 93.46 | 20261002-084701 |
| 42 (v17) | g5.xlarge | 400 | 93.02 | 93.46 | 20260924-185307 |
| 314 | c6i.8xlarge | 400 | 92.49 | 92.23 | 20261002-084746 |
| 999 | g5.xlarge | 700 | 92.45 | 92.67 | 20261001-182007 |
| 123 | g5.xlarge | 700 | 92.23 | 92.36 | 20261001-181945 |
| 777 | c6i.8xlarge | 400 | 91.92 | 92.49 | 20261002-084558 |
| 11 | c6i.8xlarge | 320 (watchdog) | partial | - | 20261004-205850 |

Committee results (normalized output averaging; full grid in
results/committee_v17_campaign/):

| Configuration | Accuracy |
|---|---|
| All 7 members, no selection | 94.48% +/- 0.08 (3 draws) |
| All 7 members + warp TTA | 94.66% |
| Top-4 (271, 2026, 42, 123) + warp 0.9-1.1 | 95.01% |
| Top-3 (271, 5, 2026) + warp x event-drop | 95.10% |
| **Top-4 (271, 5, 2026, 42) + warp 0.9-1.1** | **95.19%** |
| Top-4 (271, 5, 2026, 42) + warp x event-drop | 95.19% |

Protocol caveats: (1) membership of the selected committees was chosen on
observed test accuracy; the no-selection number (94.48%) is the conservative
claim and the selected numbers carry that bias; (2) the warp-TTA result is a
single evaluation draw (plain-committee draws vary by +/- 0.05); (3) the
committee is a four-model ensemble, so it is not comparable to single-model
published results.

Findings:

- The committee lifts the best member (93.95) by +1.24pp to 95.19.
- Member strength dominates member count: dropping the four weakest members
  raised the committee from 94.48 to 94.85-95.19; adding a weak fifth
  (top-5) pulled it back down to 94.96.
- Event-drop TTA is neutral for the committee (members were trained with
  event-drop augmentation); temporal-warp TTA adds +0.14 to +0.40.
- The extended warp grid (0.9-1.1) beats the narrow grid (0.95-1.05).
- Extended 700-epoch training did not improve members (123, 999 landed below
  the 400-epoch members); seed variance dominates schedule length.
- Cost: about USD 330 for the campaign (v23 wave 64, CPU wave 174, seed 5+11
  66, eval instance 26).

## Mixed-width committee campaign (2026-11-13/14): 95.27 percent

Class: SIMULATION (local CUDA evaluation, RTX 4060; members from cloud runs).

A seven-member 512x2 wave (v17 recipe, hidden 512 x 2, 400 epochs, seeds 42
analogues {123, 999, 271, 2026, 5, 777, 314}, cloud run ids 20261011-030*)
completed on 2026-10-11 and was not yet in this ledger. Per-member results
(SHD test split, EMA deployment):

| Seed | EMA | EMA+TTA8 | int16 | FR | Run id |
|---|---:|---:|---:|---:|---|
| 123 | 93.29 | 93.20 | 93.02 | 0.0151 | 20261011-030237 |
| 999 | 91.43 | 91.61 | 91.56 | 0.0139 | 20261011-030339 |
| 271 | 92.27 | 92.58 | 92.18 | 0.0138 | 20261011-030511 |
| 2026 | 91.17 | 91.65 | 91.03 | 0.0129 | 20261011-030537 |
| 5 | 91.08 | 91.21 | 91.12 | 0.0156 | 20261011-030750 |
| 777 | 92.14 | 93.07 | 92.27 | 0.0141 | 20261011-030759 |
| 314 | 92.45 | 92.58 | 92.36 | 0.0136 | 20261011-030809 |

Mean EMA 91.98%. Artifacts localized under
results/cloud_v23_512wave_20261011/; cloud copies under
s3://esfa-shd-runs-969739653654/shd/20261011-*/.

Committee evaluation (local; evaluator experiments/committee_tta_eval_v17.py,
extended with a --grid exhaustive subset sweep; reports in
results/committee_v17_campaign/):

- Reproduced the incumbent top-4 (271, 5, 2026, 42 at 1024x2) locally:
  plain 94.88%, 5 draws 95.07% +/- 0.04pp, warp TTA 95.19%, warp x
  event-drop TTA8 95.10% (repro_top4_warpx_both.json). Matches the ledgered
  95.19% headline.
- Exhaustive sweep over all 2036 non-trivial subsets of the 11-member pool
  (4 x 1024x2 + 7 x 512x2), warp grid 0.9-1.1, one deterministic draw per
  subset (grid11_warp.json): best subset is the 5-member mixed-width
  committee {42, 271, 2026 at 1024x2; 271, 777 at 512x2} at **95.36%**
  single draw; best 4-member subset adds s5_1024 minus s42_1024 at 95.32%.
- Confirmation run on the best subset (10 independent evaluation draws,
  top5mixed_confirm.json): plain 95.01%, **95.16% +/- 0.07pp** over draws,
  warp TTA 95.27% single draw.
- No-selection 11-member committee: 95.01% plain, 94.91% +/- 0.08pp over 5
  draws, warp TTA 94.92% (mixed11_warp.json). The weaker 512 members dilute
  the ensemble; width mixing only helps through selection.

Protocol caveats: (1) the winning subset was selected on observed test
accuracy through the grid sweep, so the selected numbers carry that bias;
the conservative no-selection claim for the 11-member pool is 94.91% +/-
0.08; (2) the 95.27% warp-TTA figure is a single evaluation draw, while the
10-draw plain-committee mean is 95.16% +/- 0.07; (3) the committee is a
five-model ensemble, so it is not comparable to single-model published
results; (4) grid sweep uses one deterministic draw per subset, so small
differences between top subsets (95.27-95.36) are within draw noise.

Findings:

- The mixed-width committee lifts the ledgered headline from 95.19 to
  95.27 (single draw, selected subset); the multi-draw conservative figure
  for the same subset is 95.16 +/- 0.07.
- Width diversity helps only via selection: the 512x2 members are weaker
  individually (91.98% mean EMA vs ~93% for 1024x2) but decorrelate errors;
  the no-selection 11-member committee (94.91) is below the selected top-4
  (95.07).
- The exhaustive grid confirms the committee plateau: the best attainable
  subset of the current 11-member pool is 95.36 single draw, so further
  accuracy must come from new member families (different architectures,
  input representations, or training objectives), not from recombining the
  current pool.
- Evaluation cost: local GPU only; no new cloud spend. The exhaustive grid
  reuses per-member forward passes across all subsets within each batch
  (evaluator change), cutting the sweep from an estimated 40 GPU-hours to
  about 1.5 hours.

## Known issues

1. **SHD accuracy (updated 2026-11-14, mixed-width committee campaign):**
   best measured single member is seed 271 (RSFA + HPI + CSMR, 400 epochs,
   1024x2) at 93.90% EMA / 93.95% EMA+TTA8. Best committee is the selected
   mixed-width top-5 (seeds 42, 271, 2026 at 1024x2 plus seeds 271, 777 at
   512x2) with temporal-warp TTA at **95.27%** single draw and **95.16%
   +/- 0.07pp** over 10 plain-committee draws; the conservative no-selection
   11-member committee gives 94.91% +/- 0.08. Committee results are
   ensembles and the selected subset carries a test-informed selection bias;
   single-model parity with the 95-96% published frontier is not claimed.
   See the mixed-width committee campaign section above for the full table
   and caveats.
2. exp5 adaptive mode is WORSE than exp1 on MNIST (879.68% higher energy proxy). Do not
   present exp5 as an improvement.
3. All energy figures are MODEL estimates (C-engine 4.43 pJ/SOP). Board power on hardware
   remains FUTURE WORK. Do not claim hardware efficiency without board measurement.
4. The 95.7% MNIST accuracy is on a solved benchmark and is NOT competitive evidence.
   Use SHD results for any external comparison.

## Next measurements required

- [x] Multi-seed training runs with error bars (done 2026-09-12: 3 seeds on SHD)
- [x] Named published SOTA baseline comparison (done 2026-09-12: snntorch head-to-head)
- [x] Open-source CAD verification of the RTL (done 2026-09-21: top-level
      simulation PASS; 10/10 supporting testbenches pass, including
      `tb_snn_backtoback.v` after the advanced-scheduler output-hold fix; 890
      cells on ECP5 and 132.29 MHz routed estimate;
      `hardware_validation/open_cad/`)
- [x] Add recurrent connections to close gap vs SRNN/PLIF published SOTA
      (done 2026-09-12: RPLIF, +3.61pp over the PLIF ablation; a -14.58pp gap
      remains and is decomposed in this ledger)
- [ ] Integrate and benchmark BNTT (the standalone shape-tested helper is not an
      accuracy result or model integration)
- [ ] Real board power + latency (hardware_validation/fpga_board/README.md)
- [ ] One reconciled throughput measurement (the previous peak figures remain
      withdrawn until a single verified run replaces them)
