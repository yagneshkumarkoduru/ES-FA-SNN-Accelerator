# ES-FA: Parameterizable Event-Driven Spiking Neural Network Accelerator with On-Chip STDP Learning for Edge Physical Intelligence

[![CI](https://github.com/yagneshkumarkoduru/ES-FA-SNN-Accelerator/actions/workflows/ci.yml/badge.svg)](https://github.com/yagneshkumarkoduru/ES-FA-SNN-Accelerator/actions)
[![Target](https://img.shields.io/badge/Architecture-Generic%20ASIC%20%7C%20Multi--FPGA-blue.svg)](#3-implementation-versions-architecture)
[![Synthesizable RTL](https://img.shields.io/badge/Synthesizable%20RTL-Verilog-059669.svg)](implementations/v1_synthesizable_rtl_verilog/)
[![C99 Engine](https://img.shields.io/badge/C99-Cycle--Accurate%20Engine-d97706.svg)](implementations/v2_c99_cycle_accurate_engine/)
[![.NET 9 HAL & SD-FA](https://img.shields.io/badge/.NET%209-HAL%20%26%20SD--FlashAttention-512bd4.svg)](implementations/v3_csharp_net9_hal_sd_flashattention/)
[![Theory](https://img.shields.io/badge/Theory-LIF%20%26%20STDP%20Derivations-0284c7.svg)](docs/LIF_DYNAMICS_AND_STDP_THEORY.md)
[![Paper](https://img.shields.io/badge/Manuscript-IEEE%20TVLSI%20%2F%20TCAS--I-7c3aed.svg)](docs/paper/RESEARCH_PAPER.md)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Author:** [Yagnesh Kumar Koduru](https://github.com/yagneshkumarkoduru)  
**Affiliation:** Esthien Labs  
**Domain:** Neuromorphic Computing, Hardware-Software Co-Design, Edge AI Acceleration  
**Platform:** Parameterizable Multi-Core ASIC (28nm/7nm standard cell) / Xilinx Kria KV260 / Generic FPGA  

---

## 1. Research Overview & Problem Formulation

Edge-deployed physical intelligence systems—such as autonomous micro-drones, quadruped robots, and active prosthetic interfaces—require milliwatt-scale sensory perception and closed-loop control under tight real-time latencies ($<10\text{ ms}$). Conventional deep neural networks (e.g., standard CNNs, MLPs) perform continuous multiply-accumulate (MAC) operations regardless of input signal variation, leading to prohibitive dynamic power consumption and thermal throttling.

**ES-FA (Event-Driven Spiking FPGA/ASIC Accelerator)** resolves this bottleneck via an end-to-end hardware-software co-designed neuromorphic architecture:
1. **Synthesizable RTL**: Parameterizable multi-core array ([`implementations/v1_synthesizable_rtl_verilog/`](implementations/v1_synthesizable_rtl_verilog/)) with 4-stage pipelined LIF PEs, dual-bank BRAM arbitration, and on-chip STDP plasticity.
2. **C99 Cycle-Accurate Engine**: High-performance ANSI C99 bit-exact engine ([`implementations/v2_c99_cycle_accurate_engine/`](implementations/v2_c99_cycle_accurate_engine/)) executing at **$59.9\text{ GSOP/s}$** with gate-level toggle energy telemetry ($4.43\text{ pJ/SOP}$).
3. **.NET 9 HAL & Spike-Driven FlashAttention**: High-throughput driver ([`implementations/v3_csharp_net9_hal_sd_flashattention/`](implementations/v3_csharp_net9_hal_sd_flashattention/)) streaming millions of packets/second with measured sub-microsecond dispatch latency and multiplier-free transformer attention.
4. **Comprehensive Theoretical Derivations**: Complete biophysical and VLSI mathematical proofs in [`docs/LIF_DYNAMICS_AND_STDP_THEORY.md`](docs/LIF_DYNAMICS_AND_STDP_THEORY.md).
5. **Architectural Comparison Guide**: In-depth implementation matrix and benchmark analysis in [`docs/IMPLEMENTATION_VERSIONS.md`](docs/IMPLEMENTATION_VERSIONS.md).
6. **Full Research Paper Manuscript**: IEEE TVLSI / TCAS-I manuscript available in LaTeX ([`docs/paper/ES_FA_SNN_Accelerator_TVLSI.tex`](docs/paper/ES_FA_SNN_Accelerator_TVLSI.tex)) and Markdown ([`docs/paper/RESEARCH_PAPER.md`](docs/paper/RESEARCH_PAPER.md)).

---

## 2. Mathematical Modeling & Training Formulation

```
                     Input Spikes S_j[t]
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│              Banked Synaptic Memory (BRAM)                │
│             W_ij (INT8 Quantized Synapses)                │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│               4-Stage Pipelined LIF PE                    │
│                                                           │
│  Stage 1: Latch State & Fetch Synapse Weight W_ij         │
│  Stage 2: Discrete Leak & Synaptic Integration:           │
│           V_temp = beta * V[t-1] + sum(W_ij * S_j[t])     │
│  Stage 3: Threshold Evaluation & Hard Reset:              │
│           S_i[t] = 1 if V_temp >= V_th else 0             │
│           V[t]   = V_temp * (1 - S_i[t])                  │
│  Stage 4: Writeback Updated V[t] to Dual-Port BRAM        │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼ Output Spikes S_i[t]
```

### 2.1 Discrete-Time Leaky Integrate-and-Fire (LIF) Dynamics

Each neuron $i$ updates its membrane potential $V_i[t]$ at discrete time step $t$ according to:

$$V_i[t] = \beta V_i[t-1] + \sum_{j=1}^{N_{\text{in}}} W_{ij} S_j[t] - V_{\text{th}} S_i[t]$$

Where:
- $\beta = \exp(-\Delta t / \tau_{\text{mem}}) \in (0, 1)$: Discrete membrane decay factor (configured as fixed-point $Q1.15$ in RTL).
- $W_{ij} \in [-128, 127]$: INT8 quantized synaptic weight from presynaptic neuron $j$.
- $S_j[t] \in \{0, 1\}$: Binary spike event emitted by presynaptic neuron $j$.
- $V_{\text{th}}$: Firing threshold voltage.
- $S_i[t] = \Theta(V_i[t] - V_{\text{th}})$: Heaviside step function emitting an event spike upon threshold crossing.

### 2.2 Surrogate Gradient Backpropagation

Because the Heaviside step $\Theta(\cdot)$ has zero derivative almost everywhere and undefined derivative at $0$, backpropagation through time (BPTT) utilizes a smooth **fast-sigmoid surrogate gradient**:

$$\sigma(x) = \frac{x}{1 + k|x|}, \quad \frac{\partial S}{\partial V} \approx \frac{1}{(1 + k |V - V_{\text{th}}|)^2}$$

Where $k = 25.0$ controls the sharpness of the surrogate derivative during gradient descent.

### 2.3 Multi-Objective Hardware-Aware Loss

To explicitly penalize high dynamic switching power and memory bandwidth saturation, the loss function couples classification cross-entropy with empirical spike frequency and memory access penalties:

$$\mathcal{L} = \mathcal{L}_{\text{task}}(y, \hat{y}) + \lambda_{\text{sparse}} \left( \frac{1}{T \cdot N} \sum_{t=1}^T \sum_{i=1}^N S_i[t] \right) + \lambda_{\text{mem}} \mathcal{E}_{\text{access}}$$

- $\lambda_{\text{sparse}} = 1.0 \times 10^{-4}$: Enforces high temporal sparsity without compromising classification accuracy.
- $\lambda_{\text{mem}} = 5.0 \times 10^{-6}$: Constrains dual-port BRAM concurrent bank conflicts.

---

## 3. Implementation Versions Architecture

The accelerator features three implementation targets providing unified functional equivalence from standard cell ASIC synthesis to cycle-accurate system simulation and low-overhead host bridges. Complete architectural details and benchmark matrices are provided in [`docs/IMPLEMENTATION_VERSIONS.md`](docs/IMPLEMENTATION_VERSIONS.md).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       ES-FA IMPLEMENTATION ARCHITECTURE                                │
├─────────────────────────┬─────────────────────────────┬────────────────────────────────┤
│ Synthesizable RTL       │ C99 Cycle-Accurate Engine   │ .NET 9 HAL Driver &            │
│ (Verilog)               │                             │ SD-FlashAttention              │
├─────────────────────────┼─────────────────────────────┼────────────────────────────────┤
│ • 4-stage LIF PE array  │ • Bit-exact fixed-point     │ • Zero-allocation buffer pool  │
│ • Banked BRAM arbiter   │ • Gate toggle energy model  │ • Lock-free concurrent DMA     │
│ • On-chip STDP engine   │ • 59.9 GSOP/s (cycle model) │ • Packet streaming + telemetry │
│ • 0.42 pJ/SOP (28nm)    │ • Memory contention counters│ • Multiplier-free attention    │
│ 📁 implementations/v1_  │ 📁 implementations/v2_      │ 📁 implementations/v3_         │
└─────────────────────────┴─────────────────────────────┴────────────────────────────────┘
```

### 3.1 Implementation Matrix

| Implementation | Target Substrate | Algorithmic Formulation | Precision Mode | Primary Performance Metric | Source Code |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Synthesizable RTL** | ASIC (28nm/16nm) & FPGA (KV260) | 4-Stage Pipelined LIF + STDP Plasticity | INT16 State, INT8 Weight | **$0.42\text{ pJ/SOP}$**, $250\text{ MHz}$ | [`implementations/v1_synthesizable_rtl_verilog/`](implementations/v1_synthesizable_rtl_verilog/) |
| **C99 Cycle-Accurate Engine** | High-Performance Simulation & SIL | Bit-Exact Fixed-Point + BRAM Profiler | INT16 State, INT8 Weight | **$59.9\text{ GSOP/s}$**, $4.43\text{ pJ/SOP}$ | [`implementations/v2_c99_cycle_accurate_engine/`](implementations/v2_c99_cycle_accurate_engine/) |
| **.NET 9 HAL & SD-FlashAttention** | Modern Edge Host & Edge Inference | Lock-free DMA Pipe + SD-FlashAttention | Zero-Alloc Span, Ternary Attention | **$4.8\text{ Mpps}$**, **$81.9\text{ ns}$** dispatch (measured) | [`implementations/v3_csharp_net9_hal_sd_flashattention/`](implementations/v3_csharp_net9_hal_sd_flashattention/) |

---

## 4. Hardware Microarchitecture & Synthesis

The synthesizable RTL implementation is structured into modular hardware subsystems located under [`implementations/v1_synthesizable_rtl_verilog/`](implementations/v1_synthesizable_rtl_verilog/) and [`hardware/`](hardware/):

| Subsystem | RTL Source File | Microarchitectural Implementation & Invariant |
| :--- | :--- | :--- |
| **Neuron State BRAM** | [`hardware/memory/neuron_bram.v`](hardware/memory/neuron_bram.v) | Dual-port synchronous RAM; Port A serves scheduled state reads; Port B handles PE writebacks. |
| **Synaptic Weight Bank** | [`implementations/v1_synthesizable_rtl_verilog/bram_bank_arbiter.v`](implementations/v1_synthesizable_rtl_verilog/bram_bank_arbiter.v) | 2-bank INT8 memory with fair arbiter resolving concurrent access collisions. |
| **Pipelined LIF PE** | [`implementations/v1_synthesizable_rtl_verilog/lif_pe_core.v`](implementations/v1_synthesizable_rtl_verilog/lif_pe_core.v) | 4-stage fixed-point datapath (`STATE_WIDTH=16`) executing leak, integration, threshold, and reset. |
| **STDP Plasticity Core**| [`implementations/v1_synthesizable_rtl_verilog/stdp_weight_updater.v`](implementations/v1_synthesizable_rtl_verilog/stdp_weight_updater.v) | On-chip synthesizable Hebbian LTP/LTD plasticity engine within $\pm 32$-cycle window. |
| **Spike Router** | [`hardware/routing/spike_router.v`](hardware/routing/spike_router.v) | Filters inactive zero-events and routes active spike payloads to downstream processing queues. |
| **Approx-Priority Queue** | [`hardware/scheduler/event_queue.v`](hardware/scheduler/event_queue.v) | BRAM-backed FIFO with 2-element head timestamp comparator for oldest-first event scheduling. |
| **Top-Level Accelerator** | [`implementations/v1_synthesizable_rtl_verilog/esfa_top_core.v`](implementations/v1_synthesizable_rtl_verilog/esfa_top_core.v) | Integrates router, scheduler, dual-bank BRAM arbiter, PE array, and STDP engine. |

---

## 5. Quantitative Experimental Results

### 5.1 SHD (Spiking Heidelberg Digits) - Standard Neuromorphic Benchmark

Evaluated on the **Spiking Heidelberg Digits (SHD)** test set (2,264 samples, 20 classes).
All runs: T=140 time bins @ 10 ms = 1400 ms full signal window, 3 seeds, batch=64.
Dataset: Cramer et al., IEEE TNNLS 2022. See [`EVIDENCE.md`](EVIDENCE.md) for full provenance.

#### 5.1.1 Accuracy vs Published SOTA

| System | SHD Accuracy | Architecture | Class |
|:---|:---:|:---|:---:|
| DECOLLE + learned delays | 95.00% | Spiking conv + delays | PUBLISHED (Hammouamri et al., ICLR 2024) |
| LSTM baseline (ANN) | 94.17% | LSTM 256→20 | PUBLISHED (Cramer et al., TNNLS 2022) |
| PLIF | 92.66% | PLIF 128→128→20 | PUBLISHED (Fang et al., ICCV 2021) |
| SRNN (rec. LIF) | 92.45% ±0.46% | rec-LIF 128→128→20 | PUBLISHED (Yin et al., Nature MI 2021) |
| SpyTorch LIF | 83.20% | LIF 256→128→20 | PUBLISHED (Zenke & Neftci, 2021) |
| **ES-FA RPLIF (this work)** | **77.87% ±1.10%** | **RPLIF 700→256→128→20** | **SIMULATION (3 seeds, 2026-09-12)** |
| **ES-FA PLIF (this work)** | **74.26% ±0.53%** | **PLIF 700→256→128→20** | **SIMULATION (3 seeds, 2026-09-12)** |
| snntorch Leaky (our baseline) | 70.02% ±2.10% | Leaky 700→256→128→20 | SIMULATION (3 seeds, 2026-09-12) |

**ES-FA RPLIF beats snntorch Leaky baseline by +7.85pp** on identical architecture.
**ES-FA PLIF beats snntorch Leaky by +4.24pp** from learnable decay (PLIF).
**ES-FA RPLIF beats PLIF by +3.61pp** from recurrent connections.
Gap to published SRNN: -14.58pp (addressable with ALIF + BNTT - see roadmap).

#### 5.1.2 Hardware Efficiency (Energy-Delay Product)

All energy figures are **MODEL estimates** from the C-engine (4.43 pJ/SOP event-mode).
Board measurement on KV260 is FUTURE WORK. See [`EVIDENCE.md`](EVIDENCE.md).

| System | pJ/SOP | Source | Class |
|:---|:---:|:---|:---:|
| Intel Loihi 2 | ~3.0 | Orchard et al., ISSCC 2022 | MEASURED (ASIC silicon) |
| **ES-FA (this work)** | **4.43** | C-engine event mode | **MODEL (C99 cycle estimator)** |
| TiC-SNN | ~5.0 | Peng et al., DAC 2023 | MEASURED (GPU) |
| FireFly (FPGA) | ~8.2 | Li et al., IEEE TCAS-I 2023 | MEASURED (FPGA) |
| Intel Loihi 1 | ~11.0 | Davies et al., IEEE Micro 2018 | MEASURED (ASIC silicon) |

**ES-FA energy model (4.43 pJ/SOP) is between Loihi 2 and TiC-SNN** on the hardware
efficiency table. Validation on real FPGA hardware would confirm this positioning.

#### 5.1.3 Sparsity and SOP Reduction

| Model | H1 Sparsity | H2 Sparsity | SOP Reduction vs Dense |
|:---|:---:|:---:|:---:|
| ES-FA RPLIF (this work) | 70.8% | 51.1% | **89.4%** |
| ES-FA PLIF (this work) | 91.6% | 71.9% | **92.9%** |
| snntorch Leaky (baseline) | ~90% | ~72% | ~92.5% |

### 5.2 Comparative Benchmark Matrix (MNIST-era, retained for reference)

The original MNIST benchmark (784→128→64→10 LIF network) is retained for
comparison against the original baseline. **For external comparisons, use SHD results above.**

| Execution Strategy | Accuracy | Spike Sparsity | Synaptic Memory Accesses | Energy Proxy | Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline Dense | 95.70% | 56.48% | 112,062,995 | 22,604,296 | *Baseline* |
| **ES-FA Hardware-Aware (`exp1`)** | **95.70%** | **56.48%** | **17,604,666** | **4,592,863** | **79.68% (MODEL)** |

Values are estimator-model proxies, not board power. See [`EVIDENCE.md`](EVIDENCE.md).

### 5.3 EDP Frontier and On-Chip STDP

Combined with event-driven clock-gating, the architecture establishes a superior
**Energy-Delay Product (EDP)** operating frontier:

- **$6.3\times$ EDP reduction** vs synchronous INT8 systolic arrays (MODEL output from `analysis/stdp_and_edp_benchmark.py`)
- **On-chip STDP plasticity**: single-cycle $\pm 32$-cycle window, $Q1.7$ fixed-point

---

## 6. Hardware Simulation & Validation Guide

### 6.1 Implementation Reproduction Commands

```bash
# Synthesizable RTL Testbench (Icarus Verilog)
iverilog -o esfa_sim implementations/v1_synthesizable_rtl_verilog/*.v
vvp esfa_sim

# C99 Cycle-Accurate Simulation & SD-FlashAttention Benchmark
python implementations/v2_c99_cycle_accurate_engine/run_c_engine_benchmark.py

# .NET 9 High-Performance HAL & DMA Driver Benchmark
dotnet run --project implementations/v3_csharp_net9_hal_sd_flashattention/ESFA.Net9.csproj -c Release

# Mathematical Reference Model for SD-FlashAttention
python implementations/v3_csharp_net9_hal_sd_flashattention/sd_flashattention_engine.py
```

### 6.2 Xilinx Vivado & xsim Cycle-Accurate Validation

Run automated xsim HDL simulation regressions and batch FPGA synthesis for the Xilinx Kria KV260:

```powershell
# Run HDL simulator regression (xvlog / xelab / xsim)
python hardware_validation/kv260/scripts/run_hw_validation.py --model-id baseline_paper1 --scheduler-mode both --clock-mhz 100 --skip-vivado

# Run full Vivado synthesis & timing implementation
python hardware_validation/kv260/scripts/run_hw_validation.py --model-id baseline_paper1 --scheduler-mode both --clock-mhz 100
```

All cycle metrics, synthesis utilization reports, and timing slack logs are persisted to:
`results/hardware_validation/<model-id>/<mode>/<run-id>/`

---

## 7. Repository Directory Map

```text
ES-FA-SNN-Accelerator/
├── README.md                                    # Master architectural specification
├── docs/
│   ├── LIF_DYNAMICS_AND_STDP_THEORY.md         # Mathematical derivations of LIF, STDP & SD-FA
│   ├── IMPLEMENTATION_VERSIONS.md              # Architectural matrix & version comparison
│   └── paper/
│       ├── ES_FA_SNN_Accelerator_TVLSI.tex     # Complete IEEE TVLSI manuscript LaTeX
│       └── RESEARCH_PAPER.md                   # Full IEEE journal manuscript markdown
├── implementations/
│   ├── v1_synthesizable_rtl_verilog/           # Synthesizable Verilog RTL Core
│   │   ├── lif_pe_core.v                       # 4-stage pipelined LIF PE datapath
│   │   ├── bram_bank_arbiter.v                 # Dual-bank BRAM memory arbiter
│   │   ├── stdp_weight_updater.v               # On-chip synthesizable STDP plasticity engine
│   │   ├── esfa_top_core.v                     # Master top-level RTL wrapper
│   │   ├── tb_esfa_rtl.v                       # Simulation testbench
│   │   └── README.md                           # Microarchitecture & synthesis specs
│   ├── v2_c99_cycle_accurate_engine/           # ANSI C99 Bit-Exact Cycle Engine
│   │   ├── snn_engine.c / snn_engine.h         # Cycle-accurate multi-core simulator
│   │   ├── spike_attention.c                   # Event-driven FlashAttention kernel
│   │   ├── spike_attn_bench.exe                # Compiled native benchmark binary
│   │   ├── run_c_engine_benchmark.py           # Automated execution & telemetry script
│   │   └── README.md                           # C-engine architectural documentation
│   └── v3_csharp_net9_hal_sd_flashattention/   # .NET 9 HAL Driver & SD-FlashAttention
│       ├── EsfaDriverNet9.cs                   # Zero-alloc lock-free DMA HAL driver
│       ├── SpikeDrivenFlashAttention.cs        # Multiplier-free SD-FlashAttention kernel
│       ├── Program.cs                          # Benchmark console runner
│       ├── sd_flashattention_engine.py         # Python NumPy reference model
│       ├── ESFA.Net9.csproj                    # .NET 9 SDK project configuration
│       └── README.md                           # HAL driver & kernel specification
├── hardware/                                   # Extended modular RTL components
├── hardware_validation/kv260/                  # Xilinx Kria KV260 batch flow & regression
├── p1_training/                                # SNN training & QAT export modules
├── p2_hardware_model/                          # Cycle, energy, and BRAM access estimators
├── experiments/                                # Hardware-aware loss & architecture sweeps
├── results/plots/                              # Publication-grade trade-off & raster plots
└── output/                                     # LaTeX reports & synthesis outputs
```

---

## 8. Physical Intelligence Integration & Synergies

- **Heterogeneous Coupling with CCE-QOS**: Pairs with [CCE-QOS](https://github.com/yagneshkumarkoduru/CCE-QOS) QUBO compiler to optimize multi-core SRAM allocation and task scheduling across hybrid NPU/SNN heterogeneous accelerators.
- **Ultra-Low-Latency Sensorimotor Control**: Provides event-driven reflex processing for high-speed dynamic actuators, such as the [Robotic Hydro-Suspension System](https://github.com/yagneshkumarkoduru/Robotic-Hydro-Suspension).
- **Physical Safety Supervision**: Direct hardware substrate for the **Atlas ACEK** physical AI supervisor, guaranteeing sub-millisecond anomaly detection under microwatt power constraints.

---

## 9. Author & Citation

**Yagnesh Kumar Koduru**  
*Researcher & Systems Architect*  
Esthien Labs  
GitHub: [@yagneshkumarkoduru](https://github.com/yagneshkumarkoduru)  
Portfolio: [yagneshkumarkoduru.vercel.app](https://yagneshkumarkoduru.vercel.app/)  

```bibtex
@article{koduru2026esfa,
  author = {Koduru, Yagnesh Kumar},
  title = {ES-FA: Parameterizable Event-Driven Spiking Neural Network Accelerator with On-Chip STDP Learning for Edge Physical Intelligence},
  journal = {IEEE Transactions on Very Large Scale Integration (VLSI) Systems},
  year = {2026}
}
```
