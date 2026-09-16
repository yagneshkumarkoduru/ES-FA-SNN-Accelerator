# ES-FA: A Parameterizable Event-Driven Spiking Neural Network Accelerator with On-Chip STDP Learning and Banked Synaptic Memory for Edge Physical Intelligence

**Author:** Koduru Yagnesh Kumar  
**Affiliation:** Independent Researcher  
**Contact:** `yagneshkumarkoduru@gmail.com`  
**Target Publication Venue:** IEEE Transactions on Very Large Scale Integration Systems (TVLSI) / IEEE TCAS-I  

---

## Abstract

Real-time physical intelligence on energy-constrained robotic and embedded platforms is critically bottlenecked by the high power consumption and continuous memory traffic of synchronous multiply-accumulate (MAC) tensor accelerators. Spiking Neural Networks (SNNs) offer a biologically grounded alternative where computation is triggered exclusively by discrete spatio-temporal events. However, existing neuromorphic silicon either relies on static offline weights or experiences severe memory bank contention under bursty event rates.

In this paper, we present **ES-FA**, an open, parameterizable, technology-independent neuromorphic computing architecture and verification framework designed for edge physical AI. ES-FA delivers four foundational contributions:
1. **A 4-stage pipelined Leaky Integrate-and-Fire (LIF) processing element** that executes exact fixed-point discrete leaky integration with homeostatic threshold adaptation without hardware floating-point multipliers;
2. **A dual-banked parity-interleaved synaptic SRAM arbiter** for concurrent access arbitration (contention-reduction figures pending a dedicated arbiter benchmark; the 68.4% figure elsewhere belongs to the NPU bank model, not this arbiter);
3. **A synthesizable on-chip Spike-Timing-Dependent Plasticity (STDP) learning engine** enabling autonomous local adaptation in the field; and
4. **An open verification stack** spanning a bit-exact C99 cycle engine, a .NET 9 host driver, a self-checking Verilog testbench, and a vendor-free synthesis and place-and-route flow, with every artifact number ledgered in `EVIDENCE.md`.

**Evaluation (3 seeds unless noted).** The manuscript also provides mathematical derivations for synaptic weight boundedness under Poisson jitter and worst-case memory arbitration stalls (proof review pending). On the Spiking Heidelberg Digits benchmark, the recurrent PLIF configuration reaches **77.87% ± 1.10** (versus 74.26% ± 0.53 for the non-recurrent ablation and 70.02% ± 2.10 for an identically trained snntorch baseline) while cutting synaptic operations by 89.4% (model). The published SRNN and PLIF references remain 14.6-14.8 points higher; the gap is decomposed into window, epoch, and regularization factors in `EVIDENCE.md`. The RTL core passes its self-checking simulation (Icarus Verilog), synthesizes to 890 cells on ECP5, and routes at a 132.29 MHz maximum-frequency estimate against a 50 MHz target. The .NET host driver streams at 8.97 Mpps with 55.7 ns mean dispatch latency on the author's workstation (host-dependent). Energy figures remain model estimates from the C engine (4.43 pJ/SOP event-mode); the earlier MNIST-era figures (95.7% accuracy, 79.68% proxy reduction) are retained only as historical records, and peak-throughput figures (128.0 GSOP/s, 59.9 GSOP/s, 3.89 pJ/SOP) are withdrawn pending one reconciled measurement run. Board power measurements remain future work; see `EVIDENCE.md`.

---

## 1. Introduction

Autonomous physical machines—including agile quadrupeds, active prosthetic joints, and high-frequency active suspension systems—must execute continuous state estimation and closed-loop motor control under sub-millisecond reaction times and sub-5 W power budgets. Modern deep convolutional neural networks and vision transformers deployed on conventional systolic tensor arrays fail to meet these constraints: their synchronous clocking forces every processing element (PE) to toggle every cycle regardless of whether sensory inputs have changed, generating excessive dynamic power dissipation ($P_{\text{dyn}} = \alpha C V_{dd}^2 f$).

In biological nervous systems, information processing is fundamentally event-driven: biological neurons remain quiescent until action potentials arrive, achieving over $80\%$ temporal sparsity. However, realizing these theoretical gains in physical silicon encounters severe architectural and algorithmic bottlenecks:
1. **Memory Access Contention:** High-fanout spike bursts generate concurrent memory read requests to shared synaptic memory arrays, causing pipeline starvation and arbitration stalls.
2. **Offline Learning Rigidity:** Prominent neuromorphic processors (such as IBM TrueNorth) restrict on-chip synapses to static read-only weights trained offline via backpropagation. Consequently, deployed agents cannot adapt to mechanical wear, changing actuator dynamics, or novel environmental friction in the field.
3. **Hardware-Software Disconnect:** Neuromorphic algorithms are frequently developed in high-level Python libraries that obscure cycle-accurate hardware timing, memory hierarchies, and bus arbitration latencies.

**ES-FA** overcomes these challenges via full-stack architectural parameterization, synthesizable on-chip plasticity, and a bit-accurate dual C/C# verification ecosystem.

---

## 2. Mathematical Theory & Formal Convergence Proofs

### 2.1 Discretized Leaky Integrate-and-Fire Dynamics
The continuous-time subthreshold dynamics of a biological neuron membrane potential $V_i(t)$ subject to synaptic input currents $I_i(t)$ and external bias $I_{\text{ext}}(t)$ are defined by:

$$\tau_m \frac{dV_i(t)}{dt} = -(V_i(t) - V_{\text{rest}}) + R_m \sum_{j=1}^{N_{\text{in}}} w_{ij} S_j(t) + I_{\text{ext}}(t)$$

where $\tau_m = R_m C_m$ is the membrane time constant, $w_{ij} \in \mathbb{R}$ represents the synaptic coupling strength, and $S_j(t) = \sum_k \delta(t - t_j^k)$ models the input Dirac spike train.

To synthesize this model on digital silicon with zero floating-point overhead, we apply forward Euler discretization over timestep $\Delta t$:

$$V_i[t] = V_i[t-1] - \lfloor V_i[t-1] \gg \beta \rfloor + \sum_{j=1}^{N_{\text{in}}} w_{ij} S_j[t]$$

where $\beta \in \mathbb{Z}^+$ is an integer arithmetic right-shift factor related to $\tau_m$ by $\beta = -\log_2(1 - \Delta t / \tau_m)$. For a typical parameterization ($\Delta t = 1\text{ ms}$, $\tau_m = 8\text{ ms}$), $\beta = 3$, giving an exact decay multiplier of $1 - 2^{-3} = 0.875$.

When $V_i[t] \ge V_{\text{th}}[t]$, the neuron emits an action potential $S_i[t] = 1$, the membrane potential is hard-reset to $V_{\text{reset}} = 0$, and an internal counter enforces a refractory period $T_{\text{ref}}$:

$$S_i[t] = \Theta\left(V_i[t] - V_{\text{th}}[t]\right), \quad V_i[t] \leftarrow V_i[t] \cdot (1 - S_i[t]) + V_{\text{reset}} \cdot S_i[t]$$

---

### 2.2 Theorem 1: Synaptic Weight Boundedness Under Poisson Jitter

**Theorem 1.** *Let input spike arrivals follow a Poisson renewal process with rate $\lambda_{\text{pre}}$, and let output firing occur with rate $\lambda_{\text{post}}$. If the integral of the depression kernel strictly dominates the potentiation kernel:*

$$\int_{-\infty}^0 A_- e^{t/\tau_-} dt > \int_0^\infty A_+ e^{-t/\tau_+} dt \iff A_- \tau_- > A_+ \tau_+$$

*then for any bounded input rate $\lambda_{\text{pre}} < \infty$, the expected weight trajectory $\mathbb{E}[w(t)]$ is mathematically bounded:*

$$\lim_{t \to \infty} \mathbb{E}[w(t)] \le w^* < \infty$$

*preventing unbounded weight saturation without requiring artificial hard clamping.*

**Proof:**  
Consider the continuous expectation of synaptic drift $\frac{d\mathbb{E}[w]}{dt}$:

$$\frac{d\mathbb{E}[w]}{dt} = \int_{-\infty}^\infty \Delta w(s) \cdot C_{\text{pre,post}}(s) \, ds = \lambda_{\text{pre}} \lambda_{\text{post}} \left[ \int_0^\infty A_+ e^{-s/\tau_+} ds - \int_{-\infty}^0 A_- e^{s/\tau_-} ds \right] + \lambda_{\text{pre}} \int_0^\infty A_+ e^{-s/\tau_+} \Delta P(s) \, ds$$

Evaluating the steady-state integrals:

$$\int_0^\infty A_+ e^{-s/\tau_+} ds = A_+ \tau_+, \quad \int_{-\infty}^0 A_- e^{s/\tau_-} ds = A_- \tau_-$$

Substituting into the drift expression:

$$\frac{d\mathbb{E}[w]}{dt} = \lambda_{\text{pre}} \lambda_{\text{post}} (A_+ \tau_+ - A_- \tau_-) + \mathcal{O}(\Delta P)$$

Because $A_- \tau_- > A_+ \tau_+$, the leading coefficient $(A_+ \tau_+ - A_- \tau_-) = -\epsilon < 0$. Therefore, as $\mathbb{E}[w]$ increases, $\lambda_{\text{post}}$ increases via the activation function, amplifying the negative drift term $-\epsilon \lambda_{\text{pre}} \lambda_{\text{post}}$. By Lyapunov stability with candidate function $L(w) = \frac{1}{2} w^2$, $\dot{L}(w) < 0$ for all $w > w^*$, proving global asymptotic stability and boundedness of the synaptic weight distribution. $\blacksquare$

---

### 2.3 Theorem 2: Worst-Case Memory Bank Contention Bound

**Theorem 2.** *For a neuromorphic core servicing $M$ active postsynaptic neuron updates per cycle across $B$ interleaved physical SRAM banks with random hash distribution:*

$$P_{\text{conflict}}(M, B) = 1 - \prod_{k=0}^{M-1} \left(1 - \frac{k}{B}\right)$$

*Under our dual-bank parity architecture ($B = 2$) with destination LSB steering ($k = \text{Neuron\_ID} \pmod 2$), memory conflicts are completely eliminated ($P_{\text{conflict}} = 0$) for any pair of adjacent neurons $(2j, 2j+1)$, since adjacent destination identifiers are guaranteed to map to distinct physical banks. Quantitative stall-reduction ratios for arbitrary (non-adjacent) access interleavings depend on the live event stream and are reported separately from the dedicated arbiter benchmark rather than asserted here.*

---

## 3. Silicon Microarchitecture & Hardware Implementation

```text
+-----------------------------------------------------------------------------------+
|                        ES-FA MULTI-CORE NEUROMORPHIC FABRIC                       |
|                                                                                   |
|   +--------------------------+                  +--------------------------+      |
|   |         CORE 0           |                  |         CORE 1           |      |
|   |  +--------------------+  |                  |  +--------------------+  |      |
|   |  | 4-Stage LIF PE     |  |                  |  | 4-Stage LIF PE     |  |      |
|   |  +--------------------+  |                  |  +--------------------+  |      |
|   |  | Parity SRAM (B0/B1)|  |                  |  | Parity SRAM (B0/B1)|  |      |
|   |  +--------------------+  |  Low-Latency     |  +--------------------+  |      |
|   |  | On-Chip STDP Unit  |  |<---------------->|  | On-Chip STDP Unit  |  |      |
|   |  +--------------------+  |  Event Mesh NoC  |  +--------------------+  |      |
|   |  | Adaptive Leak Unit |  |                  |  | Adaptive Leak Unit |  |      |
|   |  +--------------------+  |                  |  +--------------------+  |      |
|   +--------------------------+                  +--------------------------+      |
|                ^                                              ^                   |
|                |                                              |                   |
|   +---------------------------------------------------------------------------+   |
|   | AXI4-Lite Control Slave  |  AXI4-Stream Ingestion  |  AXI4-Stream Egress  |   |
|   +---------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------+
```

### Core Synthesizable Verilog Modules:
1. **[`hardware/top/snn_accelerator_generic.v`](../../hardware/top/snn_accelerator_generic.v)**: Technology-independent multi-core array with parameterizable dimensions (`NUM_CORES`, `NEURONS_PER_CORE`), AXI4-Lite control, and AXI4-Stream event bus.
2. **[`hardware/compute/lif_neuron_pe.v`](../../hardware/compute/lif_neuron_pe.v)**: 4-stage pipeline isolating state latching, leaky integration, threshold comparison, and write-back.
3. **[`hardware/compute/adaptive_leak_engine.v`](../../hardware/compute/adaptive_leak_engine.v)**: Homeostatic firing rate stabilizer adjusting $V_{\text{th}}[t]$ and leak shift without multipliers.
4. **[`hardware/compute/spike_driven_flash_attention.v`](../../hardware/compute/spike_driven_flash_attention.v)**: Multi-head Spike-Driven FlashAttention accelerator eliminating $O(N^2)$ Softmax and floating-point multipliers via event-driven ternary coincidence accumulation.
5. **[`hardware/compute/stdp_learning_engine.v`](../../hardware/compute/stdp_learning_engine.v)**: Hardware plasticity unit updating synapses in a single clock cycle.
6. **[`hardware/memory/weight_bram_bank.v`](../../hardware/memory/weight_bram_bank.v)**: Dual-banked memory with parity interleaving.

---

## 4. Hardware-Software Co-Design

1. **Cycle-Accurate C99 Simulation Engine ([`c_engine/`](../../c_engine/))**:
   - Exact fixed-point pipeline emulation matching RTL.
   - Comprehensive gate-level toggle power modeling ($P_{\text{dyn}} = \frac{1}{2}\alpha C V_{dd}^2 f$).
   - Compiles via GCC/Clang: `gcc -O3 main.c snn_engine.c -o snn_simulator.exe -lm`.
2. **Real-Time C# Embedded HAL Driver ([`csharp_driver/`](../../csharp_driver/))**:
   - Zero-allocation lock-free ring buffering in .NET 9.
   - Measured throughput: **8.97 Million packets/sec** with **55.7 ns** mean dispatch latency (author workstation, .NET 9.0.305, 2026-09-16; host-dependent; see `EVIDENCE.md`). Earlier 2.46 Mpps / 405 ns and 4.8 Mpps / 81.9 ns figures are superseded by this ledgered run.

---

## 5. Evaluation on Spiking Heidelberg Digits

Protocol: 2,264-sample test split, 20 classes, T=140 bins at 10 ms (1,400 ms
window), batch 64, three seeds {42, 123, 999}; dataset from Cramer et al.,
IEEE TNNLS 2022. All figures below are ledgered in `EVIDENCE.md`.

| System | SHD accuracy | Architecture | Class |
| :--- | :---: | :--- | :---: |
| SRNN (reference) | 92.45% ± 0.46% | rec-LIF 128-128-20 | PUBLISHED (Yin et al., Nature MI 2021) |
| PLIF (reference) | 92.66% | PLIF 128-128-20 | PUBLISHED (Fang et al., ICCV 2021) |
| **ES-FA RPLIF (this work)** | **77.87% ± 1.10%** | RPLIF 700-256-128-20 | SIMULATION (3 seeds) |
| ES-FA PLIF ablation | 74.26% ± 0.53% | PLIF 700-256-128-20 | SIMULATION (3 seeds) |
| snntorch Leaky baseline | 70.02% ± 2.10% | Leaky 700-256-128-20 | SIMULATION (3 seeds) |

- Recurrence alone accounts for +3.61 points over the PLIF ablation and
  +7.85 points over the identically trained snntorch baseline.
- The remaining gap to the published references (-14.58 points to SRNN) is
  decomposed into three measured factors in `EVIDENCE.md`: window geometry
  (T=140 at 10 ms versus SRNN's T=50 at 2 ms), training length (50 epochs
  versus 200+), and sparsity regularization; removing the regularization
  was measured and did not help (75.81%, ledgered as a negative result).
- Spike-operation reduction of 89.4% (model) applies at the RPLIF
  operating point; energy remains a C-engine model estimate (4.43 pJ/SOP
  event mode), not a board measurement.

### 5.1 Open-source CAD verification of the RTL core

The four RTL modules plus a self-checking testbench pass under Icarus
Verilog (membrane accumulation, threshold firing through the bank arbiter,
STDP writeback). Yosys maps the top core to 890 cells on the Lattice ECP5
fabric, and nextpnr-ecp5 routes the netlist with a maximum-frequency
estimate of **132.29 MHz** against a 50 MHz target. The flow is
reproducible with `py -3 hardware_validation/open_cad/run_open_cad.py`;
full provenance in `EVIDENCE.md`.

---

## 6. Comparative Silicon Benchmarking

| Metric | IBM TrueNorth | Intel Loihi 1 | Intel Loihi 2 | Tsinghua Tianjic | SpiNNaker-2 | ES-FA (Ours) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Process Node** | 28 nm | 14 nm FinFET | Intel 4 | 28 nm | 22 nm FD-SOI | **Generic / 28 nm target** |
| **Target Platform** | Custom ASIC | Custom ASIC | Custom ASIC | Hybrid ASIC | Many-core ASIC | **Param. ASIC / FPGA** |
| **On-Chip Learning** | None (Offline) | Programmable | Microcode | None | Software C | **Synthesizable STDP** |
| **Clock Frequency** | 1 MHz | 1000 MHz | 1000 MHz | 300 MHz | 250 MHz | **132.29 MHz (ECP5 routed estimate)** |
| **Energy / Synaptic Op** | 26.0 pJ | 23.6 pJ | 19.4 pJ | 12.0 pJ | 18.0 pJ | **4.43 pJ/SOP (C-engine model)** |
| **Peak Throughput** | 46.0 GSOP/s | 100.0 GSOP/s | 120.0 GSOP/s | 150.0 GSOP/s | 125.0 GSOP/s | **withdrawn (no verified run)** |
| **Memory Banking** | Monolithic | Interleaved | Interleaved | Banked | SRAM/Core | **Parity Dual-Bank** |
| **Host Interface** | Proprietary | PCIe | PCIe | Custom | AXI4 / Ethernet | **AXI4-Lite / Stream** |
| **EDP Advantage** | 1.0x (Ref) | 4.2x | 5.1x | 3.8x | 4.0x | **6.3x (model)** |

Reference-system figures are published specifications. The ES-FA column
carries measured or model values from `EVIDENCE.md`: the routed estimate,
the C-engine model energy, and the modeled EDP; the peak-throughput and
28 nm silicon-frequency figures are not verified and are marked
accordingly rather than quoted.

---

## 7. Conclusion

ES-FA is an open, parameterizable event-driven architecture with native
on-chip STDP adaptation, parity-banked synaptic memory, and homeostatic
threshold scaling, verified end to end without vendor tools: the RTL core
passes its self-checking simulation, maps to 890 cells on ECP5, and routes
at a 132.29 MHz maximum-frequency estimate; the .NET host driver measures
8.97 Mpps with 55.7 ns dispatch; and the recurrent PLIF configuration
reaches 77.87% ± 1.10% on SHD with an 89.4% modeled reduction in synaptic
operations. The remaining accuracy gap to published recurrent models and
the board-level power measurements are stated openly as the next
milestones; see `EVIDENCE.md` for every ledgered number.
