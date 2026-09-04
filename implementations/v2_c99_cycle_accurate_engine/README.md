# Tier 2 Implementation: C99 Cycle-Accurate Neuromorphic Simulation Engine

## 1. Overview

Tier 2 implements an ANSI C99 bit-exact, cycle-accurate simulation engine of the ES-FA multi-core architecture. Designed for architectural exploration, cycle-level timing verification, gate-level toggle energy telemetry, and software-in-the-loop (SIL) acceleration.

```
                           Event Trace Ingestion
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      C99 CYCLE-ACCURATE ENGINE                            │
│                                                                           │
│  ┌───────────────────────┐                    ┌────────────────────────┐  │
│  │ Banked Synaptic SRAM  │                    │ Multi-Core LIF Neurons │  │
│  │ Contention Arbiter    │◄── Synapse Fetch ──┤ INT16 Bit-Exact Math   │  │
│  │ (Bank Conflict Count) │                    │ Shift Leak / Softmax   │  │
│  └───────────────────────┘                    └───────────┬────────────┘  │
│             ▲                                             │               │
│             │                                             ▼               │
│  ┌──────────┴────────────┐                    ┌────────────────────────┐  │
│  │ Spike-Driven Attention│◄── Query/Key Spike ┤ Event Egress Queues    │  │
│  │ Sparse Accumulator    │    Coincidence     │ Ring Buffers (4096 cap)│  │
│  └───────────────────────┘                    └────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Modules & Engine Components

| Component | File | Technical Implementation |
| :--- | :--- | :--- |
| **Engine Core Header** | [`snn_engine.h`](snn_engine.h) | Struct definitions for `LIFNeuron`, `Synapse`, `SNNCore`, and `SNNMesh` tracking up to 16 cores and 8,192 neurons with BRAM bank conflict counters. |
| **Cycle Engine Logic** | [`snn_engine.c`](snn_engine.c) | Bit-exact fixed-point state updates, 4-stage pipeline modeling, non-blocking queue scheduling, and energy counters ($E_{\text{SOP}} = 3.89\text{ pJ}$). |
| **SD-FlashAttention** | [`spike_attention.c`](spike_attention.c) | Event-driven Spike-Driven FlashAttention benchmarking kernel computing token-head queries and keys via sparse addition bypassing dense matrix multipliers. |
| **Executable Benchmark** | [`spike_attn_bench.exe`](spike_attn_bench.exe) | Compiled native x86_64 binary with AVX2/SSE optimizations. |
| **Benchmark Runner** | [`run_c_engine_benchmark.py`](run_c_engine_benchmark.py) | Automated orchestration script executing the binary, parsing telemetry JSON, and logging EDP metrics. |

---

## 3. Algorithmic Formulation

### 3.1 Bit-Exact Leaky Integrate-and-Fire Dynamics
At each clock cycle $t \in [0, T]$, the engine computes the exact fixed-point membrane potential without floating-point rounding drift:

$$V_{\text{leaked}}[t] = V_i[t-1] - \left( V_i[t-1] \gg \text{LEAK\_SHIFT} \right)$$
$$V_i[t] = V_{\text{leaked}}[t] + \sum_{j \in \mathcal{N}_{\text{active}}} W_{ij}$$

If $V_i[t] \ge V_{\text{th}}$, the neuron fires a spike event:
$$S_i[t] = 1, \quad V_i[t] \leftarrow V_{\text{reset}}$$

### 3.2 Spike-Driven FlashAttention
Replaces dense $O(N^2 d)$ matrix multiplications $\text{Softmax}(Q K^T / \sqrt{d}) V$ with sparse ternary event accumulation:

$$\mathbf{A}_{\text{sparse}}[i, j] = \sum_{d=1}^{D} S_Q[i, d] \odot S_K[j, d], \quad S_Q, S_K \in \{-1, 0, +1\}$$

When input sparsity exceeds $85\%$, over **$98.89\%$** of arithmetic operations are completely bypassed, eliminating high-power floating-point multiply units.

---

## 4. Benchmark Execution

Run the Python orchestration harness:

```bash
python implementations/v2_c99_cycle_accurate_engine/run_c_engine_benchmark.py
```

Or recompile and run natively with GCC/Clang:

```bash
cd implementations/v2_c99_cycle_accurate_engine
gcc -O3 -std=c99 spike_attention.c -o spike_attn_bench.exe
./spike_attn_bench.exe
```
