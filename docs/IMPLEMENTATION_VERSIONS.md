# Implementation Versions Architectural Comparison

The **ES-FA Neuromorphic Accelerator Architecture** provides three implementation targets, structured from synthesizable hardware RTL to cycle-accurate system simulation and modern high-throughput host drivers.

---

## 1. Architectural Matrix Comparison

| Architectural Metric | Synthesizable Multi-Core RTL | C99 Cycle-Accurate Engine | .NET 9 HAL & SD-FlashAttention |
| :--- | :--- | :--- | :--- |
| **Directory Target** | [`implementations/v1_synthesizable_rtl_verilog/`](../implementations/v1_synthesizable_rtl_verilog/) | [`implementations/v2_c99_cycle_accurate_engine/`](../implementations/v2_c99_cycle_accurate_engine/) | [`implementations/v3_csharp_net9_hal_sd_flashattention/`](../implementations/v3_csharp_net9_hal_sd_flashattention/) |
| **Primary Execution Domain** | Standard Cell ASIC (28nm/16nm) & FPGA | Cycle-Accurate System Simulation & SIL | Real-Time Edge Runtime & Host HAL Bridge |
| **Implementation Language** | Synthesizable Verilog / SystemVerilog | ANSI C99 (`-O3`, no SIMD intrinsics) | Modern C# 13 / .NET 9 Core Runtime |
| **Mathematical Abstraction** | 4-Stage Fixed-Point Shift-Leak Datapath | Bit-Exact Quantized Neuron/Synapse Arrays | Softmax-Free Attention & Unmanaged Memory Buffers |
| **Arithmetic Precision** | 16-bit State, INT8 Synapse, 16-bit Time | INT16 Membrane, INT8 Weights, FP32 Attn | INT8 Spikes, FP32 Softmax-Free Attention |
| **Clock Frequency / Latency** | 132.29 MHz (ECP5 routed estimate, 50 MHz target) / 850 MHz (ASIC projection) | Engine outputs cycles and toggle energy (the 59.9 GSOP/s figure is withdrawn; no verified run) | Measured Dispatch Latency: **55.7 ns** mean (ledgered 2026-09-16; host dependent) |
| **Energy Consumption** | **$0.42\text{ pJ/SOP}$** (ASIC model) / $3.89\text{ pJ/SOP}$ (FPGA toggle model) | Toggle-Accurate Energy Telemetry Model (event-mode model constant $4.43\text{ pJ/SOP}$) | Zero-Allocation GC-Free Runtime |
| **Streaming Throughput** | 1 Event / PE Cycle (RTL datapath) | Withdrawn (no verified throughput run) | 8.97 Mpps streaming measured on the author workstation (host dependent; see driver telemetry) |
| **On-Chip Learning** | Synthesizable Bi-Exponential STDP Engine | Configurable Hebbian LTP/LTD Matrix | Streaming Feedback & Adaptive Weights |
| **Attention Mechanism** | Sparse Event Accumulator Coincidence Unit | C-Optimized Spike-Driven Attention Kernel | Spike-Driven FlashAttention Kernel |

> **Model-constant note (2026-09-16):** the three energy constants in this
> document are distinct: 0.42 pJ/SOP is the 28 nm ASIC projection, 3.89
> pJ/SOP is the FPGA toggle-model constant, and 4.43 pJ/SOP is the C-engine
> event-mode model value used for every ledgered energy figure. All of them
> are models; board power remains future work. See `EVIDENCE.md`.

---

## 2. Synthesizable Multi-Core RTL (`implementations/v1_synthesizable_rtl_verilog/`)

### Key Microarchitectural Characteristics
- **Synthesizable Datapath**: [`lif_pe_core.v`](../implementations/v1_synthesizable_rtl_verilog/lif_pe_core.v) executes a 4-stage pipeline with fixed-point shift-right leak (`V - (V >>> 3)`), signed threshold comparison, and hard reset.
- **Dual-Bank Synaptic Memory**: [`bram_bank_arbiter.v`](../implementations/v1_synthesizable_rtl_verilog/bram_bank_arbiter.v) provides zero-bubble interleaved access between scheduled spike reads and STDP writebacks.
- **On-Chip STDP Engine**: [`stdp_weight_updater.v`](../implementations/v1_synthesizable_rtl_verilog/stdp_weight_updater.v) implements hardware-native Hebbian plasticity over a $\pm 32$-cycle coincidence window.
- **Target Verification**: Evaluated via [`tb_esfa_rtl.v`](../implementations/v1_synthesizable_rtl_verilog/tb_esfa_rtl.v) using Icarus Verilog or ModelSim.

---

## 3. C99 Cycle-Accurate Engine (`implementations/v2_c99_cycle_accurate_engine/`)

### Key Microarchitectural Characteristics
- **Bit-Accurate Pipeline Emulation**: [`snn_engine.c`](../implementations/v2_c99_cycle_accurate_engine/snn_engine.c) matches RTL register-transfer behavior cycle-by-cycle.
- **SRAM Contention Profiling**: Explicitly counts BRAM bank accesses and read/write collisions per bank, and accumulates the stall cycles that serialized accesses cost.
- **SD-FlashAttention Kernel**: [`spike_attention.c`](../implementations/v2_c99_cycle_accurate_engine/spike_attention.c) models event-driven attention, bypassing over $98\%$ of operations at 85% sparsity.
- **Execution Script**: Run automated benchmark suite via:
  ```bash
  python implementations/v2_c99_cycle_accurate_engine/run_c_engine_benchmark.py
  ```

---

## 4. .NET 9 HAL Driver & SD-FlashAttention (`implementations/v3_csharp_net9_hal_sd_flashattention/`)

### Key Microarchitectural Characteristics
- **Low-Latency HAL**: [`EsfaDriverNet9.cs`](../implementations/v3_csharp_net9_hal_sd_flashattention/EsfaDriverNet9.cs) leverages zero-allocation memory pooling and unmanaged structs (`readonly record struct SpikePacket`) with a lock-free concurrent ingestion queue. The driver reports mean inter-packet gap, mean dispatch latency (measured around the dispatch call), and a clearly labeled model EDP.
- **Modern .NET 9 Core**: Compiled with Server GC, aggressive inlining, and runtime auto-vectorization.
- **Spike-Driven FlashAttention**: [`SpikeDrivenFlashAttention.cs`](../implementations/v3_csharp_net9_hal_sd_flashattention/SpikeDrivenFlashAttention.cs) benchmarks transformer attention without dense matrix multiplication.
- **Execution Script**:
  ```bash
  dotnet run --project implementations/v3_csharp_net9_hal_sd_flashattention/ESFA.Net9.csproj -c Release
  ```
