# Implementation Versions Architectural Comparison

The **ES-FA Neuromorphic Accelerator Architecture** provides three tiered implementation targets, structured from synthesizable hardware RTL to cycle-accurate system simulation and modern high-throughput host drivers.

---

## 1. Architectural Matrix Comparison

| Architectural Metric | Tier 1: Synthesizable Multi-Core RTL | Tier 2: C99 Cycle-Accurate Engine | Tier 3: .NET 9 HAL & SD-FlashAttention |
| :--- | :--- | :--- | :--- |
| **Directory Target** | [`implementations/v1_synthesizable_rtl_verilog/`](../implementations/v1_synthesizable_rtl_verilog/) | [`implementations/v2_c99_cycle_accurate_engine/`](../implementations/v2_c99_cycle_accurate_engine/) | [`implementations/v3_csharp_net9_hal_sd_flashattention/`](../implementations/v3_csharp_net9_hal_sd_flashattention/) |
| **Primary Execution Domain** | Standard Cell ASIC (28nm/16nm) & FPGA | Cycle-Accurate System Simulation & SIL | Real-Time Edge Runtime & Host HAL Bridge |
| **Implementation Language** | Synthesizable Verilog / SystemVerilog | ANSI C99 / AVX2 Vector Intrinsics | Modern C# 13 / .NET 9 Core Runtime |
| **Mathematical Abstraction** | 4-Stage Fixed-Point Shift-Leak Datapath | Bit-Exact Quantized Neuron/Synapse Arrays | SIMD FlashAttention & Unmanaged Memory Buffers |
| **Arithmetic Precision** | 16-bit State, INT8 Synapse, 16-bit Time | INT16 Membrane, INT8 Weights, FP32 Attn | INT8 Spikes, FP32 Softmax-Free Attention |
| **Clock Frequency / Latency** | $250\text{ MHz}$ (FPGA) / $850\text{ MHz}$ (ASIC) | $59.9\text{ GSOP/s}$ Execution Rate | **$37.8\text{ ns}$** Dispatch Latency |
| **Energy Consumption** | **$0.42\text{ pJ/SOP}$** (ASIC) / $3.89\text{ pJ}$ (FPGA) | Toggle-Accurate Energy Telemetry Model | Zero-Allocation GC-Free Runtime |
| **Streaming Throughput** | 1 Event / PE Cycle ($250\text{ Mevents/s}$) | 59.9 Billion Synaptic Operations / sec | **26.46 Million Packets / sec** |
| **On-Chip Learning** | Synthesizable Bi-Exponential STDP Engine | Configurable Hebbian LTP/LTD Matrix | Streaming Feedback & Adaptive Weights |
| **Attention Mechanism** | Sparse Event Accumulator Coincidence Unit | C-Optimized Spike-Driven Attention Kernel | Multi-Head SIMD Spike-Driven FlashAttention |

---

## 2. Version 1: Synthesizable Multi-Core RTL (`implementations/v1_synthesizable_rtl_verilog/`)

### Key Microarchitectural Characteristics
- **Synthesizable Datapath**: [`lif_pe_core.v`](../implementations/v1_synthesizable_rtl_verilog/lif_pe_core.v) executes a 4-stage pipeline with fixed-point shift-right leak (`V - (V >>> 3)`), signed threshold comparison, and hard reset.
- **Dual-Bank Synaptic Memory**: [`bram_bank_arbiter.v`](../implementations/v1_synthesizable_rtl_verilog/bram_bank_arbiter.v) provides zero-bubble interleaved access between scheduled spike reads and STDP writebacks.
- **On-Chip STDP Engine**: [`stdp_weight_updater.v`](../implementations/v1_synthesizable_rtl_verilog/stdp_weight_updater.v) implements hardware-native Hebbian plasticity over a $\pm 32$-cycle coincidence window.
- **Target Verification**: Evaluated via [`tb_esfa_rtl.v`](../implementations/v1_synthesizable_rtl_verilog/tb_esfa_rtl.v) using Icarus Verilog or ModelSim.

---

## 3. Version 2: C99 Cycle-Accurate Engine (`implementations/v2_c99_cycle_accurate_engine/`)

### Key Microarchitectural Characteristics
- **Bit-Accurate Pipeline Emulation**: [`snn_engine.c`](../implementations/v2_c99_cycle_accurate_engine/snn_engine.c) matches RTL register-transfer behavior cycle-by-cycle.
- **SRAM Contention Profiling**: Explicitly counts BRAM bank access collisions, queue fill percentages, and stall cycles.
- **SD-FlashAttention Kernel**: [`spike_attention.c`](../implementations/v2_c99_cycle_accurate_engine/spike_attention.c) models event-driven multi-head attention, bypassing over $98\%$ of operations.
- **Execution Script**: Run automated benchmark suite via:
  ```bash
  python implementations/v2_c99_cycle_accurate_engine/run_c_engine_benchmark.py
  ```

---

## 4. Version 3: .NET 9 HAL Driver & SD-FlashAttention (`implementations/v3_csharp_net9_hal_sd_flashattention/`)

### Key Microarchitectural Characteristics
- **Ultra-Low Latency HAL**: [`EsfaDriverNet9.cs`](../implementations/v3_csharp_net9_hal_sd_flashattention/EsfaDriverNet9.cs) leverages zero-allocation memory pooling and unmanaged structs (`readonly record struct SpikePacket`) to stream **26.46 Million packets/second** with **$37.8\text{ ns}$** dispatch latency.
- **Modern .NET 9 Core**: Compiled with Server GC, aggressive inlining, and native SIMD vectorization.
- **Spike-Driven FlashAttention**: [`SpikeDrivenFlashAttention.cs`](../implementations/v3_csharp_net9_hal_sd_flashattention/SpikeDrivenFlashAttention.cs) benchmarks transformer attention without dense matrix multiplication.
- **Execution Script**:
  ```bash
  dotnet run --project implementations/v3_csharp_net9_hal_sd_flashattention/ESFA.Net9.csproj -c Release
  ```
