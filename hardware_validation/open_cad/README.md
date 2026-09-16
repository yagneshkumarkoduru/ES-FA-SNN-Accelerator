# Open-source CAD verification flow (ES-FA v1 RTL core)

Vendor-free evidence flow for the synthesizable RTL core, runnable on any
machine with the OSS CAD Suite. It complements the Vivado-based flow in
`hardware_validation/fpga_board/` (which requires vendor tools and a board)
and answers the timing question without either.

## What it does

1. **Simulation**: Icarus Verilog compiles the four RTL modules plus the
   self-checking testbench (`tb_esfa_rtl.v`) and runs it; the bench checks
   pipelined membrane accumulation, threshold firing through the bank
   arbiter, and the STDP weight writeback.
2. **Supporting testbench suite**: all nine unit testbenches under
   `hardware/tb/` are compiled with their modules and run self-checking
   (scheduler ordering, event queue ordering, LIF PE thresholding, BRAM
   readback, bank arbitration, spike routing, and both top-level
   integrations).
3. **Synthesis**: Yosys synthesizes `esfa_top_core` for the Lattice ECP5
   fabric and reports the mapped cell statistics.
4. **Place and route**: nextpnr-ecp5 places and routes the netlist and
   reports the routed maximum clock frequency estimate at a target of
   50 MHz.

## Run

```powershell
py -3 hardware_validation/open_cad/run_open_cad.py
```

Set `ESFA_OSS_CAD_ROOT` (or pass `--oss-cad-root`) if the OSS CAD Suite is
not at `C:\oss-cad-suite`. The parsed report is written to
`out/open_cad_report.json`.

## Latest results (2026-09-16)

| Stage | Result |
| :--- | :--- |
| Simulation | PASS: 6 spikes fired, 12 active cycles, 1 STDP weight update (weight 22 as expected) |
| Unit testbenches | **9/9 PASS** (see `supporting_testbenches` in the report) |
| Synthesis | 890 cells on ECP5: LUT4 308, TRELLIS_FF 395, TRELLIS_DPR16X4 48, CCU2C 97, PFUMX 37, L6MUX21 2 |
| Place and route | Routed Fmax estimate **132.29 MHz** at a 50 MHz target (PASS) |
| Tools | Icarus Verilog 14.0, Yosys 0.68+120, nextpnr-ecp5 (OSS CAD Suite) |

Testbench fixes made while wiring this flow (2026-09-16): the FlashAttention
module used undeclared loop indices (`d_idx`, `r_idx`), three testbenches
sampled registered outputs one delta before the registers updated
(`tb_neuron_bram`, `tb_spike_router`, `tb_weight_bram_bank`), the LIF PE
testbench printed values without asserting them, and one testbench used a
non-standard success marker. All nine now carry explicit PASS criteria and
pass under Icarus Verilog.

## Boundary (carried in the report)

Open-source toolchain evidence only: behavioral simulation, generic
synthesis statistics, and a routed timing estimate. No vendor timing
signoff, no board measurement, and no power measurement. Silicon targets
in the manuscript remain architectural projections until taped out; this
flow is the reproducible bridge between the RTL source and a mapped
netlist.
