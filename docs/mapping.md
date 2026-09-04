# ES-FA Repository Mapping and Evolution Plan

## Phase 0 Status
- Repo scanned: completed.
- LaTeX files scanned: completed (initially no in-repo `.tex`; now `docs/*.tex` and `paper_output/final_paper.tex` are present and reviewed).
- Working components identified: completed.
- Mapping file created: completed.

## Current File-to-Role Mapping (Preserve)
- `paper1_training/`: config-driven SNN training core, baseline run, INT8 export helpers.
- `paper2_hardware_model/estimators.py`: energy, memory, latency, spike-cost estimators.
- `experiments/`: experiment configs + runners (`exp1`..`exp5`) aligned to proposal themes.
- `iterations/`: iterative v1/v2/v3 refinement on top of experiment outputs.
- `analysis/compare.py`: ranking + plots + best-model selection.
- `paper_output/generate_paper_output.py`: draft markdown paper sections.
- `hardware/`: RTL modules (LIF PE, memory, router, basic/advanced schedulers, top-level).
- `hardware/tb/`: per-module and top-level testbenches.
- `scripts/`: env setup + pipeline runners.
- `software/`: legacy stage-1 training/export/mapping flow.

## What Already Works
- End-to-end software research loop: baseline -> experiments -> iterations -> analysis -> paper markdown.
- Estimator integration during training and metrics logging.
- INT8 export artifacts and spike-raster export.
- RTL hierarchy and testbench coverage for existing hardware blocks.

## What Was Missing Before This Evolution
- Target structure elements for system layer and deployment runtime.
- KV260-local `xsim` + Vivado Tcl batch reporting flow.
- Standardized run manifest schema across all runs.
- Structured hardware report parser for utilization/timing summary.
- Mandatory evidence tables with baseline/optimized absolute + percentage deltas.
- In-repo LaTeX deliverables (`checklist.tex`, `manual.tex`, `final_paper.tex`, `code_notes.tex`).
- Weekly-friendly checkpoint tracker document.

## Preserve / Refactor / Add Decisions
### Preserve
- Keep training core, estimators, experiments, iterations, and core RTL sources.
- Keep existing script entry points for compatibility.

### Refactor (Compatibility-First)
- Extend training artifacts with standardized run manifests and layer-activity exports.
- Add wrappers and new scripts around existing hardware RTL instead of replacing modules.
- Extend analysis to produce evidence-first baseline comparison tables.

### Add
- `hardware_validation/kv260/`: Tcl-driven build flow, xsim regressions, report parsers.
- `system_layer/`: intent engine, task router, and modular execution components.
- `deployment/cli_interface/`: runtime CLI with execution metadata.
- `deployment/runtime/`: adaptive policy and runtime helpers.
- `docs/*.tex`: checklist, manual, code notes.
- `paper_output/final_paper.tex`: NeurIPS-style full paper draft.
- `docs/checkpoints.md`: concise progress ledger.

## Current -> Target Path Mapping
- Existing `paper1_training` -> retained as target `paper1_training`.
- Existing `paper2_hardware_model` -> retained as target `paper2_hardware_model`.
- Existing `hardware/` -> source RTL retained; wrapped by `hardware_validation/kv260/verilog` filelists and build scripts.
- Existing `analysis/` -> retained and extended with evidence generators.
- Existing `paper_output/` -> retained and extended with `final_paper.tex`.
- New target additions: `paper3_extension`, `system_layer`, `deployment`, `hardware_validation`.

## Toolchain Flow Mapping (xsim -> Vivado -> KV260)
1. Local simulation: run `xvlog/xelab/xsim` regressions for dense/event mode and gather cycle/latency logs.
2. Local implementation metrics: run `vivado -mode batch -source build.tcl` and persist all `.rpt` + `.log`.
3. Local parsing: normalize utilization/timing/cycle metrics into JSON under `results/hardware_validation`.
4. Final physical stage: run board measurements on KV260 with the same model/scheduler settings and merge into the same schema.

## Methodology Guardrails
- No deletion of working code.
- All structural changes preserve ES-FA intent and FPGA validation compatibility.
- Dense path remains as hardware/software baseline for mandatory comparisons.
- Adaptive behavior is explicit and fully logged for reproducibility.
