# ES-FA Checkpoints

## Milestone Progress
- Phase 0: Completed (full repo + LaTeX scan, mapping documented).
- Phase 1: Completed (core modularization, manifests, exports, reproducibility controls).
- Phase 2: Completed (xsim + Vivado Tcl flow, report capture, parsing to normalized JSON).
- Phase 3: Completed (intent engine, task router, system modules).
- Phase 4: Completed (CLI runtime with execution metadata and diagnostics).
- Phase 5: Completed (adaptive policy + decision logs + static/adaptive comparator outputs).
- Phase 6: Completed (mandatory raw tables, percentage tables, and interpretations).
- Phase 7: Completed (training/raster/trade-off/correlation/adaptive plots).
- Phase 8: Completed (checklist/manual/final paper/code notes LaTeX outputs).
- Phase 9: Active (weekly tracker maintained here).

## Completed Work
- Preserved existing training, estimator, experiments, iterations, and analysis foundations.
- Added standardized `run_manifest.json` and expanded per-layer activity exports.
- Added `hardware_validation/kv260` with:
  - xsim regression (`dense`/`event`),
  - Vivado batch implementation flow (`build.tcl`),
  - full `.rpt`/`.log` persistence and parser outputs (`vivado_metrics.json`, `hardware_metrics.json`).
- Added system runtime stack:
  - `system_layer/intent_engine`,
  - `system_layer/task_router`,
  - `system_layer/modules/{snn_module,math_module,control_module}`,
  - `deployment/runtime/adaptive_policy.py`,
  - `deployment/cli_interface/main.py`.
- Added evidence pipeline:
  - `analysis/compare.py` for training/raster/trade-off plots,
  - `analysis/evidence_report.py` for mandatory baseline comparisons and percentage deltas.
- Added documentation deliverables:
  - `docs/checklist.tex`,
  - `docs/manual.tex`,
  - `docs/code_notes.tex`,
  - `output/final_paper.tex`.

## Current Status
- Local-toolchain hardware evidence exists for baseline and adaptive models in `dense` and `event` modes.
- Analysis artifacts and comparison tables are regenerated from current results.
- Pipeline is runnable in smoke/full modes with strict phase ordering.

## Next Tasks
1. Add KV260 board-level physical runs (same model/mode matrix) into the existing hardware schema.
2. Calibrate estimator parameters to reduce current estimator-vs-measured latency error.
3. Expand model matrix beyond 2 models for stronger dense/event and adaptive conclusions.

## Blockers
- No active software/toolchain blocker for local xsim + Vivado flow.
- Remaining blocker is physical board measurement scheduling for final KV260 evidence closure.

## Run History
- 2026-05-04T20:04:02.795142Z | Phase 0 | ok | Mapping file validated
- 2026-05-04T20:32:01.943384+00:00 | Phase 0 | ok | Mapping file validated
- 2026-05-04T20:32:58.377773+00:00 | Phase 1 | ok | Baseline training completed in smoke mode
- 2026-05-04T20:33:53.280781+00:00 | Phase 3 | ok | Experiments/iterations finished
- 2026-05-05T03:45:43+00:00 | Phase 2 | ok | Baseline/optimized model matrix validated with xsim + Vivado
- 2026-05-05T04:12:00+00:00 | Phase 6-7 | ok | Evidence tables and plots regenerated from latest metrics
- 2026-05-05T06:01:17.111379+00:00 | Phase 0 | ok | Mapping file validated
- 2026-05-05T06:12:04.834102+00:00 | Phase 0 | ok | Mapping file validated
- 2026-05-05T06:12:49.225115+00:00 | Phase 1 | ok | Baseline training completed in smoke mode
- 2026-05-05T06:14:02.554884+00:00 | Phase 2 | ok | Experiments/iterations finished
- 2026-05-05T06:14:02.568885+00:00 | Phase 3 | skipped | Hardware validation skipped by flag
- 2026-05-05T06:14:02.577885+00:00 | Phase 4 | ok | System layer modules available
- 2026-05-05T06:14:02.842957+00:00 | Phase 5 | ok | CLI query paths validated
- 2026-05-05T06:14:03.302422+00:00 | Phase 6 | ok | Static/adaptive runtime comparison generated
- 2026-05-05T06:14:05.126258+00:00 | Phase 7 | ok | Analysis outputs generated
- 2026-05-05T06:14:06.162993+00:00 | Phase 8 | ok | Visualization and evidence outputs generated
- 2026-05-05T06:14:06.296936+00:00 | Phase 9 | ok | Paper output markdown generation completed
- 2026-05-05T06:14:06.309969+00:00 | Phase 10-11 | ok | Pipeline run completed
- 2026-05-05T06:21:56.360020+00:00 | Phase 0 | ok | Mapping file validated
- 2026-05-05T06:22:21.515993+00:00 | Phase 1 | ok | Baseline training completed in smoke mode
- 2026-05-05T06:23:14.574037+00:00 | Phase 2 | ok | Experiments/iterations finished
- 2026-05-05T06:23:14.596648+00:00 | Phase 3 | skipped | Hardware validation skipped by flag
- 2026-05-05T06:23:14.604162+00:00 | Phase 4 | ok | System layer modules available
- 2026-05-05T06:23:14.842400+00:00 | Phase 5 | ok | CLI query paths validated
- 2026-05-05T06:23:15.286458+00:00 | Phase 6 | ok | Static/adaptive runtime comparison generated
- 2026-05-05T06:23:16.939179+00:00 | Phase 7 | ok | Analysis outputs generated
- 2026-05-05T06:23:17.937081+00:00 | Phase 8 | ok | Visualization and evidence outputs generated
- 2026-05-05T06:23:18.044028+00:00 | Phase 9 | ok | Paper output markdown generation completed
- 2026-05-05T06:23:18.059718+00:00 | Phase 10-11 | ok | Pipeline run completed
