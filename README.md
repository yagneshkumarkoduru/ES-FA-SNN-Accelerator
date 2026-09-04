# ES-FA / BTP

This is my BTP project workspace for an event-driven SNN FPGA accelerator
(ES-FA). The repo is set up to show the full path from a small spiking neural
network, through hardware-aware estimates, to local FPGA validation scripts and
a simple CLI demo.

The important point for this project is that it runs locally. Training outputs,
plots, Vivado/xsim reports, runtime logs, and PDFs are written back into this
repo under `results/`, `output/`, and `docs/`.

## Current State

What works right now:

- baseline SNN training for a `784 -> 128 -> 64 -> 10` LIF network;
- hardware-aware estimators for spike activity, memory accesses, energy proxy,
  and latency proxy;
- experiment runs for hardware-aware loss and adaptive dataflow;
- analysis outputs: ranking tables, plots, best-model summary, and evidence
  tables;
- Verilog RTL blocks for router, scheduler, event queue, memories, LIF PE, and
  top-level integration;
- KV260-oriented local validation scripts for xsim and Vivado batch runs;
- a small system layer and CLI demo for control, math, and SNN-style classify
  requests;
- LaTeX setup/progress documents in `docs/` and `output/`.

What is still pending:

- physical KV260 board measurements;
- better estimator calibration against measured hardware latency;
- a larger hardware/model matrix for stronger dense-vs-event conclusions.

## Folder Map

```text
p1_training/          Current training code and model exports
p2_hardware_model/    Hardware proxy estimators used during training
experiments/              Proposal experiments, one folder per idea
iterations/               Follow-up refinement runs
analysis/                 Ranking, plots, evidence tables
hardware/                 Verilog RTL and testbenches
hardware_validation/      KV260 xsim/Vivado validation flow
system_layer/             Intent engine, router, execution modules
deployment/               CLI and adaptive runtime policy
software/                 Earlier train/export/bank-mapping scripts
scripts/                  Setup and pipeline entry points
results/                  Local generated metrics, plots, reports, logs
docs/                     Human-readable notes, setup guide, manual
output/             Paper/report sources and compiled PDFs
```

The repo may contain local Vivado/XSim files after a run. They are generated
artifacts, not hand-written source. The `.gitignore` marks the common generated
ones so the committed repo stays clean.

Loose Vivado/XSim files from the last local run were moved out of the root and
kept here:

```text
results/local_tool_runs/root_artifacts_20260505/
```

## First Setup

Create and activate a Python environment:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or use the helper script:

```powershell
.\scripts\setup_env.ps1 -PythonExe .\.venv312\Scripts\python.exe
```

Quick import check:

```powershell
python -c "import torch, torchvision, matplotlib; import p1_training.training_core; print('ok')"
```

## Demo Run

For a quick local demo without Vivado:

```powershell
.\scripts\run_esfa_pipeline.ps1 -PythonExe .\.venv312\Scripts\python.exe -Mode smoke -SkipVivado -SkipHardware
```

For software-only manual steps:

```powershell
python p1_training/train_baseline.py
python experiments/run_first2.py
python experiments/run_remaining.py
python iterations/run_all.py
python analysis/compare.py
python analysis/evidence_report.py
```

For CLI examples:

```powershell
python deployment/cli_interface/main.py --query "hi"
python deployment/cli_interface/main.py --query "strike rate 167.55 balls 39"
python deployment/cli_interface/main.py --query "classify results/runtime/sample_signal.bin"
```

## Hardware Validation

With Vivado/xsim available locally:

```powershell
python hardware_validation/kv260/scripts/run_hw_validation.py --model-id baseline_paper1 --scheduler-mode both --clock-mhz 100
```

To skip Vivado implementation and only run simulator-side validation:

```powershell
python hardware_validation/kv260/scripts/run_hw_validation.py --model-id baseline_paper1 --scheduler-mode both --clock-mhz 100 --skip-vivado
```

Outputs are written under:

```text
results/hardware_validation/<model-id>/<mode>/<run-id>/
```

## Documents

Main project documents:

- `docs/first_time_user_setup_guide.tex`
- `output/progress_report.tex`
- `output/final_paper.tex`
- `docs/architecture.md`
- `docs/checkpoints.md`

Compile the two presentation-facing PDFs locally with `pdflatex`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output docs/first_time_user_setup_guide.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output output/progress_report.tex
```
