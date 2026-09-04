# KV260 Hardware Validation Flow

This folder adds a research-grade local validation pipeline around the existing ES-FA RTL.

## Flow Order
1. `xsim` regression for dense/event scheduler modes.
2. Vivado batch implementation:
   - synthesis reports
   - implementation utilization reports
   - timing summary
3. Report parsing into normalized JSON for analysis.
4. Final KV260 board measurements (optional final stage) merged into the same schema.

## Key Directories
- `vivado/`: Tcl build scripts and constraints.
- `tb/`: xsim-oriented validation testbenches.
- `scripts/`: orchestration and report parsing utilities.

## Main Entry
Run validation for one model:

```bash
python hardware_validation/kv260/scripts/run_hw_validation.py --model-id baseline_paper1 --scheduler-mode dense
```

Run both dense/event:

```bash
python hardware_validation/kv260/scripts/run_hw_validation.py --model-id baseline_paper1 --scheduler-mode both
```

Outputs are written under:

`results/hardware_validation/<model-id>/<mode>/<run-id>/`

