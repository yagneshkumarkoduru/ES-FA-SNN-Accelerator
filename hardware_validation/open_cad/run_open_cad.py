"""ES-FA open-source CAD flow: simulate, synthesize, place-and-route, report.

Runs the complete open-toolchain evidence flow for the v1 synthesizable RTL
core without any vendor tools:

1. Icarus Verilog compiles the four RTL modules plus the self-checking
   testbench and runs the simulation; the testbench reports PASS/FAIL.
2. Yosys synthesizes ``esfa_top_core`` for the Lattice ECP5 fabric and
   reports cell statistics.
3. nextpnr-ecp5 places and routes the netlist and reports the routed
   maximum clock frequency estimate.

Artifacts (all under ``out/`` by default): ``esfa_tb.vvp`` (compiled
simulation), ``esfa_ecp5.json`` (synthesized netlist), ``esfa_ecp5.config``
(routed bitstream configuration), and ``open_cad_report.json`` (parsed
results with tool versions).

The flow expects the OSS CAD Suite; set ``ESFA_OSS_CAD_ROOT`` or pass
``--oss-cad-root`` if it is not at ``C:\\oss-cad-suite``.

Usage:
    py -3 hardware_validation/open_cad/run_open_cad.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RTL_DIR = PROJECT_ROOT / "implementations" / "v1_synthesizable_rtl_verilog"
RTL_MODULES = (
    "lif_pe_core.v",
    "bram_bank_arbiter.v",
    "stdp_weight_updater.v",
    "esfa_top_core.v",
)
TOP = "esfa_top_core"

HARDWARE_DIR = PROJECT_ROOT / "hardware"
SUPPORTING_MODULES = (
    "compute/adaptive_leak_engine.v",
    "compute/lif_neuron_pe.v",
    "compute/spike_driven_flash_attention.v",
    "compute/stdp_learning_engine.v",
    "memory/neuron_bram.v",
    "memory/weight_bram_bank.v",
    "routing/spike_router.v",
    "scheduler/basic_scheduler.v",
    "scheduler/event_queue.v",
    "scheduler/advanced_scheduler.v",
    "top/snn_top.v",
    "top/snn_accelerator_generic.v",
)
TESTBENCH_DIR = HARDWARE_DIR / "tb"


def _tool_env(oss_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    bin_dir = oss_root / "bin"
    lib_dir = oss_root / "lib"
    if not bin_dir.exists():
        raise SystemExit(f"OSS CAD Suite not found at {oss_root}; set --oss-cad-root")
    env["PATH"] = os.pathsep.join([str(bin_dir), str(lib_dir), env.get("PATH", "")])
    return env


def _resolve(name: str, env: dict[str, str]) -> str:
    resolved = shutil.which(name, path=env["PATH"])
    if resolved is None:
        raise SystemExit(f"{name} not found in the OSS CAD Suite")
    return resolved


def _run(command: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, env=env, cwd=str(cwd))
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(command)}")
    return result


def _parse_yosys_stat(output: str) -> tuple[dict[str, int], int]:
    marker = f"=== {TOP} ==="
    sections = output.split(marker)
    if len(sections) < 2:
        return {}, 0
    cells: dict[str, int] = {}
    total = 0
    in_cells = False
    for line in sections[-1].splitlines():
        stripped = line.strip()
        if stripped.startswith("=== ") or stripped.startswith("End of script"):
            break
        if not in_cells:
            total_match = re.match(r"^(\d+)\s+cells$", stripped)
            if total_match:
                total = int(total_match.group(1))
                in_cells = True
            continue
        if not stripped:
            break
        cell_match = re.match(r"^(\d+)\s+(\S+)$", stripped)
        if cell_match and not cell_match.group(2).startswith("$"):
            cells[cell_match.group(2)] = int(cell_match.group(1))
    return cells, total


def _parse_fmax(output: str) -> float | None:
    matches = re.findall(r"Max frequency for clock '[^']+': ([0-9.]+) MHz", output)
    if not matches:
        return None
    return min(float(value) for value in matches)


def main() -> int:
    parser = argparse.ArgumentParser(description="ES-FA open-source CAD flow")
    parser.add_argument(
        "--oss-cad-root",
        default=os.environ.get("ESFA_OSS_CAD_ROOT", r"C:\oss-cad-suite"),
    )
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "out"))
    parser.add_argument("--freq", type=float, default=50.0)
    args = parser.parse_args()

    oss_root = Path(args.oss_cad_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _tool_env(oss_root)

    rtl_paths = [str(RTL_DIR / name) for name in RTL_MODULES]

    # 1. Simulation
    vvp_path = out_dir / "esfa_tb.vvp"
    iverilog = _resolve("iverilog", env)
    vvp = _resolve("vvp", env)
    yosys = _resolve("yosys", env)
    nextpnr = _resolve("nextpnr-ecp5", env)
    _run(
        [iverilog, "-g2012", "-o", str(vvp_path), *rtl_paths, str(RTL_DIR / "tb_esfa_rtl.v")],
        env,
        PROJECT_ROOT,
    )
    simulation = _run([vvp, str(vvp_path)], env, PROJECT_ROOT)
    simulation_pass = "PASS:" in simulation.stdout
    spike_line = re.search(r"Total spikes fired\s*:\s*(\d+)", simulation.stdout)
    stdp_line = re.search(r"STDP weight updates\s*:\s*(\d+)", simulation.stdout)
    cycles_line = re.search(r"Total active cycles\s*:\s*(\d+)", simulation.stdout)

    # 1b. Supporting unit testbenches under hardware/ (self-checking).
    supporting_paths = [str(HARDWARE_DIR / name) for name in SUPPORTING_MODULES]
    testbench_results: list[dict[str, object]] = []
    for tb in sorted(TESTBENCH_DIR.glob("tb_*.v")):
        tb_exe = out_dir / f"{tb.stem}.vvp"
        _run([iverilog, "-g2012", "-o", str(tb_exe), *supporting_paths, str(tb)], env, PROJECT_ROOT)
        tb_run = _run([vvp, str(tb_exe)], env, PROJECT_ROOT)
        testbench_results.append(
            {
                "testbench": tb.name,
                "pass": "PASS:" in (tb_run.stdout + tb_run.stderr),
            }
        )
    testbenches_passed = sum(1 for entry in testbench_results if entry["pass"])

    # 2. Synthesis
    read_cmd = f"read_verilog {' '.join(Path(p).as_posix() for p in rtl_paths)}"
    synth_json = out_dir / "esfa_ecp5.json"
    _run(
        [yosys, "-q", "-p", f"{read_cmd}; synth_ecp5 -top {TOP} -json {synth_json.as_posix()}"],
        env,
        PROJECT_ROOT,
    )
    synth_stat = _run([yosys, "-p", f"{read_cmd}; synth_ecp5 -top {TOP}; stat"], env, PROJECT_ROOT)
    cells, total_cells = _parse_yosys_stat(synth_stat.stdout)

    # 3. Place and route
    config_path = out_dir / "esfa_ecp5.config"
    pnr = _run(
        [
            nextpnr,
            "--json",
            str(synth_json),
            "--textcfg",
            str(config_path),
            "--25k",
            "--package",
            "CABGA381",
            "--speed",
            "6",
            "--freq",
            str(args.freq),
            "--seed",
            "1",
            "--lpf-allow-unconstrained",
        ],
        env,
        PROJECT_ROOT,
    )
    fmax = _parse_fmax(pnr.stdout + pnr.stderr)

    yosys_version = _run([yosys, "-V"], env, PROJECT_ROOT).stdout.strip()
    nextpnr_version_run = _run([nextpnr, "--version"], env, PROJECT_ROOT)
    nextpnr_lines = (nextpnr_version_run.stdout + nextpnr_version_run.stderr).strip().splitlines()
    nextpnr_version = nextpnr_lines[0] if nextpnr_lines else ""
    iverilog_version = _run([iverilog, "-V"], env, PROJECT_ROOT).stdout.splitlines()[0].strip()

    report = {
        "flow": "esfa-open-cad",
        "top": TOP,
        "target": "Lattice ECP5 25k (CABGA381, speed grade 6)",
        "freq_target_mhz": args.freq,
        "simulation": {
            "testbench": "tb_esfa_rtl.v",
            "pass": simulation_pass,
            "spikes_fired": int(spike_line.group(1)) if spike_line else None,
            "active_cycles": int(cycles_line.group(1)) if cycles_line else None,
            "stdp_weight_updates": int(stdp_line.group(1)) if stdp_line else None,
        },
        "supporting_testbenches": {
            "total": len(testbench_results),
            "passed": testbenches_passed,
            "results": testbench_results,
        },
        "synthesis": {"total_cells": total_cells, "cells": cells},
        "place_and_route": {
            "fmax_mhz": fmax,
            "passed_target": fmax is not None and fmax >= args.freq,
        },
        "tools": {
            "iverilog": iverilog_version,
            "yosys": yosys_version,
            "nextpnr": nextpnr_version,
        },
        "boundary": (
            "Open-source toolchain evidence only: behavioral simulation, generic "
            "synthesis statistics, and a routed timing estimate. No vendor timing "
            "signoff, no board measurement, no power measurement."
        ),
    }

    report_path = out_dir / "open_cad_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print("ES-FA open-source CAD flow")
    print(f"  simulation : {'PASS' if simulation_pass else 'FAIL'} "
          f"({report['simulation']['spikes_fired']} spikes, "
          f"{report['simulation']['stdp_weight_updates']} STDP updates)")
    print(f"  testbenches: {testbenches_passed}/{len(testbench_results)} PASS")
    print(f"  synthesis  : {total_cells} cells "
          f"({cells.get('LUT4', 0)} LUT4, {cells.get('TRELLIS_FF', 0)} FF)")
    print(f"  place/route: Fmax estimate {fmax} MHz at a {args.freq} MHz target "
          f"({'PASS' if report['place_and_route']['passed_target'] else 'FAIL'})")
    print(f"  report     : {report_path}")
    if not simulation_pass or testbenches_passed != len(testbench_results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
