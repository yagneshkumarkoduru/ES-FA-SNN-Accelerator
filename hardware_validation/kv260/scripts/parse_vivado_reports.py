"""Parse Vivado synthesis/implementation reports into normalized metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _parse_first_number(line: str) -> Optional[float]:
    m = re.search(r"[-+]?\d+(\.\d+)?", line.replace(",", ""))
    if not m:
        return None
    return float(m.group(0))


def _parse_utilization(report_text: str) -> Dict[str, Optional[float]]:
    lines = report_text.splitlines()
    out: Dict[str, Optional[float]] = {
        "lut": None,
        "bram": None,
        "dsp": None,
        "ff": None,
    }
    for line in lines:
        norm = line.strip()
        if not norm:
            continue
        if "CLB LUTs" in norm and out["lut"] is None:
            out["lut"] = _parse_first_number(norm)
        elif ("CLB Registers" in norm or "Slice Registers" in norm) and out["ff"] is None:
            out["ff"] = _parse_first_number(norm)
        elif ("Block RAM Tile" in norm or "RAMB36" in norm) and out["bram"] is None:
            out["bram"] = _parse_first_number(norm)
        elif "DSPs" in norm and out["dsp"] is None:
            out["dsp"] = _parse_first_number(norm)
    return out


def _parse_timing(report_text: str, clock_mhz_target: float) -> Dict[str, Optional[float]]:
    lines = report_text.splitlines()
    wns = None
    tns = None
    fmax_mhz = None

    patterns = [
        r"WNS\(ns\)\s+TNS\(ns\)\s+TNS Failing Endpoints",
        r"Design Timing Summary",
    ]
    if any(re.search(p, report_text) for p in patterns):
        for idx, line in enumerate(lines):
            if "WNS(ns)" in line and "TNS(ns)" in line:
                # Try next non-empty line as values row.
                for nxt in lines[idx + 1 : idx + 6]:
                    if nxt.strip():
                        nums = re.findall(r"[-+]?\d+(\.\d+)?", nxt.replace(",", ""))
                        if nums:
                            vals = re.findall(r"[-+]?\d+(?:\.\d+)?", nxt.replace(",", ""))
                            if len(vals) >= 2:
                                wns = float(vals[0])
                                tns = float(vals[1])
                            break
                break

    # Fallback parse for explicit slack lines.
    if wns is None:
        m = re.search(r"Slack\s*\(WNS\)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", report_text)
        if m:
            wns = float(m.group(1))
    if tns is None:
        m = re.search(r"TNS\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", report_text)
        if m:
            tns = float(m.group(1))

    target_period_ns = 1000.0 / max(clock_mhz_target, 1e-9)
    if wns is not None:
        achieved_period_ns = max(1e-9, target_period_ns - wns)
        fmax_mhz = 1000.0 / achieved_period_ns

    return {
        "wns_ns": wns,
        "tns_ns": tns,
        "target_clock_mhz": clock_mhz_target,
        "fmax_mhz_estimated": fmax_mhz,
    }


def parse_reports(report_dir: Path, clock_mhz_target: float) -> Dict[str, object]:
    synth_util = _parse_utilization(_read(report_dir / "synth_utilization.rpt"))
    impl_util = _parse_utilization(_read(report_dir / "impl_utilization.rpt"))
    impl_timing = _parse_timing(_read(report_dir / "impl_timing_summary.rpt"), clock_mhz_target)
    synth_timing = _parse_timing(_read(report_dir / "synth_timing_summary.rpt"), clock_mhz_target)

    return {
        "synthesis": {
            "utilization": synth_util,
            "timing": synth_timing,
        },
        "implementation": {
            "utilization": impl_util,
            "timing": impl_timing,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse Vivado reports to JSON.")
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--clock-mhz", type=float, default=200.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    parsed = parse_reports(args.report_dir, args.clock_mhz)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    print(f"Parsed Vivado reports -> {args.out}")


if __name__ == "__main__":
    main()

