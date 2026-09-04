"""Resolve and invoke Xilinx tools across inconsistent shell environments.

This module allows the project to run in environments where Vivado/Vitis/xsim
are installed but not always available on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional


def _candidate_bins() -> List[Path]:
    candidates: List[Path] = []

    env_keys = [
        "XILINX_VIVADO",
        "XILINX_VITIS",
        "XILINX_HOME",
        "XILINX_INSTALL",
        "VIVADO_HOME",
        "VITIS_HOME",
    ]
    for key in env_keys:
        val = os.environ.get(key, "").strip()
        if val:
            p = Path(val)
            candidates.append(p / "bin")
            candidates.append(p)

    roots = [
        Path(r"C:\Xilinx"),
        Path(r"C:\AMDDesignTools"),
        Path(r"C:\Program Files\Xilinx"),
        Path(r"D:\Xilinx"),
        Path(r"E:\Xilinx"),
        Path.home() / "Xilinx",
    ]
    for root in roots:
        if not root.exists():
            continue
        # Common tool layouts.
        for pattern in (
            "*/Vivado/bin",
            "*/Vitis/bin",
            "*/Vitis_HLS/bin",
            "*/bin",
            "*/*/bin",
        ):
            for p in root.glob(pattern):
                candidates.append(p)

    seen = set()
    dedup: List[Path] = []
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
    return dedup


def resolve_executable(name: str) -> Optional[str]:
    direct = shutil.which(name)
    if direct:
        return direct

    if os.name == "nt":
        exts = [".bat", ".cmd", ".exe", ""]
    else:
        exts = ["", ".exe", ".bat", ".cmd"]
    for bin_dir in _candidate_bins():
        for ext in exts:
            p = bin_dir / f"{name}{ext}"
            if p.exists():
                return str(p)
    return None


def ensure_tools_available(names: Iterable[str]) -> None:
    missing = [n for n in names if resolve_executable(n) is None]
    if missing:
        raise FileNotFoundError(
            "Missing required Xilinx tools: "
            + ", ".join(missing)
            + ". Set PATH or XILINX_* environment variables."
        )


def run_tool(
    name: str,
    args: List[str],
    cwd: Path,
    log_path: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    exe = resolve_executable(name)
    if exe is None:
        raise FileNotFoundError(
            f"Could not resolve tool `{name}`. Set PATH or XILINX_* environment variables."
        )

    cmd = [exe] + args
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"$ {' '.join(cmd)}\n\n")
            f.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                text=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
    else:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
        )

    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Tool `{name}` failed with code {proc.returncode}. "
            f"See log: {log_path if log_path else 'captured output'}"
        )
    return proc
