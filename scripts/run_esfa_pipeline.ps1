Param(
    [string]$PythonExe = "python",
    [ValidateSet("smoke","full")]
    [string]$Mode = "smoke",
    [double]$ClockMHz = 100.0,
    [switch]$SkipVivado,
    [switch]$SkipHardware
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

. (Join-Path $ProjectRoot "scripts\toolchain\bootstrap_xilinx.ps1")

$argsList = @(
    "scripts/run_esfa_pipeline.py",
    "--mode", $Mode,
    "--clock-mhz", "$ClockMHz"
)
if ($SkipVivado) { $argsList += "--skip-vivado" }
if ($SkipHardware) { $argsList += "--skip-hardware" }

Push-Location $ProjectRoot
try {
    & $PythonExe @argsList
}
finally {
    Pop-Location
}
