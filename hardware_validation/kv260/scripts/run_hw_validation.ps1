Param(
    [string]$PythonExe = "python",
    [string]$ModelId = "baseline_paper1",
    [string]$SchedulerMode = "both",
    [double]$ClockMHz = 100.0,
    [switch]$SkipVivado
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")

. (Join-Path $ProjectRoot "scripts\toolchain\bootstrap_xilinx.ps1")

$argsList = @(
    "hardware_validation/kv260/scripts/run_hw_validation.py",
    "--model-id", $ModelId,
    "--scheduler-mode", $SchedulerMode,
    "--clock-mhz", "$ClockMHz"
)
if ($SkipVivado) { $argsList += "--skip-vivado" }

Push-Location $ProjectRoot
try {
    & $PythonExe @argsList
}
finally {
    Pop-Location
}

