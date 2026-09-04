Param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

Push-Location $ProjectRoot
try {
    & $PythonExe scripts/run_research_pipeline.py
}
finally {
    Pop-Location
}
