Param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

Write-Host "Project root: $ProjectRoot"
Push-Location $ProjectRoot
try {
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install -r requirements.txt
    Write-Host "Environment setup complete."
}
finally {
    Pop-Location
}
