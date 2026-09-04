Param(
    [string]$PythonExe = "python",
    [int]$Epochs = 5,
    [int]$BatchSize = 64,
    [int]$TimeSteps = 16,
    [string]$LayerToMap = "fc1"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

$ArtifactsDir = Join-Path $ProjectRoot "software/artifacts"
$ExportDir = Join-Path $ArtifactsDir "export_int8"
$MapDir = Join-Path $ArtifactsDir "fpga_mem"
$BestCkpt = Join-Path $ArtifactsDir "checkpoints/best_model.pt"

Push-Location $ProjectRoot
try {
    Write-Host "Running SNN training..."
    & $PythonExe software/train_snn.py `
        --epochs $Epochs `
        --batch-size $BatchSize `
        --time-steps $TimeSteps `
        --output-dir $ArtifactsDir

    Write-Host "Exporting INT8 weights..."
    & $PythonExe software/export_weights.py `
        --checkpoint $BestCkpt `
        --output-dir $ExportDir

    Write-Host "Mapping exported weights to banked FPGA memory files..."
    & $PythonExe software/map_weights_to_banks.py `
        --export-dir $ExportDir `
        --layer $LayerToMap `
        --num-banks 2 `
        --bank-depth 64 `
        --out-dir $MapDir

    Write-Host "Software pipeline complete."
    Write-Host "Artifacts: $ArtifactsDir"
}
finally {
    Pop-Location
}
