Param(
    [string]$IcarusExe = "iverilog",
    [string]$VvpExe = "vvp",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$HardwareRoot = Join-Path $ProjectRoot "hardware"

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $HardwareRoot "sim_out"
}
New-Item -ItemType Directory -Force $OutDir | Out-Null

$rtl = @(
    "memory/neuron_bram.v",
    "memory/weight_bram_bank.v",
    "compute/lif_neuron_pe.v",
    "routing/spike_router.v",
    "scheduler/basic_scheduler.v",
    "scheduler/event_queue.v",
    "scheduler/advanced_scheduler.v",
    "top/snn_top.v"
)

$tbList = @(
    "tb_neuron_bram",
    "tb_weight_bram_bank",
    "tb_lif_neuron_pe",
    "tb_spike_router",
    "tb_basic_scheduler",
    "tb_event_queue",
    "tb_advanced_scheduler",
    "tb_top"
)

Push-Location $HardwareRoot
try {
    foreach ($tb in $tbList) {
        $tbFile = "tb/$tb.v"
        $simOut = Join-Path $OutDir "$tb.out"
        Write-Host "Compiling $tbFile"

        & $IcarusExe -g2012 -o $simOut @rtl $tbFile
        Write-Host "Running $tb"
        & $VvpExe $simOut
    }
    Write-Host "All RTL tests completed."
}
finally {
    Pop-Location
}
