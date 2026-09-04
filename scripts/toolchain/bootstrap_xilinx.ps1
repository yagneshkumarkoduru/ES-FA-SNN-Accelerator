Param()

$ErrorActionPreference = "Stop"

if (-not $env:PATH) { $env:PATH = "" }

$candidateBins = @()
foreach ($k in @("XILINX_VIVADO","XILINX_VITIS","XILINX_HOME","XILINX_INSTALL","VIVADO_HOME","VITIS_HOME")) {
    $val = (Get-Item -Path ("Env:" + $k) -ErrorAction SilentlyContinue).Value
    if ($val) {
        $candidateBins += (Join-Path $val "bin")
        $candidateBins += $val
    }
}

$roots = @("C:\AMDDesignTools", "C:\Xilinx", "D:\Xilinx", "E:\Xilinx", "$HOME\Xilinx")
foreach ($r in $roots) {
    if (Test-Path $r) {
        Get-ChildItem -Path $r -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            if (Test-Path (Join-Path $_.FullName "Vivado\\bin")) { $candidateBins += (Join-Path $_.FullName "Vivado\\bin") }
            if (Test-Path (Join-Path $_.FullName "Vitis\\bin")) { $candidateBins += (Join-Path $_.FullName "Vitis\\bin") }
            if (Test-Path (Join-Path $_.FullName "Vitis_HLS\\bin")) { $candidateBins += (Join-Path $_.FullName "Vitis_HLS\\bin") }
        }
        Get-ChildItem -Path $r -Recurse -Directory -Filter bin -ErrorAction SilentlyContinue | ForEach-Object {
            $candidateBins += $_.FullName
        }
    }
}

$unique = $candidateBins | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique
foreach ($b in $unique) {
    if ($env:PATH -notlike "*$b*") {
        $env:PATH = "$b;$env:PATH"
    }
}

Write-Host "Xilinx PATH bootstrap complete."
Write-Host "vivado: $(Get-Command vivado -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)"
Write-Host "xvlog : $(Get-Command xvlog -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)"
