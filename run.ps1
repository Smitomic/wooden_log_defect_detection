# Auto-detect CUDA and launch the appropriate Docker profile
#
# Usage:
#   .\run.ps1          # auto-detect
#   .\run.ps1 cpu      # force CPU
#   .\run.ps1 gpu      # force GPU
#   .\run.ps1 build    # auto-detect + force rebuild

param(
    [string]$Force = "",
    [switch]$Build
)

$Rebuild = if ($Build -or $Force -eq "build") { "--build" } else { "" }
if ($Force -eq "build") { $Force = "" }

# Detect CUDA
if (-not $Force) {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $gpuName = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null |
                   Select-Object -First 1
        if ($LASTEXITCODE -eq 0 -and $gpuName) {
            Write-Host "NVIDIA GPU detected: $gpuName" -ForegroundColor Green
            $Profile = "gpu"
        } else {
            Write-Host "nvidia-smi found but no GPU available — using CPU build." -ForegroundColor Yellow
            $Profile = "cpu"
        }
    } else {
        Write-Host "No NVIDIA GPU detected — using CPU build." -ForegroundColor Yellow
        $Profile = "cpu"
    }
} else {
    $Profile = $Force
    Write-Host "Profile forced: $Profile" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Starting wood-defect:$Profile on http://localhost:8000" -ForegroundColor Cyan
Write-Host ""

$cmd = "docker compose --profile $Profile up $Rebuild".Trim()
Invoke-Expression $cmd
