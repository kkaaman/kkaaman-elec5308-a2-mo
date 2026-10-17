param([switch]$CpuOnly)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

& py -3.12 --version
if ($LASTEXITCODE -ne 0) { throw "Install Python 3.12 before continuing." }
if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}
$projectPython = Join-Path $PWD ".venv\Scripts\python.exe"
& $projectPython -c "import sys; assert sys.version_info[:2] == (3,12), 'This venv is not Python 3.12'"
if ($LASTEXITCODE -ne 0) { throw "Existing environment uses another Python version. Keep it and create a separate Python 3.12 environment." }
& $projectPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
$torchIndex = if ($CpuOnly) { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/cu128" }
& $projectPython -m pip install --upgrade torch==2.10.0 torchvision==0.25.0 --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
& $projectPython -m pip install "ultralytics==8.4.177" -e ".[test,docs,carla]"
if ($LASTEXITCODE -ne 0) { throw "Project installation failed." }
& $projectPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency consistency check failed." }
if (-not $CpuOnly) {
    & $projectPython -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; x=torch.rand(512,512,device='cuda'); y=x@x; torch.cuda.synchronize(); print('CUDA computation: PASS'); print('GPU:',torch.cuda.get_device_name(0)); print('Torch:',torch.__version__)"
    if ($LASTEXITCODE -ne 0) { throw "GPU computation failed. Save this output before attempting performance tests." }
}
Write-Host "Environment ready. Next: & .\scripts\run_batch35.ps1 -Stage Offline"
