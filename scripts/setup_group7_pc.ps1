<#
.SYNOPSIS
  Set up a Windows PC for the Group 7 mapless work. No admin rights needed; run it on
  each PC.

.DESCRIPTION
  1. Machine check: GPU, free disk, Python 3.12, git and git-lfs.
  2. Finds CARLA 0.9.16 (CarlaUE4.exe) and its Python 3.12 wheel. If CARLA is missing,
     it says so and carries on.
  3. Clones the team repo (remote "origin") if it isn't there already, adds the
     private repo (remote "private"), checks out feature/mapless-autonomy and pulls
     the LFS weights.
  4. Creates .venv with Python 3.12, then runs setup_environment.py:
     - the project (editable, test and docs extras);
     - torch with CUDA;
     - the CARLA wheel.
  5. Checks:
     - torch sees the GPU;
     - carla imports;
     - the mapless unit tests pass.
  6. Writes setup_report.txt in the repo and prints the next commands.

  Safe to re-run: steps that are already done are skipped.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File "$HOME\Downloads\setup_group7_pc.ps1"
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\setup_group7_pc.ps1 -CarlaRoot "C:\Users\me\CARLA_0.9.16"
#>
param(
    [string]$Root = "$HOME\Desktop",
    [string]$CarlaRoot = "",
    [string]$CudaTag = "cu128",
    [switch]$SkipTests
)
# Native tools (git, py) write progress to stderr; check exit codes instead of stopping on it.
$ErrorActionPreference = "Continue"
$TeamUrl = "https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7.git"
$PrivateUrl = "https://github.com/kkaaman/kkaaman-elec5308-a2-mo.git"
$Branch = "feature/mapless-autonomy"
$Repo = Join-Path $Root "26-S2-Assignment2-Group7"
$Report = New-Object System.Collections.Generic.List[string]

function Step([string]$Text) { Write-Host "`n== $Text ==" -ForegroundColor Cyan }
function Note([string]$Text) { Write-Host $Text; $Report.Add($Text) }
function Problem([string]$Text) { Write-Host "PROBLEM: $Text" -ForegroundColor Yellow; $Report.Add("PROBLEM: $Text") }
function Have([string]$Command) { return [bool](Get-Command $Command -ErrorAction SilentlyContinue) }

# 1. Machine ---------------------------------------------------------------------
Step "1. Machine"
Note "Computer: $env:COMPUTERNAME  User: $env:USERNAME  Date: $(Get-Date -Format s)"
if (Have "nvidia-smi") {
    $gpu = nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    Note "GPU: $gpu"
} else {
    Problem "nvidia-smi not found: no NVIDIA driver on PATH. CARLA and CUDA torch need one."
}
$drive = Get-PSDrive -Name ($Root.Substring(0, 1))
Note ("Free space on {0}: {1:N0} GB" -f $drive.Name, ($drive.Free / 1GB))
if ($drive.Free -lt 40GB) { Problem "Less than 40 GB free. CARLA needs about 20 GB, and captures need more." }
if (-not (Have "py")) { throw "The Python launcher 'py' is missing. Install Python 3.12 (user install, no admin) and re-run." }
$pyVersion = & py -3.12 --version 2>&1
if ($LASTEXITCODE -ne 0) { throw "Python 3.12 not found (py -3.12). Install it for your user and re-run." }
Note "Python: $pyVersion"
if (-not (Have "git")) { throw "git is missing. Install Git for Windows (user install) and re-run." }
Note "git: $(git --version)"
$lfs = git lfs version 2>&1
$HasLfs = ($LASTEXITCODE -eq 0)
if ($HasLfs) { Note "git-lfs: $lfs" } else { Problem "git-lfs missing: the .pth weights will stay as pointer files." }

# 2. CARLA ------------------------------------------------------------------------
Step "2. CARLA 0.9.16"
if (-not $CarlaRoot) {
    $places = @($HOME, "$HOME\Documents", "$HOME\Desktop", "$HOME\Downloads", "C:\CARLA", "D:\") | Where-Object { Test-Path -LiteralPath $_ }
    $found = $null
    foreach ($place in $places) {
        $found = Get-ChildItem -LiteralPath $place -Filter "CarlaUE4.exe" -Recurse -Depth 3 -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { break }
    }
    if ($found) { $CarlaRoot = $found.DirectoryName }
}
$Wheel = $null
if ($CarlaRoot -and (Test-Path -LiteralPath (Join-Path $CarlaRoot "CarlaUE4.exe"))) {
    Note "CARLA: $CarlaRoot"
    $Wheel = Get-ChildItem -LiteralPath (Join-Path $CarlaRoot "PythonAPI\carla\dist") -Filter "carla-0.9.16-cp312-*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($Wheel) { Note "CARLA wheel: $($Wheel.FullName)" } else { Problem "No carla-0.9.16 cp312 wheel under PythonAPI\carla\dist." }
} else {
    Problem "CARLA 0.9.16 not found. Download the Windows package from the CARLA 0.9.16 GitHub release, extract it into your user folder, then re-run with -CarlaRoot <folder>."
}

# 3. Repository ---------------------------------------------------------------------
Step "3. Repository"
if (-not (Test-Path -LiteralPath (Join-Path $Repo ".git"))) {
    Note "Cloning team repo into $Repo (a browser sign-in may appear)"
    git clone $TeamUrl $Repo
    if ($LASTEXITCODE -ne 0) { throw "Clone failed. Sign in to GitHub as kkaaman and re-run." }
} else {
    Note "Repo exists: $Repo"
}
Push-Location $Repo
try {
    git config user.name "kkaaman"
    git config user.email "judehcrabs12@gmail.com"
    $remotes = git remote
    if ($remotes -notcontains "private") { git remote add private $PrivateUrl; Note "Added remote 'private'" }
    git fetch origin --quiet
    git fetch private --quiet
    if ($LASTEXITCODE -ne 0) { throw "Fetch from the private repo failed. Check the GitHub sign-in." }
    $dirty = git status --porcelain
    $current = git branch --show-current
    if ($current -ne $Branch) {
        if ($dirty) { throw "Uncommitted changes on $current; commit or stash them before switching to $Branch." }
        $local = git branch --list $Branch
        if ($local) { git switch $Branch } else { git switch -c $Branch --track "private/$Branch" }
    }
    $behind = git rev-list --count "HEAD..private/$Branch"
    if ([int]$behind -gt 0) {
        if ($dirty) { Problem "Branch is $behind commit(s) behind private/$Branch, but has local changes; not pulling." }
        else { git merge --ff-only "private/$Branch"; Note "Fast-forwarded $behind commit(s)" }
    }
    Note "Branch: $(git branch --show-current) at $(git log -1 --format='%h %s')"
    if ($HasLfs) {
        git lfs install --local | Out-Null
        git lfs pull private
        if ($LASTEXITCODE -ne 0) { git lfs pull origin }
        $pointers = git lfs ls-files | Where-Object { $_ -match " - " }
        if ($pointers) { Problem "Some LFS files are still pointers: $($pointers -join '; ')" } else { Note "LFS weights present" }
    }

    # 4. Python environment ------------------------------------------------------------
    Step "4. Python environment (.venv, Python 3.12)"
    $python = Join-Path $Repo ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) { py -3.12 -m venv .venv; Note "Created .venv" }
    & $python -m pip install --upgrade pip --quiet
    $setupArgs = @("setup_environment.py", "--cuda-tag", $CudaTag, "--skip-lfs")
    if ($Wheel) { $setupArgs += @("--carla-wheel", $Wheel.FullName) } else { $setupArgs += "--skip-carla" }
    & $python @setupArgs
    if ($LASTEXITCODE -ne 0) { Problem "setup_environment.py reported a failure; see the output above." }

    # 5. Checks --------------------------------------------------------------------------
    Step "5. Checks"
    $torch = & $python -c "import torch; print(torch.__version__, 'CUDA' if torch.cuda.is_available() else 'NO-CUDA', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1
    Note "torch: $torch"
    if ($torch -match "NO-CUDA") { Problem "torch cannot see the GPU. Re-run with -CudaTag cu126 or cu130 (see pytorch.org)." }
    $carlaPkg = & $python -c "import carla, importlib.metadata as m; print(m.version('carla'))" 2>&1
    if ($LASTEXITCODE -eq 0) { Note "carla client: $carlaPkg" } else { Problem "carla does not import in .venv" }
    if (-not $SkipTests) {
        & $python -m pytest -q tests/unit/test_mapless_sensor_suite.py tests/unit/test_mapless_frame_geometry.py tests/unit/test_satnav.py tests/unit/test_mapless_firewall.py tests/unit/test_sensor_rig.py tests/unit/test_mapless_capture_check.py tests/unit/test_sensor_rig_check_smoke.py tests/unit/test_rig_calibration.py tests/unit/test_local_controller.py
        if ($LASTEXITCODE -eq 0) { Note "Mapless unit tests: PASS" } else { Problem "Mapless unit tests failed" }
    }
} finally {
    Pop-Location
}

# 6. Summary --------------------------------------------------------------------------
Step "6. Summary"
$problems = $Report | Where-Object { $_ -like "PROBLEM:*" }
$Report | Set-Content -LiteralPath (Join-Path $Repo "setup_report.txt") -Encoding UTF8
if ($problems) { Write-Host "$($problems.Count) problem(s); see above and setup_report.txt" -ForegroundColor Yellow }
else { Write-Host "Setup complete, no problems." -ForegroundColor Green }
$carlaExe = if ($CarlaRoot) { Join-Path $CarlaRoot "CarlaUE4.exe" } else { "<CARLA folder>\CarlaUE4.exe" }
Write-Host @"

Next:
  1. Start CARLA:   & "$carlaExe"     (wait for the city view)
  2. In the repo:   cd "$Repo"; .\.venv\Scripts\Activate.ps1
  3. Rig check:     python scripts\check_sensor_rig.py --output "$HOME\Documents\Group7_runs\rigcheck_v1_$(Get-Date -Format yyyyMMdd_HHmm)"
"@
