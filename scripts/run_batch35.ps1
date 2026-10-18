param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("Offline","LiveSetup","Quick","Verify","Capture","Perception","Tune","Navigation","Freeze","Smoke","Acceptance","ReplayFaults","Train","Evaluate","Audit","Archive")]
    [string]$Stage,
    [int]$Samples = 12000,
    [double]$TrainingHours = 4,
    [string]$CarlaExe = "",
    [string]$StateFile = "",
    # Capture: town to record. The demo map (default) is the measurement dataset;
    # another town (e.g. Town05) is stored separately as the training dataset.
    [string]$Map = "Town10HD_Opt",
    # Train: explicitly allow fine-tuning on the demo map's own images.
    [switch]$AllowDemoMap,
    # Perception: detector input size/confidence to measure; Freeze copies them
    # into the candidate config so Tune/Navigation/Acceptance run with them.
    [int]$DetectorImageSize = 640,
    [double]$DetectorConfidence = 0.25
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$repoPath = $PWD.Path
$projectPython = Join-Path $repoPath ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $projectPython)) { throw "Run scripts/setup_batch35.ps1 first." }
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$downloadsPath = Join-Path $HOME "Downloads"
if (-not $StateFile) { $StateFile = Join-Path $downloadsPath "Group7_batch35_state.json" }
$batchState = @{}
if (Test-Path -LiteralPath $StateFile) {
    $savedState = Get-Content -LiteralPath $StateFile -Raw | ConvertFrom-Json
    foreach ($property in $savedState.PSObject.Properties) { $batchState[$property.Name] = $property.Value }
    if ($batchState["repo"] -ne $repoPath) { throw "This state file belongs to another checkout. Supply a different -StateFile in Downloads." }
}
$batchState["repo"] = $repoPath
$stageFolder = Join-Path $downloadsPath ("Group7_batch35_" + $Stage + "_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff"))
New-Item -ItemType Directory -Path $stageFolder -ErrorAction Stop | Out-Null
$stageLog = Join-Path $stageFolder "terminal.txt"
$batchState["last_stage_output"] = $stageFolder
$batchState["last_stage"] = $Stage

function Invoke-ProjectPython {
    param([string[]]$PythonArgs)
    $previousErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $projectPython @PythonArgs 2>&1 | Tee-Object -FilePath $stageLog -Append
        $pythonExitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousErrorPreference }
    if ($pythonExitCode -ne 0) { throw "Command stopped (exit $pythonExitCode). Evidence is in $stageFolder. Read the last error before retrying." }
}
function Need-Result {
    param([string]$Name)
    if (-not $batchState.ContainsKey($Name) -or -not (Test-Path -LiteralPath $batchState[$Name])) {
        throw "Missing '$Name' in saved state. Run its preceding stage first."
    }
    return [string]$batchState[$Name]
}
function Save-BatchState {
    $batchState | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $StateFile -Encoding utf8
}
function Get-LiveArgs {
    param([switch]$RequireEvents)
    $missionPath = Need-Result "mission"
    $depthPath = Need-Result "depth"
    $result = @("--engine","carla","--mission",$missionPath,"--depth-calibration",$depthPath,"--follow")
    # Collision/lane metrics are only scored after the live BL-B-07 Verify stage.
    if ($RequireEvents) {
        $result += @("--event-verification",(Need-Result "event_verification"))
    } elseif ($batchState.ContainsKey("event_verification") -and (Test-Path -LiteralPath $batchState["event_verification"])) {
        $result += @("--event-verification",[string]$batchState["event_verification"])
    }
    return $result
}

try {
    switch ($Stage) {
        "Offline" {
            Invoke-ProjectPython -PythonArgs @("-m","ruff","check",".")
            Invoke-ProjectPython -PythonArgs @("-m","pytest","-m","not carla","-q")
            Invoke-ProjectPython -PythonArgs @("scripts/check_simulator_boundary.py","--smoke")
            Invoke-ProjectPython -PythonArgs @("-m","sphinx","-W","-b","html","docs",(Join-Path $stageFolder "docs"))
            Invoke-ProjectPython -PythonArgs @("scripts/run_fault_suite.py","--output",(Join-Path $stageFolder "faults"))
            $batchState["offline"] = $stageFolder
        }
        "LiveSetup" {
            if ($CarlaExe) {
                if (-not (Test-Path -LiteralPath $CarlaExe)) { throw "CARLA executable not found." }
                if (-not (Get-Process "CarlaUE4-Win64-Shipping" -ErrorAction SilentlyContinue)) {
                    Start-Process -FilePath $CarlaExe -ArgumentList "-quality-level=Low","-carla-rpc-port=2000","-windowed","-ResX=1280","-ResY=720"
                    Write-Host "Wait for the CARLA map to appear, then rerun LiveSetup."
                    return
                }
            }
            Invoke-ProjectPython -PythonArgs @("scripts/prepare_mission.py","--output",(Join-Path $stageFolder "mission"))
            $batchState["mission"] = Join-Path $stageFolder "mission\mission.json"
            Save-BatchState
            Invoke-ProjectPython -PythonArgs @("scripts/calibrate_depth.py","--output",(Join-Path $stageFolder "depth"))
            $batchState["depth"] = Join-Path $stageFolder "depth\depth_calibration.json"
            Save-BatchState
            Invoke-ProjectPython -PythonArgs @("scripts/calibrate_longitudinal.py","--follow","--output",(Join-Path $stageFolder "braking"))
            $batchState["braking"] = Join-Path $stageFolder "braking\longitudinal.json"
        }
        "Quick" {
            # Early integration smoke on the PROVISIONAL config: one navigation-only
            # S0, one full-stack S0 and one S1 stop/hold/resume. Diagnostic only;
            # it does not replace Navigation/Freeze/Acceptance evidence.
            $liveArgs = Get-LiveArgs
            # After Perception/Tune, smoke the actual candidate (detector size,
            # derived cruise, tuned gains) before Navigation locks the source.
            $quickConfig = "configs/baseline_stack.json"
            if ($batchState.ContainsKey("candidate") -and (Test-Path -LiteralPath $batchState["candidate"])) {
                $quickConfig = [string]$batchState["candidate"]
            }
            Write-Host "Quick config: $quickConfig"
            $quickRuns = @(
                @{ Name = "S0-navigation"; Args = @("--scenario","S0","--navigation-only") },
                @{ Name = "S0-full"; Args = @("--scenario","S0") },
                @{ Name = "S1-full"; Args = @("--scenario","S1") }
            )
            foreach ($run in $quickRuns) {
                Invoke-ProjectPython -PythonArgs (@("scripts/run_scenario.py") + $run.Args + @("--config",$quickConfig,"--max-sim-s","120","--output",(Join-Path $stageFolder $run.Name)) + $liveArgs)
            }
            # RV-06: lane monitor replayed on the recorded full-stack runs (diagnostic only).
            if (Test-Path -LiteralPath "lane_unet_best.pth") {
                foreach ($name in @("S0-full","S1-full")) {
                    Invoke-ProjectPython -PythonArgs @("scripts/replay_lane_monitor.py","--run",(Join-Path $stageFolder $name),"--mission",(Need-Result "mission"),"--output",(Join-Path $stageFolder ($name + "-lane-monitor")))
                }
            }
            $batchState["quick"] = $stageFolder
        }
        "Verify" {
            # BL-B-07: live known-distance truth gap, collision dedup and lane-invasion checks.
            $missionPath = Need-Result "mission"
            Invoke-ProjectPython -PythonArgs @("scripts/verify_event_sensors.py","--mission",$missionPath,"--follow","--output",(Join-Path $stageFolder "events"))
            $batchState["event_verification"] = Join-Path $stageFolder "events\event_verification.json"
        }
        "Capture" {
            $null = Need-Result "depth"
            $datasetFolder = Join-Path $stageFolder "dataset"
            Invoke-ProjectPython -PythonArgs @("scripts/capture_frames.py","--samples",[string]$Samples,"--sequence-frames","20","--ray-reference","--map",$Map,"--output",$datasetFolder)
            # Perception/Freeze measure the demo map; a capture on another town is
            # kept apart as training data so it never replaces those measurements.
            $datasetKey = if ($Map -in @("Town10HD_Opt","Town10HD")) { "dataset" } else { "training_dataset" }
            $batchState[$datasetKey] = $datasetFolder
            Save-BatchState
            Invoke-ProjectPython -PythonArgs @("scripts/review_dataset.py",$datasetFolder)
            Start-Process (Join-Path $datasetFolder "review.html")
        }
        "Perception" {
            $datasetPath = Need-Result "dataset"
            $depthPath = Need-Result "depth"
            $detectorArgs = @("--image-size",[string]$DetectorImageSize,"--confidence",[string]$DetectorConfidence)
            Invoke-ProjectPython -PythonArgs (@("scripts/benchmark_perception.py","--dataset",$datasetPath,"--weights","yolo26n.pt","--depth-calibration",$depthPath,"--output",(Join-Path $stageFolder "detector")) + $detectorArgs)
            $batchState["perception"] = Join-Path $stageFolder "detector\perception.json"
            Save-BatchState
            Invoke-ProjectPython -PythonArgs (@("scripts/analyze_depth.py","--dataset",$datasetPath,"--depth-calibration",$depthPath,"--segmentation-weights","segmentation_model.pth","--weights","yolo26n.pt","--output",(Join-Path $stageFolder "depth_analysis")) + $detectorArgs)
            $batchState["depth_analysis"] = Join-Path $stageFolder "depth_analysis\depth_analysis.json"
            Invoke-ProjectPython -PythonArgs @("scripts/preview_lane_monitor.py","--dataset",$datasetPath,"--weights","lane_unet_best.pth","--output",(Join-Path $stageFolder "lane_monitor"))
            $batchState["lane_monitor"] = Join-Path $stageFolder "lane_monitor\monitor.json"
            $brakingPath = Need-Result "braking"
            Invoke-ProjectPython -PythonArgs @("scripts/freeze_thresholds.py","--config","configs/baseline_stack.json","--braking",$brakingPath,"--perception",$batchState["perception"],"--depth-calibration",$depthPath,"--output",(Join-Path $stageFolder "candidate.json"))
            $batchState["candidate"] = Join-Path $stageFolder "candidate.json"
        }
        "Tune" {
            $liveArgs = Get-LiveArgs -RequireEvents
            $candidatePath = Need-Result "candidate"
            Invoke-ProjectPython -PythonArgs (@("scripts/tune_controller.py","--config",$candidatePath,"--output",(Join-Path $stageFolder "sweep")) + $liveArgs)
            $batchState["candidate"] = Join-Path $stageFolder "sweep\selected.json"
            $batchState["tuning"] = Join-Path $stageFolder "sweep\tuning.json"
        }
        "Navigation" {
            $liveArgs = Get-LiveArgs -RequireEvents
            $candidatePath = Need-Result "candidate"
            Invoke-ProjectPython -PythonArgs (@("scripts/run_acceptance_suite.py","--navigation-only","--config",$candidatePath,"--output",(Join-Path $stageFolder "runs")) + $liveArgs)
            $batchState["navigation"] = Join-Path $stageFolder "runs\acceptance.json"
        }
        "Freeze" {
            $candidatePath = Need-Result "candidate"
            $brakingPath = Need-Result "braking"
            $perceptionPath = Need-Result "perception"
            $depthPath = Need-Result "depth"
            $navigationPath = Need-Result "navigation"
            Invoke-ProjectPython -PythonArgs @("scripts/freeze_thresholds.py","--config",$candidatePath,"--braking",$brakingPath,"--perception",$perceptionPath,"--depth-calibration",$depthPath,"--navigation-report",$navigationPath,"--output",(Join-Path $stageFolder "frozen.json"))
            $batchState["frozen"] = Join-Path $stageFolder "frozen.json"
        }
        "Smoke" {
            $liveArgs = Get-LiveArgs -RequireEvents
            $frozenPath = Need-Result "frozen"
            Invoke-ProjectPython -PythonArgs (@("scripts/run_scenario.py","--scenario","S1","--config",$frozenPath,"--output",(Join-Path $stageFolder "S1")) + $liveArgs)
            $batchState["s1"] = Join-Path $stageFolder "S1"
        }
        "Acceptance" {
            $liveArgs = Get-LiveArgs -RequireEvents
            $frozenPath = Need-Result "frozen"
            $null = Need-Result "s1"
            Invoke-ProjectPython -PythonArgs (@("scripts/run_acceptance_suite.py","--config",$frozenPath,"--output",(Join-Path $stageFolder "runs")) + $liveArgs)
            $batchState["acceptance"] = Join-Path $stageFolder "runs\acceptance.json"
        }
        "ReplayFaults" {
            $datasetPath = Need-Result "dataset"
            $depthPath = Need-Result "depth"
            foreach ($faultName in @("flicker","id_reset")) {
                Invoke-ProjectPython -PythonArgs @("scripts/benchmark_perception.py","--dataset",$datasetPath,"--weights","yolo26n.pt","--depth-calibration",$depthPath,"--fault",$faultName,"--output",(Join-Path $stageFolder $faultName))
            }
            $batchState["replay_faults"] = $stageFolder
        }
        "Train" {
            # Prefer a dataset recorded on another town; train_lane.py refuses the
            # demo map's own images unless -AllowDemoMap is given.
            $datasetPath = if ($batchState.ContainsKey("training_dataset")) { Need-Result "training_dataset" } else { Need-Result "dataset" }
            $trainArgs = @("training/batch35/train_lane.py","--repo",".","--dataset",$datasetPath,"--initial","lane_unet_best.pth","--output",(Join-Path $stageFolder "training"),"--hours",[string]$TrainingHours)
            if ($AllowDemoMap) { $trainArgs += "--allow-demo-map" }
            Invoke-ProjectPython -PythonArgs $trainArgs
            $batchState["training"] = Join-Path $stageFolder "training"
        }
        "Evaluate" {
            $datasetPath = Need-Result "dataset"
            $trainingPath = Need-Result "training"
            Invoke-ProjectPython -PythonArgs @("training/batch35/evaluate_lane.py","--repo",".","--dataset",$datasetPath,"--baseline","lane_unet_best.pth","--candidate",(Join-Path $trainingPath "lane_candidate.pth"),"--output",(Join-Path $stageFolder "lane_evaluation"))
            $batchState["lane_evaluation"] = Join-Path $stageFolder "lane_evaluation"
        }
        "Audit" {
            $datasetPath = Need-Result "dataset"
            $depthPath = Need-Result "depth"
            Invoke-ProjectPython -PythonArgs @("scripts/benchmark_perception.py","--dataset",$datasetPath,"--weights","yolo26n.pt","--depth-calibration",$depthPath,"--split","test","--output",(Join-Path $stageFolder "heldout"))
            $batchState["heldout"] = Join-Path $stageFolder "heldout\perception.json"
        }
        "Archive" {
            $sourcePaths = @("src","tests","configs","scripts","docs","training","examples","pyproject.toml","README.md",".github",".gitignore",".gitattributes") | Where-Object { Test-Path -LiteralPath $_ }
            Compress-Archive -Path $sourcePaths -DestinationPath (Join-Path $stageFolder "source.zip") -ErrorAction Stop
            Copy-Item -LiteralPath $StateFile -Destination (Join-Path $stageFolder "state.json") -ErrorAction Stop
            foreach ($key in @("perception","braking","depth","navigation","frozen","acceptance","heldout")) {
                if ($batchState.ContainsKey($key) -and (Test-Path -LiteralPath $batchState[$key] -PathType Leaf)) {
                    Copy-Item -LiteralPath $batchState[$key] -Destination (Join-Path $stageFolder ($key + ".json"))
                }
            }
            Write-Host "Source and compact reports archived. Raw run/dataset folders remain at the paths in state.json."
        }
    }
    Save-BatchState
    Write-Host "Stage finished: $Stage"
    Write-Host "Saved to: $stageFolder"
    Write-Host "State file: $StateFile"
} catch {
    Save-BatchState
    Write-Host "Stage stopped: $Stage. Existing results were retained in $stageFolder"
    throw
}
