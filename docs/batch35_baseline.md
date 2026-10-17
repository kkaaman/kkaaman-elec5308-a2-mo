# Baseline implementation batch: BL-A-04 through TEST-09

This batch connects the recovered A1 perception and controller APIs into a fixed-route stopping baseline. It includes implementation and evidence tools for 35 board tasks. **It does not declare those 35 tasks accepted.** Live CARLA measurements, held-out model results and named team review remain outstanding. The task-by-task record is in [batch35_tasks.md](batch35_tasks.md).

## What runs together

A single fixed legal mission is passed to the existing Pure Pursuit/PID controller at mission start. Each exact-frame RGB-D bundle goes through the existing YOLO detector, box-depth association, track confirmation, route corridor, speed governor and behaviour state machine. The vehicle adapter applies the resulting command. The lane model supplies a diagnostic overlay and never replaces the mission path or writes steering commands.

The core uses metres, seconds, radians, x-forward/y-left coordinates and a rear-axle vehicle reference. The adapter converts CARLA's coordinate convention. The front bumper offset is applied once when calculating the stopping gap. World ground truth is retrieved after the control decision and is reserved for scenario orchestration and evaluation.

The same policy runs in the bicycle harness and in `CarlaWorld`. Core modules do not import CARLA. The CARLA-free harness provides functional tests, not evidence of simulator performance, detector accuracy or real braking.

## Behaviour and limits

- The behaviour states are CRUISE, APPROACH, STOPPING, HOLD, RESUME, FINISHED and FAULT_STOP. Endpoint completion is distinct from waiting for an obstruction.
- Tracks require two hits in three frames. IDs include source/model/episode identity; anonymous detections still block conservatively. Track expiry alone never establishes clearance.
- Resuming requires five fresh clear observations spanning at least 0.4 seconds. Invalid depth, stale frames, expired commands and insufficient visibility fail closed.
- The governor combines route curvature, measured braking, processing/filter delay and perceived front-bumper gap. Speed decreases can take effect immediately; speed increases are limited.
- Planned obstacle braking uses at least 0.4 brake when the controller requests braking. Longitudinal calibration measures 0.4, 0.7 and 1.0 brake at 2, 5 and 8 m/s; a reserve is applied to the measured deceleration.
- Runtime configuration starts PROVISIONAL. Freezing requires depth and braking measurements, a usable validation detection range and five matching navigation-only CARLA runs. Failed attempts remain in the evidence.

The current model assumes stationary obstacles, a locally planar road, a fixed Tesla Model 3 and the recorded camera rig. It does not model tire slip in the harness, arbitrary terrain, reliable detection of every road user, or the camera's near-field blind area. Visibility samples are a limited clearance check, not proof that every point in space is empty. Captured training scenes use parked actors and do not establish moving-pedestrian tracking performance. Changes to vehicle, map, rig, weights, runtime configuration or implementation require the relevant measurements to be repeated.

## Windows setup and stage commands

Apply the downloadable bundle using its `START_HERE.md` first. From the repository root, use PowerShell:

```powershell
& .\scripts\setup_batch35.ps1
& .\scripts\run_batch35.ps1 -Stage Offline
```

The setup creates a Python 3.12 virtual environment, installs the declared project extras, installs the selected CUDA PyTorch build and checks a GPU matrix operation. `-CpuOnly` is available for offline development; GPU training and performance acceptance require a working CUDA environment. VS Code can use `.venv\Scripts\python.exe` as its interpreter.

Obtain the existing `lane_unet_best.pth`, `segmentation_model.pth` and `yolo26n.pt` from the repository's LFS assets or the team's verified model storage. Keep their names in the repository root. Do not train or download a replacement YOLO model simply to satisfy a missing-file check. `git lfs pull` retrieves files actually tracked by LFS; it cannot retrieve an untracked team asset.

Start CARLA 0.9.16 and wait until Town10HD_Opt has loaded. Close other clients that advance the world. The suite owns synchronous ticking and requires an empty vehicle/sensor scene. Run these stages in order, stopping at the first failure:

```powershell
& .\scripts\run_batch35.ps1 -Stage LiveSetup
& .\scripts\run_batch35.ps1 -Stage Capture -Samples 12000
& .\scripts\run_batch35.ps1 -Stage Perception
& .\scripts\run_batch35.ps1 -Stage Tune
& .\scripts\run_batch35.ps1 -Stage Navigation
& .\scripts\run_batch35.ps1 -Stage Freeze
& .\scripts\run_batch35.ps1 -Stage Smoke
& .\scripts\run_batch35.ps1 -Stage Acceptance
& .\scripts\run_batch35.ps1 -Stage ReplayFaults
```

Paste one stage at a time. Every invocation uses a new dated folder under Downloads; the state file `Group7_batch35_state.json` remembers result paths. A failure retains the terminal log and partial evidence. Do not delete a failed run to improve the reported success rate. The latest result location is available with:

```powershell
Get-Content "$HOME\Downloads\Group7_batch35_state.json"
```

`LiveSetup` produces the fixed mission, route overlay, off-axis axial/radial depth comparison and braking measurements. The route must also be visually reviewed for legality through junctions. `Capture` creates RGB, instance-derived labels, raw depth, camera/pose metadata and evaluation-only ground truth, then opens the dataset review page. `Perception` measures the existing YOLO model, depth and optional segmentation-mask ablation on validation sequences. Low range or excessive runtime stops progression; those measurements must not be replaced with guessed thresholds.

`Tune` compares four public controller configurations on clear-road S0 runs. It does not alter private controller state or use obstacle acceptance trials for tuning. `Navigation` then runs five independent S0 trials. `Freeze` verifies that their hashes match the candidate settings. `Smoke` is the first full-stack S1 stop/hold/resume run. `Acceptance` runs five seeds per S0-S5, with an actual 60-second stationary hold in S4. A failed cleanup aborts subsequent live trials.

## Measurements and evidence

Each run contains an initial INCOMPLETE manifest, source/config/model/mission identities, frame samples, events, measured stage durations and a cleanup result. A run becomes COLLECTED only after normal completion and successful cleanup. Collection alone is not acceptance. Aggregation recomputes metrics from hash-checked raw files and retains failed attempts.

Metrics include moving fixed-route rear-axle CTE, stable-cruise speed error, capture-to-confirmed-application wall latency, simulation age, collision contacts, lane incursions, stop margin, hold duration/speed, resume and endpoint completion. Sensor callback-to-application latency is also recorded separately. Harness wall-clock deadlines are isolated for deterministic functional tests and cannot count toward the 100 ms live criterion. The live sensor pipeline's lag may fail that criterion; it must be measured, not hidden by reporting simulation time instead.

The acceptance fixture remains authoritative: mean moving absolute CTE <= 0.3 m; stable-cruise speed RMS error <= 0.5 m/s; capture-to-application p95 <= 100 ms; no collisions or unplanned incursions; required stopped margins 2.5-6 m; hold speed <= 0.1 m/s. Missing metrics fail rather than becoming zero. S3 roadside runs are paired with matching S0 runs. Completion is not required for persistent-block S4; its continuous hold must last at least 60 seconds. Lane-invasion events count conservatively unless explicitly annotated as a legal crossing; junctions are not exempted wholesale.

The A1 recorder uses nearest-lane tangent CTE at the actor origin. This differs from A2's fixed-route rear-axle CTE. `scripts/compare_runs.py` retains both definitions, plots A2 traces and reports DEFINITION_REVIEW_REQUIRED. It deliberately makes no percentage-improvement claim. TEST-03 needs an agreed comparable measurement before acceptance.

`check_gates.py --help` lists evidence inputs for G1/G2 review. Its statuses request review; they do not merge a PR or mark the board. A measured result is not a substitute for inspecting route/video overlays and the underlying logs.

## Additional commands and one PR

Use `scripts/compare_runs.py --help` for the A1/A2 comparison and `scripts/check_gates.py --help` for the gate summary. Python scripts are run with `.venv\Scripts\python.exe`. Their output paths should be new folders under Downloads.

```powershell
& .\scripts\run_batch35.ps1 -Stage Archive
```

Archive saves source, state and compact reports in Downloads; it does not embed large datasets, weights or every video. Keep the referenced raw folders and share them through team storage. In the PR, commit compact measurements plus a manifest of raw evidence locations and hashes, rather than committing a multi-gigabyte dataset. Use one new feature branch targeting `USYD-ELEC5308/26-S2-Assignment2-Group7:main`. Keep acceptance claims limited to completed measurements.

## Verification available with this delivery

The delivery includes a local verification record under `docs/evidence/batch35-local-verification.txt`. Python unit/integration tests, six synthetic scenarios, fault cases and a one-epoch training software smoke were run in a Linux CPU environment. PowerShell scripts were reviewed statically. CARLA, Windows execution, CUDA performance and useful learned-model accuracy have not been verified in that environment.
