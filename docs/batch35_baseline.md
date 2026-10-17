# Baseline implementation batch: BL-A-04 through TEST-09

This batch connects the recovered A1 perception and controller APIs into a fixed-route stopping baseline. It includes implementation and evidence tools for 35 board tasks. **It does not declare those 35 tasks accepted.** Live CARLA measurements, held-out model results and named team review remain outstanding. The task-by-task record is in [batch35_tasks.md](batch35_tasks.md).

## What runs together

A single fixed legal mission is passed to the existing Pure Pursuit/PID controller at mission start. Each exact-frame RGB-D bundle goes through the existing YOLO detector, box-depth association, track confirmation, route corridor, speed governor and behaviour state machine. The vehicle adapter applies the resulting command. The lane model supplies a diagnostic overlay and never replaces the mission path or writes steering commands.

The core uses metres, seconds, radians, x-forward/y-left coordinates and a rear-axle vehicle reference. The adapter converts CARLA's coordinate convention. The front bumper offset is applied once when calculating the stopping gap. World ground truth is retrieved after the control decision and is reserved for scenario orchestration and evaluation.

The same policy runs in the bicycle harness and in `CarlaWorld`. Core modules do not import CARLA. The CARLA-free harness provides functional tests, not evidence of simulator performance, detector accuracy or real braking.

## Behaviour and limits

- The behaviour states are CRUISE, APPROACH, STOPPING, HOLD, RESUME, FINISHED and FAULT_STOP. Endpoint completion is distinct from waiting for an obstruction.
- Tracks require two hits in three frames. IDs include source/model/episode identity; an observation without a tracker ID confirms only by staying within 1.5 m (world XY) of the same class. Track expiry alone never establishes clearance.
- Every observation, confirmed or not, limits speed and blocks resumption. Only a confirmed observation, or an unknown-depth box over the route seen in two of three frames, can force STOPPING. A single flickering frame cannot. Emergency braking still reacts to any observation inside the emergency margin. The behaviour latch engages only once a stop has been required.
- Resuming requires five fresh clear observations spanning at least 0.4 seconds. Invalid depth, stale frames, expired commands and insufficient visibility fail closed.
- The visible-route horizon uses a stationary-scene memory of corridor cells seen clear in the last 15 s. The 90-degree camera cannot see the inside edge of the mission's 10 m-radius right turn while in it, although it saw that edge on the approach. Without the memory every full-stack run held in the junction indefinitely. Out-of-view cells may use the memory. An occluded or invalid observation of a cell erases its record, so only a later clear observation restores it. An in-view occluded or invalid cell always ends the horizon, and a formerly blocked obstacle area still needs a current observation. On the mission harness the oldest record actually relied on was 5.2 s (S0) and 5.9 s (S5), so 15 s leaves about 2.5x margin for slower live turns. The memory is an explicit stationary-scenario assumption; it is not valid for moving road users.
- After a brief stop the speed ramp restarts from the measured speed, not from zero.
- The governor combines route curvature, measured braking, processing/filter delay and perceived front-bumper gap. Speed decreases can take effect immediately; speed increases are limited.
- Planned obstacle braking uses at least 0.4 brake when the controller requests braking. Longitudinal calibration measures 0.4, 0.7 and 1.0 brake at 2, 5 and 8 m/s; a reserve is applied to the measured deceleration.
- Runtime configuration starts PROVISIONAL. Freezing requires depth and braking measurements, a usable validation detection range and five matching navigation-only CARLA runs. Failed attempts remain in the evidence.
- Obstacle footprints and the unknown-box exclusion test use only measured points at least 0.4 m above the ego ground plane. The box around a car seen at an angle also contains road. On the first live full-stack S0 run (17 Oct), counting those road pixels moved a kerb-side Town10HD parked SUV (a map-layer prop about 2.0 m right of the lane centre) progressively into the corridor (estimated near edge 1.77 → 0.80 m). The car stopped about 13 m into the route and held indefinitely. Lateral padding is a fixed 0.2 m, because each pixel is projected at its own depth and the depth spread of an oblique surface is geometry, not error. The mission harness now renders roadside parked cars as a regression test (S0 and S1, with the near side on the lane edge). Obstacles lower than 0.4 m are out of scope. An unknown-depth box is excluded from the route only when none of its measured, raised points (every pixel valid, near boxes subsampled to about 4k pixels) lies within the swept corridor plus 0.3 m. A single bounding rectangle around a roadside car and the background in the same box straddled the turning corridor on the second live S0 run and braked the car from 5 to 2.3 m/s about 10 m before the turn.
- In synchronous CARLA runs the wall-clock guards are hang watchdogs: 5 s observation age and 1 s step. The simulator is paused while the client computes, so a host stall cannot make an observation stale relative to the world. Sim-time freshness is still enforced by frame-matched packets and the sim-time command TTL. The same live run showed a single 1.3 s stall (and a 375 ms one on the navigation run), which the 0.5 s guard had turned into a latched fault. The 0.1 s step budget is still measured. `summary.json` → `timing` reports budget overruns, maximum latencies and stalls over 500 ms, and capture-to-apply p95 remains the scored criterion. The harness keeps the strict 0.5 s guard.

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
& .\scripts\run_batch35.ps1 -Stage Verify
& .\scripts\run_batch35.ps1 -Stage Quick
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

`Verify` (BL-B-07) checks evaluation truth and event sensors live: known-distance bumper-to-surface gaps for cars and walkers, a roadside placement outside the corridor, two deliberate low-speed contacts that must count as exactly two, and a lane change across a broken marking that must count unless explicitly declared. Collision and lane-invasion metrics are scored only with a passing report for the same map build; Tune, Navigation, Smoke and Acceptance require it. `Quick` is a 10-15 minute integration smoke on the provisional configuration: one navigation-only S0, one full-stack S0 and one S1. Run it before the long capture so integration faults surface first. Its runs are diagnostic and never count as acceptance evidence.

`LiveSetup` produces the fixed mission, route overlay, off-axis axial/radial depth comparison and braking measurements. The route must also be visually reviewed for legality through junctions. `Capture` creates RGB, instance-derived labels, raw depth, camera/pose metadata and evaluation-only ground truth, then opens the dataset review page. `Perception` measures the existing YOLO model, depth and optional segmentation-mask ablation on validation sequences. The masking ablation runs on both ground-truth and YOLO boxes and records a `masking_decision`: KEEP_OFF unless detector-box depth error or invalid-rate gains clear a threshold within a 30 ms mask budget (BL-B-05). Low range or excessive runtime stops progression; those measurements must not be replaced with guessed thresholds.

`Tune` compares four public controller configurations on clear-road S0 runs. It does not alter private controller state or use obstacle acceptance trials for tuning. `Navigation` then runs five independent S0 trials. `Freeze` verifies that their hashes match the candidate settings. `Smoke` is the first full-stack S1 stop/hold/resume run. `Acceptance` runs five seeds per S0-S5, with an actual 60-second stationary hold in S4. A failed cleanup aborts subsequent live trials.

## Measurements and evidence

Each run contains an initial INCOMPLETE manifest, source/config/model/mission identities, frame samples, events, measured stage durations and a cleanup result. A run becomes COLLECTED only after normal completion and successful cleanup. Collection alone is not acceptance. Aggregation recomputes metrics from hash-checked raw files and retains failed attempts.

`CarlaWorld` reads the frame it has just ticked. In synchronous mode the simulator waits while the client collects that frame's callbacks, so there is no reason to act on older images. `capture_apply_s` follows the board definition: host-monotonic receipt of the frame's first sensor callback to the `apply_control` call. The more conservative `tick_to_apply_s` (tick submission to the returned tick that carries the command) is kept as a report-only figure.

Each scenario obstacle records where it landed: corners, map lane type and whether it overlaps the swept corridor. A placement that contradicts its definition (S3 inside the corridor, S1/S2/S4/S5 outside it) stops the run. Seeds come from each scenario YAML. Lane-invasion events carry the rear-axle route s and are excluded only by an entry in `configs/declared_crossings.json` covering that s interval and marking type; none are declared by default.

Metrics include moving fixed-route rear-axle CTE, stable-cruise speed error, capture-to-confirmed-application wall latency, simulation age, collision contacts, lane incursions, stop margin, hold duration/speed, resume and endpoint completion. Sensor callback-to-application latency is also recorded separately. Harness wall-clock deadlines are isolated for deterministic functional tests and cannot count toward the 100 ms live criterion. The live sensor pipeline's lag may fail that criterion; it must be measured, not hidden by reporting simulation time instead.

The acceptance fixture remains authoritative: mean moving absolute CTE <= 0.3 m; stable-cruise speed RMS error <= 0.5 m/s; capture-to-application p95 <= 100 ms; no collisions or unplanned incursions; required stopped margins 2.5-6 m; hold speed <= 0.1 m/s. Missing metrics fail rather than becoming zero. S3 roadside runs are paired with matching S0 runs. Completion is not required for persistent-block S4; its continuous hold must last at least 60 seconds. Lane-invasion events count conservatively unless explicitly annotated as a legal crossing; junctions are not exempted wholesale.

The A1 recorder uses nearest-lane tangent CTE at the actor origin. This differs from A2's fixed-route rear-axle CTE. `scripts/compare_runs.py` therefore recomputes the A1 definition for A2 runs from their logged rear-axle poses (plus the 1.386 m axle-to-origin offset) and the exported lane topology. Applied to the committed TEST-01 capture, the offline definition reproduces A1's logged values within 1.3 cm outside junctions and 3 cm inside them. The tool plots CTE traces and distributions, wall latency (A1 cycle before apply versus A2 receipt to apply) and command age/control period, and lists the route, timing, speed and vehicle differences that remain. Its status stays REVIEW_REQUIRED: the numbers are measured, the interpretation is the team's.

The RV-06 lane monitor is replayed on recorded run video by `scripts/replay_lane_monitor.py` (the Quick stage runs it automatically when `lane_unet_best.pth` is present). LaneUNet's centreline is projected to the ground, fitted and evaluated 8 m ahead of the rear axle, just beyond the nearest road the camera sees; the fixed route's offset is evaluated at the same point. Evaluating at the rear axle would extrapolate into the camera's blind zone (0.5 m errors on curves). The output is a plot, HUD screenshots on curves and straights, sign agreement and offset differences with confidence and validity. The monitor never runs inside the control loop.

`check_gates.py --help` lists evidence inputs for G1/G2 review. Its statuses request review; they do not merge a PR or mark the board. A measured result is not a substitute for inspecting route/video overlays and the underlying logs.

## Additional commands and one PR

Use `scripts/compare_runs.py --help` for the A1/A2 comparison and `scripts/check_gates.py --help` for the gate summary. Python scripts are run with `.venv\Scripts\python.exe`. Their output paths should be new folders under Downloads.

```powershell
& .\scripts\run_batch35.ps1 -Stage Archive
```

Archive saves source, state and compact reports in Downloads; it does not embed large datasets, weights or every video. Keep the referenced raw folders and share them through team storage. In the PR, commit compact measurements plus a manifest of raw evidence locations and hashes, rather than committing a multi-gigabyte dataset. Use one new feature branch targeting `USYD-ELEC5308/26-S2-Assignment2-Group7:main`. Keep acceptance claims limited to completed measurements.

## Verification available with this delivery

The delivery includes a local verification record under `docs/evidence/batch35-local-verification.txt`. Python unit/integration tests, six synthetic scenarios, fault cases and a one-epoch training software smoke were run in a Linux CPU environment. PowerShell scripts were reviewed statically. CARLA, Windows execution, CUDA performance and useful learned-model accuracy have not been verified in that environment.
