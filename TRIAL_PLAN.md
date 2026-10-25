# Live trial plan — issues #42 and #45

Branch `feat/RV-06-08-TEST-22-lane-monitor-sensor-docs-timing` (base: team main
`d1edba6`). Lab PC: Windows, CARLA 0.9.16 server + client, `Town10HD_Opt`,
project `.venv`. Every command below is PowerShell from the repository root.
Nothing here changes a frozen threshold; every result goes to
`runs/<set>/<id>` and is attached to the issue as evidence (copy curated files
into `docs/evidence/` in the PR that closes the issue).

## 0. Session setup (once per CARLA session)

```powershell
Set-Location 'D:\26-S2-Assignment2-Group7'      # adjust to the clone path
git fetch origin
git checkout feat/RV-06-08-TEST-22-lane-monitor-sensor-docs-timing
git rev-parse HEAD | Tee-Object -Variable candidateHash  # record in every evidence folder
$py = '.\.venv\Scripts\python.exe'
$carlaRoot = 'D:\CARLA_0.9.16'
& $py -m pip install -e . --no-deps
& $py -m pytest tests -q -p no:cacheprovider           # must be all green before any trial
& $py scripts/check_simulator_boundary.py --smoke
& $py scripts/verify_assets.py
```

Start the server (repaired runtime per `docs/carla_rgb_readback_fix.md` if this
PC needs it; otherwise the stock executable):

```powershell
# Windowed ("HUD"/spectator visible):
Start-Process "$carlaRoot\CarlaUE4.exe" -ArgumentList '-quality-level=Epic','-carla-rpc-port=2000'
# Headless alternative used in section 4:
# Start-Process "$carlaRoot\CarlaUE4.exe" -ArgumentList '-RenderOffScreen','-quality-level=Epic','-carla-rpc-port=2000'
Start-Sleep -Seconds 40
& $py scripts/check_baseline_mission.py --verify-map   # map/OpenDRIVE hash must match the frozen route
```

Preconditions enforced by the runner: no other client, no stray vehicles/
walkers/sensors, `no_rendering_mode` off for RGB runs. If
`check_baseline_mission.py --verify-map` fails, stop; the installed map differs
from the frozen route.

## 1. TEST-10 — rerun the S0 suite on the candidate hash

**Runnable now.** Evidence for: unchanged baseline behaviour on this branch
(both PRs), and the `headless` reference profile for TEST-22.

```powershell
$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
& $py scripts/run_scenario.py --runs 5 --output "runs/S0/rerun-$candidateHash-$runId"
# Exit code 0 only when summary.json status == PASS (5/5 runs).
Get-Content "runs/S0/rerun-$candidateHash-$runId/summary.json" | Select-String '"status"','worst_run_metrics' -Context 0,8
```

Produces `run-01..run-05/{manifest,samples.csv,events,summary}.json` and the
suite `summary.json` (worst-run CTE, cruise RMS, capture-to-apply p95, false
stops, collisions, lane invasions; `execution_provenance.git_head` must equal
`$candidateHash`). Acceptance evidenced: TEST-10 rerun; S0 limits per
`tests/fixtures/acceptance_definitions.json`.

Quick offline harness check (no CARLA) for completeness:

```powershell
& $py scripts/run_s0_acceptance.py --output runs/S0-offline
& $py scripts/evaluate_runs.py (Get-ChildItem runs/S0-offline -Directory | ForEach-Object FullName) --output runs/S0-offline/aggregate.json
```

## 2. TEST-22 — headless versus HUD timing against the 100 ms budget

**Runnable now** for the configurations that exist. The S0 runner has no
in-loop HUD; the comparison varies (a) server rendering and (b) the RGB
pipeline, which is exactly the "background workload" the issue asks about. Run
each configuration as a full RGB-enabled run (latency evidence requires RGB).

```powershell
# (a) headless server: restart CarlaUE4.exe with -RenderOffScreen, re-run check_baseline_mission --verify-map, then
$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
& $py scripts/run_scenario.py --runs 1 --output "runs/TEST-22/headless-$runId"

# (b) windowed server with the spectator following the ego (the live "HUD"); restart without -RenderOffScreen, then
& $py scripts/run_scenario.py --runs 1 --output "runs/TEST-22/hud-$runId"

# (c) warm admitted models (YOLO26n + tracker + governor), same-frame path, two episodes x 120 frames
& $py scripts/measure_operational_governance.py --host 127.0.0.1 --output "runs/TEST-22/warm-$runId" --frames 120 --episodes 2

# Profile and gate
& $py scripts/profile_loop_timing.py --output "runs/TEST-22/profile-$runId" `
    --run "headless=runs/TEST-22/headless-$runId/run-01" `
    --run "hud=runs/TEST-22/hud-$runId/run-01" `
    --run "warm=runs/TEST-22/warm-$runId/timing.jsonl" `
    --baseline headless --final hud
```

Produces `latency_profile.json` (per-label p50/p95/max per stage and end-to-end,
real-time factor, frame age, deltas versus headless, `gate` block) and
`latency_profile.png` (CDFs with the 100 ms line, per-stage p95 bars). The
script exits 0 only when the `--final` configuration's p95 <= 100 ms with >= 20
samples; otherwise the gate is `BLOCKED` with the reason. Acceptance evidenced:
TEST-22 "final baseline config p95 <= 100 ms or gate explicitly blocked". Run
the five-run suite from section 1 under the configuration you intend to admit
and point `--final` at its `run-0N` folders as well (one label per run) so the
worst run is visible.

Remaining live gap: an in-loop HUD renderer does not exist yet (E3). When it
does, log its wall cost per tick as `hud_s` in the run samples; the profiler
picks the column up automatically.

## 3. RV-06 — recorded-curve overlay check

**Runnable now, offline on the lab PC** (needs the LFS lane checkpoint). The S0
runner does not persist images, and the only driving recorder on main that does
is the A1 observational capture (`src/carla_adapter/a1_capture.py`): it writes
`video.avi` (processed display frames, 320 x 180 unless the A1 settings differ),
`video_frames.csv` (`video_frame`, `sim_time_s`, `latest_control_sample`) and
`samples.csv` (`actor_x_m`, `actor_y_m`, `actor_yaw_rad`,
`nearest_lane_cte_signed_m`). Its frames carry the A1 overlay, so this is a
convention/overlay check, not a lane-model accuracy claim.

```powershell
New-Item -ItemType Directory -Force runs/RV-06 | Out-Null
& $py src/carla_adapter/a1_capture.py --repo . --detector-weights yolo26n.pt --output runs/RV-06/clip --smoke-seconds 120
```

Then run the monitor over the clip. The map reference is the A1 definition
already used by TEST-03 (`nearest_lane_cte_signed_m`, positive left, at the
actor origin, 1.386 m ahead of the rear axle — a negligible lateral difference
on the route's gentle bends); the frozen route CTE is also available through
`carla_adapter.s0_runner.signed_cross_track` when the clip follows the
baseline route.

```powershell
@'
import csv, json, math, cv2
from pathlib import Path
from autonomy.paths import model_path
from evaluation.a1_comparison import REAR_AXLE_TO_ACTOR_ORIGIN_M, LaneCentrelines
from geometry.camera import CameraIntrinsics
from perception.lane_inference import LaneSegmenter
from perception.lane_path import extract_ego_path
from planning.lane_graph import LaneTopology
from visualisation.hud import draw_departure_overlay
from visualisation.lane_departure import LaneDepartureMonitor, lane_estimate_from_centerline

clip = Path("runs/RV-06/clip"); out = Path("runs/RV-06/overlay"); out.mkdir(parents=True, exist_ok=True)
samples = list(csv.DictReader((clip / "samples.csv").open()))
index = list(csv.DictReader((clip / "video_frames.csv").open()))
lanes = LaneCentrelines(LaneTopology.load(Path("docs/evidence/baseline-batch/junction-topology.json")))
video = cv2.VideoCapture(str(clip / "video.avi"))
ok, first = video.read(); height, width = first.shape[:2]; video.set(cv2.CAP_PROP_POS_FRAMES, 0)
cal = CameraIntrinsics.from_horizontal_fov(width, height, math.radians(90)).calibration(
    ((0, 0, 1, 1.5 + REAR_AXLE_TO_ACTOR_ORIGIN_M), (-1, 0, 0, 0), (0, -1, 0, 2.0), (0, 0, 0, 1)))
segmenter, monitor, rows = LaneSegmenter(model_path("lane")), LaneDepartureMonitor(), []
for record in index:
    ok, bgr = video.read()
    if not ok: break
    sample = samples[int(record["latest_control_sample"])]
    # A1 logged the signed nearest-lane offset (positive left) at the actor origin;
    # TEST-03 reproduced it offline to ~1.3 cm. Junction frames have no single reference.
    map_cte = float(sample["nearest_lane_cte_signed_m"])
    in_junction = lanes.signed_offset(float(sample["actor_x_m"]), float(sample["actor_y_m"]))[2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    path = extract_ego_path(segmenter.predict(rgb))
    t = float(record["sim_time_s"]); frame = int(record["video_frame"])
    estimate = None if path is None else lane_estimate_from_centerline(path["centerline"], cal, frame_id=frame, sim_time_s=t)
    report = monitor.update(estimate, None if in_junction else map_cte, now_sim_time_s=t)
    rows.append(dict(report.as_record(), in_junction=bool(in_junction)))
    if frame % 5 == 0:
        cv2.imwrite(str(out / f"{frame:06d}.png"), cv2.cvtColor(draw_departure_overlay(rgb, report), cv2.COLOR_RGB2BGR))
(out / "monitor.json").write_text(json.dumps(rows, indent=2))
valid = [r for r in rows if r["lane_monitor_valid"] and abs(r["lane_map_cte_m"]) > 0.1]
agree = sum(math.copysign(1, r["lane_image_offset_m"]) == math.copysign(1, r["lane_map_cte_m"]) for r in valid)
print(f"{len(rows)} frames, {len(valid)} valid with |CTE|>0.1 m, sign agreement {agree}/{len(valid)}, "
      f"mean |difference| {sum(abs(r['lane_difference_m']) for r in valid) / max(1, len(valid)):.3f} m")
'@ | Set-Content runs/RV-06/overlay_check.py
& $py runs/RV-06/overlay_check.py
```

Produces per-frame overlays (every fifth frame) and `monitor.json`
(`lane_image_offset_m`, `lane_map_cte_m`, `lane_difference_m`, confidence,
validity, reasons, junction flag). Acceptance evidenced: RV-06 "correct sign and
overlay verified on recorded curves" — report sign agreement on the valid
frames inside the clip's left and right turns and the mean |difference|. The
camera mount (1.5 m / 2.0 m from the actor origin) is the A2 rig; if the A1
capture used a different mount, edit `cal` to match before judging magnitudes.
"Monitor no longer rebuilds controller route" is evidenced offline by the code
(no route interface; `tests/unit/test_lane_departure.py`).

## 4. TEST-05 — five S3 and five S4 trials

**BLOCKED on the scenario runner.** `scripts/run_scenario.py` runs the frozen
obstacle-free S0 mission only; obstacle actors, the stop/HOLD/resume state
machine (BL-D-08) and the evaluation-truth gap stream (`carla_adapter/scenario.py`,
`ground_truth.py` placeholders) are not on main. Do not improvise a runner for
these trials: the acceptance evidence needs `phase == hold` samples, a
`bumper_gap_m` column from evaluation truth and `obstacle_spawned` /
`obstacle_removed` / `obstacle_confirmed` events, as defined in
`src/evaluation/scenario_metrics.py` (module docstring).

What is ready now and runs unchanged once the runner lands (expected layout:
`runs/S3/<id>/run-01..05`, each with `manifest.json`, `samples.csv`,
`events.json`, `summary.json`):

```powershell
# S3: car outside the corridor, five trials, matched against an S0 run from section 1
& $py scripts/evaluate_scenario_runs.py S3 (Get-ChildItem "runs/S3/$runId" -Directory -Filter 'run-*' | ForEach-Object FullName) `
    --matched-s0 "runs/S0/rerun-$candidateHash-$runId/run-01" --output "runs/S3/$runId/summary.json"

# S4: persistent obstacle, 60 s HOLD, five trials; route completion is NOT APPLICABLE by the fixture
& $py scripts/evaluate_scenario_runs.py S4 (Get-ChildItem "runs/S4/$runId" -Directory -Filter 'run-*' | ForEach-Object FullName) `
    --output "runs/S4/$runId/summary.json"
```

Produces per-run metrics with the frozen limits read from
`tests/fixtures/acceptance_definitions.json`: S3 `false_stops` (= 0) and
`cruise_speed_drop_mps` (<= 0.5 versus the matched S0 stable-cruise mean) plus
the shared S0 rows; S4 `hold_duration_s` (>= 60), `hold_max_speed_mps`
(<= 0.1), `stop_margin_min_m` (>= 2.5), `stop_margin_max_m` (<= 6.0), with
`route_completion` reported `NOT_APPLICABLE`. Missing evidence is
`INSUFFICIENT_EVIDENCE`, never PASS. Acceptance evidenced: TEST-05.

## 5. TEST-07 — five S5 bend-obstacle trials

**BLOCKED** for the same reason as section 4. Evaluation command once runs exist:

```powershell
& $py scripts/evaluate_scenario_runs.py S5 (Get-ChildItem "runs/S5/$runId" -Directory -Filter 'run-*' | ForEach-Object FullName) `
    --mission configs/mission_baseline.json --output "runs/S5/$runId/summary.json"
```

Produces the S0 rows, stop margins, HOLD max speed, `resume_delay_median_s`
(report-only) and the new `report_only.speed_envelope` record: ego speed and
evaluation-truth gap at first `obstacle_confirmed`, the required stopping
distance `v^2 / (2 a) + stop_margin` with the frozen `deceleration_mps2`
(1.4976) and `stop_margin_m` (6.0) from the mission freeze block, and
`envelope_ok`. Acceptance evidenced: TEST-07 (five trials) and the S5
visibility/speed envelope the issue asks to record.

The bend site must be chosen so the obstacle becomes visible at or beyond the
usable first-admission range recorded in the freeze block (22.3 m, stationary
vehicle only); `scripts/capture_operational_approach.py --preflight` is the
existing tool for checking a site before scripting it.

## 6. TEST-17 — sensor rig profile with warm models

**Partly runnable now.** The warm RGB-D + YOLO + filter + governor costs come
from section 2(c) (`timing.jsonl`): `sensor_availability`, `bundle`,
`detector`, `filter`, `controller`, `decision_and_overhead` p50/p95 and the
real-time factor are in `latency_profile.json`. Segmentation and lane-monitor
costs require a consumer that logs `segmentation_s` / `lane_monitor_s` per tick
(none exists in the runner yet; the demos are legacy and not instrumented).
Offline, the warm per-frame cost of each model can be measured on recorded
frames:

```powershell
@'
import time, cv2, numpy as np
from pathlib import Path
from autonomy.paths import model_path
from perception.lane_inference import LaneSegmenter
from perception.lane_path import extract_ego_path
from segmentation import SemanticSegmenter
video = cv2.VideoCapture("runs/RV-06/clip/video.avi"); images = []
while len(images) < 65:
    ok, bgr = video.read()
    if not ok: break
    images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
lane, seg = LaneSegmenter(model_path("lane")), SemanticSegmenter(str(model_path("segmentation")))
for image in images[:5]: lane.predict(image); seg.predict(image)          # warm-up
rows = []
for image in images:
    t0 = time.perf_counter(); mask = lane.predict(image); t1 = time.perf_counter()
    extract_ego_path(mask); t2 = time.perf_counter(); seg.predict(image); t3 = time.perf_counter()
    rows.append({"lane_model_s": t1 - t0, "lane_monitor_s": t2 - t1, "segmentation_s": t3 - t2, "end_to_end_ms": (t3 - t0) * 1000})
import json; Path("runs/TEST-17").mkdir(parents=True, exist_ok=True)
Path("runs/TEST-17/optional-workloads.json").write_text(json.dumps(rows))
'@ | Set-Content runs/TEST-17/measure_optional.py
& $py runs/TEST-17/measure_optional.py
& $py scripts/profile_loop_timing.py --output runs/TEST-17/profile `
    --run "warm=runs/TEST-22/warm-$runId/timing.jsonl" --run optional=runs/TEST-17/optional-workloads.json --baseline warm
```

Acceptance evidenced: TEST-17 "real-time factor and p95 E2E reported" (warm
label) and the per-workload costs that justify the `RateLimitedWorkload`
cadence (`perception.cadence`; stale masks are reported invalid, never reused as
current — `tests/unit/test_cadence.py`). LiDAR is not implemented; its row stays
"not applicable" in the report.

## Evidence checklist per issue

| Issue | Row | Live set | Attach |
|---|---|---|---|
| #42 | RV-06 | section 3 | `runs/RV-06/overlay/monitor.json` + 3 overlay PNGs from each curve |
| #42 | TEST-07 | section 5 (blocked) | `runs/S5/<id>/summary.json` once the runner exists |
| #42 | TEST-10 | section 1 | `runs/S0/rerun-<hash>-<id>/summary.json` |
| #42 | TEST-22 | section 2 | `runs/TEST-22/profile-<id>/latency_profile.{json,png}` |
| #45 | RV-08 | none (docs) | Sphinx build log; `docs/sensor_rig.md` |
| #45 | TEST-05 | section 4 (blocked) | `runs/S3/<id>/summary.json`, `runs/S4/<id>/summary.json` |
| #45 | TEST-17 | section 6 | `runs/TEST-17/profile/latency_profile.json`, `optional-workloads.json` |
