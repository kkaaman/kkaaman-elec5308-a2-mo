# TEST-01: reproducible A1-settings comparison capture

Owner: Abdullah. Reviewer: Zhaolin. Live evidence is required before completion.
The supplied 0.22 m / 1.14 m figures are UNVERIFIED SUPPLIED RESULTS, not measured
results of this recorder and not acceptance thresholds.

## Baseline identity

This runs the inherited `src/demo_fused_lane_following.py` with its existing A1
settings, on the reviewed source snapshot from team main 3a518cb. It is an
A1-settings comparison using the recovered library, not a pristine historical
A1 checkout. RV-01/RV-03 and other integrated dependency changes remain visible
in source hashes and a complete source ZIP. No lane, planning, controller, miss
handling or traffic-light algorithm is changed by the recorder. The provisional
A2 vehicle geometry adapter and synchronous world session are not substituted.

Original settings include 3 m/s cruise, 120 s wall-clock duration, 2.5 m
controller wheelbase, 0.6 rad steering normalization, 5 m lookahead, repeated lane
route replacement, original junction choice/spawn heuristics and asynchronous
simulation. Vehicle blueprint choice remains the demo's first library match;
its exact blueprint, attributes, requested spawn and physics data are recorded.
The requested sensor mounts and actual camera attributes are also archived.
For later matched comparisons, use this exact recorded blueprint/spawn/config.

The recorder redirects weight paths to identical local assets, wraps final
actuator application/display recording and redirects existing failure snapshots.
It does not write to inherited source files. It refuses unexpected source hashes,
wrong segmentation/lane hashes, absent local detector weights, another town,
synchronous mode or existing vehicles/sensors. Use an isolated CARLA instance
with no other client spawning actors or changing settings. Cleanup only targets
actors observed spawning through the demo wrapper, including model-load failures
that occur before the original demo's finally block.

## Commands

Install the project. With CARLA running Town10HD_Opt and a local yolo26n.pt asset:

```powershell
.\.venv\Scripts\python.exe examples/a1/run_baseline.py --repo . --detector-weights "C:\path\to\yolo26n.pt" --check
.\.venv\Scripts\python.exe examples/a1/run_baseline.py --repo . --detector-weights "C:\path\to\yolo26n.pt" --output runs/TEST-01/smoke-01 --smoke-seconds 10
.\.venv\Scripts\python.exe examples/a1/run_baseline.py --repo . --detector-weights "C:\path\to\yolo26n.pt" --output runs/TEST-01/baseline-01
```

Outputs must be new directories. Smoke runs are labelled SMOKE_ONLY and do not
complete TEST-01. Allow the full run to finish, or record any early termination
and physical failure honestly. Collision/failure evidence is part of the baseline;
a successful capture does not mean successful driving. Press q in the OpenCV
window to stop; early quit is recorded. No substitute YOLO model or silent weight
download is used. Preserve the exact detector file/checksum for reproduction.

## Raw evidence and definitions

- `samples.csv`: applied commands, requested angle, actual speed/pose, simulation
  frame/time, wall timestamps, driving mode, misses and both CTE definitions.
- `video.avi`: processed camera display frames, MJPG at nominal 10 fps.
- `video_frames.csv`: actual wall/simulation timestamps and latest command sample
  for every video frame. Video playback is not wall-time accurate at a slow loop.
- `manifest.json`: source hashes, Git revision/status, hardware, package/device
  details, world/weather/map fingerprint, model hashes, constants and spawn data.
- `pip-freeze.txt` and `source_snapshot.zip`: environment and pinned source.
- `summary.json`, optional `error.txt` and existing demo failure PNGs: measured
  summaries, capture completeness, collisions, early quit and cleanup limitations.

Nearest-lane CTE is signed perpendicular distance from actor origin to the
nearest driving-lane centre tangent. It uses simulator map data for evaluation
only and does not enter control. Its absolute mean/RMS/max are reported. At
junctions or after lane departure, nearest lane identity can change; road/lane
IDs are logged. It is not a fixed-mission-route completion metric.

Controller CTE comes from a newly returned PathFollower result for that cycle;
stale reused diagnostics are excluded. Receding-horizon paths may produce low
controller CTE while actual lane tracking is poor. These metrics must not be
combined or compared directly to the unverified prompt figures.

The asynchronous A1 camera callback does not expose image frame IDs/timestamps.
Logged simulation frame/time refer to command observation, not proof of aligned
camera/pose. `cycle_wall_before_apply_s` is elapsed cycle wall time, not measured
sensor age. Logging and video encoding add overhead and can affect asynchronous
behaviour. CPU-only inference is recorded as such; do not claim GPU or real-time
performance. Repeatability means archived inputs/settings/provenance, not bitwise
repeatability of asynchronous physics or perception.

Keep raw run folders, video, exact weights and an offline archive outside Git's
normal code commits (`runs/` is ignored). Before the PR, archive the run to
Downloads/shared evidence storage, verify the copy, and add a compact evidence
page under docs/evidence with the measured results and archive checksums. Retain
failures and limitations. Do not mark TEST-01 complete from unit tests alone.
