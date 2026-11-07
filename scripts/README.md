# scripts/

Command-line entry points. Run them from the repository root after
`pip install -e .`, for example `python scripts/run_scenario.py --help`.

**Rules:**
- Keep scripts thin. Parse arguments, build objects from the packages in
  `src/`, and call them. Logic belongs in a package, where it can be tested.
- Import by package name (`from planning.route_manager import ...`). Never use
  `sys.path` tricks, and never `cd src`.
- Find files with `autonomy.paths` (`config_path`, `model_path`, `run_dir`).
  Don't use relative paths like `"../x.pth"`.
- CARLA objects come from `carla_adapter` only.
- Write outputs under `runs/<scenario>/<run-id>/`, which git ignores. Copy
  curated evidence into `docs/` instead.

| Script | Purpose | Task (owner) |
|---|---|---|
| `run_scenario.py` | Run one scenario on CARLA or the harness | BL-E-04 (Abdullah) |
| `run_acceptance_suite.py` | One-command S0–S5 regression with N runs each | BL-F-07 (Saviour) |
| `calibrate_vehicle.py` | Steering and throttle/brake response measurements | BL-E-02/03 (Abdullah) |
| `capture_frames.py` | Aligned RGB-D capture with ground truth | BL-B-03 (Mohammad) |
| `evaluate_depth.py` | Depth convention and range accuracy | BL-B-04 (Mohammad) |
| `evaluate_runs.py` | Aggregate run metrics | BL-F-05 (Saviour) |
| `benchmark_detector.py` | Detection and confirmation range offline | BL-C-05 (Ricky) |
| `benchmark_latency.py` | Loop timing and real-time factor | TEST-03 (Abdullah), TEST-17 (Mohammad) |
| `evaluate_weather.py` | Recall and range across weather | E1-03 (Ricky) |
| `latency_speed_test.py` | Local, LAN and cloud inference latency | E4-07 (Saviour) |
| `capture_lidar_sweeps.py` | Vehicle-frame LiDAR sweeps around a known target, with transform checks | E2-02 (Mohammad) |
| `replay_lidar_obstacles.py` | LiDAR ground removal, clustering, tracking and fusion replay to JSON | E2-03/04 (Mohammad) |

## Lane topology (BL-A-02)

`export_lane_graph.py` exports real lane topology, validates JSON without CARLA,
and verifies saved geometry/connectivity against a live map or local OpenDRIVE
file. Its `audit` command checks raw OpenDRIVE links/geometry independently,
without CARLA. See [lane-topology](../docs/lane-topology.md) for commands and limitations.

## Mapless extension (Phase 0)

See [mapless_autonomy](../docs/mapless_autonomy.md).

| Script | Purpose |
|---|---|
| `check_sensor_rig.py` | Live calibration gate for the mapless rig: conversions, signs, extrinsics, coverage sweep, camera mask and reprojection, radar range and range rate, timing. `--set sensor.field=value` for rig experiments |
| `make_satnav_route.py` | Sat-nav route (GPS points and commands) for a saved mission, from the live map |
| `capture_mapless_dataset.py` | Training-data capture on Town01–Town06 (refuses Town07, the held-out test town, and Town10HD) |
| `inspect_mapless_capture.py` | Inspection of a capture run: conditions, class shares, small pedestrians, LiDAR-to-label projection and RGB-to-label pairing checks per camera, integrity, encounters, a verdict, contact sheets (writes `<run>/inspection/`) |
| `repair_mapless_capture.py` | Repairs schema-1 capture runs in place: truth box centres from each blueprint's bounding-box offset (read from a CARLA server once, exportable) |
| `run_mapless_drive.py` | G1b live run: the mapless car drives a straight, a bend, or stops for a parked car, from sensors and a sat-nav route only; scored from truth |

New PC setup (Windows, no admin rights): `powershell -ExecutionPolicy Bypass -File scripts\setup_group7_pc.ps1`.

## Wayfinder (E3-01..03)

`wayfinder_replay.py` plans a route between two named destinations from
`configs/wayfinder/pois_town10hd.json` and replays poses along it offline,
writing the route document, a guidance log (remaining distance, next turn,
ETA) and mini-map PNG frames. No simulator is used. See
[wayfinder](../docs/wayfinder.md).
## Documentation and offline preflight (DOC-03 / DOC-05)

- `python scripts/check_reproduction.py --assets-only`: verify the three model
  binaries with the standard library before importing optional ML packages.
- `python scripts/check_reproduction.py`: also compare the recorded runtime subset.
- `python scripts/generate_module_notes.py`: regenerate the source inventory;
  use `--check` in CI to reject stale documentation.

These preparation/documentation utilities deliberately use only the standard
library so they can diagnose an environment before the project is installed.
They locate the source tree using their own file path. See
[reproduction](../docs/reproduction.md) for disconnected installation and the
read-only `python -m carla_adapter.environment_check` live prerequisite check.
