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
| `profile_loop_timing.py` | Loop timing profile: per-stage and capture-to-apply p50/p95, real-time factor, headless vs HUD gate | TEST-22 (Abdullah), TEST-17 (Mohammad) |
| `evaluate_scenario_runs.py` | S3/S4/S5 reporting from persisted runs (false stops, slowdown, HOLD, resume, envelope) | TEST-05, TEST-07 |
| `evaluate_weather.py` | Recall and range across weather | E1-03 (Ricky) |
| `latency_speed_test.py` | Local, LAN and cloud inference latency | E4-07 (Saviour) |

## Lane topology (BL-A-02)

`export_lane_graph.py` exports real lane topology, validates JSON without CARLA,
and verifies saved geometry/connectivity against a live map or local OpenDRIVE
file. Its `audit` command checks raw OpenDRIVE links/geometry independently,
without CARLA. See [lane-topology](../docs/lane-topology.md) for commands and limitations.
