docs(RV-08), feat(TEST-17): sensor rig/segmentation lifecycle docs, docstring units and validity, rate-limited optional workloads, S3/S4 reporting

Closes #45 once the live evidence is attached (TEST-05 S3 x5 / S4 x5 and the
TEST-17 warm-rig profile). This PR delivers the Sphinx pages, docstrings,
cadence control, scenario reporting and the exact trial plan; no scenario
acceptance is claimed by this description.

## What it adds

- `docs/sensor_rig.md` (toctree): the rig as implemented (640 x 360 / 90 deg,
  1.5 m / 2.0 m actor-origin mount, `sensor_tick = 0` at 20 Hz, S0's 320 x 180
  observation camera, collision/lane-invasion streams, LiDAR not implemented),
  exact-frame matching rules of `FrameJoiner` and the S0 `RGBReceiptQueue`,
  calibration scripts and current records (axial convention, 107 samples,
  ~0.050 m), weather limits (ClearNoon acceptance only; `configs/weather/`
  empty; wet/fog/night unmeasured), the LaneUNet vs BiSeNetV2 trade-off with
  `KEEP_OFF` masking, cadence/stale-mask rules, lifecycle and a units/validity
  table.
- Docstrings with units and validity rules on the public sensor and
  segmentation API: `carla_adapter.conversions` (`bgra_array`, `decode_rgb`,
  `decode_depth`, `axial_depth`), `segmentation.segmentation` /
  `preprocessing`, `perception.lane_inference.LaneSegmenter`,
  `perception.lane_path.extract_ego_path`. Behaviour unchanged.
  `carla_adapter/sensors.py` is hash-audited by the BL-D-06 freeze review
  (`tests/unit/test_baseline_freeze_review.py` fails on any byte change), so
  its public API units/validity are tabulated in `docs/sensor_rig.md` instead of
  edited in place; re-audit is a separate freeze task.
- `perception.cadence.RateLimitedWorkload` / `CadenceResult`: run an optional
  workload every `period_ticks`; a result older than `max_age_ticks` is
  reported `valid = False` with `value = None` (kept only as `stale_value` for
  labelling) and a failed computation discards the cache. Per-call wall cost is
  recorded for profiling. The A1 demos' cached mask is documented as the
  behaviour this replaces; the demos are not modified.
- `evaluation.loop_timing` + `scripts/profile_loop_timing.py` (shared with
  #42): per-sensor/stage p50/p95 cost, end-to-end p95 and real-time factor from
  `timing.jsonl` / S0 samples / optional-workload JSON; any `<name>_s` column
  (e.g. `segmentation_s`, `lane_monitor_s`, `lidar_s`) is picked up.
- `evaluation.scenario_metrics` + `scripts/evaluate_scenario_runs.py`: S3
  `false_stops` and `cruise_speed_drop_mps` vs a matched S0 run; S4
  `hold_duration_s`, `hold_max_speed_mps`, `stop_margin_min/max_m` with route
  completion `NOT_APPLICABLE`; limits read from
  `tests/fixtures/acceptance_definitions.json`, missing evidence is
  `INSUFFICIENT_EVIDENCE`. Shared S0 rows reused via `evaluate_s0_run`.
- `TRIAL_PLAN.md`, `docs/loop_timing.md`, `scripts/README.md` rows.

## Criteria -> evidence

| Criterion | Evidence | State |
|---|---|---|
| RV-08: sensor/segmentation Sphinx pages build | `docs/sensor_rig.md`, `docs/loop_timing.md`, `docs/lane_departure_monitor.md` in `docs/index.rst`; `sphinx-build -W` clean | Offline, in this PR |
| RV-08: frame matching, weather limits, calibration, segmentation cadence documented | `docs/sensor_rig.md` sections "Frame matching", "Weather limits", "Calibration", "Segmentation trade-off and cadence", "Lifecycle" | Offline, in this PR |
| RV-08: every new public function documented with units and validity | Docstrings listed above; `api.rst` automodule entries for `evaluation.loop_timing`, `evaluation.scenario_metrics`, `perception.cadence`, `visualisation.*` build without warnings | Offline, in this PR |
| TEST-05: S3 false stops and slowdown vs frozen threshold reported | `evaluate_scenario_run("S3", ...)`; `tests/unit/test_scenario_metrics.py::test_s3_reports_false_stops_and_slowdown_against_matched_s0` | Offline, in this PR |
| TEST-05: S4 HOLD duration with route completion N/A | `evaluate_scenario_run("S4", ...)`; `test_s4_hold_duration_with_route_completion_not_applicable` | Offline, in this PR |
| TEST-05: five S3 + five S4 trials | `TRIAL_PLAN.md` section 4 (blocked on the scenario runner) | Live, blocked |
| TEST-17: RGB-D, monitor, segmentation, LiDAR costs with warm models | Section 2(c)/6 of `TRIAL_PLAN.md`; profiler tested on `timing.jsonl` layout (`test_load_run_folder_and_timing_jsonl`) | Live, pending (LiDAR N/A) |
| TEST-17: real-time factor and p95 E2E reported | `profile_rows` -> `real_time_factor`, `end_to_end.p95_ms` | Offline, in this PR; live values pending |
| TEST-17: optional workloads rate-limited without misusing stale masks | `RateLimitedWorkload`; `tests/unit/test_cadence.py::test_stale_result_is_invalid_and_not_returned_as_current` | Offline, in this PR |

## Offline checks (this branch, Linux dev box)

- `pytest tests -q -p no:cacheprovider`: 857 passed, 1 skipped, 3 subtests
  (baseline on `d1edba6`: 832 passed; +25 new tests in `test_lane_departure`,
  `test_cadence`, `test_loop_timing`, `test_scenario_metrics`).
- `ruff check .`: clean.
- `scripts/check_simulator_boundary.py --smoke`: passed (85 core modules).
- `sphinx-build -W -q -b html docs`: clean.

## What remains live

TEST-05 trials need the S3/S4 scenario runner (obstacle actors, BL-D-08 HOLD
state machine, evaluation-truth `bumper_gap_m` and `obstacle_*` events) that is
not on main; the evaluation command is fixed and tested. TEST-17 warm-rig
numbers come from `measure_operational_governance.py` + the optional-workload
measurement in section 6. No frozen threshold, config or baseline behaviour is
changed; the S0 runner and legacy demos are untouched.
