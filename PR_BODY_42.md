feat(RV-06, TEST-22): lane departure monitor + HUD hook, loop-timing profiler, S5 envelope reporting

Closes #42 once the live evidence is attached (TEST-07 S5 trials, TEST-10 rerun,
TEST-22 headless/HUD profile, RV-06 recorded-curve overlay). Until then this PR
delivers the offline code, tests and the exact trial plan; no acceptance is
claimed by this description.

## What it adds

- `visualisation.lane_departure`: `LaneEstimate`, `MonitorConfig`,
  `LaneDepartureMonitor`, `DepartureReport` and
  `lane_estimate_from_centerline` (image centreline pixels -> ground-plane
  quadratic fit -> lateral offset m / heading rad / confidence). Compares the
  image offset with the map-based signed CTE, reports the difference, a
  confidence and a validity flag (`no_estimate`, `stale`, `low_confidence`,
  `outside_range`, `no_map_cte`). Positive-left convention shared with
  `signed_cross_track` and `a1_comparison`. No route, waypoint or command
  interface; the A1 `lane_to_waypoints` route rebuild is not used.
- `visualisation.hud`: `departure_hud_lines` (pure text) and
  `draw_departure_overlay` (panel + signed offset bar; lazy `cv2`; input not mutated).
- `evaluation.loop_timing` + `scripts/profile_loop_timing.py`: per-stage and
  capture-to-apply p50/p95/max from S0 `samples.csv`, warm-model `timing.jsonl`
  or `Diagnostics`; real-time factor and frame age; labelled comparison
  (headless vs HUD) with an explicit `gate` block (`ADMIT` only on p95 <= 100 ms
  with >= 20 samples, else `BLOCKED` with reason). Writes
  `latency_profile.json` + `latency_profile.png`.
- `evaluation.scenario_metrics` + `scripts/evaluate_scenario_runs.py` (shared
  with #45): S5 rows (stop margins, HOLD speed, resume delay) and the new
  report-only `speed_envelope` record (speed and truth gap at first
  confirmation vs `v^2/(2a) + stop_margin` with the frozen deceleration).
- `perception.cadence.RateLimitedWorkload` (shared with #45) for rate-limiting
  optional workloads such as the lane monitor without reusing stale results.
- Docs: `docs/lane_departure_monitor.md`, `docs/loop_timing.md` (toctree +
  API), `TRIAL_PLAN.md` with the PowerShell commands for every live set.

## Criteria -> evidence

| Criterion | Evidence | State |
|---|---|---|
| RV-06: monitor no longer rebuilds controller route | `visualisation.lane_departure` has no route/command interface; `tests/unit/test_lane_departure.py::test_disagreement_is_reported_not_acted_on`; controller untouched (`navigation_control`, `planning` unchanged) | Offline, in this PR |
| RV-06: correct sign on curves | Synthetic R = 40 m left-hand bend, ego +/-0.4 m: image offset and map CTE agree in sign and within 0.15 m (`test_recorded_style_curve_agrees_with_map_cte_sign_convention`) | Offline, in this PR |
| RV-06: overlay verified on recorded curves | `TRIAL_PLAN.md` section 3 -> `runs/RV-06/overlay/monitor.json` + overlay PNGs | Live, pending |
| RV-06: HUD shows confidence and validity | `departure_hud_lines` / `draw_departure_overlay`; `test_hud_lines_and_overlay_do_not_mutate_input` | Offline, in this PR |
| TEST-22: headless vs HUD with admitted models, capture-to-apply latency recorded | `scripts/profile_loop_timing.py`; `tests/unit/test_loop_timing.py` on synthetic diagnostics, S0 CSV and `timing.jsonl` | Offline, in this PR |
| TEST-22: final config p95 <= 100 ms or gate explicitly blocked | `TRIAL_PLAN.md` section 2 -> `runs/TEST-22/profile-<id>/latency_profile.json` `comparison.gate` | Live, pending |
| TEST-07: five S5 bend-obstacle trials | `TRIAL_PLAN.md` section 5 (blocked on the scenario runner); evaluation path tested in `tests/unit/test_scenario_metrics.py::test_s5_resume_delay_and_speed_envelope` | Live, blocked |
| TEST-10: rerun on candidate hash | `TRIAL_PLAN.md` section 1 -> `runs/S0/rerun-<hash>-<id>/summary.json` | Live, pending |

## Offline checks (this branch, Linux dev box)

- `pytest tests -q -p no:cacheprovider`: 857 passed, 1 skipped, 3 subtests
  (baseline on `d1edba6`: 832 passed; +25 new tests in `test_lane_departure`,
  `test_cadence`, `test_loop_timing`, `test_scenario_metrics`).
- `ruff check .`: clean.
- `scripts/check_simulator_boundary.py --smoke`: AST boundary passed; core
  modules imported with simulator imports blocked (85 modules, up from 80).
- `sphinx-build -W -q -b html docs`: clean; new pages in the toctree.

## What remains live

Sections 1, 2 and 3 of `TRIAL_PLAN.md` run today on the lab PC. Section 5 (S5)
is blocked: main has no scenario runner with obstacle actors, BL-D-08 state
machine or evaluation-truth gap stream; the evaluation side is ready and the
fields it consumes are listed in `evaluation.scenario_metrics`. No frozen
threshold, config or baseline behaviour is changed; `scripts/run_scenario.py`
and `carla_adapter.s0_runner` are untouched.
