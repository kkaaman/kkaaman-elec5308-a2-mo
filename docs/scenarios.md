# Deterministic obstacle scenarios and evaluation truth — BL-B-06 / BL-B-07

Issue #39. Owner: @kkaaman. Reviewer: @at0941. Status: implementation and
offline tests in this PR; live verification evidence pending. The issue starts
after G1 passes, so this PR stays Draft until the G1 decision.

This page defines how S1–S5 place and remove their obstacles, and how collisions,
lane invasions and stopping margins are measured. It does not run the S1–S5
acceptance suite: that needs the integrated perception-to-control path (BL-F-07,
issue #41), which will call these modules.

## Scenarios (BL-B-06)

`configs/scenarios/S1.json` … `S5.json`, loaded by `evaluation.scenarios.load_scenario`.

**Placement.** Every obstacle is placed by route arc length `route_s_m` and signed
lateral offset `lateral_m` (positive = left of travel) on the frozen mission route
(`configs/mission_baseline.json`, 169.8 m).

| ID | Obstacle | Route s | Lateral | Removal | Expected |
|---|---|---|---|---|---|
| S1 | car (`vehicle.tesla.model3`) | 40.0 m (road 20, before junction 841 at 49.2 m) | 0.0 m | after a 2.0 s steady HOLD | stop, hold, resume |
| S2 | adult pedestrian (`walker.pedestrian.0001`), facing across the road | 40.0 m | 0.0 m | after a 2.0 s steady HOLD | stop, hold, resume |
| S3 | car | 40.0 m | +3.5 m (adjacent lane) | never | no stop |
| S4 | car | 40.0 m | 0.0 m | never | HOLD for at least 60 s |
| S5 | car | 158.0 m (6.4 m after the junction 532 exit at 151.6 m) | 0.0 m | after a 2.0 s steady HOLD | stop, hold, resume |

**What makes a run reproducible:**
- The pose is a pure function of the route and the file (`placement_pose`). Each
  placement is snapped to the ground below the road. Physics is disabled, so
  contact cannot move the obstacle.
- Removal is a pure function of simulation time, true ego speed and the
  obstacle-in-path evidence (`RemovalSchedule`). The obstacle is removed on the
  first tick at which the ego has been at or below 0.1 m/s for 2.0 s of continuous
  simulation time while the obstacle is in its path (`obstacle_hold_evidence`:
  present, in the ego's corridor, within 15 m ahead). A dropped tick (gap above
  1.5 × dt), any motion above 0.1 m/s, or a tick without that evidence restarts the
  count, so the standstill at mission start or a fault stop short of the obstacle
  never removes it (review of PR #98).
- A removal counts only once CARLA's `destroy()` has succeeded; a failed
  destruction leaves the obstacle present (the truth keeps reporting it) and is
  recorded in `removal_failures`.
- Seeds 5308–5312 label repeatability runs. The scripted world draws no random
  numbers.
- Each file's SHA-256 (line-ending independent) is recorded with every placement.

`ScenarioObstacle.spawn` writes a placement record:
- requested route coordinates, core pose and CARLA transform;
- ground height, bounding box and footprint corners;
- map road/lane and whether it is a junction.

`remove` records the frame and simulation time of the scripted removal.

## Events (BL-B-07)

`ScenarioEventSensors` attaches CARLA's collision and lane-invasion sensors to the
ego and keeps every raw callback. `event_records` turns the raw callbacks into
records in the live S0 runner's schema, so `evaluation.s0_metrics` counts S1–S5
exactly as it counts S0 (checked in `tests/unit/test_scenario_events.py`).

| Event | One record per | Deduplication | Excluded from the count |
|---|---|---|---|
| Collision | sensor callback | Callbacks against the same other actor share a `contact_id` while consecutive callbacks are ≤ 0.5 s of simulation time apart (`collision_dedup_s`). A later touch is a new contact. A callback with no actor id is its own contact. | never |
| Lane invasion | sensor callback (unique `event_id`) | none | Only a crossing covered by a declared legal manoeuvre: an entry of the route's `legal_lane_invasion_exclusions` (`id`, `from_s_m`, `to_s_m`, `markings`, `reason`) covering its route s, for every crossed marking. Junctions are never exempted wholesale. The baseline route declares none. |

Raw callbacks are written for failed runs as well as successful ones. Counts are
always derived from them and never stored in their place.

## Stop-margin truth (BL-B-07)

`evaluation.stop_truth.route_surface_gap`, called per tick by
`carla_adapter.scenario_world.ego_truth`. The reference points match
`tests/fixtures/acceptance/golden.json`:

- **Ego:** the rear axle is projected onto the continuous route (`s_ego`). The
  front bumper is at `s_ego + front_offset_m`, where `front_offset_m` = rear-axle
  offset + bounding-box front face (3.78 m for the Tesla). It is applied exactly
  once.
- **Obstacle:** its bounding-box footprint is sampled every 5 cm along its edges.
  The surface is the smallest route s of a sample inside the evaluation corridor
  and ahead of the ego rear axle. The corridor is the ego half-width plus 0.15 m
  on each side.
- **Gap:** `truth_front_gap_m = surface_s − s_ego − front_offset_m`. It is None
  when no sample is in the corridor ahead.

`hold_metrics` derives the acceptance quantities from the logged rows:
- **HOLD:** starts once speed ≤ 0.1 m/s for 1.0 s with an obstacle in the path,
  and ends at the scripted removal (S4: when the obstacle leaves the path).
- **Stopped margin:** min and max gap over the HOLD.
- **HOLD speed:** maximum speed over the HOLD, so any creep counts.
- **Continuous HOLD (S4):** the longest run of stopped ticks with the obstacle in
  the path; motion, a missing tick or a row without obstacle evidence breaks the run.
- **Resume delay:** from removal to the first speed above 0.1 m/s.

A goal stop has no obstacle in the path, so it is never mistaken for a HOLD.
Missing evidence gives None, never zero.

## Truth stays out of the control path

- `tests/unit/test_evaluation_truth_isolation.py`:
  - no package in `autonomy`, `geometry`, `navigation_control`,
    `obstacle_perception`, `perception`, `planning`, `segmentation` or
    `traffic_perception` imports `evaluation`, `carla_adapter` or `sim_harness`,
    in any import form;
  - the only stack input is a `FrameBundle`, and none of its fields carries truth,
    events or scenario data.
- `scripts/check_simulator_boundary.py` independently keeps CARLA out of the core.
- In the live checks below, the scripted drives use the ego's own pose. Truth is
  only logged.

## Live verification

With CARLA 0.9.16 running Town10HD_Opt and no other clients:

```powershell
python scripts/verify_scenarios.py --output "$HOME\Downloads\Group7_BL-B-07_verification"
```

It writes `scenario_verification.json`, with every check's raw callbacks and
placement records. The status is `MEASURED_PASS` only if every check passes,
settings are restored and no actors leak.

| Check | Pass condition |
|---|---|
| Placement S1–S5 | Two clean spawns agree within 1 cm and 0.1°. The pose after a tick equals the requested pose. In the corridor for S1, S2, S4 and S5; outside it for S3. |
| Truth gap | Car at 3, 6 and 10 m and pedestrian at 3 and 6 m, placed face-to-face along the straight start: truth within 5 cm of the face-to-face route distance. Car 0.8 m left: still in the corridor, same tolerance. Car 3.5 m left: not in the corridor. |
| Collision deduplication | The ego touches a stationary car, backs off 1.5 m, waits 1.5 s and touches again: exactly two contacts, both against that car. |
| Lane invasion | `PathFollower` drives a smooth lane change across a broken marking on a straight two-lane road. The crossing is reported and counted, and a declared legal manoeuvre covering it removes it from the count. |

These are static-scene checks of the scoring path. They are not S1–S5 acceptance.

## Offline tests

| File | Tests |
|---|---|
| `tests/unit/test_scenarios.py` | Scenario schema, placements on the real route, determinism and removal timing |
| `tests/unit/test_scenario_events.py` | Contact rule, agreement with the live S0 runner's contact ids, legal exclusions and S0-metric counting |
| `tests/unit/test_stop_truth.py` | Golden gap and lateral sign, partial overlap, HOLD/resume metrics, missing evidence |
| `tests/unit/test_scenario_world.py` | Placement, footprint and truth through a fake left-handed `carla`; raw event capture; the live check's known-distance and repeatability arithmetic |
| `tests/unit/test_evaluation_truth_isolation.py` | Static isolation of the control path |
