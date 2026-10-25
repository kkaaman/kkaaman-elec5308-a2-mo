# Lane departure monitor and HUD — RV-06

Owner: Abdullah (E3). Status: implemented offline with synthetic-curve tests;
live overlay verification on recorded Town10HD curves is pending (see
`TRIAL_PLAN.md`, RV-06 row).

## What changed

In A1, `perception.lane_path.extract_ego_path` fed
`lane_to_waypoints.build_local_route`, which rebuilt the controller's route every
cycle from the detected lane. That path is not used by the A2 baseline: the
controller follows the frozen map route (`planning.baseline_mission`) and the
S0 runner records the signed cross-track error against it
(`carla_adapter.s0_runner.signed_cross_track`, retained as `signed_cte_m`).

`visualisation.lane_departure` repurposes the lane estimate as an **observer**:

```
lane mask -> extract_ego_path()["centerline"]  (pixels)
          -> lane_estimate_from_centerline(...)  (metres, radians, confidence)
          -> LaneDepartureMonitor.update(estimate, map_cte_m, now_sim_time_s=...)
          -> DepartureReport  ->  HUD overlay / run log columns
```

The monitor has no route, waypoint or command interface and
`visualisation` never imports the simulator or the controller. The acceptance
clause "monitor no longer rebuilds controller route" is therefore structural:
`lane_to_waypoints` remains only as a legacy shim for the A1 demos.

## Sign convention and units

Ego coordinates are +X forward, +Y left (`docs/interfaces.md`). All three
quantities are **positive when the rear axle is left** of the reference:

| Quantity | Reference | Source |
|---|---|---|
| `image_offset_m` | detected lane centre | `lane_estimate_from_centerline` (quadratic ground fit evaluated at x = 0, negated) |
| `map_cte_m` | frozen route polyline | `signed_cross_track` / `samples.csv` `signed_cte_m` |
| `difference_m` | `image_offset_m - map_cte_m` | monitor |

`image_heading_rad` is positive when the nose points left of the lane
tangent. `confidence` is dimensionless in [0, 1]: point support divided by
`expected_points`, damped by the ground-fit RMS residual
(`1 / (1 + rms / residual_scale_m)`).

The unit test `tests/unit/test_lane_departure.py` builds a left-hand bend
(R = 40 m) as both an image centreline (projected through the A2 camera
calibration) and a map route, displaces the ego 0.4 m left and right, and
checks that the image offset and the map CTE carry the same sign and agree to
within 0.15 m. Straight-lane and yaw cases check magnitude and heading sign.

## Validity rules

`MonitorConfig` defaults: `max_age_s = 0.5`, `min_confidence = 0.3`,
`max_abs_offset_m = 2.5`, `max_abs_heading_rad = 30 deg`, `disagreement_m = 0.5`.

| Reason | Condition |
|---|---|
| `no_estimate` | lane extractor returned `None` / fewer than `min_points` ground points |
| `stale` | `now_sim_time_s - estimate.sim_time_s` outside `[0, max_age_s]` |
| `low_confidence` | `confidence < min_confidence` |
| `outside_range` | offset or heading beyond the configured plausible range |
| `no_map_cte` | no finite route CTE supplied |

A report with any reason is `invalid` and has `difference_m = None`; a valid
report is `agree` or `disagree` (`|difference| > disagreement_m`). Disagreement
is **reported, not acted on**. `DepartureReport.as_record()` gives the
`lane_*` columns to append to a run's samples; the S0 runner is unchanged by
this task and does not yet write them.

## HUD hook

`visualisation.hud.departure_hud_lines(report)` returns the text lines
(status, confidence, offsets in metres, heading in degrees, reasons) and
`draw_departure_overlay(image_rgb, report)` draws a panel plus a signed offset
bar (filled = image offset, tick = map CTE, left of centre = positive) on a copy
of a nonempty `HxWx3` uint8 RGB frame. OpenCV is imported lazily so the
`visualisation` package imports without it. Rendering cost should be measured
as a `hud_s` stage with `scripts/profile_loop_timing.py` (TEST-22).

## Limits

- The ground projection assumes a flat local road; slopes and junctions bias
  the image offset. Camera 1.5 m forward / 2.0 m up from the actor origin is
  taken from `CameraRig`; the rear-axle offset belongs in `camera_to_ego`.
- `extract_ego_path` uses fixed pixel thresholds inherited from A1; sharp
  curves, missing markings and other resolutions are not validated.
- No recorded curve fixture with a logged image-derived offset exists in the
  repository (TEST-01 logged the A1 controller's path distance, not a lane
  offset), so the sign check is synthetic until the live overlay run below.
- Live acceptance for RV-06 is the recorded-curve overlay in `TRIAL_PLAN.md`;
  the offline tests only establish the convention and validity logic.
