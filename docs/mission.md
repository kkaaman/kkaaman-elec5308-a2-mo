# Controller audit and mission design (BL-A-01)

A2 keeps the A1 path follower (`navigation_control.PathFollower`: Pure
Pursuit steering and a speed PID). It repairs how the follower is *used*,
not the algorithm. This page records the controller's contract and what the
A1 demo got wrong. It also sets out the rules and format for the fixed
mission route. The concrete route is fixed on the map in BL-A-04.

## 1. Controller contract (unchanged public API)

| Item | Contract |
|---|---|
| Construction | `PathFollower(ControllerConfig(...))`. One instance per vehicle and episode. Not thread-safe |
| State input | `VehicleState(x_m, y_m, yaw_rad, speed_mps)`: **rear-axle** position in the world frame (`interfaces.md`). Yaw is counter-clockwise from +X. **Speed must be ≥ 0** |
| Command output | `ControlCommand(steering_angle_rad, throttle, brake)`: front-wheel-equivalent angle, **positive = left**, clamped to `ControllerConfig.max_steering_angle_rad`. Throttle and brake in [0, 1], never both positive |
| `set_path(waypoints)` | Copies an open polyline (at least 2 distinct points, not closed) and **resets** progress, completion and the PID history. A rejected path leaves the old one intact |
| `reset()` | Clears progress, completion and PID history and keeps the path |
| `update(state, target_speed_mps, dt_s)` | `target_speed_mps` ≥ 0 is the caller's ceiling. `dt_s` > 0 is the control period. Raises `RuntimeError` with no path, and `ValueError` for invalid input or a target behind the car |
| Progress | Arc length of the nearest point on the **whole** path, never decreasing (`max(previous, projection)`) |
| Endpoint braking | Internal ceiling `sqrt(2·a·max(0, remaining − margin))` with `a = stopping_deceleration_mps2` and `margin = min(goal_tolerance/2, 0.1·length)`. The car slows into the real destination by itself |
| Completion | Latched FINISHED once progress is near the end (≥ max(½·length, length − tolerance)), the goal is within tolerance, and speed ≤ `stopped_speed_mps`. Holds the brake until `reset()` or `set_path()` |
| Temporary stop | `target_speed_mps = 0` away from the goal: status STOPPING, brake ≥ `hold_brake`, PID integral cleared, steering still follows curves while moving. A later positive target resumes. It does **not** latch FINISHED (A1 tests `test_temporary_stop_*`) |
| Exceptions | The caller must stop the actuators. The library never does |

Steering limits:
- `ControllerConfig.max_steering_angle_rad` is the **controller's** cap.
- The vehicle's **physical** limit is `VehicleGeometry.max_steer_rad`,
  measured in BL-E-01.
- The CARLA adapter normalises by the physical limit:
  `steer = −angle / max_steer_rad`, clamped to [−1, 1]. It never normalises
  by the controller cap.

## 2. What the A1 integration got wrong, and the A2 rule

| A1 demo (`src/demo_fused_lane_following.py`) | Effect | A2 rule | Task |
|---|---|---|---|
| Calls `set_path` with a fresh local route every camera cycle | Progress and the PID integral reset every frame, leaving proportional-only speed control. Pinned by `test_setting_the_path_every_cycle_discards_speed_integral` | `set_path` **once per mission**. Replace the path only for an explicit mission change or reroute | BL-A-04 |
| Wall-clock `dt` clamped to [0.02, 0.5] s with `time.sleep` | Control period does not match the simulated step | Synchronous mode. `dt_s` = the fixed simulator step | BL-B-01, BL-A-05 |
| `rear_axle_offset_m` left at 0 | The controlled point is the actor origin, not the rear axle | Rear-axle offset measured from wheel positions | BL-E-01 |
| Wheelbase 2.5 m and steering limit 0.6 rad used as both controller cap and normaliser | Steering gain wrong by the ratio of the physical limit to 0.6 rad | Measured wheelbase, with physical limit and controller cap kept separate | BL-E-01, BL-E-02 |
| Reuses the last command or creeps when perception misses | Can keep moving without valid input | Explicit fault stop with `ControlPacket.stop()`, never the last moving command | BL-F-02 |
| `turn_decision.build_turn_route` joins the pre-junction waypoint straight to the chosen exit, choosing from the first fan-out only | A chord across the junction can cut corners or curbs | `JunctionPlanner` follows each candidate through the junction and stitches real lane-centre samples | BL-A-03 |
| `navigation_control` docstring points to a missing `docs/control.md` | Broken reference | Now points here | (this task) |

## 3. Mission route rules

The baseline drives **one fixed route**, set once at mission start.

1. **Open and non-self-crossing.** Non-adjacent parts of the route stay at
   least **10 m** apart. The controller projects onto the nearest segment of
   the whole path. A route that doubles back nearby can make progress jump
   ahead, and the look-ahead target then lies behind the car, so `update()`
   raises. Pinned by
   `test_progress_jumps_to_a_nearby_later_leg_and_the_controller_fails`. The
   stack still turns any controller exception into an emergency stop.
2. **Legal lanes only.** Lane centre-lines are taken from the map's lane
   graph, through junctions as well. There are no lane changes and no chords
   across junctions.
3. **Contents.** At least one straight section, one left turn and one right
   turn. Length 100–250 m, short enough for 30+ baseline runs and the demo.
4. **Endings.**
   - Start on a straight lane at least 20 m before the first junction.
   - End on a straight segment with at least 30 m of straight approach, so
     the endpoint cap brakes smoothly.
5. **S5 bend.** One bend with a known sight distance, long enough for the
   stopping budget (`depth-association.md`, §7).
6. **Environment.**
   - Traffic lights on the route are held green by the scenario.
   - No background traffic.
   - Town10HD only, recorded with the map name and lane-graph hash.
7. **Speed.**
   - Cruise candidate 5 m/s, lowered at G1 if the measured stopping and
     detection range demand it.
   - Curve speed is capped by `v_curve = sqrt(a_lat_max / |κ|)` with a
     look-ahead envelope (BL-D-07).

## 4. Tentative manoeuvre list

The route is fixed on the map, and the coordinates filled in, by BL-A-04
after a visual check in Town10HD.

| # | Section | Manoeuvre |
|---|---|---|
| 1 | Start on a straight, ≥ 20 m before junction A | – |
| 2 | Junction A | `right` |
| 3 | Straight section, S1/S2 obstacle placement zone | – |
| 4 | Junction B | `straight` |
| 5 | Junction C (bend/turn used by S5) | `left` |
| 6 | Straight finish, ≥ 30 m | – |

The list goes in `Mission.maneuvers` as `[RIGHT, STRAIGHT, LEFT]`. If the map
does not offer this sequence on a short open route, BL-A-04 records the
substitute and why in `decisions.md`.

## 5. Mission file format

`configs/scenarios/S0.yaml`, written by BL-A-04 and loaded with
`autonomy.paths.config_path("scenarios/S0")`:

```yaml
name: S0
map: Town10HD_Opt
lane_graph_hash: "<sha256 of the exported LaneGraph JSON>"
start:                     # world frame (interfaces.md), rear axle
  x_m: 0.0
  y_m: 0.0
  yaw_rad: 0.0
maneuvers: [right, straight, left]
destination_xy_m: [0.0, 0.0]
cruise_speed_mps: 5.0
traffic_lights: hold_green
background_traffic: false
obstacles: []              # S1–S5 add entries (BL-B-06)
```

## 6. How the stack uses the controller

```python
follower = PathFollower(controller_config)        # from measured geometry
follower.set_path(route_manager.mission_path())   # once
while running:
    state = bundle.ego.to_vehicle_state()          # rear axle, forward-only
    target = governor.target_speed(...)            # min(cruise, curve, obstacle) ≥ 0
    try:
        result = follower.update(state, target, dt_s=sim_step_s)
        packet = ControlPacket(bundle.frame_id, result.command, reason=behaviour.state)
    except (ValueError, RuntimeError) as exc:
        packet = ControlPacket.stop(bundle.frame_id, reason=f"controller: {exc}")
```

- Temporary obstacle stops pass `target = 0` to the same follower. The
  path, progress and completion are untouched.
- Only the controller's endpoint cap ends the mission.
- BL-A-06 verifies this with the full stack.
