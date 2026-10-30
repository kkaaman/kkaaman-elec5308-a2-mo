# Mapless autonomy: rig, mapless driver, turns and the G2/G3 stack

Branch `feature/mapless-autonomy`, built from `feature/shadow-localisation`. This is an
extension: the frozen baseline is unchanged. The sections below the status table are
the working record, oldest first; the table is the current state.

## Current status (27 Oct)

"Implemented" means the code exists; "offline" that it passes the harness and replay
tests; "live" that it has passed on CARLA at least once; "repeated" that it has passed on
more than one site or run without retuning.

| Gate | Scope | Implemented | Offline | Live | Repeated | Open |
|---|---|---|---|---|---|---|
| G0 | Rig v2, health, calibration gate | yes | yes | MEASURED_PASS, both PCs | yes | — |
| G1a | Local trajectory controller | yes | yes | yes (inside G1b) | yes | — |
| G1b | Straight, curve, obstacle stop at 2 m/s, sensor-only | yes | yes | MEASURED_PASS (Town10HD) | straight at 4 m/s; stack runs; stop and curve again on the Stage 1 code | railing-kerb calibration (Town05) fixed offline, not yet live; Stage 1 stopping cases (holds, clearance, low objects, bends) offline only |
| G1c | Commanded 90° turn | yes | yes | right (Town10HD) and left (Town03) MEASURED_PASS | no | Stage 3 (oncoming gate, standstill at the mouth, medians, exact arcs, evaluator outline) offline only; far exit-kerb bias (up to 1 m on the synthetic verge); skewed junctions unsupported (sites now within 10° of a right angle); exit-line uncertainty not reported |
| G2 | Segmenter, painting, road model, stack | yes | yes | stack straight and stop MEASURED_PASS (Town10HD site 1) | no | segmentation does not yet place the lane; S0–S5 × 5 not run; segmenter retrained on corrected labels (`road_seg_b0_v2.pt`, mIoU 0.879), the old checkpoint refused by the loader |
| G3 | Tracker, radar, prediction, following, M04–M06 | yes | yes | braking lead, cut-in, crossing pedestrian MEASURED_PASS (Town10HD site 1), again on `6b1f705` with no false yields; braking lead at 4 m/s MEASURED_PASS with and without radar (Town05 site 0) | no | A/B/C ablation with a lead still moving beyond LiDAR range (in A6/B6 it had stopped before either sensor saw it: first tracked at 30.5 m LiDAR-only, 34.5 m with radar, both static, so nothing was followed); the radar adds ~45 ms of tracking (155 tracks against 61); grades are outside the terrain model's scope; split LiDAR clusters and a separate radar track per parked car; a car within the tight link (0.3 m + 1 cm/m) of a long structure stays part of it and is left to the LiDAR obstacle stop; Stage 2 identity fixes (radar association, twins, lead body, reachability, structure split) offline only |
| G4–G6 | Urban behaviours, learning, final report | no | — | — | — | — |

Stack compute on the lab PCs (27 Oct, `e401188`, v2 segmenter): 58–62 ms p50 and about
215 ms p95 per 50 ms tick on Town10HD, 140 ms p50 and 295–310 ms p95 on Town05. The
segmentation is 56 ms p50 and 150–180 ms p95 on the ticks it runs; clustering is 19 ms on
Town10HD and 78 ms on Town05. The 50 ms budget is not met.

## Goal and label

The car drives with **no simulator truth and no map**. It sees only its sensors, and it
is given a sat-nav route: sparse GPS points, each with a command. The rules are inspired
by the CARLA Leaderboard 2.0 **SENSORS track**. It is not "Leaderboard-compliant" until
the official wrapper and evaluator run on this CARLA version.

Report label: a mapless autonomous-driving research prototype, evaluated within a
declared and tested operating domain. No SAE level is claimed. The working plan is
`claude/plans/mapless_autonomy_plan.md` (v2.1) in the project.

## What the car receives (`src/mapless`)

| Module | Contents |
|---|---|
| `sensor_suite` | Sensor-suite schema and the SENSORS-track limits. Ground-truth sensor types are refused with the reason. |
| `frame` | `SensorFrame`: cameras, LiDAR sweeps, radar returns, GNSS fix, IMU, speedometer. No pose, route, map or labels. |
| `geometry` | Mount transforms, camera calibration, compass-to-heading conversion. |
| `satnav` | `SatNavRoute` (GPS points and commands), the car's own tangent-plane projection, `RouteProgress`. |

**Mountable sensors**, with limits:

| Type | Limit |
|---|---|
| RGB camera | 8 |
| Ray-cast LiDAR | 2 |
| Radar | 4 |
| GNSS | 1 |
| IMU | 1 |
| Speedometer | 1 |

The speedometer stands for a wheel encoder: true speed with a seeded scale error and
noise added.

**Refused:**
- the depth camera and segmentation cameras;
- semantic LiDAR, the obstacle detector, optical flow and normals;
- collision and lane-invasion sensors (evaluator only).

**Rig v1** (`configs/sensor_suites/mapless_v1.json`):

| Sensor | Setup |
|---|---|
| Front camera | 1280×720, 90° |
| Front-left and front-right cameras | ±60°, 100°, 640×360 |
| Rear camera | 100°, 640×360 |
| Roof LiDAR | 32 channels, 20 Hz (one full sweep per 0.05 s tick), 70 m, CARLA noise and drop-off |
| Radars | Front (30°, 100 m) and two rear corners (90°, 60 m) |
| GNSS | Consumer: 5 Hz, 0.5 m white noise plus a seeded constant offset (1σ = 1 m) |
| IMU | Gyro, accelerometer, compass |
| Speedometer | Wheel-encoder model |

**Firewall** (`tests/unit/test_mapless_firewall.py`):
- `src/mapless` may import only the standard library, numerical packages and
  simulator-free core packages, checked both directly and transitively.
- No string in it names a town, mission, lane graph or OpenDRIVE file.
- `SensorFrame` has exactly the reviewed fields.

## Sat-nav route

**Generation (evaluator side).** `evaluation/satnav_route.py` turns a mapped mission into
route points:
- start;
- each junction entry, carrying LEFT, RIGHT or STRAIGHT;
- each junction exit, carrying LANEFOLLOW;
- fill points, so that no gap is longer than 50 m;
- goal.

This is the phone sat-nav's job; the car never imports it.

**Live export.** `scripts/make_satnav_route.py` writes the file from the live map. It
also confirms that `carla_geolocation`, a pure copy of CARLA's geolocation formula,
agrees with the server.

**On the car.** `RouteProgress` reads the car's own position estimate and reports:
- the next point's command, which says what is coming;
- the command of the point just passed, which says whether a turn is in progress;
- the distances to the next point and to the goal.

Progress only moves forwards. Passing a point counts as reaching it, so GNSS offsets of
a metre or two don't stall the route.

## Simulator side (`src/carla_adapter`)

### `sensor_rig.py`

Spawns the suite and joins every sensor on the exact frame.

**Conversions** from CARLA's left-handed frame:

| Sensor | Conversion |
|---|---|
| LiDAR | y flipped, then the mount transform |
| Radar | azimuth sign flipped; velocity sign from `RADAR_VELOCITY_IS_RANGE_RATE` |
| IMU acceleration | y flipped |
| IMU angular velocity | pseudovector rule: `(-gx, gy, -gz)` |

**Noise:** compass and speedometer noise come from one seeded generator, applied on the
main thread.

### `sensor_rig_check.py`

Run with `scripts/check_sensor_rig.py`. A live check, noise-free by default, that
proves the conversions before anything is built on them.

**Stationary, with two parked targets in the neighbour lane:**
- every sensor delivers every tick;
- each LiDAR sweep covers 360° with no self-hits;
- LiDAR extrinsics: points fall inside the true target box, not its mirror image;
- camera and LiDAR agree on where the target is in the image;
- the radars see the targets on the correct side;
- the IMU reads gravity at rest, the compass matches the true heading, and GNSS fixes
  arrive at the configured rate.

**Driving:**
- GNSS displacement matches the true displacement;
- the speedometer matches the true speed;
- the gyro yaw rate and the accelerometer x have the correct sign;
- the radar velocity sign matches the car's own closing speed on static returns.

**Output:** timing (tick, sensor arrival, frame build) and sample images, including a
top-down LiDAR view.

### `mapless_capture.py`

Run with `scripts/capture_mapless_dataset.py`. Training-data capture on **Town01–Town06
only**; Town07 (the held-out final test town, plan section 12) and Town10HD (the
development benchmark) are refused.

- **Driving:** the ego drives on Traffic Manager autopilot through background vehicles
  and pedestrians. This is for data collection only.
- **Weather:** cycles through seven presets.
- **Each sample saves:**
  - what the car sees;
  - a semantic label image per camera;
  - truth: ego pose, velocity and control, traffic-light state, nearby actors as
    vehicle-frame boxes, and the lane with its marking types.
- **Status of labels and truth:** training and evaluation data, never car inputs.
- **Splits:** by whole run and whole town.

## P0.1 additions

### Rig v2

`configs/sensor_suites/mapless_v2.json` is now the default; v1 is kept for comparison.

| Sensor | v2 setting |
|---|---|
| Radars | 6,000 points/s; front 60° × 15°, 80 m; rear 90° × 15°, 60 m; mounted just outside the bumpers |
| LiDAR | z 2.2 m; −30° to +10°; 60 m; nominal drop-off 0.20 |
| Side and rear cameras | 800×450 |

**Radar density against FOV is an experiment, not a settled choice.** Vary the suite
from the command line:

```powershell
python scripts\check_sensor_rig.py --output <folder> --set radar_front.points_per_second=3000 --set radar_front.horizontal_fov_deg=30
```

The rig records the attributes it set, and also the attributes CARLA reports on each
actor (`effective_attributes`).

### Per-sensor health

**What each frame carries.** Every `SensorFrame` has one `SensorStatus` per mounted
sensor, plus the `calibration_id` of the suite and mount geometry used.

| State | Meaning |
|---|---|
| OK | Data present |
| NOT_DUE | Scheduled gap, such as GNSS between its 5 Hz fixes |
| MISSING | Due, but no sample arrived |
| INVALID | Rejected, with a reason (bad size, wrong timestamp, decode error) |

A sensor's data is present exactly when its state is OK.

**Degraded, not fatal.** The rig now delivers a degraded frame instead of stopping the
car. `strict=True` keeps the old refuse-and-brake behaviour; capture uses the default
mode and skips degraded samples, and the rig check counts them.

**Ages.** `mapless.health.HealthTracker` turns statuses into ages and overdue flags,
measured against each sensor's expected period.

**Policy belongs to the planner, and depends on conditions.** The planner asks
`stale(names, max_age_s)` for the sensors covering its current corridor and manoeuvre.
Any age limit must fit inside the response time assumed in the stopping-distance
calculation.

### Calibration gate (`scripts/check_sensor_rig.py`)

Run with noise off by default. The evaluator-side geometry is in
`evaluation/rig_calibration.py`.

1. **Stationary:**
   - completeness, using the statuses;
   - full LiDAR sweep;
   - ego-body LiDAR hits (reported; perception masks them);
   - no radar self-returns;
   - IMU gravity and stillness, compass, GNSS rate.
2. **Coverage sweep.** One parked car at a time:
   - ego lane at 5, 10, 20, 30 and 50 m ahead, and 8 and 15 m behind;
   - each side lane alongside, 6 and 15 m ahead, and 10 m behind.

   Then box props, one per camera view. Per placement, the check records:
   - LiDAR box points against the mirror-image box;
   - per camera: whether the target is seen, mask consistency (the share of its LiDAR
     points that land on its instance mask), and silhouette edge error in pixels;
   - per radar: hits and range error along each return's ray.

   Outputs: `coverage` in the JSON and `coverage_top_down.png`.
3. **Radar motion:** stationary ego with a target receding at 3 m/s; ego and a lead car
   both at 3 m/s.
4. **Driving** (straight, gentle left curve, stop):
   - GNSS displacement, speedometer, gyro yaw rate and accelerometer sign;
   - radar sign;
   - range rate on static-world returns, with and without the lever arm `ω × r`.

**Gates:**

| Check | Pass |
|---|---|
| LiDAR extrinsics | Every placement within 45 m has 30+ box points, and fewer than 20% of that count in the mirrored box |
| Mask consistency | ≥ 0.85 per camera |
| Prop reprojection | Edge error p95 ≤ 3 px over at least 8 edges |
| Radar range | Error p95 ≤ 0.5 m at 5–40 m |
| Radar range rate | Error p95 ≤ 0.5 m/s |
| Radar sign | Correct in every case |

Car silhouettes and radar detection coverage are reported, not gated.

**Status:**
- MEASURED_PASS: every gated check passes;
- MEASURED_FAIL: any gated check fails;
- INCOMPLETE: there was too little evidence to decide a check.

**What the fake-world smoke test proves.** It runs the whole gate against an
independently written fake CARLA world:
- a consistent world passes;
- a flipped radar sign fails;
- a missing LiDAR y-flip fails.

### First live run of the gate (22 Oct, PC 2, Town10HD_Opt, rig v2, noise off)

Status MEASURED_FAIL. Every failure traced to a measurement assumption, not to a
broken conversion, and the gate was corrected as follows.

| Finding | Evidence | Change |
|---|---|---|
| CARLA's radar `velocity` is the range rate (positive when receding), despite the docs' "towards the sensor" | Measured/expected ratio −0.99 for a receding target (834 returns) and for a moving ego against the static world (6,838 returns); the equal-speed lead read about 0 | `RADAR_VELOCITY_IS_RANGE_RATE = True` |
| Road returns were counted as target points | The mirror-image box over empty road held 20–40% as many LiDAR points as the real one | Box counts start 0.2 m above the box bottom |
| The road's instance code won the mask vote | Front-camera masks spanned half the image (72,817 px) | Scenery semantic tags are excluded; car targets must carry a vehicle tag |
| IMU accelerometer spikes after spawn | Mean \|a\| = 1,448 m/s² at rest, while gyro and compass were exact | The first 5 ticks are excluded and recorded |
| GNSS schedule slips by a tick | 1 MISSING out of 40 ticks | MISSING only when a fix is more than one tick late |
| Destroyed actors listed straight after the session | The ego and its 14 sensors were reported as leaked | Waits up to 5 s for the server to process the destruction |
| Top edges of box props | About 4 px at 3 m, while left, right and bottom edges agreed to under 0.5 px | Top edges reported, not gated (the box01 mesh sits below its bounding box) |
| Car bounding boxes are not radar truth | Positive range errors up to about 3 m on car bodies | Radar range gated on box props; cars reported |
| LiDAR self-hits | 382 points on the ego roof at z 2.2 m, −30° | Reported; perception masks the ego body |
| Timer resolution | `time.monotonic` ticks every 15.6 ms on Windows | `perf_counter` |

**Already good on the first run:**
- compass error 8×10⁻⁶ °;
- gyro yaw rate error 1×10⁻⁶ rad/s;
- GNSS displacement error 0.65° and 0.004%;
- speedometer exact;
- radar detection 7/7 (front) and 3/3 (each rear) inside their fields of view;
- full 360° LiDAR sweep at about 27,000 points.

**Coverage finding:** no radar sees a car directly alongside, in either side lane.
Lane-change checks there rely on LiDAR and the side cameras.

### Follow-up live runs (22 Oct, both PCs, identical results)

- **Spawned cars are invisible to cameras for their first ticks.** The segmentation
  inside a spawned car's projected box held only road, lane markings and pavement for 3
  ticks. Every camera, at every distance, first showed the car on tick 4 (0.2 s).
  LiDAR saw it at once. Placements now settle 12 ticks (`--settle-ticks`).
  - Consequence for scenarios: spawn scripted actors at least 4 ticks before they
    matter, or out of view.
- **The cameras do not lag the simulation.** A settled car jumped 1.5 m sideways 4
  times, and the camera matched LiDAR on every jump (`camera_minus_lidar_ticks` all
  0). The first jump after spawn took effect a tick late for both sensors, which is a
  teleport delay, not render lag.
- **Coverage directly alongside:** outside every radar. The first runs also reported
  it as unseen by the cameras, but that came from the check itself, which skipped any
  target with a box corner behind a camera's image plane; the side cameras do see a
  car alongside, partially (fixed in `G1b-4`: partial targets are judged from the
  corners in front). Before lane changes, consider two side-facing cameras at ±90° (rig
  v3; the track allows 8 cameras) for full coverage.
- **With 12 settle ticks,** every gated check passes on both PCs: mask consistency,
  prop reprojection, LiDAR extrinsics, radar range, range rate and sign, IMU, GNSS,
  compass, speedometer and cleanup.
- **Capture pilot (22 Oct, PC 1, Town03):** 2,000 samples over 17 episodes. The run
  ended "Cleanup incomplete" because 32 walker actors had already been removed by the
  server; the data was complete. The session now skips actors the server no longer has
  and the manifest counts them (`actors_removed_by_server`).
- **Pilot inspection (23 Oct):** accepted as training data despite the manifest's
  INCOMPLETE (cleanup only).
  - Labels and LiDAR agree on every camera: ground points on ground labels 0.978 to
    0.990, points in vehicle boxes on vehicle labels 0.966 to 0.991.
  - Seven weather presets, all daytime; median speed 8 m/s (Traffic Manager); no
    degraded frames.
  - **Too few road users:** 0.8 cars and 0.2 pedestrians within 30 m on average; of 104
    front-camera pedestrians in 200 samples, 60 were under 20 px tall and 12 were 40 px
    or taller. Capture defaults are now 80 cars, 150 walkers and a 0.3 road-crossing
    factor.
  - CARLA labels stop lines as ordinary road lines; a stop-line class would have to come
    from the map's stop positions at capture time (traffic lights, G4).
- **G0 closed (22 Oct).** The record run on `01a086d` (PC 2) is `MEASURED_PASS`, with
  render lag 0 on all four trials. PC 1 passed the same checks on `ffbbc21`.

### Local trajectory controller (G1a)

`navigation_control.local.LocalTrajectoryController` follows moving-horizon
`LocalTrajectory` updates from a planner.

**What it fixes in the baseline `PathFollower`.** Every `set_path()` resets its speed
PID and progress, and its stopping logic treats the path end as the destination.

**Behaviour:**
- **Updates keep state.** The speed PID survives updates. A manoeuvre change restarts
  only the progress along the new trajectory.
- **The horizon end is not the goal.** Only `final_goal=True` can latch FINISHED.
- **Never outrun the plan.** Speed is capped so the car can stop `stop_margin_m` before
  the horizon end. A planner avoids that cap by sending horizons longer than
  `minimum_horizon_m(speed)`.
- **Stale plans brake.** An expired trajectory brakes along its own path and reports
  EXPIRED.
- **Implausible plans are refused.** A trajectory that starts more than 2 m from the
  car, or has already expired, is rejected and the current one kept.

**Offline evidence** (`tests/unit/test_local_controller.py`):
- replanning every 0.1 s gives no speed dips;
- the baseline follower under the same replanning settles about 0.5 m/s low, because
  each `set_path()` clears its PID integral;
- short horizons are capped;
- an expired plan brakes in lane;
- the final goal stops within tolerance;
- a manoeuvre change keeps speed continuity.

The live part of G1a comes with the first mapless drive.

## G1b: first mapless drive (`src/mapless`, `scripts/run_mapless_drive.py`)

The car is `mapless.driver.MaplessDriver`: `SensorFrame`s and a sat-nav route in, one
`ControlCommand` out per tick. It never sees the map, the simulator's poses or any actor.

| Stage | Module | What it does |
|---|---|---|
| Ego motion | `ego_motion.py` | Odometry frame (wheel speed and gyro, gyro bias measured at standstill) for driving. EKF route pose initialised from GNSS and compass averaged over 1 s at standstill, then GNSS and compass updates; used only for route progress and the goal |
| Terrain | `terrain.py` | Per sweep: road height in 2 m bands grown out from the car; points labelled ROAD, RAISED (kerb, pavement, 0.10 to 0.30 m), OBSTACLE (0.30 to 2.4 m). Evidence accumulates on a grid in the odometry frame (kerbs decay over 3 s, obstacles over 0.5 s) |
| Corridor | `corridor.py` | Lateral profiles along last tick's centreline; right kerb = first raised cell with road seen next to it (obstacles such as parked cars are OCCUPIED, never an edge); the kerb may not jump in either direction (side roads, cars); kerb evidence must lie within 0.75 m of where last tick's line puts it; lane centre `keep_right_m` left of the kerb, measured at a standstill before the car moves (the kerb's raw inner contour from 10 m behind to 15 m ahead fitted as a quadratic, or a line when only one side is seen, accepted once 10 sweeps agree within 0.15 m, else a FAULT with the reason; never a default); a corridor under 5.5 m wide is one lane, driven in its middle; polynomial fit (degree from how far the evidence reaches) with last tick's line as a prior, continued along its tangent past the evidence, short evidence that bends on its own gets a quadratic, or last tick's line carried unchanged when there is too little evidence or the car stands still. Kerb cells count only if hit within 3 s, the road ahead must be observed within 1 s (an empty LiDAR ends the corridor), and the evidence time is the newest kerb hit, so a carried line ages and expires after 5 s |
| Obstacles | `obstacles.py` | This sweep's OBSTACLE points in the car-width tube along the centreline (nearest dense cluster), widened on the outside of a bend by the front corner's sweep (`tube_bounds`); RAISED points under the body width, 12 up to 5 m and fewer with range (never under 3); and how far the tube is *seen clear*: ground observed in 0.5 m slices from near the car outwards |
| Planner | `planner.py` | Cruise speed, curve limit, braking ramp to a stop 2.5 m short of an obstacle (bumper) or at the goal (final-goal trajectory). One hold per obstacle, each in three states: BLOCKED (seen), CLEAR (road seen clear 2 m past where it stood, and ground in every 0.3 m bin of its own region just behind its near face), UNKNOWN (neither: the stop stays). The stop is the nearest hold's, worked out from the obstacle's own position on each tick's path. Released only after 1 s of continuous CLEAR |
| Control | `navigation_control.local` | `LocalTrajectoryController` in the odometry frame |
| Driver | `driver.py` | Modes STARTUP (GNSS and compass averaging), CALIBRATING (kerb distance), DRIVING, STOPPING, FINISHED, FAULT. Faults: LiDAR older than 0.15 s, IMU or speedometer older than 0.2 s, GNSS older than 3 s, route position sigma over 4 m, failed calibration, no corridor. A fault brakes along the last trajectory while odometry is trusted (IMU and speed fresh), otherwise holds the last steering: never wheels straight in a bend. From a standstill the car moves only on kerb evidence fresher than 0.5 s; a lost corridor at a standstill is re-acquired from the start-up assumption |

**Sat-nav frame.** The local plane is built from the route's first point with the
receiver's `north_sign` from the sensor suite: +1 on Earth, −1 on CARLA 0.9.16, where
the GNSS sensor's latitude grows towards the compass's *south* (measured by
`scripts/check_gnss_frame.py` on a Town05 capture: north per core y = −0.9993; found when
the live curve run drove the bend correctly while its route frame went the other way and
the goal was never reached). The rig check now measures the sign (`gnss_north_sign`) and
fails when the suite's value is wrong. `scripts/check_geolocation.py` on the live server
(Town04, 23 Oct) gave +0.9993 m of latitude per metre of CARLA +y for both the GNSS
sensor and `map.transform_to_geolocation`, so the route and the receiver share one frame
and the sign is the whole fix; the offline formula `carla_geolocation` was corrected to
match the server (it had followed the source comment, which says the opposite).

**Start-up assumption, explicit:** the car starts stopped, centred and aligned in a lane
with a kerb on its right, as the harness places it. The calibration checks what it can
(a clean kerb line within 10° of the heading, stable over 10 sweeps) and fails otherwise.
A car placed off centre keeps that offset; camera lane evidence (G1c/G2) removes this.

LiDAR intensity in the simulator depends on range only, so lane paint is invisible to
it. G1b therefore keeps to the lane by following the right kerb. Camera lane evidence is
G1c and later.

**Offline:** `sim_harness/mapless_road.py` ray-casts the rig's LiDAR against a kerbed road
(straight and arc pieces, side-road openings, road crown, parked cars) with the suite's
GNSS, IMU and speedometer noise. In closed loop, the driver keeps the lane to under
0.3 m p95 on straights, 25 m radius bends and a crowned road, stops 3.2 to 3.7 m short of
a parked car, and reaches the goal within 3 m (`tests/unit/test_mapless_g1b.py`).
`tests/unit/test_mapless_g1b_review.py` runs the fault sequence from the review of
`94b5caa`: an empty LiDAR at start-up holds the car and calibration fails with a reason;
calibrated drives (straight, a 3.5° skewed start, a kink like live run e's, 25 m and 12 m
bends) keep the whole body in the lane from the first tick; IMU loss in a 12 m bend
stops the car in its lane with the steering held, LiDAR loss braking along the bend;
stale IMU, speedometer and GNSS fault and recover; loss of road evidence does not renew
the corridor; an occluded obstacle holds the stop with zero creep for 10 s; a removed
one resumes only after 1 s of observed clearance. Standstill calibration reads the kerb
within 0.07 m on straights and skewed starts and 0.1 m on a 100 m bend.
`tests/unit/test_mapless_drive_smoke.py` runs the live harness end to end on a fake
CARLA built on that road.

**Live:** `scripts/run_mapless_drive.py --scenario straight|curve|stop`. The harness
searches the map's junction-free lane runs (topology segments, from 8 m past a junction)
for the rightmost driving lane with the scenario's shape, preferring a mapped pavement
straight after the lane. The default route is 60 m (`--goal-m`), so a site needs 75 m
without a junction; the first live attempt on Town10HD found no 115 m run from a spawn
point. If nothing fits, the error gives a census of why and the longest runs. The
harness writes the route from the map's geolocation and spawns the parked car for `stop`
(35 m ahead) before the car starts. It scores from truth only:

| Check | Pass |
|---|---|
| Start-up | Localised from GNSS and compass within 3 s, no spawn pose |
| No collision | No collision event |
| Lane keeping | After the first 15 m: offset p95 ≤ 0.35 m, max ≤ 0.70 m |
| Lane footprint | Whole run, calibration and start included: no part of the body outside the lane (rear axle and front bumper within half a lane minus half the car) |
| Calibration | Kerb distance measured at a standstill and accepted, within 0.3 m of the map's |
| Goal (straight, curve) | FINISHED, rear axle within 3 m of the goal |
| Obstacle stop (stop) | Standing still, bumper gap 1.5 to 5.0 m, held for 5 s |
| Speed | Never above cruise + 0.5 m/s |
| Faults | Under 2% of driving ticks in FAULT |

Reported: driver step time, start-up route-pose error, and the LiDAR kerb's bias while
driving. `--keep-right` replaces the measurement (the calibration check then only
reports). The raw sweeps until the car first moves are saved in `sweeps/` so a
calibration can be replayed offline; snapshots show ROAD, BARRIER and OCCUPIED cells.

**First live attempt (23 Oct, PC 2, Town10HD, `2a7e9da`): the car never moved.**
- Town10HD's rightmost lanes have two shoulder lanes before the pavement: the kerb is
  4.39 m from the lane centre, not half a lane. With a fixed 1.75 m the corridor centre
  was 2.6 m right of the car, every trajectory was refused (more than 2 m from the car),
  and the driver braked while reporting DRIVING.
- The truth lane started at the first waypoint, 1.4 m ahead of the rear axle, so the
  log showed a 1.39 m offset that did not exist.
- Fixes: the kerb offset is measured at start-up; refused trajectories are a FAULT with
  the reason; parked cars on the shoulder can no longer pose as the kerb (their sills
  are kerb-high, so a raised cell that has also seen tall points is OCCUPIED); the road
  height comes from the car's own path first (on a narrow road the LiDAR's blind ring
  left only pavement beside the car); the truth lane extends 15 m behind the start. The
  synthetic road now has a Town10HD-like case: kerb 4.4 m away and cars parked on the
  shoulder.
- Limitation: measuring at start-up assumes the car starts aligned in its lane, as the
  harness places it. A car placed off centre holds that offset; camera lane evidence
  (G1c/G2) removes this.

**Second live attempt (23 Oct, `f982781`): drove 50 m sensor-only at 2 m/s, then
stopped for nothing.**
- The kerb offset measured at a standstill from single points was 3.81 m against the
  map's 4.39 m. The driving detector found the kerb 0.14 m inside the map's, so the car
  held 0.45 m right of centre (lane p95 0.49 m, FAIL). The two measurements disagreed:
  the first raised point on a rounded kerb and gutter is closer than where raised cells
  take over.
- About 7 m before a parked car it stopped and held to the time limit (goal 10 m away).
  Offline, the same pattern came from the corridor itself: a stray kerb edge behind a
  parked car at 26 m bent the cubic fit, and the bent path ran into the next parked car.
- Fixes (`G1b-3`): the kerb offset is measured on an 8 m calibration drive along the start
  line with the driving detector itself; lateral profiles are sampled every 5 cm so that
  the calibration and driving read the same kerb cell; kerb evidence must lie within
  0.75 m of where last tick's line puts it; the fit's degree follows the evidence's
  reach and the line continues along its tangent beyond it; the obstacle's lateral
  offset is logged. Offline: lane p95 0.05 to 0.20 m on straights, the shoulder-parking
  road and bends down to 12 m radius with the live controller tuning; no phantom stops.

**Third live attempt (23 Oct, run e, `94b5caa`): drove, held 0.64 m right, stopped and
crept.**
- The calibration drive read 3.6 m against the map's 4.39 m. The log shows why: the
  lane runs about 2.5° off the start heading soon after the start, and an 8 m drive along
  the start heading drifted 0.39 m off the lane while measuring. A heading-dependent
  measurement cannot be made by driving the heading.
- The stop was for a real parked car in the tube, because the car was 0.64 m right of
  centre. The hold then released on 1 s of *missed* detections and the car crept 0.6 m
  towards the parked car between flickering sightings.
- A review of `94b5caa` (nine points, all taken) led to `G1b-4`:
  - **Calibration at a standstill, as a line fit.** The kerb is fitted as a line from
    10 m behind to 15 m ahead and the axle's perpendicular distance to it is the lane
    centre's; a skewed start is measured across the kerb, not along the car. A
    quadratic was tried and rejected: at a standstill the kerb is seen only where
    LiDAR rings cross it, each edge quantised to a cell, and the curve read 1.55 m for
    a 1.75 m straight kerb. The line's cost is a bend: 0.1 m on a 100 m radius. The
    measurement must settle (10 clean sweeps within 0.15 m) within 4 s or the drive is a
    FAULT with the reason. There is no default.
  - **Freshness.** Kerb cells count only if hit within 3 s; the road ahead must be
    observed within 1 s, so an empty or dead LiDAR ends the corridor rather than leaving
    old road looking current; the corridor's evidence time is the newest kerb hit behind
    it, so a carried line ages; from a standstill the car moves only on evidence fresher
    than 0.5 s.
  - **Ego sensors.** IMU and speedometer older than 0.2 s, GNSS older than 3 s and a
    route sigma over 4 m are faults; the gyro bias is never learned from a reused
    reading.
  - **Obstacle hold in three states** (BLOCKED, CLEAR, UNKNOWN): the stop releases only
    on observed clearance 2 m past where the obstacle stood, for 1 s.
  - **Fault braking** along the last trajectory while odometry is trusted, else with the
    steering held.
  - **Harness:** a whole-run lane-footprint gate (calibration and start included), a
    calibration gate, raw sweeps saved until the car moves, OCCUPIED in snapshots.
  - **Capture:** truth boxes placed from CARLA's world vertices with their corners
    (schema 2; schema-1 runs are repaired in place by
    `scripts/repair_mapless_capture.py`, which reads each blueprint's box offset from a
    CARLA server once and can export it); Town07 refused (held-out final test town);
    the manifest records the source commit and the spawned counts; the inspection adds
    integrity, per-camera counts, an RGB-to-label pairing check, measured encounters
    and a verdict.
  - **Rig check:** a target partly behind a camera's image plane is judged from the
    corners in front (the "only LiDAR sees a car alongside" finding came from skipping
    those); failing evidence fails a check even when evidence is thin; skipped phases
    leave their gates INCOMPLETE.

Each run writes `drive_report.json`, `drive_log.jsonl`, `route.json`, `drive_plot.png`,
top-down terrain snapshots and the start-up sweeps.

**Fourth live attempt (23 Oct, `2b7e14d`, three scenarios): drove all three.**
- Straight: finished, lane p95 0.26 m, zero faults. Stop: stopped 3.5 m short and held 130 s
  with zero creep. Curve: drove the 90° bend at −0.11 m the whole way.
- What failed, and why: the standstill calibration read 3.98/4.13 m against the driving
  detector's 4.24 m. The saved sweeps showed the "straight" site bends at 58 m radius; a
  line fit through 25 m of kerb reads the chord. The stop run's faults were the new
  stale-evidence rule misfiring at a standstill (only the LiDAR's ring crossings hit the
  kerb when stopped: 4 edges, too few to refit, so the carried line was treated as
  unsupported and reset every tick). The curve run never reached its goal because the
  route frame was mirrored (above) and ran past the end of the truth lane.
- Fixes (`G1b-5`, with a second review of `2b7e14d`): calibration from the raw kerb contour
  as a quadratic (reads within 0.07 m on straights, skewed starts, kinks and bends to 50 m
  radius offline; 4.13 m on the live sweeps); fresh kerb hits renew a carried line even when
  too few to refit, and the corridor is reset only once its evidence has expired; the
  obstacle hold is checked before the moving-off rule. From the review: an IMU or
  speedometer outage integrates nothing (holding the last speed invented 6 m for 0.8 m
  driven) and the odometry is marked untrusted, nothing is placed in the grid meanwhile,
  and on recovery the car stops and rebuilds grid, corridor, obstacle hold and trajectory
  together; the clearance count restarts on any gap (a fault, a missing sweep); seen-clear
  needs ground in 2 of 3 strips across the tube; kerb-high objects in the tube are
  obstacles; per-class cell timestamps so a fresh road hit cannot vouch for an old kerb;
  the footprint gate uses the four body corners; the harness ends a run cleanly past the
  goal or the truth lane's end; the capture verdict splits integrity, geometry and
  coverage (coverage only warns; the edge ratio is a warning).

**G1b closed (23 Oct, PC 2, Town10HD_Opt, `955740b`/`1348ec6`): straight, stop and curve
all MEASURED_PASS**, sensor-only at 2 m/s, no map, no simulator pose, no actor truth in
the car.
- Straight (60 m): lane p95 0.26 m, zero faults, goal within 3 m.
- Stop: stopped with a 3.5 m bumper gap and held, zero faults, no creep.
- Curve (90° at 39 m radius, 63 m): lane p95 0.06 m, max 0.064 m; worst body corner 1.32 m
  from the lane centre (half lane 1.75 m); calibration 4.20 m against the map's 4.385 m
  (the map measures to the sidewalk lane, the LiDAR to the 10 cm contour); route-pose
  error at start 1.9 m; driver step p95 24 ms.
- The first curve run with the same code failed only the footprint gate's own
  measurement (the lane heading read by index drifted 10° in the bend); fixed in `1348ec6`.
- Beyond the gate: the straight at 4 m/s (`--speed 4`) also MEASURED_PASS, lane offset
  within 0.16 m, zero faults, no retuning.

## G1c: first commanded turn (`src/mapless/junction.py`)

A 90° turn at a junction from the sat-nav's LEFT/RIGHT command, LiDAR only, planned in
the car's own frame. Developed in the synthetic world first (`SideRoad` in
`sim_harness/mapless_road.py`: a side road with filleted pavement corners, seen by the car
only through its LiDAR); the harness scenario and live runs come next.

- **Junction context from the route.** The entry point carries the command, the next point
  is the exit; the exit heading is ±90° from the car's own corridor (the two GNSS points
  are only good to ~8°, so the route's heading is used only when it disagrees by more
  than 35°). The route point locates the exit road, never the exit lane (review of
  `697d53c`, point B).
- **The kerb confirms the junction.** A right turn is entered only once the right kerb is
  seen to end for ≥3 m within 6 m of the route's entry, with observed road where the
  kerb would be (a kerb that runs out of the observed range is unknown ground, not an
  opening). A route that claims a junction the kerb does not show holds the car short of
  the entry. The mouth (the kerb opening, or the entry less a margin) only comes nearer
  while waiting, so a flickering opening cannot let the car creep forward.
- **Exit lane from LiDAR.** The exit road is read from the terrain grid along a probe
  line through the route's exit point in the exit heading: at each 1 m station the cell
  labels across the line (the corridor estimator's own profile search) give the last road
  cell and the first raised cell right of the line; the kerb is their midpoint while the
  unseen strip between them is ≤2 m. At 20 m the LiDAR's rings are a metre apart, so
  either cell alone is up to a ring spacing out (0.35–0.7 m on the synthetic turns); the
  midpoint reads 0.1–0.4 m from the truth and improves as the car nears. Raw sweep points
  were tried first and placed the kerb 1–2 m off at 25 m (a pavement top seen at a grazing
  angle reads as road). The stations are kept 2 s and fitted as a line (≥3 stations over
  ≥4 m, MAD ≤0.4 m). The exit direction is the prior's: a fitted direction is used only
  between 6° and 15° off it, because the range bias of the stations read a straight kerb
  as 2.5° skewed and the exit line ran into it 10 m on. The prior is a right angle to
  the entry lane's direction, latched in the odometry frame as the approach begins
  (nearer the mouth the kerb runs out and the corridor's fit tilts with it; measured from
  the car's heading instead, a 1.5° error drifted the exit line 0.6 m).
- **Arc candidates.** Entry line → arc (R 8.5–16 m, 0.5 m steps) → exit line; each
  candidate's full body footprint is swept over the grid, with the 0.2 m planning margin
  on the arc and the lane-keeping margin (0.05 m) on the straights (in a 3.5 m lane the
  body has 0.7 m to the kerb and kerb cells reach 0.25 m into the road: with the margin
  on the exit straight too, a lane placed 0.3 m off left no feasible radius). Unknown
  cells count against `verified_m` (review point A). Clearance is the body's nearest
  approach to a kerb or obstacle cell over the arc (margin excluded; counted twice, a
  correct lane read 0.25 m and the commit test was a knife edge). The chosen candidate is
  the nearest to the nominal 9.5 m among those with ≥0.3 m clearance — the widest
  clearance swung 2.6 m into the exit road's opposing lane (point E) — else the widest.
  The choice is latched within 1 m and committed when the exit is kerb-measured, the body
  clears by ≥0.3 m and the tangent is within 3 m. While committed the latched path is
  followed to its end whatever the route says; the car is only ever driven as far as the
  footprint lies on observed road.
- **Hand-back.** Once the heading is within 10° of the exit and the arc is behind, the
  kerb corridor is seeded with the exit line and takes over only when it finds its own
  corridor. A sensor loss mid-turn faults with the steering held while braking (the
  `_fault` steering is now zeroed only once the car is measured stationary, not when
  untrusted odometry reports zero speed; review point 1). Seen-clear needs the centre
  strip plus one of the outer two (point 2).
- **Synthetic results (offline tests, `tests/unit/test_mapless_g1c.py`):** right turn
  (10 m fillets) max lane offset 0.52 m against the truth arc in the box (the planned
  9.5 m arc is not the truth path), 0.07 m on exit; left turn 0.50 m / 0.03 m; both hand
  back and finish. Deliberate failures: a route junction the kerb does
  not show holds short of the entry; an exit point shifted 3 m is corrected by the kerb; a
  car parked across the exit lane leaves no clearing radius and the car waits at the
  mouth; LiDAR loss mid-turn brakes with the steering held. Known limit: a 6 m fillet
  right turn is infeasible for this body (rear-axle minimum ≈8.25 m); the live site search
  must measure the junction's lane radius.

## G1c harness, G2 and G3 frameworks (24 Oct)

The remaining gates have their code structure in place: contracts, modules with the
plan's numbers, the harness hooks and unit tests on synthetic data. Each is a framework
to debug against captures and live CARLA, not a result.

### Contracts (`src/mapless/contracts.py`, plan section 7)
`ObjectTrack` (odometry frame, `[x, y, vx, vy]` with covariance, class probabilities
over `car/truck/bike/pedestrian/unknown`, lifecycle status, sensor support, measured
footprint), `Prediction` (time-indexed occupied regions), `LocalRoadModel` (grid layers
occupancy/free/unknown/ground height/road/marking/pavement/age, lane, exit and stop-line
candidates, visibility) and `DecisionRecord` (proposed and executed action, veto reason,
constraints, coverage, stage timings). Every value validates itself.

### G1c harness (`turn_left`, `turn_right`; `evaluation/junction_truth.py`)
- **Site search** (`find_turn_sites`): a rightmost lane run ending at a junction whose
  connecting lane turns about 90° to the commanded side (±30°) into a rightmost exit lane
  of `--exit-m`; approach of `--approach-m`. The connecting lane is walked waypoint by
  waypoint until it leaves the junction.
- **Route** (`make_turn_route`): LANEFOLLOW every 25 m on the approach, the entry point
  with LEFT/RIGHT at the lane's last waypoint before the junction (where CARLA's routes
  put it), the exit point 5 m into the exit lane, the goal at the exit run's end.
- **Truth and gates** (review point F): the permitted area is approach, connecting and
  exit lane, each with its own width (the connecting lane is wider, plus 0.5 m, carried
  5 m into the exit as the turn winds out); body corners project onto it. Gates:
  `turn_executed`, `junction_footprint` (no corner outside the permitted area),
  `correct_exit` (ends on the commanded exit lane within its half width),
  `exit_lane_keeping` (p95 ≤ 0.35 m from 8 m past the junction), `approach_lane_keeping`.
  Tested on a fake map (L-shaped topology with a straight-on branch) and a synthetic log.

### G2 (`road_segmentation`, `painting`, `road_model`, `detections`, `stack`; `mapless_training`)
- **Segmentation**: `RoadSegmenter` interface over `road/marking/pavement/road_user/other`;
  `ConstantSegmenter` (road below the horizon at low confidence) so the stack runs
  without a model; `TorchRoadSegmenter` runs either the A1 BiSeNetV2 (from scratch,
  fastest, detail branch for sharp kerb edges) or SegFormer-B0 (ImageNet-pretrained MiT-B0
  encoder via `transformers`, 3.7 M parameters, converges on little data, logits at 1/4
  resolution) with a 5-class head, chosen by the checkpoint's `architecture`. The A1
  BiSeNetV2 result is not evidence against the architecture — it was trained from
  scratch on 19 Cityscapes classes with little data — but the pretrained encoder is the
  safer bet on moderate data; both are trained with `--architecture` and compared on the
  captures by pavement boundary agreement, marking recall and ms per image (section 9),
  not mIoU. YOLOv8n-seg is an instance model for road users, not a road-surface
  segmenter; the detector stays the A1 YOLO for boxes. The checkpoint carries provenance. The car's code names no town, so the loader checks the
  training side's attestation `test_towns_excluded` and the class list, and refuses a
  path inside any `labels` folder.
- **Training** (`scripts/train_mapless_segmentation.py`): capture runs indexed from
  `samples.jsonl`, CARLA tags mapped to the five classes, split by town (train 01/03/04/
  05/06, validate 02; Town07 and Town10HD refused at index time), inverse-frequency class
  weights, auxiliary heads, per-class IoU and recall, pavement boundary agreement within
  2 px and ms per image reported per epoch. Exercised end to end on a two-sample
  synthetic run (CPU, one epoch) and loaded back by the car's loader.
- **Painting**: LiDAR points projected into each camera in depth order (nearest per
  4-px bucket; a point more than 1 m or 8 % deeper is hidden); class probabilities
  averaged over the cameras that saw a point; unpainted points stay uniform.
- **Road model**: semantic evidence on the terrain grid's own lattice, following its
  recentring, decayed (τ 4 s) and dropped when older than 3 s; layers assembled from the
  terrain labels; lane candidates from marking bands (quadratic fits), stop lines from
  transverse marking runs, visibility along the car's axis. Ground height is a zero layer
  until section 9 item 4 is built.
- **Detections**: range-tuned clustering of obstacle points (link 0.6 m + 0.03 m/m, in
  10 m rings); `Detector` interface with `YoloDetector` (A1 ultralytics detector) and
  `NoDetector`; each box takes its foreground cluster (nearest 20th-percentile depth among
  clusters covering ≥30 % of the box), never the median of every point in the box; boxes
  with no cluster become camera-only class evidence with a bearing hint.
- **Stack** (`MaplessStack`): the G1 driver, then segment/paint (every 2nd tick), cluster/
  detect/join, radar (as admitted), track, predict, follow; a following speed or a
  predicted conflict within 3 s reaches the driver as `speed_cap_mps`; a `DecisionRecord`
  per tick with stage timings. Admission presets A (camera+LiDAR), B (+front radar),
  C (+rear radars) for the section 10 ablation. On the synthetic world with a grey camera
  and a synthetic front radar the stack tracks a parked car (confirmed, LiDAR-supported)
  and the driver stops short of it. `scripts/run_mapless_stack.py` runs it on the drive
  harness (`--admission`, `--segmenter`, `--detector`), writing `decisions.jsonl`.

### G3 (`radar`, `tracking`, `prediction`, `following`; `carla_adapter/mapless_scenarios.py`)
- **Radar**: returns to the vehicle frame; ego-motion compensation with the lever arm
  (`v_sensor = v_ego + ω × r`, range rate positive when the range grows, as measured on
  CARLA 0.9.16); conservative single-linkage clustering (1.5 m, 1 m/s) with wide clusters
  split; an empty frame is reported empty, never as clear.
- **Tracker**: one CV-EKF per track; gates on space and class, then Mahalanobis χ² 9.21
  (2-D position) and 6.63 (radial speed, checked separately: an inconsistent rate updates
  range and bearing only); one-to-one nearest-first assignment; tentative → confirmed at
  3 hits in 5 frames, coasting to 0.3 s, deleted after 0.75 s; tentative tracks within
  12 m ahead are reported as hazards. Synthetic: a car moving away at 3 m/s is confirmed
  with its velocity within 0.6 m/s; radar-only tracks start without an extent.
- **Prediction**: constant velocity 4 s at 0.2 s (6 s for lane changes), footprint
  inflated by 2σ position uncertainty + 0.3 m, pedestrians widening at 1 m/s;
  `first_conflict` finds the earliest path point inside a region.
- **Following**: gap 4 m + 2 s·v, acceleration ≤ 1 m/s², jerk ≤ 2 m/s³ (rate-limited
  target), worst-case cap from the lead braking at 6 m/s² with a 0.5 s response and
  4 m/s² ego braking, stopping 2.5 m + 1 m short; nearest in-lane track ahead is the lead.
- **Scenarios M04–M06** (`crossing_pedestrian`, `braking_lead`, `cut_in`): scripted
  actors on the harness site with a `separation` gate (oriented-polygon distance between
  the ego box and the actor's box never below 1 m; INCOMPLETE without measurements) and
  per-scenario `completion` (yield-resume-goal, safe gap then goal, sustained safe hold —
  see the review of `6773e24` below). Written against the CARLA API, not yet run live.

### Review of `36ba0ca` (fresh-eyes read, 24 Oct) — all taken
Would have broken a live run: the walker spawned on the *left* of the lane (CARLA's
right vector is (−sin yaw, cos yaw)); the radar bearing innovation was not wrapped
across ±π in the Kalman update (a track behind the car jumped 60 m); the cut-in car
accepted the opposing lane as "left lane" (same-direction check on `lane_id` sign
added). Silently wrong: the turn gates compared the harness's route-start `s_m` against
path-start junction bounds (15 m apart; `s_offset_m`); the stack's conflict cap timed the
path with the already-capped planned speeds and flickered (now timed at the uncapped
speed); following measured the gap from the corridor line's start, 2 m behind the axle;
clusters were cut at 10 m range rings (one global pass now); tentative 3-point clusters
could lead the car (confirmed/coasting only; coasting past 0.3 s leaves the live set,
`Tracker.live`). Also: the road model labels only a 45 m window around the car and is
built at 5 Hz (was 21–31 ms over the full 120 m grid every 2nd tick). Stack stage times
on the dev CPU, synthetic world: drive 11 ms, paint 5, detect <1, track 1, road model
~10 at 5 Hz; total p50 18 ms.

### Second review, of the G1c turn code before its first live run (24 Oct) — all taken
A separate read of `junction.py`, the driver's junction block and the turn harness, with
instrumented synthetic runs, found:
- **The exit lane is placed `keep_right` left of the exit kerb** — `keep_right` is the
  approach lane's kerb distance (4.4 m on Town10HD's shoulder lanes). Emulated with a
  12 m main road turning into a 7 m side road: the car finished 2.6 m into the opposing
  lane, cleanly, and nothing in the car can tell (the opposing lane is road; the
  hand-back corridor finds the kerb at `keep_right` and is content). Until lane markings
  are read (G2) this is a declared limit: the harness picks only sites whose exit kerb
  distance matches the approach's within `--max-kerb-mismatch` 0.3 m (census key
  `kerb_mismatch`), and ends a turn run when a body corner is more than 1 m outside the
  permitted area.
- **The exit heading prior inherited the corridor's tilt.** The lane direction was taken
  from the quadratic's slope at the car, which the kerb running out into the mouth 10 m
  ahead tilts by 2.3° even on the noise-free synthetic road; that skew went into the
  exit heading and, absorbed by the median over stations ~14 m out, moved the exit lane
  0.95 m. Now: the lane direction is a line through the kerb evidence beside and behind
  the car (`Corridor.heading_near`), the exit-kerb line is fitted and anchored at the exit
  point itself, and its direction is used from 1.5° up to 5° (a larger fitted skew is two
  station clusters levering, not a skew — one such 11° fit put the car 2.8 m off). The
  harness also refuses approaches bent more than 5°.
- **The exit search looked outward from the probe line only.** With a 3 m route error
  toward the kerb (GNSS bias, seed 3) the line lay on the pavement, found a barrier and
  no road, and the exit was never measured: the car held at the mouth and faulted. The
  exit road is now the run of ROAD cells nearest the line on either side; its right end
  and the first raised cell beyond bracket the kerb. `exit_max_distance_m` 7.5 m (4.4 m
  shoulder + 3 m route error; the profile spans ±8 m), `max_entry_offset_m` 10 m (the
  opening is read ~4 m into the fillet, the box edge up to 10 m past it).
- **A blocked latched path stopped the car in the junction for good.** A kerb cell newly
  seen under the latched path made `follow` fail and nothing re-planned. Now: while the
  car has turned less than 20° from the entry heading the latch is dropped and the turn
  re-planned the same tick; deeper into the arc the car stops along the turn path (not
  the pre-turn kerb line, which would have been a steering jump).
- **The kerb edge rule.** With the exit kerb bracket in hand, the road's outer edge is the
  kerb when the unseen strip is wider than two cells (the strip lies on the pavement side
  in both geometries measured: the kerb's shadow seen from the road, sparse pavement
  returns seen end-on); the midpoint sat 0.3–0.6 m onto the pavement. Far exit lanes are
  sampled by a ring or two, so a road run of 0.5 m counts, the strip may be 3 m (ring
  spacing at 25–30 m) and three stations over 2 m fit the lateral position (the direction
  is the prior's unless eight stations over 10 m say otherwise).
- **Mid-arc.** A blocked latched path deep in the arc is re-planned as one arc tangent to
  the car's heading and to the exit line re-measured from there (closer, so better); blocked
  within 10° of the exit heading, the kerb corridor is seeded from the turn path and takes
  over at once (the car is on the exit lane already). Only when both fail does the car stop,
  along the turn path.
- **The exit heading stays the prior's.** A line fitted through the stations reads a
  slope anyway (the kerb estimate's bias grows with range), and a fitted −4° sent one exit
  straight 1.6 m off a straight kerb; the fitted skew is now reported only, and the lateral
  position is read from the nearest stations (within 8 m of the nearest), where the
  estimate is best. A latched path blocked far along (the exit straight against a kerb cell
  at 20 m) is re-anchored on the re-measured exit line with the same arc and tangent point,
  so the car's heading does not jump; the entry-line re-plan is used only while the car has
  turned less than 5°.
- **Outcome over fourteen synthetic turns** (both sides, GNSS seeds 1–5, seed 5308 with and
  without noise): the exit kerb is read within 0.5 m of the truth in all fourteen (ten
  within 0.35 m), all complete without collision, and the largest excursion against the
  truth arc is 1.9 m on one left turn (0.2–1.2 m on the rest; a different radius is a
  different path, and the permitted-area gate, not this number, is the safety measure). Driver step time through the planning phase 28 ms p50 / 42 ms
  p95 on the dev CPU (the swept check is now one vectorised placement, and radii are tried
  nearest the nominal first, stopping at the first with the commit clearance — the full
  sweep of sixteen cost 70 ms).
- The nominal radius is `nominal_fillet_m` (7.75) + `keep_right`, so a shoulder lane gets
  the wider arc its connecting lane has (9.5 m cut 1.1–1.4 m inside the map centreline
  and would have failed the footprint gate). The console now prints the turn
  diagnostics (`entry`, `open`, `R`, `exit`, `kerb@`, `ver`, `clr`, the turn state) and a
  run waiting at the mouth ends after `--junction-hold-s` 25 s with the reason, instead
  of sitting out `--max-s`. Also found: the synthetic side road was 40 m long and the
  exit lane's road surface ran out 2.4 m before the goal on some seeds (now 80 m).

### First live turn attempt (24 Oct, PC 2, Town10HD_Opt, `6773e24`): drove to the junction
The site search found a matching-kerb site at once; the approach was clean (offset
≤ 0.06 m) and the junction context appeared 17 m out. Three faults, all fixed:
- **The kerb opening was 9 m beyond the route's entry point** (CARLA's junction boundary
  lies well before the physical corner) and, once the car stopped, the opening was no
  longer confirmed (the cells that confirmed it are not re-hit at a standstill), so the
  hold point fell back to the route entry and the car waited 9 m short of the real mouth,
  where it could not see the exit road. An opening seen once is now kept in the odometry
  frame while the car waits.
- **Observed road expired while standing still:** the turn's verified distance shrank
  from 35 m to 14 m in four seconds at the mouth. Freshness for the junction checks is now
  measured from the car's last motion (`JunctionTurn.standing_s`), as nothing changes
  while it stands.
- **Roll-back on the brake crashed the run:** the baseline adapter refuses reverse motion
  beyond 0.05 m/s and the car rolled back a few cm/s at the hold. The harness uses
  `HoldingVehicleAdapter` (0.2 m/s tolerance, roll-back ticks reported) and the driver now
  holds the brake at a junction stop as it does at an obstacle stop.
Also seen: a phantom obstacle 1.5 cm ahead for one tick as the car braked hard (LiDAR
pitch under braking), released by the clearance rule.

### Second live turn (24 Oct, PC 2, Town10HD_Opt, `3c6a457`): first right turn driven
Every gate passed but `junction_footprint`: the car waited at the real mouth, committed
a 12 m arc on a kerb-measured exit line, drove it, handed back to the kerb corridor on the
exit lane and reached the goal (exit lane keeping p95 within 0.35 m, no collision). The
footprint gate failed by 0.50 m: the arc ran ~1.1 m left of the true exit lane. The exit
kerb had been measured once, from 25–35 m (`kerb@1.46` at t=21), and never again: the
latched path was never blocked (shoulder clearance 1.45–2.2 m), so the better readings
from closer by were never used, and the lane offset at the arc end was +1.1 m before the
kerb corridor corrected it to +0.03 m.

**Fix: the committed path follows the exit measurement while the car is on the arc.**
Each tick before the arc end the exit line is re-measured; when it has moved by more than
`refine_shift_m` (0.5 m, above the synthetic re-measurement scatter of ±0.3 m) across its own direction, on at least as much kerb evidence as
the latched line had, the path is re-anchored (`replan_same_arc(min_shift_m=…)`). Keeping
the tangent point moves the arc's start along the entry straight, so once the car is in
the arc the re-anchored path no longer passes under it (on the synthetic left turns it
jumped 0.8 m sideways); when the re-anchored path is more than `refine_max_jump_m`
(0.25 m) from the car, the refinement is instead one fresh arc from the car's own pose to
the new exit line (`_mid_arc`, the geometry `replan_mid_arc` uses). Synthetic turns: 11
of 12 noisy seeds complete as before (refinement fired on 9 of them, offsets at the
arc end unchanged or better); left turn seed 2 stops in the junction box, as it did at
`6773e24` (commits a 15 m arc on a knife-edge 0.3 m clearance that is then blocked), a
known synthetic failure to revisit before the live left turns.

### Third live turn (24 Oct, PC 2, `7b1e3c6`): same path, refinement did not fire
Identical to the second run to the centimetre (same seed, same world) and the refinement
never moved the path: the console showed `exit=None kerb@-` from the moment the turn was
committed. Two changes so the next run tells why and acts sooner:
- The refinement's verdict is printed each tick in brackets (`[exit kerb not read from
  here]`, `[exit line +0.9 m off for 0.3 s]`, `[… re-anchored]`, `[… fresh arc from here
  (not clear)]`); `turn_refine` in the drive log.
- The evidence test is temporal, not a station count: a re-measured line that differs from
  the latched by more than `refine_shift_m` replaces it once it has held within
  `refine_candidate_m` (0.3 m) of itself for `refine_persist_s` (0.5 s). The kerb estimate
  jumps by a metre from one tick to the next as a far station enters or leaves the fit;
  on the synthetic right turn (seed 5308) the line committed on was 1.15 m off the one
  read every tick afterwards, and the station-count rule (fewer stations from closer in)
  blocked the correction for 3 s. It now fires 0.5 s after commit.

### Fourth live turn (24 Oct, PC 2, `b4959ea`): refinement fired, footprint still +0.47 m
The console verdicts showed what the exit estimate does from inside the junction: right
after commit a reading 2.7 m off held for 0.5 s and re-anchored the path (the trajectory
hardly changed, as the car was still on the entry straight), then readings within ±0.3 m
of the latched for five seconds, then at t=28 (car two thirds round the arc) a reading
0.86 m right of the latched replaced it with a fresh 8.9 m arc from the car's pose: lane
offset at the arc end 0.28 m against 1.1 m before, but the corner excess had already
peaked at +0.47 m. The commit measurement (`kerb@1.5`, 115 stations, MAD 0.014, read at
27 m) was consistent and wrong by ~1.3 m — the true exit kerb is 4.385 m from the lane
centre (site truth), the car's keep-right 4.204 m, and the path 1.1 m left of the lane.
A consistent error at range is a bias of the kerb-edge rule (the road's last ROAD cell
read as the kerb; from 27 m the unseen strip between the last road return and the kerb
lies on the road side), the opposite of the synthetic world's geometry that chose it.

**Instead of guessing on the next live run, the harness now records the sensor frames**
(`--record-frames turns` by default: LiDAR as int16 cm, each radar's returns as float32
since 29 Oct, GNSS, IMU, speed and status per tick under `<run>/frames`, ~95 KB a tick) and `turn_truth.json` (the site's lane
centrelines and kerb distances), and the drive log carries the latched and measured exit
points and the kerb stations in odometry. `scripts/replay_mapless_frames.py` runs the
same driver over the recording open loop, ties odometry to truth by a rigid fit, and
reports the kerb-station error and lane error per tick; configuration variants
(`--junction exit_kerb_edge='"midpoint"'`) run against the same frames. The smoke test
checks the replay reproduces the live driver tick for tick.

### Fifth live turn (24 Oct, PC 2, `3c4177e`), recorded: the cause, found by replay
Same result (footprint +0.49 m), but this run carried the sensor frames. The replay
(`scripts/replay_mapless_frames.py`, rigid fit odometry→truth RMS 0.28 m) reproduces the
live driver tick for tick and scores the exit-kerb stations against the site's true kerb:

| rule for the kerb within the unseen strip | kerb error at commit (t=21, 27 m) | lane error | at t=29 |
|---|---|---|---|
| `road` (the road's last ROAD cell; the rule shipped) | −1.20 m (into the road) | −1.04 m | −0.32 m |
| `midpoint` | −0.57 m | −0.41 m | −0.32 m |
| `barrier` / `raised` (the first raised cell beyond the road) | −0.16 m | +0.01 m | −0.19 m |

(Scored through one global rigid fit, RMS 0.28 m; the replay now takes each tick's own
odometry/truth pose pair instead, which reads the `raised` rule at −0.08 m kerb, +0.07 m
lane and a latched line 0.34 m right of truth — the same picture without the drift.)

The grid across the exit kerb at t=21 (`kerb_cells.py`): BARRIER cells sit within a cell
of the true kerb line, then ~1.5 m of unseen cells on the *road* side, then the road.
The point classes say why: the road near the kerb gets a return only where a ring lands
(8 road points within 2 m of the kerb in the whole sweep at 28 m), while the pavement
behind the kerb is furnished — hedges, poles, trees — and returns raised and obstacle
points right behind the kerb line. The `road` rule was chosen on the synthetic world,
whose pavements are bare: there nothing raised is seen for metres behind a 0.15 m kerb
at range (its shadow), so the first raised cell lies a metre onto the pavement and the
road's edge is the better kerb (`edge_eval.py`: `road` 0.0–0.4 m, `barrier` +0.8–1.1 m on
right turns). Two worlds, two answers; the live one decides. Taken:
- **Kerb = the first cell with fresh raised evidence beyond the road** (`exit_kerb_edge`
  `"raised"`; the kerb face returns raised points even where a cell's road returns outvote
  them into ROAD, `TerrainGrid.raised_evidence`). On the recording the lane is placed
  within 0.03 m of truth from the first measurement and the latched line is 0.22 m right
  of truth (toward the kerb — the side the clearance checks guard).
- **The synthetic side roads carry pavement furniture** (`side_road_hedges`: a 0.25 m
  verge 0.3 m behind both kerbs — raised, not an obstacle, so its cells read BARRIER as
  the live kerb's do), which halves the bare-pavement bias on synthetic right turns
  (+0.3 m). The synthetic left turn reads its exit kerb from ~38 m across the box, where
  the verge's raised returns lie up to a metre behind the kerb: it commits ~1.3 m right of
  truth and swings wide in the box before the refinement and hand-back correct it (test
  bound 2.6 m, noted). The live footprint gate is the measure; the synthetic world's
  kerb-at-range behaviour is a known limit of that world.
- **The latched exit line is the measured one** the path was built on; the route point is
  kept separately as the probe the exit road is searched around (external review of
  `7b1e3c6`: the refinement had compared new measurements against the route point, 2.7 m
  from the lane, so its first verdict was a spurious "2.7 m off" and the re-anchor a
  no-op). When no same-radius arc reaches a moved exit line from the entry straight, the
  refinement falls through to a fresh arc from the car instead of returning.

### Sixth live turn (25 Oct, PC 1, `dc21548`): **right turn MEASURED_PASS, every gate**
Lane offset through the box within ±0.17 m (was +1.1 m), corner excess at most −0.49 m,
exit kerb read at 2.5–2.8 m from the probe at commit (was 1.5), no refinement needed:
the replay's kerb rule confirmed live. (PC 2 and PC 3 stopped launching CARLA after the
overnight sign-out — "Fatal error!" before any log, same driver 596.71 as PC 1, which
was never signed out and still runs it: an IT matter, not ours.)

**Left turn (Town03, `--approach-m 30 --exit-m 20`): MEASURED_PASS, every gate.** Corner
excess at most −0.18 m, exit lane keeping p95 0.05 m; the in-box offset bulged to +1.0 m
(left, away from the kerb). Town10HD has no left-turn site for the filters (the 8 left
exits fail `no_matching_exit`); the first Town03 attempt sat on a hill and the planar
adapter aborted it at 5° of tilt two thirds round the box — the harness now rejects sites
with any waypoint steeper than `max_slope_deg` 3°, ends the run with a report when the
adapter refuses the vehicle state, and removes an empty output folder when setup fails.

Both left-turn recordings replayed: the exit kerb, read across the box from ~40 m with
the `raised` rule, lies 1.3–1.6 m **beyond** the true kerb (the Town03 pavement there is
wide and bare — the furniture is metres behind the kerb, the bare-pavement geometry the
synthetic world had warned of), so the lane is placed 1.3 m right of truth and the car
drives the bulge; the `road` and `midpoint` rules do not find that kerb at all from
there and would have held the car at the mouth. The site's kerb distance is 6.4 m
(parking + two shoulders), which stretches every measurement. Passing with the junction
tolerance absorbing the bulge is not the same as right: a closer-range correction for
far kerbs (the refinement saw the line move +0.3 m, under its 0.5 m threshold) is the
next G1c refinement, with both recordings as the offline cases.

### G2 segmenter decision (25 Oct, PC 1, RTX 4000 Ada): SegFormer-B0
Trained on the six accepted captures (48k images, Town02's 8k held out), 8 epochs,
512×256. SegFormer-B0 (pretrained MiT-B0 encoder) best epoch 6: mIoU 0.870, road 0.972,
pavement 0.888, marking 0.690 (noisy, 0.44–0.69 across epochs: thin and over-painted,
recall 0.95), road_user 0.834, pavement boundary within 2 px 0.826, 5.5 ms/image. The
BiSeNetV2 run (from scratch, batch 16) was killed at epoch 6 by a branch checkout in the
same clone (its loader workers re-import the script on Windows spawn), but at equal epoch
4 it was behind on everything that matters — mIoU 0.788 vs 0.846, boundary 0.74 vs 0.82,
marking 0.45 vs 0.64 — at 2.2 vs 5.4 ms/image. **SegFormer-B0 is the segmenter**
(`models/road_seg_b0.pt`, epoch 6; `_last.pt` beside it). Training now keeps its loader
workers alive across epochs and writes the best checkpoint after every improving epoch
(provenance `selection.in_progress`), so a run that dies keeps its best. The BiSeNetV2
rerun is only for the report's comparison table.

**These numbers are obsolete.** `road_seg_b0.pt` was trained on a tag table that read
CARLA 0.9.16's tag 8 (TrafficSign) as pavement and left tag 25 (Ground) as "other" (review
of `3e9083a`). The car's loader now refuses it (its recorded `label_mapping` differs);
`--evaluate` scores it on today's labels for the comparison.

### Segmenter retrained on the corrected labels (27 Oct, PC 1): `road_seg_b0_v2.pt`

SegFormer-B0, same six captures (48k training images from Town01/03/04/05/06, Town02's 8k
for validation; Town07 and Town10HD refused), 8 epochs at a constant rate, best epoch 6
saved:

| Class | IoU | Recall | Precision |
|---|---|---|---|
| road | 0.974 | 0.978 | 0.995 |
| marking | 0.712 | 0.861 | 0.804 |
| pavement | 0.921 | 0.980 | 0.939 |
| road_user | 0.816 | 0.985 | 0.826 |
| other | 0.975 | 0.978 | 0.997 |

mIoU 0.879; pavement boundary within 2 px 0.842 (recall); 2.3 ms/image (RTX 4000 Ada, no
CARLA running). Precision here is derived from IoU and recall; the training and
`--evaluate` reports now print it, with the boundary's precision. The constant rate was
unstable: loss spikes in epochs 4 and 7, and in epoch 8 the marking class died (IoU 0.0,
mIoU 0.729), so `road_seg_b0_v2_last.pt` is not for use. Training now decays the rate
(cosine per step to 5%, `--schedule constant` for the old behaviour). Marking stays the
weakest class: high recall, lower precision (over-painted thin lines).

Old against retrained on the same corrected Town02 labels (`--evaluate`, 8,000 images):

| | `road_seg_b0.pt` (old table) | `road_seg_b0_v2.pt` |
|---|---|---|
| mIoU | 0.874 | **0.879** |
| pavement IoU / precision / recall | 0.904 / 0.915 / 0.987 | **0.921 / 0.939** / 0.980 |
| pavement boundary (2 px) precision / recall | 0.838 / 0.829 | **0.911 / 0.842** |
| marking IoU / precision / recall | 0.690 / 0.714 / 0.953 | **0.712 / 0.804** / 0.861 |
| road_user IoU / precision / recall | **0.834 / 0.846** / 0.983 | 0.816 / 0.826 / 0.985 |
| road IoU | 0.972 | 0.974 |

The gain is where the car uses it: pavement boundary precision 0.838 → 0.911 (the old
model painted pavement on sign panels, so it drew kerb edges that are not there).
Road users are slightly worse (0.82 against 0.83 precision); mIoU alone (+0.005) hides
both. `road_seg_b0_v2.pt` is the segmenter.

### Fresh-eyes review of `6773e24..58adaf3` (25 Oct, three independent reads): all taken
Turn code (`junction.py`, `driver.py`):
1. **A mid-arc re-plan committed a path its own follow then rejected**, and the latch had
   already been swapped, so every later follow failed and the car stopped for good. The
   re-plan solves the radius so the tangent point is at the car; `_arc_path` emitted a
   zero-length first segment whose arbitrary heading swung the footprint over the kerb
   (rounding decided which half of the re-plans failed). No degenerate segment now, the
   swept check takes headings from real segments, and every re-latch goes through
   `_relatch`: if the follow rejects the new path the old latch stands.
2. **A block far along the exit straight stopped the car dead at the mouth** (a 1.5 m wall
   on the approach pavement tilted the lane-heading prior 4°, the exit straight read a kerb
   cell 28 m out, all repairs failed identically, and at a standstill nothing changes, so
   nothing released it — 71 s). `follow` tolerates a hazard beyond the stopping envelope
   (`near_cover_m`): the path is verified to a metre short of it, the car approaches and
   keeps measuring. A mid-arc repair tries the heading prior first and the stations'
   fitted heading second (a prior a few degrees off is the likeliest cause of a block at
   the exit straight; on straight kerbs the fitted slope is range bias, so second).
3. **`raised_evidence` ignored the tall veto**: a parked car flush with the exit kerb
   became the kerb (lane 1.9 m into the road). The veto `labels()` applies now applies.
4. The jump guard and the fresh-arc fallback apply to the repair as well as the refinement
   (`fresh_arc` in both modes); `_mid_arc` refreshes the entry frame so the next refinement
   rebuilds from the new arc; a reading hovering about the refinement threshold no longer
   resets the candidate every other tick (dropped only within 0.2 m of the latched); the
   "read from nearer" rule the comment promised but the code did not apply is dropped from
   the comment (`range_m` is reported only); `_moving_t` is updated before the early
   returns (a fault while moving must not stretch a later standstill's freshness); the
   junction brake hold applies only at the hold point itself (`junction_hold_within_m`
   0.4 m — a car stopped a metre short of a verified frontier at a standstill never saw
   the frontier move); the log keeps at most 60 kerb stations a tick (was 800, 11 KB).
   Tests: a followable mid-arc re-plan with latch rollback, the wall scenario finishing,
   the tall veto, and the latch test allows the radius to change once per re-plan.

Harness, recorder, replay (`mapless_drive.py`, `frame_record.py`, `mapless_replay.py`):
5. **A crossing walker's yield counted towards `--hold-s`**, so the run ended the tick the
   walker cleared, with no ticks for the car to resume (completion FAIL on a correct
   yield). Hold ticks count only while a hold may end the run.
6. **The replay's one global rigid fit smeared odometry drift** (0.3–0.5 m, up to a degree
   over the run) into the kerb and lane errors; each tick's own odometry/truth pose pair
   now takes estimates to the core frame (the global fit stays as a sanity figure).
7. The recording is self-contained (the suite's full spec and sha in `meta.json`; the
   replay no longer needs the config file or the report's 3-digit-rounded overrides);
   `frames.jsonl` is flushed per row; int16 centimetres are checked against the LiDAR's
   reach; a second LiDAR's status is dropped on load; ASCII-only console (Windows
   cp1252); `HoldingVehicleAdapter` tolerance 0.2 m/s (ten times the roll-back seen).

Stack and training (`stack.py`, `prediction.py`, `train_segmentation.py`, `segmentation_data.py`):
8. **Fresh tracks tapped the brakes**: a just-confirmed track carries the filter's initial
   5 m/s velocity sigma, which the prediction inflated to a 30 m region at 3 s, and every
   new parked-car cluster capped the speed to zero for a few ticks 26 m out. Only
   confirmed/coasting tracks are predicted onto the path, and the velocity sigma counted
   is bounded (`max_velocity_sigma_mps` 1.5). Test: no zero cap with the parked car more
   than 15 m ahead. The veto label now names its cause (lead at a standstill, following,
   predicted conflict) instead of always "predicted conflict".
9. Training: `drop_last` on the training loader (BiSeNetV2's context embedding batch-norms
   a 1×1 map and raises on a final batch of one — latent, every epoch had identical
   batches); the dataset holds only data so Windows `--workers > 0` can pickle it; a
   progress line every 200 steps with images/s and ETA, `seconds` per epoch.

Left as noted limits: the raised rule takes the first raised cell up to 3 m behind the
road (`exit_max_unseen_m`), so a bare pavement with furniture 2.5 m back would place the
lane 2.6 m toward the pavement — the clearance checks refuse such a path rather than drive
it; `braking_lead`'s hold passes with any gap ≥ 1 m; a walker that never triggers leaves
the run to `--max-s`.

### External review of `6773e24` (ChatGPT, 24 Oct): nine findings, seven taken
Taken, each with tests:
1. **Verified turn space** — within the stopping envelope (`near_cover_m` 8 m) a pose is
   verified only when observed road spans the body's width (`near_column_share` 0.8 of
   the footprint's lateral columns hold a ROAD cell), with the 1 m gap allowance only;
   beyond it the 40% share rule stands as tentative support. A half-unknown footprint
   and an unknown centre strip no longer verify (`test_mapless_swept_check.py`).
2. **Conflict check covers the whole car** — `first_footprint_conflict`: oriented ego
   box at time-aligned poses along the path (time zero at the car, not the corridor's
   first sample), tested at each prediction step's window ends and every path sample
   inside it. An object under the bonnet is a conflict at the first step.
3. **Prediction uncertainty in the object's frame** — P_pos(t) = P_pp + t(P_pv + P_vp) +
   t²P_vv + (q t³/3)I rotated into the heading frame; a scene rotated 90° gives the same
   footprints; a 3 m along-track σ inflates by 6 m along, 0.2 m across.
4. **Stack follows the frame rebuild** — `reset_spatial()` on the driver's `frame_rebuilt`
   (tracker, road model, following, cached segmentation, cap); nothing added while
   `rebuilding`; `DecisionRecord.frame_generation`.
5. **Segment and paint the same frame** — painting only on segmentation ticks with a
   `SegmentedFrame(frame_id, …)`; `paint` refuses another frame's segmentation; the road
   model ages its evidence between.
6. **No confirmation bypass** — only a previously confirmed track coasts; a tentative miss
   stays TENTATIVE (hazard, never a lead); `lead_ahead` accepts CONFIRMED/COASTING only.
7. **Moving-scenario evaluator** — separation is the oriented-polygon distance between
   the ego box and the actor's bounding box (its local offset included), overlap → 0 →
   FAIL, no measurements → INCOMPLETE; completion per scenario: `crossing_pedestrian`
   must yield, then resume and reach the goal after the walker clears (a hold ends the
   run only once the walker is past the lane); `cut_in` must settle a safe gap and reach
   the goal; `braking_lead` may end in a sustained safe hold. `required` is reported.
   Training: `index_runs` requires integrity and geometry PASS from the capture
   inspection (`allow_unaccepted` for smoke), best validated checkpoint saved (`_last.pt`
   kept; no validation → `_smoke.pt`, provenance `validated: false`), class weights
   sampled across runs and cameras, synchronised warmed-up GPU timing, seed recorded;
   the stack records `cap_applied_mps` (this command) and `speed_cap_next_mps`.

Documented as limits, not patched: (8) the semantic road model is built but does not yet
select the corridor or the exit lane — the turn still places the exit lane `keep_right`
left of the exit kerb, so training a better segmenter alone does not solve shoulder →
plain-lane transitions; the matching-kerb site filter is a test-domain restriction and
those transitions are unsupported until the G2 lane-position milestone. The near-cover
rule is coverage, not visibility reasoning: an occluded exit still reads as unknown and
holds the car at the verified distance, which is the intended conservative behaviour.

### First live stack runs (25–26 Oct, PC 1 and PC 2, Town10HD and Town05)

Each failure below was found from the run's log (and, from `405b056`, the `cap_track`
field naming the track behind a conflict or yield) and replayed offline where the sweeps
were saved.

| Run | Symptom | Cause | Fix |
|---|---|---|---|
| straight B, site 0 | held at the start | a parked object 28 m ahead; stationary tracks charged a moving road user's process noise; centroid jitter read as motion; ~390 radar-only tracks from pavement returns | `--site 1`; `b7b50bf`, `8140f4c`, `abf5e3a` |
| straight B6 | 0.3–1 m/s sawtooth | radar returns from the tarmac clustered as a lead 5 m ahead; a 0.006 brake stopped the car dead | `278e4f9` (radar height window, coasting deadband) |
| stop B1 | stopped 6.3 m short, never held | the parked lead was a "predicted conflict" before the driver's stop point | `47e693c` (a stopped lead is the obstacle stop's) |
| cut-in B1 | held at s = 17 m | a car parked at the kerb tracked as a conflict; the cut-in car never cut in | `4d38a03` (static world left to the corridor check; scripted approach) |
| pedestrian B1–B3 | no yield, then false yields | a crossing walker read as a stopped lead; no yield rule; uniform classes broke ties on "car"; walls sliding past the car read as walkers | `7998276`, `e8ef4a1`, `405b056`, `c5f0b74` |
| braking lead, Town05 | crash on the first tick; then "kerb not seen" | no-corridor cap returned None; a railing on the kerb vetoed every calibration station | `2d85cd5`, `066d85b` |
| braking lead B3 and 4 m/s A5, Town05 (railing calibration live: 5.70 m) | stop-and-go behind a lead under 3 m away (negative for 100–145 ticks); both ended with a tilt refusal at the same spot | on a grade the road rising ahead and behind read as OBSTACLE and single linkage chained it with both kerb railings into one 63 x 26 m cluster; its mean sat 2.8 m ahead in the lane and it was tracked as a 12 x 12 m vehicle (external review, confirmed by replaying the saved sweeps) | straight and actor sites reject grades (3° max, 2° change); clusters larger than a road user (12 x 4 m) are re-split at a tight link and the rest is structure, never a track; the tilt refusal logs pitch, roll, position and speed |
| pedestrian B6 / cut-in B3 (on `c5f0b74` or earlier: started the minute `081cd04` was pushed) | proactive yield scored as none; false yields to the cut-in car and to a one-tick fragment | M04 gate took the already-slowed speed as reference; the yield took a car's rear face (under 2 m) for a walker and acted on single ticks | gate reference = fastest speed in the 3 s before entry; yield only to crossers (across the path at 0.3 m/s or more) at pedestrian speed (2.5 m/s max) after 0.3 s |
| braking lead B4 (Town10HD), 4 m/s A6/B6 (Town05), cut-in B4, pedestrian B7, all on `6b1f705` | all MEASURED_PASS (yield 1.99 to 0.14 m/s and resume; no yield during the cut-in); at 4 m/s, stop-and-go for the first 10 s; stop-and-go in the last 10 m behind the stopped lead | an 80 x 7 m kerb railing tracked as a 12 x 7 m object moving with the car (71 conflict ticks in both runs); 0.3 m pieces of the lead's roof and wheels 1.5–4 m beyond its rear face jittering sideways, and once the lead duplicated as a fresh LiDAR-only track | the split above (thin pieces of a structure are structure too); tracks inside the lead's body are not conflict candidates |
| braking lead B8 (Town05, 4 m/s, lead at 2 m/s braking at 90 m), on `b8c3a46` | MEASURED_PASS; clean start (the railing fix); a stop at 11 s and a slowdown at 23 s | far-right scenery (13–23 m off the path, out of the radar's view) with centroid motion of 4–6 m/s read as vehicles heading for the path; the lead tracked twice, radar-only (followed) and LiDAR (a 'crossing walker'); 148 tracks, 45 ms tracking, 81 ms clustering | radar twins merged into LiDAR tracks; near static radar-only clutter dropped; one linking pass for the split (two passes cost ~18 ms); exact radar Jacobian and vectorised gates. Open: LiDAR-only motion of extended scenery (needs shape/point registration, not a centroid) |
| braking lead A9/B9 (Town05, 4 m/s, lead 2 m/s braking at 90 m), on `44e4f96` | both MEASURED_PASS, 3.48 m clearance; first A/B table above; both stopped once at ~10 s and stopped short twice as the lead came to rest | walls 13–23 m right read as cars heading for the path (radar blind there); the radar's reading of the lead braking rejected nine ticks running; the stopping lead's rear face 'uncertain', so its own envelope was a conflict | vehicles cannot reach the path through a kerb, railing or object (`road_reachable`, terrain grid); persistent one-sided radial speed is a manoeuvre; a known or car-wide lead at rest is the obstacle stop's; far static radar spawns only within 8 m of the line ahead |
| braking lead A10/B10 (Town10HD site 1, 2 m/s), on `44e4f96` | both MEASURED_PASS (3.46/3.49 m); A braked from 2 m/s to rest at 19.6 s, B stopped at 6 s and 17 s | a 0.3 m piece of the lead 1.5 m to its side for three ticks (zero cap); a 3.3 x 2.2 m roadside object at 6.5 m right, its position pulled between the radar's corner return and the LiDAR centroid (0.4-0.5 m/s) | graded conflict cap (`conflict_speed_cap`, 2 m/s^2, 2 m short); radar radial speed only for LiDAR-placed tracks; lead body margin 0.8 m; `stopped_before_lead_s` counts only after the car first moved |
| braking lead 4 m/s A10/B10 (Town05), on `cabfd4e` | both MEASURED_PASS; radar A/B table above (braking seen 0.15 s against 0.60 s); the walls no longer stop the car (`unreachable` on 211–239 ticks) | A: a railing piece within kerb + 2 m approaching at the car's speed (7 ticks); B: a piece 2.7 m beyond the followed lead's rear face, its footprint narrower than a car's rear | a followed lead is a body whatever its measured footprint; the graded conflict cap |

### External review of `405b056` (ChatGPT, 26 Oct): all five high-priority findings taken

- A tick without perception kept no cap: the restriction now holds until perception
  re-evaluates it.
- Too little evidence of motion was treated as standing still: tracks are static,
  moving or uncertain; uncertain ones are predicted where they stand with velocity
  uncertainty capped at walking pace and are tested in the timed conflict check (not the
  yield), and static ones are left to the corridor check.
- Velocity uncertainty was dropped when the filter was unsettled, so more uncertainty
  gave a smaller envelope: it is now capped, never dropped.
- A diagonal walker selected as the lead was followed and dropped from the crossing
  checks: `lead_role` decides on motion along and across the path at the lead, and a
  non-vehicle lead stays in the crossing checks.
- The following gap ran to the lead's centre and its speed along the car's axis: the
  gap is to the footprint's rear along the path and the speed is along the path tangent.
- Smaller items taken: coasting only for an excess up to 0.3 m/s; a sensor supports a
  track for 0.5 s after its last update; run reports carry the commit, argv, segmenter
  checkpoint hash and stack configuration hash; the junction-hold test allows numerical
  noise but bounds creep.
- Deferred to the next G1c/G2 work: exit-line uncertainty and road/raised disagreement in
  the turn diagnostics, separate furnished and bare-pavement synthetic cases, lane
  markings in lane placement, and coverage that scales with speed.

### LiDAR and radar roles (26 Oct, after the external review of the sensor dynamic)

- **LiDAR** gives the geometry: the corridor, the obstacle stop (from its points, never
  from tracks) and object candidates. A cluster larger than a road user is split, and
  what stays larger is structure: never a track, still an obstacle where its points lie.
- **Front radar** gives range, bearing and radial speed, lever-arm compensated. It
  updates the tracks the LiDAR forms, and starts radar-only tracks only where it adds
  something: beyond 30 m (where the LiDAR first clustered a car on Town05) or for a
  moving return. A static return near the car that no track claims is clutter, and so
  is a static radar-only track carried in from beyond 30 m that the LiDAR has not
  confirmed by 25 m (B8 still carried 148 tracks after the spawn rule alone). A
  radar-only track inside a LiDAR track's footprint (grown by 1 m) is the same object:
  merged into it, the older id kept (B8: the lead was followed as radar-only track 166
  from 32 m to 12 m while its LiDAR twin read as a crossing walker and stopped the car).
- **Disagreement: an outlier is rejected, a persistent one is a manoeuvre.** A radial
  speed the track's velocity cannot explain updates the position only and is counted
  (`doppler_rejects`, in the lead and `cap_track` notes; `radar_stats` lists the
  innovations). Three running with the same sign (0.15 s) is the object's motion
  changing: the velocity variance along the line of sight is opened by the innovation
  and the radial speed is used (`doppler_manoeuvres`). B9 showed why: the lead stopped
  within 0.2 s, the radar read it 1.7 m/s slower nine ticks running, every reading was
  rejected, and the track slowed only as fast as the LiDAR positions allowed (0.6 s);
  replayed, the track now reads ~0 by the third update. Only the radial component moves,
  so a walker crossing square on (radial ~0) keeps its crossing speed.
- **Cameras** run the road segmenter; the object detector (`--detector yolo`) has not
  been run live yet, so classes come from footprint and motion alone.
- **Rear radars** (admission C) feed the tracker but have no consumer until lane
  changes; their tests (approaching, alongside, leaving the blind spot) belong to G4.
- **First evidence for the radar (B8, lead at 2 m/s):** a lead selected at 32 m, beyond
  the LiDAR's object range, its speed read 1.8–1.9 m/s (true 2.0) and followed.
- **First A/B (A9 against B9, Town05, 4 m/s, lead at 2 m/s braking at 90 m, `44e4f96`):**

  | | A (LiDAR) | B (+ front radar) |
  |---|---|---|
  | first lead (time, true gap) | 9.0 s, 32.5 m | 8.85 s, 32.8 m |
  | lead-speed error RMSE / p95 | 0.29 / 0.37 m/s | 0.26 / 0.13 m/s |
  | braking seen after | 0.45 s | 0.50 s (radial speeds rejected; fixed above) |
  | stopped while the lead moved | 0.45 s | 1.05 s (scenery out of the radar's view) |
  | minimum clearance | 3.48 m | 3.48 m |
  | tracking p50 (tracks) | 4 ms (58) | 16 ms (130); was 45 ms before `44e4f96` |

  The radar did not see the lead earlier (the LiDAR clustered it at 32.5 m as well) and
  its one clear gain so far is the speed error's tail. The braking delay is the
  measurement it should win, now that a persistent radial speed is used.
- **Third A/B, the one that shows the radar's value (A10 against B10, Town05, 4 m/s,
  lead at 2 m/s braking at 90 m, `cabfd4e`, manoeuvre rule in):**

  | | A (LiDAR) | B (+ front radar) | change |
  |---|---|---|---|
  | braking seen after | 0.60 s | 0.15 s | 75% sooner |
  | lead-speed error RMSE / p95 | 0.49 / 0.99 m/s | 0.10 / 0.08 m/s | 80% / 92% lower |
  | first lead (time, true gap) | 9.0 s, 32.5 m | 8.85 s, 32.8 m | none |
  | stopped while the lead moved | 0.45 s | 0.75 s | one false stop each (fixed below) |
  | minimum clearance | 3.48 m | 3.51 m | none |
  | step p50 / p95 | 127 / 269 ms | 129 / 252 ms | +2 ms p50, p95 lower |
  | tracking p50 (tracks) | 4 ms (58) | 11 ms (86) | +7 ms |

  Against plan section 10's admission (at least 20% lower velocity error or braking
  delay, no extra adverse events, at most 5 ms of added p95 compute): the error and the
  delay pass by a wide margin, the step's p95 did not grow, and the one extra stop was a
  piece of the followed lead (now inside its body). One pair on one site: to repeat.
- **Second A/B (A10 against B10, Town10HD site 1, 2 m/s, lead at 3 m/s braking at 45 m,
  `44e4f96`):** first lead the same (2.05 s, 18.5 m: inside LiDAR range); lead-speed
  error RMSE/p95 0.40/0.19 against 0.37/0.11 m/s; braking seen after 0.45 s against
  0.35 s (22% sooner, before the manoeuvre rule); tracking 7 ms (81 tracks) against
  20 ms (133); step p50 46 against 68 ms. Both stopped once before the lead did: A on a
  three-tick piece of the lead 9 m ahead (a zero cap braked it from 2 m/s to rest), B on
  a roadside object whose position the radar and the LiDAR pulled between them. Fixed:
  a conflict caps to a speed that stops short of it (zero only up close); a track the
  LiDAR placed this tick takes only the radar's radial speed, the LiDAR giving position
  and extent; the lead's body margin is 0.8 m.
- **Measurement:** every stack braking-lead and cut-in run now reports `lead_tracking`
  (first lead and the true gap then, lead-speed error RMSE/p95, gap error, brake
  detection delay, stops while the lead still moved, sensor support), and the log
  carries the actor's truth per tick. Next: admission A against B on the same site and
  actor parameters, with compute.

### Stage 1, reliable stopping (27 Oct; offline, and the G1b regression live)

After the full review of `73ae759` and the external review of it, the agreed order is:
1 reliable stopping, 2 object identity, 3 junction commitment, 4 honest evaluation,
5 sensor benefit. Stage 1 is done offline; live, only the G1b regression has been run.

**Live regression (PC 2, Town10HD_Opt, `8653753`): stop and curve MEASURED_PASS**, as on
23 Oct. Stop: held 3.51 m short (3.5 m before), lane p95 0.11 m, worst corner 1.28 m,
no faults; standing, the corridor grew from 9 to 13 m on accepted refits. Curve: lane p95
0.06 m, max 0.064 m, worst corner 1.32 m (identical), goal within 2.8 m; step p95 25 ms.
The stop log showed one more fault, fixed after the run: a missed parked car 27 m out
on a 24 m line put its held stop at the line's end, 3.9 m short (a projection past the
end clamps to the end point, and its "lateral" is the distance to it, so an obstacle to
one side of the line's continuation could be dropped as off the path). Holds are now
measured along the line continued past its end.

- **One hold per obstacle (S2).** A detection within 2 m (along the path) of a held
  obstacle is that obstacle; anything else gets its own hold. The stop is the nearest
  hold's. Before, one sweep that missed the held obstacle 7 m ahead but saw another at
  22 m moved the stop to 15.7 m and forgot the first.
- **Clearance over the obstacle's own region (S1).** CLEAR also needs ground in every
  0.3 m lateral bin across the obstacle's span (±0.3 m), in the 0.5 m just behind its near
  face, where it stood. Ground beside it, or in front of it, proves nothing; missing
  evidence is UNKNOWN. A 0.5 m box at the lane centre and a car over the right half of
  the footprint each "cleared" while still there before. The external review's sparse
  sampling concern was real: one sweep put ground in that strip only within about 6 m
  (rings 3 m apart at 18 m), and in the live stack runs on `e93154f` a walker who had
  crossed out of the lane, and a lead that had driven on, held the car almost to a stop
  (0.06 and 0.57 m/s). Each hold now gathers the in-tube ground points of every sweep
  since its obstacle was last detected (odometry frame), and the strip is judged on all
  of them: as the car drives, the rings sweep the strip; an obstacle still there keeps it
  in shadow. A closed-loop test: down to 0.23 m/s before, over 1.5 m/s now; the
  missed-obstacle cases stay held. Judging it on the terrain grid's cell hit times instead
  was unsafe: 0.25 m cells leaked the ground beside and in front of a 0.5 m box into its
  bins, and the car drove into it. The log names what a hold still lacks (`hold_note`:
  region bins with ground, points gathered, clearance against what it needs).
- **Holds across a standstill rebuild (S3).** If the car has not moved since odometry
  dropped out, the rebuild keeps the holds (clearance count restarted, UNKNOWN).
- **The stop comes from the obstacle.** A hold keeps the obstacle's near face across its
  span, in odometry, and its stop is worked out on each tick's path. A stop point kept on
  an old centreline projected 2.4 m behind the car when the path was refitted, and released.
- **Swept body on bends.** The tube widens on the outside by front² · |k| / 2 (0.59 m at
  12 m radius, at most 1 m), for detection and for the hold region (`tube_bounds`).
- **Low objects (S4).** RAISED points count under the body's full width (was the inner
  0.85 m): 12 up to 5 m from the axle, then 12 · (5 / d)² and never under 3. The
  32-channel roof LiDAR gives a 0.25 m brick 17 raised points at 6 m, 5 at 9 m, 3 at 11 m
  and none past 20 m. All nine low-object cases now stop about 3.4 m short (first
  detection 11–23 m from the axle). Before, the brick under the left wheel was hit at
  2 m/s, and the other kerb-high ones were never obstacles: each ended in a corridor
  FAULT 1.8–2.0 m short and stayed stuck.
- **Bumper mask.** Points ahead of the bumper are masked only 0.05 m out (was 0.3 m, which
  hid all 272 points of a person standing just in front). Beside and behind the body the
  0.3 m margin stays (mirrors, the rear radars): a person there is beside the path, not in
  it.
- **Corridor at a standstill.** Kerb cells inside the LiDAR's ~4 m blind ring age out
  while standing, and refits went straight across a 12 m bend: the held pole fell out of
  the tube and was hit. Standing still, a refit is taken only if it reaches as far as the
  carried line and lies within 0.25 m of the line the car stopped on (not of the last
  refit: accepted refits otherwise walked 0.28 m in 4 s). Freezing the line outright
  instead left a departed parked car's shadow as the line's end, so its road could never
  be seen clear.
- **Corridor fit and support.** Evidence under 8 m that bends on its own (a line leaves
  over 0.1 m rms and a quadratic halves it) is fitted with a quadratic. A raised strip with
  road within 2 m beyond it no longer ends the support (that is the obstacle check's
  job). Support observed to the line's last 0.5 m sample is the line's end: the sample
  quantisation cut each carried line by about 1 m a tick.

Offline evidence (synthetic kerbed road, roof LiDAR v2, 2 m/s), all on the final code:

| Case | `73ae759` | Now |
|---|---|---|
| Nine low objects in the lane: 0.25 m bricks (centred, under each wheel, 0.7 m off centre), a kerbstone, a 0.4 m box, legs, a toddler-size box, a 1.6 m wide low strip | brick under the left wheel hit at 2 m/s; the centred and right-wheel bricks, kerbstone and low strip: corridor FAULT 1.8–2.0 m short, stuck | all held 3.37–3.41 m short, first seen 10.9–22.9 m from the axle |
| 0.5 m box at the lane centre, its returns removed once held | hit at 2.1 m/s | held, moved 0.0 m |
| The same with a parked car 22 m beyond it | — | held, moved 0.0 m |
| Parked car 0.5 m into the footprint, returns removed once held | hit at 1.75 m/s | held, moved 0.0 m |
| Parked car over the right half of the footprint, returns removed | hit at 2.15 m/s | held, moved 0.0 m |
| 0.3 s IMU or speedometer outage while held at a standstill | drove 1.24 m towards the occluded car | held, moved 0.0 m |
| Pole 1.5 m left of the lane centre, outside of a 12 m bend | no contact (rear axle stopped at s 32.4, pole at 33.1) | held at s 26.9, no contact |
| The same bend without the pole | finished | finished; body corner at most 1.44 m left of the lane centre |
| 3 m deep, 40 m long lay-by | finished, no pavement contact | the same |
| Parked car removed once held | resumed on observed clearance | the same |
| LiDAR OK but empty sweeps from 12 s, at 2 m/s | — | FAULT after 0.95 s and 1.93 m |
| 0.12 m speed hump across the lane | corridor FAULT, stuck | held for as an obstacle |

Known limits: a 0.12 m speed hump across the lane is held for as an obstacle (before: a
corridor fault); the stack cannot tell a traversable hump from a low object. At 4 m/s a
0.25 m brick first seen 13 m from the axle leaves about 1.5 m of the 2.5 m stop gap.
Regression tests: `tests/unit/test_mapless_stage1_stopping.py` (11 unit, 4 closed loop).

### Stage 2, object identity (27 Oct, offline only)

From the review of `3e9083a` (H1, H4, M1, M2, M6, M9) and the external review of the full
review of `73ae759`. Not yet run live.

- **Radar association (H4, M1).** A return associates on range, bearing and radial speed
  together (chi-squared 11.34, allowing 1 m/s of speed change). For a track the LiDAR
  holds, range and bearing are loosened by its footprint's half-extent along and across
  the line of sight, so a return off a car's corner joins the car's track. Before, a return
  joined only within about 0.8 m of the LiDAR centroid, and past that it spawned a
  radar-only twin, merged again on 37–64% of radar ticks in B9 and B10. A return whose
  radial speed fails on its own (6.63) is dropped whole and counted.
- **Manoeuvres (H4).** A braking lead (B9) is still taken from its radial speed by the
  third return. The run must be:
  - three rejections within 0.3 s, all with the same sign;
  - agreeing with the run's first and last innovation;
  - centred on the track: the mean range and bearing innovations within 1.5 sigma.

  The velocity variance opens by at most 10 m/s² × the time since the first rejection,
  not by the innovation squared.
- **Radar twins.** A radar-only track is folded into a LiDAR track only when:
  - it has lain inside that track's footprint for two ticks;
  - their radial speeds agree within 3 sigma plus 0.5 m/s;
  - the LiDAR track is bigger than a walker (1.2 m).

  The two velocity estimates are combined, so the radar's speed is kept. The longer
  history (trail, origin) is kept, moved to the LiDAR position. A radar-held track's first
  LiDAR fix re-anchors it on the cluster and moves its history with it. A walker-sized
  cluster cannot take over a radar-only track moving faster than 6 m/s.
- **Radar clutter.** A static radar-only track near the car is dropped only where the LiDAR
  hit something within 0.3 s (ground included) in its cell or a neighbouring one. Where it
  saw nothing (occluded, between rings), the radar is the only evidence, and the track
  stays.
- **Structure splitting (H1).** A piece split off an oversized cluster is structure only if
  it does not have a road user's shape:
  - still oversized;
  - a line: under 0.5 m wide, at least 2.5 m long and three times its width;
  - a one-ring sliver: under 0.08 m across, or thin and under 0.3 m tall;
  - or wholly above 1.4 m over the road (a wall or hedge top, a branch).

  Narrow alone was the rule before. The sliver and floor rules came from the first live
  Town05 runs on `e93154f`: on the start-up sweep the line rule alone made 66 objects
  where there had been 30, and the extra slivers became tracks and conflicts. With them
  it makes 31–35.
- **Lead body (external review, M9).** A track inside the lead's body is dropped as a piece
  of the lead only if it never was a road user of its own: never seen outside the body for
  0.5 s, and not classed pedestrian or bike. The lead's own pieces are born inside it; a
  walker beside it came from somewhere. The filter now also covers a followed walker or
  cyclist lead, which stays in the crossing checks itself.
- **Reachability (M2).** A vehicle's run to the path is blocked only by a wall:
  - persistent tall evidence (20 or more, about a second of steady hits) that no track's
    footprint covers;
  - across ±1 m of the run;
  - for 3 ticks running.

  A kerb or pavement does not block (a car leaves a driveway over it), and neither does a
  parked car (a track) or another crossing car's fading trail (about 10).

Offline evidence (the reviews' reproductions):

| Case | Before | Now |
|---|---|---|
| A parked car's corner return 2.6 m beside a 5 m/s lead | lead 0.02 m/s | 5.0 m/s |
| A neighbour at 10 m/s beside it | lead 9.97 m/s, and it stays | 5.0 m/s |
| A neighbour 3.3 m beside a radar-only lead, at 7 m/s, its own return missing | pulled 0.8 m sideways, 6.9 m/s | 0.0 m, 5.0 m/s; the neighbour gets its own track |
| Radar point 1.0–2.5 m off the LiDAR centroid (merge churn) | twins spawned and merged (37–64% of radar ticks live) | 30/30 returns associated, none spawned, none merged |
| The radar after a merged lead stops dead | its speed lost (the fresh twin's kept) | 2.0 → 0.16 m/s by the third return |
| Parked car held by the radar, then clustered | "uncertain" about 1 s, travelled 1.8 m for good | static, travelled 0.0 |
| Three rejections spread over 1.5 s | a manoeuvre | not counted |
| Opposite-motion tracks (+5 / −5 m/s) | merged, one velocity | separate |
| A walker-sized cluster beside a radar-held 8 m/s car | took over the car's track | two tracks |
| Pedestrian 0.7 m from a wall, 1.2 m from one at 25 m, cyclist by a fence, pedestrian between parked cars, a lead's rear face by a railing | structure: never tracked | objects |
| Town05 grade sweep (railings, road rising) | no lead | no lead |
| A walker beside the lead, seen 0.6 s on the pavement first | dropped as a piece of the lead | kept |
| A car's run to the path over pavement / through another car's trail / a parked car | dropped | kept |

**Live regression on `e93154f` (27 Oct).** PC 2, Town10HD site 1, admission B, constant
segmenter:
- braking lead, cut-in and crossing pedestrian all MEASURED_PASS;
- braking seen 0.15 s after it began;
- lead-speed error RMSE 0.41 m/s, p95 0.07 m/s;
- no stops before the lead stopped.

They also showed the lingering holds fixed above. PC 1, the Town05 pair at 4 m/s, both
MEASURED_PASS, but not comparable with 26 Oct: without `lead_start_m=50` and
`--goal-m 100` the route ended before the lead braked. Sliding railing tracks and the
extra slivers made 27–39 conflict ticks; the slivers are fixed above. The step was 63–68
ms p50 on PC 2 and 104–116 ms on PC 1, over the 50 ms budget, with the constant
segmenter.

Known limits: a car seen side-on within the link distance of a barrier (about 0.6 m)
is a line, so structure. Requiring a collinear second stretch kept it as an object, but
turned seven more Town05 railing and road-ring stretches back into objects. Its points
stay OBSTACLE points for the driver's stop. A rear face within the tight link (0.5 m) of a
railing chains into it. A rate-rejected return is consumed by its track while that
track's own return is missing. An unclassified walker that first appears inside the lead's
body band is ignored there, where the lead shadows it, until it steps out of it (a test
pins both). Long thin railing tracks that were never split (under 12 m) still slide and
make conflicts on Town05, as before. Regression tests: `tests/unit/test_mapless_stage2_identity.py`.

**Live on `e401188` with the v2 segmenter (27 Oct).** All four runs MEASURED_PASS, and the
reports record `road_seg_b0_v2.pt` (sha256 `e8c38004aa6d…`).
- **PC 2, Town10HD site 1, admission B.**
  - Crossing pedestrian: one yield (0.17 m/s), then 1.95 m/s to the goal. The second
    near-stop is gone.
  - Braking lead: no slowdown on the approach; final gap 3.47 m; lead-speed RMSE 0.18
    m/s, p95 0.05 m/s; braking seen after 0.15 s. The lead still stops before the car
    arrives, so this remains "approach a stopped car" (Stage 4).
- **PC 1, Town05 at 4 m/s, like-for-like** (`--goal-m 100`, `lead_start_m=50`,
  `brake_at_m=90`):

  | | A: no radar | B: front radar |
  |---|---|---|
  | Lead-speed RMSE / p95 | 0.35 / 0.70 m/s | 0.13 / 0.11 m/s |
  | Braking seen after | 0.60 s | 0.15 s |
  | Final gap | 3.46 m | 3.49 m |

  Both stopped for about 1.2 s near 11 s. Car-sized tracks 14–15 m right of the road
  appeared to move towards it at 2–7 m/s. There is no other traffic in the scenario, so
  these are scenery, and the reachability filter does not drop them (open).

Stages 1 and 2 stay open on compute, on actor-matched identity metrics and on the live
stopping edge cases.

### Stage 3, junction commitment (27 Oct, offline only)

From the full review of `73ae759` (S5, S6, S7 and the junction medium items) and the
review of `3e9083a` (H2, H3, H5, M4, M5, M7, M8). The reviewers' reproductions were re-run
in the synthetic world.

- **Oncoming traffic on a left turn (S5).** The body may enter the opposing lane only once
  its approach is clear. That is the lane beside the car's own, 3.5 m wide. "Clear" means:
  - clear for as long as the car needs to be across it, plus 1.5 s;
  - that time is worked out from where the car is, accelerating at 1 m/s² to its cruise
    speed;
  - oncoming traffic is assumed to drive at 30 km/h.

  Until then the car holds 0.3 m short of the lane. Once its body is in the lane it clears
  the lane, and the gate never stops it there.
  - **The LiDAR alone:** any obstacle cell hit in the last 0.3 s counts as traffic, out to
    40 m. That is 4.7 s at 30 km/h, while a 2 m/s turn from a standstill needs about 8 s,
    so the driver alone certifies only part of the gap (the log says how much).
  - **With the stack:** it decides from tracks.
    - A track that can reach the crossing in time blocks, at its own speed plus 1 m/s.
    - A track whose motion is unknown blocks within reach.
    - Parked, departing, far or slow tracks are excused, and the LiDAR check does not
      count their cells.
    - Radar tracks reach beyond 40 m.
  - **In the opposing lane:** a vehicle predicted where the car already stands no longer
    sets a zero cap. The driver's obstacle stop still stops for anything in the path
    ahead.
- **Standing at the mouth (S7).**
  - **Grid:** standing still, the grid keeps the static evidence of cells the current
    sweep does not hit. The rings re-hit the same cells and never the ones between them.
  - **Freshness credit:** the observed road keeps its freshness for the length of the
    standstill until the car has moved 3 m on. Kerbs do not get this credit: a kerb read
    0.9 m out from 22 m was otherwise kept fresh and never refined.
  - **Mouth:** a car standing at the mouth moves its stop on to a kerb opening that it
    reads steadily further on for 1 s. It never stands past the tangent point of the
    nominal arc to the exit line, and that point only comes nearer.
  - **Radii:** a kerb cell under the exit straight, 2 m past the arc, no longer rules a
    radius out. The verified distance stops the car short of it, and the line is refined
    as the car closes. Something standing on the exit still rules it out.
  - **Held regions:** standing still, a hold's ground is the current sweep only. Over
    seconds, the noise of identical sweeps put a point in the strip behind a missed
    pedestrian's feet.
- **Divided exit roads (S6).** A raised strip with road again within 2.5 m beyond it is a
  median: the exit carriageway is the outermost one. A refinement further than 1.5 m from
  the latched exit line is not taken.
- **Exact arcs (exit offset).** A tangent point up to 1.5 m behind the car is entered where
  the arc passes the car, a few centimetres off and a few degrees on. The arc is no longer
  shifted forward, so the exit straight lies on the exit line. It lay up to 1.5 m off.
- **Obstacle tube on bends.** The outer widening uses the mean curvature over the car's
  length behind each point. Where a bend begins, the body has not swung out yet. On a left
  turn, the full widening put the kerb at the arc start in the tube as a "low obstacle",
  and its hold could never clear.
- **Turn evaluator.** The body outline is tested every 0.25 m, not just its four corners. A
  side over the exit kerb passed before. Turn sites must turn within 10° of a right angle
  (was 30°). The car keeps the right-angle prior, and a 100° junction ended 1.85 m
  towards the opposing lane. **This changes the turn-site numbering**, so `--site N` for
  turns may name a different junction than before.
- **Stack.**
  - **H2:** the conflict cap is the lowest over every road user with a timed conflict.
  - **H5:** leads and conflicts are placed on the path the driver planned along this tick
    (a turn's, in a turn).
  - **M8:** a tick with no planned path holds the cap in force.
  - **M4:** a vehicle coming towards the car, reversing or crossing is kept for the
    conflict check, not handed to the obstacle stop.
  - **M5:** a lead must reach into the swept tube, and a pedestrian within reach is
    always a candidate.
  - **H3:** off the road, a mover heading for the path is kept unless it shows the
    sliding-wall signature or stands on persistent tall evidence. The kerb-plus-2 m reach
    alone used to drop a cyclist out of a side road. Yields cover cyclist speeds (6 m/s).
  - **M7:** a slow unclassified cluster up to 4 m across is a group on foot: it is
    yielded to and is never a "vehicle" a wall can make unreachable.

| Case (synthetic world) | Before | Now |
|---|---|---|
| Held 12 s / 30 s before the mouth | committed, verified 21.9 → 0 m, stuck | turns, reaches the goal |
| Walker on the exit lane until the car has stood 3 s (8 m fillet) | stuck at the mouth | waits, then turns |
| Left turn, oncoming car at 8 m/s released 30–60 m out | entered with a 13 m (1.6 s) gap, stopped in its lane, hit | waits and enters behind it; no collision in 6 cases |
| Divided exit road, route point 3–4 m out (two seeds) | 4.6 m refinement onto the opposing carriageway | near carriageway, within 0.1 m |
| Arc with its tangent point 0.5–1.5 m behind the car | exit straight 0.25–1.5 m off the line | on the line |
| Right/left turns, bare/furnished, seeds 4–7, 14 m fillet | finish | finish |
| Missed pedestrian / side intrusions after a hold (Stage 1 cases) | held | held |
| The stack review's ten failing tests (3e9083a) | fail | pass |

Known limits:
- **Exit kerb bias.** On the synthetic verge the exit kerb reads up to 1 m onto the
  pavement from 20 m or more, and the car often commits on that line. A walker on the
  exit at a 10 m fillet now leaves the car stuck mid-turn: the exact arc no longer
  shifts the exit straight back towards the lane by accident. At a 6 m fillet the car
  still waits at the mouth for good, as before. On Town10HD the raised returns sat at the
  kerb line; this needs the live turn runs.
- **Oncoming check (LiDAR only):** it cannot tell a parked car in the opposing lane from
  a stopped one, so it waits for both.
- **Lane geometry:** the opposing lane is assumed to be one 3.5 m lane beside the car's
  own.
- **Not covered:** traffic on the cross road at a four-way junction, and traffic signals.

Regression tests: `tests/unit/test_mapless_stage3_junctions.py`,
`tests/unit/test_mapless_stage3_stack.py`, and the evaluator test in
`tests/unit/test_mapless_g1c_harness.py`.

**Live on `ff1910b` (28 Oct).** Two of the three runs passed:

| Run | Result | Footprint margin | Exit-lane p95 |
|---|---|---|---|
| Right turn, driver only (PC 2, Town10HD site 0) | MEASURED_PASS, every gate | −0.50 m, on the full body outline | 0.12 m |
| Left turn, driver only (PC 1, Town03) | MEASURED_PASS, every gate | −0.43 m (−0.18 m on 25 Oct) | 0.04 m |
| Right turn, stack (PC 2) | INCOMPLETE | — | — |

- **Left turn.** The oncoming gate found the opposing lane clear (0 occupied cells) and
  the car entered it at 14.3 s. The exit kerb was read close by.
- **Stack right turn.** It went stop-go every 2 s past a row of parked cars. Their
  centroids drift 0.3–0.46 m/s the car's way as the visible part changes, which read as
  "uncertain" motion, and that envelope reached the lane from 2.4 m beside it. It then
  stood 45 s at the junction mouth: a one-tick detection of a parked car's corner was
  held 0.8 m past a corridor that had shrunk to 6.8 m (the kerb hidden by the parked
  cars), so "clear to 7.0 of 9.6" could never be met.
- **Fixes.**
  - A vehicle beside the path slower than 1 m/s, not coming towards it, whose near side
    (centre less half its width; a partial view's yaw is not trusted) lies 0.1 m or more
    outside the swept body, is not a conflict.
  - Held regions are judged on ground straight past the line's end. Past the end, the
    path need only be clear to the line's end, and a hold's span is placed past the end,
    no longer clamped to it.
  - Groups on foot must be at least 0.6 m across: a 3.4 × 0.37 m railing stretch was
    yielded to.
  - Off-road movers need a sustained recent speed and must reach the lane within the
    horizon plus 1 s. A 0.3 m "runner" 23 m out set a zero cap.

  Replaying the run's recorded frames through the fixed stack (open loop) cut the
  zero-cap ticks from 95 to 3. The 3 left are a new track at start-up.

**Live on `7eaca3d` (28 Oct): stack right turn MEASURED_PASS, every gate** (PC 2,
Town10HD site 0): footprint margin −0.43 m, exit-lane p95 0.17 m, approach p95 0.06 m;
driver step 203 ms p50, 361 ms p95. It still had 104 ticks of zero cap from false
triggers, and open-loop replay of its frames found four causes:

- parked cars whose near side read 1.03–1.12 m from the path, inside the 0.1 m
  clearance (their centroid sits on the face towards the car);
- one-cell clusters at 6–10 m/s, 13–23 m off the path at the junction;
- an 8.8 × 0.32 m stretch of wall sliding past, as a vehicle-length track;
- blobs on the junction corners, kept because they might be people.

Fixes: the parked-car clearance is −0.1 m (the near side may lie 0.1 m inside the swept
body's side line, not outside it). Off-road non-vehicles beyond reach must be at least
0.5 m across and no faster than `vru_max_speed_mps`. A thin vehicle-length track beyond
reach, standing on tall evidence, is a wall. Replay: 114 zero-cap ticks down to 40, all
from the junction-corner blobs (tracks 480 and 455). Those are kept, because they might
be people.

### Third external review, of `e93154f` (28 Oct): all taken

Two of its suspected defects were real (R1 and R5 below). Each change has regression
tests; those for R1, R2, R5 and R7 fail on the code before it.

- **F3 / R1, the lead hand-over.** The source gap the review named was closed in `ff1910b`
  (M4: a vehicle moving across or towards the car is a "crossing" role, kept in the
  conflict check). The R1 sweep then found a second gap: a car cutting in that became the
  followed lead lost its restriction. Following started from the car's own speed. Its
  jerk limit also applied whenever the gap term, not the worst-case cap, was the lowest,
  even with the target far above the cap. A 1.4 m/s conflict cap became a 6 m/s target that
  fell 0.005 m/s per tick. Fixes:
  - following brakes at up to `ego_brake_mps2` whenever its target is above the
    worst-case cap;
  - a lead taken on under a restriction starts from it (`FollowingController.start_from`);
  - the log names what covers the lead (`lead.checked_by`: conflict check, following or
    obstacle stop), its role and its position.

  Test: `test_r1_...` sweeps a confirmed car diagonally into the lane at three speeds along
  and three across, closed loop on the cap.
- **R5, a pedestrian on the pavement.** A walker the segmenter classed as a pedestrian,
  standing 2.5–3.5 m off the lane centre, stopped the car for good. Its walking-pace
  envelope reached the path. A standing vulnerable road user whose near edge is more than
  0.5 m beyond the tube is now passed at `roadside_mps` (3 m/s), slowed for at 2 m/s² as
  the car draws level. Once it moves, it is a crosser and the yield applies. One standing
  at the lane edge is still stopped for. Replay of the s3b turn: zero-cap ticks 40 → 4.
  The junction-corner blobs are now passed slowly instead of stopped for.
- **F1, actor identity.** The stack logs its tracks around the car (`tracks_near`: id,
  position, velocity, extent) and the lead's position. The evaluator matches tracks to
  the actor's true box within 1 m, on its own side. New REPORT check `actor_tracking`:
  - matched, missed and fragmented ticks, and the longest miss;
  - the actor's track ids, and ID switches;
  - false-lead ticks, and ticks the actor was in the lane but not the lead.

  `lead_tracking` now scores speed and gap errors, and brake detection, only on ticks
  whose lead was the actor (`unmatched_lead_ticks` counts the rest). Logs from before
  this change are scored as before, and the report says so.
- **F2, holds.** Holds have ids. A detection joins a hold only within 2 m along it and
  0.5 m beside its lateral span; another obstacle beside it gets its own hold. Merged, a
  second obstacle 1 m on moved the stop 1 m towards a first that was only missed (R2).
  Each tick's `hold_reports` gives every hold's state, place, detections and its release
  prerequisites: region bins with ground, clear-to against needed, continuous-clear time,
  and what it is `waiting_for`. R3: rings either side of a gone walker's region keep it
  UNKNOWN, waiting for "ground across its region", and it releases once a ring sweeps the
  region. No timeout was added.
- **F4, driving quality.** New REPORT check `behaviour`, beside the unchanged gates.
  Every tick's restriction (a cap below cruise, or a stop within braking range) is
  attributed to the actor or not, and its time and the time it cost are summed. Per
  scenario it adds:
  - braking lead: restriction not from the actor, and unjustified slowing (0.5 m/s below
    min(cruise, the lead) with an ample gap), before the lead brakes;
  - pedestrian: near-stops, restriction and progress after the walker cleared;
  - cut-in: the first restriction from the cut-in car and the first brake, both from the
    cut-in, and the least time to collision.

  **The two observed events, reconstructed** from the `e93154f` logs:
  - Braking lead: the car slowed to 0.37 m/s at 8.6 s, before the lead braked at 9.95 s.
    The cause was a hold made at the start on the lead itself while it stood, which
    stayed UNKNOWN after it drove off (2.45 s of restriction not from the actor).
  - Pedestrian: a near-stop after the walker cleared, from a hold on the walker's place,
    UNKNOWN for 2.75 s.

  Both were the hold-clearance defect that `e401188` fixed (ground gathered since last
  seen). On the `e401188` runs, the restriction not from the actor was 0 s before the
  brake, and 0 s with no near-stop after the walker cleared. The Town05 pair still shows 2.3 s of predicted
  conflicts from scenery tracks (the known 11 s false stop).
- **R7, refused plans.** When this tick's plan is refused (it starts over 2 m from the
  car, or has expired), the controller keeps the last trajectory it accepted. That one is
  now driven no faster than the refused plan allows within 1 m ahead of the car
  (`refused_plan_limit_mps`). A refused stop now brakes at once, not after the 0.5 s of
  validity left on the kept trajectory.
- **Braking envelope.** `scripts/braking_table.py <runs> [--calibration
  longitudinal.csv]` lists braking episodes from drive logs or BL-E-03's brake steps: lag,
  mean, peak and effective deceleration (v0² / 2d, lag included), and stopping distance.
  It then sets each rule's assumed stop, v × response + v² / 2a, against the measured stops
  from full brake. A first version compared a 3-tick peak deceleration, which the last
  ticks before a standstill distort. That version read the live data as not showing
  following's 4 m/s². Stopping distances do show it (live results below).
- **Compute.** `driver_step_time` now reports p50, p95, p99, max, the share over the
  50 ms target, and each stage's p50 / p95 / max: the stack's (drive, paint, detect,
  track, plan, road_model) and the driver's own (`driver.ego`, `classify`, `grid`,
  `corridor`, `junction`, `obstacles`, `plan`, `control`).
- **R4.** A lead holding 3 or 5 m/s, its velocity jittering across at 0.15 m/s as on the
  live logs (p95 0.05–0.12 m/s), is followed with no conflict cap.
- **R8**, sensor dropout, was already covered:
  - `test_driver_brakes_when_the_lidar_goes_stale` and
    `test_stale_ego_sensors_fault_and_recover` (G1b);
  - `test_an_odometry_outage_does_not_poison_the_grid_or_the_hold` (G1b review);
  - `test_a_missed_lidar_sweep_keeps_the_cap_in_force` (G2);
  - `test_closed_loop_an_imu_outage_while_held_keeps_the_hold` and
    `test_closed_loop_a_held_pedestrian_whose_returns_go_missing_keeps_the_car_stopped`
    (Stage 1).
- **R6, R9 and the ML A/B** are live runs: the radar A/B/C with actor matching, `--seed`
  repeats, and constant against v2 segmenter on identical scenarios.
- **The run report's three dimensions**, as the review proposes:
  - safety: the existing gates, unchanged;
  - mission: `behaviour`;
  - perception and runtime: `actor_tracking`, `lead_tracking` and `driver_step_time`.
- **Not done yet:** a speed limit from the range at which a small object is seen (the
  review's observability point; Stage 5), and optimising the stages the profile shows
  over budget.

**Live on `6370b98` (PC 1, Town05) and `7eaca3d` (PC 2, Town10HD site 1), 28 Oct.**
Every run was MEASURED_PASS. PC 2 had not pulled, so its runs are on `7eaca3d` without
the new logging. They were scored afterwards with the new evaluator; `actor_tracking`
needs `tracks_near`, so it is not available for them.

| Town05 braking lead, 4 m/s (PC 1) | A: LiDAR | B: + front radar | C: + rear radars |
|---|---|---|---|
| Lead speed RMSE, actor-matched | 0.29 m/s | 0.16 m/s | 0.16 m/s |
| Brake detection delay | 0.50 s | 0.20 s | 0.20 s |
| Actor tracked (of in-view ticks) | 94.4 % | 94.8 % | 94.7 % |
| Longest miss | 1.70 s | 0.75 s | 0.75 s |
| Ticks with two tracks on the actor | 146 | 309 | 288 |
| ID switches, false-lead ticks | 1, 0 | 1, 0 | 1, 0 |
| Restriction not from the actor | 0.80 s | 1.55 s | 1.55 s |
| Driver step p50 / p95 / max | 216 / 539 / 675 ms | 252 / 618 / 845 ms | 155 / 276 / 420 ms |

- **Radar.** The radar benefit holds with tracks matched to the actor: speed error
  halved, braking seen 0.3 s sooner, misses shorter. The rear radars add nothing for a
  lead ahead. With radar, the actor carries a second track on about half the ticks.
  Twins still form, though no false lead came of it.
- **Town05 scenery tracks.** The restriction not from the actor is all "predicted
  conflict". It comes from LiDAR-only CONFIRMED tracks 14–22 m right of the path:
  3.5–6 m long, mostly 0.3–0.85 m wide, "approaching" at 4–7 m/s. Some are young: no
  recent speed yet. This is the known Town05 scenery false stop, now attributed and
  measured. B and C slowed to 0.01 m/s at 6.7 s, 38 m short of the lead. Not fixed yet. These runs have no recorded
  frames to replay a fix on.
- **Braking (BL-E-03, Town05).** Full brake stopped from 2.3 m/s in 0.71 m and from
  8.1 m/s in 4.52 m (3.8 and 7.2 m/s² effective, lag 0.1–0.25 s). The 5 m/s trials read
  0.85 m at 0.7 and 1.0 brake (14.7 m/s²) and are implausible. Every rule's assumed
  stop is longer than the measured one: planner +1.96 m, conflict rule +0.62 m, following
  (4 m/s² after 0.5 s) +1.11 m. The live closed-loop stops agree (5 stops from 2–3.8 m/s:
  +1.68, +0.68 and +1.18 m). The module's own conservative figure is 1.56 m/s² (0.7 × the
  0.4-brake minimum).
- **Segmenter A/B (PC 2, `7eaca3d`).** Constant and v2 gave identical commands on every
  tick of the braking lead, the crossing pedestrian and the cut-in. The v2 model changed
  no driving decision in these scenarios, and cost about 160 ms at p95 (96–98 → 256–258 ms;
  p50 69–72 → 75–79 ms).
- **Seeds (PC 2, braking lead, v2).** Seeds 5308, 1 and 2: lead speed RMSE 0.16–0.18 m/s,
  brake seen in 0.15 s in all three. Seed 2 stopped for 0.35 s at 9.3 s, before the lead
  braked. The cause was a 0.74 × 0.3 m track 6.7 m right "approaching" at 2.1 m/s,
  under the `7eaca3d` filters.
- **Behaviour, PC 2.** The restriction not from the actor was 0 s on the braking lead
  (seeds 5308 and 1) and on the pedestrian (no near-stop after clearance). The cut-in car
  drives off faster than the ego, so nothing restricted the car, and this scenario does
  not exercise the R1 hand-over.
- **Compute.** Driver step on Town10HD: 69–72 ms p50 and 96–98 ms p95 with the constant
  segmenter, 75–79 ms and 254–267 ms with v2. On Town05: 155–252 ms p50, and 97 % of ticks over the 50 ms target. Town05
  stage p50s:
  - detect (clustering): 74–82 ms;
  - paint (v2): 53–57 ms;
  - the driver's obstacle observation: 18–20 ms;
  - the stack's plan: 16–19 ms;
  - road model: 12–13 ms.

### Fourth technical review, of `6f2aec3` (28 Oct)

**Live on `6f2aec3` (PC 2, Town10HD site 1), every run MEASURED_PASS.** With the lead
logged, actor matching works on Town10HD too: the actor is tracked on 96–98 % of in-view
ticks, with no false leads and no ID switches. The ~1 s miss is the start-up.
- **Slower cut-in** (`--actor-param lead_mps=1.5`): R1 is now exercised live. The car became
  the lead at 15.05 s under a 0.8 m/s cap, following started from it and nothing lifted;
  separation stayed at 6.9 m or more. The run also showed two new defects, fixed below:
  - the cut-in car (3.9 × 1.6 m at 1.4 m/s) read as a group on foot;
  - the following stop-go behind it.
- **Braking lead:** no restriction not from the actor. Brake detection was 0.65 s, because
  the stopped lead was not taken as the lead for 0.55 s. The driver's obstacle stop held
  it. Tracks are now logged with their status and support, so the next run shows why.
- **Pedestrian:** no restriction after clearance, no near-stop.

**PC 1, Town05 braking lead with frames recorded (`r3b_t05_B`).** It replays the false stop
deterministically (same as `r3_t05_B`). Open loop through the stack, the restricted ticks
fell from 56 to 20:
- **Building faces 14–22 m right** (tracks 195, 526 and 557, which gave the 6.7 s
  near-stop): CONFIRMED, LiDAR-only, 3–9 m long and 0.3–2.4 m across, "approaching" at
  4–7 m/s. In a 6 × 6 m window round each, the grid had no ROAD cell; around 40 BARRIER,
  20 OCCUPIED and 110 UNKNOWN. **Fix:** beyond the kerb's reach, a vehicle-sized track
  counts only if the grid shows road (3 cells or more) within 3 m, or has seen too little
  of the area (under 20 cells) to say (`road_near`). A car on a side road has its road
  round it, while its own cells read OCCUPIED. Paired tests: T01, the face, is dropped;
  T02, the same track with a side road round it, still gives cap 0; the same track in
  unseen ground also still gives cap 0.
- **Fence stretches** 5–6 m right, 4–8 m long, "oncoming" at 5 m/s along the road: beyond
  reach they reached the path only through their envelopes. **Fix:** beyond reach, a vehicle
  must be heading for the path (`approach_min_mps`). One turning towards it is kept the
  tick it does.
- **Left open:** two such fence tracks (119 and 129, 14 ticks) lie *within* reach, because
  Town05's kerb is 5.7 m from the lane centre. Their signature is incoherence rather than
  geometry:
  - the footprint jumps between 0.3 and 7.9 m long from tick to tick;
  - they are COASTING on most ticks (8–9 hits in 1.5 s);
  - a 5–9 m/s velocity is built on two or three detections.

  A coherence gate in the tracker is the fix the review asks for. It needs a paired
  real-vehicle case first, so it is not done.

**Following behind a slower car.** The target closed its error to the wanted speed in one
tick, with only the jerk limit between them. It overshot into a limit cycle: 1.9 → 0.4 → 1.9
m/s every 3 s behind a car at 1.4 m/s, made worse by CARLA's low-speed coasting drop
(0.9 m/s in a tick at zero throttle). **Fix:** the target closes the error over
`speed_tau_s` (1 s). Braking for the worst case is still immediate. Simulated with a 0.3 s
vehicle lag:
- behind 1.4 m/s: settles at 1.4 m/s and the desired 6.8 m gap (was 0.87–1.91 m/s);
- a lead braking at 3 or 6 m/s²: the minimum gap is 4.0 m either way.

**Groups on foot** are no wider than `group_max_width_m` (1.4 m): the cut-in car seen as
3.9 × 1.6 m was yielded to at 0.8 m/s. A 2.4 × 1.0 m group is still yielded to.

**Diagnostics and evaluation (the review's 5, 7.3, 7.4, 9.1):**
- A cap's cause track carries `admitted`, the rule that let it through
  (`plausible_crossers(..., why=)`): for example "vehicle beyond reach, road near".
- The tracker logs every radar-only track it did not fold into a LiDAR one within 5 m,
  with the gate that kept it (`radar_stats.twins_kept`). The evaluator reports, on ticks
  with two tracks on the actor:
  - those with a radar-only second track;
  - the reasons the tracker gave;
  - ticks where the second track, not the lead, set the cap.
- `tracks_near` carries each track's status and support, keeps up to 40 tracks, and logs how
  many it left out (`tracks_near_omitted`). A tick with no match while tracks were left out
  is "not judged", not a miss.
- The run report records its `braking_assumptions` (planner, conflict rule, following
  deceleration and reaction). `scripts/braking_table.py` checks against them, refuses runs
  that disagree, and lists the excluded implausible stops with the reason.
- New REPORT check `exercised`: did the run reach the transition its scenario is for?
  - braking lead: the lead slows within 50 m;
  - pedestrian: the walker enters the band within 30 m;
  - cut-in: the car became the lead under a conflict or yield cap, and the ego then closed
    on it.

  The `r3b_cutin_slow` run is exercised; the earlier cut-in, whose car drove off, is not.

**Compute: LiDAR clustering** (the review's 6.1 and T11). Linking pairs of OBSTACLE points was
the largest Town05 stage. It sorted the points by x and compared each block with every
point in the same x-band, across the whole width of the street. It now:
- hashes the points into squares no smaller than the largest link and compares each square
  with its neighbours;
- does this by range band, so near the car, where the points are densest and the links
  shortest, the squares are small;
- keeps each pair once, in the band of its nearer point; a farther point joins a band's
  pass only if its own link reaches into it.

On the 134 recorded Town05 sweeps (about 4900 OBSTACLE points each) the pair sets are
identical, at 26 ms p50 and 35 ms p95, against 80 and 112 ms. The random-scene test now
checks the pairs themselves against brute force, as well as the partition. Numpy only:
scipy is not on the lab PCs.

**Not yet:**
- the tracker coherence gate (above);
- arrival-time intervals in the conflict check (7.1);
- radar-only protection before LiDAR confirmation (7.2, T08);
- `weights_only` checkpoint loading (7.5);
- the segmenter in shadow mode;
- observability-limited speed.

**Live on `c882389` (28 Oct), every run MEASURED_PASS.**

- **Town05 braking lead, radar (PC 1).** The fixes hold live:
  - restriction not from the actor 1.55 s → 0.05 s; the 6.7 s near-stop is gone (lowest
    speed before the lead braked 1.97 m/s, was 0.0);
  - actor tracked on every in-view tick; lead speed RMSE 0.13 m/s; braking seen in 0.15 s;
    no false leads;
  - clustering (`detect`) 44 ms p50 against 74–82 ms.

  The step is still 178 ms p50 and 469 ms p95: the v2 segmenter's paint is 295 ms at p95,
  detect 95 ms and track 42 ms. The twins: 332 ticks with two tracks on the lead, 232 of
  them a radar-only one; 177 of the 216 kept twins lay outside the LiDAR footprint; the
  second track set the cap on no tick.
- **Slower cut-in (PC 2).** The stop-go is gone: following settles at the lead's 1.35 m/s.
  The cut-in car is no longer taken for a group. With the group yield gone, the car became
  the lead under a 3.2 m/s conflict cap, above the 2 m/s cruise; `exercised` reports, rightly,
  that the hand-over met no restriction (R1 was exercised on `6f2aec3`).

  It still stood 1.4 s at the hand-over. Just above the worst-case cap, the target was braked
  at 4 m/s² past the cap to the gap term's zero, and the jerk limit then held it at zero
  while the car ahead moved off. **Fix:** above the cap the target comes down only to the
  cap; there is no wind-up at a standstill. Simulated: lowest speed 0.8 m/s (was 1.95 s
  below 0.3), gap at least 4.2 m.
- **Braking lead (PC 2).** Brake detection was 1.15 s: the lead flickered in and out of being
  the lead. The kerb it hides ends the corridor at about its rear, and its centre then lay
  just past the line's end. **Fix:** a track up to 8 m past the end counts, on the line
  continued straight (`beyond_end_m`).

  In that run the second track set the cap on 4 ticks. `stopped_before_lead` no longer
  counts the goal stop.
- **Radar twins** that lie behind a LiDAR face along the line of sight (up to 3.5 m, the
  face under 1.5 m deep) are now merged. A car queued 6.5 m behind keeps its own track.
  Recorded frames hold no radar, so this waits on a live run.

**Live on `fcac32a` (PC 2, Town10HD site 1, constant segmenter), both MEASURED_PASS.**
- **Slower cut-in:** no stop at the hand-over. The lowest speed was 0.92 m/s, then the car
  settled behind the cut-in car at 1.33 m/s, the gap never under 5.4 m, nothing stopped
  before the lead.
- **Braking lead:**
  - the lead stays the lead (539 ticks, from 1.1 s), so braking is seen in 0.15 s again;
  - ticks with a radar-only second track on it: 125 → 30, and "outside its footprint"
    131 → 1: the face-twin merge works.

  The start is slower: the lead now counts while it stands at the start, so the ego pulls
  away under following's 1 m/s² and jerk limits (0.26 m/s at 2 s, against 2.15 before).
  That is 0.95 s lost, reported as 1.7 s "unjustified slow".
- **Compute without the segmenter:** driver step 58–60 ms p50, 85–86 ms p95, 92–99 ms max
  (was 69–72 / 96–98 ms; 75–79 / 254–267 ms with v2); 76–80 % of ticks over 50 ms. Clustering
  (`detect`) is 10–11 ms p50, against 19–20. What is left, p50:
  - drive 19–21 ms, of which obstacle observation 11–15 ms;
  - road model 14 ms;
  - tracking 13 ms;
  - plan 5 ms.

### Fifth technical review, of `fcac32a` (28 Oct)

**F5-01, slow start behind a distant lead (live, P1).** The cause, from the `r5_brake` log: at
1.45 s the lead, 16.7 m ahead and moving off at 2.1 m/s, became the followed lead. Following's
target began at the car's measured 0.42 m/s while the car was lurching to 2.0 m/s, and the
cap braked it back to 0.1 m/s. Following was not needed at all: the gap term asked 8 m/s and
the worst case 8.6 m/s, against a 2 m/s cruise. **Fix:** watching a lead and restricting
the car are now two decisions (`FollowingController.target_speed`):
- while the gap term and the worst-case cap both allow cruise, and the car is within the
  cap, there is no following cap: the driver accelerates as with no lead, and the lead's
  numbers are logged (`lead.following`: mode, gap speed, worst-case cap);
- following takes over the tick either falls below cruise, from the car's speed, or from the
  cap in force if that cap was set for this same road user (R1: the car cutting in);
- it lets go once its own target is back at cruise with both terms above it.

A cap held for someone else is no longer inherited. On the Town05 frames, two ticks' cap
for a fence stretch held following under 2 m/s for 4.5 s behind a lead 30 m ahead.

Paired regression (`test_f5_01_*`): a lead standing or moving off 20, 40 or 60 m ahead, at
the live 2 m/s cruise and at 6 m/s. The start is the same as with no lead until the gap or
the worst case needs a cap; then following restricts and never lifts the cap. The same
leads braking at 6 m/s² at 4 or 7 s: following takes over the tick it is needed, and the
car stops at least 2.5 m behind (the obstacle stop's gap). Against the old controller on the
same pairs:
- speed at 2 s: 2.0 m/s instead of 1.56–1.78;
- deceleration after the lead brakes begins at the same time or earlier, never later;
- the minimum gap is unchanged.

**F5-03, the lead past the line's end.** The line was continued straight for 8 m. Now:
- the continuation keeps the curvature of the line's last 4 m (up to 0.15 1/m), so a car on
  the bend is the lead and one straight on is not;
- where the route turns off at a junction while the path is still the kerb corridor
  (`MaplessDriver.turn_entry_m`), the continuation stops at the turn. A car straight on past
  the junction is not the lead; one that ended the line before it still is;
- a lead on the continuation is logged with `continuation` (its distance past the end, its
  rear's, `uncertain`). One whose body begins more than 1.5 m past the end did not end the
  line, so its place on the car's way is uncertain: it is followed for the worst case only,
  not the gap.

Tests: bends both ways, a junction ahead, a straight road, the stack's worst-case-only cap.

**F5-02 and 4.3, motion coherence.** The tracker scores each vehicle-sized track's last
second (`ObjectTrack.motion_coherent`, `coherence`):
- the share of updates that measured it;
- the position residual about a constant-acceleration fit through the measured positions;
- the spread of its footprint's long side.

On the Town05 recorded frames (c882389), the followed car's residual stayed at 0.073 m or
under, with every update measured. The building faces and fence stretches ran 0.1–0.9 m,
measured 5–60 % of the time. Incoherent means a residual over 0.25 m, or one over 0.1 m (or
too few points) with under 60 % measured.
- **4.3:** a vehicle-sized track with incoherent motion, outside the lane band, is predicted
  standing where it is. Its motion is no basis for a 3 s prediction, and its occupancy is the
  obstacle stop's, as any static object's. In the lane band its motion stands, coherent or
  not. Walker-sized tracks are not judged.
- **F5-02:** off the road (beyond the kerb's reach, no road in a well-seen 6 × 6 m window),
  a vehicle-sized track now counts if it moves as a car leaving a driveway, a car park or a
  verge does: coherent, 4 m/s or slower, tracked 30 updates or more as one object of one
  size (footprint spread at most 0.5). Missing road alone is no longer enough to drop it.
  The size-and-history test rejects a Town05 fragment that merged into an 8.5 × 3.5 m
  cluster, jumped and then moved coherently; the relaxed rule first let it in.
- Paired tests: a driveway exit (coherent, 3.5 m/s) counts; a building's face (incoherent,
  too fast, or a merged fragment) does not. A car park at road level, an access road, a
  side road at an odd angle and a partly hidden entrance count as before. An in-reach fence
  (incoherent) gives no cap, while the same track moving coherently does. The tracker test
  covers a jumping, intermittent track against a car.
- **Town05 frames, open loop:** ticks restricted by a predicted conflict fell from 22 to 8
  (head `fcac32a` against this commit). The two left are a 0.8 m object seen 8 times
  (walking-pace envelope, cap 2.8–3.5 m/s) and a 12 m fence at 35 m, too young to judge
  (2 ticks). Ticks capped below cruise overall: 144 → 129.

**F5-04, twin fusion.** The radar and LiDAR twins' velocities were fused as independent and
the cross terms dropped. **Fix:**
- they are fused by covariance intersection (`tracking.covariance_intersection`), consistent
  whatever their correlation;
- the LiDAR track's position–velocity cross terms are kept, halved only as often as needed
  to stay positive definite.

Monte Carlo, two 2-D estimates (NEES 2 is consistent):

| Correlation | Independent fusion | Covariance intersection |
|---|---|---|
| 0.0 | 1.97 | 1.04 |
| 0.5 | 2.92 | 1.49 |
| 0.8 | 3.45 | 1.74 |

Live checking: the lead now logs its speed's own sigma (`speed_sigma_mps`) and the mean LiDAR
NIS. `lead_tracking` reports the speed NEES against the truth by sensor set (mean, share
under 3.84) and the NIS mean. A merged (`lidar+radar`) lead far above 1 would show
overconfidence.

**F5-06, radar-only crossing threats.** A radar-only track is predicted onto the path when:
- it moves on corroborated motion: "moving", 1 m/s or faster, 2 m travelled;
- its velocity explains its radial speed (no Doppler rejections);
- it stands where the LiDAR has seen nothing (`_lidar_saw`). Where the LiDAR sees the place,
  an object there would be its own LiDAR track, so the radar's is a twin or a ghost.

It is predicted as a 4.5 × 2 m vehicle along its motion (`radar_threats` in the log).
Clutter stays out: standing, barely travelled, Doppler-rejected and LiDAR-seen tracks are
not predicted. Recorded frames carry no radar, so this needs a live radar-only crossing run.

**F5-07, the left-turn gate and stopped cars.** A static track in the opposing lane was
excused, so a car stopped at the junction could move off as the car crossed. Now every track
in the lane is judged by the time it needs to reach the crossing, accelerating at
`oncoming_accel_mps2` (2 m/s², stated) from its speed now (from rest when static) up to
30 km/h. A static one is excused regardless only on parked evidence: never seen to move
(under 1 m over 100 updates) and standing in the outer part of the lane (its centre past 60 %
of the lane's width). A stopped car mid-lane, one that arrived and stopped, and one only just
seen all block while they could still arrive in time. Paired tests: parked at the far kerb
against stopped mid-lane; stopped near against too far to arrive; creeping off at 1 m/s
(excused before, at its own speed plus 1 m/s) against driving away. The gap stays
conservative: a stopped car within about 37–50 m of the crossing (`required_s` 6.5–8 s)
holds the turn until it leaves or is seen parked.

**4.1, arrival-time intervals.** The conflict check timed the path at one speed. It now also
times it as the car is driving (`arrival_profiles`): accelerating from its speed to the
nominal one at 1 m/s², or slowing at 2 m/s² to the cap in force, standing where it stops for
the rest of the horizon. A conflict on any profile counts. The nominal profile is still the
earliest arrival, so nothing it found is lost. Test: a car crossing 6 m ahead at 1.5 m/s
from 10 m out has gone before a car holding 4 m/s arrives. It meets one moving off from rest,
one speeding up from 1 m/s, and one slowing to 1 m/s or to a stop, and those now cap the car.

**4.2, observability.** A run of the `stop` scenario now reports `observability`: the first
held detection of the obstacle where the truth has it (true bumper gap, speed then), and the
speed that distance supports, `d >= v*0.25 + v^2/(2*1) + 2.5` (the planner's 1 m/s², the
measured brake lag, the stop gap). `--obstacle-blueprint` puts a smaller object in place of
the parked car, so the envelope per size comes from runs (cone, box, car at 2, 3 and 4 m/s).
Nothing sets the car's speed from it yet: that waits on the measured envelope.

**F5-05, compute.** Profiled on the Town05 recorded frames (this container is about 1.5–2.2
times slower than the lab PC, stage by stage):

| Change | Before | After | Same result |
|---|---|---|---|
| Point-to-polyline projection (the driver's ground points against the corridor), component-wise and in place | 17 ms | 3 ms | to the bit |
| Clustering: squares joined unmeasured where every pair in them is linked | 52 ms | 20 ms | identical clusters and structures on 1452 sweeps from four runs (two Town05, two Town10HD) |
| Road-model labels on array slices | 14.3 ms a build | 5.4 ms | identical labels |
| Static tracks not predicted (never candidates) | 53 predictions a tick | movers only | same candidates |
| Coherence fit only for vehicle-sized tracks, by normal equations | 3.4 s | 1.3 s | |

Two scheduling changes:
- segmentation and the road model no longer share a tick (`road_model_phase`);
- the run report gives `control_path_ms` (drive, detect, track, plan: what must meet 50 ms)
  apart from `optional_ms` (paint, road model).

The whole stack on recorded frames, in this container (constant segmenter, no profiler):

| Frames | `fcac32a` p50 / p95 | now p50 / p95 | control path now, p50 / p95 |
|---|---|---|---|
| Town10HD right turn (junction, 300 ticks) | 135 / 288 ms | 52 / 68 ms | 48 / 65 ms |
| Town05 braking lead (walls, 300 ticks) | 139 / 185 ms | 72 / 94 ms | 68 / 89 ms |

The lab PC ran these stages 1.5–2.2 times faster, so live Town10HD should be well under
50 ms, and Town05 near it.

**Second pass (29 Oct), every change checked for the same output:**
- The driver's ground points: a point is left out before the exact projection when a bound
  shows it beyond the tube's widest reach (half the tube + 1 m), or not in the stretch past
  the line's end. The bound is the distance to the line's chords less how far the line
  strays from them. Identical observations on 2156 ticks of three runs; 12.0 → 6.9 ms.
- Clustering: all of a band's farther offsets are looked up in one search instead of one
  per offset, and a piece's size is taken from the shape already measured. Identical
  clusters and structures; 18.4 → 10.1 ms.
- The conflict geometry tests every pose of a prediction against its steps at once
  (`first_footprint_conflict`, `first_occupied_arc`). Identical results on 3000 random
  scenes and on every tick of the three replays (caps, reasons, causes, leads).

| Frames, here | `fcac32a` p50 / p95 | now p50 / p95 | control path now, p50 / p95 |
|---|---|---|---|
| Town10HD right turn | 135 / 288 ms | 43 / 58 ms | 40 / 53 ms |
| Town05 braking lead | 139 / 185 ms | 57 / 81 ms | 54 / 77 ms |

The full
configuration's cost is the v2 segmenter's paint (295 ms p95 live on Town05): see the ML
section below.

**ML: the segmenter in shadow (the review's stages 1 and 2).**
- **Resolution.** The segmenter's probabilities now stay at the model's own resolution
  (`StackConfig.segment_full_resolution`, off). Painting takes the cell covering each pixel,
  instead of resizing, normalising and checking a full-camera-size array per image every
  segmented tick. For an image whose classes change every 4 pixels, the painted
  probabilities match the full-resolution ones (test).
- **Shadow measure.** Each road-model tick logs `semantic_shadow`: of the road-model cells
  in the lane band along the path ahead (25 m) that carry semantic evidence, the share of
  road and of pavement. The run report summarises it (`semantic_shadow`: median and p10
  road share, ticks mostly pavement). No rule reads it. A path the segmenter calls
  pavement is where it disagrees with the kerb corridor. That is the evidence a later
  "controlled assistance" stage, at junction exits and occluded edges, needs first.

**New scenario `crossing_vehicle` (M07).** A car waits `side_m` (6 m) right of the lane
centre at `brake_at_m`, facing across the lane: a driveway, or with more `side_m` a side
road. It crosses at `cross_mps` (2 m/s) once the ego is within `trigger_m`, held by
CARLA's constant-velocity mode, so its velocity and radar Doppler are real, and stops past
the lane's left edge. Scoring is the pedestrian's (yield, clear, resume, goal), with the path
clear only when the car's whole body has left the band. It is the live pair for F5-02 (a
driveway exit off the road) and, with a far trigger, for F5-06 (radar first).

**Testing.** A run's status is now PASS, FAIL, INCOMPLETE or NOT_EXERCISED. A run whose
gates all pass without the scenario's target transition (`exercised` false) is
NOT_EXERCISED, not MEASURED_PASS.

**Also:**
- `cut_in` takes `--actor-param cut_brake_after_s=S`: the cut-in car brakes to a stop S
  seconds after cutting in (the matrix's "slow cut-in, then the actor brakes"). Holding
  behind it, stopped, at a safe gap completes the run.
- Segmenter checkpoints load with `weights_only=True` (the third review's 7.5). One that
  holds other objects still loads as before, and the run report says so
  (`stack.segmenter_load`).

**Live scenarios for the rest of the matrix (29 Oct):**
- `turn_left --oncoming stopped|parked` (M08). A car in the opposing lane, 12 m past the
  junction straight across (`oncoming_m`):
  - "stopped": it stands mid-lane and pulls away towards the junction at 6 m/s
    (`oncoming_mps`), 2 s (`go_after_s`) after the ego first comes within 12 m of the entry.
    The gate must hold the turn for it;
  - "parked": it stands 0.9 m towards its own kerb (`parked_offset_m`) and never moves. The
    gate may excuse it once it has been seen standing long enough.

  `exercised` checks that the car pulled away while the ego stood within 40 m of it, or that
  the ego came within 25 m of the parked one. The turn gates and separation judge the run.
- `braking_lead --actor-param second_gap_m=G`: a second car G m ahead of the lead's front
  holds its speed and brakes with it. These are two cars close along the radar's line of
  sight; the twin merge must not fold them into one track.
- The run report's `driver_step_time.response_budget` adds one tick, the control path's p99
  and the 0.25 s brake lag, and compares the sum with the following rule's 0.5 s
  `response_s`. Synchronous CARLA hides the processing time; this shows whether the
  budget would hold in real time.

**Still open:**
- radar-only and twin behaviour need live radar runs (no radar in recorded frames);
- the parked-evidence rule and the stopped-car gate need the M08 runs;
- the supported-speed envelope needs the stop runs per obstacle size.

### Sixth technical review, of `36b49ef` (29 Oct)

All seven findings and the ML item were taken.

**Recorded runs.** Seven recorded runs replay to the same restriction on every tick (the
cap, its reason and its track) before and after this work: 5720 ticks on Town10HD, Town03
and Town05, three of them with a braking lead. On these runs, none of the new rules adds or
lifts a restriction.

The recordings carry LiDAR only, with no radar or camera frames. They therefore test the
LiDAR-side rules (F6-03, F6-04, F6-05). The radar rules (F6-02) and the semantic worker
(F6-01) rest on their tests, on the live logs, and on the next live batch.

**The first live batch on PC Temp, `36b49ef`, before this work.**
- Town10HD, 11 runs: 9 MEASURED_PASS.
  - The 4 m/s car stop is INCOMPLETE: it reached the car with 94 settled ticks of the 100
    that lane keeping needs.
  - The braking cut-in is NOT_EXERCISED, correctly. The conflict cap at the hand-over was
    3.46 m/s, above the 2 m/s cruise, so it restricted nothing.
- Town05, braking lead, 4 runs: all MEASURED_PASS. The brake was seen 0.15–0.2 s after it
  began, the lead's speed error (RMSE) was 0.11–0.14 m/s, and its speed NEES averaged
  0.45–0.71 with 99 % of ticks within the 95 % bound. No run had a false lead.
- Control path, p50: 33–37 ms on Town10HD and 46–52 ms on Town05. The optional stages added
  7–9 ms at p50 and 18–27 ms at p95, before the command (F6-01).

**F6-01, semantic work off the control path (P0).** Segmentation, painting and the road
model inform no command. Run inline, `MaplessStack.step` still ran them before it returned,
so the command waited for them.
- `StackConfig.semantic_mode` "thread" runs them on `mapless.semantic_worker.SemanticWorker`.
- Each tick the stack hands the worker an immutable `SemanticJob`:
  - the sweep's points (a copy), the camera images and the pose;
  - the frame id and time;
  - the stack's spatial epoch, bumped on every reset of the odometry frame;
  - the terrain grid's placement (`TerrainGrid.view`) or, on a road-model tick, a copy of
    the grid (`TerrainGrid.snapshot`). The worker never reads arrays the control path is
    writing.
- The queue holds one job. A waiting job is dropped for a newer one, except that a waiting
  road-model build is not dropped for a job that only segments.
- The stack takes the newest finished model only if it comes from the current epoch and is
  no older than 0.5 s (`semantic_max_age_s`). Stale results are counted and dropped. An
  exception in the worker is counted (`semantic_worker.errors`, `last_error`) and the car
  drives on.
- The step now times only the hand-off (`semantic_submit`); the worker's own stage times
  and counters are logged every tick as `semantic_worker`.
- "inline" stays the library default, so replays and tests remain deterministic. The live
  harness defaults to `--semantic thread`.
- On the synthetic parked-car drive, every command and every cap is identical in the two
  modes (test).
- The run report gives both times and says which governs actuation:
  - `driver_step_time.actuation_governed_by`: the whole step when the work is inline, the
    control path when it is on the worker;
  - `response_budget` is computed on the governing time.

**F6-02, radar threats earlier, ghosts kept out.**
- A radar-only track becomes a *provisional* threat after 0.5 m of travel
  (`radar_provisional_travel_m`), at 1 m/s or faster, with its velocity explaining its
  radial speed. It is predicted along its velocity. Its conflict cap is floored at 1.5 m/s
  (`radar_provisional_cap_mps`): it may slow the car, not stop it.
- Once it has full evidence (motion "moving", 2 m of travel), it is a full threat and may
  stop the car.
- **Exclusions, both stricter than before:**
  - **Free ground.** Before, a track was excluded when the LiDAR had *seen* its cell, and a
    hit on the car itself counted as seen. Now it is excluded only where the LiDAR saw *free
    ground*: road hit in the cell and the four beside it, and no obstacle hit (`_lidar_free`).
  - **Twins.** It is excluded within 3.5 m of a LiDAR-supported track (`radar_twin_m`).
- **Both travels are at least two bearing sigmas across the track's range**
  (`radar_travel_bearing_sigmas`). The Doppler confirms motion along the line of sight, not
  across it. In the live crossing run, the radar-only track on the parked car, 46 m out
  and 4–6 m right, "moved" 2 m across in 0.7 s on bearing noise alone. It was a full threat
  for three ticks before the LiDAR took it over. The floor at 46 m is now 3.2 m; at 10 m it
  is 0.7 m.
- The log's `radar_threats` lists `[id, "provisional" | "full"]`. For the crossing car, the
  report adds a `radar_timeline` (`score_radar_timeline`), measured from when the car began
  to move:
  - when the radar, the radar alone, and the LiDAR first had it;
  - when it first became a provisional or a full threat;
  - the first restriction, the first brake, and when it entered the band;
  - whether the LiDAR already had it standing (`lidar_before_moving_s`), and how long the
    radar alone had it (`radar_only_s`).
- On the live crossing run, the LiDAR had the parked car from 1.9 s, at 46 m. The first
  restriction came 3.2 s after the car moved off. At this site and geometry a radar-first
  case does not arise.

**F6-03, off-road admission by the quality of the track.** The 4 m/s limit is gone. A
vehicle-sized track off the road is admitted if:
- its motion is coherent;
- its sustained motion bears out its filter speed (`recent_speed_mps` at least 0.6 of it);
- it was tracked 30 times as one object of one size (footprint spread at most 0.5);
- its speed is within a physical bound (12 m/s);
- it can reach the lane band within the conflict horizon plus a second.

Paired tests:
- Cars leaving a driveway at 2, 3.5, 5 and 7 m/s count.
- These do not:
  - a 7 m/s track whose sustained motion is 2 m/s;
  - an incoherent track;
  - a 2 m/s car 20 m out, 8.9 s from the band.

On the recorded runs, the rule is checked call by call. It admits no more than the old one,
except 10 m-long stretches seen at 4–6 m/s. The checks after it drop those, and the
restrictions are unchanged.

**F6-04, parked evidence from the whole history.**
- The tracker keeps each track's furthest excursion from where it was first seen
  (`ObjectTrack.max_excursion_m`) and how long it has stood still (`still_s`, since its
  sustained speed was last 0.3 m/s or more).
- The left-turn gate's parked rule uses them, not net displacement:
  - excursion of at most 1 m;
  - standing still for at least 5 s.
- A car that moved off and came back to where it was first seen is not parked (test).

**F6-05, following tied to the lead.** `FollowingController.target_speed` takes the lead's
id. A different lead resets the controller's state, so a far lead that replaces a near
slow one does not inherit its target (two-lead test).

**F6-06, variant-specific exercised conditions.** `score_exercised` takes the scenario's
parameters and adds a variant's own condition (`EXERCISE_VARIANT`):
- a crossing with `radar_first` 1 needs the radar alone on the moving car for at least
  0.25 s before the LiDAR (`radar_first_min_s`);
- a cut-in with `cut_brake_after_s` needs the cut-in car to brake, 0.5 m/s below its
  speed then, after it became the lead.

**F6-07, the schedule checked at start-up.** The two optional stages share a tick exactly
when the road-model phase is a multiple of gcd(`segment_every`, `road_model_every`).
- `optional_schedule` replaces such a phase with the first that does not share a tick.
- If every phase does (segmenting every tick), it reports the overlap as unavoidable.
- The result is `MaplessStack.schedule` and the run report's `stack.optional_schedule`.

**ML (evaluator side).** Each road-model tick's `semantic_shadow` adds:
- `coverage`: the share of the lane band with semantic evidence;
- the road's right edge as the segmenter shows it (`right_edge_m`): the median, over the
  stations along the path, of the midpoint between the last road cell and the first
  pavement cell.

The run report summarises:
- coverage, median and p10;
- the mean pavement share (road/pavement disagreement);
- the edge error against the site's lane-centre-to-kerb distance, evaluator truth: median,
  MAE and p95 (`right_edge_error_m`).

No rule reads any of it.

**Also seen in the live batch (not changed):** below about 2 m/s, a brake command of
0.05–0.15 stops the car within a tick or two in CARLA. Every stop and the close cut-in came
in as two or three such pulses.

### Live on PC Temp (29 Oct, `36b49ef` and `dd9344c`): the left-turn deadlock

**The runs.**
- Reruns on `36b49ef`:
  - braking cut-in at 4 m/s: MEASURED_PASS, exercised (the conflict cap now restricted
    the car at the hand-over);
  - car stop at 4 m/s with the car at 50 m: MEASURED_PASS;
  - three Town03 left turns (plain, `--oncoming stopped`, `--oncoming parked`): all
    MEASURED_FAIL, held 25 s at the mouth.
- The r7 batch on `dd9344c`:
  - car stop: MEASURED_PASS;
  - braking cut-in at 4 m/s: MEASURED_PASS, exercised with its braking variant (the car
    braked 2.5 s after it led);
  - braking lead, inline and threaded, with the v2 segmenter: both MEASURED_PASS,
    identical behaviour;
  - two left turns: MEASURED_FAIL as before;
  - two crossing cars: MEASURED_FAIL on completion. The evaluator was at fault (below).
- The Town05 crossing car could not be placed 6 m or 8 m right of the lane at that site.

**Semantic work, inline against threaded (v2 segmenter, the same braking-lead run):**

| | inline | threaded |
|---|---|---|
| whole step p50 / p95 / p99 / max | 52 / 117 / 129 / 411 ms | 51 / 70 / 76 / 115 ms |
| control stages p50 / p95 | 34 / 45 ms | 50 / 68 ms |

- The worker kept up: 390 jobs, 1 dropped. It built 130 road models and every one was
  taken; none was stale and none failed.
- Its segmentation took 139 ms p50 on the wall clock, sharing the interpreter. Inline, the
  paint stage, segmentation included, took 22 ms p50.
- The control stages slowed by about 15 ms at p50 because the worker's Python competes for
  the interpreter and the cores. Moving the worker to a process is the next step if the
  tail must come down further.

**Why the left turns held, found by replaying the recorded frames.** There were three
causes, each enough on its own.
1. **The exit kerb's span.** From near the mouth only the exit kerb's first three stations
   are in view. Three stations a metre apart span 2 m, the fit's minimum, only nominally.
   Re-projected into a probe frame that moves by centimetres, they read 1.99–2.06 m, and
   the 1.99 dropped the exit measurement at 7 m from the entry.
   - Fix: `JunctionConfig.exit_span_tolerance_m` (0.25 m).
2. **The exit measurement forgotten at the mouth.** The car held at the mouth, where those
   three stations are all it can see, and its 2 s memory dropped what it had measured on
   the approach.
   - Fix: stations now age only while the car moves (2 s of motion). The fit still reads
     the last 2 s of stations first, as before. Only if those make no line does it fall
     back to the stations kept from before the standstill.
   - Read from farther away, those carry more range bias. Fitting them all at once let
     the approach's 1.1 m outvote the standstill's 1.3 m in the synthetic S7 pause, and
     the car was held on its latched path.
   - The live yield before it (below) and CARLA's low-speed brake pulse had cost the car
     the 1.5 s that let the memory run out.
3. **A passing car's wheels read as kerb.** Standing at the mouth, the car keeps static
   evidence it no longer hits, so the cells between the rings do not decay (S7). The
   oncoming car's wheels left raised hits in four cells of the opposing lane (raised 6–9,
   little tall), so they read as kerb. Those cells were kept for the whole standstill, and
   no turn radius ever cleared them ("no radius clears the terrain" for 25 s).
   - Fix: evidence of a class first added during the standstill, to a cell already
     observed, decays as usual once the sweeps stop hitting that cell. Nothing static
     appears and vanishes while the car stands.
   - A cell first observed during the standstill (the whole grid at start-up) keeps, as
     before; the G1c turns depend on it.

**Replays of the recorded turns, open loop:**
- Plain, parked and oncoming left turns: the exit stays measured at the mouth. On the
  oncoming run the turn is planned again 4 s after the car has passed, instead of never.
- The five turns that passed before (three left on Town03, two right on Town10HD, 4338
  ticks): the driver's mode, stop distance, exit source and turn note are identical on
  every tick.
- The stack's caps on three Town05 braking-lead recordings: identical.

**Also from these runs:**
- **A yield to clutter at the junction.** A 0.7 × 0.4 m LiDAR track 33 m ahead and 18 m left
  "approached" at 0.7 m/s and brought a 0.8 m/s yield for a second. Recorded frames carry
  no radar, so the replay's tracker does not reproduce it; it is logged for the next run.
- **Crossing-car completion (evaluator).** A crossing car must now be yielded to only if
  the car, at the speed it was making, would reach it before it cleared, with a second's
  margin. Arriving a second or more after it has cleared counts as giving way.
  - r7 at 4 m/s: arrived 1.85 s after the crossing car cleared. Now passes.
  - The 5 m/s crossing: gone 8 s before the car arrived. Now passes.
  - Holding 4 m/s into a 2 m/s crossing 30 m ahead still fails (test).
- **Lead tracking.** The actor's braking is now looked for only from the tick it first led
  the car. The braking cut-in's slowing from its lane change no longer reads as a brake
  2.5 s before it led (brake delay 5.05 → 0.5 s).

### Seventh technical review, of `dd9344c` (29 Oct)

All six findings were taken.

**F7-01, radar-only time from the ticks the radar had the car.** The radar-only time was
the span from the first radar-only tick to the first LiDAR tick. A radar that had the car
for one tick was credited the whole span.
- `score_radar_timeline` now counts the ticks a radar-only track was actually on the car:
  - `radar_only_observed_s`;
  - the longest unbroken run (`radar_only_longest_s`, which is `radar_only_s`);
  - the gaps, a loss before the LiDAR included (`radar_only_gaps`,
    `radar_only_longest_gap_s`);
  - the span (`radar_only_span_s`);
  - whether a radar threat was on the car before the LiDAR had it (`threat_before_lidar`).
- The radar-first variant needs both an unbroken 0.25 s and that threat.

**F7-02, the whole step governs.** The command goes out when `stack.step` returns, so the
whole step's measured wall time now governs actuation in both modes, and the response
budget is taken on it.
- The control stages stay as a decomposition.
- `uninstrumented_ms` is what the stage timers miss: the step's time less every stage's.

**F7-03, the provisional floor.** The provisional radar threat's 1.5 m/s floor held
whatever the evidence and however close the conflict.
- The floor now falls linearly to 0 as the track's travel grows from the provisional
  threshold to the full one. Both thresholds are at least the bearing floor.
- The floor is never above the speed from which the car stops short of the conflict at
  4 m/s² (`radar_provisional_hard_decel_mps2`).
- 11 m out, on 0.6 m of travel, it still holds the car at 1.5 m/s: the clutter it is
  there for. 8 m out, it lets the car stop.
- At the first restriction, the radar timeline now reports:
  - `restriction_room_m`, from the car's front to the crossing car;
  - the speed then;
  - `restriction_decel_needed_mps2`, the deceleration a stop in that room needs. The live
    crossings needed 0.1–0.7 m/s².
- **Radar-first in CARLA.** The roof LiDAR reaches 60 m and tracked the parked crossing car
  at 46 m. In CARLA the radar is blocked by whatever blocks the LiDAR, so with this rig a
  radar-first crossing cannot arise within the conflict horizon. The live radar-first runs
  therefore shorten the LiDAR (`--set roof.range_m=25`, recorded in the report's suite
  overrides): a degraded LiDAR, against which the radar has to earn its place.

**F7-04, wall-clock age of semantic results.** Each job carries its submission time on the
wall clock; each result its start and completion.
- Per taken result, the log's `semantic_take` gives:
  - its simulation age;
  - its wall-clock age, from submission to use;
  - its queue wait and run time.
- Results older than 0.5 s on the wall clock are kept (shadow information) but counted
  (`late_wall`).
- The report's `semantic_worker` gives:
  - the shares of jobs dropped and failed, and of results stale or late;
  - the ages' p50, p95 and max.

**F7-05, the stop route's length.** The stop scenario's car now stands as far along as lane
keeping needs for its 100 settled ticks at the cruise speed: 35 m at 2 m/s, 45 m at 4 m/s.
The requirement itself is unchanged. A shorter `--obstacle-m` says so in the INCOMPLETE.

**F7-06, the shadow by section and mode.** `semantic_shadow` splits the band ahead into
near, mid and far (0–10, 10–20 and 20+ m). For each section it logs:
- the coverage;
- the right edge;
- the stations with an edge;
- the stations where road was seen right of the line but no edge (`missed`).

The report gives coverage, the edge's signed error and MAE against the kerb, and the
missed share, by section and by the driver's mode (driving or turning). A downstream
driving benefit is still to be shown: the same route and seed with the semantic
correction on and off.

### Live on PC 1 and PC 2 (29 Oct, `b15b76f`): the r8 batch

| Run | Status | |
|---|---|---|
| Radar-first crossing, Town10HD, LiDAR at 25 m, trigger 35 m, 2 m/s | MEASURED_PASS | exercised |
| Radar-first crossing, trigger 30 m, 1.5 m/s | NOT_EXERCISED | no radar threat before the LiDAR |
| Braking lead, constant segmenter, inline | MEASURED_PASS | |
| Braking lead, constant segmenter, threaded | MEASURED_PASS | the same driving |
| Traffic cone at 4 m/s | MEASURED_PASS | 169 lane-keeping ticks |
| Crossing car, Town05 `--site 1`, 4 m/s | MEASURED_PASS | yield needed and given |
| Left turn, `--oncoming stopped` | MEASURED_FAIL | junction footprint, 0.21 m |
| Left turn, `--oncoming parked` | NOT_EXERCISED | |
| Left turn, plain | not run | CARLA timed out |

**Radar-first, exercised.** The radar had the crossing car alone for 2.6 s, unbroken. Its
provisional threat came at 7.8 s, the full one at 7.85 s, the LiDAR at 8.4 s. The first
restriction (9.05 s) had 20.5 m of room at 2.9 m/s. At 1.5 m/s from 30 m the radar had the
car alone for only 0.85 s, never as a threat.

**Inline against threaded, constant segmenter.** The driving was the same: separation
3.435 m and brake delay 0.2 s in both.

| | inline | threaded |
|---|---|---|
| whole step p50 / p95 / p99 | 45.9 / 59.0 / 62.4 ms | 44.2 / 61.8 / 64.8 ms |
| control stages p50 | 33.4 ms | 43.2 ms |

- The threaded worker dropped nothing. Results were 79 ms old on the wall clock (p50).
- Uninstrumented time was 0.8 ms p50.
- Without the v2 segmenter there is little to take off the control path, and the worker's
  interpreter contention still costs the control stages about 10 ms.

**The left turn with the oncoming car: one arc dropped for another.** The car yielded to
the oncoming car at the mouth, as it should. Then the car swept through the latched 14 m
arc's path, and its cells failed the arc's re-check.
- The driver dropped the latch at once and planned afresh from where the car stood. That
  was past the 14 m arc's tangent point, so for 2 s no radius cleared the car's fading
  cells, and then only 11.5 m did.
- The tighter arc cut 0.21 m into the exit road's far lane (`junction_footprint`).
- Fix: a latched path blocked by an object, not a kerb, is waited for. Its first hazard
  OCCUPIED, within the stopping envelope: the path is verified to a metre short of it, so the
  car holds there on the same arc, for up to `JunctionConfig.blocked_wait_s` (4 s). A kerb,
  or an object still there after that, is repaired as before. The swept check now says which
  it met (`why`); the log carries `turn_block`.
- Open-loop replay of the run: the latched arc is held for 2.15 s and then clears, and the
  car goes on along the 14 m arc 0.25 s after the live car took the 11.5 m one.
- The eleven other recorded turns: the driver's decisions are identical on every tick.
- A synthetic left turn with a box set on the latched arc for 2.5 s reproduces the failure
  before the fix (re-committed at 12.5 m against 15 m) and keeps one arc after it.

**The parked oncoming car was never in reach.** Parked 12 m beyond the far side of this
junction, it was 25.4 m from the car at its nearest, beyond the 25 m the exercise needs and
the range the LiDAR forms objects at. No track was ever on it, and the gate read the lane
clear. The verdict is right; the placement is not. The rerun parks it 4 m beyond the
junction (`--actor-param oncoming_m=4`).

**The crossing car beside the road: a stop for a car that had stopped.** In all three
crossing runs, the ego braked from 3–4 m/s to a stop, cap 0 for "predicted conflict on the
path", after the crossing car had cleared and stopped 1.9 m beside the path.
- The track's cluster was measured at its mean. The mean sits where the points are
  densest, on the faces in view. As the ego drew level it saw only the car's end face, and
  the mean moved 1.2 m towards the path in 2 s while the car stood still.
- The track read 0.3–0.6 m/s towards the path: "moving" (Town05, 0.92 m/s over the last
  second) or "uncertain" (Town10HD, 0.36 m/s). It was predicted into the path, and the car
  held 0.3–0.5 s until the drift stopped with it.
- Fix, the tracker's remembered shape (`TrackerConfig.shape_*`). A vehicle-sized track is
  measured at the centre of its remembered box, placed against the points in view:
  - Along each of the box's axes: the middle of the points' span where the span is the
    box's size, or where the sensor lies within it (a face square on).
  - Otherwise: the end nearer the sensor, plus half the box's size (the far side is out of
    view).
  - The box is the third-largest length and width the track has shown, so a merged
    cluster or two do not set it. Its axes are those of the last view that showed that
    length.
  - The views are fitted as L-shapes (`_rectangle`: the orientation whose rectangle has the
    points closest to its edges). A cluster's principal axis runs between an L's two faces,
    and the smallest rectangle is no better (an L's hull is a right triangle).
  - Vehicle-sized means 3–6 m long and 1.45–2.6 m wide (wider than a group of walkers, no
    wider than a lorry), not decided a pedestrian or a bike.
  - A cluster that shows none of the box's sides whole is a fragment. It associates where it
    lies on the box, behind any whole cluster of the object, and does not move the track.
    Placed against the box, a 0.4 m fragment of a 5.9 × 4.1 m roadside blob on Town05 put
    its centre metres off; the blob's next whole cluster then fitted a new track better, and
    that track's jump read 4.9 m/s and capped the car to 0 for a tick in the replay.
  - Whenever the box changes (first known, resized, turned, or given up when the object
    outgrows a vehicle's, as that blob did), the reference point changes with it. The track,
    its trail and origin move by the difference the cluster shows: a change of reference,
    not motion, as at a radar-held track's first LiDAR fix.
- A synthetic crossing car (4.2 × 2.0 m, the points a scanning LiDAR returns off it) that
  crosses and stops beside the road reproduces the drift. At the cluster's mean the track
  reads up to 1.0 m/s towards the path and "moving". At the box's centre it reads under
  0.1 m/s, stays static and lies within 0.15 m of the car's centre with the car's own box.
- The crossing runs recorded no frames, so the live case is to be seen on the rerun (with
  `--record-frames all`). Open-loop stack replays of seven recorded runs (no radar in the
  recordings), head against the fix:
  - Three Town05 braking-lead runs: following the lead identical on every tick. Two have
    identical caps. The third has two more ticks capped at 2.0–2.4 m/s by clutter beside
    the road, where the two versions' fragment tracks associate differently.
  - Two recorded oncoming-car left turns: the oncoming car's track lies 0.45 m from it at
    the median, against 1.4–1.5 m. Its speed held at 5.6–5.8 m/s as it passed, against a
    "deceleration" to 4.6 m/s. The car is released 17–21 ticks sooner once it has gone by.
  - Two left turns without traffic: identical.

### Eighth technical review (29 Oct; the review names no commit)

All findings were taken, and the open items it lists are now measured or enforced.

**F8-01, a rejected radar return observes nothing.** A return whose radial speed the
track's motion cannot explain was dropped as a measurement but still counted as a hit, as
radar support and as freshness. A confirmed radar-only track whose returns were all
rejected therefore stayed alive, supported and a lead candidate.
- Now it observes nothing: no hit, no support, no update of the last-seen time.
- Such a track coasts and is deleted as if unseen (test: a track fed only rejected returns
  is gone within the coasting limit).

**F8-02, the oncoming verdict expires.** The stack's verdict on the opposing lane (which
tracks it excuses, whether it is clear) is now stamped with its time and frame generation.
- The left-turn gate takes it as computed up to `JunctionConfig.oncoming_override_max_age_s`
  (0.075 s, so the normal one-tick-old verdict is fresh).
- Older, its excuses lapse: the LiDAR gate counts every occupied cell again. A "not clear"
  still holds. The log says so (`turn_gate_override`).
- A perception outage can therefore no longer keep a parked car excused.

**F8-03, a young vehicle off the road is a provisional threat.** Off the road a vehicle was
admitted as a crosser only after 30 hits (1.5 s: 7.5 m at 5 m/s for a car out of a
driveway).
- From 8 hits it is now admitted provisionally (`young_vehicle_off_the_road`) when:
  - it is at least 1 m wide;
  - its motion is not judged incoherent;
  - its filter speed is borne out by its sustained motion;
  - it shows no wider spread.
- It is treated as a provisional radar threat: a floor of 1.5 m/s that falls to 0 as its
  hits grow to 30, and never above the speed from which the car stops short at 4 m/s².
- The fifth review's road-near test (a slow car beside the road is no threat) now reads
  provisional for that track, and the scenery tests are unchanged.

**F8-04, restrictions attributed by actor.** A radar restriction counts as the target
actor's only when its cause track is one matched to that actor. Others are reported apart
as possible false interventions (`unmatched_radar_restrictions`: ticks, seconds, the first,
the tracks).

**F8-05, every actor scored.** Scripted actors now report each of their cars with a role
(the braking lead and the car ahead of it with `--actor-param second_gap_m=…`).
- `multi_actor_tracking` (REPORT) scores each role's tracking:
  - matched share and track ids;
  - id switches;
  - the ticks one track was the nearest match of two cars at once;
  - track ids that moved from one car to the other;
  - which car the lead was, and when that changed.

**Timing, sensor to actuator.** The harness now stamps five wall-clock marks per tick:
- the synchronous world tick returning;
- the rig's frame read;
- the step's start and end;
- the command applied.

`sensor_to_actuator` (REPORT) gives each interval's p50/p95/p99/max and the share of ticks
over 50 ms. In synchronous CARLA the command still takes effect at the next world tick:
these are what real time would cost.

**P1, observability-limited speed now governs.** The car never drives faster than it can
stop `stop_gap_m` short of where its view of the road ends:
- The bound uses the tube seen clear, `clear_to_m`, and the measured brake lag of 0.25 s
  (`PlannerConfig.sight_response_s`): `v·t + v²/2a ≤ clear − front − gap`.
- What it has not seen is treated as standing where its view ends. A small object seen
  late is covered by the same bound: the obstacle ends the clear tube.
- The log carries `sight_speed_mps`. `evaluation.observability` reports the same bound
  from each run's first detection.
- On all 15 recorded r8 and r9 runs the bound lay below the car's speed only with an
  obstacle in view, where the stop for it binds too: at the 2–4 m/s cruise it is a
  guarantee, not a new behaviour.

**P2, semantic wall-clock admission.** `SemanticWorker` refuses a result older than
`max_wall_age_s` on the wall clock (`stale_wall`), as it refuses a stale one.
- `StackConfig.semantic_max_wall_age_s` sets it. It is off (None) while the semantic layers
  are shadow only, and must be set before any of them steers or sets a speed.
- Learned segmentation still has no shown driving benefit: it remains experimental.

**`_lidar_free` in a changing scene.** A new test drives an actor onto ground seen free
0.2 s before.
- The first LiDAR hit on it ends "seen empty" at once.
- Ground seen 0.3 s or more ago no longer counts.
- The window in which a radar return there is judged a ghost is at most the time to the
  first LiDAR hit.

### Live on PC 1 and PC 2 (29 Oct, `b4280cd`): the r9 batch

| Run | Status | |
|---|---|---|
| Crossing car, Town10HD `--site 1`, 4 m/s (PC 2) | MEASURED_PASS | no stop after the car cleared |
| Radar-first crossing, Town10HD, LiDAR at 25 m, trigger 35 m, 2 m/s (PC 2) | MEASURED_PASS | radar alone 2.6 s |
| Cut-in then brake, Town10HD, 4 m/s (PC 2) | MEASURED_PASS | separation 2.15 m |
| Braking lead, Town10HD (PC 2) | MEASURED_PASS | separation 3.44 m |
| Crossing car, Town05 `--site 1`, 4 m/s (PC 1) | MEASURED_PASS | |
| Left turn, oncoming car pulling away (`--oncoming stopped`) | MEASURED_FAIL | separation 0.50 m, then held 130 s |
| Left turn, parked car 4 m past the junction | MEASURED_FAIL | exit 1.58 m off, held 115 s |

**Crossings.** The remembered box did what the r8 batch needed. No crossing run stopped
for the crossing car after it had cleared: 0.6 s of other restrictions after the clear,
none by the actor.

**Step time.** On PC 2 (Town10HD) the whole step was 56–60 ms at p50 against 44–46 ms
on `b15b76f`. The shape fits cost it: every vehicle-sized cluster of every track was fitted,
every tick (29 fits, 8 ms a tick on a Town10HD street: walls, hedges, parked cars). The
fits are now budgeted:
- confirmed tracks only;
- every fifth hit;
- at most four a tick, nearest first;
- at most 48 points a fit.

Replayed here on the recorded frames (this machine is slower than PC 1 or PC 2), the
tracking stage's p50:

| | `b15b76f` | `b4280cd` | now |
|---|---|---|---|
| Town10HD crossing (r9_cross4) | 18.8 ms | 29.0 ms | 20.8 ms |
| Town10HD radar-first, LiDAR at 25 m (r9_rf_35_2) | 7.8 ms | 14.9 ms | 10.8 ms |

The whole step moved with it (crossing p50 66.7 / 71.1 / 62.4 ms, within this machine's
run-to-run spread). The live step and `sensor_to_actuator` times are in the r10 section.

**The oncoming car: let go into its way.** The gate frame (the lane, latched as the
approach began) was 1.4° off. Nineteen metres on, the gate had the body "in the opposing
lane" while its corner was 0.6 m short of it.
- The gate therefore latched "entered" and stopped looking. The oncoming car was then 44 m
  out and just pulling away: beyond the gate's 40 m, and not yet tracked by the radar
  (first tracked at 26 m; the radar was not recorded, so the replay cannot say why).
- When the track appeared, the stack stopped the car at once, its corner 1.1 m clear of
  the oncoming car's side. Right.
- One tick later the clearing rule (S5: in the opposing lane, clear it rather than stop in
  it) let it go, on the gate frame's word. It drove its corner to the lane's edge.
- The LiDAR stop then held it there. The oncoming car passed 0.50 m away.
- Fix: the clearing rule needs the car to be in the vehicle's way (`in_the_way_of`). Its
  body, where it stands, must lie in the strip the vehicle drives along: the vehicle's
  width (at least 1.8 m) plus 0.3 m either side.
  - The strip is taken both along the track's velocity and along the lane (the gate frame's
    direction), and must hold on both. 20 m out, a few degrees of a track's heading move
    its strip a metre: a LiDAR-only track of this car read "in the way" for 0.35 s in the
    replay, 1.2 m clear.
  - The gate frame's place no longer decides it. Out of the way, the car waits.
- The gate's view stays short of a crossing at 2 m/s (it needs 11 s; 40 m is 4.7 s at
  30 km/h). Beyond it the stack's radar tracks must cover, and here they did not. The
  recordings now carry the radars (frame schema 2), so the next run can be replayed with
  them.

**Then held for good: the blind zone.** The car that passed 1 m in front of the bumper was
held. Its region could never show ground from there: the lowest ring the bonnet leaves
meets the road 1.4 m or more past the bumper.
- A height rule ("tall, and no longer seen") is no proof: a person who crouches there
  drops below the rings, and the passing car's corner in the tube read only 0.8 m.
- Fix: the stack says when what stood on a hold has left (`mark_departed`). The planner
  then releases the hold without ground on its region, once it has gone unseen for 0.5 s
  and the tube beyond is seen clear for the usual 1 s.
- When the hold is last detected, the stack notes the LiDAR tracks within 0.5 m of its
  near face. It is gone when every one of them is a vehicle, moving, LiDAR-supported now,
  1.5 m clear of the face and driving away from it.
- A walker, or a track lost, never releases a hold this way. The driver alone (no stack)
  waits, as before.
- The low parts of a car (sills, wheels), kerb-high and nearer than its body, were the
  detection on some ticks and carried no height. They now count as the object behind
  them (`ObstacleConfig.same_object_m`). The log has `obstacle_height_m` and
  `hold_departures`.

**The parked car: a refinement taken from a wrong far reading.** The first commit's exit
line was right, and the readings agreed with it for 4 s. Then the stations near the
junction read 1.2 m short, from 29 m across the junction, where the road's edge is between
rings. That reading held 0.5 s and was taken: the path moved 1.1 m.
- The right line, read from nearer, then lay +1.5 m from the latched one: "another road",
  refused for good. The car ended 1.6 m off the exit lane.
- Fix: a reading within the 1.5 m cap of the first commit's line is a correction back, not
  another road.
- A hold at 6.5 m on a 7.1 m corridor needed 8.5 m seen clear. The tube clearance is now
  needed only to the line's end (0.5 m short), for every hold.
- Open-loop replay: the right line is re-taken at 25.1 s.
- Not changed: why stations at 29 m read short (the road-edge rule for a wide unseen
  strip, at that grazing angle).

**Open-loop stack replays, `b4280cd` against this commit** (12 recorded runs, LiDAR only:
the recordings had no radar):
- Town05 braking-lead runs (three), the r6 plain left turn, the r7 oncoming left turn: the same
  decisions on every tick, bar two ticks of a 2.0–2.4 m/s cap.
- r9 Town05 crossing (PC 1): `b4280cd` stopped 18 ticks for the crossing car after it had
  stopped beside the road (its track at 0.5–0.75 m/s). Now it reads 0.17–0.21 m/s, and no
  stop. The PC 2 crossings (Town10HD): the same 8 conflict ticks each.
- r9 oncoming left turn: the clearing rule let the car go on 104 ticks, now 5 (with its
  corner at the lane's edge). The hold is released at 22.4 s, 1.45 s after its last
  detection, not held to the end (2581 ticks → 29).
- r9 parked left turn, r8 left turns: one to three ticks differ (a parked car's drift capped
  one tick, out of its way).
- The observability bound fell below 2 m/s only in the Town05 braking-lead runs, with the
  stopping lead in view; the stops are unchanged.

### Live on PC 1 and PC 2 (29 Oct, `a73f5e0`): the r10 batch, all eight passed

| Run | Status | |
|---|---|---|
| Left turn, oncoming car pulling away (Town03, PC 1) | MEASURED_PASS | separation 1.36 m; footprint −0.53 m; exit 0.05 m off |
| Left turn, parked car 4 m past the junction | MEASURED_PASS | footprint −0.44 m; exit 0.01 m off |
| Left turn, no traffic | MEASURED_PASS | exit 0.00 m off |
| Crossing car, Town05 `--site 1`, 4 m/s (PC 1) | MEASURED_PASS | separation 2.75 m |
| Braking lead, a second car 8 m ahead of it (Town10HD, PC 2) | MEASURED_PASS | separation 3.47 m; both cars tracked on 96% of ticks |
| Crossing car, Town10HD `--site 1`, 4 m/s | MEASURED_PASS | separation 2.73 m |
| Radar-first crossing, LiDAR at 25 m, trigger 35 m, 2 m/s | MEASURED_PASS | radar alone 2.6 s, threat before the LiDAR |
| Cut-in then brake, 4 m/s | MEASURED_PASS | separation 2.22 m |

**The oncoming left turn.** The gate again read "clear for 40 m" with the oncoming car
standing 46 m out.
- This time the exit reading dropped to "route" for a few ticks (14.25 s). The car slowed
  at the mouth and reached the gate's entry later.
- At 16.85 s the radar picked the car up, 30 m out and moving. The stack's verdict said
  not clear, and the gate held the car at the mouth, 1.36 m from the passing car.
- The hold on the latched path was released by the new departure rule at 21.15 s. The gate
  then waited for the passed car's cells to fade (24.1 s), and the car turned.
- **Why the radar was late.** The recordings now hold the radar. Along the oncoming car's
  bearing the front radar had no return between 30 and 50 m on any of 100 ticks: road at
  5–30 m, buildings beyond 50 m at 1.7° or more up. From 0.7 m up, the road beyond about
  30 m is out of its sight on this site; the LiDAR, 2.2 m up, sees further.
- So here a car standing 45 m out is beyond the radar and beyond the gate's 40 m. The gate
  cannot see what the crossing needs (11 s at 2 m/s), and this pass came from timing. The
  r9 fixes (clearing only in a vehicle's way, holds released on departure) keep it safe
  when the timing is against it. The report now says how far short the gate's view was
  (`oncoming_gate_view`: on this run, every one of its 155 clear verdicts).

**Step time, PC 2 (Town10HD).** The whole step against r9 on `b4280cd`, p50:

| | r9 | r10 |
|---|---|---|
| crossing, 4 m/s | 60.2 ms | 42.6 ms |
| braking lead (r10 with the second car) | 56.2 ms | 39.7 ms |
| cut-in then brake | 58.3 ms | 48.8 ms |
| radar-first | 39.1 ms | 28.9 ms |

- Over 50 ms: 1–46% of ticks.
- PC 1's last two runs were about 35% slower in every stage, the code included or not (the
  same left turn: 43.9 ms with traffic, 62.7 ms without). That is the machine's load.

**Sensor to actuator** (the five wall-clock marks), PC 2, p50: 64–89 ms.
- Delivery, from the world tick to the frame assembled (four cameras, the LiDAR, three
  radars): 26–31 ms.
- The frame recorder: 8–9 ms (only with `--record-frames`).
- The step, then 0.06 ms to apply the command.
- Over 50 ms on 95–97% of ticks. Delivery is the largest fixed cost. In synchronous CARLA
  it slows real time, not the simulated car.

**The braking lead's identity: ninth review, N9-01, seen live.**
- The scripted lead stopped from 2.8 m/s in 0.15 s, about 18 m/s². 3 m/s² was asked: the
  fixed pedal (`brake_mps2 / 8`) is far stronger than that at low speed. The cut-in car
  braked at 14 m/s².
- The lead's radar return was rejected (its radial speed far from the track's). A second
  return off the lead, unclaimed, started a radar track. That track took the next returns,
  then the LiDAR cluster: its large uncertainty made it the cheapest match.
- The lead's own track coasted at 2.6 m/s and was dropped. The new one read the stop at
  once (0 m/s), but as a new lead with no history. For 0.35 s its noisy first motion read
  "moving", so it was predicted as a crossing conflict; for a tick the car even yielded to
  it as a road user crossing (cap 0.8 m/s).
- Separation stayed 3.47 m: the LiDAR obstacle stop does not wait for the tracker.
- The report's brake delay read 0.5 s. It had skipped every tick whose lead was predicted as
  crossing. It now takes the lead's speed on those ticks too: 0.15 s on r10 (the new
  track's), 0.2 s on r9, and 0.15 s for both cut-ins.
- The fixes are in the ninth review's section below.

**Two cars ahead.** Each car was tracked on 96% of ticks. No track stood on both, none moved
from one car to the other, and the lead was the lead car on every tick: no hand-over to the
car ahead of it.

**Replay with the radar.** The stack replayed over these recordings, radar included, makes
the live decisions: the same cap on 534 of 535 ticks (two cars), 513 of 517 (radar-first),
441 of 449 (crossing) and 380 of 390 (cut-in).

### Ninth technical review (29 Oct): all taken

**N9-01, the radar's rejection must not cost the braking lead.** The review asked whether
rejecting a return whole loses a hard-braking lead. On r10 it did (above).
- **Fix: the object's other returns are its own.** An unclaimed radar return is not
  started as a track when it lies on a LiDAR-placed track's object (inside its footprint
  grown by 1 m, or up to 3.5 m behind its face), and that track's own return this tick was
  rejected with a radial speed within 1 m/s of this one's. The returns agree with each
  other, not with the track: the object's motion changed.
  - They then go to the track, whose rejection run is taken as a manoeuvre in 0.15 s.
  - A different object's returns behind a parked car are untouched: the parked car's own
    return is not rejected.
  - Counted per tick (`radar_stats.same_object`).
- Replayed on r10_lead_pair with the radar: the lead keeps its one track (no id switch).
  Its speed reads 0.26 m/s at 10.1 s, 0.05 s after the lead came to rest, and 0.06 m/s by
  10.2 s. No twin is started, and the brake is seen 0.15 s after it began.
- **The scripted brake now follows the asked rate** (`braking_profile_control`): the brake in
  proportion to the speed over a constant-deceleration profile, and full once stopped.
  - Against a plant as strong as CARLA's at low speed, it holds 1, 3 and 4 m/s² within 15%.
  - `lead_tracking` reports the rate achieved (`brake_achieved_mps2`: 18.5 and 13.9 m/s² on
    r10).
- **The review's braking matrix, as tests** (a lead 20 m ahead, its LiDAR face and two
  radar returns off it each tick):

| Case | Result |
|---|---|
| 1 m/s² from 6 m/s | one track on the lead throughout; speed within 0.3 m/s of the truth |
| 4 m/s² | one track; the radial-speed test rejects from the second tick, and the estimate trails by up to 1.1 m/s until the run is taken as a manoeuvre |
| 18 m/s² from 2.8 m/s (the scripted stop) | one track; under 0.6 m/s 0.3 s after the stop |
| radar gone for 1 s while braking | the LiDAR carries the track; within 0.6 m/s |
| both sensors gone for 0.5 s | out of the live set after 0.3 s, its velocity uncertainty growing; back under the same id |
| corrupted Doppler on a lone radar track | F8-01's test: no hit, no freshness, deleted |

- Not covered: 18 m/s² from 6 m/s. The LiDAR gate is left within 0.25 s and a new track
  takes the stopped car, while the old one only coasts. No car brakes that hard.
- A wider radial-speed test (8 m/s² of change between updates) took the 4 m/s² case to a
  0.7 m/s lag. It also let scattered Doppler through on five existing tests, so it was not
  kept.

**N9-02, greedy against optimal association.** `TrackerConfig.assignment = "optimal"` is
the gated global nearest neighbour.
- It finds the least total cost, each track or measurement left unassigned costing the
  gate. It is solved exactly per group of gated pairs (Kuhn-Munkres), and the groups are a
  few tracks and clusters each.
- The review's example (track 1 at 1.0 and 2.0, track 2 at 1.5 and out of gate) keeps both
  tracks. On 200 random gated matrices it is never worse in total cost.
- Benchmarked on eight recorded runs, actor-matched (below). The default stays greedy.

Open-loop stack replays of the recorded runs (radar included for r10; actor-matched
identity switches as `actor_tracking` scores them; ticks capped under 0.5 m/s; the
tracking stage's p50 on this machine):

| Run | greedy switches | optimal switches | greedy caps | optimal caps | tracking p50 |
|---|---|---|---|---|---|
| r10 two cars ahead | 0 | 0 | 0 | 0 | 42.5 / 44.5 ms |
| r10 crossing, Town10HD | 0 | 1 | 3 | 0 | 33.4 / 33.9 ms |
| r10 cut-in | 0 | 0 | 20 | 20 | 46.3 / 47.4 ms |
| r10 radar-first | 0 | 9 | 1 | 15 | 18.0 / 18.1 ms |
| r10 crossing, Town05 | 10 | 5 | 0 | 0 | 15.3 / 15.8 ms |
| r10 oncoming left turn | 0 | 0 | 80 | 76 | 29.5 / 29.1 ms |
| r9 crossing, Town05 (no radar) | 0 | 0 | 0 | 0 | 6.5 / 6.6 ms |
| r8 oncoming left turn (no radar) | 1 | 0 | 66 | 66 | 18.9 / 20.2 ms |

- Optimal is better on the Town05 crossing (the crossing car's fragments) and the r8 turn.
- It is worse on the radar-first run: the radar-only crossing car's returns spread over
  ten tracks, and 14 more ticks were capped.
- Elsewhere the two are the same. The cost is 0.1–2 ms.
- With one or two road users near the car there is little for a global assignment to
  resolve, and where it changed the outcome it did not improve it on balance: 15 switches
  against 11. Greedy stays; the option stays for a denser scene.

**N9-03, is the tracker's confidence calibrated?** `lead_tracking` now gives the lead
speed's NEES by phase:
- the first second of a lead track (acquired);
- cruising;
- the actor braking;
- stopped.

It also gives the LiDAR NIS's verdict (about 2 when its stated noise is right).

| | acquired | cruise | braking | stopped | LiDAR NIS |
|---|---|---|---|---|---|
| r10 two cars | 2.13 (40 ticks) | 0.04 | 83 (2 ticks) | 0.72 | 0.03 |
| r10 cut-in | 0.01 | 0.08 | 28 (1 tick) | 0.70 | 0.06 |
| r9 braking lead | 3.35 | 0.04 | 83 (2 ticks) | 1.18 | 0.04 |

- Cruising, the speed's stated uncertainty is about five times the error (NEES 0.04 where 1
  is consistent): the 1.5 m/s² process noise.
- In the scripted 18 m/s² stop the filter is overconfident for the tick or two before the
  rejection run opens its variance.
- Stopped, it is consistent. The LiDAR's 0.3 m position noise is conservative: NIS 0.03–0.06
  where 2 is consistent. Its position updates are slower than they need be.
- Not changed: a manoeuvre mode (an interacting multiple model) or tuned noise. These
  numbers are the baseline for either, and the evaluator keeps truth to itself.

**Section 3, the eighth review's safety regressions**, where each now stands:

| Concern | Covered by |
|---|---|
| Oncoming override after sensor loss | F8-02: the verdict is stamped and its excuses lapse after 0.075 s (test). New: `--drop-sensor NAME:START:END` takes a sensor out of the live frames for a window (status MISSING, recorded, in the report): the live T4 |
| Fast emerging vehicle taken for scenery | F8-03: provisional from 8 hits (test); live T5 below |
| Radar-first attribution | F8-04: `unmatched_radar_restrictions`, 0 on r10's radar-first run |
| Multi-vehicle evaluation | F8-05: `multi_actor_tracking`, two cars on r10 (above) |
| Provisional radar restriction strength | F7-03: the floor falls to 0 and never exceeds a 4 m/s² stop. On r10's radar-first run the first restriction had 20.5 m of room, 0.21 m/s² needed |
| Object observability | the planner's speed bound (eighth review, P1); the cone runs measure the envelope |

**Section 4, real time.** Measured above, on r10:
- the whole step p50 29–49 ms on PC 2, p95 42–66 ms;
- sensor to actuator p50 64–89 ms, mostly the sensor data's delivery to the client;
- the semantic worker's results 70–96 ms old on the wall clock (constant segmenter).

The v2 segmenter on the worker, with no frame recording, is in r11. The LiDAR obstacle stop
stays on the command path. Nothing required before the command is left out of the step.

**Section 5, ML.** Unchanged and not claimed:
- the learned segmentation runs in shadow (configuration B);
- `semantic_shadow` measures its coverage and edge error by section;
- no configuration C (semantic-assisted) exists yet.

The review's suggested experiment (a road edge hidden by a parked car, or an ambiguous
junction exit) fits the r9 parked-car refinement: the stations read short at 29 m are a
known geometric failure to test it on. Model checkpoints are hashed in every report
(`stack.segmenter_sha256`), and the seed is now recorded (`seed`).

**Section 6, the test matrix.**

| | | Where |
|---|---|---|
| T1 | rejected Doppler | F8-01 test |
| T2 | hard-braking lead | N9-01 tests; live braking lead at 1, 3 and 4 m/s² (r11) |
| T3 | close cars | live, second car at 3 and 8 m (r11) |
| T4 | oncoming car during sensor loss | live, `--drop-sensor roof` across the gate (r11) |
| T5 | driveway exit | live crossing car from off the road, two seeds (r11) |
| T6 | stationary Town05 structure | the r3b/r4/r6 replays; live straight run on the same site (r11) |
| T7 | radar-first | r8, r9, r10 |
| T8 | radar-first with clutter | `unmatched_radar_restrictions` on every radar-first run |
| T9 | lead change | cut-in; two cars |
| T10 | small obstacle | cone at 2 and 4 m/s (r11) |
| T11 | threaded semantics | r11 with the v2 segmenter, inline against threaded |
| T12 | semantic-assisted following | not implemented |

The evaluator already keeps PASS, FAIL, NOT_EXERCISED and INCOMPLETE apart.

### Live on PC 1 and PC 2 (30 Oct, `84fb071`): the r11 batch, all eleven passed

| Run | Status | |
|---|---|---|
| Left turn, oncoming car, front radar out 14–17 s (Town03, PC 1) | MEASURED_PASS | separation 1.36 m; footprint −0.49 m; exit 0.03 m off |
| Left turn, parked car 4 m past the junction, LiDAR out 15–15.5 s | MEASURED_PASS | footprint −0.33 m; exit 0.03 m off; re-planned twice in the turn |
| Crossing car, Town05 `--site 1`, 4 m/s, seed 1 | MEASURED_PASS | separation 2.77 m |
| Crossing car, seed 2, triggered at 20 m | MEASURED_PASS | separation 2.80 m; a stop from 3.9 m/s beside the stopped car |
| No traffic, Town05 `--site 1`, 4 m/s | MEASURED_PASS | no restriction |
| Braking lead at 1 m/s² (Town10HD, PC 2) | MEASURED_PASS | 1.13 m/s² achieved; brake seen in 0.2 s; separation 3.48 m |
| Braking lead at 3 m/s² | MEASURED_PASS | 4.61 m/s² achieved; brake seen in 0.15 s |
| Braking lead at 4 m/s², a second car 3 m ahead of it | MEASURED_PASS | 5.03 m/s² achieved; lead tracked 96%, second car 76% |
| Traffic cone, 4 m/s | MEASURED_PASS | first seen 24.5 m out; held 3.4 m short |
| Braking lead, v2 segmenter inline | MEASURED_PASS | step p95 126 ms |
| Braking lead, v2 segmenter on its worker | MEASURED_PASS | step p95 73 ms; results 183 ms old |

**The radar outage missed the oncoming car.** Neither sensor had it before 17.0 s: the radar
first saw it 29 m out at 17.0 s, the LiDAR 25 m out at 17.5 s (the radar's ~30 m line of
sight on this site, r10). The outage ended at 17.0 s. Every decision was r10's: the same
stop at 17.5 s, the same 1.36 m. The next run drops the radar over 16.5–20 s.

**A hard brake at the mouth (14.2 s, r10 and r11 alike).**
- Approaching the mouth, the exit kerb's near stations go out of view. When they aged out
  of the 2 s memory, the stations still in view spanned 1.06 m: no line. A tick before,
  52 stations had made one.
- The measured exit was lost for 0.25 s, so the turn was not committed and the car was held
  at the mouth, 0.7 m ahead. It braked at 0.95 from 2 m/s and moved off again at once.
- **Fix: the stations are kept 2 s longer, read only when the last 2 s's make no line**
  (`JunctionConfig.exit_fallback_s`; the fit says `memory: fallback`). The recent stations
  still come first.

**Two re-plans in the parked-car turn: the ground read along the heading.**
- From 18.4 s the exit road, level with the car 20–30 m to its left, read as kerb, and from
  20.4 s as obstacle: 0.1–0.5 m above "the road". The road profile is grown in bands along
  the car's heading; mid-turn that is the street straight on, which dips 0.1–0.7 m past
  the junction. A point far to the side shares only its x with those bands.
- The exit kerb was then read 1.2 m into the road (stations at 28 m, the reading the r9
  notes left open). It held 0.5 s and was taken: R 14 → 11.9 m, the clearance fell to
  0.05 m, and the car stopped at 21.75 s. The right line, read from nearer at 24.5 s, came
  back within the first-commit cap: R 20.4 m, a second stop, 0.83 m off the line in the
  junction. The exit was 0.03 m off.
- **Fix: far points are referenced along the path the car drives.** The same profile is
  grown over arc length and distance along the last tick's corridor (straight back from its
  start and on from its end). Points 10 m or more away, outside the bands' 4 m window, take
  its height at their arc length (`classify_sweep(path_xy=...)`, `TerrainConfig.path_profile*`).
  - On a straight road the two are the same, and a path within 1 m of the heading is not
    computed. The near field, where the corridor reads its kerbs, is always the bands'.
  - Not while the turn is still being chosen at the approach. There the path moves with
    each reading, and the first version held the synthetic left turns at the mouth (three
    unit tests): the arc's clearance read 0.2 m against the 0.3 m it needs. From commitment
    the path is the latched exit's, and the profile applies.
  - Replayed: the exit line stays within 0.3 m of the latched one, R stays 14 m and the
    clearance 2.05 m. Behind the turning car the approach road's pavement, which the bands
    had climbed onto, reads raised again.
  - Cost: about 3 ms a sweep while the path leaves the heading.

**The LiDAR outage (15–15.5 s).** The driver faulted (brake 0.6) and stood until sweeps came
back, then drove on. As designed.

**A stop beside a car that had stopped (seed 2, 19.75 s).**
- The crossing car stopped 6 m left of the path, across it, its near end 3.9 m out. As the
  car drew level at 3.9 m/s, the track read 0.5–0.7 m/s towards the path: the cluster's mean
  slid to the near end face as the side went out of view. "Vehicle within reach" became a
  predicted conflict, and the car braked to a stop.
- The remembered box (r8) exists for this. Live, the track was not measured against one:
  its footprint shrank to 3.1 m, below any box it could have had. In replay it was, but the
  replay's tracks part from the live ones at 1.5 s (the recording keeps the LiDAR to the
  centimetre). With the shape memory off, the replay stops the same way.
- **Fix: motion is believed when the object's leading edge moves with it.** Each LiDAR
  cluster 1.5 m or longer keeps its extent along 16 directions; `edge_speed_mps` is how fast
  its farthest extent along the track's velocity advanced over the last second. A rigid
  object that moves carries it; a static one whose visible part changes does not.
  - Live recordings, the stopped car: 0.00–0.08 m/s for a 0.5–0.7 m/s track. Crossing,
    oncoming and cut-in cars: 0.85–1.0 of their speed. The cut-in braking to rest dropped
    below that only at its stop.
  - A LiDAR track no faster than 2 m/s, not a pedestrian, whose edge advanced at most a
    quarter of its speed (or 0.15 m/s) is predicted where it stands, as an incoherent one
    is, outside the lane band (`StackConfig.edge_*`, `stands_still`). The log lists them
    (`edge_held`) and gives each cap track's `edge_speed_mps`.
  - Faster phantoms are untouched: walls sliding at 4–5 m/s as the turning car's view swept
    them (parked-car turn, 21.85 and 25.0 s) move their visible edge too.

**Scripted braking overshot.** The pedal profile achieved 4.61 and 5.03 m/s² for 3 and 4
asked, the lead's speed jumping 2.8 → 1.0 → 1.6 m/s in 0.2 s; at 1 m/s² it achieved 1.13
in steps. CARLA's brake at low speed is stronger than a 20 Hz loop can follow.
- **Fix: the lead's velocity is set each tick to the profile's** (`scripted_braking`,
  CARLA's `set_target_velocity`), the pedals released, and held on the brake once stopped.
  The cut-in car brakes the same way. A stand-in actor keeps the pedal profile.

**Tracking under braking (N9-03).** Speed NEES by phase: cruise 0.04, stopped 0.05–0.26,
braking 3.8 at 1 m/s² (50 ticks), 22.8 at 3 (15 ticks) and 32 at 4 (10 ticks). The
constant-velocity filter lags a hard stop for a few ticks; the brake was seen in 0.15–0.2 s
each time. LiDAR NIS 0.03 (conservative).

**The close pair (3 m).** The lead was tracked on 96% of ticks, one track, never handed over.
The second car, mostly behind it, on 76%, with one id switch; no track stood on both.

**The cone.** First seen 24.5 m out at 3.5 m/s; the speed that view supports is 6.4 m/s
(1 m/s², 0.25 s, 2.5 m), so 4 m/s is covered. Held 3.4 m short.

**The v2 segmenter, inline against its worker** (braking lead at 3 m/s², no recording).
- Inline: the step p50 59 ms, p95 126 ms (painting 23 ms, the road model 7 ms); sensor to
  actuator p95 155 ms, 572 ms on the first inference.
- On the worker: the step p50 53 ms, p95 73 ms; sensor to actuator p95 99 ms. The worker
  ran 13 ms a job but its results waited 150 ms in the queue (p50) and were 183 ms old when
  taken; 32% of jobs were dropped for newer ones. The design trade (latest wins, nothing
  on the command path waits); shadow only, so no rule reads them.
- Both read the right edge 0.76 m inside the kerb (near section, MAE 0.86 m).

**Step time.** PC 2: p50 45–47 ms, p95 63–66 ms. PC 1: 35–46 ms, but 76 ms on the parked
turn, every stage slower (classify 2.5 ms against 1.7 on the oncoming run: the machine's
load) and more scenery to track (24.7 ms). Sensor to actuator p50 77–90 ms; 129 ms on the
parked turn.

**Open-loop replays, `84fb071` against this commit** (29 recordings, r6–r11, the radar
replayed from r10 on): 24,309 ticks.
- Ticks under a zero cap: 505 → 497. Stopping: 1,521 → 1,496. Ticks whose stop was an
  obstacle hold: 1,758 → 1,085.
- **The holds that went were phantoms on exit roads mid-turn** (the ground fix). A one-tick
  "obstacle" 25 m on put a stop point on the path for 6–8 s in the r8, r10 and r11 oncoming
  turns, and for 8.6 s in r10's turn with no traffic.
- **The caps that went were parked or stopped cars passed close** (the leading-edge rule):
  - in the cone run, a 3.5 x 1.4 m "vehicle within reach" 4.6 m right of the path at
    2.7–3.1 s (live: caps of 3.2 m/s and then 0);
  - the crossing car stopped after crossing, in the r9 and r10 radar-first runs and r9's
    crossing.
- **Genuine threats are unchanged**: the crossings (Town05, Town10HD), the cut-in, the
  oncoming cars and the braking leads keep the same caps, bar single ticks where the
  cause's track id differed.
- The r9 and r11 parked-car turns complete on their first arc and hand back on time.
- Not fixed: wall stretches sliding at 4–5 m/s still cap the car for 0.1–0.25 s in the
  parked-car turns, on other ticks than before (live at 21.85 and 25.0 s). r10's turn with
  no traffic gains one: 0.2 s at 19.35 s, a coasting 5 m "car" 27 m out at 4.9 m/s.

### Tenth technical review (30 Oct): all taken

The review could not see the code. It sets out the invariants the fixes must keep, and asks
for transition tests. Each invariant is now a test (`tests/unit/test_mapless_tenth_review.py`
unless named). Two of them failed and are fixed (2C), and one gap is closed (2B).
Replayed over 13 recordings (the r8, r10 and r11 left turns, crossings, leads, the cut-in,
the cone), the stack's caps, modes and stops are the same on every tick: the new rules act
only when a blocker or lead is lost, which none of those recordings has.

**2A, radar rejection.** A return whose radial speed is rejected observes nothing (F8-01):
no hit, no support, no freshness. Three consistent rejections in a row are taken as a
manoeuvre (N9-01).
- Compared with keeping the return's range and bearing (`TrackerConfig.doppler_reject_policy
  = "position"`, still no freshness). The case: a braking lead, the LiDAR lost at 0.8 s, the
  radar alone to 3 s, at 1–12 m/s², with and without 0.3 m of scatter on the returns.
  - Both kept one track throughout and saw the stop on the same tick.
  - Speed RMSE while braking: the same to within 0.01 m/s up to 8 m/s²; at 12 m/s², 0.79
    against 0.84 m/s.
  - Replayed over the seven recordings with radar and a braking or crossing car (the r9
    and r10 radar-first crossings, r10's crossing, braking cut-in and close pair, r11's
    leads at 3 and 4 m/s²): not one tick's cap or stop differs.
- The default stays "drop". The review's other properties: a rejected return is not
  freshness (F8-01's test); a hard-braking lead keeps its track (N9-01's matrix); a lost
  track does not lift the cap (2C below); the driver's LiDAR obstacle stop never depends on
  the tracker.

**2B, junction clearance.** The driver takes the stack's oncoming verdict only while it is
fresh (F8-02): one tick old. Older, its excuses lapse and a "not clear" stands.
- **The gap.** A radar-only car 41 m from the crossing blocks the turn. The radar drops
  out; the track coasts, then is deleted. The verdict alone then reads clear, with the car
  still beyond the LiDAR's 40 m.
- **Fix:** a blocker that is no longer observed and was not seen to leave (deleted, or held by
  no sensor) keeps the verdict "not clear" for 3 s after it last blocked
  (`StackConfig.oncoming_lost_hold_s`). At 30 km/h that brings a car from 40 to 65 m out
  into the LiDAR's view. A blocker observed driving away, parked or too far releases it at
  once, as before. The note says which track is held.
- Replayed on r10's oncoming turn with the radar cut from 17.2 s, 0.2 s after it first saw
  the car: the LiDAR had taken the same track over by 17.25 s, nothing was lost, and every
  verdict is the same. On Town03 the radar sees the car only about 0.25 s before the
  LiDAR, so a live run exercises the hold only if both lose it. r12 runs the standing car
  4 m past the junction, pulling away with the radar out (the stationary-to-moving
  transition the review describes).

**2C, restrictions continuous across transitions.** The suite runs the stack's cap logic
closed loop. It checks that the permitted speed never rises faster than the car's own
1 m/s² while the hazard stands, across six transitions:
- a crossing car becoming the followed lead;
- the followed lead's track replaced (coasting old track, tentative new one);
- the lead lost for 0.5 s;
- a radar-only crosser handed to the LiDAR;
- a standing car moving off into the path (the tracker in the loop);
- a stop hold released while another conflict's zero cap stands (the planner).

Two of them lifted the restriction:
- **The lead lost for 0.5 s.** After its 0.3 s coast the cap went to cruise for 0.2 s. A
  followed lead whose track is gone, not seen to leave the lane, is now followed for 1 s
  where it would be at its last speed: the lower of its last cap and the worst-case cap
  behind it (`lead_memory_s`). A new lead ends that at once.
- **A radar-only crosser at the LiDAR's first fix.** The radar track was dropped as the twin
  of a LiDAR track whose motion was not known yet, and the cap went from 2.8 m/s to none
  for three ticks before the two merged. The radar track now carries the threat until its
  twin's motion is known.

**3, admission and prediction.**
- **Paired test** (the review's suggestion): the same roadside place, 5 m left and 22 m
  ahead. A 9 m fence whose visible stretch slides as the car passes never restricts it. A
  car pulling out of a driveway there at 2 m/s restricts it 0.65 s after it moves, the
  same with the leading-edge rule off, and the car stops short.
- **Prediction coverage.** The stack logs each predicted region at 1 and 2 s
  (`predictions_near`). The new `prediction_coverage` check reports, per horizon:
  - the share of ticks whose actor's true position, that far on, was inside the region;
  - the centre error and the worst miss;
  - the region's area over the actor's own (how conservative it is).

**4, the 50 ms requirement.** Every run already reports the step's p50, p95, p99 and maximum,
the deadline share, sensor to actuator, and the semantic results' age.
- New, for the review's three configurations:
  - `--semantic off`, the geometric stack alone. It drives exactly as inline (test).
  - `--cpu-load N`: N processes spinning a core each.
  - `--gpu-load`: back-to-back matrix products on the CUDA device.
- **`latency_budget`** puts each rule's assumed response beside the measured sensor-to-actuator
  p99 plus one tick: 0.15–0.24 s on r11 (0.62 s worst, inline v2's first inference).
  - Following (0.5 s) and the observability bound (0.25 s) cover it.
  - The planner's ramp and the conflict cap assume no response. Their margins (2.5 m stop
    gap, 2.0 m conflict margin) carry the lag: 0.6–1.0 m at 4 m/s. The check reports that
    distance.

**5, observability.** The planner applies the bound every tick (eighth review, P1).
- **`sight_bound`** counts the ticks the car was more than 0.3 m/s over it.
- **The cone report's `reliable`** gives the range the detection held from (every 1 s
  window 80% seen, no gap over three ticks), not only the first sighting, with the speed it
  supports.

**6, ML.** Unchanged, and not claimed. The learned segmentation runs in shadow. On r11 the
v2 segmenter put the right edge 0.76 m inside the kerb (near section, MAE 0.86 m), against
the geometric kerb's few centimetres. That is not good enough to assist, so there is no
configuration C. The experiment the review describes (hard cases: occluded kerbs and
junction exits like the r9/r11 parked-car turn; held-out locations; seeds) is the test it
would have to pass.

**7, the release gate.**

| Test | Where |
|---|---|
| Rejected radar Doppler | F8-01 test; N9-01 matrix; 2A comparison |
| Temporarily lost oncoming observation | 2B test; F8-02 test; r10 replay with the radar cut; r12 radar out from 16.5 s, and a standing car pulling away with no radar (`oncoming_gate_view.lost_blocker_held_ticks`: 0 is not exercised) |
| Conflict-to-following hand-over | 2C suite (six transitions) |
| Static Town05 scenery | r11 Town05 plain, no restriction; the r3b/r4/r6 replays |
| Fast emerging real vehicle | paired driveway test; r12 crossings seed 2 and 3 |
| Radar-first crossing | r10, r12 `rf_35_2`; `radar_timeline` |
| Multi-vehicle tracking | `multi_actor_tracking` per role (r10 8 m, r11 3 m) |
| Small obstacle | cone `observability` with `reliable`; `sight_bound` |
| Full threaded execution | `driver_step_time`, `sensor_to_actuator`, `latency_budget`; r12 off / thread / thread under load |
| Semantic-assisted driving | not implemented (6) |

The evaluator keeps PASS, FAIL, NOT_EXERCISED and INCOMPLETE apart. A transition that
never occurred is `NOT_EXERCISED`, never a pass.

## Running Phase 0 live

Run with CARLA 0.9.16 and no other clients:

```powershell
# 1. Calibration gate on the demo map (no training data is kept)
python scripts/check_sensor_rig.py --output "$HOME\Documents\Group7_runs\rigcheck_v2"

# 2. Sat-nav route for the baseline mission (the same saved mission used for live runs)
python scripts/make_satnav_route.py --mission <saved mission.json> --name baseline `
    --output configs/satnav/baseline.json

# 3. Training data, one run per town (Town01-Town06; Town07 is the held-out test town)
python scripts/capture_mapless_dataset.py --map Town03 --samples 2000 `
    --output "$HOME\Documents\Group7_runs\capture_pilot_Town03_s5308"

# 3b. Inspect a capture run (summary.json, verdict and contact sheets in <run>\inspection)
python scripts/inspect_mapless_capture.py --run "$HOME\Documents\Group7_runs\capture_pilot_Town03_s5308"

# 3c. Repair schema-1 runs (truth box centres); needs a CARLA server once, then --offsets
python scripts/repair_mapless_capture.py --run "$HOME\Documents\Group7_runs\capture_pilot_Town03_s5308" `
    --run "$HOME\Documents\Group7_runs\capture_pilot2_Town05_s5309" --export-offsets configs/box_offsets_0916.json

# 4. G1b first mapless drive (current map; --map to load another)
python scripts/run_mapless_drive.py --scenario straight --output "$HOME\Documents\Group7_runs\g1b_straight"
python scripts/run_mapless_drive.py --scenario curve --output "$HOME\Documents\Group7_runs\g1b_curve"
python scripts/run_mapless_drive.py --scenario stop --output "$HOME\Documents\Group7_runs\g1b_stop"

# 5. G1c first commanded turn (site census in the error message if none fits; --site N for another)
python scripts/run_mapless_drive.py --scenario turn_right --output "$HOME\Documents\Group7_runs\g1c_right"
python scripts/run_mapless_drive.py --scenario turn_left --output "$HOME\Documents\Group7_runs\g1c_left"
#    Turn runs record the sensor frames under <output>\frames (--record-frames none|turns|all, ~90 KB a
#    tick); zip the run folder for the offline replay on the development machine:
python scripts/replay_mapless_frames.py --run <run folder> --junction exit_kerb_edge='"midpoint"' --out replay.jsonl

# 6. G2 road segmenter on the accepted captures (needs `pip install -e .[test,docs]` for transformers;
#    not while CARLA is rendering on the same GPU). Compare the two by the per-epoch validation JSON.
#    One --run per accepted capture (Town02 is the validation town, the rest train); --allow-unaccepted
#    takes a run whose inspection failed only on manifest status (an interrupted wrap-up). ~48k training
#    images at 512x256: batch 16 and 6 loader workers by default (RTX 4000 Ada, 20 GB); a
#    progress line every 200 steps gives the ETA; --max-samples 20000 for a first result in an hour.
python scripts/train_mapless_segmentation.py --run "$HOME\Documents\Group7_runs\capture_Town01_s5311" `
    --run "$HOME\Documents\Group7_runs\capture_Town02_s5314" --run ... --output models\road_seg_b0.pt --architecture segformer_b0 --epochs 8
python scripts/train_mapless_segmentation.py --run ... --output models\road_seg_bisenet.pt --architecture bisenetv2 --epochs 8
# 6b. Score checkpoints on Town02 with today's labels (no training; one JSON line each, with the
#     checkpoint's own tag table and whether it matches): old against retrained.
python scripts/train_mapless_segmentation.py --run ... --evaluate models\road_seg_b0.pt --evaluate models\road_seg_b0_v2.pt

# 7. G2/G3 stack on the drive harness (after G1c; --admission A|B|C, --segmenter <checkpoint>, --detector yolo)
python scripts/run_mapless_stack.py --scenario straight --admission B --output "$HOME\Documents\Group7_runs\stack_straight"
```

**Phase 0 gate:**
- `rig_check.json` status is `MEASURED_PASS` for the exact rig (v2). If the radar sign
  checks fail, flip `RADAR_VELOCITY_IS_RANGE_RATE` and re-run.
- Any change to mounts, FOV, resolution or timing means rerunning the gate.
- The firewall tests pass.

## Next (Phase 1)

- Road and lane-marking segmentation trained on the captures.
- LiDAR-painted top-down road map.
- LiDAR ranging inside detector boxes, replacing depth.
- Object tracker with velocities.
- Traffic-light state.
- ~~Heading initialised from the compass; the EKF fed GNSS, IMU and the speedometer.~~
  Done in G1b (`ego_motion.py`).
