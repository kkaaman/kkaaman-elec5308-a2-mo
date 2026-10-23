# Mapless autonomy: Phase 0, P0.1 and G1b (contract, rig, health, calibration gate, local controller, first mapless drive)

Branch `feature/mapless-autonomy`, built from `feature/shadow-localisation`. This is an
extension: the frozen baseline is unchanged.

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
| Corridor | `corridor.py` | Lateral profiles along last tick's centreline; right kerb = first raised cell with road seen next to it (obstacles such as parked cars are OCCUPIED, never an edge); the kerb may not jump in either direction (side roads, cars); kerb evidence must lie within 0.75 m of where last tick's line puts it; lane centre `keep_right_m` left of the kerb, measured at a standstill before the car moves (the kerb fitted as a line from 10 m behind to 15 m ahead with this same detector, accepted once 10 sweeps agree within 0.15 m, else a FAULT with the reason; never a default); a corridor under 5.5 m wide is one lane, driven in its middle; polynomial fit (degree from how far the evidence reaches) with last tick's line as a prior, continued along its tangent past the evidence, or last tick's line carried unchanged when there is too little evidence. Kerb cells count only if hit within 3 s, the road ahead must be observed within 1 s (an empty LiDAR ends the corridor), and the evidence time is the newest kerb hit, so a carried line ages and expires after 5 s |
| Obstacles | `obstacles.py` | This sweep's OBSTACLE points in the car-width tube along the centreline (nearest dense cluster), and how far the tube is *seen clear*: ground observed in 0.5 m slices from near the car outwards |
| Planner | `planner.py` | Cruise speed, curve limit, braking ramp to a stop 2.5 m short of an obstacle (bumper) or at the goal (final-goal trajectory). Obstacle hold in three states: BLOCKED (seen), CLEAR (road seen clear 2 m past where it stood), UNKNOWN (neither: the stop stays). Released only after 1 s of continuous CLEAR |
| Control | `navigation_control.local` | `LocalTrajectoryController` in the odometry frame |
| Driver | `driver.py` | Modes STARTUP (GNSS and compass averaging), CALIBRATING (kerb distance), DRIVING, STOPPING, FINISHED, FAULT. Faults: LiDAR older than 0.15 s, IMU or speedometer older than 0.2 s, GNSS older than 3 s, route position sigma over 4 m, failed calibration, no corridor. A fault brakes along the last trajectory while odometry is trusted (IMU and speed fresh), otherwise holds the last steering: never wheels straight in a bend. From a standstill the car moves only on kerb evidence fresher than 0.5 s; a lost corridor at a standstill is re-acquired from the start-up assumption |

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
