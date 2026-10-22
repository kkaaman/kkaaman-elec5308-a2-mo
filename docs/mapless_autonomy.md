# Mapless autonomy: Phase 0 (contract, sensor rig, sat-nav route, capture)

Branch `feature/mapless-autonomy`, built from `feature/shadow-localisation`. This is an
extension: the frozen baseline is unchanged.

## Goal and honest label

The car drives with **no simulator truth and no map**. It sees only its sensors, and it
is given a sat-nav route: sparse GPS points, each with a command. The rules follow the
CARLA Leaderboard 2.0 **SENSORS track**, and scoring will use that track's driving score.

This is not SAE Level 5. Level 5 means no limit on where or when the car can drive, and
a handful of simulator towns cannot show that. The defensible claim is a Level 4 system
with a broad operating domain, measured on Town10HD and on a town it never trained on.

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

Run with `scripts/capture_mapless_dataset.py`. Training-data capture on **Town01–Town07
only**; Town10HD is refused.

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

## Running Phase 0 live

Run with CARLA 0.9.16 and no other clients:

```powershell
# 1. Rig check on the demo map (calibration only; no training data is kept)
python scripts/check_sensor_rig.py --output "$HOME\Downloads\Group7_mapless_rigcheck"

# 2. Sat-nav route for the baseline mission (the same saved mission used for live runs)
python scripts/make_satnav_route.py --mission <saved mission.json> --name baseline `
    --output configs/satnav/baseline.json

# 3. Training data, one run per town
python scripts/capture_mapless_dataset.py --map Town03 --samples 2000 `
    --output "$HOME\Downloads\Group7_mapless_capture_Town03_s5308"
```

**Phase 0 gate:**
- `rig_check.json` status is `MEASURED_PASS`. If `radar_velocity_sign` fails, flip
  `RADAR_VELOCITY_IS_RANGE_RATE` and re-run.
- The firewall tests pass.

## Next (Phase 1)

- Road and lane-marking segmentation trained on the captures.
- LiDAR-painted top-down road map.
- LiDAR ranging inside detector boxes, replacing depth.
- Object tracker with velocities.
- Traffic-light state.
- Heading initialised from the compass; the EKF fed GNSS, IMU and the speedometer.
