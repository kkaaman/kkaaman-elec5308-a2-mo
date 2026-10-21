# Localisation from the car's own sensors (GNSS + IMU + wheel odometry)

A2 drives on the simulator's vehicle pose. Shadow localisation measures how well the
car could know its pose from its own sensors before any driving depends on it:
- the estimate is computed every frame and logged beside the true pose;
- the controller still uses the true pose;
- `tests/integration/test_shadow_localisation.py` checks that a run with the estimate
  produces exactly the same commands and trajectory as one without.

## What it uses

**Sensors:**
- **GNSS:** CARLA's `sensor.other.gnss`, mounted over the rear axle. Its noise model is
  white noise plus a constant offset, drawn once per run from the seed and converted
  from metres to CARLA's degrees.
- **Gyro:** CARLA's `sensor.other.imu`, z rate, with white noise and a constant bias,
  drawn per run. The core frame flips CARLA's y, so the core yaw rate is minus the
  gyro's z.
- **Wheel speed:** CARLA has no wheel encoder, so this is the true longitudinal speed
  with a seeded constant scale error and white noise. It's the same model the harness
  uses.

**Estimator** (`autonomy/localisation.py`): an EKF over x, y, yaw and gyro bias.
- It is initialised at the first GNSS fix, snapped to the nearest route point and given
  that point's heading (map matching while stationary, where GNSS gives no heading).
- Wheel speed and gyro predict; GNSS fixes correct x and y.
- A constant GNSS offset is not observable from these sensors, so it remains as
  position error. Removing it needs a lane-relative measurement, such as the camera
  lane estimate (RV-06). Every run logs its raw inputs so such an estimator can be
  replayed offline on the same runs.

**Profiles** (`configs/localisation_*.json`):

| Profile | GNSS white | GNSS constant offset (1σ per axis) | Rate |
|---|---|---|---|
| `consumer_gnss` | 0.5 m | 1.0 m | 5 Hz |
| `rtk_gnss` | 0.02 m | none | 5 Hz |

Both use the same gyro (0.005 rad/s white, 0.002 rad/s bias) and wheel model (1% scale,
0.05 m/s).

## Running it

Any stage that drives (Quick, Navigation, Smoke, Acceptance) takes `-Localisation`;
`run_scenario.py` takes `--localisation`.

```powershell
$state = "$HOME\Downloads\Group7_a1route_state.json"
.\scripts\run_batch35.ps1 -Stage Navigation -StateFile $state -Localisation configs\localisation_consumer_gnss.json
$s = Get-Content $state -Raw | ConvertFrom-Json
.venv\Scripts\python.exe scripts\localisation_report.py (Split-Path $s.navigation)
```

**Outputs:**
- `samples.jsonl` gets a `localisation` field per frame: the estimate, total, lateral,
  along-route and yaw errors, and the raw inputs.
- `summary.json` → `localisation` holds the error statistics while moving, and the
  gyro-to-true-rate correlation as a sign/scale check (it should be close to +1).
- `manifest.json` → `localisation` records the config and the per-run noise draws.

These statistics are report-only and are not acceptance criteria.

## Live result (CARLA, A1 route, S0 navigation, seeds 5308-5312, 21 Oct)

Shadow mode: the car drove on the true pose. All ten runs were
`MEASURED_PASS_REVIEW_REQUIRED`.

| Profile | Lateral error, mean | Lateral p95 | Lateral max | Along-route p95 | Yaw p95 |
|---|---|---|---|---|---|
| consumer | 0.40-1.36 m | 0.77-1.84 m | 1.95 m | 1.2-2.3 m | 1.2-2.3° |
| RTK | 0.009-0.011 m | 0.021-0.029 m | 0.050 m | 0.04-0.07 m | 0.4-0.5° |

**Sensor checks:**
- The gyro-to-true-rate correlation was +0.995 in every run, so the yaw-rate sign is
  right.
- RTK's 1 cm error confirms the rear-axle antenna mount and the latitude/longitude
  conversion. An error in either would show as a constant offset.
- About 258 fixes per run, with 2 late at the start of each run.

**Reading:** with consumer GNSS, the estimate is off sideways by about as much as A1
drove off its lane. With RTK it would support driving at A2's accuracy.

## Driving on the estimate

`--drive-on-estimate` (`-DriveOnEstimate` on the stages), together with `--localisation`:
- The policy receives the estimated pose and the measured wheel speed instead of the
  simulator's.
- The estimator starts at the mission start, where the car is parked, and GNSS takes
  over from there.
- Scoring (CTE, goal distance, stop margins, collisions) still uses the true pose, so
  the metrics show how the car really drove.
- `manifest.json` records `pose_source: estimate`.

**Live CARLA, A1 route, RTK profile (21 Oct):**

| Run | True CTE mean (rear axle, route) | Route completed | Stop margin |
|---|---|---|---|
| S0 navigation, seeds 5308-5312 | 0.017-0.024 m (0.018 m on the simulator pose) | yes, all five | n/a |
| S1 full stack, seed 5308 | 0.018 m | yes | 4.774 m, no collision, every S1 threshold passes |

On A1's CTE definition, the S0 runs give 0.023-0.031 m, against 0.025 m on the simulator
pose and 0.918 m for A1.

The measured wheel-speed scale error shows up seed by seed in the true speed (within
0.42%). So the car really was steering and holding speed on its own sensors.

Details, the figure and the limits of the result are in
`docs/evidence/TEST-03-a1-route-estimate/`.

**Harness, A1 route, seeds 5308/5309:**

| Profile | True CTE mean | Route completed | S1 stop margin |
|---|---|---|---|
| RTK | 0.011-0.015 m | yes | 4.47-4.50 m, no collision |
| consumer | 0.34-1.32 m | no: stops more than 1 m from the goal because of along-route error | not run |

The consumer case is an honest failure, not a bug: the car drives where its biased GNSS
says the lane is. The next step is a lane-relative correction from the camera lane
estimate.

## Harness result (A1 route, S0, seeds 5308-5312)

| Profile | Lateral error, mean | Lateral p95 | Along-route p95 | Yaw p95 |
|---|---|---|---|---|
| consumer | 0.33-1.37 m (follows each run's GNSS offset) | 0.56-1.97 m | 1.4-2.3 m | 1.0-1.7° |
| RTK | 0.008 m | 0.017-0.021 m | 0.04-0.07 m | 0.2-0.3° |

With consumer GNSS alone, the estimate is about as far off sideways as A1 drove (A1:
0.92 m mean on its own CTE definition). Pose-free driving at A2's current lateral
accuracy needs either RTK-grade positioning or a lane-relative correction. These are
harness numbers on flat ground; the live CARLA runs give the real ones.

## Camera-lane correction for consumer GNSS (offline first)

**Problem.** Consumer GNSS has a constant offset of about 1 m, which GNSS, gyro and
wheel speed cannot see. `autonomy/lane_localisation.py` adds the RV-06 lane monitor as
a measurement.

**Measurement.** LaneUNet's lane centre, projected to the ground, gives the signed
offset of the point 8 m ahead of the rear axle. The filter predicts the same point's
offset from the mission route, which follows lane centrelines. One measurement
therefore corrects lateral position and heading together. Lane measurements more
than 3σ from the prediction are rejected (a lane-marking confusion, for example).

**Extra states.** The EKF adds three states to x, y, yaw and gyro bias:
- **GNSS offset (x, y), world frame.**
  - On a straight road, the lane sees only the component across the road.
  - After a bend, the other component becomes visible as well. Once the offset is
    known, along-route error goes too, which matters for stopping at the goal.
- **Wheel-speed scale error.** The GNSS-only filter treats this as noise. With consumer
  GNSS, a 0.6% scale error then leaves the estimate about 0.2 m behind the car.

**Offline replay.** `scripts/replay_lane_localisation.py` runs on recorded runs:
1. It runs LaneUNet on each run's `video.avi`.
2. It replays three estimators on the run's logged inputs, using the lane offsets from
   step 1:
   - GNSS only, as run live. It must reproduce the logged estimate exactly, which
     checks the replay.
   - GNSS + lane.
   - GNSS + lane, with the offset and wheel scale estimated.
3. It scores each estimator against the true pose.

The car drove on whatever its run used; only the estimates change.

```powershell
$s = Get-Content "$HOME\Downloads\Group7_a1route_state.json" -Raw | ConvertFrom-Json
.venv\Scripts\python.exe scripts\replay_lane_localisation.py <consumer Navigation folder> --mission $s.mission --output "$HOME\Downloads\Group7_lane_localisation_consumer"
```

Filter assumptions are in `configs/lane_localisation_consumer.json`. The sensors' true
noise is the run's own localisation profile.

**Harness result.** A1 route, consumer GNSS, seeds 5308-5312. The lane offsets were
synthetic: the true 8 m offset plus 8 cm white noise, with 2% outliers of ±3 m.

| Estimator | Lateral mean | Lateral p95 | Along-route, last 20 m (mean) | Yaw p95 |
|---|---|---|---|---|
| GNSS only (as run live) | 0.33-1.37 m | 0.56-1.97 m | 0.55-2.31 m | 1.0-1.7° |
| GNSS + lane | 0.02-0.08 m | 0.10-0.50 m | 0.11-1.83 m | 1.1-6.2° |
| GNSS + lane, offset and scale estimated | **0.016-0.027 m** | **0.04-0.13 m** | **0.05-0.19 m** | 0.6-2.0° |

- The GNSS offset was recovered to within 6 cm in every run.
- Lane without the offset states fixes lateral error but not along-route error. GNSS
  and lane then disagree, which shows up as heading error.
- The largest errors are at the start, before the opening bend makes the offset
  observable: up to 0.34 m lateral and about 1.5 m along-route.

**Before anything drives on it:**
1. **Live replay.** The real test is the replay on the live consumer runs. LaneUNet's
   errors are correlated from frame to frame and may be biased on curves, unlike the
   white noise above.
2. **Team decision.** Driving on this estimate puts the lane monitor in the control
   path. RV-06 currently defines it as diagnostic only.
