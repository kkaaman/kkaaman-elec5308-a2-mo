# Sensor rig, segmentation trade-off and lifecycle — RV-08

Owner: Mohammad. Status: documentation of the code on main as of this task;
no new live measurement is claimed here. Numbers quoted from evidence folders
carry their source. Companion pages: [RGB-D association](depth-association.md),
[depth validation evidence](evidence/TEST-13/README.md),
[world lifecycle](world_lifecycle.md), [lane departure monitor](lane_departure_monitor.md)
and [loop timing](loop_timing.md).

## Rig as implemented

| Item | Value | Where |
|---|---|---|
| Cameras | `sensor.camera.rgb` + `sensor.camera.depth`, one rigid mount | `carla_adapter.sensors.RGBDCapture` |
| Resolution / FOV | 640 x 360 px, 90 deg horizontal (`CameraRig` defaults) | `carla_adapter.sensors.CameraRig`; frozen copy in `configs/mission_baseline.json` `camera_rig` |
| Mount | 1.5 m forward, 2.0 m up from the **actor origin**; no rotation; `lens_k = lens_kcube = 0` | `CameraRig.x_actor_m`, `z_actor_m` |
| Rear-axle offset | Actor origin is 1.38641 m ahead of the rear axle (Tesla Model 3, BL-E-01) | `evaluation.a1_comparison.REAR_AXLE_TO_ACTOR_ORIGIN_M`; applied through `camera_to_ego` |
| Capture rate | `sensor_tick = 0`: one frame per synchronous tick, `fixed_delta_seconds = 0.05` s (20 Hz) | `RGBDCapture.__enter__`, mission `runtime` |
| S0 observation camera | RGB only, 320 x 180 px, 90 deg, same mount, `enable_postprocess_effects = false` | `carla_adapter.s0_runner.RunSensors` |
| Collision / lane invasion | `sensor.other.collision`, `sensor.other.lane_invasion` at the actor origin; evaluation stream only | `RunSensors` |
| LiDAR | Not implemented. The `E2-02` placeholder in `carla_adapter/__init__.py` remains open | — |

Intrinsics are derived from the horizontal FOV:
`fx = fy = width / (2 tan(fov/2))`, `cx = width/2`, `cy = height/2`, all in
pixels at the capture resolution (`geometry.camera.CameraIntrinsics.from_horizontal_fov`).
Any resize of the RGB frame must scale the intrinsics and keep depth aligned
(`docs/interfaces.md`).

## Public sensor API (units and validity)

`src/carla_adapter/sensors.py` is part of the hash-audited operational runtime
(`docs/evidence/BL-D-06/operational-range-review.json`,
`scripts/review_baseline_freeze.py`), so its docstrings are documented here
rather than edited in place until the next freeze re-audit.

| Symbol | Units / validity |
|---|---|
| `CameraRig(width, height, fov_deg, x_actor_m, z_actor_m)` | positive integer pixels; horizontal FOV strictly in (0, 180) degrees; metres from the vehicle actor origin (not the rear axle); invalid values raise `ValueError` |
| `AlignedCapture` | exact-frame RGB (`uint8 HxWx3`), `decoded_depth_m` (float32 m, NaN invalid, convention not yet established), `received_monotonic_s` = later camera arrival on `time.monotonic()` |
| `AlignedCapture.to_bundle(calibration, established_depth_convention=, capture_started_monotonic_s=)` | convention must be `"axial"` or `"radial"` from `scripts/calibrate_depth.py`, never inferred; `capture_started_monotonic_s` is the host clock before the tick request and must not exceed receipt |
| `FrameJoiner(episode_id, capacity=8)` / `.put(kind, episode, frame, timestamp, value)` | `frame` nonnegative integer CARLA frame; `timestamp` nonnegative simulation seconds; retired/late/duplicate/wrong-episode observations are counted in `stats()` and dropped; arrays stored read-only |
| `FrameJoiner.wait(frame, timeout_s, stop)` | `timeout_s` positive wall seconds; RGB/depth/ego timestamps within 1e-6 s and matching dimensions, else the frame is retired, `stop()` runs and `ValueError`/`TimeoutError` is raised; a returned capture is never reused |
| `RGBDCapture(owner, vehicle, episode_id, rig=None, capacity=8)` | context manager; sensors exist only inside an active `SynchronousWorldSession`; decode errors retained in `callback_errors` (last eight) |
| `RGBDCapture.record_snapshot(frame, ego)` / `.read(frame, timeout_s=5.0)` / `.capture(frame, ego, timeout_s=5.0)` | `ego` is an `EgoPose` (rear axle, right-handed m / rad); a snapshot on another frame triggers the owner's emergency stop |
| `RGBDCapture.close()` | closes the joiner first, then stops and destroys owned sensors; errors collected and raised once |

## Frame matching

`carla_adapter.sensors.FrameJoiner` holds at most `capacity = 8` frames per
stream (`rgb`, `depth`, `ego`) and joins on the **exact CARLA frame number**.
The rules that make a join valid:

- RGB, depth and ego timestamps agree within 1e-6 simulation seconds; otherwise
  the frame is retired and `ValueError` is raised (`timestamp_mismatch` counter).
- RGB is `uint8 HxWx3` and depth is `HxW` at the same resolution.
- The ego pose comes from `world.get_snapshot()` **on the same frame**
  (`RGBDCapture.record_snapshot`); a snapshot on a different frame triggers the
  owner's emergency stop.
- A frame is retired once returned or once a newer frame is returned; late,
  duplicate, wrong-episode and overflowed observations are counted in
  `FrameJoiner.stats()` and never reused.
- `wait()` times out after `timeout_s` wall seconds (default 5 s in
  `RGBDCapture.read`); the caller's `stop` runs outside the callback lock.

Decoding is done in the callback (`carla_adapter.conversions`): RGB is BGRA
channels `[2, 1, 0]`; depth is `1000 * (R + 256 G + 65536 B) / 16777215`
metres with code 0 and the saturated 1000 m code set to NaN. The core only ever
sees `FrameBundle.depth_m` as **axial optical Z** in metres; the axial/radial
decision is passed explicitly to `AlignedCapture.to_bundle(established_depth_convention=...)`.

The S0 runner uses a lighter path on purpose: `RGBReceiptQueue` keeps camera
receipts (frame, simulation timestamp, host receipt time) without image data and
`read_latest(frame - 4, frame, timeout)` admits up to four frames of GPU
pipelining, joining the receipt to the exact ego snapshot stored for that frame.
Receipt time is taken **before** any buffer access; capture-to-apply latency in
`samples.csv` is `apply_monotonic_s - capture_receipt_monotonic_s`.

## Calibration

| Calibration | Script | Current record |
|---|---|---|
| Depth convention (axial vs radial) | `scripts/calibrate_depth.py` (off-axis ray casts; `choose_convention` needs >= 20 valid samples, best median error <= 0.3 m and the other >= max(0.6 m, 3x)) | `axial`, 107 samples, median abs error ~0.050 m vs 1.886 m radial ([TEST-13](evidence/TEST-13/README.md)) |
| Depth range accuracy | `scripts/analyze_depth.py` on `carla_adapter.dataset_capture` output | Car/person bins through 25 m median abs error < 0.5 m ([TEST-13](evidence/TEST-13/README.md)); this is not a visibility certificate |
| Vehicle geometry / steering | `scripts/calibrate_vehicle.py` | [steering calibration](steering_calibration.md) |
| Longitudinal response | `scripts/calibrate_longitudinal.py` | `docs/evidence/BL-E-03` |

Calibration outputs are consumed by hash: `scripts/measure_operational_governance.py`
refuses to run unless `depth-calibration.json` matches the installed build, map
OpenDRIVE hash and rig, and `autonomy.baseline_operation.baseline_operational_stack`
rejects a capture whose dimensions or focal length differ from the frozen rig.
Re-run the depth convention calibration after any CARLA upgrade, map change or
rig edit before trusting depth-based clearance.

## Weather limits

- The S0 runner (`FrozenEnvironment`) forces `WeatherParameters.ClearNoon` and
  frozen green lights for every acceptance run, then restores and verifies the
  original weather. Acceptance evidence therefore exists for **clear noon only**.
- `carla_adapter.dataset_capture` cycles `ClearNoon`, `CloudyNoon`, `WetNoon` for
  the TEST-13 depth dataset; depth accuracy was reviewed on that mix, detector
  recall was not.
- `configs/weather/` is empty; the E1 weather presets and
  `scripts/evaluate_weather.py` (E1-03) do not exist yet. Rain, fog, night and
  wet-road detector/lane behaviour is **unmeasured** and outside the admitted
  envelope. The CARLA depth camera is ideal (noise-free), so weather affects RGB
  perception and lane masks, not the depth values themselves.

## Segmentation trade-off and cadence

Two segmenters exist:

| Model | Module | Input | Output | Role |
|---|---|---|---|---|
| LaneUNet (ResNet18 encoder) | `perception.lane_inference.LaneSegmenter` (`lane_unet_best.pth`) | nonempty `HxWx3` uint8 **RGB**, resized to 360 x 640 | `HxW` uint8 lane mask (1 = marking) at input resolution | Lane boundary / centreline for the RV-06 departure monitor and HUD only |
| BiSeNetV2, Cityscapes 19 | `segmentation.SemanticSegmenter` (`segmentation_model.pth`) | nonempty `HxWx3` uint8 RGB, resized to 512 x 256 | `HxW` uint8 trainId mask (0-18) at input resolution | Display overlay; optional depth masking stays `KEEP_OFF` (`evaluation.masking`) |

Neither mask enters the controller. The baseline obstacle path is YOLO boxes
plus axial depth; `evaluation.masking.masking_decision` turns masking on only if
at least 30 detector boxes show a >= 0.2 m median error gain or >= 10 % fewer
invalid boxes **and** the warm mask p95 fits a 30 ms budget. That evidence does
not exist yet, so the decision remains `KEEP_OFF`.

Cadence rule (TEST-17): optional workloads run through
`perception.cadence.RateLimitedWorkload(period_ticks, max_age_ticks)`. A result
is returned as current only while its age is `<= max_age_ticks`; older results
are handed back as `stale_value` with `valid = False` and must be labelled
stale on the HUD, never applied as if current. The A1 demo's
`SEGMENTATION_SKIP_INTERVAL = 3` with an indefinitely cached mask is the
behaviour this replaces; the demos themselves are untouched legacy code behind
the simulator boundary. Recommended starting cadence for the 20 Hz loop:
segmentation every 3 ticks (`max_age_ticks = 2`), lane monitor every tick when
the lane model is warm, LiDAR not applicable. Measured costs per stage are
reported by `scripts/profile_loop_timing.py` ([loop timing](loop_timing.md)).

## Lifecycle

1. One `SynchronousWorldSession` owns ticks; sensors are created only inside an
   active session (`RGBDCapture.__enter__` calls `owner._require_owner()`).
2. Sensors register with the owner's actor list **before** `listen()` so a
   failed spawn is still destroyed on cleanup.
3. Per tick: `owner.tick()` -> `record_snapshot(frame, ego)` -> `read(frame)` ->
   `to_bundle(...)` -> `stack.step(bundle)`. `capture_started_monotonic_s` is the
   host clock before the tick request; the freshness gate rejects frames older
   than `max_observation_age_s = 0.5` s or with a decision older than
   `command_ttl_s = 0.1` s.
4. `close()` closes the joiner first (so callbacks stop being admitted), then
   stops and destroys owned sensors in reverse order; errors are collected and
   raised once. The S0 `RunSensors.close()` adds a bounded 100 ms callback
   quiescence fence before the event snapshot is frozen.
5. Episode end restores settings, weather and traffic lights and verifies the
   actor inventory ([world lifecycle](world_lifecycle.md)).

## Validity rules summary

| Quantity | Unit | Valid when |
|---|---|---|
| RGB frame | uint8, u right, v down | nonempty `HxWx3`, same frame id as depth and ego |
| Depth | float32 m, axial optical Z | finite positive; NaN invalid; convention established by calibration |
| Receipt / apply time | s, `time.monotonic()` | same host; capture before receipt before apply |
| Simulation time | s within one episode | strictly increasing per frame; never mixed with host time |
| Lane mask | uint8 per pixel | computed from this frame; otherwise stale |
| Semantic mask | uint8 trainId 0-18 | age `<= max_age_ticks`; otherwise stale, display-only |
| Lane estimate | m, rad, positive left | see [lane departure monitor](lane_departure_monitor.md) |

## Open items

- LiDAR (`E2-02`) and weather presets (`E1-01`) are not implemented; the plan
  rows for them are blocked, not failed.
- The live S3/S4/S5 scenario actors and their evaluation-truth gap stream are
  not in `carla_adapter` yet (`scenario.py`, `ground_truth.py` placeholders).
  `evaluation.scenario_metrics` defines the fields it will consume.
