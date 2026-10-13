# Interfaces and conventions (BL-F-01)

Every A2 package talks to the others through the plain-data types in
`src/autonomy/interfaces.py`. Nothing in that module imports a simulator or a
neural-network library, so core code can be tested without CARLA or a GPU.

**Status:** proposed for review at gate G0. All six owners should check the
units, axes, frame IDs, clocks and lifecycle below before building on them.

## Conventions

| Topic | Convention |
|---|---|
| Units | SI only: metres, seconds, radians, m/s. Names carry the unit (`speed_mps`, `dt_s`). |
| World frame | Right-handed. `x` along the map's +X, `y` left, `z` up. Yaw counter-clockwise from +X. Same as A1 `VehicleState`. |
| CARLA → world | CARLA is left-handed (+Y right). The adapter maps `y_world = -y_carla` and `yaw_world = -radians(yaw_carla)`. Nothing outside `carla_adapter` does this. |
| Vehicle frame | Origin at the rear-axle centre on the ground. `x` forward, `y` left, `z` up. |
| Sensor mounts | `Extrinsics`: sensor body pose in the vehicle frame (m, rad). Zero angles look along vehicle +x. |
| Camera optical frame | OpenCV: `x` right, `y` down, `z` along the optical axis. From the body frame: `x_opt = -y_body`, `y_opt = -z_body`, `z_opt = x_body`. |
| Pixels | `u` right, `v` down from the top-left corner. CARLA's pinhole has `fx = fy = W / (2 tan(fov/2))`, `cx = W/2`, `cy = H/2`. |
| Images | `rgb` is `(H, W, 3)` uint8 in **RGB** order. The adapter converts CARLA's BGRA. |
| Depth | `depth_m` is `(H, W)` float32 metres, `NaN` where invalid. Whether CARLA reports optical-axis depth or ray distance is recorded in `CameraModel.depth_kind` and must be confirmed (BL-B-04). |
| Speed sign | `MotionState.speed_mps` is signed (negative in reverse). A1's `VehicleState` stays forward-only; `to_vehicle_state()` converts and refuses real reverse motion. |
| Steering | A1 `ControlCommand`: positive = left, radians, front-wheel equivalent. Throttle and brake in [0, 1], never both positive. |

### Clocks

Three clock domains exist. Values from different domains are never
subtracted from each other.

| Field | Clock | Meaning |
|---|---|---|
| `sim_time_s` | Simulator | Elapsed simulation time of the frame |
| `frame_id` | Simulator | Frame counter; every derived result carries it |
| `host_tick_s` | Host `perf_counter()` | When the stack asked the simulator to advance: the documented capture-origin approximation |
| `host_receive_s` | Host | When the frame bundle was complete |
| `host_start_s`, `host_end_s` | Host | Around a detector call. For a remote detector these are client-side request and response times |
| `host_command_s` | Host | When the control packet was handed to the simulator |

End-to-end latency for the course metric is `host_command_s − host_tick_s`
for the same `frame_id`. Frame age is `host_receive_s − host_tick_s`.
Real-time factor comes from `sim_time_s` against host time over a run.

## Lifecycle

Only the stack advances the world. Per control cycle:

```text
bundle = world.reset(scenario)            # first frame
loop:
    packet = stack.step(bundle)           # perception → planning → control
    world.apply(packet)                   # command for the frame just processed
    bundle = world.tick(timeout_s)        # advance one fixed step, next frame
                                          # SensorTimeout → stack stops the car
world.close()                             # in `finally`: restore settings,
                                          # destroy only this run's actors
```

`SimulatorInterface` is a structural protocol. CARLA (`carla_adapter`), the
kinematic harness (`sim_harness`) and replay all implement the same five
members: `vehicle_geometry`, `reset`, `apply`, `tick`, `evaluation`, `close`.

## Contracts

| Type | Produced by | Consumed by | Notes |
|---|---|---|---|
| `FrameBundle` | Simulator adapter | Stack, perception | RGB, depth, optional LiDAR, camera model and ego state for **one** frame. `valid` is false if any listed sensor failed |
| `CameraModel`, `Extrinsics` | Adapter (from the sensor rig config) | Geometry, depth association | Intrinsics, mount pose, depth convention |
| `MotionState` | Adapter | Planning, control, parking | Rear-axle pose, signed speed, yaw rate |
| `VehicleGeometry` | Adapter (measured, BL-E-01) | Corridor, gap, control | Wheelbase, bumper offsets, width, physical steering limit |
| `SensorStatus` | Adapter | Stack | Per-sensor validity with a reason |
| `ControlPacket` | Stack | Adapter | A1 `ControlCommand` plus reason, gear, drive mode and emergency flag. `ControlPacket.stop()` is the fault stop |
| `LaneSegment`, `LaneGraph` | Topology adapter (BL-A-02) | Route manager, Wayfinder | Lane centrelines and legal successors, serialised deterministically |
| `Mission` | Config (`configs/scenarios/`) | Route manager | Start pose, junction choices, destination |
| `Diagnostics` | Stack | Logger, HUD | Behaviour state, speed caps, counters, gap, CTE, timings, fault |
| `EvaluationPacket` | Adapter | **Evaluation only** | Ground-truth actors, collisions, lane invasions, scenario events |

### Evaluation boundary

Obstacle actor poses, boxes and traffic-light states come only through
`EvaluationPacket` and feed only `evaluation/`. Perception, planning and
control see the world through `FrameBundle` alone. The map topology and the
ego pose are the declared exceptions, pending tutor confirmation (PROC-03).

### Failure semantics

- A sensor that misses its frame produces `SensorStatus(valid=False, reason=...)`,
  or the tick raises `SensorTimeout`. Either way the stack stops the car with
  `ControlPacket.stop()`. It never repeats the last moving command.
- A `FrameBundle` whose sensors disagree on `frame_id` cannot be constructed.

## Worked example: one CARLA frame

A camera mounted 1.5 m ahead of and 2.4 m above the rear axle, 800×600, 90°
horizontal field of view:

```python
from math import radians
from autonomy.interfaces import CameraModel, Extrinsics

cam = CameraModel.from_fov(800, 600, radians(90), Extrinsics(1.5, 0.0, 2.4))
# cam.fx_px == cam.fy_px == 400.0, cam.cx_px == 400.0, cam.cy_px == 300.0
```

If CARLA reports the rear axle at `(x=10.0, y=5.0)` with yaw 30°, the
`MotionState` in the world frame is `x_m=10.0, y_m=-5.0, yaw_rad=-0.5236`.

## Extension hooks

These are already present so the extensions plug in without changing
signatures:
- **E2:** `FrameBundle.lidar`.
- **E4:** detector source and timing fields.
- **E5:** `Gear`, `DriveMode` and signed `MotionState`.
- **E3:** `LaneGraph` and `Mission`.

Every extension flag defaults to off.
