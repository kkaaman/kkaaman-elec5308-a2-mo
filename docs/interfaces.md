# Shared contracts — BL-F-01

Version: 1. Status: proposed implementation; owner reviews pending.

The core accepts ordinary Python dataclasses and NumPy arrays. CARLA imports
and simulator objects stay in `carla_adapter`. `navigation_control` remains the
owner of `VehicleState` and `ControlCommand`; `autonomy.interfaces` re-exports
these types rather than creating competing controller commands.

## Units and coordinate frames

| Quantity | Contract |
|---|---|
| World coordinates | Right-handed metres; adapter reflects CARLA Y and negates yaw |
| Ego coordinates | +X forward, +Y left, +Z up |
| Ego pose reference | Rear axle, not actor origin or camera origin |
| Yaw and steering | Radians; positive yaw counterclockwise, positive steering left |
| Speed | Nonnegative m/s; forward-only baseline |
| Pedals | Dimensionless [0,1]; throttle and brake mutually exclusive |
| Image | Nonempty HxWx3 uint8 RGB; pixel u right, v down |
| Optical coordinates | X right, Y down, Z forward |
| Intrinsics | fx, fy, cx, cy in pixels at the bundle image resolution |
| Camera extrinsics | Rigid 4x4 camera-to-ego transform; translation in metres |
| Depth | HxW float32/float64 axial optical Z in metres, aligned to RGB |
| Invalid depth | NaN; never reinterpret as clearance or zero range |

CARLA's encoded camera depth must be decoded in the adapter. Whether the
installed CARLA build reports axial optical depth or radial range must be
established by an off-axis depth-convention validation before any conversion
is applied. The core contract always receives axial optical Z depth in metres.
Raw BGRA bytes and encoded 24-bit depth must not cross into the core. Any RGB resize requires corresponding
intrinsics scaling and depth alignment. Ground projection belongs to workstream D.

## Packets

- `FrameBundle`: episode identity, RGB frame ID, simulation capture time,
  host receipt time, rear-axle ego pose, image, calibration, optional depth.
  Depth requires an explicitly matching frame ID and dimensions. A missing
  depth sensor is `None`; a dropped required depth frame must be handled as an
  invalid observation by the stack, not treated as an obstacle-free scene.
- `LaneGraph`: directed edges with sampled XY polylines and speed limits.
  Nodes have unique IDs and XYZ positions. Every edge endpoint must exist and
  match its polyline endpoint. Legal transitions are represented by edges;
  legality itself must be checked by the topology exporter and route planner.
  Elevation is recorded on nodes; the current controller remains planar.
- `Diagnostics`: per-stage wall durations, observation validity and explicit
  failure reasons. Values are measurements, never substitute acceptance results.
- `ControlPacket`: existing physical steering/pedal command, source frame and
  episode, source simulation time and simulation expiry. An expiry equal to
  the current simulation time is still valid. `is_current` checks episode/time;
  the adapter additionally rejects decreasing source frame IDs.

Arrays are copied into read-only snapshots at construction so subsequent sensor
buffer updates cannot mutate a queued observation. This adds allocation/copy cost;
measure it in loop profiling. NumPy write flags prevent normal accidental writes,
not adversarial mutation. Calibrations and graphs normalize nested sequences to tuples.

## Time, ordering and lifecycle

Simulation timestamps are seconds within one episode and reset with the episode.
Host receipt timestamps and stage durations use `time.perf_counter()` seconds on
one process clock. Never subtract simulation time from host time, or compare
remote host monotonic clocks. RGB, depth and ego must correspond to the same
simulator tick; frame IDs alone cannot prove ego alignment.

One world owner advances simulation. `World.reset(seed=...)` returns the first
bundle and graph; `AutonomyStack.reset(episode_id=..., graph=...)` clears filters,
tracker state, PID history and route progress. `step` consumes exactly one frame
and returns a packet tied to it. Reject wrong-episode, expired, future-source and
out-of-order packets. On missing data, timeout or exception, stop using old
commands: apply the configured controlled stop. Safety policy and braking values
are implemented and calibrated in later tasks; these contracts alone do not
provide automatic stopping.

`close()` must be idempotent: stop sensor callbacks, destroy only owned actors,
restore prior world/Traffic Manager settings and clear queues. A later concrete
world implementation must test these behaviours, including exception paths.
Protocol runtime checks verify method presence only, not this semantic behaviour.

## Sensor inputs versus evaluation truth

Allowed navigation inputs: map topology and ego pose, subject to tutor scope
confirmation. Perceived obstacle inputs must come from sensor processing.
Actor IDs, ground-truth obstacle positions and collision-event records belong in
an independent evaluation stream and must not be fields in `FrameBundle` or
controller inputs. A later evaluation packet can be defined by workstream B/F.

Detector/obstacle types and track identity are owned by BL-C-01. This change does
not pre-empt that contract. Frame metadata remains attached through perception,
including remote inference, so stale observations can be rejected.

## Review checklist

All six owners must review units, axes, clocks, frame identity and lifecycle.
No approval is recorded by this implementation.

| Owner | Review focus | Status |
|---|---|---|
| Saviour | Stack lifecycle, packet expiry, harness | Pending |
| Ricky | Detection metadata and simulator boundary | Pending |
| Mohammad | RGB-D tick alignment and invalid observations | Pending |
| Alan | Optical transforms and depth convention | Reviewed - 14 Oct 2026 |
| Zhaolin | Graph transitions and rear-axle navigation frame | Pending |
| Abdullah | Vehicle conversion and physical steering convention | Pending |

Reviewer from task board: Ricky. Coordinate contracts before dependent BL-F-02,
BL-B-01 and BL-D-02 implementations are merged.
