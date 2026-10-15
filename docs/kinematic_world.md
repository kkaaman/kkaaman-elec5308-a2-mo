# BL-F-03: fixed-step kinematic world and synthetic observations

`sim_harness.KinematicWorld` implements the existing `autonomy.interfaces.World`
protocol without CARLA, model weights or downloads. It provides a planar vehicle
and explicit sensor fixtures for controller and contract tests. `ContractWorld`
remains the stationary BL-F-02 fixture. The production stack is unchanged.

Owner: Saviour. Reviewer from the task board: Ricky; review is pending.

## Dynamics and configuration

`KinematicConfig` validates finite positive `dt_s`, `wheelbase_m`, physical steering
limit, full-throttle acceleration and full-brake deceleration. The steering limit
must be below pi/2. Defaults are 0.05 s, 2.8 m, 0.6 rad, 2 m/s² and 4 m/s². These
are illustrative harness values, not measured Tesla/CARLA parameters. Supply
the controller geometry separately and keep its steering cap at/below the world
limit; it does not replace the world's physical limit.

The rear axle is the pose reference, with world/ego +Y left, counterclockwise yaw
and left steering positive. Speed is nonnegative and distances use metres.
Each valid command is held for one fixed step:

- Acceleration is throttle times configured acceleration minus brake times
  configured deceleration. The existing `ControlCommand` rejects simultaneous
  throttle and brake.
- Steering is clipped to the world's limit. Curvature is `tan(steering)/wheelbase`.
- Travel distance integrates constant acceleration, ending early if braking
  reaches zero speed within the step. The remainder of the step is stationary.
- Position follows the exact constant-curvature arc for that distance, with a
  stable straight/small-angle limit. Yaw changes by curvature times distance.

There is no tire slip, suspension, drag, grade, collision response, actuator lag,
steering rate limit or measured speed-dependent steering response. Coasting has
constant speed. This model verifies algorithm behaviour, not physical stopping
guarantees, safe clearance or simulator performance.

## Lifecycle, clocks and command rejection

`reset(seed=...)` restores the configured initial `VehicleState`, restarts the
observation script at frame zero and returns `(FrameBundle, LaneGraph)`. Supply
a graph explicitly when needed; the default graph is empty. The world neither
selects routes nor asserts that a synthetic route is road-legal.

Each reset creates a fresh episode ID even for the same seed, preventing old
commands and track identities from being reused. There is currently no stochastic
model: seeds do not introduce noise. Identical initial conditions, scripts and
commands reproduce the same states, RGB/depth values and relative observations.
Episode IDs, receipt timestamps and actual stack processing durations are excluded
from deterministic comparisons.

Frame IDs increase once per accepted step; simulation time is `frame_id * dt_s`.
Receipt time uses an injectable process-local monotonic clock, compatible with
`SkeletonStack`'s freshness checks. RGB, depth and ego state share the same tick.

`step` checks packet type, command validity, episode, exact source frame and source
simulation time, and inclusive expiry at dispatch. Invalid commands select full
brake and raise without consuming a tick. Sensor snapshot construction failures
also leave the last committed state/time intact and select full brake. There is
no background ticking: selecting brake does not instantly set speed to zero. To
advance braking after a rejected command, handle the error and submit a valid
brake packet for the unchanged current observation. A stack emergency packet is
accepted normally, including its invalid-observation diagnostics.

Ordinary stopping and resuming uses brake and throttle commands in the same
episode, preserving controller progress. Resetting the world begins a new episode;
reset the stack, detector and obstacle source to its returned episode ID as well.
`close()` selects full brake and is idempotent; closed components cannot restart.
Create new instances for a new session after closing.

## Synthetic sensor path and visibility limits

`SyntheticScenario` is an immutable script shared by the world and
`SyntheticDetector`. Each `SyntheticFrame` specifies a frame ID, zero or more
`SyntheticDetection` entries, depth-sensor availability and detector validity.
Each entry contains the existing `DetectionObservation` (box, class, confidence,
optional track ID) and a positive optical-Z distance, or `None` for missing depth.

The world emits coloured RGB rectangles and constant axial-depth patches beneath
the corresponding boxes. Unknown pixels are NaN. Missing depth sensors produce
`depth_m=None`; invalid detection frames need an explicit reason. Overlapping
raster ROIs, duplicate frames/track IDs and boxes outside the configured image
are rejected rather than silently inventing occlusion behaviour. Unspecified
frames explicitly mean a valid empty detection frame, with all-NaN depth.
Script every relevant tick when persistent obstacles are required.

`SyntheticDetector.detect/track(frame)` injects the script's observations with the
input episode/frame/time and validates image dimensions/calibration. It does not
run inference on the rectangles. Detector timing is explicitly a zero-duration
fixture value, not an inference benchmark. Track IDs use the existing
episode/source/model namespace, and no IDs are fabricated when they are absent.

`SyntheticObstacleSource.observe(frame, batch)` requires a valid same-frame batch
and aligned depth. It takes the median valid values within each controlled ROI,
then unprojects the box centre using the existing geometry utilities. Missing
sensor/depth or invalid observations raise instead of implying clear space, even
when a missing sensor accompanies an empty detection batch. A valid empty batch
with an available depth sensor returns an empty tuple; it does not prove clearance
outside the scripted field of view.

The distance represents the box-centre plane in optical-Z metres, not radial
range or a measured nearest obstacle surface. The camera is at rear-axle X/Y and
the configured height, facing ego +X; optical right/down map to ego -Y/-Z. The
source uses an explicit fixed full-XYZ extent prior (default 1 m per axis), zero
yaw prior and the detection confidence/track ID. Extent and orientation are not
measured ground truth or accurate object geometry.

There is no scene-object truth in this harness. Visibility, disappearance, faults
and distances are scripted observations, not projections of actors. They do not
automatically change with ego motion. There is no occlusion, lighting, tire or
learned-detector realism. This minimal association fixture does not implement
BL-D-03's robust depth rejection, a route corridor, filtering or a speed governor.
The complete perception-to-control stopping pipeline remains BL-F-06.

## Runnable minimal observation example

After the repository's normal editable installation, run this Python example.
The policy consumes only `frame` and `batch`, associating their synthetic depth.
It returns a fixed throttle command to exercise the interface; it is not an
obstacle-avoidance policy.

```python
from autonomy.stack import SkeletonStack
from navigation_control.types import ControlCommand
from obstacle_perception.types import DetectionObservation, PixelBox
from sim_harness import (
    KinematicWorld, SyntheticDetection, SyntheticDetector,
    SyntheticFrame, SyntheticObstacleSource, SyntheticScenario,
)

scenario = SyntheticScenario((SyntheticFrame(0, (
    SyntheticDetection(
        DetectionObservation("car", 0.9, PixelBox(60, 30, 80, 50), 3),
        axial_depth_m=12.0,
    ),
)),))
world = KinematicWorld(scenario=scenario)
source = SyntheticObstacleSource(extent_prior_m=(4.0, 2.0, 1.5))


def policy(frame, batch):
    obstacle, = source.observe(frame, batch)
    print("observed ego XYZ (m):", obstacle.center_ego_m)
    return ControlCommand(0.0, 0.25, 0.0)


stack = SkeletonStack(SyntheticDetector(scenario), motion_policy=policy)
try:
    frame, graph = world.reset(seed=7)
    source.reset(episode_id=frame.episode_id)
    stack.reset(episode_id=frame.episode_id, graph=graph)
    packet = stack.step(frame)
    assert packet.diagnostics.observation_valid, packet.diagnostics.reasons
    next_frame = world.step(packet)
    assert next_frame.frame_id == 1
    print("next state:", next_frame.ego.state)
finally:
    try:
        stack.close()
    finally:
        source.close()
        world.close()
```

For repeated runs, use a policy object with `reset(episode_id, graph)` and `close()`
hooks to manage its controller and obstacle source. The existing stack passes
only frame and detection batch to its policy. No world/scene lookup is needed.

## Verification

```text
python -m pytest tests/unit/test_kinematic_world.py tests/unit/test_synthetic_observations.py
python -m pytest
python -m ruff check .
python scripts/check_simulator_boundary.py --smoke
python -m sphinx -W -b html docs docs/_build/html
```

Tests compare accelerating straight motion and left/right constant-steering arcs
to analytic references, exercise steering limits and within-step stopping, hold
and resume without replacing the route, replay/reset isolation, packet rejection
and failure atomicity. Multi-step `SkeletonStack`/`PathFollower` tests complete
straight and curved synthetic routes twice with identical physical traces.
Observation tests check projection, metadata, track namespaces, missing/invalid
depth, empty observations, lifecycle and stack fault latching.

Task evidence is recorded in `docs/evidence/BL-F-03-regression.txt`. These are
CARLA-free harness results; no live simulator result or production S1 acceptance
is claimed. CI scenario wiring, reproducible metric aggregation and full obstacle
stopping integration remain BL-F-04, BL-F-05 and BL-F-06 respectively.
