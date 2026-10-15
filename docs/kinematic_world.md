# KinematicWorld

`KinematicWorld` is a deterministic, simulator-free implementation of the
shared `SimulatorInterface`. It allows the navigation stack to be exercised in
CI without starting CARLA.

## Behaviour

- Uses a fixed-time-step kinematic bicycle model.
- Accepts normal `ControlPacket` commands.
- Provides frame bundles with a synthetic RGB image.
- Produces projected synthetic obstacle objects through `detector_scene()`.
  These objects can be passed to the fake detector boundary.
- Keeps evaluation ground truth in `evaluation()` only; control code should use
  detector outputs and must not read simulator truth directly.
- Supports reset, accelerate, steer, brake, stop, and restart scenarios.

## Limitations

This is a test harness, not a physically realistic vehicle simulator. It does
not model tyre slip, suspension, actuator delay, collisions, moving actors,
occlusion, illumination, sensor noise, or camera-depth geometry. Obstacle
detections are deterministic geometric projections. CARLA remains the required
environment for realistic perception and closed-loop validation.