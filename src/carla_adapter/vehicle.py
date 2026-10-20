"""Fixed Tesla adapter. CARLA imports stay inside this package."""

import math

from .vehicle_geometry import forward_state, geometry_from_wheels, normalized_steering


class FixedVehicleAdapter:
    """Bind to a live Tesla; inspect after physics has updated its pose.

    Nominal steering normalization is not a calibrated dynamic angle model.
    Lifecycle/tick ownership belong to the world adapter, not this class.
    """

    def __init__(self, vehicle, controller_steering_limit_rad=math.radians(20)):
        if vehicle.type_id != "vehicle.tesla.model3":
            raise ValueError("Baseline requires vehicle.tesla.model3")
        self.vehicle = vehicle
        physics = vehicle.get_physics_control()
        self.geometry = geometry_from_wheels(
            [[w.position.x, w.position.y, w.position.z] for w in physics.wheels],
            vehicle.get_transform().get_inverse_matrix(),
            [w.max_steer_angle for w in physics.wheels],
            controller_steering_limit_rad,
        )
        if (
            abs(self.geometry.wheelbase_m - 3.00465) > 0.02
            or abs(self.geometry.rear_axle_offset_m - 1.38641) > 0.02
            or abs(math.degrees(self.geometry.physical_steering_limit_rad) - 70) > 0.1
        ):
            raise ValueError("Vehicle geometry differs from verified baseline")

    def state(self):
        if not self.vehicle.is_alive:
            raise RuntimeError("Vehicle is no longer alive")
        pose = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        if abs(pose.rotation.pitch) > 5 or abs(pose.rotation.roll) > 5:
            raise ValueError("Vehicle tilt exceeds planar adapter scope")
        return forward_state(
            pose.location.x,
            pose.location.y,
            pose.rotation.yaw,
            velocity.x,
            velocity.y,
            self.geometry,
        )

    def stop(self):
        import carla

        if self.vehicle.is_alive:
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0, reverse=False))

    def apply(self, command, *, client=None):
        """Dispatch control, or wait for an acknowledged batch when client is supplied.

        The acknowledged path does not tick the world; its owner remains the
        only simulation tick source. Evidence timing must use this path because
        CARLA's ordinary apply_control call dispatches asynchronously.
        """
        import carla

        try:
            self.state()
            steer = normalized_steering(command, self.geometry)
            control = carla.VehicleControl(
                throttle=command.throttle,
                brake=command.brake,
                steer=steer,
                reverse=False,
                hand_brake=False,
                manual_gear_shift=False,
            )
            if client is None:
                self.vehicle.apply_control(control)
            else:
                responses = client.apply_batch_sync(
                    [carla.command.ApplyVehicleControl(self.vehicle.id, control)], False
                )
                if len(responses) != 1 or responses[0].error:
                    error = responses[0].error if len(responses) == 1 else "Unexpected response count"
                    raise RuntimeError(f"Control application was not acknowledged: {error}")
            return control
        except Exception:
            self.stop()
            raise


def main():
    """Parked live smoke check; run when this client may advance a sync world."""
    import argparse
    import json
    from dataclasses import asdict
    from pathlib import Path

    import carla

    from navigation_control.types import ControlCommand

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Evidence file already exists; choose a new output name")
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60)
    world = client.get_world()
    vehicle = None
    try:
        blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
        for spawn in world.get_map().get_spawn_points():
            vehicle = world.try_spawn_actor(blueprint, spawn)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("No free spawn point")
        vehicle.apply_control(carla.VehicleControl(brake=1))
        for _ in range(10):
            if world.get_settings().synchronous_mode:
                world.tick()
            else:
                world.wait_for_tick(30)
        adapter = FixedVehicleAdapter(vehicle)
        state = adapter.state()
        left = adapter.apply(ControlCommand(math.radians(10), 0, 1))
        right = adapter.apply(ControlCommand(-math.radians(10), 0, 1))
        if not left.steer < 0 < right.steer:
            raise RuntimeError("Adapter steering sign check failed")
        adapter.stop()
        report = {
            "server": client.get_server_version(),
            "map": world.get_map().name,
            "geometry": asdict(adapter.geometry),
            "state": asdict(state),
            "left_normalized_steer": left.steer,
            "right_normalized_steer": right.steer,
            "scope": "installed adapter geometry/state and parked command mapping; not dynamic calibration",
        }
    finally:
        if vehicle is not None:
            if not vehicle.destroy():
                raise RuntimeError("Vehicle cleanup failed")
            print("Vehicle cleanup: PASS")
    report["cleanup"] = "PASS"
    report["status"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
