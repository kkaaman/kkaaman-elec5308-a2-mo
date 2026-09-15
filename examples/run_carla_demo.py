r"""Run a minimal CARLA path-following demonstration.

Start CARLA first, then run:
    $env:PYTHONPATH = "$PWD\src"
    python examples/run_carla_demo.py
"""

import argparse
import math
import sys
import time

import carla

from carla_adapter import apply_command, carla_vehicle_to_state
from navigation_control.controller import PathFollower
from navigation_control.types import (
    ControlStatus,
    ControllerConfig,
    Waypoint,
)


def connect_to_carla(host: str, port: int, timeout_s: float) -> tuple[carla.Client, carla.World]:
    """Connect to a running CARLA server or raise a clear error."""

    client = carla.Client(host, port)
    client.set_timeout(timeout_s)

    try:
        world = client.get_world()
    except RuntimeError as error:
        raise RuntimeError(
            f"Could not connect to CARLA at {host}:{port}. "
            "Start CarlaUE4.exe first, then retry."
        ) from error

    return client, world


def build_route(
    world: carla.World,
    start_location: carla.Location,
    waypoint_spacing_m: float,
    waypoint_count: int,
) -> list[Waypoint]:
    """Build a simple lane-centre route starting from a CARLA spawn point."""

    carla_waypoint = world.get_map().get_waypoint(start_location)
    route = [
        Waypoint(
            x_m=carla_waypoint.transform.location.x,
            y_m=-carla_waypoint.transform.location.y,
        )
    ]

    for _ in range(waypoint_count - 1):
        next_waypoints = carla_waypoint.next(waypoint_spacing_m)
        if not next_waypoints:
            break

        carla_waypoint = next_waypoints[0]
        location = carla_waypoint.transform.location
        route.append(Waypoint(x_m=location.x, y_m=-location.y))

    if len(route) < 2:
        raise RuntimeError("Could not build a route with at least two waypoints.")

    return route


def spawn_vehicle(world: carla.World) -> carla.Vehicle:
    """Spawn one vehicle at the first available CARLA spawn point."""

    blueprints = world.get_blueprint_library().filter("vehicle.tesla.model3")
    if not blueprints:
        blueprints = world.get_blueprint_library().filter("vehicle.*")

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("The current CARLA map has no vehicle spawn points.")

    vehicle = world.try_spawn_actor(blueprints[0], spawn_points[0])
    if vehicle is None:
        raise RuntimeError("Could not spawn the demo vehicle.")

    return vehicle


def run_demo(args: argparse.Namespace) -> None:
    """Run the CARLA path-following control loop."""

    _, world = connect_to_carla(args.host, args.port, args.timeout)

    vehicle = None
    try:
        vehicle = spawn_vehicle(world)
        route = build_route(
            world,
            vehicle.get_location(),
            args.waypoint_spacing,
            args.waypoint_count,
        )

        config = ControllerConfig(
            wheelbase_m=2.875,
            max_steering_angle_rad=math.radians(35.0),
            lookahead_m=5.0,
            kp=0.6,
            ki=0.02,
            stopping_deceleration_mps2=1.5,
        )
        follower = PathFollower(config)
        follower.set_path(route)

        print(
            f"Demo started: {len(route)} waypoints, "
            f"target speed {args.target_speed_mps:.1f} m/s."
        )

        previous_time = time.monotonic()
        end_time = previous_time + args.max_duration_s

        while time.monotonic() < end_time:
            now = time.monotonic()
            dt_s = max(0.01, min(now - previous_time, 0.2))
            previous_time = now

            state = carla_vehicle_to_state(
                vehicle,
                rear_axle_offset_m=args.rear_axle_offset_m,
            )
            result = follower.update(
                state=state,
                target_speed_mps=args.target_speed_mps,
                dt_s=dt_s,
            )
            apply_command(
                vehicle,
                result.command,
                config.max_steering_angle_rad,
            )

            if result.status == ControlStatus.FINISHED:
                print("Demo finished: route goal reached.")
                return

            time.sleep(0.05)

        print("Demo stopped: maximum duration reached.")

    finally:
        if vehicle is not None:
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
            time.sleep(0.5)
            vehicle.destroy()


def parse_arguments() -> argparse.Namespace:
    """Read the demo configuration from the command line."""

    parser = argparse.ArgumentParser(description="Run the CARLA navigation demo.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--target-speed-mps", type=float, default=5.0)
    parser.add_argument("--waypoint-spacing", type=float, default=2.0)
    parser.add_argument("--waypoint-count", type=int, default=40)
    parser.add_argument("--max-duration-s", type=float, default=60.0)
    parser.add_argument("--rear-axle-offset-m", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_demo(parse_arguments())
    except RuntimeError as error:
        print(f"CARLA demo error: {error}", file=sys.stderr)
        raise SystemExit(1)