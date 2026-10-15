"""Short fixed-step Tesla steering sweeps through the installed vehicle adapter."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from evaluation.steering_calibration import SpeedServo, analyze, inferred_angle
from navigation_control.types import ControlCommand

from .vehicle import FixedVehicleAdapter
from .world import SynchronousWorldSession, WorldTiming, _settings_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/vehicle_steering_sweep.yaml")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--analyze", type=Path)
    args = parser.parse_args()
    if args.analyze:
        report = analyze(args.analyze)
        print(json.dumps(report, indent=2))
        return
    if args.output is None:
        parser.error("--output is required; use a new folder in Downloads")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if config != json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs/vehicle_steering_sweep.yaml"
        ).read_text(encoding="utf-8-sig")
    ):
        parser.error("Use the repository sweep configuration")
    if (
        config["blueprint"] != "vehicle.tesla.model3"
        or config["speeds_mps"] != [2, 5, 8]
        or config["angles_deg"] != [2, 4, 6]
    ):
        parser.error("Unexpected sweep configuration")
    timing = WorldTiming(fixed_delta_seconds=config["fixed_delta_seconds"])
    if (
        timing.fixed_delta_seconds != 0.05
        or config["sweep_seconds"] != 1
        or config["max_acceleration_seconds"] != 12
    ):
        parser.error("Timing differs from the reviewed short sweep")
    if (
        not 0 < config["max_distance_from_spawn_m"] <= 48
        or not 0 < config["max_lateral_from_spawn_m"] <= 1.4
    ):
        parser.error("Invalid travel bounds")

    import carla

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30)
    world = client.get_world()
    road_map = world.get_map()
    if road_map.name.split("/")[-1] != config["map"]:
        parser.error("Wrong map; no map change performed")
    if any(a.type_id.startswith(("vehicle.", "sensor.")) for a in world.get_actors()):
        parser.error("Use an isolated world with no vehicles/sensors")
    weather = world.get_weather()
    if weather.wetness > 0 or weather.precipitation > 0:
        parser.error("Calibration requires dry conditions")
    spawn = road_map.get_spawn_points()[config["spawn_index"]]
    # Recheck the selected site's forward map geometry before any vehicle spawn.
    waypoint = road_map.get_waypoint(spawn.location)
    origin_yaw = waypoint.transform.rotation.yaw
    if waypoint.is_junction:
        parser.error("Selected spawn is at a junction")
    for _ in range(30):
        following = waypoint.next(2)
        if len(following) != 1:
            parser.error("Site lacks the checked 60 m straight corridor")
        waypoint = following[0]
        angle = abs((waypoint.transform.rotation.yaw - origin_yaw + 180) % 360 - 180)
        if (
            waypoint.is_junction
            or angle > 2
            or abs(waypoint.transform.rotation.pitch) > 2
        ):
            parser.error("Site geometry no longer matches the checked corridor")
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    initial_settings = _settings_record(world.get_settings())
    initial_ids = {actor.id for actor in world.get_actors()}
    spectator = world.get_spectator()
    original_view = spectator.get_transform()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    manifest = {
        "source_commit": revision.stdout.strip(),
        "runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "weather": _settings_record(weather),
        "opendrive_sha256": hashlib.sha256(
            road_map.to_opendrive().encode()
        ).hexdigest(),
        "server": client.get_server_version(),
        "map": road_map.name,
        "config": config,
        "smoke_only": args.smoke,
        "python": sys.version,
        "initial_settings": initial_settings,
        "trials": [],
        "clipping_policy": "commands above controller cap rejected, not silently normalized to full lock",
        "status": "RUNNING",
    }
    stream = (folder / "samples.csv").open("w", newline="", encoding="utf-8")
    fields = [
        "trial",
        "phase",
        "phase_time_s",
        "frame",
        "sim_time_s",
        "target_speed_mps",
        "requested_angle_rad",
        "applied_steer",
        "speed_mps",
        "core_yaw_rate_rad_s",
        "inferred_angle_rad",
        "wheel_fl_deg_raw",
        "wheel_fr_deg_raw",
        "throttle",
        "brake",
        "distance_from_spawn_m",
        "lateral_from_spawn_m",
        "actor_x_m",
        "actor_y_m",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    speeds = [2.0] if args.smoke else config["speeds_mps"]
    angles = [4.0] if args.smoke else config["angles_deg"]
    failure = None
    try:
        with SynchronousWorldSession(client, timing) as owner:
            trial_id = 0
            for target in speeds:
                for degrees in angles:
                    for sign in (1, -1):
                        trial_id += 1
                        print(
                            f"Trial {trial_id}: {target:g} m/s, {'LEFT' if sign > 0 else 'RIGHT'} {degrees:g} deg",
                            flush=True,
                        )
                        vehicle = owner.try_spawn(
                            world.get_blueprint_library().find(config["blueprint"]),
                            spawn,
                        )
                        if vehicle is None:
                            raise RuntimeError("Selected spawn is occupied")
                        try:
                            vehicle.apply_control(carla.VehicleControl(brake=1))
                            for _ in range(20):
                                owner.tick()
                            adapter = FixedVehicleAdapter(vehicle)
                            start = vehicle.get_transform()
                            yaw = math.radians(start.rotation.yaw)
                            if not hasattr(vehicle, "get_wheel_steer_angle"):
                                raise RuntimeError(
                                    "Installed build lacks physical wheel-angle readback"
                                )
                            try:
                                adapter.apply(
                                    ControlCommand(
                                        adapter.geometry.controller_steering_limit_rad
                                        + 0.01,
                                        0,
                                        1,
                                    )
                                )
                            except ValueError:
                                cap_rejected = True
                            else:
                                raise RuntimeError("Controller cap was not enforced")
                            physics = vehicle.get_physics_control()
                            manifest["trials"].append(
                                {
                                    "trial": trial_id,
                                    "target_speed_mps": target,
                                    "requested_deg": sign * degrees,
                                    "geometry": asdict(adapter.geometry),
                                    "controller_cap_rejection": cap_rejected,
                                    "steering_curve_raw": [
                                        [p.x, p.y] for p in physics.steering_curve
                                    ],
                                    "spawn": {
                                        "x": start.location.x,
                                        "y": start.location.y,
                                        "z": start.location.z,
                                        "yaw_deg": start.rotation.yaw,
                                    },
                                }
                            )
                            servo = SpeedServo()
                            stable = 0
                            requested = sign * math.radians(degrees)

                            def step(phase, elapsed, steering):
                                state = adapter.state()
                                throttle, brake = servo.update(
                                    target, state.speed_mps, timing.fixed_delta_seconds
                                )
                                adapter.apply(ControlCommand(steering, throttle, brake))
                                frame = owner.tick()
                                pose = vehicle.get_transform()
                                state = adapter.state()
                                snapshot = world.get_snapshot()
                                if not math.isclose(
                                    snapshot.timestamp.delta_seconds,
                                    timing.fixed_delta_seconds,
                                    abs_tol=1e-6,
                                ):
                                    raise RuntimeError("Unexpected simulation timestep")
                                rate = -math.radians(vehicle.get_angular_velocity().z)
                                dx = pose.location.x - start.location.x
                                dy = pose.location.y - start.location.y
                                distance = math.hypot(dx, dy)
                                lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
                                if (
                                    distance > config["max_distance_from_spawn_m"]
                                    or abs(lateral) > config["max_lateral_from_spawn_m"]
                                ):
                                    raise RuntimeError(
                                        "Travel bound exceeded; trial stopped"
                                    )
                                if abs(pose.location.z - start.location.z) > 0.25:
                                    raise RuntimeError(
                                        "Vehicle height changed; possible kerb/contact"
                                    )
                                control = vehicle.get_control()
                                estimate = (
                                    inferred_angle(
                                        adapter.geometry.wheelbase_m,
                                        state.speed_mps,
                                        rate,
                                    )
                                    if state.speed_mps > 0.5
                                    else None
                                )
                                writer.writerow(
                                    dict(
                                        zip(
                                            fields,
                                            [
                                                trial_id,
                                                phase,
                                                elapsed,
                                                frame,
                                                snapshot.timestamp.elapsed_seconds,
                                                target,
                                                steering,
                                                control.steer,
                                                state.speed_mps,
                                                rate,
                                                estimate,
                                                vehicle.get_wheel_steer_angle(
                                                    carla.VehicleWheelLocation.FL_Wheel
                                                ),
                                                vehicle.get_wheel_steer_angle(
                                                    carla.VehicleWheelLocation.FR_Wheel
                                                ),
                                                control.throttle,
                                                control.brake,
                                                distance,
                                                lateral,
                                                pose.location.x,
                                                -pose.location.y,
                                            ],
                                        )
                                    )
                                )
                                stream.flush()
                                if args.follow:
                                    heading = math.radians(pose.rotation.yaw)
                                    spectator.set_transform(
                                        carla.Transform(
                                            carla.Location(
                                                x=pose.location.x
                                                - 7 * math.cos(heading),
                                                y=pose.location.y
                                                - 7 * math.sin(heading),
                                                z=pose.location.z + 4,
                                            ),
                                            carla.Rotation(
                                                pitch=-20, yaw=pose.rotation.yaw
                                            ),
                                        )
                                    )
                                    time.sleep(timing.fixed_delta_seconds)
                                return state.speed_mps

                            for index in range(
                                int(
                                    config["max_acceleration_seconds"]
                                    / timing.fixed_delta_seconds
                                )
                            ):
                                measured = step(
                                    "accelerate",
                                    (index + 1) * timing.fixed_delta_seconds,
                                    0,
                                )
                                stable = (
                                    stable + 1 if abs(measured - target) <= 0.2 else 0
                                )
                                if stable >= 10:
                                    break
                            else:
                                raise RuntimeError("Could not stabilize target speed")
                            for index in range(20):
                                step(
                                    "sweep",
                                    (index + 1) * timing.fixed_delta_seconds,
                                    requested,
                                )
                            adapter.stop()
                            owner.tick()
                        finally:
                            try:
                                if vehicle.is_alive:
                                    vehicle.apply_control(carla.VehicleControl(brake=1))
                            finally:
                                if vehicle.is_alive and not vehicle.destroy():
                                    raise RuntimeError("Trial vehicle cleanup failed")
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        stream.close()
        spectator.set_transform(original_view)
        manifest["failure"] = failure
        manifest["settings_restored"] = (
            _settings_record(world.get_settings()) == initial_settings
        )
        manifest["no_actor_leaks"] = {a.id for a in world.get_actors()} == initial_ids
        manifest["status"] = (
            "COLLECTED"
            if failure is None
            and manifest["settings_restored"]
            and manifest["no_actor_leaks"]
            else "INCOMPLETE"
        )
        (folder / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "trials": len(manifest["trials"]),
                    "settings_restored": manifest["settings_restored"],
                    "no_actor_leaks": manifest["no_actor_leaks"],
                    "output": str(folder),
                    "failure": failure,
                },
                indent=2,
            )
        )
    if manifest["status"] != "COLLECTED":
        raise RuntimeError("Collection incomplete; inspect manifest")
    report = analyze(folder)
    print(
        f"Analysis and plot saved. Usable short windows: {sum(t['quality'] == 'USABLE_SHORT_WINDOW' for t in report['trials'])}/{len(report['trials'])}"
    )


if __name__ == "__main__":
    main()
