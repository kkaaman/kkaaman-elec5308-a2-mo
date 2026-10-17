"""BL-E-03 throttle/brake step collection, bounded to a checked straight site."""

import argparse
import csv
import json
import math
from pathlib import Path

from carla_adapter.vehicle import FixedVehicleAdapter
from carla_adapter.world import SynchronousWorldSession, _settings_record
from evaluation.longitudinal import braking_summary


def collect(client, output, follow=False):
    import carla

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    world = client.get_world()
    map_ = world.get_map()
    before = _settings_record(world.get_settings())
    initial_ids = {a.id for a in world.get_actors()}
    weather = world.get_weather()
    view = world.get_spectator().get_transform()
    if any(a.type_id.startswith(("vehicle.", "walker.", "sensor.")) for a in world.get_actors()):
        raise RuntimeError("Calibration requires empty world")
    if client.get_server_version() != "0.9.16" or client.get_client_version() != "0.9.16":
        raise ValueError("CARLA 0.9.16 required")
    candidates = []
    for i, pose in enumerate(map_.get_spawn_points()):
        wp = map_.get_waypoint(pose.location)
        ok = True
        for distance in range(2, 61, 2):
            options = wp.next(distance)
            if len(options) != 1 or options[0].is_junction:
                ok = False
                break
            angle = math.radians(options[0].transform.rotation.yaw - pose.rotation.yaw)
            if abs(math.atan2(math.sin(angle), math.cos(angle))) > math.radians(2):
                ok = False
                break
        if ok:
            candidates.append((i, pose))
    if not candidates:
        raise RuntimeError("No 60 m straight calibration site")
    site, spawn = candidates[0]
    rows = []
    error = None
    try:
        with SynchronousWorldSession(client) as owner:
            world.set_weather(carla.WeatherParameters.ClearNoon)
            trial = 0
            for target in (2, 5, 8):
                for brake in (0.4, 0.7, 1.0):
                    trial += 1
                    print(f"Longitudinal trial {trial}/9: {target} m/s, brake {brake}", flush=True)
                    vehicle = owner.try_spawn(
                        world.get_blueprint_library().find("vehicle.tesla.model3"), spawn
                    )
                    if vehicle is None:
                        raise RuntimeError("Fixed calibration spawn failed")
                    vehicle.apply_control(carla.VehicleControl(brake=1))
                    for _ in range(20):
                        owner.tick()
                    adapter = FixedVehicleAdapter(vehicle)
                    initial = adapter.state()
                    phase = "throttle"
                    braking_started = None
                    for tick in range(400):
                        state = adapter.state()
                        now = world.get_snapshot().timestamp.elapsed_seconds
                        distance = math.hypot(state.x_m - initial.x_m, state.y_m - initial.y_m)
                        if state.speed_mps >= target and phase == "throttle":
                            phase = "brake"
                            braking_started = now
                        effort = brake if phase == "brake" else 0
                        rows.append(
                            dict(
                                trial=trial,
                                target_mps=target,
                                phase=phase,
                                sim_time_s=now,
                                speed_mps=state.speed_mps,
                                distance_m=distance,
                                throttle=0.6 if phase == "throttle" else 0,
                                brake=effort,
                            )
                        )
                        if distance > 48:
                            raise RuntimeError("Calibration travel bound exceeded")
                        if abs(state.yaw_rad - initial.yaw_rad) > math.radians(4):
                            raise RuntimeError("Calibration left straight heading")
                        if (
                            phase == "brake"
                            and now - braking_started > 0.2
                            and state.speed_mps <= 0.1
                        ):
                            break
                        vehicle.apply_control(
                            carla.VehicleControl(
                                throttle=0.6 if phase == "throttle" else 0, brake=effort
                            )
                        )
                        owner.tick()
                        if follow:
                            pose = vehicle.get_transform()
                            world.get_spectator().set_transform(
                                carla.Transform(
                                    carla.Location(
                                        x=pose.location.x, y=pose.location.y, z=pose.location.z + 15
                                    ),
                                    carla.Rotation(pitch=-85, yaw=pose.rotation.yaw),
                                )
                            )
                    else:
                        raise RuntimeError("Longitudinal trial did not stop")
                    adapter.stop()
                    if not vehicle.destroy():
                        raise RuntimeError("Calibration vehicle cleanup failed")
                    owner.tick()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        world.set_weather(weather)
        world.get_spectator().set_transform(view)
        with (output / "longitudinal.csv").open("w", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    report = braking_summary(rows)
    report.update(
        error=error,
        site=site,
        server=client.get_server_version(),
        map=map_.name,
        settings_restored=_settings_record(world.get_settings()) == before,
        no_actor_leaks={a.id for a in world.get_actors()} == initial_ids,
    )
    if error or not report["settings_restored"] or not report["no_actor_leaks"]:
        report["status"] = "INCOMPLETE"
    (output / "longitudinal.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}, indent=2))
    if report["status"] != "MEASURED_PASS":
        raise RuntimeError("Longitudinal calibration incomplete; raw trials retained")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(60)
    collect(client, args.output, args.follow)
