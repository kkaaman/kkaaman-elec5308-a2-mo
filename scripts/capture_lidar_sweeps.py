"""Capture vehicle-frame LiDAR sweeps around a known parked target (E2-02 evidence).

Reuses SynchronousWorldSession, FixedVehicleAdapter and the new LidarCapture. The
ego stays parked on a checked straight road; a target vehicle is placed
kinematically at a few known distances ahead. Every sweep is saved as a vehicle
frame .npy with its frame ID, simulation time, point count and density, and the
target's expected rear-axle-frame box (evaluation-only truth from the actor's
bounding vertices) is compared with the observed points. The result is a LiDAR
dataset for offline replay, not driving acceptance.
"""

import argparse
import hashlib
import json
import math
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from carla_adapter.lidar import LidarCapture, LidarRig, known_object_check
from carla_adapter.vehicle import FixedVehicleAdapter
from carla_adapter.world import SynchronousWorldSession, WorldTiming, _settings_record
from obstacle_perception.lidar_synthetic import FIXTURE_SCHEMA_VERSION

SENSOR_READ_TIMEOUT_S = 5.0
SETTLE_TICKS = 12
DEFAULT_DISTANCES_M = (10.0, 20.0, 30.0)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_target_box(world_to_actor, vertices_world, rear_axle_offset_m):
    """Evaluation only: CARLA world bounding vertices -> rear-axle ego box.

    ``world_to_actor`` is the ego actor's 4x4 inverse transform (CARLA left-handed).
    Returns ``(center_ego_m, extent_m)`` in the right-handed rear-axle frame.
    """
    points = np.column_stack(
        (np.asarray(vertices_world, dtype=float), np.ones(len(vertices_world)))
    )
    local = (np.asarray(world_to_actor, dtype=float) @ points.T).T[:, :3]
    ego = np.column_stack((local[:, 0] + rear_axle_offset_m, -local[:, 1], local[:, 2]))
    low, high = ego.min(axis=0), ego.max(axis=0)
    return [float(v) for v in (low + high) / 2], [float(v) for v in np.maximum(high - low, 1e-3)]


def serialise_sweep(sweep, npy_name, directory, *, check, target, distance_m):
    """Manifest row for one captured sweep; the array is saved alongside."""
    path = Path(directory) / npy_name
    np.save(path, sweep.points_xyzi)
    return dict(
        frame_id=sweep.frame_id,
        sim_time_s=sweep.sim_time_s,
        npy=npy_name,
        sha256=sha256(path),
        point_count=sweep.point_count,
        density_fraction=sweep.density_fraction,
        azimuth_coverage_deg=sweep.azimuth_coverage_deg(),
        target_distance_m=distance_m,
        objects=[target] if target is not None else [],
        known_object_check=check,
        depth_gap_m=None,
        depth_valid=False,
        note="live_capture",
    )


def validate_environment(client_version, server_version, map_name):
    if client_version != "0.9.16" or server_version != "0.9.16":
        raise ValueError("Matching CARLA 0.9.16 client/server required")
    if map_name.split("/")[-1] != "Town10HD_Opt":
        raise ValueError("Town10HD_Opt required")


def straight_site(map_):
    """A spawn point followed by 60 m of single-successor, straight, non-junction road."""
    for index, pose in enumerate(map_.get_spawn_points()):
        waypoint = map_.get_waypoint(pose.location)
        for distance in range(2, 61, 2):
            options = waypoint.next(distance)
            if len(options) != 1 or options[0].is_junction:
                break
            delta = math.radians(options[0].transform.rotation.yaw - pose.rotation.yaw)
            if abs(math.atan2(math.sin(delta), math.cos(delta))) > math.radians(2):
                break
        else:
            return index, pose, waypoint
    raise RuntimeError("No checked 60 m straight capture site")


def target_pose(carla, waypoint, distance):
    options = waypoint.next(distance)
    if len(options) != 1 or options[0].is_junction:
        raise RuntimeError("Target left checked straight road")
    pose = options[0].transform
    pose.location.z += 0.05
    return carla.Transform(pose.location, pose.rotation)


def collect(args):
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(20)
    world = client.get_world()
    map_ = world.get_map()
    validate_environment(client.get_client_version(), client.get_server_version(), map_.name)
    if any(a.type_id.startswith(("vehicle.", "sensor.", "walker.")) for a in world.get_actors()):
        raise RuntimeError("Empty world required; stop other capture/driving clients first")
    if world.get_settings().synchronous_mode:
        raise RuntimeError("Another client owns synchronous ticks")
    before = _settings_record(world.get_settings())
    actors_before = {a.id for a in world.get_actors()}
    site, spawn, waypoint = straight_site(map_)
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    (args.output / "capture-script.py").write_bytes(source.read_bytes())
    rig = LidarRig(z_actor_m=args.mount_height)
    episode = f"lidar-capture-{uuid.uuid4()}"
    document = dict(
        schema_version=FIXTURE_SCHEMA_VERSION,
        origin="recorded",
        generator=str(source.name),
        capture_status="INCOMPLETE",
        episode_id=episode,
        fixed_delta_seconds=args.dt,
        rig=rig.to_dict(),
        environment=dict(
            server=client.get_server_version(),
            client=client.get_client_version(),
            map=map_.name,
            opendrive_sha256=hashlib.sha256(
                map_.to_opendrive().replace("\r\n", "\n").encode()
            ).hexdigest(),
            site=site,
            weather=str(world.get_weather()),
        ),
        provenance=dict(capture_script=dict(path=str(source), sha256=sha256(source))),
        route_xy_m=[[0.0, 0.0], [60.0, 0.0]],
        sweeps=[],
    )
    owned_ids = []
    error = None
    try:
        with SynchronousWorldSession(
            client,
            WorldTiming(fixed_delta_seconds=args.dt),
            ownership_key=f"{args.host}:{args.port}",
        ) as owner:
            ego = owner.try_spawn(world.get_blueprint_library().find("vehicle.tesla.model3"), spawn)
            if ego is None:
                raise RuntimeError("Ego spawn failed")
            owned_ids.append(ego.id)
            ego.apply_control(carla.VehicleControl(brake=1))
            for _ in range(20):
                owner.tick()
            adapter = FixedVehicleAdapter(ego)
            ego.set_simulate_physics(False)
            offset = adapter.geometry.rear_axle_offset_m
            document["front_bumper_from_rear_axle_m"] = float(
                ego.bounding_box.location.x + ego.bounding_box.extent.x + offset
            )
            document["rear_axle_offset_m"] = offset
            target = owner.try_spawn(
                world.get_blueprint_library().find("vehicle.tesla.model3"),
                target_pose(carla, waypoint, DEFAULT_DISTANCES_M[0]),
            )
            if target is None:
                raise RuntimeError("Target spawn failed")
            owned_ids.append(target.id)
            target.set_simulate_physics(False)
            with LidarCapture(
                owner, ego, episode, rear_axle_offset_m=offset, rig=rig, capacity=16
            ) as capture:
                owned_ids.append(capture.sensor.id)
                document["config"] = asdict(capture.config)
                document["sensor_to_ego"] = [
                    [float(v) for v in row] for row in capture.sensor_to_ego
                ]
                document["actor_ids"] = dict(ego=ego.id, target=target.id, lidar=capture.sensor.id)
                for distance in args.distances:
                    target.set_transform(target_pose(carla, waypoint, distance))
                    for _ in range(SETTLE_TICKS):
                        owner.tick()
                    for _ in range(args.frames):
                        frame_id = owner.tick()
                        snapshot = world.get_snapshot()
                        if snapshot.frame != frame_id:
                            raise RuntimeError("Snapshot frame does not match tick")
                        sweep = capture.read(frame_id, SENSOR_READ_TIMEOUT_S)
                        if abs(sweep.sim_time_s - snapshot.timestamp.elapsed_seconds) > 1e-6:
                            raise RuntimeError("Sweep timestamp does not match the tick snapshot")
                        center, extent = expected_target_box(
                            ego.get_transform().get_inverse_matrix(),
                            [
                                [v.x, v.y, v.z]
                                for v in target.bounding_box.get_world_vertices(
                                    target.get_transform()
                                )
                            ],
                            offset,
                        )
                        check = known_object_check(sweep, center_ego_m=center, extent_m=extent)
                        target_box = dict(
                            x_m=center[0],
                            y_m=center[1],
                            size_x_m=extent[0],
                            size_y_m=extent[1],
                            height_m=extent[2],
                            base_z_m=center[2] - extent[2] / 2,
                            label="target",
                        )
                        document["sweeps"].append(
                            serialise_sweep(
                                sweep,
                                f"sweep_{frame_id:06d}.npy",
                                args.output,
                                check=check,
                                target=target_box,
                                distance_m=distance,
                            )
                        )
                        print(
                            f"frame {frame_id} t={sweep.sim_time_s:.3f}s points={sweep.point_count} "
                            f"density={sweep.density_fraction:.2f} known_object={'PASS' if check['passed'] else 'FAIL'} "
                            f"face_error={check['near_face_error_m']}"
                        )
                document["capture_records"] = capture.records
                document["queue_events"] = capture.queue.stats()
                document["callback_errors"] = list(capture.callback_errors)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        remaining = {a.id for a in world.get_actors()} - actors_before
        document["cleanup"] = dict(
            owned_ids=owned_ids,
            leaked_owned_ids=sorted(set(owned_ids) & remaining),
            settings_restored=_settings_record(world.get_settings()) == before,
        )
        document["error"] = error
        checks = [row["known_object_check"]["passed"] for row in document["sweeps"]]
        document["known_object_checks_passed"] = int(sum(checks))
        document["known_object_checks_total"] = len(checks)
        document["capture_status"] = (
            "COMPLETE" if error is None and checks and all(checks) else "INCOMPLETE"
        )
        (args.output / "sweeps.json").write_text(json.dumps(document, indent=2) + "\n")
        print(f"Wrote {args.output / 'sweeps.json'} ({document['capture_status']})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--dt", type=float, default=0.05, help="fixed_delta_seconds; LiDAR spins at 1/dt"
    )
    parser.add_argument(
        "--frames", type=int, default=10, help="sweeps captured per target distance"
    )
    parser.add_argument(
        "--distances",
        type=float,
        nargs="+",
        default=list(DEFAULT_DISTANCES_M),
        help="target distances ahead of the ego spawn (metres)",
    )
    parser.add_argument(
        "--mount-height",
        type=float,
        default=2.4,
        help="LiDAR height above the actor origin (metres)",
    )
    parser.add_argument("--output", type=Path, required=True, help="new directory under runs/")
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
