"""Parked #40 LiDAR geometry screening; no clearance authorisation or driving.

Reuses canonical capture/transform/ground/cluster path. Every attempted sweep
and candidate is retained. Truth is setup/evaluation only.
"""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from carla_adapter.lidar import LidarCapture, LidarRig, known_object_check
from carla_adapter.vehicle import FixedVehicleAdapter
from carla_adapter.world import SynchronousWorldSession, WorldTiming, _settings_record
from obstacle_perception.lidar_obstacles import (
    CLASS_OBSTACLE,
    CLASS_RAISED,
    CLASS_ROAD,
    EgoBody,
    LidarObstaclePolicy,
    extract_obstacles,
    remove_ground,
)
from planning.corridor import RouteGeometry
from scripts.capture_lidar_sweeps import (
    expected_target_box,
    straight_site,
    target_pose,
    validate_environment,
    verify_cleanup,
)

CANDIDATES = [
    (
        "baseline",
        {
            "x_actor_m": 0.0,
            "z_actor_m": 2.4,
            "channels": 32,
            "points_per_second": 640000,
            "lower_fov_deg": -30.0,
        },
    ),
    ("roof_wider_lower", {"lower_fov_deg": -45}),
    ("roof_pitch_down", {"pitch_deg": -15}),
    ("front_low64", {"x_actor_m": 2.5, "z_actor_m": 0.9, "lower_fov_deg": -85, "channels": 64}),
    ("front_mid64", {"x_actor_m": 2.5, "z_actor_m": 1.3, "lower_fov_deg": -85, "channels": 64}),
    (
        "front_low128",
        {
            "x_actor_m": 2.5,
            "z_actor_m": 0.9,
            "lower_fov_deg": -85,
            "channels": 128,
            "points_per_second": 1280000,
        },
    ),
    (
        "front_mid128",
        {
            "x_actor_m": 2.5,
            "z_actor_m": 1.3,
            "lower_fov_deg": -85,
            "channels": 128,
            "points_per_second": 1280000,
        },
    ),
    (
        "front_high128",
        {
            "x_actor_m": 2.5,
            "z_actor_m": 1.5,
            "lower_fov_deg": -85,
            "channels": 128,
            "points_per_second": 1280000,
        },
    ),
]


def ray_box_mask(origin, points, low, high):
    """Conservative ray-segment intersection with known ego box,not obstacle truth."""
    direction = points - origin
    start = np.zeros(len(points))
    end = np.ones(len(points))
    possible = np.ones(len(points), bool)
    for axis in range(3):
        parallel = np.abs(direction[:, axis]) < 1e-10
        possible &= ~(parallel & ((origin[axis] < low[axis]) | (origin[axis] > high[axis])))
        denom = np.where(parallel, 1.0, direction[:, axis])
        a = (low[axis] - origin[axis]) / denom
        b = (high[axis] - origin[axis]) / denom
        start = np.maximum(start, np.where(parallel, -np.inf, np.minimum(a, b)))
        end = np.minimum(end, np.where(parallel, np.inf, np.maximum(a, b)))
    return possible & (start < end - 1e-8) & (end > 1e-6) & (start < 1 - 1e-6)


def coverage(points, classes, front, *, length=6.5, half_width=1.5, bin_m=0.5, strips=3):
    n = int(math.ceil(length / bin_m))
    counts = np.zeros((n, strips), int)
    occupied = np.zeros_like(counts)
    x = points[:, 0] - front
    y = points[:, 1]
    inside = (x >= 0) & (x < length) & (abs(y) < half_width)
    idx = np.floor(x[inside] / bin_m).astype(int)
    lateral = np.clip(
        np.floor((y[inside] + half_width) / (2 * half_width) * strips).astype(int), 0, strips - 1
    )
    label = classes[inside]
    road = label == CLASS_ROAD
    obj = np.isin(label, [CLASS_OBSTACLE, CLASS_RAISED])
    np.add.at(counts, (idx[road], lateral[road]), 1)
    np.add.at(occupied, (idx[obj], lateral[obj]), 1)
    good = counts >= 2
    anyroad = x[inside & (classes == CLASS_ROAD)]
    return dict(
        road_counts=counts.tolist(),
        occupied_counts=occupied.tolist(),
        cells=counts.size,
        road_supported_cells=int(good.sum()),
        complete_road_support=bool(good.all()),
        unknown_cells=int((~good & (occupied == 0)).sum()),
        occupied_cells=int((occupied > 0).sum()),
        first_road_gap_m=float(anyroad.min()) if len(anyroad) else None,
        first_1_5m_supported_cells=int(good[:3].sum()),
        first_1_5m_cells=3 * strips,
        positive_clearance_authorised=False,
    )


def screen(args):
    import carla

    c = carla.Client(args.host, 2000)
    c.set_timeout(20)
    w = c.get_world()
    m = w.get_map()
    validate_environment(c.get_client_version(), c.get_server_version(), m.name)
    if w.get_settings().synchronous_mode or any(
        a.type_id.startswith(("vehicle.", "sensor.", "walker.")) for a in w.get_actors()
    ):
        raise RuntimeError("Exclusive empty asynchronous world required")
    before = _settings_record(w.get_settings())
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "screen-script.py").write_bytes(Path(__file__).read_bytes())
    (output / "lidar-adapter.py").write_bytes(
        (Path(__file__).resolve().parents[1] / "src/carla_adapter/lidar.py").read_bytes()
    )
    document = dict(
        status="INCOMPLETE",
        server=c.get_server_version(),
        map=m.name,
        fixed_delta_seconds=0.05,
        required_length_m=6.150012,
        evaluated_full_bin_length_m=6.5,
        corridor_half_width_m=1.5,
        bins=dict(longitudinal_m=0.5, strips=3, min_road_returns=2),
        candidates=[],
        failures=[],
        truth_usage="setup/evaluation only",
    )
    owned = []
    try:
        site, spawn, waypoint = straight_site(m, args.site)
        document["site"] = site
        with SynchronousWorldSession(
            c, WorldTiming(fixed_delta_seconds=0.05), ownership_key=f"{args.host}:2000"
        ) as owner:
            ego = owner.try_spawn(w.get_blueprint_library().find("vehicle.tesla.model3"), spawn)
            if ego is None:
                raise RuntimeError("Ego spawn failed")
            owned.append(ego.id)
            ego.apply_control(carla.VehicleControl(brake=1))
            for _ in range(20):
                owner.tick()
            adapter = FixedVehicleAdapter(ego)
            offset = adapter.geometry.rear_axle_offset_m
            ego.set_simulate_physics(False)
            box = ego.bounding_box
            front = box.location.x + box.extent.x + offset
            rear = box.location.x - box.extent.x + offset
            width = box.extent.y
            bottom = box.location.z - box.extent.z
            top = box.location.z + box.extent.z
            body = EgoBody(
                front_m=front, rear_m=rear, half_width_m=width, height_m=top, margin_m=0.3
            )
            exact = replace(body, margin_m=0)
            document["ego"] = dict(
                front_from_rear_m=front,
                front_from_actor_m=front - offset,
                rear_axle_offset_m=offset,
                half_width_m=width,
                bottom_z_actor_m=bottom,
                top_z_actor_m=top,
                actor_transform=ego.get_transform().get_matrix(),
            )
            target = owner.try_spawn(
                w.get_blueprint_library().find("vehicle.tesla.model3"),
                target_pose(carla, waypoint, 30),
            )
            if target is None:
                raise RuntimeError("Target spawn failed")
            owned.append(target.id)
            target.set_simulate_physics(False)
            route = RouteGeometry(((0, 0), (60, 0)))
            policy = LidarObstaclePolicy()
            selected = [v for v in CANDIDATES if not args.candidate or v[0] in args.candidate]
            for name, changes in selected:
                folder = output / name
                folder.mkdir()
                rig = replace(
                    LidarRig(
                        x_actor_m=0.0,
                        z_actor_m=2.4,
                        channels=32,
                        points_per_second=640000,
                        lower_fov_deg=-30.0,
                    ),
                    **changes,
                )
                result = dict(name=name, rig=rig.to_dict(), frames=[], status="INCOMPLETE")
                document["candidates"].append(result)
                try:
                    with LidarCapture(
                        owner,
                        ego,
                        "geometry-" + name,
                        rear_axle_offset_m=offset,
                        rig=rig,
                        capacity=16,
                        raw_directory=folder / "raw",
                    ) as capture:
                        owned.append(capture.sensor.id)
                        result["sensor_to_ego"] = capture.sensor_to_ego.tolist()
                        for distance in [30.0, 10.0, 20.0, 5.0]:
                            target.set_transform(target_pose(carla, waypoint, distance))
                            for _ in range(6):
                                owner.tick()
                            for _ in range(args.frames):
                                frame = owner.tick()
                                snap = w.get_snapshot()
                                sweep = capture.read(frame)
                                assert (
                                    frame == snap.frame
                                    and abs(sweep.sim_time_s - snap.timestamp.elapsed_seconds)
                                    < 1e-6
                                )
                                data = sweep.points_xyzi
                                np.save(folder / f"ego_{frame}.npy", data)
                                centre, extent = expected_target_box(
                                    ego.get_transform().get_inverse_matrix(),
                                    [
                                        [v.x, v.y, v.z]
                                        for v in target.bounding_box.get_world_vertices(
                                            target.get_transform()
                                        )
                                    ],
                                    offset,
                                )
                                check = known_object_check(
                                    sweep,
                                    center_ego_m=centre,
                                    extent_m=extent,
                                    exclude_bounds=([rear, -width, bottom], [front, width, top]),
                                )
                                default = remove_ground(sweep, policy, body)
                                no_extra_mask = remove_ground(sweep, policy, exact)
                                rawroad = np.where(
                                    abs(data[:, 2]) < 0.08, CLASS_ROAD, CLASS_OBSTACLE
                                )
                                raw_cov = coverage(data[:, :3], rawroad, front)
                                filtered = coverage(default.points_xyz, default.classes, front)
                                exact_cov = coverage(
                                    no_extra_mask.points_xyz, no_extra_mask.classes, front
                                )
                                near = (
                                    (data[:, 0] >= front)
                                    & (data[:, 0] <= front + 6.15)
                                    & (abs(data[:, 1]) < 1.5)
                                    & (abs(data[:, 2]) < 0.08)
                                )
                                hits = ray_box_mask(
                                    np.array(sweep.sensor_origin_ego_m),
                                    data[near, :3],
                                    np.array([rear, -width, bottom]),
                                    np.array([front, width, top]),
                                )
                                batch = extract_obstacles(sweep, route, policy=policy, body=body)
                                result["frames"].append(
                                    dict(
                                        frame_id=frame,
                                        sim_time_s=sweep.sim_time_s,
                                        distance_m=distance,
                                        expected_target=dict(center=centre, extent=extent),
                                        point_count=len(data),
                                        density=sweep.density_fraction,
                                        known_object=check,
                                        raw_road=raw_cov,
                                        current_filter=filtered,
                                        exact_ego_mask=exact_cov,
                                        possible_self_occluded_near_road_rays=int(hits.sum()),
                                        checked_near_road_rays=int(near.sum()),
                                        valid=batch.valid,
                                        nearest_gap_m=batch.nearest_gap_m,
                                    )
                                )
                        result["capture_records"] = capture.records
                        result["callback_errors"] = capture.callback_errors
                        result["status"] = "CAPTURED"
                except Exception as error:
                    result["error"] = f"{type(error).__name__}: {error}"
                    document["failures"].append(dict(candidate=name, error=result["error"]))
                (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                for _ in range(2):
                    owner.tick()
                (output / "progress.json").write_text(json.dumps(document, indent=2) + "\n")
        document["status"] = "CAPTURED"
    finally:
        document["cleanup"] = verify_cleanup(w, owned, before)
        (output / "screen.json").write_text(json.dumps(document, indent=2) + "\n")
    for candidate in document["candidates"]:
        frames = [f for f in candidate["frames"] if f["distance_m"] == 30]
        print(
            candidate["name"],
            "checks",
            sum(f["known_object"]["passed"] for f in candidate["frames"]),
            "/",
            len(candidate["frames"]),
            "raw",
            [(f["raw_road"]["road_supported_cells"], f["raw_road"]["cells"]) for f in frames],
            "default",
            [
                (f["current_filter"]["road_supported_cells"], f["current_filter"]["cells"])
                for f in frames
            ],
            "exactmask",
            [
                (f["exact_ego_mask"]["road_supported_cells"], f["exact_ego_mask"]["cells"])
                for f in frames
            ],
            "self-rays",
            [f["possible_self_occluded_near_road_rays"] for f in frames],
        )
    print("cleanup", document["cleanup"])
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="172.28.176.1")
    parser.add_argument("--site", type=int, default=22)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen(args)
