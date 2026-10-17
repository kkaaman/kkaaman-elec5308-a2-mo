"""Stationary, teleported training views; never an autonomy acceptance run."""

import argparse
import hashlib
import json
import math
import random
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.dataset import (
    KIT_VERSION,
    ROADLINE_TAG,
    decode_bgra,
    decode_depth,
    road_splits,
    save_json,
    sha256,
    visible_boxes,
)


class FrameBuffer:
    """Bounded per-sensor buffers with an exact-frame join."""

    def __init__(self, names, capacity=32):
        self.buffers = {name: {} for name in names}
        self.capacity = capacity
        self.condition = threading.Condition()
        self.evicted = Counter()

    def put(self, name, image):
        with self.condition:
            self.buffers[name][image.frame] = image
            while len(self.buffers[name]) > self.capacity:
                del self.buffers[name][min(self.buffers[name])]
                self.evicted[name] += 1
            self.condition.notify_all()

    def take(self, frame, timeout=15.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not all(frame in values for values in self.buffers.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Missing exact sensor frame {frame}")
                self.condition.wait(remaining)
            images = {name: values[frame] for name, values in self.buffers.items()}
            for values in self.buffers.values():
                for old in [value for value in values if value <= frame]:
                    del values[old]
        timestamps = [image.timestamp for image in images.values()]
        if max(timestamps) - min(timestamps) > 1e-6:
            raise ValueError("Sensor timestamps do not match")
        transforms = [np.asarray(image.transform.get_matrix()) for image in images.values()]
        if any(
            not np.allclose(transforms[0], matrix, atol=1e-4, rtol=0) for matrix in transforms[1:]
        ):
            raise ValueError("Camera transforms differ for joined frame")
        return images


def transform_dict(transform):
    return {
        "location_m": [transform.location.x, transform.location.y, transform.location.z],
        "rotation_deg_pitch_yaw_roll": [
            transform.rotation.pitch,
            transform.rotation.yaw,
            transform.rotation.roll,
        ],
    }


def location_for_target(carla, waypoint, distance, lateral):
    options = waypoint.next(distance)
    if len(options) != 1 or options[0].is_junction:
        return None
    transform = options[0].transform
    yaw = math.radians(transform.rotation.yaw)
    transform.location.x += -math.sin(yaw) * lateral
    transform.location.y += math.cos(yaw) * lateral
    transform.location.z += 0.65
    return carla.Transform(transform.location, transform.rotation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument(
        "--sequence-frames",
        type=int,
        default=20,
        help="Consecutive frames per stationary scene; kept in one split",
    )
    parser.add_argument(
        "--ray-reference",
        action="store_true",
        help="Evaluation-only geometry ray samples on the first frame of each scene",
    )
    parser.add_argument("--spacing-m", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=5308)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument(
        "--camera-z", type=float, default=2.0, help="Height above the Tesla actor origin, metres"
    )
    parser.add_argument("--camera-x", type=float, default=1.5)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    if (
        args.samples < 3
        or args.rounds < 1
        or not 1 <= args.sequence_frames <= 200
        or args.width < 32
        or args.height < 32
        or not 10 < args.fov < 170
        or not math.isfinite(args.spacing_m)
        or args.spacing_m < 2
        or not math.isfinite(args.camera_z)
        or args.camera_z <= 0
        or not math.isfinite(args.camera_x)
    ):
        parser.error("Invalid sample count or camera/spacing parameters")
    import carla

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    report = {
        "schema": KIT_VERSION,
        "status": "INCOMPLETE",
        "utc": datetime.now(timezone.utc).isoformat(),
        "label_tag": ROADLINE_TAG,
        "label_meaning": "CARLA RoadLine paint, including non-boundary markings; not lane centre or route",
        "sample_count": 0,
        "sequence_frames": args.sequence_frames,
        "image_size_wh": [args.width, args.height],
        "camera": {"fov_deg": args.fov, "x_actor_m": args.camera_x, "z_actor_m": args.camera_z},
        "depth_convention": "RAW_CARLA_ENCODED_UNVERIFIED",
        "scope": "Stationary RGB-D sequences with spatial road holdout; tracker initialization, not moving-target/driving acceptance",
        "settings_restored": False,
        "no_actor_leaks": False,
        "failure": None,
    }
    world = None
    original_settings = original_weather = original_view = None
    owned = []
    sensors = []
    failure = None
    counts = Counter()
    index_path = output / "samples.jsonl"
    try:
        report["server"] = client.get_server_version()
        report["client"] = client.get_client_version()
        if report["server"] != "0.9.16" or report["client"] != "0.9.16":
            raise RuntimeError(
                "This label mapping requires matching CARLA 0.9.16 client and server"
            )
        world = client.get_world()
        if world.get_settings().synchronous_mode:
            raise RuntimeError("Another client owns synchronous ticks; close that client first")
        busy = [
            actor
            for actor in world.get_actors()
            if actor.type_id.startswith(("vehicle.", "walker.", "sensor."))
        ]
        if busy:
            raise RuntimeError("Use an otherwise empty CARLA world for reproducible collection")
        carla_map = world.get_map()
        report["map"] = carla_map.name
        report["opendrive_sha256"] = hashlib.sha256(
            carla_map.to_opendrive().replace("\r\n", "\n").encode()
        ).hexdigest()
        if carla_map.name.split("/")[-1] not in ("Town10HD_Opt", "Town10HD"):
            raise RuntimeError("Load Town10HD_Opt before collecting this baseline dataset")
        points = [
            p
            for p in carla_map.generate_waypoints(args.spacing_m)
            if p.lane_type == carla.LaneType.Driving and not p.is_junction
        ]
        splits = road_splits([p.road_id for p in points])
        report["road_split"] = splits
        report["split_policy"] = (
            "Whole road IDs and whole run IDs. Never adjacent-frame random splitting. Same-town spatial holdout only."
        )
        report["rig"] = {
            "blueprint": "vehicle.tesla.model3",
            "mount_actor_m": [args.camera_x, 0.0, args.camera_z],
            "mount_rotation_deg": [0, 0, 0],
            "fov_deg": args.fov,
            "lens_k": 0,
            "lens_kcube": 0,
            "fixed_dt_s": 0.05,
            "pose_kind": "parked Tesla, physics disabled for scene collection",
        }
        rng = random.Random(args.seed)
        report["seed"] = args.seed
        groups = {
            split: [p for p in points if splits[p.road_id] == split]
            for split in ("train", "val", "test")
        }
        for values in groups.values():
            rng.shuffle(values)
        # Interleave the splits, so a small smoke capture contains each split.
        ordered = []
        for i in range(max(map(len, groups.values()))):
            for split in ("train", "val", "test"):
                if i < len(groups[split]):
                    ordered.append(groups[split][i])
        original_settings, original_weather, original_view = (
            world.get_settings(),
            world.get_weather(),
            world.get_spectator().get_transform(),
        )
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        bp_library = world.get_blueprint_library()
        ego_bp = bp_library.find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "group7_offline_training")
        ego = None
        for spawn in carla_map.get_spawn_points():
            ego = world.try_spawn_actor(ego_bp, spawn)
            if ego is not None:
                owned.append(ego)
                break
        if ego is None:
            raise RuntimeError("No free Tesla spawn point")
        ego.set_simulate_physics(False)
        frames = FrameBuffer(("rgb", "instance", "depth"))
        mount = carla.Transform(carla.Location(x=args.camera_x, z=args.camera_z))
        for name, blueprint in (
            ("rgb", "sensor.camera.rgb"),
            ("instance", "sensor.camera.instance_segmentation"),
            ("depth", "sensor.camera.depth"),
        ):
            bp = bp_library.find(blueprint)
            for key, value in {
                "image_size_x": args.width,
                "image_size_y": args.height,
                "fov": args.fov,
                "sensor_tick": 0,
                "lens_k": 0,
                "lens_kcube": 0,
            }.items():
                bp.set_attribute(key, str(value))
            if name == "rgb":
                bp.set_attribute("motion_blur_intensity", "0")
            sensor = world.spawn_actor(
                bp, mount, attach_to=ego, attachment_type=carla.AttachmentType.Rigid
            )
            owned.append(sensor)
            sensors.append(sensor)
            sensor.listen(lambda image, key=name: frames.put(key, image))
        weather_names = ("ClearNoon", "CloudyNoon", "WetNoon")
        spawned_target = None
        last_frame = None
        scene_counts = Counter()
        cases = [
            (name, distance, 0)
            for distance in (5, 10, 15, 20, 25, 30)
            for name in ("car", "person")
        ]
        cases += [("car", 15, 4), ("car", 25, -4), ("none", 0, 0), ("none", 0, 0)]
        with index_path.open("x", encoding="utf-8", buffering=1) as index:
            for round_index in range(args.rounds):
                weather_name = weather_names[round_index % len(weather_names)]
                world.set_weather(getattr(carla.WeatherParameters, weather_name))
                for point in ordered:
                    if report["sample_count"] >= args.samples:
                        break
                    if spawned_target is not None:
                        if not spawned_target.destroy():
                            raise RuntimeError("Previous target could not be removed")
                        owned.remove(spawned_target)
                        spawned_target = None
                    pose = point.transform
                    pose.location.z += (
                        0.02 - ego.bounding_box.location.z + ego.bounding_box.extent.z
                    )
                    pose.rotation.yaw += rng.uniform(-4, 4)
                    ego.set_transform(pose)
                    sample_number = report["sample_count"]
                    scene_split = splits[point.road_id]
                    target_kind, desired_surface_range, lateral = cases[
                        scene_counts[scene_split] % len(cases)
                    ]
                    scene_counts[scene_split] += 1
                    target_metadata = {"kind_requested": target_kind, "spawned": False}
                    if target_kind != "none":
                        distance = (
                            desired_surface_range
                            + args.camera_x
                            + (2.4 if target_kind == "car" else 0.4)
                        )
                        target_pose = location_for_target(carla, point, distance, lateral)
                        target_metadata.update(
                            requested_route_distance_m=distance,
                            requested_surface_range_m=desired_surface_range,
                            requested_lateral_m=lateral,
                        )
                        if target_pose is not None:
                            ground = world.ground_projection(
                                carla.Location(
                                    x=target_pose.location.x,
                                    y=target_pose.location.y,
                                    z=target_pose.location.z + 5,
                                ),
                                10,
                            )
                            if ground is None:
                                raise RuntimeError("No ground at collected target site")
                            candidates = (
                                sorted(bp_library.filter("walker.pedestrian.*"), key=lambda b: b.id)
                                if target_kind == "person"
                                else [bp_library.find("vehicle.tesla.model3")]
                            )
                            bp = candidates[(round_index + sample_number) % len(candidates)]
                            if bp.has_attribute("is_invincible"):
                                bp.set_attribute("is_invincible", "true")
                            spawned_target = world.try_spawn_actor(bp, target_pose)
                            if spawned_target is not None:
                                owned.append(spawned_target)
                                spawned_target.set_simulate_physics(False)
                                target_pose.location.z = ground.location.z
                                target_pose.location.z += (
                                    0.02
                                    - spawned_target.bounding_box.location.z
                                    + spawned_target.bounding_box.extent.z
                                )
                                spawned_target.set_transform(target_pose)
                                target_metadata.update(
                                    spawned=True,
                                    actor_id=spawned_target.id,
                                    blueprint=spawned_target.type_id,
                                )
                    if args.follow:
                        world.get_spectator().set_transform(
                            carla.Transform(
                                carla.Location(
                                    pose.location.x, pose.location.y, pose.location.z + 15
                                ),
                                carla.Rotation(pitch=-85, yaw=pose.rotation.yaw),
                            )
                        )
                    snapshots = {}
                    # GPU cameras can arrive several frames late. All poses stay fixed here.
                    for _ in range(18):
                        frame = world.tick(15.0)
                        if last_frame is not None and frame != last_frame + 1:
                            raise RuntimeError("Unexpected tick from another client")
                        last_frame = frame
                        snapshots[frame] = world.get_snapshot()
                        time.sleep(0.005)
                    pose_number = report["sample_count"]
                    for sequence_index in range(args.sequence_frames):
                        if report["sample_count"] >= args.samples:
                            break
                        if sequence_index:
                            next_frame = world.tick(15.0)
                            if next_frame != last_frame + 1:
                                raise RuntimeError("Another client advanced the world")
                            last_frame = next_frame
                            snapshots[last_frame] = world.get_snapshot()
                            for old in list(snapshots):
                                if old < last_frame - 8:
                                    del snapshots[old]
                            time.sleep(0.005)
                        sample_number = report["sample_count"]
                        frame = last_frame - 4
                        images = frames.take(frame)
                        snapshot = snapshots[frame]
                        if (
                            snapshot.frame != frame
                            or abs(snapshot.timestamp.elapsed_seconds - images["rgb"].timestamp)
                            > 1e-5
                        ):
                            raise ValueError(
                                "Ego/evaluation snapshot and camera frame do not match"
                            )
                        if spawned_target is not None:
                            actor_snapshot = snapshot.find(spawned_target.id)
                            if actor_snapshot is None:
                                raise ValueError("Target missing from matched snapshot")
                            target_metadata["transform"] = transform_dict(
                                actor_snapshot.get_transform()
                            )
                            target_metadata["bounding_box_local"] = {
                                "centre_m": [
                                    spawned_target.bounding_box.location.x,
                                    spawned_target.bounding_box.location.y,
                                    spawned_target.bounding_box.location.z,
                                ],
                                "extent_m": [
                                    spawned_target.bounding_box.extent.x,
                                    spawned_target.bounding_box.extent.y,
                                    spawned_target.bounding_box.extent.z,
                                ],
                            }
                        split = splits[point.road_id]
                        run_id = f"{split}-road{point.road_id}-round{round_index:03d}"
                        folder = output / "data" / run_id
                        folder.mkdir(parents=True, exist_ok=True)
                        sample_id = f"sample_{sample_number:06d}"
                        raw = {
                            name: decode_bgra(image.raw_data, image.width, image.height)
                            for name, image in images.items()
                        }
                        rgb = raw["rgb"][:, :, [2, 1, 0]]
                        lane = (raw["instance"][:, :, 2] == ROADLINE_TAG).astype(np.uint8)
                        paths = {
                            kind: folder
                            / f"{sample_id}_{kind}.{('npy' if kind == 'depth' else 'png')}"
                            for kind in ("rgb", "lane", "instance", "depth")
                        }
                        Image.fromarray(rgb).save(paths["rgb"])
                        Image.fromarray(lane).save(paths["lane"])
                        Image.fromarray(raw["instance"][:, :, [2, 1, 0]]).save(paths["instance"])
                        np.save(paths["depth"], decode_depth(raw["depth"]), allow_pickle=False)
                        record = {
                            "sample_id": sample_id,
                            "sequence_id": f"{run_id}-pose{pose_number:06d}",
                            "sequence_index": sequence_index,
                            "run_id": run_id,
                            "split": split,
                            "road_id": point.road_id,
                            "lane_id": point.lane_id,
                            "road_s_m": point.s,
                            "frame_id": frame,
                            "sensor_frames": {name: image.frame for name, image in images.items()},
                            "sim_time_s": images["rgb"].timestamp,
                            "weather": weather_name,
                            "camera_world": transform_dict(images["rgb"].transform),
                            "ego_world": transform_dict(snapshot.find(ego.id).get_transform()),
                            "target_evaluation_only": target_metadata,
                            "visible_boxes": visible_boxes(raw["instance"]),
                            "lane_pixels": int(lane.sum()),
                            "sha256": {kind: sha256(path) for kind, path in paths.items()},
                            **{
                                kind: path.relative_to(output).as_posix()
                                for kind, path in paths.items()
                            },
                        }
                        record["surface_references"] = []
                        if args.ray_reference and sequence_index == 0:
                            from carla_adapter.depth_calibration import ray_reference

                            decoded = decode_depth(raw["depth"])
                            for box in record["visible_boxes"]:
                                x0, y0, x1, y1 = box["xyxy"]
                                for fraction in (0.3, 0.5, 0.7):
                                    u = min(args.width - 1, int(x0 + (x1 - x0) * fraction))
                                    v = min(args.height - 1, int((y0 + y1) / 2))
                                    tag = int(raw["instance"][v, u, 2])
                                    if tag not in (12, 14):
                                        continue
                                    hit = ray_reference(
                                        world,
                                        images["rgb"].transform,
                                        u,
                                        v,
                                        args.width,
                                        args.height,
                                        args.fov,
                                    )
                                    value = float(decoded[v, u])
                                    record["surface_references"].append(
                                        dict(
                                            u=u,
                                            v=v,
                                            class_name=box["class_name"],
                                            raw_depth_m=value if math.isfinite(value) else None,
                                            reference=hit,
                                        )
                                    )
                        index.write(json.dumps(record, allow_nan=False) + "\n")
                        report["sample_count"] += 1
                        counts[split] += 1
                        if report["sample_count"] % 25 == 0:
                            print(
                                f"Captured {report['sample_count']}/{args.samples}; splits {dict(counts)}",
                                flush=True,
                            )
                if report["sample_count"] >= args.samples:
                    break
        if report["sample_count"] < args.samples:
            raise RuntimeError(
                "Not enough mapped views; increase --rounds or reduce --samples explicitly"
            )
        report["status"] = "COLLECTED"
        report["requested_samples"] = args.samples
        report["queue_evictions"] = dict(frames.evicted)
    except BaseException as error:
        failure = error
        report["failure"] = f"{type(error).__name__}: {error}"
    finally:
        cleanup = []
        for sensor in sensors:
            try:
                sensor.stop()
            except Exception as error:
                cleanup.append(str(error))
        ids = [actor.id for actor in owned]
        for actor in reversed(owned):
            try:
                if actor.is_alive and not actor.destroy():
                    cleanup.append(f"Destroy failed: {actor.id}")
            except Exception as error:
                cleanup.append(str(error))
        if world is not None and original_settings is not None:
            try:
                world.set_weather(original_weather)
                world.get_spectator().set_transform(original_view)
                world.apply_settings(original_settings)
                current = world.get_settings()
                fields = (
                    "synchronous_mode",
                    "fixed_delta_seconds",
                    "substepping",
                    "max_substep_delta_time",
                    "max_substeps",
                    "no_rendering_mode",
                )
                report["settings_restored"] = all(
                    getattr(current, k) == getattr(original_settings, k) for k in fields
                )
                report["no_actor_leaks"] = not any(actor.id in ids for actor in world.get_actors())
            except Exception as error:
                cleanup.append(str(error))
        report["cleanup_errors"] = cleanup
        report["split_counts"] = dict(counts)
        if index_path.exists():
            report["samples_sha256"] = sha256(index_path)
        if cleanup or not report["settings_restored"] or not report["no_actor_leaks"]:
            report["status"] = "INCOMPLETE"
        save_json(output / "manifest.json", report)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "status",
                        "sample_count",
                        "split_counts",
                        "failure",
                        "settings_restored",
                        "no_actor_leaks",
                    )
                },
                indent=2,
            )
        )
        print("Saved:", output)
    if failure is not None:
        raise failure
    if report["status"] != "COLLECTED":
        raise RuntimeError("Collection or cleanup incomplete; see manifest.json")


if __name__ == "__main__":
    main()
