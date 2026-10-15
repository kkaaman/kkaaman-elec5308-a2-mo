"""Observational TEST-01 recorder around the inherited A1 demo."""

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import platform
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

from evaluation.a1_capture import (
    CaptureLog,
    signed_lane_cte,
    verify_sources,
    write_json,
)


def _command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as error:
        return {"returncode": 127, "stdout": "", "stderr": str(error)}
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _asset(path, expected=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Local weights missing or empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        header = stream.read(128)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Git LFS pointer instead of weights: {path}")
        digest.update(header)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f"Weights checksum mismatch: {path.name}")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": actual}


def _scalar_attributes(obj):
    result = {}
    for name in dir(obj):
        if not name.startswith("_"):
            value = getattr(obj, name)
            if value is None or isinstance(value, (str, bool, int, float)):
                result[name] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke-seconds", type=float)
    args = parser.parse_args()
    root = args.repo.resolve()
    if args.smoke_seconds is not None and (
        not math.isfinite(args.smoke_seconds) or not 0 < args.smoke_seconds <= 120
    ):
        parser.error("Smoke duration must lie in (0, 120] seconds")
    sources = json.loads(
        (root / "configs/test01_a1_sources.json").read_text(encoding="utf-8-sig")
    )
    verify_sources(root, sources)
    assets = {
        "lane": _asset(
            root / "lane_unet_best.pth",
            "10dca01621fb8e110f1d5b8419e81fb616a8ce1b75d6aface11f5793af88e8ff",
        ),
        "segmentation": _asset(
            root / "segmentation_model.pth",
            "a631440d3fe2ccd2554631bb49b1352c0efbbaffc2316caa79c288757e8dde2a",
        ),
        "detector": _asset(args.detector_weights),
    }
    if Path(assets["detector"]["path"]).name != "yolo26n.pt":
        parser.error(
            "Use the A1 yolo26n.pt asset; do not substitute a different detector"
        )

    import carla
    import cv2
    import torch

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60)
    world = client.get_world()
    road_map = world.get_map()
    if road_map.name.split("/")[-1] != "Town10HD_Opt":
        parser.error("Open Town10HD_Opt first; recorder does not load another map")
    initial_settings = _scalar_attributes(world.get_settings())
    if world.get_settings().synchronous_mode:
        parser.error(
            "A1 comparison requires the inherited asynchronous world configuration"
        )
    if any(a.type_id.startswith(("vehicle.", "sensor.")) for a in world.get_actors()):
        parser.error(
            "Existing vehicles/sensors found. Use an isolated world; nothing was destroyed."
        )
    owned_actors = []
    preflight = {
        "status": "PREFLIGHT_PASS",
        "server": client.get_server_version(),
        "map": road_map.name,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "assets": assets,
        "source_files_verified": len(sources),
    }
    if args.check:
        print(json.dumps(preflight, indent=2))
        return
    if args.output is None:
        parser.error("--output is required for capture")
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    recorder = CaptureLog(folder)
    video_times = (folder / "video_frames.csv").open("w", newline="", encoding="utf-8")
    video_csv = csv.writer(video_times)
    video_csv.writerow(
        (
            "video_frame",
            "wall_epoch_s",
            "sim_frame",
            "sim_time_s",
            "latest_control_sample",
        )
    )
    old_argv = sys.argv[:]
    video = None
    video_frames = 0
    previous_result = None
    collisions = []
    vehicle_recorded = False
    quit_requested = False
    error_text = None
    cleanup_errors = []
    began = time.monotonic()
    original_imshow, original_waitkey = cv2.imshow, cv2.waitKey
    manifest = {
        "baseline": "A1 settings on pinned recovered A2 library source",
        "preflight": preflight,
        "run_kind": "SMOKE_ONLY"
        if args.smoke_seconds is not None
        else "FULL_A1_SETTINGS_CAPTURE",
        "git_head": _command(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "git_status": _command(["git", "-C", str(root), "status", "--short"]),
        "source_hashes": sources,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "gpu": _command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "world_settings": initial_settings,
        "weather": _scalar_attributes(world.get_weather()),
        "map_opendrive_sha256": hashlib.sha256(
            road_map.to_opendrive().encode()
        ).hexdigest(),
        "video": {
            "file": "video.avi",
            "codec": "MJPG",
            "nominal_fps": 10,
            "timeline": "processed display frames; use video_frames.csv for actual wall/simulation times",
        },
        "runtime_changes": [
            "absolute paths to identical local weights",
            "observational command/video logging",
            "failure snapshots redirected into run folder",
            "owned actor cleanup on exceptions",
        ],
        "limitations": [
            "A1 dependencies include recovered library changes; not a pristine A1 checkout",
            "asynchronous observations are not frame-aligned",
            "image source frame/time unavailable in inherited callback",
            "logging/video add wall-time overhead",
            "nearest-lane CTE is evaluation-only and may change lane identity",
            "0.22/1.14 m prompt figures are unverified supplied results",
        ],
    }
    write_json(folder / "manifest.json", manifest)
    (folder / "pip-freeze.txt").write_text(
        _command([sys.executable, "-m", "pip", "freeze"])["stdout"], encoding="utf-8"
    )
    with zipfile.ZipFile(
        folder / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in sources:
            archive.write(root / name, name)
        for name in (
            "src/carla_adapter/a1_capture.py",
            "src/evaluation/a1_capture.py",
            "configs/test01_a1_sources.json",
        ):
            archive.write(root / name, name)
    try:
        source = root / "src/demo_fused_lane_following.py"
        spec = importlib.util.spec_from_file_location("test01_inherited_a1", source)
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)

        class WorldProxy:
            def __getattr__(self, name):
                return getattr(world, name)

            def spawn_actor(self, blueprint, transform, **kwargs):
                actor = world.spawn_actor(blueprint, transform, **kwargs)
                owned_actors.append(actor)
                manifest.setdefault("spawned_actors", []).append(
                    {
                        "id": actor.id,
                        "blueprint": blueprint.id,
                        "requested_pose_carla": {
                            "x": transform.location.x,
                            "y": transform.location.y,
                            "z": transform.location.z,
                            "pitch_deg": transform.rotation.pitch,
                            "yaw_deg": transform.rotation.yaw,
                            "roll_deg": transform.rotation.roll,
                        },
                        "attributes": dict(actor.attributes),
                    }
                )
                write_json(folder / "manifest.json", manifest)
                return actor

        class ClientProxy:
            def __init__(self, *client_args, **kwargs):
                self.client = carla.Client(*client_args, **kwargs)

            def __getattr__(self, name):
                return getattr(self.client, name)

            def get_world(self):
                return WorldProxy()

            def load_world(self, *unused):
                raise RuntimeError("Recorder does not allow a map change")

        class CarlaFacade:
            Client = ClientProxy

            def __getattr__(self, name):
                return getattr(carla, name)

        demo.carla = CarlaFacade()
        demo.LANE_WEIGHTS = assets["lane"]["path"]
        demo.SEGMENTATION_WEIGHTS = assets["segmentation"]["path"]
        original_detector = demo.ObjectDetector
        demo.ObjectDetector = lambda config: original_detector(
            replace(config, model_name=assets["detector"]["path"])
        )
        if args.smoke_seconds is not None:
            demo.MAX_DURATION_S = args.smoke_seconds
            manifest["runtime_changes"].append(
                "shortened duration for SMOKE_ONLY; not final baseline"
            )
        manifest["demo_constants"] = {
            k: v
            for k, v in vars(demo).items()
            if k.isupper() and isinstance(v, (str, int, float, bool))
        }
        write_json(folder / "manifest.json", manifest)
        original_snapshot = demo._save_failure_snapshot
        demo._save_failure_snapshot = lambda rgb, mask, name: original_snapshot(
            rgb, mask, str(folder / Path(name).name)
        )
        original_apply = demo.apply_command

        def observe_apply(vehicle, command, steering_scale):
            nonlocal previous_result, collisions, vehicle_recorded
            local = inspect.currentframe().f_back.f_locals
            control = original_apply(vehicle, command, steering_scale)
            pose = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            snapshot = world.get_snapshot()
            waypoint = road_map.get_waypoint(
                pose.location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            cte = (
                None
                if waypoint is None
                else signed_lane_cte(
                    pose.location.x,
                    -pose.location.y,
                    waypoint.transform.location.x,
                    -waypoint.transform.location.y,
                    -math.radians(waypoint.transform.rotation.yaw),
                )
            )
            result = local.get("result")
            current = result is not None and result is not previous_result
            previous_result = result
            diagnostic = result.diagnostics if current else None
            collisions = local.get("collision_events", collisions)
            if not vehicle_recorded:
                physics = vehicle.get_physics_control()
                manifest["actual_vehicle"] = {
                    "blueprint": vehicle.type_id,
                    "first_recorded_pose": {
                        "x": pose.location.x,
                        "y": pose.location.y,
                        "z": pose.location.z,
                        "yaw_deg": pose.rotation.yaw,
                    },
                    "controller_config": asdict(local["config"]),
                    "steering_normalization_rad": steering_scale,
                    "wheels": [
                        {
                            "position": [w.position.x, w.position.y, w.position.z],
                            "max_steer_angle_deg": w.max_steer_angle,
                            "radius_raw": w.radius,
                        }
                        for w in physics.wheels
                    ],
                }
                write_json(folder / "manifest.json", manifest)
                vehicle_recorded = True
            recorder.append(
                {
                    "sample": len(recorder.rows),
                    "wall_epoch_s": time.time(),
                    "elapsed_wall_s": time.monotonic() - began,
                    "sim_frame": snapshot.frame,
                    "sim_time_s": snapshot.timestamp.elapsed_seconds,
                    "cycle_wall_before_apply_s": time.time() - local["cycle_start"],
                    "actual_dt_s": local["actual_dt_s"],
                    "mode": local["driving_mode"],
                    "lane_available": local.get("lane_path") is not None,
                    "misses": local["consecutive_misses"],
                    "actor_x_m": pose.location.x,
                    "actor_y_m": -pose.location.y,
                    "actor_z_m": pose.location.z,
                    "actor_yaw_rad": -math.radians(pose.rotation.yaw),
                    "speed_mps": math.sqrt(
                        velocity.x**2 + velocity.y**2 + velocity.z**2
                    ),
                    "nearest_lane_cte_signed_m": cte,
                    "nearest_lane_road_id": None
                    if waypoint is None
                    else waypoint.road_id,
                    "nearest_lane_id": None if waypoint is None else waypoint.lane_id,
                    "controller_cte_m": None
                    if diagnostic is None
                    else diagnostic.cross_track_error_m,
                    "controller_target_mps": None
                    if diagnostic is None
                    else diagnostic.internal_target_speed_mps,
                    "controller_diagnostic_current": current,
                    "requested_steering_rad": command.steering_angle_rad,
                    "throttle": command.throttle,
                    "brake": command.brake,
                    "applied_carla_steer": control.steer,
                    "applied_throttle": control.throttle,
                    "applied_brake": control.brake,
                    "traffic_light_stop": local["should_stop_for_light"],
                }
            )
            return control

        def observe_imshow(name, image):
            nonlocal video, video_frames
            if name == "Fused lane following":
                if video is None:
                    height, width = image.shape[:2]
                    video = cv2.VideoWriter(
                        str(folder / "video.avi"),
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        10,
                        (width, height),
                    )
                    if not video.isOpened():
                        raise RuntimeError("Video encoder could not open")
                video.write(image)
                snapshot = world.get_snapshot()
                video_csv.writerow(
                    (
                        video_frames,
                        time.time(),
                        snapshot.frame,
                        snapshot.timestamp.elapsed_seconds,
                        len(recorder.rows) - 1,
                    )
                )
                video_times.flush()
                video_frames += 1
            return original_imshow(name, image)

        def observe_waitkey(delay):
            nonlocal quit_requested
            key = original_waitkey(delay)
            if key & 0xFF == ord("q"):
                quit_requested = True
            return key

        demo.apply_command = observe_apply
        cv2.imshow, cv2.waitKey = observe_imshow, observe_waitkey
        sys.argv = [str(source)]
        demo.main()
    except BaseException:
        error_text = traceback.format_exc()
        (folder / "error.txt").write_text(error_text, encoding="utf-8")
    finally:
        sys.argv = old_argv
        cv2.imshow, cv2.waitKey = original_imshow, original_waitkey
        if video is not None:
            video.release()
        video_times.close()
        recorder.close()
        try:
            for actor in reversed(owned_actors):
                if actor.is_alive:
                    try:
                        if actor.type_id.startswith("vehicle."):
                            actor.apply_control(carla.VehicleControl(brake=1))
                        elif hasattr(actor, "stop"):
                            actor.stop()
                    except Exception as error:
                        cleanup_errors.append(f"stop actor {actor.id}: {error}")
                    try:
                        if not actor.destroy():
                            cleanup_errors.append(f"destroy returned false: {actor.id}")
                    except Exception as error:
                        cleanup_errors.append(f"destroy actor {actor.id}: {error}")
            if _scalar_attributes(world.get_settings()) != initial_settings:
                cleanup_errors.append("World settings changed unexpectedly")
        except Exception as error:
            cleanup_errors.append(str(error))
        readable_video = False
        if video_frames:
            reader = cv2.VideoCapture(str(folder / "video.avi"))
            readable_video, _ = reader.read()
            reader.release()
        report = recorder.summary()
        report.update(
            {
                "capture_status": "CAPTURED"
                if error_text is None
                and recorder.rows
                and readable_video
                and not cleanup_errors
                else "INCOMPLETE",
                "run_kind": manifest["run_kind"],
                "video_frames": video_frames,
                "video_readable": bool(readable_video),
                "quit_requested": quit_requested,
                "collision_events": [
                    {"wall_epoch_s": t, "other_actor_type": name}
                    for t, name in collisions
                ],
                "cleanup_errors": cleanup_errors,
                "error": error_text,
                "elapsed_wall_s_including_model_load": time.monotonic() - began,
                "cte_definition": "perpendicular distance to nearest driving-lane tangent at command application; actor origin; evaluation only",
                "controller_cte_definition": "A1 controller's current path diagnostic; local path can be replanned every cycle",
            }
        )
        write_json(folder / "summary.json", report)
        print(
            json.dumps(
                {
                    "capture_status": report["capture_status"],
                    "run_kind": report["run_kind"],
                    "samples": report["samples"],
                    "video_frames": video_frames,
                    "collisions": len(collisions),
                    "output": str(folder),
                },
                indent=2,
            )
        )
    if error_text is not None:
        print(error_text, file=sys.stderr)
    if report["capture_status"] != "CAPTURED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
