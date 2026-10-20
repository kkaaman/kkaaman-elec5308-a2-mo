"""Capture real tracker outputs on scripted RGB-D approaches; labels are evaluation only.

Uses the existing synchronous owner and aligned RGB-D path. The ego stays parked;
targets are moved kinematically. This is detector evidence, not stopping acceptance.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from autonomy.interfaces import EgoPose, FrameBundle
from carla_adapter.sensors import CameraRig, RGBDCapture
from carla_adapter.vehicle import FixedVehicleAdapter
from carla_adapter.world import SynchronousWorldSession, WorldTiming, _settings_record
from geometry.camera import CameraIntrinsics
from obstacle_perception.box_depth import estimate_box_depth
from obstacle_perception.tracking import TrackDepthFilter, TrackPolicy
from obstacle_perception.types import (
    CallTiming,
    DetectionBatch,
    DetectionObservation,
    ObservationMetadata,
)
from perception.recorded_detector import verify_local_asset

SCENARIOS = ("approach", "short_occlusion", "continuity_risk")
SENSOR_LAG_TICKS = 4


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_evidence(depth_path, braking_path, baseline_path, acceptance_path):
    paths = dict(
        depth=depth_path, braking=braking_path, baseline=baseline_path, acceptance=acceptance_path
    )
    docs = {key: json.loads(Path(path).read_text()) for key, path in paths.items()}
    depth, braking = docs["depth"], docs["braking"]
    if depth["status"] != "MEASURED_PASS" or depth["convention"] not in ("axial", "radial"):
        raise ValueError("Passing measured depth convention required")
    if braking["status"] != "MEASURED_PASS" or braking.get("error"):
        raise ValueError("Passing measured longitudinal evidence required")
    if not braking["settings_restored"] or not braking["no_actor_leaks"]:
        raise ValueError("Longitudinal cleanup was not verified")
    for key in ("deceleration_mps2", "response_budget_s"):
        if not math.isfinite(braking[key]) or braking[key] <= 0:
            raise ValueError("Invalid measured braking budget")
    rig = CameraRig(**depth["rig"])
    return (
        docs,
        rig,
        {
            key: dict(path=str(Path(path).resolve()), sha256=sha256(path))
            for key, path in paths.items()
        },
    )


def validate_environment(client_version, server, map_name, map_hash, rig, depth, braking):
    if client_version != "0.9.16" or server != "0.9.16":
        raise ValueError("Matching CARLA 0.9.16 client/server required")
    if map_name.split("/")[-1] != "Town10HD_Opt":
        raise ValueError("Town10HD_Opt required")
    if depth["server"] != server or depth["map"].split("/")[-1] != map_name.split("/")[-1]:
        raise ValueError("Depth evidence build/map differs")
    if depth["opendrive_sha256"] != map_hash or depth["rig"] != asdict(rig):
        raise ValueError("Depth evidence map geometry/rig differs")
    if braking["server"] != server or braking["map"].split("/")[-1] != map_name.split("/")[-1]:
        raise ValueError("Braking evidence build/map differs")


def projected_target(vertices, camera_inverse, intrinsics):
    """Evaluation only: CARLA world bounding vertices -> clipped image box and Z interval."""
    points = np.column_stack((np.asarray(vertices), np.ones(len(vertices))))
    local = (np.asarray(camera_inverse) @ points.T).T[:, :3]
    if np.any(local[:, 0] <= 0):
        return None
    u = intrinsics.fx_px * local[:, 1] / local[:, 0] + intrinsics.cx_px
    v = intrinsics.cy_px - intrinsics.fy_px * local[:, 2] / local[:, 0]
    box = [
        max(0.0, float(u.min())),
        max(0.0, float(v.min())),
        min(float(intrinsics.width), float(u.max())),
        min(float(intrinsics.height), float(v.max())),
    ]
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    return dict(
        xyxy=box, axial_min_m=float(local[:, 0].min()), axial_max_m=float(local[:, 0].max())
    )


def iou(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def target_indices(detections, depth, calibration, reference):
    """Evaluation-only geometric match with depth gating to reject a nearer occluder.

    Ambiguous duplicate matches stay unlabelled. No geometry is sent to inference,
    temporal filtering or vehicle control.
    """
    if reference is None:
        return []
    matches = []
    for index, d in enumerate(detections):
        box = d.bounding_box
        if d.class_name not in ("car", "truck", "bus"):
            continue
        if iou([box.x_min, box.y_min, box.x_max, box.y_max], reference["xyxy"]) < 0.3:
            continue
        estimate = estimate_box_depth(depth, box, calibration)
        if estimate.valid and (
            reference["axial_min_m"] - 0.5
            <= estimate.axial_depth_m
            <= reference["axial_max_m"] + 0.5
        ):
            matches.append(index)
    return matches if len(matches) == 1 else []


def serialise_frame(
    frame_id,
    sim_time_s,
    detections,
    timing,
    *,
    valid,
    reason,
    gap,
    labels,
    depth_file,
    rgb_file,
    filter_wall_s,
    sensor_delay_s,
):
    if valid and timing is None:
        raise ValueError("Valid inference requires actual detector timing")
    if not valid and (not reason or detections or labels):
        raise ValueError("Invalid observations require reason and no detections/labels")
    return dict(
        frame_id=frame_id,
        sim_time_s=sim_time_s,
        total_wall_s=None if timing is None else timing.total_wall_s,
        filter_wall_s=filter_wall_s,
        sensor_delay_s=sensor_delay_s,
        depth_npy=depth_file,
        rgb_npy=rgb_file,
        valid=valid,
        invalid_reason=reason,
        target_gap_m=gap,
        target_detection_indices=labels,
        detections=[
            dict(
                class_name=d.class_name,
                confidence=d.confidence,
                xyxy=[
                    d.bounding_box.x_min,
                    d.bounding_box.y_min,
                    d.bounding_box.x_max,
                    d.bounding_box.y_max,
                ],
                track_id=d.track_id,
            )
            for d in detections
        ],
    )


def capture_observation(
    capture, detector, temporal, metadata, ego_pose, calibration, rig, convention
):
    """Advance every attempted source frame, including failed RGB-D reads."""
    detections, timing, bundle = (), None, None
    reason = None
    rgb = None
    axial = np.full((rig.height, rig.width), np.nan, np.float32)
    try:
        aligned = capture.read(metadata.frame_id, 5)
        bundle = aligned.to_bundle(
            calibration,
            established_depth_convention=convention,
        )
        axial, rgb = bundle.depth_m, bundle.rgb
        detections = detector.track(rgb)
        timing = detector.last_call_timing
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if bundle is not None:
            timing = detector.last_call_timing
        detections = ()
    frame = bundle
    if frame is None:
        # Shape-only carrier for an invalid batch; never detector input or valid depth.
        frame = FrameBundle(
            metadata.episode_id,
            metadata.frame_id,
            metadata.sim_time_s,
            0,
            ego_pose,
            np.zeros((rig.height, rig.width, 3), np.uint8),
            calibration,
        )
    batch = DetectionBatch(
        metadata,
        tuple(
            DetectionObservation(d.class_name, d.confidence, d.bounding_box, d.track_id)
            for d in detections
        ),
        rig.width,
        rig.height,
        timing or CallTiming(0),
        reason is None,
        reason,
    )
    started = time.perf_counter()
    result = temporal.update(frame, batch)
    filter_wall = time.perf_counter() - started
    return detections, timing, bundle, reason, rgb, axial, filter_wall, result


def straight_site(map_):
    """Reuse BL-E-03's checked straight-site selection, without importing its collector."""
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


def world_vertices(actor):
    return [[v.x, v.y, v.z] for v in actor.bounding_box.get_world_vertices(actor.get_transform())]


def target_pose(carla, waypoint, distance, lateral):
    options = waypoint.next(distance)
    if len(options) != 1 or options[0].is_junction:
        raise RuntimeError("Target left checked straight road")
    pose = options[0].transform
    yaw = math.radians(pose.rotation.yaw)
    pose.location.x -= math.sin(yaw) * lateral
    pose.location.y += math.cos(yaw) * lateral
    pose.location.z += 0.05
    return carla.Transform(pose.location, pose.rotation)


def collect(args):
    import carla

    # Avoid Ultralytics silently installing tracker dependencies during a timed call.
    import lap
    import yaml

    from perception.object_detection import ObjectDetector, _resolve_tracker
    from perception.types import PerceptionConfig

    if not lap.__version__:
        raise RuntimeError("Install lap before capturing")
    root = Path(__file__).resolve().parents[1]
    docs, rig, provenance = read_evidence(
        args.depth_evidence,
        args.braking_evidence,
        root / "configs/mission_baseline.json",
        root / "configs/acceptance.json",
    )
    baseline, braking, depth_evidence = docs["baseline"], docs["braking"], docs["depth"]
    manifest_path = root / "configs/models.yaml"
    model_spec = json.loads(manifest_path.read_text())["models"][0]
    asset = verify_local_asset(root / model_spec["path"], model_spec["sha256"])
    config = PerceptionConfig(model_name=str(asset))
    tracker_path = Path(_resolve_tracker(config.tracker_config))
    provenance["models_manifest"] = dict(path=str(manifest_path), sha256=sha256(manifest_path))
    provenance["tracker"] = dict(path=str(tracker_path), sha256=sha256(tracker_path))
    client = carla.Client(args.host, args.port)
    client.set_timeout(20)
    world = client.get_world()
    map_ = world.get_map()
    map_hash = hashlib.sha256(map_.to_opendrive().replace("\r\n", "\n").encode()).hexdigest()
    validate_environment(
        client.get_client_version(),
        client.get_server_version(),
        map_.name,
        map_hash,
        rig,
        depth_evidence,
        braking,
    )
    actors_before = {a.id for a in world.get_actors()}
    if any(a.type_id.startswith(("vehicle.", "sensor.", "walker.")) for a in world.get_actors()):
        raise RuntimeError("Empty world required; stop other capture/driving clients first")
    if world.get_settings().synchronous_mode:
        raise RuntimeError("Another client owns synchronous ticks")
    before = _settings_record(world.get_settings())
    site, spawn, waypoint = straight_site(map_)
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    provenance["capture_script"] = dict(path=str(source), sha256=sha256(source))
    (args.output / "capture-script.py").write_bytes(source.read_bytes())
    for key, doc in docs.items():
        (args.output / f"evidence-{key}.json").write_text(json.dumps(doc, indent=2))
    policy = TrackPolicy()
    dt = baseline["runtime"]["fixed_delta_seconds"]
    document = dict(
        schema_version=2,
        origin="recorded",
        capture_status="INCOMPLETE",
        scope="Parked ego, kinematic target approach; real RGB-D and YOLO tracking; not closed-loop driving acceptance",
        settings=dict(
            confidence=config.confidence_threshold,
            image_size=config.image_size,
            allowed_classes=config.allowed_classes,
            tracker=config.tracker_config,
            tracker_settings=yaml.safe_load(tracker_path.read_text()),
            model_version=f"{model_spec['name']}:sha256:{model_spec['sha256']}",
            model_sha256=model_spec["sha256"],
            device="UNINITIALISED",
            versions={
                name: importlib.metadata.version(name)
                for name in ("ultralytics", "torch", "numpy", "opencv-python", "lap")
            },
            capture_resolution_wh=[rig.width, rig.height],
        ),
        track_policy=asdict(policy),
        proposed_cruise_mps=baseline["cruise_speed_mps"],
        braking_deceleration_mps2=braking["deceleration_mps2"],
        additional_response_s=braking["response_budget_s"],
        stopping_margin_m=docs["acceptance"]["margin_max_m"],
        provenance=provenance,
        environment=dict(
            server=client.get_server_version(),
            client=client.get_client_version(),
            map=map_.name,
            opendrive_sha256=map_hash,
            rig=asdict(rig),
            site=site,
            fixed_delta_seconds=dt,
            depth_convention=depth_evidence["convention"],
            weather=str(world.get_weather()),
        ),
        label_policy=dict(
            iou_min=0.3,
            depth_tolerance_m=0.5,
            ambiguous_matches="unlabelled",
            ground_truth_use="evaluation_only",
        ),
        scenario_settings=dict(
            frames=args.frames,
            start_distance_m=50.0,
            relative_speed_mps=baseline["cruise_speed_mps"],
            short_occlusion_indices=[60, 63],
            continuity_occlusion_indices=[80, 100],
            lateral_amplitude_m=0.8,
        ),
        sequences=[],
    )
    detector = ObjectDetector(config)
    owned_ids = []
    error = None
    try:
        with SynchronousWorldSession(
            client, WorldTiming(fixed_delta_seconds=dt), ownership_key=f"{args.host}:{args.port}"
        ) as owner:
            settings = world.get_settings()
            settings.no_rendering_mode = False
            world.apply_settings(settings)
            for scenario in SCENARIOS:
                episode = f"{scenario}-{uuid.uuid4()}"
                directory = args.output / scenario
                directory.mkdir()
                ego = owner.try_spawn(
                    world.get_blueprint_library().find("vehicle.tesla.model3"), spawn
                )
                if ego is None:
                    raise RuntimeError("Ego spawn failed")
                owned_ids.append(ego.id)
                ego.apply_control(carla.VehicleControl(brake=1))
                for _ in range(20):
                    owner.tick()
                adapter = FixedVehicleAdapter(ego)
                ego.set_simulate_physics(False)
                target = owner.try_spawn(
                    world.get_blueprint_library().find("vehicle.tesla.model3"),
                    target_pose(carla, waypoint, 50, 0),
                )
                if target is None:
                    raise RuntimeError("Target spawn failed")
                owned_ids.append(target.id)
                target.set_simulate_physics(False)
                # A tall bus is a physical occluder, moved off-axis outside its scheduled interval.
                occluder = None
                if scenario != "approach":
                    bus_spawn = target_pose(carla, waypoint, 12, 0)
                    bus_spawn.location.z += 2
                    occluder = owner.try_spawn(
                        world.get_blueprint_library().find("vehicle.mitsubishi.fusorosa"),
                        bus_spawn,
                    )
                    if occluder is None:
                        raise RuntimeError("Occluder spawn failed")
                    owned_ids.append(occluder.id)
                    occluder.set_simulate_physics(False)
                    occluder.set_transform(target_pose(carla, waypoint, 12, 7))
                intrinsics = CameraIntrinsics.from_horizontal_fov(
                    rig.width, rig.height, math.radians(rig.fov_deg)
                )
                matrix = np.array(
                    [
                        [0, 0, 1, rig.x_actor_m + adapter.geometry.rear_axle_offset_m],
                        [-1, 0, 0, 0],
                        [0, -1, 0, rig.z_actor_m],
                        [0, 0, 0, 1],
                    ],
                    float,
                )
                calibration = intrinsics.calibration(matrix)
                detector.reset(episode_id=episode)
                temporal = TrackDepthFilter(
                    source_id="front-rgbd",
                    model_version=document["settings"]["model_version"],
                    policy=policy,
                )
                temporal.reset(episode_id=episode)
                sequence = dict(
                    name=scenario,
                    episode_id=episode,
                    source_id="front-rgbd",
                    calibration=asdict(calibration),
                    start_sim_time_s=None,
                    frames=[],
                )
                document["sequences"].append(sequence)
                pending = []
                with RGBDCapture(owner, ego, episode, rig=rig, capacity=16) as capture:
                    owned_ids.extend(sensor.id for sensor in capture.sensors)
                    sequence["actor_ids"] = dict(
                        ego=ego.id,
                        target=target.id,
                        occluder=None if occluder is None else occluder.id,
                    )
                    for _ in range(12):
                        frame_id = owner.tick()
                        capture.record_snapshot(
                            frame_id, EgoPose(adapter.state(), ego.get_transform().location.z)
                        )
                    for index in range(args.frames + SENSOR_LAG_TICKS):
                        if index < args.frames:
                            distance = 50 - index * dt * baseline["cruise_speed_mps"]
                            lateral = (
                                0.8 * math.sin(index * 0.15)
                                if scenario == "continuity_risk"
                                else 0.0
                            )
                            target.set_transform(target_pose(carla, waypoint, distance, lateral))
                            hidden = (
                                60 <= index < 63
                                if scenario == "short_occlusion"
                                else 80 <= index < 100
                                if scenario == "continuity_risk"
                                else False
                            )
                            if occluder:
                                occluder.set_transform(
                                    target_pose(carla, waypoint, 12, 0 if hidden else 7)
                                )
                        frame_id = owner.tick()
                        capture.record_snapshot(
                            frame_id, EgoPose(adapter.state(), ego.get_transform().location.z)
                        )
                        now = world.get_snapshot().timestamp.elapsed_seconds
                        if index < args.frames:
                            if sequence["start_sim_time_s"] is None:
                                sequence["start_sim_time_s"] = now
                            camera_inverse = capture.sensors[0].get_transform().get_inverse_matrix()
                            reference = projected_target(
                                world_vertices(target), camera_inverse, intrinsics
                            )
                            forward = np.array(
                                [
                                    ego.get_transform().get_forward_vector().x,
                                    ego.get_transform().get_forward_vector().y,
                                    ego.get_transform().get_forward_vector().z,
                                ]
                            )
                            gap = float(
                                np.min(np.asarray(world_vertices(target)) @ forward)
                                - np.max(np.asarray(world_vertices(ego)) @ forward)
                            )
                            pending.append(
                                dict(
                                    frame_id=frame_id,
                                    sim_time_s=now,
                                    gap=gap,
                                    reference=reference,
                                    hidden=hidden,
                                    ego_pose=EgoPose(
                                        adapter.state(), ego.get_transform().location.z
                                    ),
                                )
                            )
                        if index < SENSOR_LAG_TICKS:
                            continue
                        snap = pending[index - SENSOR_LAG_TICKS]
                        metadata = ObservationMetadata(
                            episode,
                            snap["frame_id"],
                            snap["sim_time_s"],
                            "front-rgbd",
                            document["settings"]["model_version"],
                        )
                        detections, timing, bundle, reason, rgb, axial, filter_wall, _ = (
                            capture_observation(
                                capture,
                                detector,
                                temporal,
                                metadata,
                                snap["ego_pose"],
                                calibration,
                                rig,
                                depth_evidence["convention"],
                            )
                        )
                        depth_file = f"{scenario}/{snap['frame_id']}-depth.npy"
                        rgb_file = (
                            f"{scenario}/{snap['frame_id']}-rgb.npy" if rgb is not None else None
                        )
                        np.save(args.output / depth_file, axial, allow_pickle=False)
                        if rgb_file:
                            np.save(args.output / rgb_file, rgb, allow_pickle=False)
                        # Evaluation starts only after detector and filter calls have completed.
                        labels = target_indices(detections, axial, calibration, snap["reference"])
                        row = serialise_frame(
                            snap["frame_id"],
                            snap["sim_time_s"],
                            detections,
                            timing,
                            valid=reason is None,
                            reason=reason,
                            gap=snap["gap"],
                            labels=labels,
                            depth_file=depth_file,
                            rgb_file=rgb_file,
                            filter_wall_s=filter_wall,
                            sensor_delay_s=now - snap["sim_time_s"],
                        )
                        row["evaluation"] = dict(
                            target_box=snap["reference"], scripted_occlusion=snap["hidden"]
                        )
                        row["depth_observation_valid"] = bundle is not None
                        row["depth_sha256"] = sha256(args.output / depth_file)
                        if rgb_file:
                            row["rgb_sha256"] = sha256(args.output / rgb_file)
                        sequence["frames"].append(row)
                        with (directory / "frames.jsonl").open("a") as stream:
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                        predictor = getattr(detector.model, "predictor", None)
                        if predictor is not None:
                            document["settings"]["device"] = str(predictor.device)
                        if len(sequence["frames"]) % 40 == 0:
                            print(
                                f"{scenario}: {len(sequence['frames'])}/{args.frames}", flush=True
                            )
                    sequence["queue_events"] = capture.joiner.stats()
                    sequence["callback_errors"] = list(capture.callback_errors)
                    if capture.callback_errors:
                        raise RuntimeError("Sensor callback errors; see retained manifest")
                for actor in (occluder, target, ego):
                    if actor is not None and actor.is_alive and not actor.destroy():
                        raise RuntimeError("Scenario actor destruction failed")
            document["capture_status"] = "COMPLETE"
    except BaseException as exc:
        error = exc
        document["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        detector.close()
        # CARLA actor proxy liveness can lag destruction/physics updates.
        # Retry only IDs created by this invocation, never unrelated actors.
        retry = [identity for identity in owned_ids if world.get_actor(identity) is not None]
        document["owned_actor_ids"] = owned_ids
        document["cleanup_retries"] = []
        if retry:
            responses = client.apply_batch_sync(
                [carla.command.DestroyActor(identity) for identity in retry], False
            )
            document["cleanup_retries"] = [
                dict(actor_id=identity, error=response.error)
                for identity, response in zip(retry, responses)
            ]
        world.wait_for_tick(5)
        document["settings_restored"] = _settings_record(world.get_settings()) == before
        document["no_actor_leaks"] = {a.id for a in world.get_actors()} == actors_before
        if not document["settings_restored"] or not document["no_actor_leaks"]:
            document["capture_status"] = "INCOMPLETE"
        (args.output / "sequences.json").write_text(
            json.dumps(document, indent=2, allow_nan=False) + "\n"
        )
    print(
        json.dumps(
            {
                key: document[key]
                for key in ("capture_status", "settings_restored", "no_actor_leaks")
            }
        ),
        flush=True,
    )
    if error is not None:
        raise error
    if document["capture_status"] != "COMPLETE":
        raise RuntimeError("Capture incomplete; inspect retained manifest")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="172.28.176.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument(
        "--depth-evidence",
        type=Path,
        default=Path("runs/BL-B-04/depth-calibration/depth_calibration.json"),
    )
    parser.add_argument(
        "--braking-evidence", type=Path, default=Path("runs/BL-E-03/longitudinal/longitudinal.json")
    )
    args = parser.parse_args()
    if not 110 <= args.frames <= 210:
        parser.error(
            "--frames must be 110..210 to include occlusions and keep target travel bounded"
        )
    collect(args)


if __name__ == "__main__":
    main()
