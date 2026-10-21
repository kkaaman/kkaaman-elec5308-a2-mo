"""Same-frame real operational approach evidence; actuators remain physically braked."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from autonomy.interfaces import EgoPose, FrameBundle, LaneGraph
from autonomy.obstacle_policy import CorridorObservation, TrackedGovernancePolicy
from autonomy.operational import operational_stack
from autonomy.stack import StackConfig
from carla_adapter.sensors import CameraRig, RGBDCapture
from carla_adapter.vehicle import FixedVehicleAdapter
from carla_adapter.world import SynchronousWorldSession, _settings_record
from geometry.camera import CameraIntrinsics
from navigation_control.types import ControlCommand
from obstacle_perception.box_depth import DepthPolicy, estimate_box_depth
from obstacle_perception.tracking import TrackDepthFilter, TrackPolicy
from perception.recorded_detector import verify_local_asset
from perception.types import Detection
from planning.speed_governor import GovernorConfig, SpeedGovernor
from scripts.capture_obstacle_tracks import projected_target, target_indices, world_vertices
from scripts.measure_operational_governance import Controller, digest
from scripts.review_operational_range import admission_failures


class RecordedFilter(TrackDepthFilter):
    def update(self, frame, batch):
        start = time.monotonic()
        self.last = super().update(frame, batch)
        self.duration = time.monotonic() - start
        return self.last


class RecordedGovernor(SpeedGovernor):
    def limit(self, **kwargs):
        start = time.monotonic()
        self.inputs = kwargs
        self.output = super().limit(**kwargs)
        self.duration = time.monotonic() - start
        return self.output


class RecordedController(Controller):
    def __call__(self, frame, target, dt):
        result = super().__call__(frame, target, dt)
        self.last = dict(target_mps=target, command=asdict(result))
        return result


def pose_along(carla, spawn, distance, lateral=0):
    f = spawn.get_forward_vector()
    yaw = math.radians(spawn.rotation.yaw)
    return carla.Transform(
        carla.Location(
            spawn.location.x + distance * f.x - lateral * math.sin(yaw),
            spawn.location.y + distance * f.y + lateral * math.cos(yaw),
            spawn.location.z + 0.05,
        ),
        spawn.rotation,
    )


def collect(args):
    import carla
    import yaml

    from perception.object_detection import ObjectDetector, _resolve_tracker
    from perception.types import PerceptionConfig

    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "configs/mission_baseline.json").read_text())
    profile = baseline["baseline_threshold_freeze"]
    evidence = root / "docs/evidence/BL-D-06/sources"
    calibration_doc = json.loads((evidence / "depth-calibration.json").read_text())
    braking = json.loads((evidence / "longitudinal.json").read_text())
    rig = CameraRig(**profile["camera_rig"])
    model = json.loads((root / "configs/models.yaml").read_text())["models"][0]
    asset = verify_local_asset(root / model["path"], model["sha256"])
    config = PerceptionConfig(model_name=str(asset))
    tracker = Path(_resolve_tracker(config.tracker_config))
    if rig != CameraRig(**calibration_doc["rig"]):
        raise ValueError("Rig differs from measured convention")
    client = carla.Client(args.host, 2000)
    client.set_timeout(20)
    world = client.get_world()
    map_ = world.get_map()
    map_hash = hashlib.sha256(map_.to_opendrive().replace("\r\n", "\n").encode()).hexdigest()
    if (
        client.get_server_version() != "0.9.16"
        or map_.name.split("/")[-1] != "Town10HD_Opt"
        or map_hash != calibration_doc["opendrive_sha256"]
    ):
        raise ValueError("Build/map mismatch")
    actors_before = {a.id for a in world.get_actors()}
    if any(a.type_id.startswith(("vehicle.", "sensor.", "walker.")) for a in world.get_actors()):
        raise RuntimeError("Empty world required")
    before = _settings_record(world.get_settings())
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [
        "scripts/capture_operational_approach.py",
        "scripts/measure_operational_governance.py",
        "scripts/capture_obstacle_tracks.py",
        "scripts/review_operational_range.py",
    ]
    sources += [
        "src/autonomy/" + n + ".py"
        for n in ("stack", "freshness", "operational", "obstacle_policy", "interfaces")
    ]
    sources += [
        "src/perception/frame_detector.py",
        "src/perception/object_detection.py",
        "src/obstacle_perception/tracking.py",
        "src/obstacle_perception/box_depth.py",
        "src/planning/speed_governor.py",
        "src/carla_adapter/sensors.py",
        "src/carla_adapter/vehicle.py",
        "src/carla_adapter/world.py",
        "configs/mission_baseline.json",
        "configs/stack.yaml",
        "configs/acceptance.json",
        "configs/models.yaml",
    ]
    hashes = {n: digest(root / n) for n in sources}
    for n in sources:
        dest = args.output / "source" / n
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((root / n).read_bytes())
    report = dict(
        schema_version=1,
        kind="operational_same_frame_approach",
        status="INCOMPLETE",
        preflight=args.preflight,
        server=client.get_server_version(),
        client=client.get_client_version(),
        map=map_.name,
        opendrive_sha256=map_hash,
        rig=asdict(rig),
        depth_convention="axial",
        weather=str(world.get_weather()),
        fixed_delta_seconds=0.05,
        source_hashes=hashes,
        operational_profile=profile,
        source_id="front-rgbd",
        model_version=model["sha256"],
        settings=dict(
            confidence=config.confidence_threshold,
            image_size=config.image_size,
            tracker=config.tracker_config,
            tracker_settings=yaml.safe_load(tracker.read_text()),
            tracker_sha256=digest(tracker),
            model_sha256=model["sha256"],
            versions={
                n: importlib.metadata.version(n)
                for n in ("ultralytics", "torch", "numpy", "opencv-python", "lap")
            },
        ),
        scope="Parked ego, actual gated stack/governor/PathFollower shadow outputs; all tracks retained, no evaluation truth in control; conservative straight-scene supported-point provider, no empty clearance claim",
        failure_injection="Explicit labelled dropout after real read, input depth withheld; ordinary stack missing-depth latch; explicit new-episode reset on next frame",
        cases=[],
    )
    backend = None
    error = None
    try:
        backend = ObjectDetector(config)
        with SynchronousWorldSession(client, ownership_key=f"{args.host}:2000") as owner:
            settings = world.get_settings()
            settings.no_rendering_mode = False
            world.apply_settings(settings)
            cases = (
                [(site, "background_preflight") for site in args.sites]
                if args.preflight
                else [
                    (args.sites[0], n)
                    for n in ("approach", "short_occlusion", "continuity_risk", "failure_recovery")
                ]
            )
            for site, scenario in cases:
                directory = args.output / f"{site}-{scenario}"
                directory.mkdir()
                spawn = map_.get_spawn_points()[site]
                waypoint = map_.get_waypoint(spawn.location)
                # Map spawn points are elevated for physics settling. Kinematic
                # actors must instead use the actual road elevation.
                road_spawn = carla.Transform(
                    carla.Location(
                        spawn.location.x, spawn.location.y, waypoint.transform.location.z
                    ),
                    spawn.rotation,
                )
                for distance in (15, 30, 50):
                    opts = waypoint.next(distance)
                    if (
                        len(opts) != 1
                        or opts[0].is_junction
                        or abs(
                            math.atan2(
                                math.sin(
                                    math.radians(
                                        opts[0].transform.rotation.yaw - spawn.rotation.yaw
                                    )
                                ),
                                math.cos(
                                    math.radians(
                                        opts[0].transform.rotation.yaw - spawn.rotation.yaw
                                    )
                                ),
                            )
                        )
                        > math.radians(2)
                    ):
                        raise ValueError("Not checked straight 50m site")
                ego = owner.try_spawn(
                    world.get_blueprint_library().find("vehicle.tesla.model3"), spawn
                )
                if ego is None:
                    raise RuntimeError("Ego spawn failed")
                ego.apply_control(carla.VehicleControl(brake=1))
                for _ in range(20):
                    owner.tick()
                adapter = FixedVehicleAdapter(ego)
                origin = adapter.state()
                ego.set_simulate_physics(False)
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
                front = (
                    ego.bounding_box.location.x
                    + ego.bounding_box.extent.x
                    + adapter.geometry.rear_axle_offset_m
                )
                target = bus = None
                if not args.preflight:
                    target = owner.try_spawn(
                        world.get_blueprint_library().find("vehicle.tesla.model3"),
                        pose_along(carla, road_spawn, 50),
                    )
                    if target is None:
                        raise RuntimeError("Target spawn failed")
                    target.set_simulate_physics(False)
                    if scenario in ("short_occlusion", "continuity_risk"):
                        bus = owner.try_spawn(
                            world.get_blueprint_library().find("vehicle.mitsubishi.fusorosa"),
                            pose_along(carla, road_spawn, -20),
                        )
                        if bus is None:
                            raise RuntimeError("Occluder spawn failed")
                        bus.set_simulate_physics(False)
                controller = RecordedController(baseline["controller"], origin)
                temporal = RecordedFilter(
                    source_id="front-rgbd",
                    model_version=model["sha256"],
                    policy=TrackPolicy(**profile["track_policy"]),
                    depth_policy=DepthPolicy(**profile["depth_policy"]),
                )
                values = profile["candidate_values"]
                impl = profile["implementation_policy"]
                governor = RecordedGovernor(
                    GovernorConfig(
                        values["cruise_speed_mps"],
                        braking["deceleration_mps2"],
                        impl["lateral_acceleration_mps2"],
                        values["response_budget_s"],
                        values["stop_margin_m"],
                        impl["acceleration_mps2"],
                        impl["emergency_margin_m"],
                    )
                )
                state = {}

                def provider(frame, batch, tracks):
                    t = time.monotonic()
                    points = []
                    for track in tracks.tracks:
                        if track.current and track.depth_valid:
                            estimate = estimate_box_depth(
                                frame.depth_m,
                                track.detection.box,
                                frame.calibration,
                                temporal.depth_policy,
                            )
                            if estimate.valid:
                                points.append(
                                    dict(
                                        track_id=track.identity.track_id,
                                        point_rear_axle_m=list(estimate.point_ego_m),
                                        bumper_gap_m=float(estimate.point_ego_m[0] - front),
                                    )
                                )
                    gap = min((p["bumper_gap_m"] for p in points), default=None)
                    obs = CorridorObservation(batch.metadata, bool(points), gap, gap, False)
                    state.update(
                        provider=dict(
                            metadata=asdict(batch.metadata),
                            valid=obs.valid,
                            obstacle_gap_m=gap,
                            visible_distance_m=gap,
                            clearance_valid=False,
                            points=points,
                            front_offset_applied_once_m=front,
                            route_reference_world_xy=[
                                [origin.x_m, origin.y_m],
                                [
                                    origin.x_m + 100 * math.cos(origin.yaw_rad),
                                    origin.y_m + 100 * math.sin(origin.yaw_rad),
                                ],
                            ],
                            corridor_exclusion="none: all tracks retained conservatively",
                        ),
                        provider_s=time.monotonic() - t,
                        policy_failures=admission_failures(
                            tracks,
                            obs,
                            ready=stack.freshness_gate.fresh_results >= temporal.policy.hits,
                        ),
                    )
                    return obs

                motion = TrackedGovernancePolicy(
                    temporal, governor, provider, controller, dt_s=0.05
                )
                stack = operational_stack(
                    backend,
                    motion,
                    source_id="front-rgbd",
                    model_version=model["sha256"],
                    simulation_clock=lambda: world.get_snapshot().timestamp.elapsed_seconds,
                    config=StackConfig(**profile["operational_gate"]),
                )
                episode = f"{scenario}-{uuid.uuid4()}"
                stack.reset(episode_id=episode, graph=LaneGraph((), ()))
                original_track = stack.detector.track

                def recorded_track(frame):
                    batch = original_track(frame)
                    state["batch"] = batch
                    return batch

                stack.detector.track = recorded_track
                case = dict(
                    kinematic_actor_road_z_m=waypoint.transform.location.z,
                    site=site,
                    scenario=scenario,
                    episode_id=episode,
                    calibration=asdict(calibration),
                    ego_geometry=asdict(adapter.geometry),
                    front_from_rear_m=front,
                    ego_blueprint=ego.type_id,
                    ego_bbox=dict(
                        location=[
                            ego.bounding_box.location.x,
                            ego.bounding_box.location.y,
                            ego.bounding_box.location.z,
                        ],
                        extent=[
                            ego.bounding_box.extent.x,
                            ego.bounding_box.extent.y,
                            ego.bounding_box.extent.z,
                        ],
                    ),
                    rows=[],
                )
                report["cases"].append(case)
                with RGBDCapture(owner, ego, episode, rig=rig) as capture:
                    for index in range(args.frames):
                        if scenario == "failure_recovery" and index == 141:
                            stack.reset(episode_id=case["episode_id"], graph=LaneGraph((), ()))
                        hidden = (
                            140 <= index < 143
                            if scenario == "short_occlusion"
                            else 140 <= index < 160
                            if scenario == "continuity_risk"
                            else False
                        )
                        if target:
                            target.set_transform(
                                pose_along(
                                    carla,
                                    road_spawn,
                                    50 - index * 0.05 * 4,
                                    0.8 * math.sin(index * 0.15)
                                    if scenario == "continuity_risk"
                                    else 0,
                                )
                            )
                        if bus:
                            bus.set_transform(pose_along(carla, road_spawn, 12 if hidden else -20))
                        state.clear()
                        temporal.last = None
                        temporal.duration = 0
                        controller.duration_s = 0
                        controller.last = None
                        governor.duration = 0
                        governor.output = None
                        governor.inputs = None
                        t0 = time.monotonic()
                        frame_id = owner.tick()
                        sim_time = world.get_snapshot().timestamp.elapsed_seconds
                        pose = EgoPose(adapter.state(), ego.get_transform().location.z)
                        capture.record_snapshot(frame_id, pose)
                        failure = None
                        frame = None
                        aligned = None
                        read_start = time.monotonic()
                        try:
                            aligned = capture.read(frame_id, 3.0)
                            t1 = time.monotonic()
                            if scenario == "failure_recovery" and index == 140:
                                raise TimeoutError(
                                    "INJECTED dropout after source read; raw source withheld from operational frame"
                                )
                            frame = aligned.to_bundle(
                                calibration,
                                established_depth_convention="axial",
                                capture_started_monotonic_s=t0,
                            )
                        except Exception as exc:
                            t1 = time.monotonic()
                            failure = f"{type(exc).__name__}: {exc}"
                            frame = FrameBundle(
                                case["episode_id"],
                                frame_id,
                                sim_time,
                                t1,
                                pose,
                                np.zeros((rig.height, rig.width, 3), np.uint8),
                                calibration,
                                capture_started_monotonic_s=t0,
                            )
                        t2 = time.monotonic()
                        packet = stack.step(frame)
                        t3 = time.monotonic()
                        current_sim = world.get_snapshot().timestamp.elapsed_seconds
                        current = packet.is_current(frame.episode_id, current_sim, monotonic_s=t3)
                        batch = state.get("batch")
                        tracks = temporal.last
                        detections = (
                            []
                            if batch is None
                            else [
                                dict(
                                    class_name=d.class_name,
                                    confidence=d.confidence,
                                    xyxy=[d.box.x_min, d.box.y_min, d.box.x_max, d.box.y_max],
                                    track_id=d.track_id,
                                )
                                for d in batch.detections
                            ]
                        )
                        track_rows = (
                            []
                            if tracks is None
                            else [
                                dict(
                                    identity=asdict(x.identity),
                                    metadata=asdict(x.metadata),
                                    confirmed=x.confirmed,
                                    current=x.current,
                                    depth_valid=x.depth_valid,
                                    distance_m=x.distance_m,
                                    robust_distance_m=x.robust_distance_m,
                                    reason=x.invalid_reason,
                                )
                                for x in tracks.tracks
                            ]
                        )
                        provider_state = state.get("provider")
                        admitted = bool(
                            current
                            and packet.diagnostics.observation_valid
                            and motion.observation_valid
                        )
                        row = dict(
                            index=index,
                            episode_id=frame.episode_id,
                            frame_id=frame_id,
                            sim_time_s=sim_time,
                            capture_started_monotonic_s=t0,
                            sensor_available_monotonic_s=t1,
                            sensor_receipt_monotonic_s=frame.received_monotonic_s,
                            processing_started_monotonic_s=t2,
                            decision_monotonic_s=t3,
                            decision_sim_time_s=current_sim,
                            source_age_at_decision_s=t3 - t0,
                            source_sim_age_s=current_sim - sim_time,
                            sensor_wait_s=t1 - read_start,
                            sensor_availability_s=t1 - t0,
                            bundle_s=t2 - t1,
                            stack_s=t3 - t2,
                            detector_s=dict(packet.diagnostics.stage_durations_s).get(
                                "detector", 0
                            ),
                            filter_s=temporal.duration,
                            provider_s=state.get("provider_s", 0),
                            governor_s=governor.duration,
                            controller_s=controller.duration_s,
                            stage_durations=dict(packet.diagnostics.stage_durations_s),
                            backend_call_timing=None
                            if batch is None
                            else asdict(backend.last_call_timing),
                            detector_invoked=batch is not None,
                            detections=detections,
                            tracks=track_rows,
                            expired=[] if tracks is None else [asdict(x) for x in tracks.expired],
                            untracked=[]
                            if tracks is None
                            else [
                                dict(class_name=x.class_name, track_id=x.track_id)
                                for x in tracks.untracked
                            ],
                            input_depth_present=frame.depth_m is not None,
                            source_failure=failure,
                            injected_failure=scenario == "failure_recovery" and index == 140,
                            provider=provider_state,
                            policy_failures=state.get("policy_failures", []),
                            gate=dict(
                                generation=backend.generation,
                                fresh_results=stack.freshness_gate.fresh_results,
                                required_results=temporal.policy.hits,
                                reason=stack.freshness_gate.reason,
                            ),
                            policy_observation_valid=motion.observation_valid,
                            packet=asdict(packet),
                            packet_current=current,
                            admitted=admitted,
                            governor_output=governor.output,
                            governor_inputs=governor.inputs,
                            policy_target_mps=motion.last_target_mps,
                            policy_emergency=motion.last_emergency,
                            controller_output=controller.last,
                            ego_pose=asdict(pose),
                            scripted_occlusion=hidden,
                            explicit_reset=scenario == "failure_recovery" and index == 141,
                        )
                        row["decision_overhead_s"] = row["stack_s"] - sum(
                            row[k]
                            for k in (
                                "detector_s",
                                "filter_s",
                                "provider_s",
                                "governor_s",
                                "controller_s",
                            )
                        )
                        # Evaluation and artifact writing occur only after decision.
                        row["evaluation"] = None
                        if target:
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
                            reference = projected_target(
                                world_vertices(target),
                                capture.sensors[0].get_transform().get_inverse_matrix(),
                                intrinsics,
                            )
                            labels = (
                                target_indices(
                                    tuple()
                                    if batch is None
                                    else tuple(
                                        Detection(d.class_name, d.confidence, d.box, d.track_id)
                                        for d in batch.detections
                                    ),
                                    frame.depth_m,
                                    calibration,
                                    reference,
                                )
                                if frame.depth_m is not None
                                else []
                            )
                            row["evaluation"] = dict(
                                target_gap_m=gap,
                                target_detection_indices=labels,
                                target_reference=reference,
                            )
                        if frame.depth_m is not None:
                            for name, array in [("depth", frame.depth_m), ("rgb", frame.rgb)]:
                                f = directory / f"{index}-{name}.npy"
                                np.save(f, array, allow_pickle=False)
                                row[name + "_npy"] = str(f.relative_to(args.output))
                                row[name + "_sha256"] = digest(f)
                        row["physical_actuation"] = (
                            "full brake; positive packet remains shadow only"
                        )
                        dispatch_start = time.monotonic()
                        if current:
                            adapter.apply_packet(
                                replace(packet, command=ControlCommand(0, 0, 1)),
                                episode_id=frame.episode_id,
                                sim_time_s=current_sim,
                                client=client,
                            )
                        else:
                            adapter.stop()
                        row["brake_dispatch_ack_monotonic_s"] = time.monotonic()
                        row["brake_dispatch_wall_s"] = (
                            row["brake_dispatch_ack_monotonic_s"] - dispatch_start
                        )
                        case["rows"].append(row)
                        with (directory / "rows.jsonl").open("a") as stream:
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                        if index % 25 == 0:
                            print(
                                f"{site}/{scenario}: {index}/{args.frames} admitted={admitted} reason={packet.diagnostics.reasons}",
                                flush=True,
                            )
                    case["queue_events"] = capture.joiner.stats()
                    case["callback_errors"] = capture.callback_errors
                for actor in (bus, target, ego):
                    if actor is not None and actor.is_alive:
                        actor.destroy()
                owner.tick()
                case["admissible_frames"] = sum(r["admitted"] for r in case["rows"])
                case["background_preflight_suitable"] = bool(
                    args.preflight
                    and all(
                        r["detector_invoked"]
                        and r["gate"]["fresh_results"] >= r["gate"]["required_results"]
                        and all(t["current"] and t["depth_valid"] for t in r["tracks"])
                        and not r["untracked"]
                        and not r["expired"]
                        for r in case["rows"][-10:]
                    )
                )
                print(
                    f"CASE COMPLETE site={site} scenario={scenario} admissible={case['admissible_frames']} preflight_suitable={case['background_preflight_suitable']}",
                    flush=True,
                )
            report["settings"]["device"] = str(backend.model.predictor.device)
            report["status"] = "COMPLETE"
    except BaseException as exc:
        error = exc
        report["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if backend is not None:
            backend.close()
        world.wait_for_tick(5)
        report["settings_restored"] = _settings_record(world.get_settings()) == before
        report["no_actor_leaks"] = {a.id for a in world.get_actors()} == actors_before
        if not report["settings_restored"] or not report["no_actor_leaks"]:
            report["status"] = "INCOMPLETE"
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps({k: report[k] for k in ("status", "settings_restored", "no_actor_leaks")}),
            flush=True,
        )
    if error:
        raise error
    if report["status"] != "COMPLETE":
        raise RuntimeError("Incomplete capture")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="172.28.176.1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sites", type=int, nargs="+", default=[22])
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.frames < 20 or (not args.preflight and args.frames != 200):
        parser.error("Preflight >=20; final exactly 200 frames per scenario")
    collect(args)


if __name__ == "__main__":
    main()
