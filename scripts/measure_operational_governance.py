"""Measure the actual same-frame warm tracked-governance path, with parked actuators."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from autonomy.interfaces import EgoPose, LaneGraph
from autonomy.obstacle_policy import CorridorObservation, TrackedGovernancePolicy
from autonomy.operational import operational_stack
from autonomy.stack import StackConfig
from carla_adapter.sensors import CameraRig, RGBDCapture
from carla_adapter.vehicle import FixedVehicleAdapter
from carla_adapter.world import SynchronousWorldSession, _settings_record
from geometry.camera import CameraIntrinsics
from navigation_control import PathFollower
from navigation_control.types import ControlCommand, ControllerConfig, Waypoint
from obstacle_perception.box_depth import DepthPolicy, estimate_box_depth
from obstacle_perception.tracking import TrackDepthFilter, TrackPolicy
from perception.recorded_detector import verify_local_asset
from planning.speed_governor import GovernorConfig, SpeedGovernor


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TimedFilter(TrackDepthFilter):
    duration_s = 0.0

    def update(self, frame, batch):
        start = time.monotonic()
        result = super().update(frame, batch)
        self.duration_s = time.monotonic() - start
        return result


class Controller:
    def __init__(self, config, pose):
        self.follower = PathFollower(ControllerConfig(**config))
        self.follower.set_path(
            [
                Waypoint(pose.x_m, pose.y_m),
                Waypoint(
                    pose.x_m + 100 * math.cos(pose.yaw_rad), pose.y_m + 100 * math.sin(pose.yaw_rad)
                ),
            ]
        )
        self.duration_s = 0.0

    def reset(self):
        self.follower.reset()

    def __call__(self, frame, target, dt):
        start = time.monotonic()
        result = self.follower.update(frame.ego.state, target, dt).command
        self.duration_s += time.monotonic() - start
        return result

    def hold(self, frame, dt):
        self(frame, 0.0, dt)


def collect(args):
    import carla
    import lap
    import yaml

    from perception.object_detection import ObjectDetector, _resolve_tracker
    from perception.types import PerceptionConfig

    if not lap.__version__:
        raise RuntimeError("Install tracking dependencies first")
    root = Path(__file__).resolve().parents[1]
    evidence = root / "docs/evidence/BL-D-06/sources"
    calibration_doc = json.loads((evidence / "depth-calibration.json").read_text())
    braking = json.loads((evidence / "longitudinal.json").read_text())
    navigation = json.loads((evidence / "navigation-baseline.json").read_text())
    manifest = json.loads((root / "configs/models.yaml").read_text())
    model = manifest["models"][0]
    asset = verify_local_asset(root / model["path"], model["sha256"])
    settings = PerceptionConfig(model_name=str(asset))
    tracker_path = Path(_resolve_tracker(settings.tracker_config))
    client = carla.Client(args.host, 2000)
    client.set_timeout(15)
    world = client.get_world()
    map_ = world.get_map()
    rig = CameraRig(**calibration_doc["rig"])
    map_hash = hashlib.sha256(map_.to_opendrive().replace("\r\n", "\n").encode()).hexdigest()
    if (
        client.get_server_version() != "0.9.16"
        or client.get_client_version() != "0.9.16"
        or map_.name.split("/")[-1] != "Town10HD_Opt"
        or map_hash != calibration_doc["opendrive_sha256"]
        or calibration_doc["convention"] != "axial"
    ):
        raise ValueError("Calibrated CARLA build/map/rig required")
    if any(a.type_id.startswith(("vehicle.", "walker.", "sensor.")) for a in world.get_actors()):
        raise RuntimeError("Use an empty world")
    before = _settings_record(world.get_settings())
    ids = {a.id for a in world.get_actors()}
    args.output.mkdir(parents=True, exist_ok=False)
    config = StackConfig.load(root / "configs/stack.yaml")
    config = replace(config, require_depth=True)
    policy = TrackPolicy()
    report = dict(
        status="INCOMPLETE",
        scope="Same-frame real YOLO/filter/governor/PathFollower decision timing; ego physically held by brake, positive decisions are shadow outputs",
        host=args.host,
        spawn_index=args.spawn_index,
        server=client.get_server_version(),
        map=map_.name,
        opendrive_sha256=map_hash,
        rig=asdict(rig),
        stack_config=asdict(config),
        track_policy=asdict(policy),
        settings=dict(
            confidence=settings.confidence_threshold,
            image_size=settings.image_size,
            model_sha256=model["sha256"],
            tracker=settings.tracker_config,
            tracker_settings=yaml.safe_load(tracker_path.read_text()),
            device="UNINITIALISED",
            versions={
                k: importlib.metadata.version(k) for k in ("ultralytics", "torch", "numpy", "lap")
            },
        ),
        provenance={
            str(path.relative_to(root)): digest(path)
            for path in [
                evidence / "depth-calibration.json",
                evidence / "longitudinal.json",
                root / "configs/stack.yaml",
                root / "configs/models.yaml",
                Path(__file__).resolve(),
                tracker_path,
            ]
            if path.is_relative_to(root)
        },
        clock="Host monotonic before tick request is a conservative capture-time lower bound; all stages are contiguous non-overlapping host durations. Simulation expiry is checked separately.",
        rows=[],
        episodes=[],
    )
    report["source_hashes"] = {
        str(p.relative_to(root)): digest(p)
        for p in [
            root / "src/autonomy/stack.py",
            root / "src/autonomy/freshness.py",
            root / "src/autonomy/operational.py",
            root / "src/autonomy/obstacle_policy.py",
            root / "src/perception/frame_detector.py",
            root / "src/obstacle_perception/tracking.py",
            root / "src/carla_adapter/sensors.py",
            root / "src/planning/speed_governor.py",
            root / "src/autonomy/interfaces.py",
            root / "src/perception/object_detection.py",
            root / "src/carla_adapter/vehicle.py",
        ]
    }
    for path in report["source_hashes"]:
        dest = args.output / "source" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((root / path).read_bytes())
    backend = None
    failure = None
    try:
        constructed = time.monotonic()
        backend = ObjectDetector(settings)
        report["construction_wall_s"] = time.monotonic() - constructed
        with SynchronousWorldSession(client, ownership_key=f"{args.host}:2000") as owner:
            changed = world.get_settings()
            changed.no_rendering_mode = False
            world.apply_settings(changed)
            spawn = map_.get_spawn_points()[args.spawn_index]
            ego = owner.try_spawn(world.get_blueprint_library().find("vehicle.tesla.model3"), spawn)
            if ego is None:
                raise RuntimeError("Ego spawn failed")
            ego.apply_control(carla.VehicleControl(brake=1))
            wp = map_.get_waypoint(spawn.location).next(25)
            if len(wp) != 1:
                raise RuntimeError("Ambiguous capture site")
            target_pose = wp[0].transform
            target_pose.location.z += 0.65
            target = owner.try_spawn(
                world.get_blueprint_library().find("vehicle.tesla.model3"), target_pose
            )
            if target is None:
                raise RuntimeError("Target spawn failed")
            target.apply_control(carla.VehicleControl(brake=1))
            for _ in range(20):
                owner.tick()
            adapter = FixedVehicleAdapter(ego)
            origin = adapter.state()
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
            front_from_rear = (
                ego.bounding_box.location.x
                + ego.bounding_box.extent.x
                + adapter.geometry.rear_axle_offset_m
            )
            report["ego_front_from_rear_axle_m"] = front_from_rear
            depth_policy = DepthPolicy(max_depth_m=25.0)

            def provider(frame, batch, tracks):
                gaps = []
                for track in tracks.tracks:
                    if track.current and track.depth_valid:
                        estimate = estimate_box_depth(
                            frame.depth_m, track.detection.box, frame.calibration, depth_policy
                        )
                        if estimate.valid:
                            gaps.append(float(estimate.point_ego_m[0] - front_from_rear))
                gap = min(gaps) if gaps else None
                # Conservative shadow straight-scene input: no empty detection
                # becomes coverage/clearance, and no evaluation target geometry is read.
                return CorridorObservation(batch.metadata, bool(gaps), gap, gap, False)

            controller = Controller(navigation["controller"], origin)
            for episode_index in range(args.episodes):
                episode = f"warm-operational-{episode_index}"
                temporal = TimedFilter(
                    source_id="front-rgbd",
                    model_version=model["sha256"],
                    policy=policy,
                    depth_policy=depth_policy,
                )
                governor = SpeedGovernor(
                    GovernorConfig(
                        4.0,
                        braking["deceleration_mps2"],
                        1.5,
                        config.max_observation_age_s
                        + braking["response_budget_s"]
                        + config.command_ttl_s,
                        6.0,
                        1.0,
                        2.0,
                    )
                )
                motion = TrackedGovernancePolicy(
                    temporal,
                    governor,
                    provider,
                    controller,
                    dt_s=navigation["runtime"]["fixed_delta_seconds"],
                )
                stack = operational_stack(
                    backend,
                    motion,
                    source_id="front-rgbd",
                    model_version=model["sha256"],
                    simulation_clock=lambda: world.get_snapshot().timestamp.elapsed_seconds,
                    config=config,
                )
                reset_started = time.monotonic()
                stack.reset(episode_id=episode, graph=LaneGraph((), ()))
                report["episodes"].append(
                    dict(
                        episode=episode,
                        reset_wall_s=time.monotonic() - reset_started,
                        startup_state="HOLD",
                        generation=backend.generation,
                    )
                )
                with RGBDCapture(owner, ego, episode, rig=rig) as capture:
                    for index in range(args.frames):
                        t0 = time.monotonic()
                        frame_id = owner.tick()
                        pose = EgoPose(adapter.state(), ego.get_transform().location.z)
                        capture.record_snapshot(frame_id, pose)
                        aligned = capture.read(frame_id, 3.0)
                        t1 = time.monotonic()
                        injected = index == args.frames - 1
                        frame = aligned.to_bundle(
                            calibration,
                            established_depth_convention="axial",
                            capture_started_monotonic_s=t0
                            - (config.max_observation_age_s + 1 if injected else 0),
                        )
                        t2 = time.monotonic()
                        temporal.duration_s = 0
                        controller.duration_s = 0
                        packet = stack.step(frame)
                        t3 = time.monotonic()
                        stages = dict(packet.diagnostics.stage_durations_s)
                        warm = (
                            stack.freshness_gate.fresh_results >= policy.hits
                            and packet.diagnostics.observation_valid
                        )
                        target_speed = motion.last_target_mps if warm else 0.0
                        current_sim = world.get_snapshot().timestamp.elapsed_seconds
                        current = packet.is_current(episode, current_sim, monotonic_s=t3)
                        row = dict(
                            episode=episode,
                            index=index,
                            frame_id=frame_id,
                            sim_time_s=frame.sim_time_s,
                            capture_started_monotonic_s=t0,
                            sensor_available_monotonic_s=t1,
                            stack_started_monotonic_s=t2,
                            decision_monotonic_s=t3,
                            sensor_availability_s=t1 - t0,
                            bundle_s=t2 - t1,
                            stack_s=t3 - t2,
                            capture_to_decision_s=t3 - t0,
                            receipt_to_decision_s=t3 - frame.received_monotonic_s,
                            detector_s=stages.get("detector", 0.0),
                            filter_s=temporal.duration_s,
                            controller_s=controller.duration_s,
                            decision_and_overhead_s=t3
                            - t2
                            - stages.get("detector", 0.0)
                            - temporal.duration_s
                            - controller.duration_s,
                            warmed=warm,
                            gate_reason=stack.freshness_gate.reason,
                            packet_reason=packet.diagnostics.reasons,
                            packet_current=current,
                            target_mps=target_speed,
                            throttle=packet.command.throttle,
                            brake=packet.command.brake,
                            injected_stale=injected,
                            track_diagnostics=getattr(motion, "track_diagnostics", []),
                        )
                        report["rows"].append(row)
                        with (args.output / "timing.jsonl").open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
                        # Keep physical ego parked. Timing run does not claim closed-loop actuation.
                        if current:
                            adapter.apply_packet(
                                replace(packet, command=ControlCommand(0, 0, 1)),
                                episode_id=episode,
                                sim_time_s=current_sim,
                                client=client,
                            )
                        else:
                            adapter.stop()
                        if index % 25 == 0:
                            print(
                                f"{episode}: {index}/{args.frames} {row['gate_reason']} target={target_speed}",
                                flush=True,
                            )
                if episode_index + 1 < args.episodes:
                    # Do not close shared backend; reset through stack on next episode.
                    pass
            report["settings"]["device"] = str(backend.model.predictor.device)
            report["status"] = "COMPLETE"
    except BaseException as exc:
        failure = exc
        report["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if backend is not None:
            backend.close()
        world.wait_for_tick(5)
        report["settings_restored"] = _settings_record(world.get_settings()) == before
        report["no_actor_leaks"] = {a.id for a in world.get_actors()} == ids
        if not report["settings_restored"] or not report["no_actor_leaks"]:
            report["status"] = "INCOMPLETE"
        warm = [r for r in report["rows"] if r["warmed"] and not r["injected_stale"]]
        report["summary"] = dict(
            warm_samples=len(warm),
            positive_packets=sum(r["target_mps"] > 0 and r["throttle"] > 0 for r in warm),
            cold_or_stale_positive_packets=sum(
                r["throttle"] > 0 for r in report["rows"] if not r["warmed"] or r["injected_stale"]
            ),
            max_warm_capture_to_decision_s=max(
                (r["capture_to_decision_s"] for r in warm), default=None
            ),
            p95_warm_capture_to_decision_s=float(
                np.percentile([r["capture_to_decision_s"] for r in warm], 95)
            )
            if warm
            else None,
            p95_warm_receipt_to_decision_s=float(
                np.percentile([r["receipt_to_decision_s"] for r in warm], 95)
            )
            if warm
            else None,
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in ("status", "summary", "settings_restored", "no_actor_leaks")
                },
                indent=2,
            )
        )
    if failure is not None:
        raise failure
    if report["status"] != "COMPLETE":
        raise RuntimeError("Incomplete measurement")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="172.28.176.1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--spawn-index", type=int, default=152)
    args = parser.parse_args()
    if args.frames < 10 or args.episodes < 2:
        parser.error("At least two episodes of ten frames required")
    collect(args)


if __name__ == "__main__":
    main()
