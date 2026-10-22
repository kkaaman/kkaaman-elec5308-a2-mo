"""Verify same-frame operational logs against captured clocks and current policy."""

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from autonomy.interfaces import CameraCalibration, EgoPose, FrameBundle, LaneGraph, VehicleState
from autonomy.obstacle_policy import CorridorObservation, TrackedGovernancePolicy
from obstacle_perception.box_depth import DepthPolicy, estimate_box_depth
from obstacle_perception.tracking import TrackDepthFilter, TrackPolicy
from obstacle_perception.types import (
    CallTiming,
    DetectionBatch,
    DetectionObservation,
    ObservationMetadata,
    PixelBox,
)
from planning.speed_governor import (
    GovernorConfig,
    SpeedGovernor,
    speed_for_distance,
    stopping_distance,
)
from scripts.measure_operational_governance import Controller


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    return (
        dict(
            samples=len(values),
            min=min(values),
            median=float(np.median(values)),
            p95=float(np.percentile(values, 95)),
            max=max(values),
        )
        if values
        else None
    )


def validate_decision_clocks(row, config):
    """Check measured boundaries and reject any positive out-of-budget output."""
    packet = row["packet"]
    if (packet["episode_id"], packet["source_frame_id"], packet["source_sim_time_s"]) != (
        row["episode_id"],
        row["frame_id"],
        row["sim_time_s"],
    ):
        raise ValueError("Packet source mismatch")
    times = [
        row[k]
        for k in (
            "capture_started_monotonic_s",
            "sensor_available_monotonic_s",
            "processing_started_monotonic_s",
            "decision_monotonic_s",
        )
    ]
    if any(not math.isfinite(x) for x in times) or times != sorted(times):
        raise ValueError("Host boundary order invalid")
    if not math.isclose(
        packet["valid_until_monotonic_s"], times[0] + config["max_observation_age_s"], abs_tol=1e-8
    ):
        raise ValueError("Host expiry does not match admission budget")
    if not math.isclose(
        packet["valid_until_sim_time_s"], row["sim_time_s"] + config["command_ttl_s"], abs_tol=1e-8
    ):
        raise ValueError("Simulation expiry does not match source TTL")
    age = times[-1] - times[0]
    partition = (
        row["sensor_availability_s"]
        + row["bundle_s"]
        + sum(
            row[k]
            for k in (
                "detector_s",
                "filter_s",
                "provider_s",
                "governor_s",
                "controller_s",
                "decision_overhead_s",
            )
        )
    )
    if abs(partition - age) > 1e-8 or abs(age - row["source_age_at_decision_s"]) > 1e-8:
        raise ValueError("Incomplete or overlapping timing partition")
    if row["admitted"] and (
        age > config["max_observation_age_s"]
        or row["stack_s"] > config["max_step_wall_s"]
        or row["source_sim_age_s"] >= config["command_ttl_s"]
    ):
        raise ValueError("Admitted output violates freshness")
    if not row["admitted"] and packet["command"]["throttle"] > 0:
        raise ValueError("Invalid output authorised throttle")
    return abs(partition - age)


def audit_same_frame(path, root):
    path, root = Path(path), Path(root)
    d = json.loads(path.read_text())
    if (
        d["kind"] != "operational_same_frame_approach"
        or d["preflight"]
        or d["status"] != "COMPLETE"
        or not d["settings_restored"]
        or not d["no_actor_leaks"]
    ):
        raise ValueError("Complete final operational capture required")
    for name, digest in d["source_hashes"].items():
        if sha(path.parent / "source" / name) != digest:
            raise ValueError("Captured source hash mismatch")
        if name.startswith("src/") and sha(root / name) != digest:
            raise ValueError("Measured runtime changed")
    baseline = json.loads((path.parent / "source/configs/mission_baseline.json").read_text())
    profile = d["operational_profile"]
    values = profile["candidate_values"]
    impl = profile["implementation_policy"]
    config = profile["operational_gate"]
    if baseline["baseline_threshold_freeze"] != profile:
        raise ValueError("Profile snapshot mismatch")
    for key in ("confidence", "image_size", "tracker", "tracker_settings", "model_sha256"):
        if d["settings"][key] != profile["detector"][key]:
            raise ValueError("Measured detector profile mismatch")
    current = json.loads((root / "configs/mission_baseline.json").read_text())[
        "baseline_threshold_freeze"
    ]
    for key in (
        "operational_gate",
        "track_policy",
        "depth_policy",
        "camera_rig",
        "detector",
        "fixed_delta_seconds",
        "candidate_values",
        "implementation_policy",
    ):
        if profile[key] != current[key]:
            raise ValueError("Operational profile changed since measurement")
    summaries = []
    allrows = []
    partition_max = 0
    replay_mismatches = []
    for case in d["cases"]:
        first_pose = case["rows"][0]["ego_pose"]["state"]
        controller = Controller(baseline["controller"], VehicleState(**first_pose))
        temporal = TrackDepthFilter(
            source_id=d["source_id"],
            model_version=d["model_version"],
            policy=TrackPolicy(**profile["track_policy"]),
            depth_policy=DepthPolicy(**profile["depth_policy"]),
        )
        provider_state = {}

        def provider(frame, batch, tracks):
            gaps = []
            for track in tracks.tracks:
                if track.current and track.depth_valid:
                    e = estimate_box_depth(
                        frame.depth_m, track.detection.box, frame.calibration, temporal.depth_policy
                    )
                    if e.valid:
                        gaps.append(float(e.point_ego_m[0] - case["front_from_rear_m"]))
            gap = min(gaps) if gaps else None
            obs = CorridorObservation(batch.metadata, bool(gaps), gap, gap, False)
            provider_state["observation"] = obs
            return obs

        policy = TrackedGovernancePolicy(
            temporal,
            SpeedGovernor(
                GovernorConfig(
                    values["cruise_speed_mps"],
                    values["deceleration_mps2"],
                    impl["lateral_acceleration_mps2"],
                    values["response_budget_s"],
                    values["stop_margin_m"],
                    impl["acceleration_mps2"],
                    impl["emergency_margin_m"],
                )
            ),
            provider,
            controller,
            dt_s=d["fixed_delta_seconds"],
        )
        policy.reset(episode_id=case["episode_id"], graph=LaneGraph((), ()))
        first = None
        counts = Counter()
        rejections = Counter()
        case_rows = []
        for row in case["rows"]:
            if row["explicit_reset"]:
                policy.reset(episode_id=case["episode_id"], graph=LaneGraph((), ()))
            dep = rgb = None
            if row["input_depth_present"]:
                for key in ("depth", "rgb"):
                    p = path.parent / row[key + "_npy"]
                    if sha(p) != row[key + "_sha256"]:
                        raise ValueError("Frame artifact hash mismatch")
                dep = np.load(path.parent / row["depth_npy"], allow_pickle=False)
                rgb = np.load(path.parent / row["rgb_npy"], allow_pickle=False)
            else:
                rgb = np.zeros((d["rig"]["height"], d["rig"]["width"], 3), np.uint8)
            frame = FrameBundle(
                row["episode_id"],
                row["frame_id"],
                row["sim_time_s"],
                row["sensor_receipt_monotonic_s"],
                EgoPose(VehicleState(**row["ego_pose"]["state"]), row["ego_pose"]["z_m"]),
                rgb,
                CameraCalibration(**case["calibration"]),
                dep,
                row["frame_id"] if dep is not None else None,
                row["capture_started_monotonic_s"],
            )
            validate_decision_clocks(row, config)
            row["receipt_to_output_s"] = (
                row["decision_monotonic_s"] - row["sensor_receipt_monotonic_s"]
            )
            row["decision_to_brake_ack_s"] = (
                row["brake_dispatch_ack_monotonic_s"] - row["decision_monotonic_s"]
            )
            row["receipt_to_brake_ack_s"] = (
                row["brake_dispatch_ack_monotonic_s"] - row["sensor_receipt_monotonic_s"]
            )
            if row["admitted"] and row["decision_to_brake_ack_s"] > config["command_ttl_s"]:
                raise ValueError("Measured delivery exceeds reserved budget")
            packet = row["packet"]
            current = (
                packet["source_sim_time_s"]
                <= row["decision_sim_time_s"]
                <= packet["valid_until_sim_time_s"]
                and row["decision_monotonic_s"] <= packet["valid_until_monotonic_s"]
            )
            if current != row["packet_current"]:
                raise ValueError("Packet clock result mismatch")
            age = row["decision_monotonic_s"] - row["capture_started_monotonic_s"]
            sim_age = row["decision_sim_time_s"] - row["sim_time_s"]
            if (
                abs(age - row["source_age_at_decision_s"]) > 1e-8
                or abs(sim_age - row["source_sim_age_s"]) > 1e-8
            ):
                raise ValueError("Age clock mismatch")
            partition = (
                row["sensor_availability_s"]
                + row["bundle_s"]
                + sum(
                    row[k]
                    for k in (
                        "detector_s",
                        "filter_s",
                        "provider_s",
                        "governor_s",
                        "controller_s",
                        "decision_overhead_s",
                    )
                )
            )
            partition_max = max(partition_max, abs(partition - age))
            if partition_max > 1e-8:
                raise ValueError("Overlapping/incomplete timing partition")
            if row["provider"] is not None:
                metadata = ObservationMetadata(
                    row["episode_id"],
                    row["frame_id"],
                    row["sim_time_s"],
                    d["source_id"],
                    d["model_version"],
                )
                batch = DetectionBatch(
                    metadata,
                    tuple(
                        DetectionObservation(
                            x["class_name"], x["confidence"], PixelBox(*x["xyxy"]), x["track_id"]
                        )
                        for x in row["detections"]
                    ),
                    d["rig"]["width"],
                    d["rig"]["height"],
                    CallTiming(row["detector_s"]),
                )
                command = policy.evaluate(
                    frame, batch, fresh=True, ready="warmup_not_ready" not in row["policy_failures"]
                )
                obs = provider_state["observation"]
                recorded = row["provider"]
                if (
                    obs.valid != recorded["valid"]
                    or obs.obstacle_gap_m != recorded["obstacle_gap_m"]
                    or policy.observation_valid != row["policy_observation_valid"]
                ):
                    raise ValueError("Policy/provider replay mismatch")
                if row["packet"]["diagnostics"]["observation_valid"]:
                    if any(
                        abs(v - packet["command"][key]) > 1e-8 for key, v in asdict(command).items()
                    ):
                        replay_mismatches.append([case["scenario"], row["index"]])
                elif row["gate"]["reason"] in (
                    "capture_age_hold",
                    "processing_budget_hold",
                    "source_command_expired",
                ):
                    policy.on_hold(frame)
            elif row["detector_invoked"] or row["gate"]["reason"] in (
                "capture_age_hold",
                "processing_budget_hold",
                "source_command_expired",
            ):
                policy.on_hold(frame)
            admitted = bool(
                current
                and packet["diagnostics"]["observation_valid"]
                and row["policy_observation_valid"]
            )
            if admitted != row["admitted"]:
                raise ValueError("Admission mismatch")
            positive = packet["command"]["throttle"] > 0
            if admitted and (
                age > config["max_observation_age_s"]
                or row["stack_s"] > config["max_step_wall_s"]
                or sim_age >= config["command_ttl_s"]
                or row["gate"]["fresh_results"] < row["gate"]["required_results"]
                or any(not t["current"] or not t["depth_valid"] for t in row["tracks"])
                or row["untracked"]
                or row["expired"]
            ):
                raise ValueError("Admitted observation violates operational contract")
            if not admitted and positive:
                raise ValueError("Invalid/cold/stale frame authorised throttle")
            counts["frames"] += 1
            counts["admitted"] += admitted
            counts["positive_packets"] += positive
            counts["injected_failures"] += row["injected_failure"]
            counts["source_failures"] += row["source_failure"] is not None
            counts["occluded_frames"] += row["scripted_occlusion"]
            if not admitted:
                rejections.update(packet["diagnostics"]["reasons"])
            labels = row["evaluation"]["target_detection_indices"]
            ids = {row["detections"][i]["track_id"] for i in labels}
            target_valid = any(
                t["identity"]["track_id"] in ids
                and t["confirmed"]
                and t["current"]
                and t["depth_valid"]
                for t in row["tracks"]
            )
            counts["admitted_target_frames"] += admitted and target_valid
            if first is None and admitted and target_valid:
                first = dict(
                    index=row["index"],
                    frame_id=row["frame_id"],
                    evaluation_bumper_gap_m=row["evaluation"]["target_gap_m"],
                    operational_bumper_gap_m=row["provider"]["obstacle_gap_m"],
                    conservative_range_m=min(
                        row["evaluation"]["target_gap_m"], row["provider"]["obstacle_gap_m"]
                    ),
                    response_s=age,
                )
            allrows.append(row)
            case_rows.append(row)
        if case["rows"][0]["packet"]["command"]["throttle"] > 0:
            raise ValueError("Startup authorised positive motion")
        hidden = [r["index"] for r in case_rows if r["scripted_occlusion"]]
        fault = [r for r in case_rows if r["source_failure"] is not None]
        recovery = None
        if fault:
            if any(r["packet"]["command"]["throttle"] > 0 for r in fault):
                raise ValueError("Failure authorised positive output")
            recovered = next(
                (r for r in case_rows if r["index"] > fault[-1]["index"] and r["admitted"]), None
            )
            recovery = dict(
                failure_indices=[r["index"] for r in fault],
                explicit_reset_indices=[r["index"] for r in case_rows if r["explicit_reset"]],
                first_admitted_recovery_index=None if recovered is None else recovered["index"],
            )
            if recovered is None or not recovery["explicit_reset_indices"]:
                raise ValueError("Missing explicit fault recovery")
        summaries.append(
            dict(
                scenario=case["scenario"],
                fault_recovery=recovery,
                occlusion_indices=hidden,
                counts=dict(counts),
                rejections=dict(rejections),
                first_admitted_target=first,
                admitted_timing={
                    k: stats([r[k] for r in case_rows if r["admitted"]])
                    for k in (
                        "source_age_at_decision_s",
                        "sensor_availability_s",
                        "detector_s",
                        "filter_s",
                        "provider_s",
                        "governor_s",
                        "controller_s",
                        "stack_s",
                    )
                },
            )
        )
    if {s["scenario"] for s in summaries} != {
        "approach",
        "short_occlusion",
        "continuity_risk",
        "failure_recovery",
    }:
        raise ValueError("Operational scenario coverage incomplete")
    if replay_mismatches:
        raise ValueError(f"Controller replay differs: {replay_mismatches[:3]}")
    if any(s["first_admitted_target"] is None for s in summaries):
        raise ValueError("Missing first target admission")
    visibility = min(s["first_admitted_target"]["conservative_range_m"] for s in summaries)
    lag = json.loads((root / "docs/evidence/BL-D-06/sources/longitudinal.json").read_text())[
        "response_budget_s"
    ]
    budget = config["max_observation_age_s"] + config["command_ttl_s"] + lag
    if values["response_budget_s"] < budget:
        raise ValueError("Budget omits admitted age/delivery/vehicle lag")
    cap = speed_for_distance(
        visibility - values["stop_margin_m"], values["deceleration_mps2"], budget
    )
    return dict(
        kind="verified_operational_same_frame_approach",
        review_implementation_sha256=sha(__file__),
        status="PASS" if values["cruise_speed_mps"] <= cap else "BLOCKED_SPEED_ENVELOPE",
        capture_sha256=sha(path),
        capture_path=str(path),
        operational_profile=profile,
        capture_source_hashes=d["source_hashes"],
        environment={
            k: d[k]
            for k in (
                "server",
                "client",
                "map",
                "opendrive_sha256",
                "rig",
                "depth_convention",
                "weather",
                "fixed_delta_seconds",
                "settings",
            )
        },
        cases=summaries,
        total_frames=len(allrows),
        admissible_frames=sum(r["admitted"] for r in allrows),
        positive_packets=sum(r["packet"]["command"]["throttle"] > 0 for r in allrows),
        invalid_positive_packets=0,
        timing={
            k: stats([r[k] for r in allrows if r["admitted"]])
            for k in (
                "source_age_at_decision_s",
                "sensor_availability_s",
                "bundle_s",
                "detector_s",
                "filter_s",
                "provider_s",
                "governor_s",
                "controller_s",
                "decision_overhead_s",
                "stack_s",
                "receipt_to_output_s",
                "decision_to_brake_ack_s",
                "receipt_to_brake_ack_s",
            )
        },
        maximum_all_frame_age_s=max(r["source_age_at_decision_s"] for r in allrows),
        maximum_source_sim_age_s=max(r["source_sim_age_s"] for r in allrows),
        timing_partition_max_error_s=partition_max,
        controller_replay_mismatches=0,
        response_budget_s=budget,
        vehicle_lag_s=lag,
        usable_operational_range_m=visibility,
        exact_speed_cap_mps=cap,
        speed_cap_rounded_down_mps=math.floor(cap * 10) / 10,
        candidate_cruise_mps=values["cruise_speed_mps"],
        required_gap_at_candidate_m=values["stop_margin_m"]
        + stopping_distance(values["cruise_speed_mps"], values["deceleration_mps2"], budget),
        limitations=[
            "Parked-ego shadow decision and controlled straight approach; not closed-loop stopping certification or general full-corridor coverage.",
            "No empty observation authorises clearance; all visible tracks included.",
            "Highest calculated cap is envelope arithmetic, not evidence of driven operation at that speed.",
        ],
    )
