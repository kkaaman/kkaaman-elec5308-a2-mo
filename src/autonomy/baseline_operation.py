"""Load the frozen, bounded operating profile without bypassing its gate."""

import math
from pathlib import Path

from autonomy.admission import load_admitted_baseline
from autonomy.operational import operational_stack
from autonomy.stack import StackConfig
from planning.speed_governor import GovernorConfig


def frozen_governor_config(path):
    return _governor_config(load_admitted_baseline(path))


def _governor_config(baseline):
    freeze = baseline["baseline_threshold_freeze"]
    values = freeze["frozen_values"]
    composition = freeze["response_composition"]
    config = StackConfig(**freeze["operational_gate"])
    required = (
        config.max_observation_age_s
        + config.command_ttl_s
        + composition["measured_vehicle_response_s"]
    )
    if values["response_budget_s"] < required:
        raise ValueError("Governor response omits admitted age, delivery or vehicle lag")
    return GovernorConfig(
        cruise_mps=values["cruise_speed_mps"],
        deceleration_mps2=values["deceleration_mps2"],
        response_budget_s=values["response_budget_s"],
        stop_margin_m=values["stop_margin_m"],
        lateral_acceleration_mps2=freeze["implementation_policy"]["lateral_acceleration_mps2"],
        acceleration_mps2=freeze["implementation_policy"]["acceleration_mps2"],
        emergency_margin_m=freeze["implementation_policy"]["emergency_margin_m"],
    )


def baseline_operational_stack(
    backend, policy, *, baseline_path, source_id, model_version, simulation_clock
):
    baseline = load_admitted_baseline(baseline_path)
    freeze = baseline["baseline_threshold_freeze"]
    expected = _governor_config(baseline)
    if policy.governor.config != expected:
        raise ValueError("Use the frozen governor configuration")
    if model_version != freeze["detector"]["model_sha256"]:
        raise ValueError("Model identity differs from measured profile")
    if tuple(backend.config.allowed_classes or ()) != tuple(freeze["detector"]["allowed_classes"]):
        raise ValueError("Detector class selection differs from measured profile")
    for attribute, key in (
        ("confidence_threshold", "confidence"),
        ("image_size", "image_size"),
        ("tracker_config", "tracker"),
    ):
        if getattr(backend.config, attribute) != freeze["detector"][key]:
            raise ValueError("Detector setting differs from measured profile")
    import yaml

    from perception.object_detection import _resolve_tracker
    from perception.recorded_detector import verify_local_asset

    verify_local_asset(backend.config.model_name, freeze["detector"]["model_sha256"])
    if (
        yaml.safe_load(Path(_resolve_tracker(backend.config.tracker_config)).read_text())
        != freeze["detector"]["tracker_settings"]
    ):
        raise ValueError("Resolved tracker settings differ from measured profile")
    if policy.temporal_filter.policy.__dict__ != freeze["track_policy"]:
        raise ValueError("Confirmation policy differs from measured profile")
    if (
        policy.temporal_filter.depth_policy is None
        or policy.temporal_filter.depth_policy.__dict__ != freeze["depth_policy"]
    ):
        raise ValueError("Depth policy differs from measured profile")
    provider = policy.observation_provider

    def checked_provider(frame, batch, tracks):
        rig = freeze["camera_rig"]
        focal = rig["width"] / (2 * math.tan(math.radians(rig["fov_deg"]) / 2))
        if frame.rgb.shape[:2] != (rig["height"], rig["width"]) or not (
            math.isclose(frame.calibration.fx_px, focal, rel_tol=1e-6)
            and math.isclose(frame.calibration.fy_px, focal, rel_tol=1e-6)
        ):
            raise ValueError("Capture dimensions/intrinsics differ from measured rig")
        return provider(frame, batch, tracks)

    policy.observation_provider = checked_provider
    if policy.dt_s != freeze["fixed_delta_seconds"]:
        raise ValueError("Controller timestep differs from measured profile")
    return operational_stack(
        backend,
        policy,
        source_id=source_id,
        model_version=model_version,
        simulation_clock=simulation_clock,
        config=StackConfig(**freeze["operational_gate"]),
    )
