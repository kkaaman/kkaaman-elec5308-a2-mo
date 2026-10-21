"""Offline replay: GNSS-only vs lane-aided pose estimates on a recorded run's own inputs.

Every run with localisation logs its raw inputs per frame:
- the GNSS fix, if one arrived;
- wheel speed and gyro rate;
- the input time.

The RV-06 lane monitor gives a camera lane offset per replayed frame. This module
replays three estimators on exactly those inputs and scores each against the logged
true pose:

- ``gnss``: the GNSS-only filter, as run live. It must reproduce the logged estimate,
  which checks that the replay is faithful.
- ``gnss_lane``: GNSS plus lane offsets; the GNSS offset is treated as noise.
- ``gnss_lane_offset``: GNSS plus lane offsets, with the GNSS offset as a state.

Nothing here drives: the car drove on whatever the run used, and only the estimates
change.
"""

import math
from dataclasses import replace

import numpy as np

from autonomy.lane_localisation import LaneAidedEstimator
from autonomy.localisation import LocalisationInputs, PoseEstimator, pose_errors
from navigation_control.types import VehicleState

VARIANTS = ("gnss", "gnss_lane", "gnss_lane_offset")
MOVING_MPS = 0.5


def lane_measurements(records, min_confidence):
    """{frame_id: (signed offset, confidence)} for valid lane-monitor records."""
    lanes = {}
    for record in records:
        departure = record["departure"]
        if (
            departure["valid"]
            and departure["signed_offset_m"] is not None
            and departure["confidence"] >= min_confidence
        ):
            lanes[record["frame_id"]] = (departure["signed_offset_m"], departure["confidence"])
    return lanes


def lane_quality(records, min_confidence):
    """Camera lane offset minus the route offset at the true pose (RV-06 comparison)."""
    differences = [
        abs(r["departure"]["signed_offset_m"] - r["map_offset_m"])
        for r in records
        if r["departure"]["valid"]
        and r["departure"]["signed_offset_m"] is not None
        and r["departure"]["confidence"] >= min_confidence
        and r.get("map_offset_m") is not None
    ]
    if not differences:
        return dict(frames=len(records), usable=0)
    values = np.asarray(differences)
    return dict(
        frames=len(records),
        usable=len(values),
        median_abs_m=float(np.median(values)),
        p95_abs_m=float(np.percentile(values, 95)),
        max_abs_m=float(values.max()),
        over_0_5_m=int((values > 0.5).sum()),
    )


def _inputs(localisation):
    fix = localisation.get("gnss_xy_m")
    return LocalisationInputs(
        localisation["input_time_s"],
        localisation["wheel_speed_mps"],
        localisation["yaw_rate_rps"],
        tuple(fix) if fix else None,
    )


def replay(rows, route, config, aiding, lanes, *, drive_mode):
    """Per-variant trajectories (estimate and errors per logged frame) for one run.

    ``drive_mode``: the run drove on its estimate, which started at the mission start
    rather than at the first fix; the replay starts the same way.
    """
    logged = [r for r in rows if "localisation" in r]
    if not logged:
        raise ValueError("Run has no localisation records")
    estimators = dict(
        gnss=PoseEstimator(config.filter),
        gnss_lane=LaneAidedEstimator(config.filter, _without_offset(aiding)),
        gnss_lane_offset=LaneAidedEstimator(config.filter, _with_offset(aiding)),
    )
    if drive_mode:
        start = route.at(0.0)
        t0 = logged[0]["localisation"]["input_time_s"]
        estimators["gnss"].initialise_at(*start, t0)
        for name in VARIANTS[1:]:
            estimators[name].initialise_at(*start, t0, route)
    out = {name: [] for name in VARIANTS}
    replay_difference = 0.0
    for row in logged:
        inputs = _inputs(row["localisation"])
        truth = VehicleState(**row["ego"])
        lane = lanes.get(row["frame_id"])
        for name, estimator in estimators.items():
            if name == "gnss":
                estimate = estimator.step(inputs, route)
            else:
                estimate = estimator.step(inputs, route, lane)
            if estimate is None:
                continue
            if name == "gnss":
                logged_loc = row["localisation"]
                replay_difference = max(
                    replay_difference,
                    abs(estimate.x_m - logged_loc["x_m"]),
                    abs(estimate.y_m - logged_loc["y_m"]),
                    abs(math.remainder(estimate.yaw_rad - logged_loc["yaw_rad"], 2 * math.pi)),
                )
            out[name].append(
                dict(
                    frame_id=row["frame_id"],
                    sim_time_s=row["sim_time_s"],
                    speed_mps=row["speed_mps"],
                    progress_m=route.project(truth.x_m, truth.y_m).s_m,
                    x_m=estimate.x_m,
                    y_m=estimate.y_m,
                    yaw_rad=estimate.yaw_rad,
                    lane=getattr(estimator, "last_lane", None),
                    **pose_errors(estimate, truth, route),
                )
            )
    counts = {
        name: dict(lane_used=e.lane_used, lane_rejected=e.lane_rejected)
        for name, e in estimators.items()
        if name != "gnss"
    }
    offsets = {"gnss_lane_offset": estimators["gnss_lane_offset"].gnss_offset_m}
    return out, dict(
        replay_max_difference=replay_difference,
        lane_counts=counts,
        final_gnss_offset_m=offsets,
    )


def _with_offset(aiding):
    return replace(aiding, estimate_gnss_offset=True)


def _without_offset(aiding):
    return replace(aiding, estimate_gnss_offset=False)


def _stats(values):
    values = np.abs(np.asarray(values, dtype=float))
    return dict(
        mean=float(values.mean()), p95=float(np.percentile(values, 95)), max=float(values.max())
    )


def summarise(trajectory, *, goal_window_m=20.0):
    """Errors while moving (same statistics as the run summary), plus the goal approach."""
    moving = [r for r in trajectory if r["speed_mps"] > MOVING_MPS]
    if not moving:
        return None
    end = max(r["progress_m"] for r in trajectory)
    approach = [r for r in moving if r["progress_m"] >= end - goal_window_m]
    final = trajectory[-1]
    return dict(
        moving_samples=len(moving),
        lateral_error_m=_stats([r["lateral_error_m"] for r in moving]),
        along_error_m=_stats([r["along_error_m"] for r in moving]),
        position_error_m=_stats([r["error_m"] for r in moving]),
        yaw_error_deg=_stats([math.degrees(r["yaw_error_rad"]) for r in moving]),
        goal_approach_along_error_m=_stats([r["along_error_m"] for r in approach])
        if approach
        else None,
        final_along_error_m=float(final["along_error_m"]),
        final_lateral_error_m=float(final["lateral_error_m"]),
    )
