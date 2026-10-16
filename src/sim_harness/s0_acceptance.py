"""Offline straight-route S0 harness using the inherited A1 controller.

This is component evidence, not a CARLA or full-autonomy-stack validation.
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np

from autonomy.interfaces import (
    CameraModel,
    ControlPacket,
    Diagnostics,
    DriveMode,
    EvaluationPacket,
    Extrinsics,
    FrameBundle,
    MotionState,
)
from evaluation.metrics import evaluate_run
from evaluation.run_log import RunLogger, provenance, write_json
from navigation_control import ControllerConfig, PathFollower, VehicleState, Waypoint


def run_s0(directory, acceptance, seed, root):
    dt = acceptance["dt_s"]
    if not 0 < dt <= 0.1:
        raise ValueError("S0 dt_s must be in (0, 0.1]")
    config = ControllerConfig(2.8, 0.6, ki=0.0)
    follower = PathFollower(config)
    route = (Waypoint(0.0, 0.0), Waypoint(80.0, 0.0))
    follower.set_path(route)
    rng = random.Random(seed)
    state = VehicleState(0.0, rng.uniform(-0.05, 0.05), 0.0, 5.0)
    camera = CameraModel.from_fov(2, 2, math.pi / 2, Extrinsics(0, 0, 1))
    run_config = {"scenario": "S0", "evidence_kind": "offline A1 controller harness",
                  "acceptance": acceptance, "controller": config,
                  "route_xy": [[0, 0], [80, 0]], "dynamics": {"acceleration_mps2": 2, "braking_mps2": 4}}
    origin = time.perf_counter()
    stopped_since = None
    with RunLogger(directory, config=run_config, seed=seed, metadata=provenance(root), host_origin_s=origin) as logger:
        for frame in range(math.ceil(60 / dt) + 1):
            tick = time.perf_counter()
            sim_time = frame * dt
            ego = MotionState(state.x_m, state.y_m, state.yaw_rad, state.speed_mps)
            bundle = FrameBundle(Path(directory).name, frame, sim_time, tick, time.perf_counter(), ego, camera, np.zeros((2, 2, 3), dtype=np.uint8))
            result = follower.update(state, 5.0, dt)
            d = result.diagnostics
            endpoint = d.internal_target_speed_mps < 5.0
            diagnostic = Diagnostics(frame, sim_time, "GOAL_STOP" if endpoint else "CRUISE", DriveMode.DRIVE,
                                     d.internal_target_speed_mps, state.speed_mps,
                                     {"cruise": 5.0, "endpoint": d.internal_target_speed_mps},
                                     signed_cte_m=state.y_m, route_progress_m=d.progress_m,
                                     timings_ms={"controller": (time.perf_counter() - bundle.host_receive_s) * 1000})
            packet = ControlPacket(frame, result.command, "planned endpoint braking" if endpoint else "open route cruise", host_command_s=time.perf_counter())
            # Evaluation truth is only passed to the logger, never to the controller.
            logger.record(bundle, packet, diagnostic, EvaluationPacket(frame, ego),
                          goal_distance_m=d.distance_to_goal_m, planned_final_stop=endpoint)
            if d.distance_to_goal_m <= acceptance["goal_tolerance_m"] and state.speed_mps <= acceptance["stopped_speed_mps"]:
                if stopped_since is None:
                    stopped_since = sim_time
                if sim_time - stopped_since >= acceptance["stopped_duration_s"] + dt:
                    break
            else:
                stopped_since = None
            command = result.command
            speed = max(0.0, state.speed_mps + (2 * command.throttle - 4 * command.brake) * dt)
            travel = (speed + state.speed_mps) * dt / 2
            yaw = state.yaw_rad + travel * math.tan(command.steering_angle_rad) / config.wheelbase_m
            state = VehicleState(state.x_m + travel * math.cos((state.yaw_rad + yaw) / 2),
                                 state.y_m + travel * math.sin((state.yaw_rad + yaw) / 2), yaw, speed)
    summary = evaluate_run(directory)
    write_json(Path(directory) / "summary.json", summary)
    return summary
