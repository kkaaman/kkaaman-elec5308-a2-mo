"""Offline S0 integration: shared KinematicWorld -> SkeletonStack -> world.

The detector is empty by design; this is not learned-perception/CARLA evidence.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from autonomy.interfaces import GraphEdge, GraphNode, LaneGraph
from autonomy.stack import SkeletonStack, StackConfig
from evaluation.metrics import evaluate_run, validate_s0_criteria
from evaluation.run_log import RunLogger, provenance, write_json
from navigation_control import ControllerConfig, PathFollower, VehicleState, Waypoint
from sim_harness.kinematic import KinematicConfig, KinematicWorld
from sim_harness.skeleton import EmptyDetector
from sim_harness.synthetic import SyntheticScenario


class S0MotionPolicy:
    """A1 follower on a predeclared open route, with logged decision context."""

    def __init__(self, config, dt_s):
        self.follower = PathFollower(config)
        self.dt_s = dt_s
        self.result = None

    def reset(self, *, episode_id, graph):
        self.follower.set_path(Waypoint(*p) for p in graph.edges[0].points_xy_m)
        self.result = None

    def __call__(self, frame, batch):
        if not batch.valid or batch.detections:
            raise ValueError("S0 requires a valid empty observation batch")
        self.result = self.follower.update(frame.ego.state, 5.0, self.dt_s)
        return self.result.command


def run_s0(directory, acceptance, seed, root):
    definitions = json.loads((Path(root) / "tests/fixtures/acceptance_definitions.json").read_text(encoding="utf-8"))
    validate_s0_criteria(acceptance, definitions)
    dt = acceptance["dt_s"]
    if not 0 < dt <= 0.1:
        raise ValueError("S0 dt_s must be in (0, 0.1]")
    config = ControllerConfig(2.8, 0.6, ki=0.0)
    policy = S0MotionPolicy(config, dt)
    graph = LaneGraph((GraphNode("start", 0, 0), GraphNode("goal", 80, 0)),
                      (GraphEdge("start", "goal", ((0, 0), (80, 0)), 5.0),))
    rng = random.Random(seed)
    state = VehicleState(0.0, rng.uniform(-0.05, 0.05), 0.0, 5.0)
    dynamics = KinematicConfig(dt_s=dt)
    stack_config = StackConfig()
    world = KinematicWorld(dynamics, initial_state=state, graph=graph,
                           scenario=SyntheticScenario(image_width=8, image_height=6))
    stack = SkeletonStack(EmptyDetector(), stack_config, policy)
    run_config = {"scenario": "S0", "evidence_kind": "offline KinematicWorld/SkeletonStack with A1 follower",
                  "acceptance": acceptance, "acceptance_definitions": definitions,
                  "controller": config, "stack": stack_config,
                  "route_xy": [[0, 0], [80, 0]], "dynamics": dynamics,
                  "latency_boundary": "receipt to world.step return (conservative application upper bound)",
                  "cte_window": "all sampled S0 rear-axle states through final stopped window"}
    metadata = provenance(root)
    origin = time.monotonic()
    stopped_since = None
    try:
        with RunLogger(directory, config=run_config, seed=seed, metadata=metadata, host_origin_s=origin) as logger:
            frame, graph = world.reset(seed=seed)
            stack.reset(episode_id=frame.episode_id, graph=graph)
            for _ in range(math.ceil(60 / dt) + 1):
                packet = stack.step(frame)
                # Apply the command, then sample the SAME host clock as receipt.
                next_frame = world.step(packet)
                applied = time.monotonic()
                result = policy.result
                valid = packet.diagnostics.observation_valid and result is not None
                if result is None:
                    raise RuntimeError("S0 policy never produced a command: " + "; ".join(packet.diagnostics.reasons))
                d = result.diagnostics
                endpoint = d.internal_target_speed_mps < 5.0
                diagnostic = {"frame_id": frame.frame_id, "sim_time_s": frame.sim_time_s,
                              "behaviour": "FAULT" if not valid else "GOAL_STOP" if endpoint else "CRUISE",
                              "target_speed_mps": d.internal_target_speed_mps,
                              "actual_speed_mps": frame.ego.state.speed_mps,
                              "speed_caps_mps": {"cruise": 5.0, "endpoint": d.internal_target_speed_mps},
                              "signed_cte_m": frame.ego.state.y_m, "route_progress_m": d.progress_m,
                              "nearest_gap_m": None, "clear_count": 0,
                              "timings_ms": {k: v * 1000 for k, v in packet.diagnostics.stage_durations_s},
                              "reason": "stack fault" if not valid else "planned endpoint braking" if endpoint else "open route cruise",
                              "observation_valid": valid}
                # This world models no contacts or lane sensors. Empty events are
                # explicit synthetic evidence, not proof of CARLA collision safety.
                evaluation = {"frame_id": frame.frame_id, "ego_truth": frame.ego.state,
                              "collisions": [], "lane_invasions": [], "events": [],
                              "source": "obstacle-free kinematic fixture; contacts not simulated"}
                logger.record(frame, packet, diagnostic, evaluation,
                              host_applied_monotonic_s=applied,
                              applied_sim_time_s=frame.sim_time_s,
                              goal_distance_m=d.distance_to_goal_m, planned_final_stop=endpoint)
                if not valid:
                    break
                if d.distance_to_goal_m <= acceptance["goal_tolerance_m"] and frame.ego.state.speed_mps <= acceptance["stopped_speed_mps"]:
                    if stopped_since is None:
                        stopped_since = frame.sim_time_s
                    if frame.sim_time_s - stopped_since >= acceptance["stopped_duration_s"] + dt:
                        break
                else:
                    stopped_since = None
                frame = next_frame
    finally:
        try:
            stack.close()
        finally:
            world.close()
    summary = evaluate_run(directory)
    write_json(Path(directory) / "summary.json", summary)
    return summary
