"""Shared episode loop. Evaluation truth is logged after control computation."""

import time
from dataclasses import asdict

from autonomy.baseline import BaselinePolicy
from autonomy.stack import SkeletonStack, StackConfig
from evaluation.metrics import summarize
from evaluation.run_logger import RunLogger, json_text


def run_episode(
    world, detector, points, config, folder, manifest, *, max_sim_s=120, navigation_only=False
):
    policy = BaselinePolicy(points, config, navigation_only=navigation_only)
    stack = SkeletonStack(
        detector,
        StackConfig(
            command_ttl_s=0.3,
            max_observation_age_s=0.5,
            max_step_wall_s=manifest.get("step_wall_budget_s", 0.1),
            require_depth=not navigation_only,
        ),
        policy,
        clock=getattr(world, "control_clock", time.monotonic),
    )
    logger = RunLogger(
        folder,
        dict(
            manifest,
            dt_s=config.dt_s,
            cruise_mps=config.governor.cruise_mps,
            config=asdict(config),
            goal_tolerance_m=config.controller.goal_tolerance_m,
        ),
    )
    rows, events, error = [], [], None
    completed = False
    start = time.monotonic()
    cruise_since = None
    hold_since = None
    try:
        logger.manifest["guard_clock_scope"] = (
            "fixture frozen within step" if hasattr(world, "control_clock") else "host monotonic"
        )
        frame, graph = world.reset(seed=manifest["seed"])
        stack.reset(episode_id=frame.episode_id, graph=graph)
        if hasattr(detector, "warmup"):
            logger.manifest["detector_warmup_s"] = detector.warmup()
            if hasattr(world, "refresh"):
                frame = world.refresh()
        initial_time = frame.sim_time_s
        while frame.sim_time_s - initial_time <= max_sim_s:
            packet = stack.step(frame)
            projection = policy.route.project(frame.ego.state.x_m, frame.ego.state.y_m)
            goal = policy.route.points[-1]
            policy_record = dict(
                speed_mps=frame.ego.state.speed_mps,
                internal_target_mps=0.0,
                cte_m=projection.distance_m,
                signed_cte_m=projection.lateral_m,
                distance_to_goal_m=float(
                    ((goal[0] - frame.ego.state.x_m) ** 2 + (goal[1] - frame.ego.state.y_m) ** 2)
                    ** 0.5
                ),
                **{
                    k: v
                    for k, v in policy.last.items()
                    if k
                    not in {
                        "speed_mps",
                        "internal_target_mps",
                        "cte_m",
                        "signed_cte_m",
                        "distance_to_goal_m",
                    }
                },
            )
            if policy.last:
                policy_record["internal_target_mps"] = policy.last["internal_target_mps"]
            fault = not packet.diagnostics.observation_valid
            if fault:
                policy_record.update(
                    state="FAULT_STOP",
                    reason=";".join(packet.diagnostics.reasons),
                    target_mps=0.0,
                    finished=False,
                )
            # Evaluation-only information is fetched after policy execution.
            truth = world.evaluation(frame.frame_id)
            next_frame = world.step(packet)
            applied_at = getattr(world, "last_applied_monotonic_s", time.monotonic())
            age = getattr(world, "last_simulation_age_s", 0.0)
            if not policy_record:
                raise RuntimeError("No policy diagnostics")
            record = dict(
                policy_record,
                **truth,
                frame_id=frame.frame_id,
                sim_time_s=frame.sim_time_s,
                ego=asdict(frame.ego.state),
                command=asdict(packet.command),
                capture_apply_s=max(
                    0.0,
                    applied_at
                    - (
                        world.capture_monotonic(frame.frame_id)
                        if hasattr(world, "capture_monotonic")
                        else frame.received_monotonic_s
                    ),
                ),
                callback_apply_s=max(0.0, applied_at - frame.received_monotonic_s),
                simulation_age_s=age,
                stages=dict(packet.diagnostics.stage_durations_s),
                observation_valid=not fault,
            )
            steady = (
                record["state"] == "CRUISE"
                and record["distance_to_goal_m"] > 15
                and abs(record["internal_target_mps"] - config.governor.cruise_mps) < 0.05
            )
            cruise_since = (
                (frame.sim_time_s if cruise_since is None else cruise_since) if steady else None
            )
            record["cruise_window"] = (
                cruise_since is not None and frame.sim_time_s - cruise_since >= 2.0
            )
            logger.sample(record)
            rows.append(record)
            if hasattr(world, "record_visual"):
                world.record_visual(frame, record)
            if record["state"] == "HOLD" and frame.ego.state.speed_mps <= 0.1:
                hold_since = frame.sim_time_s if hold_since is None else hold_since
            else:
                hold_since = None
            if record["finished"] or (
                manifest["scenario"] == "S4"
                and hold_since is not None
                and frame.sim_time_s - hold_since >= 60.0
            ):
                completed = True
                break
            if fault and next_frame.ego.state.speed_mps <= 0.1:
                error = "Latched stack fault: " + record["reason"]
                break
            frame = next_frame
        if not completed and error is None:
            error = "Simulation time budget reached without scenario completion"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for component in (world, stack):
            try:
                component.close()
            except Exception as exc:
                error = (error + "; " if error else "") + f"cleanup: {exc}"
        events = list(getattr(world, "events", []))
        for event in events:
            logger.event(event)
        logger.manifest["loop_wall_s"] = time.monotonic() - start
        logger.manifest.update(getattr(world, "manifest_details", {}))
        logger.close(
            completed=completed, cleanup=getattr(world, "cleanup", {"ok": False}), error=error
        )
    summary = summarize(rows, events, logger.manifest)
    summary.update(
        collection_status=logger.manifest["status"],
        error=error,
        completed=completed,
        path_installations=policy.path_installations,
    )
    (logger.folder / "summary.json").write_text(json_text(summary) + "\n", encoding="utf-8")
    return summary
