"""BL-F-03 dynamics, command boundary and real controller-loop checks."""

import math
from dataclasses import replace

import numpy as np
import pytest

from autonomy.interfaces import ControlPacket, Diagnostics, World
from autonomy.stack import SkeletonStack, StackConfig
from navigation_control import PathFollower
from navigation_control.types import (
    ControlCommand,
    ControllerConfig,
    ControlStatus,
    VehicleState,
    Waypoint,
)
from sim_harness import KinematicConfig, KinematicWorld, SyntheticDetector


def packet(frame, command=None):
    return ControlPacket(
        frame.episode_id,
        frame.frame_id,
        frame.sim_time_s,
        frame.sim_time_s + 0.1,
        command or ControlCommand(0, 0, 0),
        Diagnostics(),
    )


@pytest.mark.parametrize("name", KinematicConfig.__dataclass_fields__)
@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "0.05"])
def test_config_rejects_invalid_values(name, value):
    with pytest.raises(ValueError):
        KinematicConfig(**{name: value})


def test_singular_steering_and_bad_constructor_types_rejected():
    with pytest.raises(ValueError):
        KinematicConfig(max_steering_angle_rad=math.pi / 2)
    for kwargs in (
        {"config": {}},
        {"initial_state": ()},
        {"graph": ()},
        {"scenario": ()},
        {"clock": 0},
    ):
        with pytest.raises(TypeError):
            KinematicWorld(**kwargs)


def test_straight_acceleration_matches_analytic_distance_and_fixed_clock():
    world = KinematicWorld(clock=lambda: 10.0)
    assert isinstance(world, World)
    frame, graph = world.reset(seed=7)
    assert frame.frame_id == 0 and frame.sim_time_s == 0
    assert frame.ego.state == VehicleState(0, 0, 0, 0)
    assert not graph.edges
    for tick in range(1, 41):
        frame = world.step(packet(frame, ControlCommand(0, 0.5, 0)))
        elapsed = tick * world.config.dt_s
        assert frame.frame_id == tick
        assert frame.sim_time_s == elapsed
        assert frame.received_monotonic_s == 10
        assert frame.ego.state.x_m == pytest.approx(0.5 * elapsed**2, abs=1e-12)
        assert frame.ego.state.speed_mps == pytest.approx(elapsed, abs=1e-12)
        assert frame.ego.state.y_m == 0 and frame.ego.state.yaw_rad == 0
    assert not frame.rgb.flags.writeable and not frame.depth_m.flags.writeable


@pytest.mark.parametrize("angle", [0.3, -0.3, 1e-12])
def test_constant_speed_curve_matches_circle_and_steering_sign(angle):
    world = KinematicWorld(initial_state=VehicleState(1, 2, 0.4, 3))
    frame, _ = world.reset(seed=2)
    for _ in range(40):
        frame = world.step(packet(frame, ControlCommand(angle, 0, 0)))
    distance = 6.0
    curvature = math.tan(angle) / world.config.wheelbase_m
    turn = distance * curvature
    half_turn = turn / 2
    chord = distance * math.sin(half_turn) / half_turn
    state = frame.ego.state
    assert state.x_m == pytest.approx(1 + chord * math.cos(0.4 + half_turn), abs=1e-11)
    assert state.y_m == pytest.approx(2 + chord * math.sin(0.4 + half_turn), abs=1e-11)
    assert state.yaw_rad == pytest.approx(0.4 + turn)
    assert state.speed_mps == 3


@pytest.mark.parametrize("sign", [-1, 1])
def test_world_steering_limit_is_applied(sign):
    world = KinematicWorld(initial_state=VehicleState(0, 0, 0, 2))
    frame, _ = world.reset(seed=0)
    frame = world.step(packet(frame, ControlCommand(sign * 1.2, 0, 0)))
    angle = sign * world.config.max_steering_angle_rad
    assert world.last_command.steering_angle_rad == angle
    assert frame.ego.state.yaw_rad == pytest.approx(2 * 0.05 * math.tan(angle) / 2.8)


def test_braking_inside_tick_holds_without_reverse_then_resumes():
    world = KinematicWorld(initial_state=VehicleState(0, 0, 0, 0.1))
    frame, _ = world.reset(seed=0)
    frame = world.step(packet(frame, ControlCommand(0, 0, 1)))
    # 0.025 s to stop; the remainder of the 0.05 s step stays stationary.
    assert frame.ego.state.x_m == pytest.approx(0.1**2 / (2 * 4))
    assert frame.ego.state.speed_mps == 0
    stopped = frame.ego.state
    for _ in range(10):
        frame = world.step(packet(frame, ControlCommand(0.4, 0, 1)))
        assert frame.ego.state == stopped
    frame = world.step(packet(frame, ControlCommand(0, 1, 0)))
    assert frame.ego.state.speed_mps > 0
    assert frame.ego.state.x_m > stopped.x_m


def test_temporary_path_follower_stop_and_resume_preserve_route_progress():
    world = KinematicWorld()
    controller = PathFollower(ControllerConfig(wheelbase_m=2.8, max_steering_angle_rad=0.5))
    controller.set_path([Waypoint(0, 0), Waypoint(60, 0)])
    frame, _ = world.reset(seed=0)
    results = []
    for target_speed, ticks in ((3.0, 100), (0.0, 200), (3.0, 50)):
        for _ in range(ticks):
            result = controller.update(frame.ego.state, target_speed, world.config.dt_s)
            frame = world.step(packet(frame, result.command))
        results.append((frame.ego.state, result))
    assert results[0][0].speed_mps > 1
    assert results[1][0].speed_mps == 0
    assert results[1][1].status == ControlStatus.STOPPING
    assert results[2][0].speed_mps > 1
    progress = [result.diagnostics.progress_m for _, result in results]
    assert progress[0] <= progress[1] < progress[2]
    assert results[2][1].status != ControlStatus.FINISHED


@pytest.mark.parametrize(
    "kind",
    [
        "foreign",
        "future_frame",
        "old_frame",
        "future_time",
        "old_time",
        "expired",
        "wrong_type",
        "damaged_command",
    ],
)
def test_rejected_packet_selects_brake_without_consuming_tick(kind):
    world = KinematicWorld(initial_state=VehicleState(0, 0, 0, 2))
    frame, _ = world.reset(seed=1)
    old_packet = packet(frame)
    frame = world.step(old_packet)
    command = packet(frame, ControlCommand(0, 1, 0))
    if kind == "foreign":
        command = replace(command, episode_id="foreign")
    elif kind == "future_frame":
        command = replace(command, source_frame_id=99)
    elif kind == "old_frame":
        command = old_packet
    elif kind == "future_time":
        command = replace(command, source_sim_time_s=frame.sim_time_s + 0.01)
    elif kind == "old_time":
        command = replace(command, source_sim_time_s=0)
    elif kind == "expired":
        command = replace(command, source_sim_time_s=0, valid_until_sim_time_s=0.01)
    elif kind == "wrong_type":
        command = None
    elif kind == "damaged_command":
        object.__setattr__(command.command, "throttle", float("nan"))
    with pytest.raises((ValueError, TypeError)):
        world.step(command)
    assert world.last_command == ControlCommand(0, 0, 1)
    recovered = world.step(packet(frame, ControlCommand(0, 0, 1)))
    assert recovered.frame_id == frame.frame_id + 1
    assert recovered.ego.state.speed_mps == pytest.approx(1.8)


def test_exact_expiry_and_emergency_brake_packet_are_accepted():
    world = KinematicWorld()
    frame, _ = world.reset(seed=1)
    command = replace(
        packet(frame, ControlCommand(0, 0, 1)),
        valid_until_sim_time_s=frame.sim_time_s,
        diagnostics=Diagnostics(observation_valid=False, reasons=("test_fault",)),
    )
    assert world.step(command).frame_id == 1


@pytest.mark.parametrize("seed", [-1, True, 1.0, None])
def test_reset_seed_validation(seed):
    with pytest.raises(ValueError):
        KinematicWorld().reset(seed=seed)


def test_reset_has_fresh_episode_replays_physics_and_blocks_old_commands():
    world = KinematicWorld()

    def trace():
        frame, _ = world.reset(seed=7)
        first = frame
        states = []
        for index in range(30):
            frame = world.step(packet(frame, ControlCommand(0.2, 0.5 if index < 20 else 0, 0)))
            states.append(frame.ego.state)
        return first, states

    first, states = trace()
    second, repeated = trace()
    assert states == repeated
    assert first.episode_id != second.episode_id
    np.testing.assert_equal(first.depth_m, second.depth_m)
    frame, _ = world.reset(seed=7)
    with pytest.raises(ValueError):
        world.step(packet(first))
    assert world.step(packet(frame)).frame_id == 1


def test_lifecycle_and_failed_sensor_capture_leave_state_recoverable():
    now = [1.0]
    world = KinematicWorld(clock=lambda: now[0])
    with pytest.raises(RuntimeError):
        world.step(None)
    frame, _ = world.reset(seed=0)
    now[0] = float("nan")
    with pytest.raises(ValueError):
        world.step(packet(frame, ControlCommand(0, 1, 0)))
    assert world.last_command.brake == 1
    now[0] = 2.0
    assert world.step(packet(frame)).ego.state.speed_mps == 0
    world.close()
    world.close()
    assert world.last_command.brake == 1
    with pytest.raises(RuntimeError):
        world.reset(seed=0)
    with pytest.raises(RuntimeError):
        world.step(packet(frame))


@pytest.mark.parametrize("curve", [False, True])
def test_path_follower_stack_completes_straight_or_curve_deterministically(curve):
    if curve:
        points = [
            Waypoint(20 * math.sin(t), 20 * (1 - math.cos(t)))
            for t in np.linspace(0, math.pi / 2, 81)
        ]
    else:
        points = [Waypoint(0, 0), Waypoint(40, 0)]

    class Policy:
        def __init__(self):
            self.controller = PathFollower(
                ControllerConfig(
                    wheelbase_m=2.8,
                    max_steering_angle_rad=0.5,
                    lookahead_m=3.0,
                )
            )
            self.errors = []
            self.status = None
            self.path_sets = 0

        def reset(self, **kwargs):
            self.controller.set_path(points)
            self.path_sets += 1

        def __call__(self, frame, detections):
            result = self.controller.update(frame.ego.state, 3.0, 0.05)
            self.errors.append(result.diagnostics.cross_track_error_m)
            self.status = result.status
            return result.command

    traces = []
    for _ in range(2):
        # A shared injected receipt clock makes wall-clock freshness independent
        # of CI load; SkeletonStack still measures actual per-step durations.
        world = KinematicWorld(clock=lambda: 10.0)
        policy = Policy()
        stack = SkeletonStack(
            SyntheticDetector(world.scenario),
            StackConfig(max_step_wall_s=5),
            policy,
            clock=lambda: 10.0,
        )
        states = []
        try:
            frame, graph = world.reset(seed=42)
            stack.reset(episode_id=frame.episode_id, graph=graph)
            for _ in range(2000):
                command = stack.step(frame)
                assert command.diagnostics.observation_valid, command.diagnostics.reasons
                frame = world.step(command)
                states.append(frame.ego.state)
                if policy.status == ControlStatus.FINISHED:
                    break
            assert policy.status == ControlStatus.FINISHED
            assert policy.path_sets == 1
            assert max(policy.errors) < 0.5
            assert frame.ego.state.speed_mps <= 0.1
            assert (
                math.hypot(
                    frame.ego.state.x_m - points[-1].x_m, frame.ego.state.y_m - points[-1].y_m
                )
                <= 1.0
            )
        finally:
            stack.close()
            world.close()
        traces.append(states)
    assert traces[0] == traces[1]
