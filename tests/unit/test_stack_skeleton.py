"""Multi-cycle stack, emergency latch and lifecycle regression tests."""

import json
import math
from dataclasses import replace

import pytest

from autonomy.interfaces import AutonomyStack, World
from autonomy.stack import SkeletonStack, StackConfig
from navigation_control.types import ControlCommand
from sim_harness.skeleton import ContractWorld, EmptyDetector


def prepared(config=None, policy=None, detector=None):
    world = ContractWorld()
    frame, graph = world.reset(seed=1)
    stack = SkeletonStack(detector or EmptyDetector(), config, policy)
    stack.reset(episode_id=frame.episode_id, graph=graph)
    return world, frame, graph, stack


def assert_stop(packet):
    assert packet.command.throttle == 0
    assert packet.command.brake == 1
    assert packet.command.steering_angle_rad == 0
    assert not packet.diagnostics.observation_valid
    assert packet.diagnostics.reasons
    assert all(
        math.isfinite(t) and t >= 0 for _, t in packet.diagnostics.stage_durations_s
    )


def test_fake_world_one_cycle_protocols_and_reason():
    world, frame, _, stack = prepared()
    assert isinstance(world, World)
    assert isinstance(stack, AutonomyStack)
    packet = stack.step(frame)
    assert packet.diagnostics.observation_valid
    assert packet.diagnostics.reasons == ("skeleton_hold",)
    assert packet.command.brake == 1
    assert "detector" in dict(packet.diagnostics.stage_durations_s)
    assert world.step(packet).frame_id == 1


@pytest.mark.parametrize(
    "config",
    [
        {"command_ttl_s": 0},
        {"require_depth": "false"},
        {"unknown": 1},
        {"max_step_wall_s": float("nan")},
        {"command_ttl_s": True},
        {"command_ttl_s": 1, "max_observation_age_s": 0.1},
    ],
)
def test_bad_config_returns_stop_not_motion(config):
    _, frame, _, stack = prepared(config, lambda f, b: ControlCommand(0, 1, 0))
    assert_stop(stack.step(frame))


def test_missing_config_file_and_json_subset(tmp_path):
    _, frame, _, stack = prepared(tmp_path / "missing.yaml")
    assert_stop(stack.step(frame))
    path = tmp_path / "valid.yaml"
    path.write_text(json.dumps({"require_depth": False}))
    assert StackConfig.load(path).require_depth is False


def test_duplicate_frame_latches_until_episode_reset():
    world, frame, graph, stack = prepared(policy=lambda f, b: ControlCommand(0, 0.2, 0))
    packet = stack.step(frame)
    assert packet.command.throttle == 0.2
    assert_stop(stack.step(frame))
    next_frame = world.step(packet)
    assert_stop(stack.step(next_frame))
    new_frame, graph = world.reset(seed=2)
    stack.reset(episode_id=new_frame.episode_id, graph=graph)
    assert stack.step(new_frame).command.throttle == 0.2


@pytest.mark.parametrize(
    "kind", ["episode", "stale", "missing_depth", "state", "wrong_type"]
)
def test_invalid_observation_stops(kind):
    _, frame, _, stack = prepared({"require_depth": kind == "missing_depth"})
    if kind == "episode":
        frame = replace(frame, episode_id="foreign")
    elif kind == "stale":
        frame = replace(frame, received_monotonic_s=0)
    elif kind == "state":
        object.__setattr__(frame.ego.state, "speed_mps", float("nan"))
    elif kind == "wrong_type":
        frame = None
    assert_stop(stack.step(frame))


@pytest.mark.parametrize("kind", ["invalid", "foreign", "exception"])
def test_bad_detector_result_stops(kind):
    class BadDetector(EmptyDetector):
        def detect(self, frame):
            if kind == "exception":
                raise RuntimeError("inference failed")
            batch = super().detect(frame)
            if kind == "invalid":
                return replace(batch, valid=False, invalid_reason="unavailable")
            return replace(batch, metadata=replace(batch.metadata, frame_id=99))

    _, frame, _, stack = prepared(detector=BadDetector())
    assert_stop(stack.step(frame))


def test_invalid_policy_output_and_processing_expiry():
    _, frame, _, stack = prepared(policy=lambda f, b: None)
    assert_stop(stack.step(frame))
    _, frame, _, stack = prepared(policy=lambda f, b: ControlCommand(0, 0.2, 0))
    calls = iter([frame.received_monotonic_s, frame.received_monotonic_s + 1])
    stack.clock = lambda: next(calls)
    assert_stop(stack.step(frame))


def test_world_rejects_foreign_packet_and_closing_is_idempotent():
    world, frame, _, stack = prepared()
    packet = stack.step(frame)
    with pytest.raises(ValueError):
        world.step(replace(packet, episode_id="foreign"))
    assert world.last_command.brake == 1
    stack.close()
    stack.close()
    world.close()
    with pytest.raises(RuntimeError):
        stack.step(frame)


def test_policy_reset_and_close_hooks():
    class Policy:
        def __init__(self):
            self.events = []

        def reset(self, **kwargs):
            self.events.append(("reset", kwargs["episode_id"]))

        def __call__(self, frame, batch):
            return ControlCommand(0, 0, 1)

        def close(self):
            self.events.append(("close", None))

    policy = Policy()
    _, frame, _, stack = prepared(policy=policy)
    stack.step(frame)
    stack.close()
    assert policy.events == [("reset", "fake-1"), ("close", None)]


def test_wall_budget_and_reset_fault_stop(monkeypatch):
    _, frame, _, stack = prepared()
    times = iter([0.0, 0.0, 0.2, 0.2, 0.2])
    monkeypatch.setattr("autonomy.stack.time.perf_counter", lambda: next(times))
    assert_stop(stack.step(frame))


def test_reset_failure_and_close_attempts_both_components():
    events = []

    class Detector(EmptyDetector):
        def reset(self, **kwargs):
            raise RuntimeError("reset failed")

        def close(self):
            events.append("detector")

    class Policy:
        def __call__(self, frame, batch):
            return ControlCommand(0, 1, 0)

        def close(self):
            events.append("policy")
            raise RuntimeError("policy cleanup failure")

    _, frame, _, stack = prepared(detector=Detector(), policy=Policy())
    assert_stop(stack.step(frame))
    with pytest.raises(RuntimeError, match="cleanup"):
        stack.close()
    assert events == ["policy", "detector"]
