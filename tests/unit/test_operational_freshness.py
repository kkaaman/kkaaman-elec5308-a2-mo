"""Cold, stale, reset and expired observations cannot authorise positive motion."""

import numpy as np
import pytest

from autonomy.interfaces import EgoPose, FrameBundle, LaneGraph
from autonomy.obstacle_policy import CorridorObservation, TrackedGovernancePolicy
from autonomy.operational import operational_stack
from autonomy.stack import StackConfig
from geometry.camera import CameraIntrinsics
from navigation_control.types import ControlCommand, VehicleState
from obstacle_perception.tracking import TrackDepthFilter
from perception.types import Detection
from planning.speed_governor import GovernorConfig, SpeedGovernor


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


class Backend:
    def __init__(self, clock):
        self.generation = 0
        self.clock = clock
        self.duration = 0.01
        self.track_id = 0

    def reset(self, *, episode_id):
        self.episode_id = episode_id
        self.generation += 1

    def track(self, rgb):
        self.clock.value += self.duration
        from obstacle_perception.types import PixelBox

        return (Detection("car", 0.9, PixelBox(0, 0, 40, 40), self.track_id),)

    def close(self):
        pass


def setup():
    clock = Clock()
    backend = Backend(clock)
    sim = [0.0]

    def provider(frame, batch, tracks):
        valid = bool(tracks.tracks and all(t.depth_valid for t in tracks.tracks))
        return CorridorObservation(
            batch.metadata, valid, 20.0 if valid else None, 20.0 if valid else None
        )

    policy = TrackedGovernancePolicy(
        TrackDepthFilter(source_id="rgb", model_version="model"),
        SpeedGovernor(GovernorConfig(4.0, 1.497566177392398, 1.5, 0.95, 6.0, 1.0, 2.0)),
        provider,
        lambda frame, target, dt: ControlCommand(0, 0.5 if target > 0 else 0, 0),
        dt_s=0.05,
    )
    stack = operational_stack(
        backend,
        policy,
        source_id="rgb",
        model_version="model",
        simulation_clock=lambda: sim[0],
        config=StackConfig(require_depth=True),
    )
    stack.clock = clock
    stack.reset(episode_id="e", graph=LaneGraph((), ()))

    def step(index, capture_age=0.0, sim_age=0.0):
        clock.value += 0.05
        sim[0] = index * 0.05 + sim_age
        frame = FrameBundle(
            "e",
            index,
            index * 0.05,
            clock(),
            EgoPose(VehicleState(0, 0, 0, 0)),
            np.zeros((40, 40, 3), np.uint8),
            CameraIntrinsics(40, 40, 20, 20, 20, 20).calibration(np.eye(4)),
            np.full((40, 40), 20.0),
            index,
            clock() - capture_age,
        )
        return stack.step(frame)

    return clock, backend, policy, stack, step


def test_startup_hold_then_fresh_confirmed_governance():
    clock, backend, policy, stack, step = setup()
    assert step(0).command.throttle == 0
    packet = step(1)
    assert packet.command.throttle > 0
    assert packet.is_current("e", 0.05, monotonic_s=clock())
    assert not packet.is_current("e", 0.05)
    assert not packet.is_current("e", 0.05, monotonic_s=clock() + 1)


def test_cold_result_never_counts_as_warm():
    _, backend, _, _, step = setup()
    backend.duration = 0.6
    assert step(0).command.throttle == 0
    backend.duration = 0.01
    assert step(1).command.throttle == 0
    assert step(2).command.throttle > 0


@pytest.mark.parametrize("ages", [(0.6, 0.0), (0.0, 0.2)])
def test_stale_capture_or_old_sim_frame_cannot_be_relabelled_fresh(ages):
    _, _, _, _, step = setup()
    packet = step(0, capture_age=ages[0], sim_age=ages[1])
    assert packet.command.throttle == 0
    assert not packet.diagnostics.observation_valid


def test_reset_generation_and_episode_reset_require_warmup_again():
    _, backend, _, stack, step = setup()
    step(0)
    assert step(1).command.throttle > 0
    backend.reset(episode_id="e")
    assert step(2).command.throttle == 0
    stack.reset(episode_id="e", graph=LaneGraph((), ()))
    assert step(3).command.throttle == 0
    assert step(4).command.throttle > 0


def test_late_controller_result_is_rejected_after_governance():
    clock, _, policy, _, step = setup()
    step(0)

    def late(frame, target, dt):
        clock.value += 0.6
        return ControlCommand(0, 1.0, 0)

    policy.controller = late
    packet = step(1)
    assert packet.command.throttle == 0 and packet.command.brake == 1
    assert not packet.diagnostics.observation_valid


def test_missing_ids_and_depth_never_clear_hold():
    _, backend, policy, _, step = setup()
    step(0)
    backend.track_id = None
    assert step(1).command.throttle == 0
    assert policy.last_target_mps == 0


def test_missing_capture_clock_cannot_authorise_motion():
    clock, _, _, stack, _ = setup()
    frame = FrameBundle(
        "e",
        0,
        0,
        clock(),
        EgoPose(VehicleState(0, 0, 0, 0)),
        np.zeros((40, 40, 3), np.uint8),
        CameraIntrinsics(40, 40, 20, 20, 20, 20).calibration(np.eye(4)),
        np.full((40, 40), 20.0),
        0,
    )
    assert stack.step(frame).command.throttle == 0


def test_hold_and_emergency_precedence_unchanged():
    _, _, policy, _, step = setup()
    step(0)
    assert step(1).command.throttle > 0

    def emergency(frame, batch, tracks):
        return CorridorObservation(batch.metadata, True, 1.0, 1.0)

    policy.observation_provider = emergency
    packet = step(2)
    assert packet.command.throttle == 0 and packet.command.brake == 1
    assert policy.last_emergency


def test_actuator_rejects_delayed_positive_packet():
    from carla_adapter.vehicle import FixedVehicleAdapter

    clock, _, _, _, step = setup()
    step(0)
    packet = step(1)
    calls = []
    adapter = FixedVehicleAdapter.__new__(FixedVehicleAdapter)
    adapter.stop = lambda: calls.append("stop")
    adapter.apply = lambda command, client=None: calls.append("positive")
    with pytest.raises(ValueError, match="Expired"):
        adapter.apply_packet(packet, episode_id="e", sim_time_s=0.05, clock=lambda: clock() + 1.0)
    assert calls == ["stop"]


def test_frozen_baseline_entry_point_uses_the_same_startup_gate(monkeypatch, tmp_path):
    import json
    from pathlib import Path

    from autonomy.baseline_operation import baseline_operational_stack, frozen_governor_config
    from obstacle_perception.box_depth import DepthPolicy
    from obstacle_perception.tracking import TrackPolicy
    from perception.types import PerceptionConfig

    root = Path(__file__).resolve().parents[2]
    document = json.loads((root / "configs/mission_baseline.json").read_text())
    freeze = document["baseline_threshold_freeze"]
    # Exercise a hypothetical accepted loader; the real baseline must stay blocked.
    freeze["status"] = "FROZEN_GATED_OPERATIONAL"
    freeze["frozen_values"] = dict(freeze["candidate_values"])
    path = tmp_path / "hypothetical-accepted-baseline.json"
    path.write_text(json.dumps(document))
    import perception.object_detection as detector_module
    import perception.recorded_detector as assets

    tracker_file = tmp_path / "tracker.yaml"
    tracker_file.write_text(json.dumps(freeze["detector"]["tracker_settings"]))
    monkeypatch.setattr(assets, "verify_local_asset", lambda path, sha: Path(path))
    monkeypatch.setattr(detector_module, "_resolve_tracker", lambda name: str(tracker_file))
    clock, backend, policy, _, _ = setup()
    backend.config = PerceptionConfig(model_name=str(root / "yolo26n.pt"))
    policy.governor = SpeedGovernor(frozen_governor_config(path))
    policy.temporal_filter = TrackDepthFilter(
        source_id="rgb",
        model_version=freeze["detector"]["model_sha256"],
        policy=TrackPolicy(**freeze["track_policy"]),
        depth_policy=DepthPolicy(**freeze["depth_policy"]),
    )
    sim = [0.0]
    stack = baseline_operational_stack(
        backend,
        policy,
        baseline_path=path,
        source_id="rgb",
        model_version=freeze["detector"]["model_sha256"],
        simulation_clock=lambda: sim[0],
    )
    stack.clock = clock
    stack.reset(episode_id="e", graph=LaneGraph((), ()))
    for i in range(2):
        clock.value += 0.05
        sim[0] = i * 0.05
        frame = FrameBundle(
            "e",
            i,
            sim[0],
            clock(),
            EgoPose(VehicleState(0, 0, 0, 0)),
            np.zeros((360, 640, 3), np.uint8),
            CameraIntrinsics(640, 360, 320, 320, 320, 180).calibration(np.eye(4)),
            np.full((360, 640), 20.0),
            i,
            clock(),
        )
        packet = stack.step(frame)
        assert (packet.command.throttle > 0) == (i == 1)
