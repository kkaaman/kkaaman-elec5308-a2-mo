"""Hybrid failures/source changes exercise the real stack, expiry and HOLD policy."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from autonomy.hybrid_detector import HybridFrameDetector, LocalTrackingBackend, hybrid_control_stack
from autonomy.interfaces import CameraCalibration, EgoPose, FrameBundle, LaneGraph
from autonomy.obstacle_policy import CorridorObservation, TrackedGovernancePolicy
from navigation_control.types import ControlCommand, VehicleState
from obstacle_perception.tracking import TrackDepthFilter
from obstacle_perception.types import (
    CallTiming,
    DetectionBatch,
    DetectionObservation,
    ObservationMetadata,
    PixelBox,
)
from perception.async_detector import AsyncDetector
from perception.hybrid_detector import DetectionDecision, HybridDetector
from planning.speed_governor import GovernorConfig, SpeedGovernor


def frame(index):
    now = time.monotonic()
    return FrameBundle(
        "episode",
        index,
        index * 0.05,
        now,
        EgoPose(VehicleState(0, 0, 0, 0)),
        np.zeros((40, 40, 3), np.uint8),
        CameraCalibration(40, 40, 20, 20, ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))),
        np.full((40, 40), 20.0),
        index,
        capture_started_monotonic_s=now,
    )


def batch(value, *, empty=False, model="weights"):
    detections = () if empty else (DetectionObservation("car", 0.9, PixelBox(1, 1, 39, 39), 0),)
    return DetectionBatch(
        ObservationMetadata(value.episode_id, value.frame_id, value.sim_time_s, "worker", model),
        detections,
        40,
        40,
        CallTiming(0),
    )


def policy():
    def observation(value, detected, tracks):
        gap = (
            20.0
            if tracks.tracks and all(t.current and t.depth_valid for t in tracks.tracks)
            else None
        )
        return CorridorObservation(detected.metadata, gap is not None, gap, gap, False)

    return TrackedGovernancePolicy(
        TrackDepthFilter(source_id="initial", model_version="weights"),
        SpeedGovernor(
            GovernorConfig(4.0, 1.497566177392398, 1.5, 0.9500000052154064, 6.0, 1.0, 2.0)
        ),
        observation,
        lambda frame, target, dt: ControlCommand(
            0, 0.5 if target > 0 else 0, 0.5 if target == 0 else 0
        ),
        dt_s=0.05,
    )


class Decisions:
    def __init__(self):
        self.source = "local"
        self.empty = False
        self.model = "weights"
        self.failed = False

    def reset(self, **kwargs):
        pass

    def submit(self, value):
        self.frame = value
        return True, True

    def poll(self):
        return DetectionDecision(
            None if self.failed else batch(self.frame, empty=self.empty, model=self.model),
            "none" if self.failed else self.source,
            self.failed,
            "scripted",
        )

    def close(self):
        pass


def setup(hybrid=None):
    hybrid = hybrid or Decisions()
    motion = policy()
    stack = hybrid_control_stack(
        hybrid,
        motion,
        model_version="weights",
        simulation_clock=lambda: (
            hybrid.frame.sim_time_s if hasattr(hybrid, "frame") and hybrid.frame else 0.0
        ),
    )
    stack.reset(episode_id="episode", graph=LaneGraph((), ()))
    return hybrid, motion, stack.detector, stack


def test_source_id_reuse_and_empty_results_cannot_release_obstacle_hold():
    hybrid, motion, detector, stack = setup()
    try:
        for index in range(3):
            value = frame(index)
            detector.submit(value)
            packet = stack.step(value)
        assert motion.guard.blocked_latch and packet.command.throttle > 0
        hybrid.source = "remote"
        value = frame(3)
        detector.submit(value)
        packet = stack.step(value)
        assert packet.command.brake == 1 and motion.guard.blocked_latch
        assert motion.temporal_filter.source_id == "hybrid:remote"
        hybrid.empty = True
        for index in range(4, 12):
            value = frame(index)
            detector.submit(value)
            assert stack.step(value).command.brake == 1
            assert motion.guard.blocked_latch
        assert detector.last_decision["frame_id"] == 11
        assert detector.last_decision["source"] == "remote"
    finally:
        stack.close()


def test_dual_failure_uses_existing_full_brake_and_packet_expiry():
    hybrid, motion, detector, stack = setup()
    try:
        hybrid.failed = True
        value = frame(0)
        packet = stack.step(value)
        assert packet.command.brake == 1 and packet.command.throttle == 0
        assert not packet.diagnostics.observation_valid
        assert not packet.is_current(
            "episode", value.sim_time_s + 0.11, monotonic_s=time.monotonic() + 0.11
        )
    finally:
        stack.close()


def test_model_mismatch_latches_existing_stack_emergency():
    hybrid, motion, detector, stack = setup()
    try:
        hybrid.model = "unreviewed"
        assert stack.step(frame(0)).command.brake == 1
        hybrid.model = "weights"
        for index in range(1, 5):
            value = frame(index)
            detector.submit(value)
            assert stack.step(value).command.brake == 1
    finally:
        stack.close()


class Worker:
    def __init__(self, gate=None):
        self.gate = gate

    def reset(self, **kwargs):
        pass

    def detect(self, value):
        if self.gate:
            self.gate.wait(2)
        return batch(value)

    def close(self):
        pass


def test_real_bounded_worker_local_fallback_never_waits_for_stalled_remote():
    gate = threading.Event()
    hybrid = HybridDetector(Worker(), AsyncDetector(Worker(gate)), deadline_s=1.0)
    _, motion, detector, stack = setup(hybrid)
    try:
        for index in range(3):
            value = frame(index)
            detector.submit(value)
            assert hybrid.local.work.done.wait(1)  # test orchestration, outside control step
            started = time.monotonic()
            packet = stack.step(value)
            assert time.monotonic() - started < 0.5
        assert packet.command.throttle > 0
        assert detector.last_decision["source"] == "local"
    finally:
        gate.set()
        stack.close()


def test_local_tracking_wrapper_does_not_replace_tracking_with_detection():
    backend = SimpleNamespace(
        track=lambda value: "tracked", reset=lambda **kwargs: None, close=lambda: None
    )
    assert LocalTrackingBackend(backend).detect(object()) == "tracked"


def test_adapter_refuses_policy_that_could_reset_or_replace_safety_contract():
    with pytest.raises(TypeError):
        HybridFrameDetector(Decisions(), object(), model_version="weights")


def test_reused_frame_identity_with_different_image_latches_safe_fault():
    from dataclasses import replace

    hybrid, motion, detector, stack = setup()
    try:
        value = frame(0)
        detector.submit(value)
        bad = replace(value, rgb=np.ones((40, 40, 3), np.uint8))
        assert stack.step(bad).command.brake == 1
        assert stack.step(frame(1)).command.brake == 1
    finally:
        stack.close()


def test_canonical_simulation_expiry_gate_still_blocks_hybrid():
    hybrid, motion, detector, stack = setup()
    try:
        stack.simulation_clock = lambda: 1000.0
        assert stack.step(frame(0)).command.brake == 1
        assert not motion.observation_valid
    finally:
        stack.close()


def test_new_remote_session_without_source_family_change_forces_hold_reconfirmation():
    hybrid, motion, detector, stack = setup()
    hybrid.source = "remote"
    backend = SimpleNamespace(session_id="first")
    hybrid.remote = SimpleNamespace(backend=backend)
    try:
        for index in range(3):
            value = frame(index)
            detector.submit(value)
            stack.step(value)
        assert motion.guard.blocked_latch
        backend.session_id = "recovered"
        value = frame(3)
        detector.submit(value)
        assert stack.step(value).command.brake == 1
        assert motion.guard.blocked_latch
        assert motion.temporal_filter.source_id == "hybrid:remote:recovered"
    finally:
        stack.close()
