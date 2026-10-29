"""TEST-18 deterministic fault injection through the real shared stack.

All images, distances, clocks and controller outputs are synthetic fixtures.
This tests command selection, not physical stopping or remote deployment.
"""
from __future__ import annotations

import builtins
import hashlib
import itertools
import json
import platform
import socket
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from autonomy.freshness import OperationalFreshnessGate
from autonomy.interfaces import EgoPose, FrameBundle, LaneGraph
from autonomy.obstacle_policy import CorridorObservation, TrackedGovernancePolicy
from autonomy.stack import SkeletonStack, StackConfig
from geometry.camera import CameraIntrinsics
from navigation_control.types import ControlCommand, VehicleState
from obstacle_perception.tracking import TrackDepthFilter
from obstacle_perception.types import PixelBox
from perception.frame_detector import FrameDetector
from perception.types import Detection
from planning.speed_governor import GovernorConfig, SpeedGovernor

PROFILES = ("local_fixture", "delayed_fixture", "remote", "hybrid")
BEHAVIOURS = ("cruise", "approach", "hold")
FAULTS = ("missing_depth", "invalid_depth", "missing_capture_clock", "stale_capture",
          "old_sim_frame", "duplicate_frame", "foreign_observation", "old_session_result",
          "wrong_result_frame", "wrong_result_source", "detector_exception", "deadline_miss",
          "generation_restart", "missing_track_id", "invalid_state", "controller_exception",
          "controller_delay", "invalid_command")


class Clock:
    def __init__(self):
        self.now = 100.0
        self.sim = 0.0

    def __call__(self):
        return self.now


class FixtureBackend:
    """Image-backend fake. Delay advances a virtual clock, never sleeps or connects."""
    def __init__(self, clock, profile):
        self.clock, self.profile = clock, profile
        self.generation = 0
        self.fault = None
        self.calls = []
        self.closed = False

    def reset(self, *, episode_id):
        self.episode_id = episode_id
        self.generation += 1

    def _run(self, method):
        self.calls.append(method)
        if self.fault == "detector_exception":
            raise RuntimeError("injected inference outage")
        self.clock.now += 0.2 if self.fault == "deadline_miss" else (
            0.02 if self.profile == "delayed_fixture" else 0.001)
        return (Detection("car", 0.9, PixelBox(0, 0, 40, 40),
                          None if self.fault == "missing_track_id" else 1),)

    def track(self, rgb):
        return self._run("track")

    def detect(self, rgb):
        return self._run("detect")

    def close(self):
        self.closed = True


class FaultFrameDetector(FrameDetector):
    """Inject bad response envelopes after exercising the real frame adapter."""
    def _run(self, frame, tracking):
        batch = super()._run(frame, tracking)
        fault = self.backend.fault
        updates = {"old_session_result": {"episode_id": "old-session"},
                   "wrong_result_frame": {"frame_id": frame.frame_id + 10},
                   "wrong_result_source": {"source_id": "foreign-source"}}
        if fault in updates:
            batch = replace(batch, metadata=replace(batch.metadata, **updates[fault]))
        self.last_result = asdict(batch.metadata)
        return batch


class Rig:
    def __init__(self, profile, behaviour, tracking):
        self.clock = Clock()
        self.behaviour = behaviour
        self.backend = FixtureBackend(self.clock, profile)
        self.gap = None if behaviour == "cruise" else 20.0
        self.closed = False

        def provider(frame, batch, tracks):
            valid = bool(tracks.tracks) and all(t.current and t.depth_valid for t in tracks.tracks)
            return CorridorObservation(batch.metadata, valid, self.gap, 30.0,
                                       clearance_valid=self.gap is None)

        def controller(frame, target, dt):
            fault = self.backend.fault
            if fault == "controller_exception":
                raise RuntimeError("injected controller outage")
            if fault == "controller_delay":
                self.clock.now += 0.2
            if fault == "invalid_command":
                return None
            return ControlCommand(0, 0.5 if target > 0 else 0, 0)

        self.policy = TrackedGovernancePolicy(
            TrackDepthFilter(source_id="fixture-front", model_version="fixture-v1"),
            SpeedGovernor(GovernorConfig(4.0, 1.5, 1.5, 0.95, 6.0, 1.0, 2.0)),
            provider, controller, dt_s=0.05)
        self.config = StackConfig(require_depth=True)
        self.gate = OperationalFreshnessGate(self.config, required_results=2)
        self.detector = FaultFrameDetector(self.backend, source_id="fixture-front", model_version="fixture-v1")
        self.stack = SkeletonStack(self.detector, self.config, self.policy, clock=self.clock,
                                   freshness_gate=self.gate, simulation_clock=lambda: self.clock.sim,
                                   use_tracking=tracking)
        self.stack.reset(episode_id="test18-session", graph=LaneGraph((), ()))
        self.index = 0

    def step(self, fault=None):
        self.backend.fault = fault
        self.detector.last_result = None
        self.clock.now += 0.05
        index = self.index
        self.index += 1
        self.clock.sim = index * 0.05
        frame = FrameBundle(
            "test18-session", index, index * 0.05, self.clock(),
            EgoPose(VehicleState(0, 0, 0, 0)), np.zeros((40, 40, 3), np.uint8),
            CameraIntrinsics(40, 40, 20, 20, 20, 20).calibration(np.eye(4)),
            np.full((40, 40), 20.0), index, self.clock())
        if fault == "missing_depth":
            frame = replace(frame, depth_m=None, depth_frame_id=None)
        elif fault == "invalid_depth":
            frame = replace(frame, depth_m=np.full((40, 40), np.nan))
        elif fault == "missing_capture_clock":
            frame = replace(frame, capture_started_monotonic_s=None)
        elif fault == "stale_capture":
            frame = replace(frame, capture_started_monotonic_s=self.clock() - 0.6)
        elif fault == "old_sim_frame":
            self.clock.sim += 0.2
        elif fault == "duplicate_frame":
            frame = replace(frame, frame_id=index - 1, sim_time_s=(index - 1) * 0.05,
                            depth_frame_id=index - 1)
        elif fault == "foreign_observation":
            frame = replace(frame, episode_id="foreign-session")
        elif fault == "generation_restart":
            self.backend.reset(episode_id="test18-session")
        elif fault == "invalid_state":
            object.__setattr__(frame.ego.state, "speed_mps", float("nan"))
        before = self.policy.guard.clear_count
        packet = self.stack.step(frame)
        return packet, {"input_session": frame.episode_id, "input_frame": frame.frame_id,
                        "source": self.detector.source_id, "session": packet.episode_id,
                        "frame": packet.source_frame_id, "generation": self.detector.generation,
                        "result_metadata": self.detector.last_result,
                        "fault": fault, "decision_reason": list(packet.diagnostics.reasons),
                        "command": asdict(packet.command),
                        "clear_count_before": before, "clear_count_after": self.policy.guard.clear_count,
                        "fresh_results": self.gate.fresh_results,
                        "blocked_latch": self.policy.guard.blocked_latch,
                        "stack_emergency": self.stack._emergency,
                        "stage_durations_s": list(packet.diagnostics.stage_durations_s)}

    def close(self):
        self.stack.close()
        self.closed = True


def full_brake(packet):
    return (packet.command.throttle == 0 and packet.command.brake == 1
            and packet.command.steering_angle_rad == 0)


def run_case(profile, behaviour, tracking, fault):
    if profile not in PROFILES[:2] or behaviour not in BEHAVIOURS or type(tracking) is not bool or fault not in FAULTS:
        raise ValueError("Unsupported fixture combination; remote/hybrid must remain blocked")
    rig = Rig(profile, behaviour, tracking)
    rows = []
    checks = {}
    try:
        # Only this single-threaded fixture's stack uses the patched virtual wall clock.
        with patch("autonomy.stack.time.perf_counter", rig.clock):
            for _ in range(4):
                packet, row = rig.step()
                rows.append(row)
            checks["positive_control"] = packet.command.throttle > 0
            if behaviour == "hold":
                rig.gap = 1.0
                packet, row = rig.step()
                rows.append(row)
                checks["blocker_hold"] = full_brake(packet) and rig.policy.guard.blocked_latch
                rig.gap = None
                for _ in range(2):
                    packet, row = rig.step()
                    rows.append(row)
                checks["partial_clearance"] = rig.policy.guard.clear_count == 2 and full_brake(packet)
            packet, row = rig.step(fault)
            rows.append(row)
            checks["fault_full_brake"] = full_brake(packet)
            checks["diagnostic_reason"] = bool(packet.diagnostics.reasons)
            if fault not in ("controller_exception", "invalid_command"):
                checks["fault_cannot_advance_clearance"] = row["clear_count_after"] <= row["clear_count_before"]
            if behaviour == "hold":
                checks["blocker_retained"] = rig.policy.guard.blocked_latch
            for _ in range(2):
                repeated, repeated_row = rig.step(fault)
                rows.append(repeated_row)
                checks["persistent_fault_full_brake"] = (
                    checks.get("persistent_fault_full_brake", True) and full_brake(repeated))
            # A valid next observation may resume an unblocked scene.
            # Existing HOLD, emergency latches and reset freshness gates must persist.
            recovery_hold_required = (behaviour == "hold" or row["stack_emergency"] is not None
                                      or row["fresh_results"] == 0)
            packet, row = rig.step()
            rows.append(row)
            if recovery_hold_required:
                checks["no_immediate_resume"] = full_brake(packet)
            else:
                checks["fresh_recovery_protected"] = (full_brake(packet) or
                    (packet.diagnostics.observation_valid and rig.policy.observation_valid
                     and rig.gate.fresh_results >= rig.gate.required_results))
            checks["diagnostic_identity"] = all(r["source"] and r["session"] and
                                               type(r["frame"]) is int for r in rows)
    except Exception as error:
        checks["harness_exception"] = False
        rows.append({"exception": f"{type(error).__name__}: {error}"})
    finally:
        rig.close()
        checks["cleanup"] = rig.closed and rig.backend.closed
    return {"profile": profile, "behaviour": behaviour, "tracking": tracking, "fault": fault,
            "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "trace": rows}


def run_suite():
    cases, blocked = [], []
    def no_network(*args, **kwargs):
        raise AssertionError("Network forbidden in TEST-18 offline harness")
    original_import=builtins.__import__
    def no_simulator(name, *args, **kwargs):
        if name=="carla" or name.startswith(("carla.", "carla_adapter.")) or name=="carla_adapter":
            raise RuntimeError("Simulator import forbidden during offline suite")
        return original_import(name, *args, **kwargs)
    with ExitStack() as scope:
        scope.enter_context(patch.object(builtins, "__import__", no_simulator))
        for name in ("connect", "connect_ex"):
            scope.enter_context(patch.object(socket.socket, name, no_network))
        scope.enter_context(patch.object(socket, "create_connection", no_network))
        for profile, behaviour, tracking in itertools.product(PROFILES, BEHAVIOURS, (False, True)):
            if profile in ("remote", "hybrid"):
                blocked.append({"profile": profile, "behaviour": behaviour, "tracking": tracking,
                                "status": "BLOCKED", "reason": "Canonical client/hybrid/server implementations now exist; this fixture profile is not a selected/admitted production flag combination. E4-10 evidence and G3 flag admission pending"})
                continue
            for fault in FAULTS:
                cases.append(run_case(profile, behaviour, tracking, fault))
    failures = sum(c["status"] == "FAIL" for c in cases)
    return {"schema": 1, "status": "FAIL" if failures else "BLOCKED",
            "scope": "Synthetic command-level integration only; not physical stopping, cloud transport, release or admission evidence",
            "python": platform.python_version(), "numpy": np.__version__,
            "fixture_configuration": {"stack": asdict(StackConfig(require_depth=True)),
                                      "governor": asdict(GovernorConfig(4.0, 1.5, 1.5, 0.95, 6.0, 1.0, 2.0)),
                                      "required_fresh_results": 2, "dt_s": 0.05,
                                      "image_shape": [40, 40, 3], "depth_m": 20.0,
                                      "normal_delays_s": {"local_fixture": 0.001, "delayed_fixture": 0.02},
                                      "fault_delay_s": 0.2, "physical_world": False},
            "dependencies": {"G3-01/#48": "PENDING", "E4-10/#58": "PENDING"},
            "enabled_combinations_source": "Harness profiles, not an admitted production flag registry",
            "matrix": {"profiles": PROFILES, "behaviours": BEHAVIOURS,
                       "tracking": (False, True), "faults": FAULTS},
            "counts": {"executed": len(cases), "passed": len(cases) - failures,
                       "failed": failures, "blocked_combinations": len(blocked)},
            "cases": cases, "blocked": blocked}


def write_report(report, output, root):
    """New directory only; preserve every failure and diagnostic row."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for folder in ("src", "configs"):
        for path in sorted((Path(root) / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                inventory[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("scripts/offline_fault_suite.py", "scripts/run_all_flags_faults.py",
                 "tests/unit/test_all_flags_faults.py", ".github/workflows/ci.yml"):
        path = Path(root) / name
        if path.is_file():
            inventory[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(report, source_sha256=inventory)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "diagnostics.jsonl").open("w", encoding="utf-8") as stream:
        for case in report["cases"]:
            identity = {k: case[k] for k in ("profile", "behaviour", "tracking", "status")}
            identity["case_fault"] = case["fault"]
            identity["case_id"] = f"{case['profile']}/{case['behaviour']}/{case['tracking']}/{case['fault']}"
            for row in case["trace"]:
                stream.write(json.dumps(dict(identity, **row), allow_nan=False) + "\n")
    lines = ["# TEST-18 offline fault report", "", f"Overall: **{report['status']}**", "",
             report["scope"], "", str(report["counts"]), "",
             "## Executed fixture cases", "", "| Profile | Behaviour | Track | Fault | Result | Failed checks |",
             "|---|---|---|---|---|---|"]
    for c in report["cases"]:
        failed = ", ".join(k for k, v in c["checks"].items() if not v)
        lines.append(f"| {c['profile']} | {c['behaviour']} | {c['tracking']} | {c['fault']} | {c['status']} | {failed} |")
    lines += ["", "## Blocked combinations", ""]
    for b in report["blocked"]:
        lines.append(f"- {b['profile']} / {b['behaviour']} / tracking={b['tracking']}: {b['reason']}")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
