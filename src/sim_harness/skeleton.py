"""Contract-only fake world/detector and one-step runnable BL-F-02 evidence."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from autonomy.interfaces import CameraCalibration, EgoPose, FrameBundle, LaneGraph
from autonomy.stack import SkeletonStack
from navigation_control.types import ControlCommand, VehicleState
from obstacle_perception.types import CallTiming, DetectionBatch, ObservationMetadata


class EmptyDetector:
    def reset(self, *, episode_id):
        self.episode_id = episode_id

    def detect(self, frame):
        if frame.episode_id != self.episode_id:
            raise ValueError("Detector episode mismatch")
        return DetectionBatch(
            ObservationMetadata(
                frame.episode_id, frame.frame_id, frame.sim_time_s, "fake", "empty-v1"
            ),
            (),
            frame.rgb.shape[1],
            frame.rgb.shape[0],
            CallTiming(0),
        )

    def track(self, frame):
        return self.detect(frame)

    def close(self):
        pass


class ContractWorld:
    """Stationary fake observations; not a kinematic or perception simulator."""

    def __init__(self):
        self.closed = False
        self.frame_id = 0
        self.episode_id = None
        self.last_command = ControlCommand(0, 0, 1)

    def _frame(self):
        return FrameBundle(
            self.episode_id,
            self.frame_id,
            self.frame_id * 0.05,
            time.monotonic(),
            EgoPose(VehicleState(0, 0, 0, 0)),
            np.zeros((4, 6, 3), dtype=np.uint8),
            CameraCalibration(4, 4, 3, 2, tuple(map(tuple, np.eye(4)))),
        )

    def reset(self, *, seed):
        if self.closed:
            raise RuntimeError("World closed")
        self.episode_id = f"fake-{seed}"
        self.frame_id = 0
        self.last_command = ControlCommand(0, 0, 1)
        return self._frame(), LaneGraph((), ())

    def step(self, packet):
        if self.closed or self.episode_id is None:
            raise RuntimeError("World is closed or not reset")
        now = self.frame_id * 0.05
        if packet.source_frame_id != self.frame_id or not packet.is_current(
            self.episode_id, now
        ):
            self.last_command = ControlCommand(0, 0, 1)
            raise ValueError("Rejected stale or foreign packet; full brake selected")
        self.last_command = packet.command
        self.frame_id += 1
        return self._frame()

    def close(self):
        self.last_command = ControlCommand(0, 0, 1)
        self.closed = True


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Evidence already exists; use another output name")
    world = ContractWorld()
    stack = SkeletonStack(EmptyDetector(), args.config)
    try:
        frame, graph = world.reset(seed=7)
        stack.reset(episode_id=frame.episode_id, graph=graph)
        packet = stack.step(frame)
        next_frame = world.step(packet)
        report = {
            "packet": asdict(packet),
            "next_frame_id": next_frame.frame_id,
            "status": "PASS"
            if packet.diagnostics.observation_valid
            else "EMERGENCY_STOP",
            "scope": "one stationary fake-world stack cycle; no CARLA or learned model",
        }
    finally:
        try:
            stack.close()
        finally:
            world.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
