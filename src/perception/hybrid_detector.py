"""E4-05 concurrent inference with fresh same-frame local fallback."""
from __future__ import annotations

import time
from dataclasses import dataclass

from obstacle_perception.types import DetectionBatch
from perception.async_detector import AsyncDetector, positive


@dataclass(frozen=True)
class DetectionDecision:
    batch: DetectionBatch | None
    source: str
    stop_required: bool
    reason: str


class HybridDetector:
    """Owns two bounded workers; submits local and remote on every frame.

    A pending/expired/invalid observation requests controlled stop. A valid
    local completion is usable on the first poll that sees it, without waiting
    for HTTP. Recovery requires N consecutive distinct timely remote frames.
    The caller must enforce stop_required through its normal control path.
    This optional component is not installed into the G1 admitted runner.
    """
    def __init__(self, local_backend, remote_worker, *, deadline_s=0.1,
                 recovery_frames=3, clock=time.monotonic):
        if type(recovery_frames) is not int or recovery_frames < 1:
            raise ValueError("recovery_frames must be positive")
        self.clock, self.deadline_s = clock, positive(deadline_s, "deadline_s")
        self.local = AsyncDetector(local_backend, clock=clock)
        self.remote = remote_worker
        self.recovery_frames = recovery_frames
        self.frame = None
        self.remote_enabled = True
        self.streak = 0
        self.counted = False

    def reset(self, *, episode_id):
        self.local.reset(episode_id=episode_id)
        self.remote.reset(episode_id=episode_id)
        self.frame = None
        self.remote_enabled, self.streak, self.counted = True, 0, False

    def submit(self, frame):
        if self.frame is not None:
            if frame.frame_id <= self.frame.frame_id:
                raise ValueError("Frames must increase; reset at episode boundary")
            if not self.counted:
                self.streak = 0
                self.remote_enabled = False
        self.frame = frame
        self.deadline = self.clock() + self.deadline_s
        self.counted = False
        # Never short-circuit: local remains warm even when remote is preferred.
        local = self.local.submit(frame, deadline=self.deadline)
        remote = self.remote.submit(frame, deadline=self.deadline)
        return local, remote

    def poll(self):
        if self.frame is None:
            return DetectionDecision(None, "none", True, "No frame submitted")
        remote, local = self.remote.poll(self.frame), self.local.poll(self.frame)
        if remote is not None:
            if not self.counted:
                self.streak += 1
                self.counted = True
                if self.streak >= self.recovery_frames:
                    self.remote_enabled = True
            if self.remote_enabled:
                return DetectionDecision(remote, "remote", False, "Timely remote result")
        else:
            work = self.remote.work
            failed = (work is None or work.frame is not self.frame
                      or (work.done.is_set() and work.error is not None)
                      or self.clock() > self.deadline)
            if failed:
                self.remote_enabled, self.streak = False, 0
        if local is not None:
            return DetectionDecision(local, "local", False, "Fresh local fallback")
        return DetectionDecision(None, "none", True, "No fresh result within current budget")

    def close(self):
        self.local.close()
        self.remote.close()
