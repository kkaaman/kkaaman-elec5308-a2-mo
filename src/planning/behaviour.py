"""Explicit stop/hold/resume guards; track expiry cannot authorize motion."""

import math
from dataclasses import dataclass
from enum import Enum


class BehaviourState(str, Enum):
    CRUISE = "CRUISE"
    APPROACH = "APPROACH"
    STOPPING = "STOPPING"
    HOLD = "HOLD"
    RESUME = "RESUME"
    FINISHED = "FINISHED"
    FAULT_STOP = "FAULT_STOP"


@dataclass(frozen=True)
class BehaviourConfig:
    stopped_speed_mps: float = 0.1
    clear_frames: int = 5
    clear_duration_s: float = 0.4

    def __post_init__(self):
        if type(self.clear_frames) is not int or self.clear_frames < 1:
            raise ValueError("Positive clear-frame count required")
        if any(
            not math.isfinite(v) or v <= 0 for v in (self.stopped_speed_mps, self.clear_duration_s)
        ):
            raise ValueError("Positive finite speed/time required")


class Behaviour:
    def __init__(self, config=None):
        self.config = config or BehaviourConfig()
        self.reset()

    def reset(self):
        self.state = BehaviourState.CRUISE
        self.clear_count = 0
        self.clear_since = None
        self.last_time = None
        self.blocked_latch = False
        self.reason = "reset"

    def fault(self, reason):
        self.state = BehaviourState.FAULT_STOP
        self.reason = str(reason)
        self.clear_count = 0
        self.clear_since = None

    def update(
        self, *, time_s, speed, obstacle, must_stop, clear, observation_valid=True, finished=False
    ):
        if not all(math.isfinite(x) for x in (time_s, speed)) or speed < 0:
            raise ValueError("Invalid behaviour observation")
        if self.last_time is not None and time_s <= self.last_time:
            raise ValueError("Behaviour time must advance")
        self.last_time = time_s
        if self.state in (BehaviourState.FAULT_STOP, BehaviourState.FINISHED):
            return self.state
        if finished:
            self.state, self.reason = BehaviourState.FINISHED, "goal_reached"
            return self.state
        if obstacle:
            self.blocked_latch = True
        if obstacle or not clear or not observation_valid:
            self.clear_count, self.clear_since = 0, None
        else:
            self.clear_count += 1
            if self.clear_since is None:
                self.clear_since = time_s
        cleared = (
            self.clear_count >= self.config.clear_frames
            and self.clear_since is not None
            and time_s - self.clear_since + 1e-9 >= self.config.clear_duration_s
        )
        if self.state == BehaviourState.HOLD:
            if cleared:
                self.blocked_latch = False
                self.state, self.reason = BehaviourState.RESUME, "fresh_visible_clearance"
            else:
                self.reason = "await_fresh_visible_clearance"
            return self.state
        if (
            must_stop
            or not observation_valid
            or (self.blocked_latch and not obstacle and not cleared)
        ):
            self.state = (
                BehaviourState.HOLD
                if speed <= self.config.stopped_speed_mps
                else BehaviourState.STOPPING
            )
            self.reason = "obstacle_or_unknown_corridor"
        elif obstacle:
            self.state, self.reason = BehaviourState.APPROACH, "obstacle_ahead"
        elif self.state == BehaviourState.STOPPING and not cleared:
            if speed <= self.config.stopped_speed_mps:
                self.state = BehaviourState.HOLD
            self.reason = "clearance_hysteresis"
        else:
            self.blocked_latch = False
            self.state, self.reason = BehaviourState.CRUISE, "visible_route"
        return self.state
