"""Validated, simulator-free stack skeleton with latched emergency stopping."""

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

from navigation_control.types import ControlCommand
from obstacle_perception.types import DetectionBatch

from .interfaces import ControlPacket, Diagnostics, FrameBundle, LaneGraph


@dataclass(frozen=True)
class StackConfig:
    command_ttl_s: float = 0.1
    max_observation_age_s: float = 0.5
    max_step_wall_s: float = 0.1
    require_depth: bool = False

    def __post_init__(self):
        for name in ("command_ttl_s", "max_observation_age_s", "max_step_wall_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if type(self.require_depth) is not bool:
            raise ValueError("require_depth must be bool")
        if self.command_ttl_s > self.max_observation_age_s:
            raise ValueError("Command TTL exceeds observation-age budget")

    @classmethod
    def load(cls, path):
        # JSON syntax is also valid YAML; this subset needs no extra dependency.
        values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict):
            raise ValueError("Configuration must be an object")
        return cls(**values)


class SkeletonStack:
    """Detect -> optional motion policy -> validated command, or full brake.

    The default policy holds still. Emergency state stays latched until reset.
    Injected policies own their algorithm and may implement reset()/close().
    No simulator import, ground truth, route selection or obstacle association.
    """

    def __init__(self, detector, config=None, motion_policy=None, clock=time.monotonic):
        self.detector = detector
        self.motion_policy = motion_policy
        self.clock = clock
        self._config_error = None
        try:
            if config is None:
                self.config = StackConfig()
            elif isinstance(config, StackConfig):
                self.config = replace(config)
            elif isinstance(config, dict):
                self.config = StackConfig(**config)
            elif isinstance(config, (str, Path)):
                self.config = StackConfig.load(config)
            else:
                raise TypeError("Unsupported config type")
        except Exception as error:
            self.config = StackConfig()
            self._config_error = f"bad_config:{type(error).__name__}"
        self._episode = None
        self._last_frame = None
        self._last_time = 0.0
        self._emergency = self._config_error
        self._closed = False

    def reset(self, *, episode_id, graph):
        if self._closed:
            raise RuntimeError("Stack is closed")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("Episode ID must be nonempty")
        self._episode = episode_id
        self._last_frame = None
        self._last_time = 0.0
        self._emergency = self._config_error
        try:
            if not isinstance(graph, LaneGraph):
                raise TypeError("Expected LaneGraph")
            LaneGraph(graph.nodes, graph.edges)
            self.detector.reset(episode_id=episode_id)
            if self.motion_policy is not None and hasattr(self.motion_policy, "reset"):
                self.motion_policy.reset(episode_id=episode_id, graph=graph)
        except Exception as error:
            self._emergency = f"reset_failed:{type(error).__name__}"

    def step(self, frame):
        if self._closed or self._episode is None:
            raise RuntimeError("Reset an open stack before stepping")
        started = time.perf_counter()
        stages = []
        command = ControlCommand(0, 0, 1)
        reason = "skeleton_hold"
        packet_frame = self._last_frame if self._last_frame is not None else 0
        packet_time = self._last_time
        try:
            if not isinstance(frame, FrameBundle):
                raise TypeError("Expected FrameBundle")
            # Revalidate even if a caller bypassed frozen dataclass protection.
            frame = replace(
                frame, ego=replace(frame.ego, state=replace(frame.ego.state))
            )
            if frame.episode_id != self._episode:
                raise ValueError("Episode mismatch")
            if self._last_frame is not None and (
                frame.frame_id <= self._last_frame
                or frame.sim_time_s <= self._last_time
            ):
                raise ValueError("Nonmonotonic observation")
            packet_frame, packet_time = frame.frame_id, frame.sim_time_s
            self._last_frame, self._last_time = packet_frame, packet_time
            age = self.clock() - frame.received_monotonic_s
            if (
                not math.isfinite(age)
                or age < 0
                or age > self.config.max_observation_age_s
            ):
                raise ValueError("Observation is stale or clock is inconsistent")
            if self.config.require_depth and frame.depth_m is None:
                raise ValueError("Required depth missing")
            if self._emergency is None:
                stage_start = time.perf_counter()
                batch = self.detector.detect(frame)
                stages.append(("detector", time.perf_counter() - stage_start))
                if not isinstance(batch, DetectionBatch):
                    raise TypeError("Detector must return DetectionBatch")
                batch = replace(
                    batch,
                    metadata=replace(batch.metadata),
                    timing=replace(batch.timing),
                    detections=tuple(
                        replace(item, box=replace(item.box))
                        for item in batch.detections
                    ),
                )
                if not batch.valid:
                    raise ValueError("Detector returned invalid observations")
                if (
                    batch.metadata.episode_id != frame.episode_id
                    or batch.metadata.frame_id != frame.frame_id
                    or batch.metadata.sim_time_s != frame.sim_time_s
                    or (batch.image_height, batch.image_width) != frame.rgb.shape[:2]
                ):
                    raise ValueError("Detector source does not match frame")
                if self.motion_policy is not None:
                    stage_start = time.perf_counter()
                    command = self.motion_policy(frame, batch)
                    stages.append(("policy", time.perf_counter() - stage_start))
                    if not isinstance(command, ControlCommand):
                        raise TypeError("Policy must return ControlCommand")
                    command = replace(command)
                    reason = "policy_command"
                if time.perf_counter() - started > self.config.max_step_wall_s:
                    raise TimeoutError("Step wall-time budget exceeded")
                if (
                    self.clock() - frame.received_monotonic_s
                    > self.config.max_observation_age_s
                ):
                    raise TimeoutError("Observation expired during processing")
        except Exception as error:
            self._emergency = (
                self._emergency or f"invalid_step:{type(error).__name__}:{error}"
            )
        if self._emergency is not None:
            command = ControlCommand(0, 0, 1)
            reason = self._emergency
        stages.append(("total", time.perf_counter() - started))
        return ControlPacket(
            self._episode,
            packet_frame,
            packet_time,
            packet_time + self.config.command_ttl_s,
            command,
            Diagnostics(tuple(stages), self._emergency is None, (reason,)),
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        for component in (self.motion_policy, self.detector):
            if component is not None and hasattr(component, "close"):
                try:
                    component.close()
                except Exception as error:
                    errors.append(str(error))
        if errors:
            raise RuntimeError("Stack cleanup failed: " + "; ".join(errors))
