"""Deterministic bicycle-model world for simulator-free integration tests.

The model deliberately provides only simple kinematics and synthetic camera
observations. It is not a tyre, collision, or visibility simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from autonomy.interfaces import (
    ActorTruth,
    CameraModel,
    ControlPacket,
    EvaluationPacket,
    Extrinsics,
    FrameBundle,
    MotionState,
    VehicleGeometry,
)
from navigation_control import ControlCommand

from .fake_detectors import ScriptedObject


@dataclass(frozen=True)
class SyntheticObstacle:
    """Static rectangular obstacle used only to generate synthetic observations."""

    key: str
    class_name: str
    x_m: float
    y_m: float
    length_m: float = 4.5
    width_m: float = 1.8
    confidence: float = 0.95


class KinematicWorld:
    """Fixed-step bicycle-model implementation of ``SimulatorInterface``.

    The controller receives a normal ``FrameBundle``. Synthetic objects are
    exposed through ``detector_scene`` so a ``FakeLocalDetector`` or
    ``FakeRemoteDetector`` can produce detections; controller code must not
    read ``evaluation()`` ground truth.
    """

    def __init__(
        self,
        *,
        dt_s: float = 0.05,
        vehicle_geometry: VehicleGeometry | None = None,
        max_accel_mps2: float = 3.0,
        max_brake_mps2: float = 8.0,
        drag_mps2: float = 0.15,
        camera_width_px: int = 640,
        camera_height_px: int = 480,
        camera_fov_rad: float = math.radians(90.0),
    ) -> None:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if max_accel_mps2 <= 0.0 or max_brake_mps2 <= 0.0:
            raise ValueError("acceleration limits must be positive")
        if drag_mps2 < 0.0:
            raise ValueError("drag_mps2 must be nonnegative")

        self._dt_s = float(dt_s)
        self._max_accel_mps2 = float(max_accel_mps2)
        self._max_brake_mps2 = float(max_brake_mps2)
        self._drag_mps2 = float(drag_mps2)

        self._vehicle_geometry = vehicle_geometry or VehicleGeometry(
            blueprint="kinematic-hatchback",
            wheelbase_m=2.8,
            rear_axle_to_front_bumper_m=3.8,
            rear_axle_to_rear_bumper_m=1.0,
            width_m=1.8,
            max_steer_rad=math.radians(35.0),
        )
        self._camera = CameraModel.from_fov(
            camera_width_px,
            camera_height_px,
            camera_fov_rad,
            Extrinsics(0.0, 0.0, 1.4),
        )

        self._closed = False
        self._run_id = ""
        self._frame_id = 0
        self._time_s = 0.0
        self._state = MotionState(0.0, 0.0, 0.0, 0.0)
        self._command = ControlCommand(0.0, 0.0, 1.0)
        self._obstacles: tuple[SyntheticObstacle, ...] = ()

    @property
    def vehicle_geometry(self) -> VehicleGeometry:
        return self._vehicle_geometry

    def reset(self, scenario: Mapping[str, Any]) -> FrameBundle:
        """Build a deterministic scene and return frame zero.

        Supported scenario keys are ``run_id``, ``initial_state`` (a
        ``MotionState`` or mapping with x_m/y_m/yaw_rad/speed_mps) and
        ``obstacles`` (a list of obstacle mappings).
        """
        self._ensure_open()

        self._run_id = str(scenario.get("run_id", "kinematic-run"))
        if not self._run_id:
            raise ValueError("scenario run_id must be non-empty")

        self._state = self._parse_state(scenario.get("initial_state", {}))
        self._obstacles = tuple(
            self._parse_obstacle(index, value)
            for index, value in enumerate(scenario.get("obstacles", ()))
        )
        self._frame_id = 0
        self._time_s = 0.0
        self._command = ControlCommand(0.0, 0.0, 1.0)
        return self._bundle()

    def apply(self, packet: ControlPacket) -> None:
        """Store the command for the current frame.

        A command must be tied to the latest observed frame. This prevents
        stale commands from silently moving the simulated vehicle.
        """
        self._ensure_ready()
        if packet.frame_id != self._frame_id:
            raise ValueError(
                f"command frame {packet.frame_id} does not match "
                f"latest frame {self._frame_id}"
            )
        self._command = packet.command

    def tick(self, timeout_s: float) -> FrameBundle:
        """Advance exactly one fixed time step and return the next frame."""
        self._ensure_ready()
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")

        command = self._command
        if command.brake > 0.0:
            acceleration = -self._max_brake_mps2 * command.brake
        else:
            acceleration = self._max_accel_mps2 * command.throttle
            acceleration -= self._drag_mps2 if self._state.speed_mps > 0.0 else 0.0

        speed_mps = max(0.0, self._state.speed_mps + acceleration * self._dt_s)
        steering_rad = max(
            -self._vehicle_geometry.max_steer_rad,
            min(self._vehicle_geometry.max_steer_rad, command.steering_angle_rad),
        )
        yaw_rate_rps = (
            speed_mps * math.tan(steering_rad) / self._vehicle_geometry.wheelbase_m
        )
        yaw_rad = self._wrap_angle(self._state.yaw_rad + yaw_rate_rps * self._dt_s)
        x_m = self._state.x_m + speed_mps * math.cos(yaw_rad) * self._dt_s
        y_m = self._state.y_m + speed_mps * math.sin(yaw_rad) * self._dt_s

        self._state = MotionState(x_m, y_m, yaw_rad, speed_mps, yaw_rate_rps)
        self._frame_id += 1
        self._time_s += self._dt_s
        return self._bundle()

    def detector_scene(self, frame_id: int) -> tuple[ScriptedObject, ...]:
        """Return projected synthetic objects for the fake detector boundary."""
        self._ensure_ready()
        if frame_id != self._frame_id:
            raise ValueError(f"scene requested for frame {frame_id}, expected {self._frame_id}")

        objects: list[ScriptedObject] = []
        for obstacle in self._obstacles:
            projection = self._project(obstacle)
            if projection is not None:
                objects.append(projection)
        return tuple(objects)

    def obstacle_distances_m(self) -> Mapping[str, float]:
        """Synthetic range readings for perception/harness code, not control."""
        self._ensure_ready()
        return {
            obstacle.key: math.hypot(
                obstacle.x_m - self._state.x_m,
                obstacle.y_m - self._state.y_m,
            )
            for obstacle in self._obstacles
        }

    def evaluation(self) -> EvaluationPacket:
        """Return ground truth for evaluation only."""
        self._ensure_ready()
        actors = tuple(
            ActorTruth(
                actor_id=index + 1,
                kind=obstacle.class_name,
                x_m=obstacle.x_m,
                y_m=obstacle.y_m,
                yaw_rad=0.0,
                length_m=obstacle.length_m,
                width_m=obstacle.width_m,
            )
            for index, obstacle in enumerate(self._obstacles)
        )
        return EvaluationPacket(self._frame_id, self._state, actors)

    def close(self) -> None:
        self._closed = True

    def _bundle(self) -> FrameBundle:
        rgb = np.zeros(
            (self._camera.height_px, self._camera.width_px, 3),
            dtype=np.uint8,
        )
        return FrameBundle(
            run_id=self._run_id,
            frame_id=self._frame_id,
            sim_time_s=self._time_s,
            host_tick_s=self._time_s,
            host_receive_s=self._time_s,
            ego=self._state,
            camera=self._camera,
            rgb=rgb,
        )

    def _project(self, obstacle: SyntheticObstacle) -> ScriptedObject | None:
        dx = obstacle.x_m - self._state.x_m
        dy = obstacle.y_m - self._state.y_m
        cos_yaw = math.cos(self._state.yaw_rad)
        sin_yaw = math.sin(self._state.yaw_rad)

        forward_m = cos_yaw * dx + sin_yaw * dy
        left_m = -sin_yaw * dx + cos_yaw * dy
        if forward_m <= 0.5:
            return None

        focal_px = self._camera.fx_px
        centre_x = self._camera.cx_px - focal_px * left_m / forward_m
        box_width = max(4.0, focal_px * obstacle.width_m / forward_m)
        box_height = max(4.0, focal_px * obstacle.length_m / forward_m)
        x_min = max(0.0, centre_x - box_width / 2.0)
        x_max = min(float(self._camera.width_px), centre_x + box_width / 2.0)
        y_min = max(0.0, self._camera.cy_px - box_height / 2.0)
        y_max = min(float(self._camera.height_px), self._camera.cy_px + box_height / 2.0)
        if x_max <= x_min or y_max <= y_min:
            return None

        from autonomy.interfaces import ImageBox

        return ScriptedObject(
            key=obstacle.key,
            class_name=obstacle.class_name,
            confidence=obstacle.confidence,
            box=ImageBox(x_min, y_min, x_max, y_max),
        )

    @staticmethod
    def _parse_state(value: MotionState | Mapping[str, Any]) -> MotionState:
        if isinstance(value, MotionState):
            return value
        return MotionState(
            float(value.get("x_m", 0.0)),
            float(value.get("y_m", 0.0)),
            float(value.get("yaw_rad", 0.0)),
            float(value.get("speed_mps", 0.0)),
        )

    @staticmethod
    def _parse_obstacle(index: int, value: Mapping[str, Any]) -> SyntheticObstacle:
        return SyntheticObstacle(
            key=str(value.get("key", f"obstacle-{index}")),
            class_name=str(value.get("class_name", "vehicle")),
            x_m=float(value["x_m"]),
            y_m=float(value["y_m"]),
            length_m=float(value.get("length_m", 4.5)),
            width_m=float(value.get("width_m", 1.8)),
            confidence=float(value.get("confidence", 0.95)),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("world is closed")

    def _ensure_ready(self) -> None:
        self._ensure_open()
        if not self._run_id:
            raise RuntimeError("reset(scenario) must be called first")

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi