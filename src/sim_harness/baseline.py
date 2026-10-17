"""Closed-loop S0-S5 fixture renderer over the existing KinematicWorld.

Synthetic coloured rectangles are sensed by a raster detector. Evaluation
positions stay in the world; the production policy receives RGB-D only.
"""

import math
import time

import cv2
import numpy as np

from autonomy.interfaces import EgoPose, FrameBundle
from geometry.camera import CameraIntrinsics
from geometry.transforms import ego_to_world_matrix, inverse_rigid
from navigation_control.types import VehicleState
from obstacle_perception.types import (
    CallTiming,
    DetectionBatch,
    DetectionObservation,
    ObservationMetadata,
    PixelBox,
)
from planning.corridor import RouteGeometry
from planning.mission import route_graph
from sim_harness.kinematic import KinematicConfig, KinematicWorld


def fixture_route(curve=False):
    points = [(float(x), 0.0) for x in range(0, 51)]
    if curve:
        radius = 18.0
        points.extend(
            (50 + radius * math.sin(t), radius * (1 - math.cos(t)))
            for t in np.linspace(0, math.pi / 2, 40)[1:]
        )
        points.extend((68.0, float(y)) for y in range(19, 80))
    else:
        points.extend((float(x), 0.0) for x in range(51, 121))
    return tuple(points)


class RasterDetector:
    """Test-only colour segmentation, not a model accuracy benchmark."""

    def reset(self, *, episode_id):
        self.episode_id = episode_id

    def detect(self, frame):
        if frame.episode_id != self.episode_id:
            raise ValueError("Raster detector episode mismatch")
        started = time.perf_counter()
        found = []
        for class_name, channel, identity in (("car", 2, 1), ("person", 0, 2)):
            mask = ((frame.rgb[:, :, channel] == 240) & (frame.rgb[:, :, 1] == 40)).astype(np.uint8)
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            for x, y, w, h, area in stats[1:count]:
                if area >= 6:
                    found.append(
                        DetectionObservation(
                            class_name,
                            0.95,
                            PixelBox(float(x), float(y), float(x + w), float(y + h)),
                            identity,
                        )
                    )
        return DetectionBatch(
            ObservationMetadata(
                frame.episode_id,
                frame.frame_id,
                frame.sim_time_s,
                "synthetic-rgb",
                "colour-fixture-v1",
            ),
            tuple(found),
            frame.rgb.shape[1],
            frame.rgb.shape[0],
            CallTiming(time.perf_counter() - started),
        )

    def close(self):
        self.episode_id = None


class BaselineWorld(KinematicWorld):
    def __init__(self, scenario="S0", *, curve=False, fault=None, seed=1):
        if scenario not in {f"S{x}" for x in range(6)}:
            raise ValueError("Unknown baseline scenario")
        self.scenario_id, self.fault, self.seed = scenario, fault, seed
        self.points = fixture_route(curve or scenario == "S5")
        self.route = RouteGeometry(self.points)
        self.object_s = 85.0 if scenario == "S5" else 55.0
        self.object_lateral = 4.5 if scenario == "S3" else 0.0
        self.person = scenario == "S2"
        self.present = scenario != "S0"
        self._hold_since = None
        self.events = []
        self.evaluations = {}
        self._collision = False
        self.calibration = CameraIntrinsics.from_horizontal_fov(640, 360, math.pi / 2).calibration(
            ((0, 0, 1, 2.88641), (-1, 0, 0, 0), (0, -1, 0, 2.0), (0, 0, 0, 1))
        )
        super().__init__(
            KinematicConfig(wheelbase_m=3.00465, max_steering_angle_rad=math.radians(20)),
            initial_state=VehicleState(*self.points[0], 0, 0),
            graph=route_graph(self.points),
        )

    def _frame(self, episode, frame_id, state):
        now = frame_id * self.config.dt_s
        s = self.route.project(state.x_m, state.y_m).s_m
        gap = self.object_s - (0.4 if self.person else 2.4) - s - 3.81152
        if (
            self.present
            and self.scenario_id not in {"S3", "S4"}
            and state.speed_mps <= 0.1
            and 0 < gap < 8
        ):
            if self._hold_since is None:
                self._hold_since = now
            if now - self._hold_since >= 2.0:
                self.present = False
                self.events.append(dict(type="obstacle_removed", sim_time_s=now, frame_id=frame_id))
        else:
            self._hold_since = None
        rgb = np.full((360, 640, 3), 25, np.uint8)
        depth = np.full((360, 640), 100.0, np.float32)
        if self.present:
            x, y, yaw = self.route.at(self.object_s, self.object_lateral)
            half = (0.4, 0.3, 0.9) if self.person else (2.4, 1.0, 0.75)
            object_to_world = ego_to_world_matrix(x, y, yaw, half[2])
            world_to_camera = inverse_rigid(self.calibration.camera_to_ego) @ inverse_rigid(
                ego_to_world_matrix(state.x_m, state.y_m, state.yaw_rad)
            )
            corners = [
                world_to_camera @ object_to_world @ np.array([dx, dy, dz, 1])
                for dx in (-half[0], half[0])
                for dy in (-half[1], half[1])
                for dz in (-half[2], half[2])
            ]
            if min(p[2] for p in corners) > 0.1:
                uv = np.array([(320 + 320 * p[0] / p[2], 180 + 320 * p[1] / p[2]) for p in corners])
                lo = np.maximum(0, np.floor(uv.min(axis=0)).astype(int))
                hi = np.minimum((640, 360), np.ceil(uv.max(axis=0)).astype(int))
                if (hi > lo).all():
                    roi = (slice(lo[1], hi[1]), slice(lo[0], hi[0]))
                    vv, uu = np.mgrid[lo[1] : hi[1], lo[0] : hi[0]]
                    rays = np.stack(((uu - 320) / 320, (vv - 180) / 320, np.ones_like(uu)), axis=-1)
                    camera_to_object = np.linalg.inv(world_to_camera @ object_to_world)
                    direction = rays @ camera_to_object[:3, :3].T
                    origin = camera_to_object[:3, 3]
                    safe = np.where(abs(direction) < 1e-12, 1e-12, direction)
                    a, b = (-np.array(half) - origin) / safe, (np.array(half) - origin) / safe
                    entry = np.max(np.minimum(a, b), axis=-1)
                    leave = np.min(np.maximum(a, b), axis=-1)
                    hit = (entry > 0) & (leave >= entry)
                    rgb[roi][hit] = (240, 40, 0) if self.person else (0, 40, 240)
                    depth[roi][hit] = entry[hit]
        in_corridor = self.present and self.scenario_id != "S3" and gap > -6
        if in_corridor and gap <= 0 and not self._collision:
            self.events.append(
                dict(type="collision", sim_time_s=now, frame_id=frame_id, other_actor_id="fixture")
            )
            self._collision = True
        self.evaluations[frame_id] = dict(
            truth_gap_m=gap if in_corridor else None, truth_in_corridor=in_corridor
        )
        if self.fault == "missing_depth" and frame_id >= 80:
            depth = None
        if self.fault == "invalid_depth" and frame_id >= 80:
            depth[:] = np.nan
        if self.fault == "flicker" and frame_id % 5 in (0, 1):
            rgb[:] = 25
        self.control_clock_s = time.monotonic()
        received = self.control_clock_s - (2.0 if self.fault == "stale" and frame_id >= 80 else 0)
        return FrameBundle(
            episode,
            frame_id,
            now,
            received,
            EgoPose(state),
            rgb,
            self.calibration,
            depth,
            frame_id if depth is not None else None,
        )

    def control_clock(self):
        """Fixture clock freezes during computation; stale faults shift receipt."""
        return self.control_clock_s

    def evaluation(self, frame_id):
        return self.evaluations.pop(frame_id)

    def close(self):
        super().close()
        self.cleanup = dict(ok=True, scope="synthetic world; no CARLA actors")
