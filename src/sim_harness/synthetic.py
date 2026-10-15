"""Scripted sensor fixtures for the CARLA-free harness, never scene truth.

Boxes and axial distances are specified in sensor coordinates per frame. RGB
rectangles and constant-depth patches are test signals, not rendered imagery.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from autonomy.interfaces import FrameBundle
from geometry.camera import CameraIntrinsics, pixel_depth_to_ego
from obstacle_perception.types import (
    CallTiming,
    DetectionBatch,
    DetectionObservation,
    ObservationMetadata,
    Obstacle,
)


def _frame_id(value):
    if type(value) is not int or value < 0:
        raise ValueError("frame_id must be a nonnegative integer")


def _roi(box):
    return (
        slice(math.floor(box.y_min), math.ceil(box.y_max)),
        slice(math.floor(box.x_min), math.ceil(box.x_max)),
    )


@dataclass(frozen=True)
class SyntheticDetection:
    """A sensor detection and its scripted optical-Z distance (None = missing)."""

    detection: DetectionObservation
    axial_depth_m: float | None

    def __post_init__(self):
        if not isinstance(self.detection, DetectionObservation):
            raise TypeError("detection must be DetectionObservation")
        object.__setattr__(
            self, "detection", replace(self.detection, box=replace(self.detection.box))
        )
        value = self.axial_depth_m
        if value is not None and (
            isinstance(value, bool) or not math.isfinite(value) or value <= 0
        ):
            raise ValueError("axial_depth_m must be positive finite metres or None")


@dataclass(frozen=True)
class SyntheticFrame:
    """One frame's scripted observations; omitted frames are valid empty scenes.

    depth_available=False means the entire depth sensor is unavailable. A valid
    empty detection batch describes only this fixture's specified field of view.
    """

    frame_id: int
    observations: tuple[SyntheticDetection, ...] = ()
    depth_available: bool = True
    valid: bool = True
    invalid_reason: str | None = None

    def __post_init__(self):
        _frame_id(self.frame_id)
        observations = tuple(self.observations)
        if any(not isinstance(item, SyntheticDetection) for item in observations):
            raise TypeError("observations must use SyntheticDetection")
        object.__setattr__(self, "observations", tuple(replace(item) for item in observations))
        if type(self.depth_available) is not bool or type(self.valid) is not bool:
            raise ValueError("depth_available and valid must be bool")
        if not self.valid:
            if not isinstance(self.invalid_reason, str) or not self.invalid_reason.strip():
                raise ValueError("invalid observations require a reason")
            if observations:
                raise ValueError("invalid frames cannot contain detections")
        elif self.invalid_reason is not None:
            raise ValueError("valid observations cannot carry an invalid reason")


@dataclass(frozen=True)
class SyntheticScenario:
    """Immutable observation script shared by a world and its synthetic detector.

    Distances do not update automatically with vehicle motion. Supply each tick's
    intended observations explicitly. Overlapping raster ROIs are rejected: this
    fixture has no occlusion or foreground/background association model.
    """

    frames: tuple[SyntheticFrame, ...] = ()
    image_width: int = 160
    image_height: int = 90
    horizontal_fov_rad: float = math.pi / 2
    camera_height_m: float = 1.5
    source_id: str = "synthetic-front"
    model_version: str = "scripted-v1"

    def __post_init__(self):
        CameraIntrinsics.from_horizontal_fov(
            self.image_width, self.image_height, self.horizontal_fov_rad
        )
        if isinstance(self.horizontal_fov_rad, bool):
            raise ValueError("horizontal_fov_rad must be a numeric angle")
        if (
            isinstance(self.camera_height_m, bool)
            or not math.isfinite(self.camera_height_m)
            or self.camera_height_m < 0
        ):
            raise ValueError("camera_height_m must be finite nonnegative metres")
        ObservationMetadata("validation", 0, 0.0, self.source_id, self.model_version)
        frames = tuple(self.frames)
        if any(not isinstance(frame, SyntheticFrame) for frame in frames):
            raise TypeError("frames must use SyntheticFrame")
        frames = tuple(replace(frame) for frame in frames)
        ids = set()
        for frame in frames:
            if frame.frame_id in ids:
                raise ValueError("duplicate scripted frame_id")
            ids.add(frame.frame_id)
            self._batch("validation", frame, 0.0)
            occupied = np.zeros((self.image_height, self.image_width), dtype=bool)
            for item in frame.observations:
                roi = _roi(item.detection.box)
                if occupied[roi].any():
                    raise ValueError("overlapping synthetic raster ROIs are unsupported")
                occupied[roi] = True
        object.__setattr__(self, "frames", frames)

    @property
    def calibration(self):
        """Optical right/down/forward -> ego forward/left/up, rear-axle origin."""
        intrinsics = CameraIntrinsics.from_horizontal_fov(
            self.image_width, self.image_height, self.horizontal_fov_rad
        )
        return intrinsics.calibration(
            ((0, 0, 1, 0), (-1, 0, 0, 0), (0, -1, 0, self.camera_height_m), (0, 0, 0, 1))
        )

    def at(self, frame_id):
        _frame_id(frame_id)
        return next(
            (frame for frame in self.frames if frame.frame_id == frame_id), SyntheticFrame(frame_id)
        )

    def _batch(self, episode_id, script, sim_time_s):
        return DetectionBatch(
            ObservationMetadata(
                episode_id, script.frame_id, sim_time_s, self.source_id, self.model_version
            ),
            tuple(item.detection for item in script.observations),
            self.image_width,
            self.image_height,
            CallTiming(0.0),
            script.valid,
            script.invalid_reason,
        )

    def sensor_arrays(self, frame_id):
        """Create fresh RGB and aligned axial depth; all unknown depth is NaN."""
        script = self.at(frame_id)
        rgb = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        depth = np.full((self.image_height, self.image_width), np.nan, dtype=np.float64)
        for item in script.observations:
            roi = _roi(item.detection.box)
            rgb[roi] = (90, 150, 210)
            if item.axial_depth_m is not None:
                depth[roi] = item.axial_depth_m
        return rgb, depth if script.depth_available else None


class SyntheticDetector:
    """Detector protocol fixture consuming a frame-indexed sensor script.

    Does not infer from RGB, read a world, or expose simulator object identities.
    Repeated calls for the same frame return the same observation. Track IDs, if
    scripted, have the normal episode/source/model namespace.
    """

    def __init__(self, scenario=None):
        self.scenario = SyntheticScenario() if scenario is None else scenario
        if not isinstance(self.scenario, SyntheticScenario):
            raise TypeError("scenario must be SyntheticScenario")
        self._episode = None
        self._closed = False

    def reset(self, *, episode_id):
        if self._closed:
            raise RuntimeError("Detector is closed")
        ObservationMetadata(
            episode_id, 0, 0.0, self.scenario.source_id, self.scenario.model_version
        )
        self._episode = episode_id

    def detect(self, frame):
        if self._closed or self._episode is None:
            raise RuntimeError("Reset an open detector before detecting")
        if not isinstance(frame, FrameBundle):
            raise TypeError("Expected FrameBundle")
        if frame.episode_id != self._episode:
            raise ValueError("Detector episode mismatch")
        if frame.rgb.shape[:2] != (self.scenario.image_height, self.scenario.image_width):
            raise ValueError("Frame dimensions differ from synthetic scenario")
        if frame.calibration != self.scenario.calibration:
            raise ValueError("Frame calibration differs from synthetic scenario")
        return self.scenario._batch(
            frame.episode_id, self.scenario.at(frame.frame_id), frame.sim_time_s
        )

    def track(self, frame):
        return self.detect(frame)

    def close(self):
        self._episode = None
        self._closed = True


class SyntheticObstacleSource:
    """Associate controlled synthetic depth patches with their detection boxes.

    Extents are an explicit fixed prior, not measured geometry. This fixture is
    not BL-D-03 robust depth association or a production perception component.
    """

    def __init__(self, *, extent_prior_m=(1.0, 1.0, 1.0)):
        # Reuse the shared extent validation without inventing another contract.
        prior = Obstacle(
            ObservationMetadata("validation", 0, 0.0, "synthetic", "prior-v1"),
            "prior",
            1.0,
            (0, 0, 0),
            extent_prior_m,
        )
        self.extent_prior_m = prior.extent_m
        self._episode = None
        self._closed = False

    def reset(self, *, episode_id):
        if self._closed:
            raise RuntimeError("Obstacle source is closed")
        ObservationMetadata(episode_id, 0, 0.0, "synthetic", "prior-v1")
        self._episode = episode_id

    def observe(self, frame, detections):
        if self._closed or self._episode is None:
            raise RuntimeError("Reset an open obstacle source before observing")
        if not isinstance(frame, FrameBundle) or not isinstance(detections, DetectionBatch):
            raise TypeError("Expected FrameBundle and DetectionBatch")
        metadata = detections.metadata
        if (
            frame.episode_id != self._episode
            or metadata.episode_id != frame.episode_id
            or metadata.frame_id != frame.frame_id
            or metadata.sim_time_s != frame.sim_time_s
            or (detections.image_height, detections.image_width) != frame.rgb.shape[:2]
        ):
            raise ValueError("Detection source does not match frame")
        if not detections.valid:
            raise ValueError("Invalid synthetic observations: " + detections.invalid_reason)
        if frame.depth_m is None:
            raise ValueError("Synthetic obstacle association requires depth")
        obstacles = []
        for detection in detections.detections:
            box = detection.box
            values = frame.depth_m[_roi(box)]
            values = values[np.isfinite(values) & (values > 0)]
            if not values.size:
                raise ValueError("Detection has no valid synthetic depth")
            center = pixel_depth_to_ego(
                (box.x_min + box.x_max) / 2,
                (box.y_min + box.y_max) / 2,
                float(np.median(values)),
                frame.calibration,
            )
            obstacles.append(
                Obstacle(
                    metadata,
                    detection.class_name,
                    detection.confidence,
                    tuple(center),
                    self.extent_prior_m,
                    track_id=detection.track_id,
                )
            )
        return tuple(obstacles)

    def close(self):
        self._episode = None
        self._closed = True
