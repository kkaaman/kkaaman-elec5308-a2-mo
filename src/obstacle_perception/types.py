"""Sensor-derived detection and obstacle contracts (BL-C-01).

No simulator actor identity or obstacle ground truth belongs in these types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")


def _number(name: str, value: float, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


@dataclass(frozen=True)
class ObservationMetadata:
    """Identity of the source observation, not the inference completion frame."""

    episode_id: str
    frame_id: int
    sim_time_s: float
    source_id: str
    model_version: str

    def __post_init__(self) -> None:
        for name in ("episode_id", "source_id", "model_version"):
            _text(name, getattr(self, name))
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")
        _number("sim_time_s", self.sim_time_s, 0)

    def track_identity(self, track_id: int) -> TrackIdentity:
        return TrackIdentity(self.episode_id, self.source_id, self.model_version, track_id)


@dataclass(frozen=True)
class TrackIdentity:
    """Tracker IDs are scoped to episode, source and model version."""

    episode_id: str
    source_id: str
    model_version: str
    track_id: int

    def __post_init__(self) -> None:
        for name in ("episode_id", "source_id", "model_version"):
            _text(name, getattr(self, name))
        if type(self.track_id) is not int or self.track_id < 0:
            raise ValueError("track_id must be a nonnegative integer")


@dataclass(frozen=True)
class PixelBox:
    """Continuous XYXY pixel coordinates; right/bottom are exclusive limits."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for name in ("x_min", "y_min", "x_max", "y_max"):
            _number(name, getattr(self, name), 0)
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("box must have positive width and height")

    @property
    def width_px(self):
        return self.x_max - self.x_min

    @property
    def height_px(self):
        return self.y_max - self.y_min


@dataclass(frozen=True)
class DetectionObservation:
    """One sensor detection. Track ID is absent for ordinary detection."""

    class_name: str
    confidence: float
    box: PixelBox
    track_id: int | None = None

    def __post_init__(self) -> None:
        _text("class_name", self.class_name)
        _number("confidence", self.confidence, 0)
        if self.confidence > 1:
            raise ValueError("confidence must be <= 1")
        if not isinstance(self.box, PixelBox):
            raise TypeError("box must be PixelBox")
        if self.track_id is not None and (type(self.track_id) is not int or self.track_id < 0):
            raise ValueError("track_id must be absent or a nonnegative integer")


@dataclass(frozen=True)
class CallTiming:
    """Durations, not absolute timestamps; remote clocks need not match."""

    total_wall_s: float
    inference_wall_s: float | None = None
    transport_wall_s: float | None = None

    def __post_init__(self) -> None:
        _number("total_wall_s", self.total_wall_s, 0)
        for name in ("inference_wall_s", "transport_wall_s"):
            value = getattr(self, name)
            if value is not None:
                _number(name, value, 0)
        # Remote durations may use another clock and must not be subtracted
        # from local timestamps. Do not assume they sum exactly to total.


@dataclass(frozen=True)
class DetectionBatch:
    metadata: ObservationMetadata
    detections: tuple[DetectionObservation, ...]
    image_width: int
    image_height: int
    timing: CallTiming
    valid: bool = True
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, ObservationMetadata) or not isinstance(
            self.timing, CallTiming
        ):
            raise TypeError("metadata and timing must use contract types")
        for value in (self.image_width, self.image_height):
            if type(value) is not int or value <= 0:
                raise ValueError("image dimensions must be positive integers")
        if type(self.valid) is not bool:
            raise ValueError("valid must be bool")
        detections = tuple(self.detections)
        if not self.valid:
            _text("invalid_reason", self.invalid_reason)
            if detections:
                raise ValueError("invalid batches cannot contain detections")
        elif self.invalid_reason is not None:
            raise ValueError("valid batches cannot carry an invalid reason")
        ids = set()
        for detection in detections:
            if not isinstance(detection, DetectionObservation):
                raise TypeError("detections must use DetectionObservation")
            if detection.box.x_max > self.image_width or detection.box.y_max > self.image_height:
                raise ValueError("box exceeds source-image bounds")
            if detection.track_id is not None:
                if detection.track_id in ids:
                    raise ValueError("duplicate track ID within one batch")
                ids.add(detection.track_id)
        object.__setattr__(self, "detections", detections)


@dataclass(frozen=True)
class Obstacle:
    """Sensor-derived extent in rear-axle ego coordinates at source frame.

    center_ego_m: X forward, Y left, Z up, metres.
    extent_m: full positive XYZ lengths in metres, not half extents.
    yaw_ego_rad: counterclockwise about ego +Z.
    """

    metadata: ObservationMetadata
    class_name: str
    confidence: float
    center_ego_m: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    yaw_ego_rad: float = 0.0
    track_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, ObservationMetadata):
            raise TypeError("metadata must be ObservationMetadata")
        _text("class_name", self.class_name)
        _number("confidence", self.confidence, 0)
        if self.confidence > 1:
            raise ValueError("confidence must be <= 1")
        for name in ("center_ego_m", "extent_m"):
            values = tuple(getattr(self, name))
            if len(values) != 3:
                raise ValueError(f"{name} requires XYZ")
            for value in values:
                _number(name, value)
                if name == "extent_m" and value <= 0:
                    raise ValueError("extent lengths must be positive")
            object.__setattr__(self, name, values)
        _number("yaw_ego_rad", self.yaw_ego_rad)
        if self.track_id is not None:
            self.metadata.track_identity(self.track_id)

    @property
    def identity(self) -> TrackIdentity | None:
        return None if self.track_id is None else self.metadata.track_identity(self.track_id)
