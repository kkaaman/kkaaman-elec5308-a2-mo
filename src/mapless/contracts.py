"""Contracts between the stages of the mapless stack (plan v2.1, section 7).

Every value here is in the car's own frames and carries its age and its evidence, so a
consumer can refuse what is too old or too thin instead of trusting a stale number. None
of it comes from the simulator: a track is built from sensor returns, a road model from
the terrain grid and the cameras, a decision from what the planner saw.

Frames: the vehicle frame (rear axle, +x forward, +y left) for anything consumed this
tick; the odometry frame (``mapless.ego_motion``) for anything kept between ticks.
"""

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

CLASSES = ("car", "truck", "bike", "pedestrian", "unknown")


def _finite(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _array(value, shape, name):
    out = np.asarray(value, dtype=float)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite array of shape {shape}")
    return out


class TrackStatus(str, Enum):
    TENTATIVE = "tentative"  # seen, not yet confirmed (3 hits in 5 frames)
    CONFIRMED = "confirmed"
    COASTING = "coasting"  # no hit this frame; predicted forward, up to 0.3 s
    DELETED = "deleted"  # not seen for 0.75 s


@dataclass(frozen=True)
class ObjectTrack:
    """One tracked road user in the odometry frame (section 10).

    ``state`` is ``[x, y, vx, vy]`` and ``covariance`` its 4x4 covariance. ``footprint``
    is the plan-view extent as ``(length_m, width_m, yaw_rad)`` when the LiDAR has
    measured it, else None. ``class_probabilities`` sums to one over ``CLASSES``.
    ``support`` names the sensors behind the last update (``lidar``, ``camera``,
    ``radar``): an empty radar frame never means "lane clear", and a track without
    LiDAR support has no measured extent.
    """

    track_id: int
    time_s: float
    state: np.ndarray
    covariance: np.ndarray
    class_probabilities: dict
    status: TrackStatus
    hits: int
    misses: int
    support: tuple = ()
    footprint: tuple | None = None
    # Net displacement (m) from the track's first measurement, or None when unknown:
    # the tracker's corroboration that an estimated velocity is motion, not jitter.
    travelled_m: float | None = None
    # Mean speed from net displacement over the last second, or None (too little history).
    recent_speed_mps: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "state", _array(self.state, (4,), "state"))
        object.__setattr__(self, "covariance", _array(self.covariance, (4, 4), "covariance"))
        probs = {k: float(self.class_probabilities.get(k, 0.0)) for k in CLASSES}
        total = sum(probs.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6) or any(p < 0 for p in probs.values()):
            raise ValueError("class_probabilities must be non-negative over CLASSES and sum to one")
        object.__setattr__(self, "class_probabilities", probs)
        if not isinstance(self.status, TrackStatus):
            raise TypeError("status must be a TrackStatus")
        if self.footprint is not None:
            length, width, yaw = self.footprint
            for name, value in (("length", length), ("width", width), ("yaw", yaw)):
                _finite(name, value)
            if length <= 0 or width <= 0:
                raise ValueError("footprint length and width must be positive")

    @property
    def position(self):
        return self.state[:2]

    @property
    def velocity(self):
        return self.state[2:]

    @property
    def speed_mps(self):
        return float(np.hypot(*self.state[2:]))

    @property
    def best_class(self):
        return max(self.class_probabilities, key=self.class_probabilities.get)


@dataclass(frozen=True)
class Prediction:
    """Where a track may be over the horizon (section 10): time-indexed occupied regions.

    ``times_s`` are offsets from the track's time; ``centres`` (K x 2, odometry frame)
    the predicted positions; ``half_extents`` (K x 2) the half-length along and half-width
    across the heading, inflated by the position uncertainty plus a margin.
    """

    track_id: int
    times_s: np.ndarray
    centres: np.ndarray
    half_extents: np.ndarray
    heading_rad: float

    def __post_init__(self):
        k = len(np.asarray(self.times_s))
        object.__setattr__(self, "times_s", _array(self.times_s, (k,), "times_s"))
        object.__setattr__(self, "centres", _array(self.centres, (k, 2), "centres"))
        object.__setattr__(self, "half_extents", _array(self.half_extents, (k, 2), "half_extents"))
        _finite("heading_rad", self.heading_rad)
        if k == 0 or (np.diff(self.times_s) <= 0).any() or (self.half_extents <= 0).any():
            raise ValueError("a prediction needs increasing times and positive extents")

    def occupies(self, xy, time_s):
        """Whether point(s) ``xy`` (N x 2) fall inside the region at ``time_s`` (nearest step)."""
        i = int(np.argmin(np.abs(self.times_s - time_s)))
        p = np.asarray(xy, dtype=float).reshape(-1, 2) - self.centres[i]
        c, s = math.cos(self.heading_rad), math.sin(self.heading_rad)
        along, across = c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1]
        return (np.abs(along) <= self.half_extents[i, 0]) & (np.abs(across) <= self.half_extents[i, 1])


@dataclass(frozen=True)
class LocalRoadModel:
    """The rolling local road model (section 9): grid layers in the odometry frame.

    ``layers`` maps a name to an H x W array on the same grid (``origin_xy`` is the
    odometry position of cell (0, 0), ``cell_m`` its size): at least ``occupancy``,
    ``free``, ``unknown``, ``ground_height``, ``road``, ``marking``, ``pavement``,
    ``age_s``. ``lane_candidates`` and ``exit_candidates`` are polylines (N x 2) with a
    score each; ``stop_lines`` likewise. ``visibility_m`` is how far ahead the road is
    currently observed. Everything is as fresh as ``time_s`` says, and nothing older
    than the builder's memory survives in it.
    """

    time_s: float
    origin_xy: tuple
    cell_m: float
    layers: dict
    lane_candidates: tuple = ()
    exit_candidates: tuple = ()
    stop_lines: tuple = ()
    visibility_m: float = 0.0

    REQUIRED = ("occupancy", "free", "unknown", "ground_height", "road", "marking", "pavement", "age_s")

    def __post_init__(self):
        _finite("time_s", self.time_s)
        _finite("cell_m", self.cell_m)
        if self.cell_m <= 0:
            raise ValueError("cell_m must be positive")
        missing = [k for k in self.REQUIRED if k not in self.layers]
        if missing:
            raise ValueError(f"road model layers missing: {missing}")
        shapes = {np.asarray(v).shape for v in self.layers.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ValueError("all layers must be 2-D arrays of one shape")

    @property
    def shape(self):
        return np.asarray(self.layers["occupancy"]).shape

    def cell(self, xy):
        """Cell indices (N x 2: row from x, column from y, as the terrain grid) of odometry
        points; may lie outside the layers."""
        p = np.asarray(xy, dtype=float).reshape(-1, 2) - np.asarray(self.origin_xy)
        return np.floor(p / self.cell_m).astype(int)


@dataclass(frozen=True)
class DecisionRecord:
    """What the stack decided on one tick and why (section 7), for the log and the report.

    ``proposed`` is what the tactical layer asked for (``FOLLOW``, ``WAIT``,
    ``PREPARE_LEFT``, ``PREPARE_RIGHT``, ``COMMIT``, ``STOP``) and ``executed`` what the
    safety checks let through; ``veto_reason`` says why they differ. ``constraints``
    lists what bounded the plan (an obstacle, a prediction, unknown ground, a hold);
    ``coverage`` the sensors that were healthy; ``timing_ms`` the stage times.

    The stack's speed cap is computed after the driver has produced this tick's command,
    so it reaches the driver one tick later: ``cap_applied_mps`` is the cap the command
    of this tick was planned under (the previous tick's result) and ``speed_cap_next_mps``
    the one computed this tick for the next (review of the G3 framework: one field read
    as if the fresh cap had shaped this command). ``frame_generation`` counts the local
    (odometry) frame rebuilds; records from different generations are not comparable in
    position.
    """

    time_s: float
    proposed: str
    executed: str
    veto_reason: str = ""
    constraints: tuple = ()
    coverage: dict = field(default_factory=dict)
    timing_ms: dict = field(default_factory=dict)
    cap_applied_mps: float | None = None
    speed_cap_next_mps: float | None = None
    frame_generation: int = 0

    ACTIONS = ("FOLLOW", "WAIT", "PREPARE_LEFT", "PREPARE_RIGHT", "COMMIT", "STOP")

    def __post_init__(self):
        _finite("time_s", self.time_s)
        for name in (self.proposed, self.executed):
            if name not in self.ACTIONS:
                raise ValueError(f"unknown action {name!r}")
        if self.proposed != self.executed and not self.veto_reason:
            raise ValueError("a vetoed action needs a veto_reason")
        for name in ("cap_applied_mps", "speed_cap_next_mps"):
            value = getattr(self, name)
            if value is not None:
                _finite(name, value)
        if type(self.frame_generation) is not int or self.frame_generation < 0:
            raise ValueError("frame_generation must be a non-negative integer")

    def as_dict(self):
        return dict(
            time_s=self.time_s,
            proposed=self.proposed,
            executed=self.executed,
            veto_reason=self.veto_reason,
            constraints=list(self.constraints),
            coverage=dict(self.coverage),
            timing_ms=dict(self.timing_ms),
            cap_applied_mps=self.cap_applied_mps,
            speed_cap_next_mps=self.speed_cap_next_mps,
            frame_generation=self.frame_generation,
        )
