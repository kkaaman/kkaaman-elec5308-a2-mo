"""Temporal RGB-D evidence. No output authorises clearance."""

import math
from collections import deque
from dataclasses import dataclass
from statistics import median

from obstacle_perception.box_depth import estimate_batch_depth
from obstacle_perception.types import DetectionObservation, ObservationMetadata, TrackIdentity


@dataclass(frozen=True)
class TrackPolicy:
    hits: int = 2
    window: int = 3
    expiry_s: float = 0.5
    distance_samples: int = 5

    def __post_init__(self):
        for name in ("hits", "window", "distance_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Track counts must be positive integers")
        if self.hits > self.window:
            raise ValueError("hits exceeds window")
        if (
            isinstance(self.expiry_s, bool)
            or not math.isfinite(self.expiry_s)
            or self.expiry_s <= 0
        ):
            raise ValueError("expiry_s must be finite and positive")


@dataclass(frozen=True)
class TrackObservation:
    identity: TrackIdentity
    metadata: ObservationMetadata
    detection: DetectionObservation
    confirmed: bool
    current: bool
    depth_valid: bool
    distance_m: float | None
    robust_distance_m: float | None
    invalid_reason: str | None


@dataclass(frozen=True)
class TrackBatch:
    tracks: tuple[TrackObservation, ...]
    expired: tuple[TrackIdentity, ...]
    untracked: tuple[DetectionObservation, ...]
    usable_for_clearance: bool = False


class TrackDepthFilter:
    """One ordered episode/source/model stream, reset explicitly between runs.

    N-of-M counts RGB hits over supplied batches, invalid batches included as
    misses. Only current confirmed detections receive valid aligned depth.
    Median history describes axial depth, not a stopping gap. None IDs remain
    unknown and unconfirmed. Callers must retain their independent HOLD policy.
    """

    def __init__(self, *, source_id, model_version, policy=None, depth_policy=None):
        ObservationMetadata("validation", 0, 0, source_id, model_version)
        self.source_id = source_id
        self.model_version = model_version
        self.policy = policy or TrackPolicy()
        self.depth_policy = depth_policy
        self._episode = None
        self._states = {}
        self._last = None

    def reset(self, *, episode_id):
        ObservationMetadata(episode_id, 0, 0, self.source_id, self.model_version)
        self._episode = episode_id
        self._states.clear()
        self._last = None

    def update(self, frame, batch):
        m = batch.metadata
        if (m.episode_id, m.source_id, m.model_version) != (
            self._episode,
            self.source_id,
            self.model_version,
        ):
            raise ValueError("Reset filter for matching episode/source/model")
        if (m.episode_id, m.frame_id, m.sim_time_s) != (
            frame.episode_id,
            frame.frame_id,
            frame.sim_time_s,
        ) or frame.rgb.shape[:2] != (batch.image_height, batch.image_width):
            raise ValueError("Detection/depth source observation mismatch")
        if self._last is not None and (
            m.frame_id <= self._last[0] or m.sim_time_s <= self._last[1]
        ):
            raise ValueError("Observations must increase in frame and time")
        estimates = (
            estimate_batch_depth(frame, batch, self.depth_policy)
            if batch.valid and frame.depth_m is not None
            else ()
        )
        self._last = (m.frame_id, m.sim_time_s)
        expired = tuple(
            identity
            for identity, state in self._states.items()
            if m.sim_time_s - state["last_seen"] > self.policy.expiry_s
        )
        for identity in expired:
            del self._states[identity]
        current = {}
        for index, detection in enumerate(batch.detections):
            if detection.track_id is None:
                continue
            identity = m.track_identity(detection.track_id)
            state = self._states.get(identity)
            if state is None or state["detection"].class_name != detection.class_name:
                state = dict(
                    hits=deque(maxlen=self.policy.window),
                    distances=deque(maxlen=self.policy.distance_samples),
                )
                self._states[identity] = state
            state.update(last_seen=m.sim_time_s, detection=detection)
            current[identity] = estimates[index] if estimates else None
        rows = []
        for identity, state in self._states.items():
            present = identity in current
            state["hits"].append(present)
            confirmed = sum(state["hits"]) >= self.policy.hits
            estimate = current.get(identity)
            valid = bool(confirmed and present and estimate is not None and estimate.valid)
            distances = state["distances"]
            while distances and m.sim_time_s - distances[0][0] > self.policy.expiry_s:
                distances.popleft()
            if estimate is not None and estimate.valid:
                distances.append((m.sim_time_s, estimate.axial_depth_m))
            reason = (
                None
                if valid
                else "unconfirmed"
                if not confirmed
                else "not_visible"
                if not present
                else "missing_depth"
                if estimate is None
                else estimate.reason
            )
            rows.append(
                TrackObservation(
                    identity,
                    m,
                    state["detection"],
                    confirmed,
                    present,
                    valid,
                    estimate.axial_depth_m if valid else None,
                    median(x[1] for x in distances) if valid and distances else None,
                    reason,
                )
            )
        return TrackBatch(
            tuple(rows),
            expired,
            tuple(d for d in batch.detections if d.track_id is None),
        )
