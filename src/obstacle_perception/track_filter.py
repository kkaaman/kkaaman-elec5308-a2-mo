"""Episode-scoped N-of-M confirmation and world-coordinate temporal filtering."""

from collections import deque
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class TrackFilterConfig:
    required_hits: int = 2
    window_frames: int = 3
    expiry_s: float = 0.6
    anonymous_gate_m: float = 1.5

    def __post_init__(self):
        if any(type(x) is not int for x in (self.required_hits, self.window_frames)):
            raise ValueError("Confirmation counts must be integers")
        if not 1 <= self.required_hits <= self.window_frames <= 30:
            raise ValueError("Invalid N-of-M window")
        if not np.isfinite(self.expiry_s) or self.expiry_s <= 0:
            raise ValueError("Expiry must be positive")
        if not np.isfinite(self.anonymous_gate_m) or self.anonymous_gate_m <= 0:
            raise ValueError("Anonymous association gate must be positive")


def _centre(corners):
    return np.asarray(corners, dtype=float).mean(axis=0)


class TrackFilter:
    """Unconfirmed and anonymous observations can already limit speed.

    Median filtering occurs in world coordinates: moving ego coordinates from
    different times must not be averaged. Lost tracks are retained until expiry,
    but expiry is never a clearance signal. The behaviour layer owns that latch.

    Tracker IDs are trusted within their episode/source/model namespace. An
    observation without an ID is joined to the nearest same-class anonymous
    record within ``anonymous_gate_m`` (world XY, stationary-obstacle model), so
    a persistent object can confirm while flicker at changing places cannot.
    """

    def __init__(self, config=None):
        self.config = config or TrackFilterConfig()
        self._episode = None
        self._tracks = {}
        self._last_frame = -1
        self._last_time = -1.0

    def reset(self, episode_id):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("Episode required")
        self._episode = episode_id
        self._tracks.clear()
        self._last_frame, self._last_time = -1, -1.0

    def _anonymous_key(self, item, metadata, index, claimed):
        centre = _centre(item.corners_world_m)
        best, best_distance = None, self.config.anonymous_gate_m
        for key, record in self._tracks.items():
            if (
                not isinstance(key, tuple)
                or key[3] != "anonymous"
                or key[:3] != (metadata.episode_id, metadata.source_id, metadata.model_version)
                or key in claimed
                or record["last"].class_name != item.class_name
            ):
                continue
            distance = float(np.linalg.norm(_centre(record["last"].corners_world_m) - centre))
            if distance <= best_distance:
                best, best_distance = key, distance
        if best is not None:
            return best
        return (
            metadata.episode_id,
            metadata.source_id,
            metadata.model_version,
            "anonymous",
            metadata.frame_id,
            index,
        )

    def update(self, footprints, metadata):
        if metadata.episode_id != self._episode:
            raise ValueError("Filter episode mismatch")
        if metadata.frame_id <= self._last_frame or metadata.sim_time_s <= self._last_time:
            raise ValueError("Track observations must advance time and frame")
        self._last_frame, self._last_time = metadata.frame_id, metadata.sim_time_s
        # Prune before association: an expired record must not be re-joined or
        # lend its old confirmation history to a reappearance after a frame gap.
        for key, record in list(self._tracks.items()):
            if metadata.sim_time_s - record["last"].metadata.sim_time_s > self.config.expiry_s:
                del self._tracks[key]
        seen = set()
        for index, item in enumerate(footprints):
            if item.metadata != metadata:
                raise ValueError("Mixed observation metadata")
            key = item.identity or self._anonymous_key(item, metadata, index, seen)
            if key in seen:
                raise ValueError("Duplicate track identity")
            seen.add(key)
            record = self._tracks.setdefault(
                key,
                {
                    "history": deque(maxlen=self.config.window_frames),
                    "hits": deque(maxlen=self.config.window_frames),
                },
            )
            if "last" in record and record["last"].class_name != item.class_name:
                record["history"].clear()
                record["hits"].clear()
            record["history"].append(item.corners_world_m)
            record["last"] = item
        result = []
        for key, record in list(self._tracks.items()):
            item = record["last"]
            if metadata.sim_time_s - item.metadata.sim_time_s > self.config.expiry_s:
                del self._tracks[key]
                continue
            record["hits"].append(key in seen)
            confirmed = sum(record["hits"]) >= self.config.required_hits
            median = np.median(np.asarray(record["history"]), axis=0)
            # Keep the latest surface as well as median: smoothing must not hide
            # a suddenly nearer obstacle. Convex camera slabs -> covering AABB.
            all_points = np.vstack((median, np.asarray(item.corners_world_m)))
            lo, hi = all_points.min(axis=0), all_points.max(axis=0)
            corners = ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))
            result.append(
                replace(
                    item,
                    corners_world_m=corners,
                    confirmed=confirmed,
                    observed_this_frame=key in seen,
                )
            )
        return tuple(result)
