"""Fuse LiDAR tracks with the RGB-D corridor observation for the governor (E2-04).

:class:`LidarCorridorSource` binds the physical LiDAR stream to the ``"lidar"``
source namespace, runs ground removal, corridor crop, clustering and N-of-M
tracking for each sweep, and answers one :class:`SourceCorridorReport` per frame.
:class:`FusedObservationProvider` wraps an existing RGB-D observation provider so
``TrackedGovernancePolicy`` keeps its contract (``observation.metadata`` equals the
detector batch metadata) while receiving the conservative fused gap:

* the nearer valid gap wins;
* a missing, stale or invalid LiDAR sweep makes the observation invalid (HOLD),
  never clear;
* a lost RGB-D observation is likewise invalid even if LiDAR sees open road;
* LiDAR asserts clearance only when it has positively seen clear road bins from
  the bumper out to at least ``policy.clear_start_m`` with no cluster in the
  corridor; one clear bin is not clearance, and the RGB-D clearance rule must
  still agree.

No simulator import; callers deliver :class:`LidarSweep` objects from the adapter.
"""

from __future__ import annotations

from autonomy.obstacle_policy import CorridorObservation
from obstacle_perception.fusion import SourceCorridorReport, fuse_corridor_reports
from obstacle_perception.lidar import LidarSweep
from obstacle_perception.lidar_obstacles import (
    LIDAR_SOURCE_ID,
    EgoBody,
    LidarObstaclePolicy,
    LidarTrackFilter,
    extract_obstacles,
)


class LidarCorridorSource:
    """Per-episode LiDAR obstacle pipeline producing source-namespaced reports.

    ``route`` is the :class:`~planning.corridor.RouteGeometry` (metres) used for
    the corridor crop; ``ego_pose``/``ego_s_m`` place the rear axle on it (default:
    the route is in the ego frame). ``max_age_s`` bounds how far (simulation
    seconds) the latest sweep may lag the frame it is asked about.
    """

    def __init__(
        self,
        route,
        *,
        policy=None,
        body=None,
        track_policy=None,
        gate_m=1.5,
        source_id=LIDAR_SOURCE_ID,
        max_age_s=0.0,
    ):
        self.route = route
        self.policy = policy or LidarObstaclePolicy()
        self.body = body or EgoBody()
        self.filter = LidarTrackFilter(source_id=source_id, policy=track_policy, gate_m=gate_m)
        self.source_id = source_id
        if max_age_s < 0:
            raise ValueError("max_age_s must be nonnegative")
        self.max_age_s = max_age_s
        self.ego_pose = None
        self.ego_s_m = 0.0
        self._episode = None
        self.last_batch = None
        self.last_tracks = None

    def reset(self, *, episode_id):
        self._episode = episode_id
        self.filter.reset(episode_id=episode_id)
        self.last_batch = None
        self.last_tracks = None

    def observe(self, sweep, *, ego_pose=None, ego_s_m=0.0):
        """Process one sweep (ego frame) and return its :class:`LidarTrackBatch`."""
        if not isinstance(sweep, LidarSweep):
            raise TypeError("observe expects a LidarSweep")
        if sweep.episode_id != self._episode:
            raise ValueError("Reset the LiDAR source for this episode")
        self.last_batch = extract_obstacles(
            sweep,
            self.route,
            policy=self.policy,
            body=self.body,
            ego_pose=ego_pose,
            ego_s_m=ego_s_m,
        )
        self.last_tracks = self.filter.update(self.last_batch)
        return self.last_tracks

    def report(self, frame):
        """Corridor report for ``frame`` (a FrameBundle-like with episode/frame/sim time)."""
        tracks = self.last_tracks
        invalid = None
        if tracks is None:
            invalid = "missing_sweep"
        elif tracks.metadata.episode_id != frame.episode_id:
            invalid = "episode_mismatch"
        elif tracks.metadata.frame_id > frame.frame_id or (
            frame.sim_time_s - tracks.metadata.sim_time_s > self.max_age_s + 1e-9
        ):
            invalid = "stale_sweep"
        elif not tracks.valid:
            invalid = tracks.invalid_reason
        elif tracks.unknown_tracks:
            invalid = "unconfirmed_tracks"
        if invalid is not None:
            return SourceCorridorReport(self.source_id, False, invalid_reason=invalid)
        gap = tracks.nearest_gap_m
        seen_clear = tracks.seen_clear_m
        # Clearance is a positive observation of clear road, not the absence of a
        # cluster: the clear bins must run from the bumper out to clear_start_m.
        clear = gap is None and seen_clear is not None and seen_clear >= self.policy.clear_start_m
        return SourceCorridorReport(
            self.source_id,
            True,
            gap,
            tracks.visible_distance_m,
            bool(clear),
            tuple(t.identity for t in tracks.tracks),
        )


def rgbd_report(batch, tracks, observation):
    """Wrap an RGB-D ``CorridorObservation`` as a source report for fusion."""
    if observation.metadata != batch.metadata:
        raise ValueError("Corridor observation source mismatch")
    valid = bool(observation.valid)
    return SourceCorridorReport(
        batch.metadata.source_id,
        valid,
        observation.obstacle_gap_m if valid else None,
        observation.visible_distance_m if valid else None,
        bool(valid and observation.clearance_valid),
        tuple(t.identity for t in tracks.tracks),
        None if valid else "rgbd_observation_invalid",
    )


class FusedObservationProvider:
    """Drop-in ``observation_provider`` combining RGB-D and LiDAR conservatively."""

    def __init__(self, rgbd_provider, lidar_source):
        self.rgbd_provider = rgbd_provider
        self.lidar_source = lidar_source
        self.last_fused = None

    def __call__(self, frame, batch, tracks):
        observation = self.rgbd_provider(frame, batch, tracks)
        rgbd = rgbd_report(batch, tracks, observation)
        lidar = self.lidar_source.report(frame)
        fused = fuse_corridor_reports(
            (rgbd, lidar), required_sources=(rgbd.source_id, lidar.source_id)
        )
        self.last_fused = fused
        return CorridorObservation(
            batch.metadata,
            fused.valid,
            fused.obstacle_gap_m,
            fused.visible_distance_m,
            fused.clearance_valid,
        )


__all__ = ["FusedObservationProvider", "LidarCorridorSource", "rgbd_report"]
