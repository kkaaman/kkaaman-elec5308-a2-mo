"""Freshness-aware composition of existing tracking, clearance and governance."""

from dataclasses import dataclass

from navigation_control.types import ControlCommand
from planning.behaviour import ClearanceGuard


@dataclass(frozen=True)
class CorridorObservation:
    """Provider-owned perceived geometry; never evaluation actor geometry.

    clearance_valid requires independently validated fresh corridor coverage.
    A valid empty tracker batch alone cannot set it.
    """

    metadata: object
    valid: bool
    obstacle_gap_m: float | None
    visible_distance_m: float | None
    clearance_valid: bool = False


class TrackedGovernancePolicy:
    def __init__(self, temporal_filter, governor, observation_provider, controller, *, dt_s):
        self.dt_s = dt_s
        self.temporal_filter = temporal_filter
        self.governor = governor
        self.observation_provider = observation_provider
        self.controller = controller
        self.guard = ClearanceGuard()
        self.last_target_mps = 0.0
        self.last_emergency = False
        self._interrupted_blocker = False
        self._ever_operational = False
        self.observation_valid = False

    def reset(self, *, episode_id, graph):
        self.episode_id = episode_id
        self.temporal_filter.reset(episode_id=episode_id)
        self.governor.reset()
        self.guard.reset()
        self.last_target_mps = 0.0
        self.last_emergency = False
        self._interrupted_blocker = False
        self._ever_operational = False
        self.observation_valid = False
        if hasattr(self.controller, "reset"):
            self.controller.reset()

    def on_hold(self, frame):
        self.last_target_mps = 0.0
        self.governor.reset()
        self._interrupted_blocker |= self._ever_operational and self.guard.blocked_latch
        self.guard.invalidate()
        if hasattr(self.controller, "hold"):
            self.controller.hold(frame, self.dt_s)
        self.temporal_filter.reset(episode_id=self.episode_id)
        # Do not fabricate a clear observation or reset the blocked latch.

    def evaluate(self, frame, batch, *, fresh, ready):
        self.last_target_mps = 0.0
        self.last_emergency = False
        self.observation_valid = False
        if not fresh or not batch.valid or frame.depth_m is None:
            self.on_hold(frame)
            return ControlCommand(0, 0, 1)
        tracks = self.temporal_filter.update(frame, batch)
        self.track_diagnostics = [
            dict(
                class_name=t.detection.class_name,
                track_id=t.identity.track_id,
                current=t.current,
                depth_valid=t.depth_valid,
                reason=t.invalid_reason,
            )
            for t in tracks.tracks
        ]
        self.track_diagnostics.extend(
            dict(
                class_name=d.class_name,
                track_id=None,
                current=True,
                depth_valid=False,
                reason="missing_id",
            )
            for d in tracks.untracked
        )
        observation = self.observation_provider(frame, batch, tracks)
        if observation.metadata != batch.metadata:
            raise ValueError("Corridor observation source mismatch")
        unknown_tracks = bool(tracks.untracked or tracks.expired) or any(
            not t.current or not t.depth_valid for t in tracks.tracks
        )
        valid = (
            ready
            and observation.valid
            and not unknown_tracks
            and observation.visible_distance_m is not None
            and (observation.obstacle_gap_m is not None or observation.clearance_valid)
        )
        self.observation_valid = valid
        clear = self.guard.update(
            time_s=frame.sim_time_s,
            obstacle_present=observation.obstacle_gap_m is not None,
            observation_valid=valid,
        )
        if not valid:
            self._interrupted_blocker |= self._ever_operational and self.guard.blocked_latch
            self.governor.reset()
            if hasattr(self.controller, "hold"):
                self.controller.hold(frame, self.dt_s)
            return ControlCommand(0, 0, 1)
        if clear:
            self._interrupted_blocker = False
        # Fresh visible blockers may be approached conservatively; an interrupted
        # existing blocker/HOLD cannot disappear into implicit clearance.
        hold = self._interrupted_blocker or (observation.obstacle_gap_m is None and not clear)
        target, emergency = self.governor.limit(
            speed_mps=frame.ego.state.speed_mps,
            dt_s=self.dt_s,
            obstacle_gap_m=observation.obstacle_gap_m,
            visible_distance_m=observation.visible_distance_m,
            hold=hold,
        )
        self.last_target_mps, self.last_emergency = target, emergency
        command = self.controller(frame, target, self.dt_s)
        if hold or emergency:
            return ControlCommand(0, 0, 1)
        self._ever_operational |= target > 0
        return command
