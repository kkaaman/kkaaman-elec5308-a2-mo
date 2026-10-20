"""One fixed route installation per episode, with explicit temporary holds."""

import math
import re
from dataclasses import dataclass
from numbers import Real
from typing import Iterable

from navigation_control import (
    ControllerConfig,
    ControlResult,
    ControlStatus,
    PathFollower,
    VehicleState,
    Waypoint,
)


@dataclass(frozen=True)
class MissionEvent:
    """A lifecycle transition after the indexed successful control step."""

    step_index: int
    event: str
    route_sha256: str


class MissionEpisode:
    """Own a controller and one route for a single forward mission episode.

    Construct a fresh instance for each episode. There is deliberately no route
    replacement or reset operation: temporary holds only change the requested
    speed and completion permission. Invalid inputs leave lifecycle history
    unchanged. The caller must stop actuators if a control step raises.
    """

    def __init__(
        self,
        config: ControllerConfig,
        waypoints: Iterable[Waypoint],
        route_sha256: str,
    ) -> None:
        if not isinstance(route_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", route_sha256
        ) is None:
            raise ValueError("route_sha256 must be a lowercase SHA-256 digest")
        follower = PathFollower(config)
        follower.set_path(waypoints)
        self._follower = follower
        self._route_sha256 = route_sha256
        self._route_install_count = 1
        self._step_count = 0
        self._status = "ready"
        self._progress_m = 0.0
        self._progress_fraction = 0.0
        self._failure_reason = None
        self._events = [MissionEvent(0, "route_installed", route_sha256)]

    @property
    def route_sha256(self) -> str:
        """Return the frozen route identity supplied by the validated loader."""
        return self._route_sha256

    @property
    def route_install_count(self) -> int:
        """Return the number of route installations in this episode."""
        return self._route_install_count

    @property
    def step_count(self) -> int:
        """Return the number of successful control updates."""
        return self._step_count

    @property
    def status(self) -> str:
        """Return ready, running, holding, finished or failed."""
        return self._status

    @property
    def progress_m(self) -> float:
        """Return monotonic route progress from the last successful update."""
        return self._progress_m

    @property
    def progress_fraction(self) -> float:
        """Return route progress, which alone is not proof of completion."""
        return self._progress_fraction

    @property
    def failure_reason(self) -> str | None:
        """Return the retained failure reason, if the caller failed the run."""
        return self._failure_reason

    @property
    def events(self) -> tuple[MissionEvent, ...]:
        """Return an immutable snapshot of route and lifecycle events."""
        return tuple(self._events)

    def step(
        self,
        state: VehicleState,
        target_speed_mps: float,
        dt_s: float,
        hold: bool = False,
    ) -> ControlResult:
        """Calculate control without reinstalling the route or clearing progress.

        ``hold`` represents a temporary obstacle stop, including near the goal.
        Releasing it permits normal endpoint completion. An already completed
        mission remains finished even if a subsequent caller requests a hold.
        """
        if self._status == "failed":
            raise RuntimeError("Cannot step a failed mission; create a new episode")
        if type(hold) is not bool:
            raise TypeError("hold must be bool")
        if (
            isinstance(target_speed_mps, bool)
            or not isinstance(target_speed_mps, Real)
            or not math.isfinite(target_speed_mps)
            or target_speed_mps < 0
        ):
            raise ValueError("target_speed_mps must be finite and nonnegative")
        result = self._follower.update(
            state,
            0.0 if hold else target_speed_mps,
            dt_s,
            completion_allowed=not hold,
        )
        self._step_count += 1
        self._progress_m = result.diagnostics.progress_m
        self._progress_fraction = result.diagnostics.progress_fraction
        previous_status = self._status
        if result.status == ControlStatus.FINISHED:
            self._status = "finished"
        else:
            self._status = "holding" if hold else "running"
        if self._status != previous_status:
            if previous_status == "holding":
                self._record("hold_released")
            if self._status == "finished":
                self._record("mission_finished")
            elif self._status == "holding":
                self._record("hold_started")
        return result

    def fail(self, reason: str) -> None:
        """Latch the first run failure; stopping actuators remains the caller's job."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("failure reason must be a nonempty string")
        if self._failure_reason is None:
            self._failure_reason = reason
            self._status = "failed"
            self._record("mission_failed")

    def _record(self, event: str) -> None:
        self._events.append(MissionEvent(self._step_count, event, self._route_sha256))
