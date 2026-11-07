"""Conservative obstacle-clearance guards for baseline behaviour (BL-D-05)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ClearanceGuardConfig:
    """Requirements for confirming that a previously blocked route is clear."""

    clear_frames: int = 5
    clear_duration_s: float = 0.4

    def __post_init__(self) -> None:
        if type(self.clear_frames) is not int or self.clear_frames < 1:
            raise ValueError("clear_frames must be a positive integer")

        if (
            isinstance(self.clear_duration_s, bool)
            or not math.isfinite(self.clear_duration_s)
            or self.clear_duration_s <= 0
        ):
            raise ValueError("clear_duration_s must be a positive finite value")


class ClearanceGuard:
    """Track conservative clearance evidence for a blocked route.

    A missing or expired obstacle track is not sufficient evidence of
    clearance. Clearance is confirmed only after fresh, valid observations
    report the corridor clear for both the configured frame count and
    duration.
    """

    def __init__(self, config: ClearanceGuardConfig | None = None) -> None:
        self.config = config or ClearanceGuardConfig()
        self.reset()

    def reset(self) -> None:
        """Reset all episode-local guard state."""

        self._blocked_latch = False
        self._clear_count = 0
        self._clear_since_s: float | None = None
        self._last_time_s: float | None = None

    @property
    def blocked_latch(self) -> bool:
        """Whether an obstacle has been observed since the last confirmed clear."""

        return self._blocked_latch

    @property
    def clear_count(self) -> int:
        """Number of consecutive fresh observations reporting a clear corridor."""

        return self._clear_count

    def invalidate(self) -> None:
        """Discard partial clearance evidence without releasing a blocked latch."""
        self._clear_count = 0
        self._clear_since_s = None

    def update(
        self,
        *,
        time_s: float,
        obstacle_present: bool,
        observation_valid: bool,
    ) -> bool:
        """Update clearance evidence and return whether clearance is confirmed.

        ``observation_valid=False`` always resets accumulated clearance
        evidence. Therefore track expiry, sensor loss or stale observations
        cannot independently authorize motion.
        """

        if isinstance(time_s, bool) or not math.isfinite(time_s):
            raise ValueError("time_s must be finite")

        if self._last_time_s is not None and time_s <= self._last_time_s:
            raise ValueError("time_s must advance monotonically")

        self._last_time_s = time_s

        if obstacle_present:
            self._blocked_latch = True

        if not observation_valid or obstacle_present:
            self._clear_count = 0
            self._clear_since_s = None
            return False

        if not self._blocked_latch:
            return True

        self._clear_count += 1

        if self._clear_since_s is None:
            self._clear_since_s = time_s

        duration_s = time_s - self._clear_since_s

        confirmed = (
            self._clear_count >= self.config.clear_frames
            and duration_s + 1e-9 >= self.config.clear_duration_s
        )

        if confirmed:
            self._blocked_latch = False
            self._clear_count = 0
            self._clear_since_s = None

        return confirmed