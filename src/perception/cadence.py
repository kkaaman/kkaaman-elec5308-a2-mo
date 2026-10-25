"""Rate-limited optional workloads with explicit stale-result handling (TEST-17).

Optional perception workloads (semantic segmentation, the lane-departure
monitor, LiDAR processing) do not need to run every control tick. The A1 demo
skipped segmentation on most cycles but reused the cached mask indefinitely,
so a consumer could not tell a fresh mask from one that was many frames old.

:class:`RateLimitedWorkload` computes a value every ``period_ticks`` ticks and
reports the age of the value it hands back. A value older than
``max_age_ticks`` is reported as *invalid* and is not returned as current;
it remains available as ``stale_value`` for diagnostics only. Wall cost of
each computation is recorded in seconds for profiling.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["CadenceResult", "RateLimitedWorkload"]


@dataclass(frozen=True)
class CadenceResult:
    """Outcome of one ``step``.

    ``value`` is the current result or ``None`` when no valid result exists.
    ``age_ticks`` counts ticks since the value was computed (0 when computed
    this tick); ``None`` when nothing has been computed yet. ``computed`` says
    whether the workload ran this tick. ``valid`` is ``age_ticks <= max_age_ticks``.
    ``stale_value`` carries the last computed value only when it is invalid,
    so display code can label it explicitly; it must never be used as current.
    ``cost_s`` is the wall duration of this tick's computation (0 when skipped).
    """

    value: Any
    age_ticks: int | None
    computed: bool
    valid: bool
    stale_value: Any
    cost_s: float


class RateLimitedWorkload:
    """Run ``compute`` every ``period_ticks`` ticks; expire results after ``max_age_ticks``.

    ``period_ticks`` must be a positive integer. ``max_age_ticks`` must be a
    nonnegative integer; the default equals ``period_ticks - 1`` so a result is
    valid exactly until the next scheduled computation. A ``period_ticks`` of 1
    runs every tick. Ticks passed to :meth:`step` must be nonnegative integers
    that never decrease within an episode; :meth:`reset` starts a new episode.
    """

    def __init__(
        self,
        period_ticks: int,
        *,
        max_age_ticks: int | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        if type(period_ticks) is not int or period_ticks < 1:
            raise ValueError("period_ticks must be a positive integer")
        if max_age_ticks is None:
            max_age_ticks = period_ticks - 1
        if type(max_age_ticks) is not int or max_age_ticks < 0:
            raise ValueError("max_age_ticks must be a nonnegative integer")
        self.period_ticks = period_ticks
        self.max_age_ticks = max_age_ticks
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        self._last_tick: int | None = None
        self._computed_tick: int | None = None
        self._value: Any = None
        self.compute_count = 0
        self.skip_count = 0
        self.costs_s: list[float] = []

    def due(self, tick: int) -> bool:
        """True when ``tick`` is a scheduled computation tick or nothing is cached."""
        self._check_tick(tick)
        return self._computed_tick is None or tick - self._computed_tick >= self.period_ticks

    def _check_tick(self, tick: int) -> None:
        if type(tick) is not int or tick < 0:
            raise ValueError("tick must be a nonnegative integer")
        if self._last_tick is not None and tick < self._last_tick:
            raise ValueError("ticks must not decrease; call reset() for a new episode")

    def step(self, tick: int, compute: Callable[[], Any]) -> CadenceResult:
        """Advance to ``tick``; run ``compute`` when due and return the current result.

        A ``compute`` exception propagates after the cached value is discarded,
        so a failed optional workload never leaves a stale value as current.
        """
        self._check_tick(tick)
        self._last_tick = tick
        cost = 0.0
        computed = False
        if self.due(tick):
            started = self._clock()
            try:
                value = compute()
            except Exception:
                self._value, self._computed_tick = None, None
                raise
            cost = self._clock() - started
            if not math.isfinite(cost) or cost < 0:
                cost = 0.0
            self._value, self._computed_tick = value, tick
            self.compute_count += 1
            self.costs_s.append(cost)
            computed = True
        else:
            self.skip_count += 1
        if self._computed_tick is None:
            return CadenceResult(None, None, computed, False, None, cost)
        age = tick - self._computed_tick
        valid = age <= self.max_age_ticks
        return CadenceResult(
            self._value if valid else None,
            age,
            computed,
            valid,
            None if valid else self._value,
            cost,
        )
