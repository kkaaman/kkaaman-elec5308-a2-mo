"""Sensor health on the car's side: how old each sensor's latest good sample is.

The rig says what happened on each tick (``SensorStatus``). This tracker turns that into
ages and overdue flags, measured against each sensor's expected period (one tick for
cameras, LiDAR, radar, IMU and speedometer; the fix period for GNSS). A scheduled
absence is never counted as a dropout.

**Policy is not fixed here.** Whether a missing sample matters depends on the motion and
on what the remaining sensors still cover. Losing rear radar while stopped is not the
same as losing front perception beside a crossing pedestrian. The planner asks
``stale(names, max_age_s)`` for the sensors that cover its current corridor and
manoeuvre. Any age limit it uses, for example a candidate 150 ms for front coverage,
must fit inside the response time assumed in its stopping-distance calculation.
"""

import math
from dataclasses import dataclass

from mapless.frame import SensorFrame, SensorState


@dataclass(frozen=True)
class SensorHealth:
    name: str
    kind: str
    expected_period_s: float
    age_s: float  # simulation time since the last OK sample (inf if never)
    overdue: bool  # age beyond the expected period plus half a tick
    consecutive_dropouts: int  # MISSING or INVALID ticks in a row
    last_state: SensorState


class HealthTracker:
    def __init__(self, suite, dt_s):
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be positive")
        self.dt_s = dt_s
        self.period = {}
        for spec in suite.sensors:
            self.period[spec.name] = spec.period_s if spec.kind == "gnss" else dt_s
        self.kind = {spec.name: spec.kind for spec in suite.sensors}
        self._last_ok = {name: None for name in self.period}
        self._dropouts = {name: 0 for name in self.period}
        self.report = {}

    def update(self, frame):
        if not isinstance(frame, SensorFrame):
            raise TypeError("frame must be a SensorFrame")
        seen = {s.name: s for s in frame.status}
        unknown = set(seen) - set(self.period)
        if unknown:
            raise ValueError(f"Frame reports sensors not in the suite: {sorted(unknown)}")
        report = {}
        for name, period in self.period.items():
            status = seen.get(name)
            state = status.state if status else SensorState.MISSING
            if state is SensorState.OK:
                self._last_ok[name] = status.capture_time_s
                self._dropouts[name] = 0
            elif state in (SensorState.MISSING, SensorState.INVALID):
                self._dropouts[name] += 1
            last = self._last_ok[name]
            age = math.inf if last is None else max(0.0, frame.sim_time_s - last)
            report[name] = SensorHealth(
                name=name,
                kind=self.kind[name],
                expected_period_s=period,
                age_s=age,
                overdue=age > period + self.dt_s / 2,
                consecutive_dropouts=self._dropouts[name],
                last_state=state,
            )
        self.report = report
        return report

    def stale(self, names, max_age_s):
        """Sensors among ``names`` whose latest good sample is older than ``max_age_s``."""
        if not math.isfinite(max_age_s) or max_age_s < 0:
            raise ValueError("max_age_s must be non-negative")
        missing = [n for n in names if n not in self.report]
        if missing:
            raise KeyError(f"No health yet for {missing}")
        return [n for n in names if self.report[n].age_s > max_age_s]
