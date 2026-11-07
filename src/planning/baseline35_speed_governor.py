"""Stopping-distance and preview curvature ceilings with immediate reductions."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GovernorConfig:
    cruise_mps: float = 5.0
    deceleration_mps2: float = 1.5
    lateral_acceleration_mps2: float = 1.0
    response_budget_s: float = 0.5
    stop_margin_m: float = 4.0
    acceleration_mps2: float = 1.0
    emergency_margin_m: float = 2.5

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.emergency_margin_m >= self.stop_margin_m:
            raise ValueError("Emergency margin must be below desired stop margin")


def stopping_distance(speed, deceleration, response_s):
    if (
        not all(math.isfinite(x) for x in (speed, deceleration, response_s))
        or speed < 0
        or deceleration <= 0
        or response_s < 0
    ):
        raise ValueError("Invalid stopping budget")
    return speed * response_s + speed * speed / (2 * deceleration)


def speed_for_distance(distance, deceleration, response_s):
    if not math.isfinite(distance):
        raise ValueError("Distance must be finite")
    stopping_distance(0, deceleration, response_s)
    return max(
        0.0,
        math.sqrt((deceleration * response_s) ** 2 + 2 * deceleration * max(0.0, distance))
        - deceleration * response_s,
    )


class SpeedGovernor:
    def __init__(self, config=None):
        self.config = config or GovernorConfig()
        self.previous_target = 0.0

    def reset(self):
        self.previous_target = 0.0

    def limit(self, *, speed, dt, curvature=(), obstacle_gap=None, visible_m=None, hold=False):
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(speed) or speed < 0:
            raise ValueError("Invalid speed or timestep")
        c = self.config
        ceiling = c.cruise_mps
        for distance, kappa in curvature:
            if not all(math.isfinite(x) for x in (distance, kappa)) or min(distance, kappa) < 0:
                raise ValueError("Invalid curvature preview")
            if kappa > 1e-8:
                curve_speed = math.sqrt(c.lateral_acceleration_mps2 / kappa)
                usable = max(0.0, distance - speed * c.response_budget_s)
                ceiling = min(ceiling, math.sqrt(curve_speed**2 + 2 * c.deceleration_mps2 * usable))
        for distance in (obstacle_gap, visible_m):
            if distance is not None:
                ceiling = min(
                    ceiling,
                    speed_for_distance(
                        distance - c.stop_margin_m, c.deceleration_mps2, c.response_budget_s
                    ),
                )
        emergency = obstacle_gap is not None and obstacle_gap <= c.emergency_margin_m + speed * dt
        if hold or emergency:
            target = 0.0
        else:
            # Rate-limit increases from whichever is higher: the last target or
            # the measured speed. Restarting from a stale zero after a brief
            # stop would otherwise command braking down to the ramp.
            start = max(self.previous_target, min(speed, ceiling))
            target = min(ceiling, start + c.acceleration_mps2 * dt)
        self.previous_target = target
        return target, emergency
