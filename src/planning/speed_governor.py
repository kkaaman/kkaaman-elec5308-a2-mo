"""Obstacle and curvature target-speed governance for BL-D-07."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GovernorConfig:
    """Measured parameters used by the baseline speed governor.

    Values are supplied explicitly so BL-D-07 does not silently freeze the
    provisional BL-D-06 calibration values.
    """

    cruise_mps: float
    deceleration_mps2: float
    lateral_acceleration_mps2: float
    response_budget_s: float
    stop_margin_m: float
    acceleration_mps2: float
    emergency_margin_m: float

    def __post_init__(self) -> None:
        positive_fields = (
            "cruise_mps",
            "deceleration_mps2",
            "lateral_acceleration_mps2",
            "stop_margin_m",
            "acceleration_mps2",
            "emergency_margin_m",
        )

        for name in positive_fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite value")

        if (
            isinstance(self.response_budget_s, bool)
            or not isinstance(self.response_budget_s, (int, float))
            or not math.isfinite(self.response_budget_s)
            or self.response_budget_s < 0
        ):
            raise ValueError("response_budget_s must be a non-negative finite value")

        if self.emergency_margin_m >= self.stop_margin_m:
            raise ValueError(
                "emergency_margin_m must be smaller than stop_margin_m"
            )


def stopping_distance(
    speed_mps: float,
    deceleration_mps2: float,
    response_s: float,
) -> float:
    """Return response distance plus constant-deceleration braking distance."""

    values = (speed_mps, deceleration_mps2, response_s)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise ValueError("stopping budget values must be numeric")

    if (
        not all(math.isfinite(value) for value in values)
        or speed_mps < 0
        or deceleration_mps2 <= 0
        or response_s < 0
    ):
        raise ValueError("invalid stopping budget")

    return (
        speed_mps * response_s
        + speed_mps**2 / (2.0 * deceleration_mps2)
    )


def speed_for_distance(
    distance_m: float,
    deceleration_mps2: float,
    response_s: float,
) -> float:
    """Return the maximum speed whose stopping distance fits ``distance_m``."""

    if (
        isinstance(distance_m, bool)
        or not isinstance(distance_m, (int, float))
        or not math.isfinite(distance_m)
    ):
        raise ValueError("distance_m must be finite")

    stopping_distance(0.0, deceleration_mps2, response_s)

    usable_distance_m = max(0.0, distance_m)

    return max(
        0.0,
        math.sqrt(
            (deceleration_mps2 * response_s) ** 2
            + 2.0 * deceleration_mps2 * usable_distance_m
        )
        - deceleration_mps2 * response_s,
    )


class SpeedGovernor:
    """Combine cruise, curvature and obstacle limits into one target speed."""

    def __init__(self, config: GovernorConfig) -> None:
        if not isinstance(config, GovernorConfig):
            raise TypeError("config must be GovernorConfig")

        self.config = config
        self.previous_target_mps = 0.0

    def reset(self) -> None:
        """Reset episode-local rate-limit state."""

        self.previous_target_mps = 0.0

    def limit(
        self,
        *,
        speed_mps: float,
        dt_s: float,
        curvature: Iterable[tuple[float, float]] = (),
        obstacle_gap_m: float | None = None,
        visible_distance_m: float | None = None,
        hold: bool = False,
    ) -> tuple[float, bool]:
        """Return ``(target_speed_mps, emergency)``.

        Speed increases are rate limited. Reductions are applied immediately,
        and HOLD or emergency conditions override all normal governor output in
        the same cycle.
        """

        if (
            isinstance(speed_mps, bool)
            or not isinstance(speed_mps, (int, float))
            or not math.isfinite(speed_mps)
            or speed_mps < 0
        ):
            raise ValueError("speed_mps must be a non-negative finite value")

        if (
            isinstance(dt_s, bool)
            or not isinstance(dt_s, (int, float))
            or not math.isfinite(dt_s)
            or dt_s <= 0
        ):
            raise ValueError("dt_s must be a positive finite value")

        config = self.config
        ceiling_mps = config.cruise_mps

        for distance_m, curvature_per_m in curvature:
            if (
                not math.isfinite(distance_m)
                or not math.isfinite(curvature_per_m)
                or distance_m < 0
            ):
                raise ValueError("invalid curvature preview")

            abs_curvature = abs(curvature_per_m)

            if abs_curvature <= 1e-8:
                continue

            curve_speed_mps = math.sqrt(
                config.lateral_acceleration_mps2 / abs_curvature
            )

            usable_distance_m = max(
                0.0,
                distance_m - speed_mps * config.response_budget_s,
            )

            reachable_curve_speed_mps = math.sqrt(
                curve_speed_mps**2
                + 2.0
                * config.deceleration_mps2
                * usable_distance_m
            )

            ceiling_mps = min(
                ceiling_mps,
                reachable_curve_speed_mps,
            )

        for distance_m in (obstacle_gap_m, visible_distance_m):
            if distance_m is None:
                continue

            if (
                isinstance(distance_m, bool)
                or not isinstance(distance_m, (int, float))
                or not math.isfinite(distance_m)
            ):
                raise ValueError("distance limits must be finite")

            ceiling_mps = min(
                ceiling_mps,
                speed_for_distance(
                    distance_m - config.stop_margin_m,
                    config.deceleration_mps2,
                    config.response_budget_s,
                ),
            )

        emergency = (
            obstacle_gap_m is not None
            and obstacle_gap_m
            <= config.emergency_margin_m + speed_mps * dt_s
        )

        if hold or emergency:
            target_mps = 0.0
        else:
            maximum_increase_mps = (
                self.previous_target_mps
                + config.acceleration_mps2 * dt_s
            )

            target_mps = min(
                ceiling_mps,
                maximum_increase_mps,
            )

        self.previous_target_mps = target_mps

        return target_mps, emergency