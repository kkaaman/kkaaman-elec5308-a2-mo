"""Geometric Pure Pursuit lateral controller."""

import math

from ._validation import clamp
from .types import ControllerConfig, VehicleState, Waypoint


class PurePursuit:
    """Compute bounded front-wheel angle from a rear-axle pose and lookahead point."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config

    def steering_angle(self, state: VehicleState, target: Waypoint) -> float:
        """Return radians, positive left; reject a target strictly behind the car.

        Curvature is 2*y_local / distance**2. Using actual chord length instead
        of configured lookahead handles interpolation and shortened end targets.
        """
        dx, dy = target.x_m - state.x_m, target.y_m - state.y_m
        cosine, sine = math.cos(state.yaw_rad), math.sin(state.yaw_rad)
        x_local = cosine * dx + sine * dy
        y_local = -sine * dx + cosine * dy
        if x_local < -1e-8:
            raise ValueError("target is behind vehicle; check heading, path order and pose frame")
        distance = math.hypot(dx, dy)
        if distance < 1e-9:
            return 0.0
        curvature = 2.0 * (y_local / distance) / distance
        angle = math.atan(self.config.wheelbase_m * curvature)
        limit = self.config.max_steering_angle_rad
        return clamp(angle, -limit, limit)
