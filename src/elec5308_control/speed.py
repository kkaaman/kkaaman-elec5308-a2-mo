"""Discrete speed PID with conditional-integration anti-windup."""

from typing import Optional

from ._validation import clamp, finite, nonnegative, positive
from .types import ControllerConfig


class SpeedPID:
    """Map forward-speed error to a signed pedal effort in [-1, 1].

    Derivative is applied to the measurement, avoiding a derivative kick when
    target speed changes. Integration is rejected if it drives saturation further.
    """

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Clear integral and measurement history."""
        self._integral = 0.0
        self._previous_speed: Optional[float] = None

    def update(self, speed_mps: float, target_speed_mps: float, dt_s: float) -> float:
        """Return positive effort for throttle and negative effort for braking."""
        nonnegative("speed_mps", speed_mps)
        nonnegative("target_speed_mps", target_speed_mps)
        positive("dt_s", dt_s)
        error = target_speed_mps - speed_mps
        derivative = 0.0
        if self._previous_speed is not None and self.config.kd > 0:
            derivative = (self._previous_speed - speed_mps) / dt_s
        integral = self._integral + error * dt_s if self.config.ki > 0 else 0.0
        base = self.config.kp * error + self.config.kd * derivative
        output = base + self.config.ki * integral
        finite("PID output", output)
        if (output > 1.0 and error > 0) or (output < -1.0 and error < 0):
            integral = self._integral
            output = base + self.config.ki * integral
        finite("PID output", output)
        self._integral = integral
        self._previous_speed = speed_mps
        return clamp(output, -1.0, 1.0)
