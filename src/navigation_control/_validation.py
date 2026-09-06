"""Small numeric validation helpers shared by the control components."""

import math
from numbers import Real


def finite(name: str, value: float) -> None:
    """Reject non-real, boolean or non-finite scalar input with a clear error."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")


def nonnegative(name: str, value: float) -> None:
    """Require a finite scalar greater than or equal to zero."""
    finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def positive(name: str, value: float) -> None:
    """Require a finite scalar strictly greater than zero."""
    finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def clamp(value: float, lower: float, upper: float) -> float:
    """Clip a finite value to the inclusive bounds."""
    return max(lower, min(value, upper))
