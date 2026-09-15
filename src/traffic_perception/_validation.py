import math
from numbers import Real

import numpy as np


def finite(name: str, value: float) -> None:
    """Require a finite real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")


def probability(name: str, value: float) -> None:
    """Require a probability in the inclusive range [0, 1]."""
    finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def positive(name: str, value: float) -> None:
    """Require a finite value greater than zero."""
    finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def rgb_image(image: np.ndarray) -> None:
    """Require a non-empty H x W x 3 NumPy image."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image dimensions must be non-zero")
