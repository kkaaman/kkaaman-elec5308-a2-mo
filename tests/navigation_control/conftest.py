"""Shared fixtures and import setup for navigation-control unit tests."""

import sys
from pathlib import Path

import pytest

# Keep this test group runnable from the repository root without requiring an
# editable installation while the team-level configuration is evolving.
SRC_DIRECTORY = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from navigation_control import (  # noqa: E402
    ControllerConfig,
    PathFollower,
    Waypoint,
)


@pytest.fixture
def config():
    """Return one consistent baseline vehicle/controller configuration."""
    return ControllerConfig(wheelbase_m=2.8, max_steering_angle_rad=0.6)


@pytest.fixture
def follower(config):
    """Return a controller with a simple 100 m straight reference path."""
    controller = PathFollower(config)
    controller.set_path([Waypoint(0, 0), Waypoint(100, 0)])
    return controller
