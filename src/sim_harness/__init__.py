"""CARLA-free fixed-step world and scripted sensor fixtures (BL-F-03).

These fixtures support deterministic controller and contract checks, not claims
of real perception, obstacle avoidance or calibrated vehicle performance.
"""

from .kinematic import KinematicConfig, KinematicWorld
from .synthetic import (
    SyntheticDetection,
    SyntheticDetector,
    SyntheticFrame,
    SyntheticObstacleSource,
    SyntheticScenario,
)

__all__ = [
    "KinematicConfig",
    "KinematicWorld",
    "SyntheticDetection",
    "SyntheticDetector",
    "SyntheticFrame",
    "SyntheticObstacleSource",
    "SyntheticScenario",
]
