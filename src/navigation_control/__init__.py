"""Forward-only path following in metres, seconds and radians.

No simulator or machine-learning dependency is required. See ``docs/mission.md``
for the controller contract and how the A2 stack uses it, and
``docs/interfaces.md`` for coordinate conventions.
"""

from .controller import PathFollower
from .types import (
    ControlCommand,
    ControlDiagnostics,
    ControllerConfig,
    ControlResult,
    ControlStatus,
    VehicleState,
    Waypoint,
)

__all__ = [
    "ControlCommand",
    "ControlDiagnostics",
    "ControllerConfig",
    "ControlResult",
    "ControlStatus",
    "PathFollower",
    "VehicleState",
    "Waypoint",
]
