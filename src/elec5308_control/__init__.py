"""Forward-only path following in metres, seconds and radians.

No simulator or machine-learning dependency is required. See ``docs/control.md``
for coordinate conventions, operating assumptions and a complete example.
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
