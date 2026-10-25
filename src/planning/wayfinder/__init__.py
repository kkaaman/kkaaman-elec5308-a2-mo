"""Wayfinder v1: A* on the exported lane graph, named destinations and guidance (E3-01..03).

* :mod:`planning.wayfinder.astar`: shortest legal path over ``LaneGraph`` edges
* :mod:`planning.wayfinder.pois`: named points snapped to legal lane nodes
* :mod:`planning.wayfinder.route`: continuous routes with manoeuvre announcements
* :mod:`planning.wayfinder.guidance`: remaining distance, next turn and ETA policy
* :mod:`planning.wayfinder.minimap`: OpenCV mini-map frames for the HUD/replay

No module here imports a simulator; see ``docs/wayfinder.md``.
"""

from planning.wayfinder.astar import SearchResult, shortest_path
from planning.wayfinder.guidance import (
    GuidanceState,
    RouteTracker,
    SpeedPolicy,
    format_distance,
    format_eta,
)
from planning.wayfinder.pois import PointOfInterest, PoiRegistry
from planning.wayfinder.route import (
    DestinationCheck,
    Maneuver,
    RoutePlanningError,
    WayfinderRoute,
    check_destination,
    plan_named_route,
    plan_route,
)

__all__ = [
    "DestinationCheck",
    "GuidanceState",
    "Maneuver",
    "PointOfInterest",
    "PoiRegistry",
    "RoutePlanningError",
    "RouteTracker",
    "SearchResult",
    "SpeedPolicy",
    "WayfinderRoute",
    "check_destination",
    "format_distance",
    "format_eta",
    "plan_named_route",
    "plan_route",
    "shortest_path",
]
