"""Turn-by-turn guidance state: remaining distance, next manoeuvre and ETA (E3-03).

Units are metres, metres per second and seconds throughout; display strings
are produced only by :func:`format_distance` and :func:`format_eta`. ETA is
derived from the remaining path length and a :class:`SpeedPolicy`:

* ``reference="measured"`` divides by the current speed, clamped to at least
  the policy's ``cruise_speed_mps`` upper bound and never below the paused
  threshold, so no division by zero can occur;
* ``reference="cruise"`` divides by the policy cruise speed, giving a
  planning ETA that does not fluctuate with the controller.

When the vehicle is at or below ``paused_threshold_mps`` the ETA is reported
as ``paused`` (``eta_s`` is ``None``) rather than infinite or divided by zero;
within ``arrival_radius_m`` of the goal it is ``arrived``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from planning.wayfinder.route import Maneuver, WayfinderRoute

__all__ = ["GuidanceState", "RouteTracker", "SpeedPolicy", "format_distance", "format_eta"]

ETA_STATES = ("moving", "paused", "arrived")


def _finite(name, value, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return float(value)


@dataclass(frozen=True)
class SpeedPolicy:
    cruise_speed_mps: float = 4.0
    paused_threshold_mps: float = 0.1
    arrival_radius_m: float = 1.0
    reference: str = "measured"

    def __post_init__(self) -> None:
        _finite("cruise_speed_mps", self.cruise_speed_mps)
        _finite("paused_threshold_mps", self.paused_threshold_mps)
        _finite("arrival_radius_m", self.arrival_radius_m, 0.0)
        if self.paused_threshold_mps <= 0 or self.cruise_speed_mps <= self.paused_threshold_mps:
            raise ValueError("require 0 < paused_threshold_mps < cruise_speed_mps")
        if self.reference not in ("measured", "cruise"):
            raise ValueError("reference must be 'measured' or 'cruise'")

    def eta_s(self, remaining_m: float, speed_mps: float, lateral_m: float = 0.0) -> tuple[str, float | None]:
        """Return ``(state, eta_s)``; ``eta_s`` is None unless the state is moving.

        Arrival needs both the remaining arc and the lateral offset from the route
        inside ``arrival_radius_m``: a car level with the goal but two metres beside
        it has not arrived.
        """
        remaining_m = _finite("remaining_m", remaining_m, 0.0)
        speed_mps = _finite("speed_mps", speed_mps, 0.0)
        lateral_m = _finite("lateral_m", lateral_m, 0.0)
        if remaining_m <= self.arrival_radius_m and lateral_m <= self.arrival_radius_m:
            return "arrived", 0.0
        if speed_mps <= self.paused_threshold_mps:
            return "paused", None
        reference = self.cruise_speed_mps if self.reference == "cruise" else min(speed_mps, self.cruise_speed_mps)
        reference = max(reference, self.paused_threshold_mps)
        return "moving", remaining_m / reference


@dataclass(frozen=True)
class GuidanceState:
    time_s: float
    x_m: float
    y_m: float
    speed_mps: float
    progress_m: float
    remaining_m: float
    off_route_m: float
    next_maneuver: str
    next_maneuver_m: float
    eta_state: str
    eta_s: float | None
    units: str = "m, m/s, s"

    def __post_init__(self) -> None:
        if self.eta_state not in ETA_STATES:
            raise ValueError("unknown ETA state")
        if (self.eta_s is None) != (self.eta_state == "paused"):
            raise ValueError("eta_s is None exactly when paused")

    def to_dict(self) -> dict:
        return asdict(self)


class RouteTracker:
    """Project poses onto one route; progress is monotonic and never resets.

    After the first fix, the nearest route segment is searched forward from the
    last matched segment, so a route that passes near itself (for example the
    opposite lane of a road already driven) cannot pull progress backwards.
    """

    def __init__(self, route: WayfinderRoute, policy: SpeedPolicy | None = None, search_window_m: float = 30.0) -> None:
        self.route = route
        self.policy = policy or SpeedPolicy()
        self.search_window_m = _finite("search_window_m", search_window_m, 1.0)
        points = [p[:2] for p in route.points_xyz_m]
        self._points = points
        self._arc = [0.0]
        for a, b in zip(points, points[1:]):
            self._arc.append(self._arc[-1] + math.dist(a, b))
        self._segment = 0
        self._progress_m = 0.0
        self._last_time_s: float | None = None

    @property
    def length_m(self) -> float:
        return self._arc[-1]

    @property
    def progress_m(self) -> float:
        return self._progress_m

    def _project(self, x: float, y: float) -> tuple[int, float, float]:
        """Return ``(segment, progress_m, lateral_m)`` for the nearest forward segment.

        The first fix searches the whole route; later fixes search forward from
        the last matched segment until the arc length exceeds the best match
        so far by ``search_window_m``.
        """
        best = None
        limit = math.inf if self._last_time_s is None else self._arc[self._segment] + self.search_window_m
        for index in range(self._segment, len(self._points) - 1):
            if self._arc[index] > limit and best is not None:
                break
            (ax, ay), (bx, by) = self._points[index], self._points[index + 1]
            dx, dy = bx - ax, by - ay
            length_sq = dx * dx + dy * dy
            t = min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / length_sq))
            px, py = ax + t * dx, ay + t * dy
            distance = math.hypot(x - px, y - py)
            progress = self._arc[index] + t * math.sqrt(length_sq)
            if best is None or distance < best[2] - 1e-9:
                best = (index, progress, distance)
                limit = max(limit, progress + self.search_window_m)
        assert best is not None
        return best

    def next_maneuver(self, progress_m: float) -> Maneuver:
        for maneuver in self.route.maneuvers:
            if maneuver.route_s_m + 1e-6 >= progress_m or maneuver.kind == "arrive":
                return maneuver
        return self.route.maneuvers[-1]

    def update(self, time_s: float, x_m: float, y_m: float, speed_mps: float) -> GuidanceState:
        time_s = _finite("time_s", time_s)
        x_m, y_m = _finite("x_m", x_m), _finite("y_m", y_m)
        speed_mps = _finite("speed_mps", speed_mps, 0.0)
        if self._last_time_s is not None and time_s < self._last_time_s:
            raise ValueError("time_s must not decrease")
        segment, progress, lateral = self._project(x_m, y_m)
        self._last_time_s = time_s
        if progress >= self._progress_m:
            self._segment, self._progress_m = segment, progress
        remaining = max(0.0, self.length_m - self._progress_m)
        maneuver = self.next_maneuver(self._progress_m)
        state, eta = self.policy.eta_s(remaining, speed_mps, lateral)
        return GuidanceState(time_s, x_m, y_m, speed_mps, self._progress_m, remaining, lateral,
                             maneuver.kind, max(0.0, maneuver.route_s_m - self._progress_m), state, eta)


def format_distance(metres: float) -> str:
    """Metres under one kilometre, otherwise kilometres with two decimals."""
    metres = _finite("metres", metres, 0.0)
    return f"{metres:.0f} m" if metres < 1000 else f"{metres / 1000:.2f} km"


def format_eta(state: str, eta_s: float | None) -> str:
    """``mm:ss`` while moving; ``paused`` or ``arrived`` otherwise."""
    if state == "paused":
        return "ETA paused"
    if state == "arrived":
        return "arrived"
    if eta_s is None or not math.isfinite(eta_s) or eta_s < 0:
        raise ValueError("moving ETA requires a finite nonnegative time")
    minutes, seconds = divmod(int(round(eta_s)), 60)
    return f"ETA {minutes:d}:{seconds:02d}"
