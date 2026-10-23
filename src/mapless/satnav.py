"""Sat-nav route: the only navigation input the mapless car receives.

The route is a list of GPS points, each carrying a high-level command, in the CARLA
Leaderboard 2.0 format. Points can be far apart: they say where to go, not where the lane
is.

Commands:
- LEFT, RIGHT and STRAIGHT sit at a junction entry. The manoeuvre runs from that point to
  the next one, which is the junction exit.
- LANEFOLLOW sits at a junction exit, along the road and at the goal.
- CHANGELANELEFT and CHANGELANERIGHT ask for a lane change on the way to that point.

``RouteProgress`` reports the command of the point ahead, which says what is coming. It
also reports the command of the point just passed, which says whether the car is in the
middle of a turn.

The car turns GPS into local metres with an ordinary tangent-plane approximation,
anchored at the route's first point (+x east, +y north). It needs nothing from the
simulator for that. Over a few kilometres this agrees with CARLA's own projection to
millimetres, far below GNSS noise.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

EARTH_RADIUS_M = 6378137.0  # WGS84 equatorial radius, as CARLA uses
SCHEMA_VERSION = 1


class Command(str, Enum):
    LANEFOLLOW = "LANEFOLLOW"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STRAIGHT = "STRAIGHT"
    CHANGELANELEFT = "CHANGELANELEFT"
    CHANGELANERIGHT = "CHANGELANERIGHT"

    @property
    def is_junction(self):
        return self in (Command.LEFT, Command.RIGHT, Command.STRAIGHT)


@dataclass(frozen=True)
class RoutePoint:
    latitude_deg: float
    longitude_deg: float
    command: Command

    def __post_init__(self):
        for name in ("latitude_deg", "longitude_deg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not -90 < self.latitude_deg < 90 or not -180 <= self.longitude_deg <= 180:
            raise ValueError("Route point out of range")
        object.__setattr__(self, "command", Command(self.command))


@dataclass(frozen=True)
class SatNavRoute:
    name: str
    points: tuple
    provenance: dict | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported sat-nav route schema")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Route needs a name")
        points = tuple(self.points)
        if len(points) < 2 or any(not isinstance(p, RoutePoint) for p in points):
            raise ValueError("Route needs at least two RoutePoint values")
        if points[-1].command is not Command.LANEFOLLOW:
            raise ValueError("The goal point must carry LANEFOLLOW")
        plane = LocalTangentPlane(points[0].latitude_deg, points[0].longitude_deg)
        xy = [plane.to_local(p.latitude_deg, p.longitude_deg) for p in points]
        if any(math.dist(a, b) < 0.5 for a, b in zip(xy, xy[1:])):
            raise ValueError("Consecutive route points must be at least 0.5 m apart")
        object.__setattr__(self, "points", points)

    def as_dict(self):
        return dict(
            schema_version=self.schema_version,
            name=self.name,
            points=[
                dict(latitude_deg=p.latitude_deg, longitude_deg=p.longitude_deg, command=p.command.value)
                for p in self.points
            ],
            provenance=self.provenance,
        )

    @property
    def sha256(self):
        """Hash of what the car receives (name and points; provenance excluded)."""
        body = dict(self.as_dict(), provenance=None)
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, document):
        unknown = set(document) - {"schema_version", "name", "points", "provenance"}
        if unknown:
            raise ValueError(f"Unknown route fields: {', '.join(sorted(unknown))}")
        points = []
        for entry in document.get("points") or ():
            if set(entry) != {"latitude_deg", "longitude_deg", "command"}:
                raise ValueError("Each route point needs exactly latitude_deg, longitude_deg, command")
            points.append(RoutePoint(**entry))
        return cls(
            document.get("name", ""),
            tuple(points),
            document.get("provenance"),
            document.get("schema_version", SCHEMA_VERSION),
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))


class LocalTangentPlane:
    """Equirectangular east/north metres about a reference fix.

    ``north_sign`` -1 flips y so that the plane stays right-handed with the compass heading
    on a receiver whose latitude grows towards the compass's south (CARLA 0.9.16; see
    ``GnssSpec.north_sign``).
    """

    def __init__(self, latitude_deg, longitude_deg, north_sign=1):
        if not (math.isfinite(latitude_deg) and math.isfinite(longitude_deg)) or abs(latitude_deg) >= 89:
            raise ValueError("Reference fix must be finite and away from the poles")
        if north_sign not in (1, -1):
            raise ValueError("north_sign must be 1 or -1")
        self.lat0, self.lon0 = float(latitude_deg), float(longitude_deg)
        self.north_sign = int(north_sign)
        self._east = math.radians(1.0) * EARTH_RADIUS_M * math.cos(math.radians(self.lat0))
        self._north = math.radians(1.0) * EARTH_RADIUS_M * self.north_sign

    def to_local(self, latitude_deg, longitude_deg):
        return (
            (longitude_deg - self.lon0) * self._east,
            (latitude_deg - self.lat0) * self._north,
        )

    def to_geodetic(self, x_m, y_m):
        return self.lat0 + y_m / self._north, self.lon0 + x_m / self._east


@dataclass(frozen=True)
class Guidance:
    index: int  # index of the point being driven towards
    command: Command  # that point's command (what is coming)
    previous: Command  # command of the point just passed (a turn in progress)
    target_xy_m: tuple  # that point in local metres
    distance_m: float  # straight-line distance to it
    following: tuple  # (command, distance between that point and the next), when one follows
    goal_distance_m: float  # straight-line distance to the goal
    finished: bool  # True once the goal point has been reached


class RouteProgress:
    """Which route point the car is heading for, from its own position estimate.

    A point counts as reached when the car is within ``reach_m`` of it, or when the car
    has passed it: beyond the point along the incoming leg, and within
    ``pass_corridor_m`` of it. Progress only moves forwards. The goal is reached only
    within ``goal_reach_m``.
    """

    def __init__(self, route, *, reach_m=4.0, pass_corridor_m=12.0, goal_reach_m=3.0, north_sign=1):
        if not isinstance(route, SatNavRoute):
            raise TypeError("route must be a SatNavRoute")
        for name, value in (("reach_m", reach_m), ("pass_corridor_m", pass_corridor_m), ("goal_reach_m", goal_reach_m)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        self.route = route
        self.plane = LocalTangentPlane(route.points[0].latitude_deg, route.points[0].longitude_deg, north_sign)
        self.xy = tuple(self.plane.to_local(p.latitude_deg, p.longitude_deg) for p in route.points)
        self.reach_m, self.pass_corridor_m, self.goal_reach_m = reach_m, pass_corridor_m, goal_reach_m
        self.index = 1
        self.finished = False

    def _reached(self, i, x, y):
        tx, ty = self.xy[i]
        distance = math.hypot(x - tx, y - ty)
        if i == len(self.xy) - 1:
            return distance <= self.goal_reach_m
        if distance <= self.reach_m:
            return True
        px, py = self.xy[i - 1]
        lx, ly = tx - px, ty - py
        beyond = (x - tx) * lx + (y - ty) * ly > 0
        return beyond and distance <= self.pass_corridor_m

    def update(self, x_m, y_m):
        if not (math.isfinite(x_m) and math.isfinite(y_m)):
            raise ValueError("Position must be finite")
        while not self.finished and self._reached(self.index, x_m, y_m):
            if self.index == len(self.xy) - 1:
                self.finished = True
            else:
                self.index += 1
        i = self.index
        tx, ty = self.xy[i]
        following = ()
        if i + 1 < len(self.xy):
            following = (self.route.points[i + 1].command, math.dist(self.xy[i], self.xy[i + 1]))
        gx, gy = self.xy[-1]
        return Guidance(
            index=i,
            command=self.route.points[i].command,
            previous=self.route.points[i - 1].command,
            target_xy_m=(tx, ty),
            distance_m=math.hypot(x_m - tx, y_m - ty),
            following=following,
            goal_distance_m=math.hypot(x_m - gx, y_m - gy),
            finished=self.finished,
        )
