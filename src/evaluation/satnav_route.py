"""Evaluator side: turn a mapped mission into the sat-nav route the mapless car receives.

This module may read the map, because the evaluator plays the role of a phone sat-nav:
it knows the road network and hands the car only sparse GPS points and commands. The
car never imports this module.

Route points, in order of route arc length:
- start (LANEFOLLOW);
- for each junction traversal: the junction entry, carrying the manoeuvre (LEFT, RIGHT or
  STRAIGHT), then the junction exit (LANEFOLLOW);
- LANEFOLLOW fill points, so that no gap is longer than ``max_spacing_m``;
- goal (LANEFOLLOW).
"""

import math

import numpy as np

from mapless.satnav import EARTH_RADIUS_M, Command, RoutePoint, SatNavRoute
from planning.corridor import RouteGeometry

GENERATOR_VERSION = "satnav-from-mission-1"
_MANOEUVRE = {"left": Command.LEFT, "right": Command.RIGHT, "straight": Command.STRAIGHT}


def carla_geolocation(reference_lat_deg, reference_lon_deg):
    """Pure copy of CARLA's ``GeoLocation::Transform`` (Mercator, scaled at the reference).

    Returns ``geolocate(x, y)`` taking CARLA x/y (left-handed, +y right) and returning
    (latitude, longitude) in degrees. The live route script checks it against
    ``map.transform_to_geolocation`` before trusting it.
    """
    scale = math.cos(math.radians(reference_lat_deg))
    mx0 = scale * math.radians(reference_lon_deg) * EARTH_RADIUS_M
    my0 = scale * EARTH_RADIUS_M * math.log(math.tan(math.radians(90.0 + reference_lat_deg) / 2))

    def geolocate(x, y):
        mx, my = mx0 + x, my0 - y
        lon = math.degrees(mx / (EARTH_RADIUS_M * scale))
        lat = math.degrees(2 * math.atan(math.exp(my / (EARTH_RADIUS_M * scale)))) - 90.0
        return lat, lon

    return geolocate


def junction_traversals(mission, topology):
    """(entry xy, exit xy, manoeuvre) for each junction the mission crosses, in order."""
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    ids = list(mission.segment_ids)
    if any(key not in by_id for key in ids):
        raise ValueError("Mission uses a lane section missing from the topology")
    runs, current = [], []
    for key in ids:
        if by_id[key].junction_id is not None:
            current.append(key)
        elif current:
            runs.append(current)
            current = []
    if current:
        raise ValueError("Mission must not end inside a junction")
    if len(runs) != len(mission.manoeuvres):
        raise ValueError(
            f"Mission crosses {len(runs)} junctions but declares {len(mission.manoeuvres)} manoeuvres"
        )
    result = []
    for run, manoeuvre in zip(runs, mission.manoeuvres):
        if manoeuvre not in _MANOEUVRE:
            raise ValueError(f"Unsupported manoeuvre {manoeuvre!r}")
        entry = by_id[run[0]].points_xyz_m[0][:2]
        exit_ = by_id[run[-1]].points_xyz_m[-1][:2]
        result.append((entry, exit_, _MANOEUVRE[manoeuvre]))
    return result


def satnav_from_mission(mission, topology, geolocate, *, name, max_spacing_m=50.0):
    """Sparse route for ``mission``. ``geolocate(x, y)`` takes CARLA x/y."""
    if not math.isfinite(max_spacing_m) or max_spacing_m < 5:
        raise ValueError("max_spacing_m must be at least 5 m")
    route = RouteGeometry(mission.points_xy_m)
    marks = [(0.0, Command.LANEFOLLOW), (route.length_m, Command.LANEFOLLOW)]
    for entry, exit_, command in junction_traversals(mission, topology):
        entry_s = route.project(*entry).s_m
        exit_s = route.project(*exit_, minimum_s=entry_s).s_m
        if not 0 < entry_s < exit_s < route.length_m:
            raise ValueError("Junction entry and exit must lie inside the mission")
        marks += [(entry_s, command), (exit_s, Command.LANEFOLLOW)]
    marks.sort(key=lambda item: item[0])
    filled = [marks[0]]
    for s, command in marks[1:]:
        start = filled[-1][0]
        gap = s - start
        # Fill along the road, never inside a junction (between an entry and its exit).
        if gap > max_spacing_m and not filled[-1][1].is_junction:
            pieces = math.ceil(gap / max_spacing_m)
            filled += [(start + gap * k / pieces, Command.LANEFOLLOW) for k in range(1, pieces)]
        filled.append((s, command))
    arc = route.arc
    points = []
    for s, command in filled:
        x = float(np.interp(s, arc, route.points[:, 0]))
        y = float(np.interp(s, arc, route.points[:, 1]))
        lat, lon = geolocate(x, -y)  # core y is left; CARLA y is right
        points.append(RoutePoint(lat, lon, command))
    return SatNavRoute(
        name,
        tuple(points),
        provenance=dict(
            generator=GENERATOR_VERSION,
            mission_sha256=mission.sha256,
            topology_sha256=topology.sha256,
            map_name=mission.map_name,
            opendrive_sha256=mission.opendrive_sha256,
            max_spacing_m=max_spacing_m,
            route_length_m=route.length_m,
            route_s_m=[round(s, 3) for s, _ in filled],
            note="Evaluator output. The car receives only the points and commands.",
        ),
    )
