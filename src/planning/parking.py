"""Deterministic forward bay approaches with certified Bezier curvature bounds."""

import math
from dataclasses import dataclass
from itertools import product

import numpy as np

from autonomy.parking_contracts import ParkingConfig, ParkingGoal, ParkingVehicle, angle_error
from navigation_control import VehicleState, Waypoint
from navigation_control.path import ReferencePath
from planning.parking_geometry import bay_polygon, contains, footprint, polygon, swept_rectangle


@dataclass(frozen=True)
class ParkingSegment:
    start_s_m: float
    end_s_m: float
    curvature_bound: float
    sweep: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ParkingPlan:
    waypoints: tuple[Waypoint, ...]
    headings_rad: tuple[float, ...]
    segments: tuple[ParkingSegment, ...]
    drivable_area: tuple[tuple[float, float], ...]

    @property
    def length_m(self):
        return self.segments[-1].end_s_m


def _split(p):
    levels = [p]
    while len(levels[-1]) > 1:
        levels.append((levels[-1][:-1] + levels[-1][1:]) / 2)
    return np.array([v[0] for v in levels]), np.array([v[-1] for v in levels[::-1]])


def _curvature_bound(p):
    velocity = (len(p) - 1) * np.diff(p, axis=0)
    acceleration = (len(velocity) - 1) * np.diff(velocity, axis=0)
    chord = p[-1] - p[0]
    length = np.linalg.norm(chord)
    if length <= 1e-10:
        return math.inf
    lower_speed = min(velocity @ (chord / length))
    if lower_speed <= 1e-10:
        return math.inf
    numerator = np.abs(
        velocity[:, None, 0] * acceleration[None, :, 1]
        - velocity[:, None, 1] * acceleration[None, :, 0]
    ).max()
    return float(numerator / lower_speed**3)


def _leaves(p, limit, step, depth=0):
    """Convex derivative hulls prove the bound over each whole interval."""
    bound = _curvature_bound(p)
    hull_length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    if bound <= limit and hull_length <= step:
        return [(p, bound)]
    if depth >= 14:
        raise ValueError("No certified curvature-feasible subdivision")
    # An endpoint's exact curvature already exceeding the limit cannot be repaired.
    velocity = 5 * np.diff(p, axis=0)
    acceleration = 4 * np.diff(velocity, axis=0)
    for v, a in ((velocity[0], acceleration[0]), (velocity[-1], acceleration[-1])):
        norm = float(np.linalg.norm(v))
        if norm < 1e-12 or abs(v[0] * a[1] - v[1] * a[0]) / norm**3 > limit:
            raise ValueError("Turning radius exceeds the controller steering limit")
    left, right = _split(p)
    return _leaves(left, limit, step, depth + 1) + _leaves(right, limit, step, depth + 1)


def plan_forward_parking(start, goal, vehicle, drivable_area, config=None):
    """Plan in an explicitly supplied convex manoeuvring area.

    Searches bounded quintic approaches followed by a straight alignment tail.
    All derivative controls must advance along the approach chord. This excludes
    loops and self crossings, but deliberately does not claim arbitrary forward
    reachability. Rejected candidates never modify an installed route.
    """
    config = ParkingConfig() if config is None else config
    if not isinstance(start, VehicleState) or not isinstance(goal, ParkingGoal):
        raise TypeError("Expected VehicleState and ParkingGoal")
    if not isinstance(vehicle, ParkingVehicle) or not isinstance(config, ParkingConfig):
        raise TypeError("Expected ParkingVehicle and ParkingConfig")
    area = polygon(drivable_area)
    target = VehicleState(goal.x_m, goal.y_m, goal.yaw_rad, 0)
    if start.speed_mps > config.max_speed_mps:
        raise ValueError("Enter P1 at or below its configured approach speed")
    if not contains(bay_polygon(goal.bay), footprint(target, vehicle, goal.clearance_m)):
        raise ValueError("Goal body and clearance do not fit inside the bay")
    if not contains(area, bay_polygon(goal.bay)):
        raise ValueError("Bay must be inside the supplied manoeuvring area")
    if not contains(area, footprint(start, vehicle, goal.clearance_m)):
        raise ValueError("Initial body is outside the manoeuvring area")
    p0, p5 = np.array([start.x_m, start.y_m]), np.array([goal.x_m, goal.y_m])
    u = np.array([math.cos(start.yaw_rad), math.sin(start.yaw_rad)])
    v = np.array([math.cos(goal.yaw_rad), math.sin(goal.yaw_rad)])
    delta = p5 - p0
    distance = float(np.linalg.norm(delta))
    if distance < max(0.6, 2 * goal.position_tolerance_m):
        raise ValueError("P1 requires a nontrivial forward approach")
    straight = (
        abs(angle_error(start.yaw_rad, goal.yaw_rad)) < 1e-6
        and float(delta @ u) > 0
        and abs(delta[0] * u[1] - delta[1] * u[0]) < 1e-6
    )
    candidates = []
    if straight:
        candidates.append([np.linspace(p0, p5, 6)])
    else:
        pregoal = p5 - config.straight_finish_m * v
        chord = pregoal - p0
        span = float(np.linalg.norm(chord))
        if span < 0.5 or min(chord @ u, chord @ v) <= 0:
            raise ValueError("No supported monotone forward approach to this goal")
        for a, b in product((0.2, 0.28, 0.36, 0.14), repeat=2):
            a, b = a * span, b * span
            controls = np.array([p0, p0 + a*u, p0 + 2*a*u,
                                 pregoal - 2*b*v, pregoal - b*v, pregoal])
            if np.any(np.diff(controls, axis=0) @ chord <= 1e-8):
                continue
            candidates.append([controls, np.linspace(pregoal, p5, 6)])
    padding = goal.clearance_m + config.tracking_padding(vehicle)
    for curves in candidates:
        try:
            leaves = [leaf for p in curves for leaf in _leaves(
                p, vehicle.curvature_limit, config.sample_step_m
            )]
            points = [Waypoint(*leaves[0][0][0])]
            headings = [start.yaw_rad]
            segments = []
            progress = 0.0
            for p, bound in leaves:
                sweep = swept_rectangle(p, vehicle, padding)
                if not contains(area, sweep):
                    raise ValueError("Full vehicle sweep leaves the manoeuvring area")
                length = float(np.linalg.norm(p[-1] - p[0]))
                segments.append(ParkingSegment(progress, progress + length, bound, sweep))
                progress += length
                points.append(Waypoint(*p[-1]))
                derivative = p[-1] - p[-2]
                headings.append(math.atan2(derivative[1], derivative[0]))
            ReferencePath(points)
            return ParkingPlan(tuple(points), tuple(headings), tuple(segments), area)
        except ValueError:
            continue
    raise ValueError("No curvature- and full-body-feasible forward candidate in this area")
