"""G1c harness framework: junction-aware truth and gates, and the turn-site search and
route construction against a fake CARLA map (an L-shaped approach, a connecting lane
through a junction and an exit lane). The live run is the G1c debug pass."""

import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from carla_adapter.mapless_drive import find_turn_sites, make_turn_route
from evaluation.junction_truth import TurnSite, heading_at, score_turn, truth_sample
from mapless.satnav import Command
from mapless.terrain import VehicleBody

BODY = VehicleBody()


def l_site(command="RIGHT"):
    approach = np.column_stack((np.arange(0.0, 41.0), np.zeros(41)))  # along +x to the box edge at x = 40
    sign = -1.0 if command == "RIGHT" else 1.0
    # A quarter arc of radius 8 from (40, 0) heading +x to heading -y (right) / +y (left).
    theta = np.linspace(0.0, math.pi / 2, 14)
    junction = np.column_stack((40.0 + 8.0 * np.sin(theta), sign * (8.0 - 8.0 * np.cos(theta))))
    exit_lane = np.column_stack((np.full(31, 48.0), sign * (8.0 + np.arange(0.0, 31.0))))
    return TurnSite(command, approach, junction, exit_lane, 3.5, 4.5, 3.5)


def test_truth_sample_knows_the_segments_and_the_permitted_area():
    site = l_site()
    start, end = site.segment_bounds()
    assert start == pytest.approx(40.0) and end > start
    on_approach = truth_sample(site, (20.0, 0.1), 0.0, BODY)
    assert on_approach["segment"] == "approach" and on_approach["corner_excess_m"] < 0
    assert on_approach["lane_offset_m"] == pytest.approx(0.1)
    # Half a lane off on the approach puts a corner on the pavement.
    off = truth_sample(site, (20.0, 1.0), 0.0, BODY)
    assert off["corner_excess_m"] > 0
    # In the junction the connecting lane is wider, plus the tolerance.
    inside = truth_sample(site, (44.5, -1.8), -math.pi / 4, BODY)
    assert inside["segment"] == "junction" and inside["corner_excess_m"] < 0.6
    on_exit = truth_sample(site, (48.0, -20.0), -math.pi / 2, BODY)
    assert on_exit["segment"] == "exit" and abs(on_exit["heading_error_rad"]) < 1e-6


def test_score_turn_gates():
    site = l_site()
    rows = []
    path, arcs = site.path, site.arcs
    for s in np.arange(1.0, arcs[-1] - site.extend_m - 1.0, 0.25):
        xy = np.array([np.interp(s, arcs, path[:, 0]), np.interp(s, arcs, path[:, 1])])
        truth = truth_sample(site, xy, heading_at(path, arcs, s), BODY) | dict(speed_mps=2.0)
        rows.append(dict(truth=truth))
    gates = {name: (ok, values) for name, ok, values in score_turn(rows, site, modes_turning=50, end="finished")}
    assert gates["turn_executed"][0] and gates["correct_exit"][0]
    assert gates["junction_footprint"][0], gates["junction_footprint"][1]
    assert gates["exit_lane_keeping"][0] and gates["approach_lane_keeping"][0]
    bad = {name: ok for name, ok, _ in score_turn(rows, site, modes_turning=0, end="collision")}
    assert not bad["turn_executed"] and not bad["correct_exit"]


# -- a fake CARLA map ------------------------------------------------------------------------
class FakeLaneType:
    Driving, Sidewalk = "Driving", "Sidewalk"


class FakeCarla:
    LaneType = FakeLaneType


@dataclass
class Location:
    x: float
    y: float
    z: float = 0.0


@dataclass
class Rotation:
    yaw: float


@dataclass
class Transform:
    location: Location
    rotation: Rotation


@dataclass
class FakeWaypoint:
    transform: Transform
    lane_type: str = "Driving"
    is_junction: bool = False
    lane_width: float = 3.5
    road_id: int = 1
    lane_id: int = -1
    section_id: int = 0
    junction_id: int | None = None
    _next: list = field(default_factory=list)
    _right: object = None

    def next(self, step):
        return list(self._next)

    def get_right_lane(self):
        return self._right

    def get_left_lane(self):
        return None


def chain(points, yaw_deg, **kw):
    wps = [FakeWaypoint(Transform(Location(x, y), Rotation(yaw_deg)), **kw) for x, y in points]
    for a, b in zip(wps, wps[1:]):
        a._next = [b]
    for wp in wps:
        wp._right = FakeWaypoint(Transform(Location(0, 0), Rotation(0)), lane_type="Sidewalk")
    return wps


class FakeMap:
    """Approach along +x (core frame; CARLA y is mirrored), a right turn through a junction, exit along -y (core)."""

    def __init__(self):
        # CARLA frame: core (x, y) = (carla.x, -carla.y); a core heading h is carla yaw -h degrees.
        self.approach = chain([(x, 0.0) for x in range(0, 61)], yaw_deg=0.0, road_id=1)
        theta = np.linspace(0.0, math.pi / 2, 14)[1:]
        junction_pts = [(60.0 + 8.0 * math.sin(t), 8.0 - 8.0 * math.cos(t)) for t in theta]  # carla y grows = core y shrinks: right turn
        self.junction = chain(junction_pts, yaw_deg=45.0, is_junction=True, lane_width=4.5, road_id=7, junction_id=3)
        self.exit = chain([(68.0, 8.0 + k) for k in range(0, 41)], yaw_deg=90.0, road_id=2)
        self.approach[-1]._next = [self.junction[0]]
        self.junction[-1]._next = [self.exit[0]]
        # The straight-on branch: a second connecting lane that does not turn.
        straight = chain([(60.0 + k, 0.0) for k in range(1, 9)], yaw_deg=0.0, is_junction=True, road_id=8, junction_id=3)
        straight_exit = chain([(68.0 + k, 0.0) for k in range(0, 41)], yaw_deg=0.0, road_id=3)
        straight[-1]._next = [straight_exit[0]]
        self.approach[-1]._next.append(straight[0])
        self.name = "FakeTown"

    def get_topology(self):
        return [(self.approach[0], self.approach[-1]), (self.exit[0], self.exit[-1])]

    def transform_to_geolocation(self, location):
        class Fix:
            latitude = -33.0 + location.x * 1e-5
            longitude = 151.0 - location.y * 1e-5

        return Fix()


def test_turn_site_search_and_route_on_a_fake_map():
    carla_map = FakeMap()
    sites, census = find_turn_sites(FakeCarla, carla_map, "RIGHT", approach_m=40.0, exit_m=30.0)
    assert census["sites"] == 1, census
    site = sites[0]
    assert len(site["approach"]) == 41 and site["approach"][-1] is carla_map.approach[-1]
    assert site["junction"][-1] is carla_map.exit[0] and len(site["exit"]) >= 31
    assert abs(site["heading_change_deg"] + 90) < 5 and site["junction_width_m"] == 4.5
    assert site["kerb_source"] == "sidewalk"
    # No LEFT turn exists at this junction.
    left, census_left = find_turn_sites(FakeCarla, carla_map, "LEFT", 40.0, 30.0)
    assert left == [] and census_left["no_matching_exit"] == 1
    route = make_turn_route(carla_map, site, "RIGHT")
    commands = [p.command for p in route.points]
    assert commands.count(Command.RIGHT) == 1
    entry = commands.index(Command.RIGHT)
    assert commands[entry + 1] is Command.LANEFOLLOW and commands[-1] is Command.LANEFOLLOW
    assert all(c is Command.LANEFOLLOW for c in commands[:entry]) and entry >= 1
    # Entry at the lane's last waypoint, exit 5 m into the exit lane (geolocation of x).
    assert route.points[entry].latitude_deg == pytest.approx(-33.0 + 60.0 * 1e-5)
    assert route.points[entry + 1].latitude_deg == pytest.approx(-33.0 + 68.0 * 1e-5)
