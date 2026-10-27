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


def test_the_whole_body_side_is_tested_against_the_lane_not_its_corners():
    """Full review of 73ae759: only the four corners were tested, so a body side over the
    exit kerb in the first metres of the exit lane (corners still inside on the bend) passed."""
    site = l_site()
    start, end = site.segment_bounds()
    path, arcs = site.path, site.arcs
    s = end + 2.0
    hd = heading_at(path, arcs, s)
    on_line = np.array([np.interp(s, arcs, path[:, 0]), np.interp(s, arcs, path[:, 1])])
    right = np.array([math.sin(hd), -math.cos(hd)])
    inside = truth_sample(site, on_line + 0.6 * right, hd, BODY)
    over = truth_sample(site, on_line + 1.15 * right, hd, BODY)  # the body side 0.45 m past the kerb
    assert inside["corner_excess_m"] <= 0.0 < over["corner_excess_m"]
    # On the arc, the middle of the inner side lies further in than its corners.
    centre = np.array([40.0, -8.0])
    pose = centre + 9.2 * np.array([math.sin(math.pi / 4), math.cos(math.pi / 4)])
    sample = truth_sample(site, pose, -math.pi / 4, BODY)
    us = np.linspace(BODY.rear_m, BODY.front_m, 40)
    c, sn = math.cos(-math.pi / 4), math.sin(-math.pi / 4)
    side = np.column_stack((pose[0] + c * us + sn * BODY.half_width_m, pose[1] + sn * us - c * BODY.half_width_m))
    from mapless.obstacles import project_onto_polyline  # noqa: PLC0415

    ss, lat, _ = project_onto_polyline(side, path)
    assert sample["corner_excess_m"] >= float((np.abs(lat) - site.half_width_at(ss)).max()) - 0.02


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
    pitch: float = 0.0


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


def test_score_turn_takes_the_log_offset_into_account():
    # The harness logs s_m from the route start, 15 m past the path's start: the gates must
    # shift the junction bounds by the same offset (review of 36ba0ca).
    site = l_site()
    path, arcs = site.path, site.arcs
    rows = []
    for s in np.arange(1.0, arcs[-1] - site.extend_m - 1.0, 0.25):
        xy = np.array([np.interp(s, arcs, path[:, 0]), np.interp(s, arcs, path[:, 1])])
        truth = truth_sample(site, xy, heading_at(path, arcs, s), BODY) | dict(speed_mps=2.0)
        truth["s_m"] -= 15.0
        rows.append(dict(truth=truth))
    gates = {name: (ok, values) for name, ok, values in score_turn(rows, site, 50, end="finished", s_offset_m=15.0)}
    unshifted = {name: (ok, values) for name, ok, values in score_turn(rows, site, 50, end="finished")}
    assert gates["exit_lane_keeping"][0] is True and gates["approach_lane_keeping"][0] is not False
    assert gates["exit_lane_keeping"][1]["ticks"] > unshifted["exit_lane_keeping"][1]["ticks"]


def test_cut_in_refuses_an_opposing_left_lane():
    from carla_adapter.mapless_scenarios import CutInCar, ScenarioParams

    class Owner:
        def try_spawn(self, bp, t):
            return object()

    class Library:
        def find(self, name):
            return name

    lane = chain([(x, 0.0) for x in range(0, 50)], yaw_deg=0.0, lane_id=-1)
    opposing = chain([(x, -3.5) for x in range(0, 50)], yaw_deg=180.0, lane_id=1)
    for w, o in zip(lane, opposing):
        w._left = o
    FakeWaypoint.get_left_lane = lambda self: getattr(self, "_left", None)
    scenario = CutInCar(FakeCarla, ScenarioParams("cut_in"), lambda loc: (loc.x, -loc.y))
    with pytest.raises(RuntimeError, match="same-direction"):
        scenario.spawn(Owner(), Library(), lane)


def test_cut_in_starts_once_the_ego_closes_to_the_cut_gap():
    """First live stack cut-in run (26 Oct): the car held 3 m/s, faster than the ego, and
    waited for the ego to draw level, so it never cut in. It now approaches at
    approach_mps and cuts in when the ego is cut_gap_m behind it, then speeds up."""
    from types import SimpleNamespace

    from carla_adapter.mapless_scenarios import CutInCar, ScenarioParams

    class Actor:
        def __init__(self, t):
            self.t, self.controls = t, []

        def get_transform(self):
            return self.t

        def get_velocity(self):
            return SimpleNamespace(x=1.0, y=0.0)

        def apply_control(self, control):
            self.controls.append(control)

    class Owner:
        def try_spawn(self, bp, t):
            return Actor(t)

    class Library:
        def find(self, name):
            return name

    carla = SimpleNamespace(LaneType=FakeLaneType, VehicleControl=lambda **kw: SimpleNamespace(**kw))
    lane = chain([(x, 0.0) for x in range(0, 80)], yaw_deg=0.0, lane_id=-1)
    left = chain([(x, 3.5) for x in range(0, 80)], yaw_deg=0.0, lane_id=-2)
    for w, o in zip(lane, left):
        w._left = o
    FakeWaypoint.get_left_lane = lambda self: getattr(self, "_left", None)
    try:
        params = ScenarioParams("cut_in")
        scenario = CutInCar(carla, params, lambda loc: (loc.x, loc.y))
        actor = scenario.spawn(Owner(), Library(), lane)  # at 20 m in the left lane
        scenario.step(0, dict(rear_axle_xy=(0.0, 0.0)))
        assert scenario.cut_start is None  # 20 m ahead: approaching at approach_mps
        assert actor.controls[-1].throttle > 0.0  # 1 m/s measured, approach target 1 m/s
        scenario.step(1, dict(rear_axle_xy=(20.0 - params.cut_gap_m, 0.0)))
        assert scenario.cut_start == 1
        assert actor.controls[-1].throttle > 0.5  # speeding up towards lead_mps
    finally:
        FakeWaypoint.get_left_lane = lambda self: None


def test_turn_site_search_refuses_a_kerb_mismatch_and_a_bent_approach():
    # Review of 1931070: the car places the exit lane keep_right (measured on the approach)
    # left of the exit kerb, so a shoulder lane turning into a plain lane would put it in the
    # opposing lane. The harness refuses such sites, and bent approaches (skewed exit heading).
    carla_map = FakeMap()
    for wp in carla_map.exit:
        wp.lane_width = 6.0  # exit kerb 3.0 m from the lane centre vs 1.75 m on the approach
    sites, census = find_turn_sites(FakeCarla, carla_map, "RIGHT", 40.0, 30.0)
    assert sites == [] and census["kerb_mismatch"] == 1
    sites, census = find_turn_sites(FakeCarla, carla_map, "RIGHT", 40.0, 30.0, max_kerb_mismatch_m=2.0)
    assert len(sites) == 1 and sites[0]["kerb_mismatch_m"] == pytest.approx(1.25)
    bent = FakeMap()
    for k, wp in enumerate(bent.approach):
        wp.transform.rotation.yaw = -8.0 * (k / len(bent.approach))  # 8 degrees of bend over the approach
    sites, census = find_turn_sites(FakeCarla, bent, "RIGHT", 40.0, 30.0)
    assert sites == [] and census["approach_bent"] == 1
    # A site on a hill (Town03's first left-turn site tilted the car past the planar
    # adapter's 5 degrees mid-box and the run aborted): any waypoint steeper than 3 degrees.
    sloped = FakeMap()
    sloped.exit[5].transform.rotation.pitch = 4.0
    sites, census = find_turn_sites(FakeCarla, sloped, "RIGHT", 40.0, 30.0)
    assert sites == [] and census["sloped"] == 1


def test_lane_heading_near_the_car_ignores_a_bend_far_ahead():
    # Review of 1931070: the corridor's quadratic slope at the car was tilted 2.3 degrees by
    # the kerb running out into the junction mouth 10 m ahead, and that skew went straight
    # into the exit heading. The near-evidence line is not tilted.
    from mapless.corridor import Corridor
    from mapless.ego_motion import Pose2D

    xs = np.arange(-10.0, 31.0, 1.0)
    ys = np.where(xs > 15.0, 0.02 * (xs - 15.0) ** 2, 0.0)  # straight, then bending away
    coefficients = tuple(np.polyfit(xs, ys, 2))
    corridor = Corridor(
        time_s=0.0,
        pose=Pose2D(0.0, 0.0, 0.0),
        centre_body=np.column_stack((xs, np.polyval(coefficients, xs))),
        centre_odometry=np.column_stack((xs, np.polyval(coefficients, xs))),
        valid_length_m=30.0,
        coefficients=coefficients,
        station_x_m=xs,
        reference_y_m=np.zeros_like(xs),
        right_edge_m=np.full_like(xs, -1.75),
        left_edge_m=np.full_like(xs, np.nan),
        blocked=np.zeros_like(xs, dtype=bool),
        evidence_body=np.column_stack((xs, ys)),
    )
    assert abs(math.degrees(corridor.heading_at(0.0))) > 1.0  # the whole-line fit is tilted at the car
    assert abs(math.degrees(corridor.heading_near())) < 0.05  # the near evidence is straight
