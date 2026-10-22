"""Sat-nav route: evaluator-side generation from a mission, car-side progress from GPS."""

import math

import numpy as np
import pytest

from autonomy.paths import FIXTURES_DIR
from evaluation.satnav_route import carla_geolocation, junction_traversals, satnav_from_mission
from mapless.satnav import Command, LocalTangentPlane, RouteProgress, SatNavRoute
from planning.lane_graph import LaneTopology
from sim_harness.baseline import mission_route

GEO = carla_geolocation(49.0, 8.0)


@pytest.fixture(scope="module")
def mission():
    return mission_route()


@pytest.fixture(scope="module")
def topology():
    return LaneTopology.load(FIXTURES_DIR / "Town10HD.lane_graph.json")


@pytest.fixture(scope="module")
def route(mission, topology):
    return satnav_from_mission(mission, topology, GEO, name="baseline")


def car_xy(route, x, y):
    """What the car computes from a fix at mission point (x, y): GPS -> its own plane."""
    plane = LocalTangentPlane(route.points[0].latitude_deg, route.points[0].longitude_deg)
    return plane.to_local(*GEO(x, -y))


def test_route_has_one_entry_and_exit_per_junction(route, mission, topology):
    traversals = junction_traversals(mission, topology)
    commands = [p.command for p in route.points]
    assert [c for c in commands if c.is_junction] == [t[2] for t in traversals]
    assert commands[0] is Command.LANEFOLLOW and commands[-1] is Command.LANEFOLLOW
    # Every junction command is followed by its exit point.
    for i, c in enumerate(commands):
        if c.is_junction:
            assert commands[i + 1] is Command.LANEFOLLOW
    s = route.provenance["route_s_m"]
    assert s == sorted(s) and s[0] == 0 and s[-1] == pytest.approx(route.provenance["route_length_m"], abs=1e-3)
    assert max(np.diff(s)) <= 50.0 + 1e-6


def test_tangent_plane_matches_carlas_projection(mission):
    lat0, lon0 = GEO(0.0, 0.0)
    plane = LocalTangentPlane(lat0, lon0)
    for x, y in ((100.0, 0.0), (0.0, 100.0), (-300.0, 250.0), (800.0, -600.0)):
        east, north = plane.to_local(*GEO(x, y))
        # CARLA y is right, so north is -y.
        assert (east, north) == pytest.approx((x, -y), abs=0.01 * max(1.0, math.hypot(x, y) / 100))
        assert plane.to_geodetic(east, north) == pytest.approx(GEO(x, y), abs=1e-9)


def test_progress_walks_the_route_in_order_and_flags_turns(route, mission):
    progress = RouteProgress(route)
    seen = []
    for x, y in mission.points_xy_m:
        g = progress.update(*car_xy(route, x, y))
        if not seen or seen[-1] != (g.index, g.command, g.previous):
            seen.append((g.index, g.command, g.previous))
    assert progress.finished
    assert [i for i, _, _ in seen] == list(range(1, len(route.points)))
    # Inside each junction the guidance says the turn is in progress.
    turns = [(c, p) for _, c, p in seen if p.is_junction]
    assert turns and all(c is Command.LANEFOLLOW for c, _ in turns)


def test_progress_survives_gnss_noise_and_never_goes_backwards(route, mission):
    rng = np.random.default_rng(1)
    bias = rng.normal(0, 1.0, 2)
    progress = RouteProgress(route)
    last = 1
    for x, y in mission.points_xy_m:
        cx, cy = car_xy(route, x, y)
        g = progress.update(cx + bias[0] + rng.normal(0, 0.5), cy + bias[1] + rng.normal(0, 0.5))
        assert g.index >= last
        last = g.index
    # The goal needs 3 m; with ~1 m of bias the last mission point is close enough.
    assert last == len(route.points) - 1


def test_route_file_round_trip_and_validation(route, tmp_path):
    path = tmp_path / "r.json"
    route.save(path)
    again = SatNavRoute.load(path)
    assert again.points == route.points and again.sha256 == route.sha256
    document = route.as_dict()
    document["points"][-1]["command"] = "LEFT"
    with pytest.raises(ValueError, match="goal"):
        SatNavRoute.from_dict(document)
    document = route.as_dict()
    document["points"][0]["heading"] = 0
    with pytest.raises(ValueError, match="exactly"):
        SatNavRoute.from_dict(document)
    document = route.as_dict()
    document["points"].insert(1, dict(document["points"][0]))
    with pytest.raises(ValueError, match="0.5 m apart"):
        SatNavRoute.from_dict(document)
