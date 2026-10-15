"""Exit classification uses complete legal branches, including initially straight turns."""

import math

import pytest

from planning.junction_planner import choose_exit, classify_exit, junction_exits
from planning.lane_graph import LaneSegment, LaneTopology
from planning.route_manager import stitch_route


def segment(road, points, successors=(), junction=None):
    return LaneSegment(
        road, 0, -1, 0, 10, tuple(points), tuple(f"{x}:0:-1" for x in successors), junction
    )


def topology(right=True, duplicate=False):
    lanes = [
        segment(1, [(-5, 0, 0), (0, 0, 0)], (2, 3) + ((4,) if right else ())),
        segment(2, [(0.1, 0, 0), (2, 0, 0), (3, 1, 0), (3, 3, 0)], (5,), 1),
        segment(3, [(0.1, 0.01, 0), (4, 0.01, 0)], (6,), 1),
        segment(5, [(3, 3.1, 0), (3, 8, 0)]),
        segment(6, [(4.1, 0.01, 0), (8, 0.01, 0)]),
    ]
    if right:
        lanes.extend(
            [
                segment(4, [(0.1, -0.01, 0), (2, -0.01, 0), (3, -1, 0), (3, -3, 0)], (7,), 1),
                segment(7, [(3, -3.1, 0), (3, -8, 0)]),
            ]
        )
    if duplicate:
        lanes[0] = segment(1, [(-5, 0, 0), (0, 0, 0)], (2, 3, 8))
        lanes.extend(
            [
                segment(8, [(0.1, -0.02, 0), (2, -0.02, 0), (4, 1, 0), (4, 3, 0)], (9,), 1),
                segment(9, [(4, 3.1, 0), (4, 8, 0)]),
            ]
        )
    return LaneTopology("fixture", "0" * 64, 1, 3, tuple(lanes))


@pytest.mark.parametrize("right", [False, True])
def test_t_and_four_way_complete_exits(right):
    t = topology(right)
    assert {x.maneuver for x in junction_exits(t, "1:0:-1")} == (
        {"left", "straight", "right"} if right else {"left", "straight"}
    )
    selected = choose_exit(t, "1:0:-1", "left")
    assert selected.segment_ids == ("1:0:-1", "2:0:-1", "5:0:-1")
    points = stitch_route(t, selected.segment_ids)
    assert (3.0, 1.0, 0.0) in points and points[-1] == (3.0, 8.0, 0.0)


def test_missing_and_ambiguous_maneuvers_fail():
    with pytest.raises(ValueError):
        choose_exit(topology(False), "1:0:-1", "right")
    with pytest.raises(ValueError):
        choose_exit(topology(False, True), "1:0:-1", "left")


def test_no_illegal_chord_or_repeated_route():
    with pytest.raises(ValueError):
        stitch_route(topology(), ("1:0:-1", "5:0:-1"))
    with pytest.raises(ValueError):
        stitch_route(topology(), ("1:0:-1", "1:0:-1"))


@pytest.mark.parametrize("angle", [30, 40, 140, 180])
def test_ambiguous_or_uturn_rejected(angle):
    with pytest.raises(ValueError):
        classify_exit(math.radians(angle))


def test_junction_cycle_and_dead_end_rejected():
    lanes = (segment(1, [(-2, 0, 0), (0, 0, 0)], (2,)), segment(2, [(0.1, 0, 0), (1, 0, 0)], (), 1))
    t = LaneTopology("dead", "0" * 64, 1, 3, lanes)
    with pytest.raises(ValueError, match="no legal exit"):
        junction_exits(t, "1:0:-1")
    with pytest.raises(ValueError):
        junction_exits(topology(), "1:0:-1", max_segments=1)


def test_boundary_tangent_jump_is_not_smoothed_into_a_chord():
    lanes = (segment(1, [(-2, 0, 0), (0, 0, 0)], (2,)), segment(2, [(0.1, 0, 0), (0.1, 2, 0)]))
    t = LaneTopology("jump", "0" * 64, 1, 3, lanes)
    with pytest.raises(ValueError, match="Discontinuous"):
        stitch_route(t, ("1:0:-1", "2:0:-1"))
