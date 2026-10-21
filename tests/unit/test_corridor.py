"""Route-corridor geometry tests for BL-D-04."""

import math
from dataclasses import dataclass

import pytest

from planning.corridor import RouteGeometry


@dataclass(frozen=True)
class FootprintStub:
    corners_world_m: tuple[tuple[float, float], ...]


def footprint(
    *,
    corners=((20, -1), (24, -1), (24, 1), (20, 1)),
):
    return FootprintStub(corners_world_m=corners)


@pytest.mark.parametrize(
    ("corners", "expected"),
    [
        (((20, -1), (24, -1), (24, 1), (20, 1)), 16),
        (((20, 3), (24, 3), (24, 5), (20, 5)), None),
        (((-10, -1), (-6, -1), (-6, 1), (-10, 1)), None),
        (((20, -4), (24, -4), (24, 4), (20, 4)), 16),
    ],
)
def test_corridor_front_offset_once(corners, expected):
    route = RouteGeometry(((0, 0), (40, 0)))

    assert route.gap(
        footprint(corners=corners),
        ego_s_m=0,
        front_bumper_from_rear_axle_m=4,
        half_width_m=1.2,
    ) == expected


def test_corridor_follows_bend_instead_of_infinite_start_tangent():
    route = RouteGeometry(((0, 0), (10, 0), (10, 20)))

    obj = footprint(
        corners=((9, 8), (11, 8), (11, 10), (9, 10)),
    )

    assert route.gap(
        obj,
        ego_s_m=0,
        front_bumper_from_rear_axle_m=4,
        half_width_m=1,
    ) == pytest.approx(14)

    assert route.project(9, 8).lateral_m == pytest.approx(1)
    assert route.project(2, -1).lateral_m == pytest.approx(-1)

    assert (
        route.footprint_interval(
            ((15, -1), (17, -1), (17, 1), (15, 1)),
            half_width_m=1,
        )
        is None
    )


@pytest.mark.parametrize("turn", [1, -1])
@pytest.mark.parametrize("heading", [0, math.pi / 4, math.pi])
def test_outer_turn_sweep_detects_obstacle_for_both_turn_directions(turn, heading):
    def rotate(x, y):
        return (
            x * math.cos(heading) - y * math.sin(heading),
            x * math.sin(heading) + y * math.cos(heading),
        )

    route = RouteGeometry(tuple(rotate(x, y) for x, y in ((0, 0), (10, 0), (10, turn * 20))))
    obj = footprint(
        corners=tuple(
            rotate(x, turn * y)
            for x, y in ((10.2, -0.8), (10.8, -0.8), (10.8, -0.2), (10.2, -0.2))
        )
    )

    assert route.gap(
        obj,
        ego_s_m=5,
        front_bumper_from_rear_axle_m=4,
        half_width_m=1,
    ) == pytest.approx(1)


def test_front_overlap_beyond_route_endpoint_is_not_discarded():
    route = RouteGeometry(((0, 0), (10, 0)))
    obj = footprint(corners=((11, -0.5), (12, -0.5), (12, 0.5), (11, 0.5)))

    assert route.gap(
        obj,
        ego_s_m=8,
        front_bumper_from_rear_axle_m=4,
        half_width_m=1,
    ) == pytest.approx(-1)


@pytest.mark.parametrize("reverse_corners", [False, True])
@pytest.mark.parametrize(
    ("corners", "expected"),
    [
        # The polygon's edge crosses the round join with all vertices outside.
        (((10.1, -1.1), (11.1, -0.1), (11.2, -0.2), (10.2, -1.2)), (10, 10)),
        # Exact tangency in the outside quadrant counts as an intersection.
        (((10.6, -0.8), (11.4, -1.4), (12, -0.6), (11.2, 0)), (10, 10)),
        # Inside the join's bounding square, but outside its circular sweep.
        (((10.8, -0.9), (10.9, -0.9), (10.9, -0.8), (10.8, -0.8)), None),
    ],
)
def test_turn_join_uses_footprint_edges_and_round_boundary(corners, expected, reverse_corners):
    route = RouteGeometry(((0, 0), (10, 0), (10, 20)))
    if reverse_corners:
        corners = tuple(reversed(corners))
    assert route.footprint_interval(corners, half_width_m=1) == expected


def test_obstacle_enclosing_turn_join_remains_detected():
    route = RouteGeometry(((0, 0), (10, 0), (10, 20)))
    assert route.footprint_interval(
        ((8, -2), (12, -2), (12, 2), (8, 2)), half_width_m=1
    ) == pytest.approx((8, 12))


@pytest.mark.parametrize("bounds", [{"minimum_s": 10.1}, {"maximum_s": 9.9}])
def test_turn_join_respects_requested_arc_interval(bounds):
    route = RouteGeometry(((0, 0), (10, 0), (10, 20)))
    assert route.footprint_interval(
        ((10.2, -0.8), (10.8, -0.8), (10.8, -0.2), (10.2, -0.2)),
        half_width_m=1,
        **bounds,
    ) is None


@pytest.mark.parametrize("front_offset", [2, 4])
def test_endpoint_extension_preserves_obstacle_coordinate(front_offset):
    route = RouteGeometry(((0, 0), (10, 0)))
    corners = ((11, -0.5), (12, -0.5), (12, 0.5), (11, 0.5))
    assert route.footprint_interval(corners, half_width_m=1) is None
    assert route.footprint_interval(
        corners, half_width_m=1, end_extension_m=front_offset
    ) == pytest.approx((11, 12))
    assert route.gap(
        footprint(corners=corners), 8, front_offset, 1
    ) == pytest.approx(11 - 8 - front_offset)


def test_explicit_arc_limit_still_clips_endpoint_extension():
    route = RouteGeometry(((0, 0), (10, 0)))
    assert route.footprint_interval(
        ((11, -0.5), (12, -0.5), (12, 0.5), (11, 0.5)),
        half_width_m=1,
        end_extension_m=4,
        maximum_s=10,
    ) is None


@pytest.mark.parametrize(
    "corners",
    [
        ((14.1, -0.5), (15, -0.5), (15, 0.5), (14.1, 0.5)),
        ((11, 2), (12, 2), (12, 3), (11, 3)),
    ],
)
def test_endpoint_extension_rejects_obstacles_outside_final_vehicle_envelope(corners):
    route = RouteGeometry(((0, 0), (10, 0)))
    assert route.gap(footprint(corners=corners), 8, 4, 1) is None


def test_endpoint_extension_follows_final_tangent_after_a_bend():
    route = RouteGeometry(((0, 0), (10, 0), (10, 10)))
    obj = footprint(corners=((9.5, 11), (10.5, 11), (10.5, 12), (9.5, 12)))
    assert route.gap(obj, 18, 4, 1) == pytest.approx(-1)


@pytest.mark.parametrize("extension", [-1, float("nan"), float("inf")])
def test_invalid_endpoint_extension_is_rejected(extension):
    route = RouteGeometry(((0, 0), (10, 0)))
    with pytest.raises(ValueError, match="End extension"):
        route.footprint_interval(
            footprint().corners_world_m, half_width_m=1, end_extension_m=extension
        )
