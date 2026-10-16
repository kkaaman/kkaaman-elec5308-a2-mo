"""Route-corridor geometry tests for BL-D-04."""

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