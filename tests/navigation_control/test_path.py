"""Unit tests for reference-path validation, projection and interpolation."""

import math

import pytest

from navigation_control import Waypoint
from navigation_control.path import ReferencePath


def test_projection_and_arc_length_interpolation():
    path = ReferencePath([Waypoint(0, 0), Waypoint(10, 0), Waypoint(10, 10)])

    projection = path.project(13, 4)

    assert projection.point == Waypoint(10, 4)
    assert projection.progress_m == pytest.approx(14)
    assert projection.distance_m == pytest.approx(3)
    assert path.point_at(12.5) == Waypoint(10, 2.5)
    assert path.point_at(-1) == Waypoint(0, 0)
    assert path.point_at(30) == Waypoint(10, 10)


def test_consecutive_duplicates_are_removed_and_input_is_copied():
    points = [Waypoint(0, 0), Waypoint(0, 0), Waypoint(10, 0)]

    path = ReferencePath(iter(points))
    points.clear()

    assert path.length_m == 10
    assert len(path.points) == 2


@pytest.mark.parametrize(
    "points",
    [
        [],
        [Waypoint(0, 0)],
        [Waypoint(1, 1)] * 3,
        [Waypoint(0, 0), Waypoint(1, 0), Waypoint(0, 0)],
    ],
)
def test_invalid_or_closed_paths_are_rejected(points):
    with pytest.raises(ValueError):
        ReferencePath(points)


def test_non_waypoint_entries_are_rejected():
    with pytest.raises(TypeError, match="Waypoint"):
        ReferencePath([(0, 0), (10, 0)])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_projection_or_interpolation_arguments_are_rejected(value):
    path = ReferencePath([Waypoint(0, 0), Waypoint(10, 0)])

    with pytest.raises(ValueError):
        path.point_at(value)
    with pytest.raises(ValueError):
        path.project(value, 0)
