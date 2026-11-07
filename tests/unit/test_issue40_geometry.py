"""Geometry support is descriptive and cannot silently authorize RESUME."""

import numpy as np

from obstacle_perception.lidar_obstacles import CLASS_OBSTACLE, CLASS_ROAD
from scripts.screen_lidar_geometry import coverage


def supported_points():
    return np.array([[x, y, 0.] for x in np.arange(.25, 6.5, .5)
                     for y in (-1., 0., 1.) for _ in range(2)])


def test_full_road_support_and_occupancy_are_independent():
    points = np.vstack((supported_points(), [[5.2, 0., .7]]))
    classes = np.array([CLASS_ROAD]*(len(points)-1)+[CLASS_OBSTACLE])
    report = coverage(points, classes, 0.)
    assert report["complete_road_support"] and report["road_supported_cells"] == 39
    assert report["occupied_cells"] == 1
    assert not report["positive_clearance_authorised"]


def test_empty_missing_and_last_full_bin_remain_unknown():
    empty = coverage(np.empty((0, 3)), np.empty(0, int), 0.)
    assert empty["unknown_cells"] == 39 and not empty["complete_road_support"]
    points = supported_points()
    points = points[points[:, 0] < 6.15]
    report = coverage(points, np.full(len(points), CLASS_ROAD), 0.)
    assert report["unknown_cells"] == 3
    assert not report["positive_clearance_authorised"]
