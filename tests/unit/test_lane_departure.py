"""RV-06: lane-departure monitor sign convention, validity rules and HUD hook."""

import math

import numpy as np
import pytest

from autonomy.interfaces import CameraCalibration
from geometry.camera import CameraIntrinsics, ego_point_to_pixel
from navigation_control.path import ReferencePath
from navigation_control.types import Waypoint
from visualisation.hud import departure_hud_lines, draw_departure_overlay
from visualisation.lane_departure import (
    DepartureReport,
    LaneDepartureMonitor,
    LaneEstimate,
    MonitorConfig,
    lane_estimate_from_centerline,
)

WIDTH, HEIGHT = 640, 360


def calibration():
    intrinsics = CameraIntrinsics.from_horizontal_fov(WIDTH, HEIGHT, math.radians(90))
    # Camera 1.5 m ahead of the rear axle, 2.0 m up, looking forward (A2 rig).
    return intrinsics.calibration(
        ((0, 0, 1, 1.5), (-1, 0, 0, 0), (0, -1, 0, 2.0), (0, 0, 0, 1))
    )


def signed_cte(path, x_m, y_m):
    """Same convention as carla_adapter.s0_runner.signed_cross_track (no CARLA import)."""
    projection = path.project(x_m, y_m)
    for index, end_s in enumerate(path.arc_lengths[1:]):
        if projection.progress_m <= end_s or index == len(path.points) - 2:
            a, b = path.points[index : index + 2]
            cross = (b.x_m - a.x_m) * (y_m - projection.point.y_m) - (b.y_m - a.y_m) * (
                x_m - projection.point.x_m
            )
            return math.copysign(projection.distance_m, cross if cross else 1.0)
    raise AssertionError


def curve_centre_ego(radius_m, ego_offset_left_m, forward_m):
    """Ego-frame lane-centre points of a left-hand bend; ego sits ``ego_offset_left_m`` left."""
    points = []
    for x in forward_m:
        # Circle of radius R to the left, tangent to +X at the origin: y = R - sqrt(R^2 - x^2).
        y_centre = radius_m - math.sqrt(radius_m**2 - x**2)
        points.append((x, y_centre - ego_offset_left_m))
    return points


def centerline_pixels(ego_points, cal):
    pixels = []
    for x, y in ego_points:
        pixel = ego_point_to_pixel((x, y, 0.0), cal)
        assert pixel is not None
        pixels.append(pixel)
    # extract_ego_path returns far-to-near order; the monitor must not care.
    return np.array(pixels[::-1])


@pytest.mark.parametrize("offset_left_m", [0.6, -0.6, 0.0])
def test_image_offset_sign_is_positive_left_on_straight_lane(offset_left_m):
    cal = calibration()
    forward = np.linspace(3.0, 15.0, 25)
    pixels = centerline_pixels([(x, -offset_left_m) for x in forward], cal)
    estimate = lane_estimate_from_centerline(pixels, cal, frame_id=3, sim_time_s=1.0)
    assert estimate is not None
    assert estimate.lateral_offset_m == pytest.approx(offset_left_m, abs=0.02)
    assert estimate.heading_rad == pytest.approx(0.0, abs=0.01)
    assert estimate.confidence > 0.9


def test_heading_sign_is_positive_when_nose_points_left_of_lane():
    cal = calibration()
    # Lane tangent rotated clockwise by 5 degrees in the ego frame means the
    # ego nose points 5 degrees to the left of the lane direction.
    angle = math.radians(-5.0)
    forward = np.linspace(3.0, 15.0, 25)
    pixels = centerline_pixels([(x, math.tan(angle) * x) for x in forward], cal)
    estimate = lane_estimate_from_centerline(pixels, cal, frame_id=0, sim_time_s=0.0)
    assert estimate.heading_rad == pytest.approx(math.radians(5.0), abs=0.01)


def test_recorded_style_curve_agrees_with_map_cte_sign_convention():
    """Synthetic left-hand bend: image offset and map CTE share the positive-left sign."""
    cal = calibration()
    radius = 40.0
    # Map route: the same left-hand bend sampled in world coordinates with the
    # ego at the origin heading +X; rear axle displaced 0.4 m to the left.
    route = ReferencePath(
        [
            Waypoint(x, radius - math.sqrt(radius**2 - x**2))
            for x in np.linspace(-5.0, 25.0, 61)
        ]
    )
    monitor = LaneDepartureMonitor(MonitorConfig(disagreement_m=0.25))
    for sign in (1.0, -1.0):
        offset_left = 0.4 * sign
        map_cte = signed_cte(route, 0.0, offset_left)
        assert math.copysign(1.0, map_cte) == sign
        forward = np.linspace(3.0, 10.0, 20)
        pixels = centerline_pixels(curve_centre_ego(radius, offset_left, forward), cal)
        estimate = lane_estimate_from_centerline(
            pixels, cal, frame_id=7, sim_time_s=2.0, max_forward_m=10.0
        )
        report = monitor.update(estimate, map_cte, now_sim_time_s=2.05)
        assert report.valid, report.reasons
        assert math.copysign(1.0, report.image_offset_m) == sign
        assert abs(report.difference_m) < 0.15
        assert report.status == "agree"


def test_monitor_flags_stale_low_confidence_and_outside_range():
    monitor = LaneDepartureMonitor(MonitorConfig(max_age_s=0.5, min_confidence=0.3))
    fresh = LaneEstimate(0.2, 0.0, 0.9, 10, 5.0)
    assert monitor.update(fresh, 0.1, now_sim_time_s=5.2).valid
    stale = monitor.update(fresh, 0.1, now_sim_time_s=5.8)
    assert not stale.valid and stale.reasons == ("stale",) and stale.difference_m is None
    future = monitor.update(fresh, 0.1, now_sim_time_s=4.9)
    assert "stale" in future.reasons
    weak = monitor.update(LaneEstimate(0.2, 0.0, 0.1, 11, 5.0), 0.1, now_sim_time_s=5.1)
    assert weak.reasons == ("low_confidence",)
    far = monitor.update(LaneEstimate(3.0, 0.0, 0.9, 12, 5.0), 0.1, now_sim_time_s=5.1)
    assert far.reasons == ("outside_range",)
    yawed = monitor.update(LaneEstimate(0.0, math.radians(45), 0.9, 13, 5.0), 0.1, now_sim_time_s=5.1)
    assert yawed.reasons == ("outside_range",)
    missing = monitor.update(None, 0.1, now_sim_time_s=5.1, frame_id=14)
    assert missing.reasons == ("no_estimate",) and missing.frame_id == 14
    no_map = monitor.update(fresh, None, now_sim_time_s=5.1)
    assert no_map.reasons == ("no_map_cte",)
    assert monitor.update_count == 8 and monitor.valid_count == 1


def test_disagreement_is_reported_not_acted_on():
    monitor = LaneDepartureMonitor(MonitorConfig(disagreement_m=0.5))
    report = monitor.update(LaneEstimate(0.9, 0.0, 0.8, 1, 1.0), -0.2, now_sim_time_s=1.0)
    assert report.valid and report.status == "disagree"
    assert report.difference_m == pytest.approx(1.1)
    record = report.as_record()
    assert record["lane_difference_m"] == pytest.approx(1.1)
    assert record["lane_monitor_status"] == "disagree"
    # The monitor exposes no route, waypoint or command interface.
    for name in ("set_path", "waypoints", "route", "command", "build_local_route"):
        assert not hasattr(monitor, name)
        assert not hasattr(report, name)


def test_estimate_returns_none_without_support():
    cal = calibration()
    assert lane_estimate_from_centerline(np.zeros((0, 2)), cal, frame_id=0, sim_time_s=0.0) is None
    # Points above the horizon never hit the ground plane.
    sky = np.array([[320.0, 10.0], [330.0, 12.0], [340.0, 14.0], [350.0, 16.0], [360.0, 18.0], [370.0, 20.0]])
    assert lane_estimate_from_centerline(sky, cal, frame_id=0, sim_time_s=0.0) is None
    with pytest.raises(ValueError):
        lane_estimate_from_centerline(np.array([[1.0, 2.0, 3.0]]), cal, frame_id=0, sim_time_s=0.0)
    with pytest.raises(ValueError):
        lane_estimate_from_centerline(np.array([[np.nan, 2.0]]), cal, frame_id=0, sim_time_s=0.0)
    with pytest.raises(TypeError):
        lane_estimate_from_centerline(np.zeros((6, 2)), object(), frame_id=0, sim_time_s=0.0)


def test_contract_validation():
    with pytest.raises(ValueError):
        LaneEstimate(math.nan, 0.0, 0.5, 0, 0.0)
    with pytest.raises(ValueError):
        LaneEstimate(0.0, 0.0, 1.5, 0, 0.0)
    with pytest.raises(ValueError):
        LaneEstimate(0.0, 0.0, 0.5, -1, 0.0)
    with pytest.raises(ValueError):
        MonitorConfig(max_age_s=-1.0)
    with pytest.raises(ValueError):
        MonitorConfig(min_confidence=2.0)
    with pytest.raises(TypeError):
        LaneDepartureMonitor().update(object(), 0.0, now_sim_time_s=0.0)
    with pytest.raises(ValueError):
        LaneDepartureMonitor().update(None, math.inf, now_sim_time_s=0.0)
    assert isinstance(calibration(), CameraCalibration)


def test_hud_lines_and_overlay_do_not_mutate_input():
    report = DepartureReport(5, 1.0, 0.3, 0.02, 0.1, 0.2, 0.8, True, "agree")
    lines = departure_hud_lines(report)
    assert lines[0].startswith("lane monitor: AGREE")
    assert any("+0.30 m" in line for line in lines)
    assert any("difference   +0.20 m" in line for line in lines)
    invalid = LaneDepartureMonitor().update(None, None, now_sim_time_s=0.0)
    assert "reasons: no_estimate, no_map_cte" in departure_hud_lines(invalid)
    assert departure_hud_lines(None) == ["lane monitor: no report"]
    image = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    before = image.copy()
    drawn = draw_departure_overlay(image, report)
    assert drawn.shape == image.shape and drawn.dtype == np.uint8
    assert np.array_equal(image, before)
    assert not np.array_equal(drawn, image)
    with pytest.raises(ValueError):
        draw_departure_overlay(np.zeros((4, 4), dtype=np.uint8), report)
    with pytest.raises(ValueError):
        draw_departure_overlay(image, report, alpha=2.0)
