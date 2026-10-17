"""RV-06: lane monitor sign and look-ahead geometry, and replay against the map."""

import math

import cv2
import numpy as np
import pytest

from evaluation.lane_monitor_replay import replay, report
from geometry.camera import CameraIntrinsics, ego_point_to_pixel
from geometry.transforms import ego_to_world_matrix, inverse_rigid
from planning.corridor import RouteGeometry
from sim_harness.baseline import mission_route
from visualisation.lane_monitor import (
    REFERENCE_X_M,
    compare_to_map,
    map_lookahead_offset,
    monitor_from_mask,
    monitor_lane,
)

CALIBRATION = CameraIntrinsics.from_horizontal_fov(640, 360, math.pi / 2).calibration(
    ((0, 0, 1, 1.5 + 1.38641), (-1, 0, 0, 0), (0, -1, 0, 2.0), (0, 0, 0, 1))
)


def draw(mask, ego_points):
    pixels = [ego_point_to_pixel((x, y, 0.0), CALIBRATION) for x, y in ego_points]
    pixels = [p for p in pixels if p is not None]
    if len(pixels) > 1:
        cv2.polylines(mask, [np.round(np.array(pixels)).astype(np.int32)], False, 1, 4)


def ego_lane_mask(centre_y_m, curvature=0.0, width_m=3.5):
    mask = np.zeros((360, 640), np.uint8)
    for side in (-1, 1):
        xs = np.linspace(4, 40, 80)
        draw(mask, [(x, centre_y_m + 0.5 * curvature * x * x + side * width_m / 2) for x in xs])
    return mask


@pytest.mark.parametrize(
    "centre,curvature", [(-0.5, 0.0), (0.5, 0.0), (-0.3, 0.02), (0.3, -0.02), (-0.4, 0.05)]
)
def test_mask_offset_sign_and_value_at_look_ahead(centre, curvature):
    departure, path, ground = monitor_from_mask(ego_lane_mask(centre, curvature), CALIBRATION)
    expected = -(centre + 0.5 * curvature * REFERENCE_X_M**2)
    assert departure.valid and path is not None and len(ground) > 20
    assert departure.signed_offset_m == pytest.approx(expected, abs=0.06)
    assert departure.confidence > 0.5


def test_no_lane_is_invalid_and_never_compared():
    departure, path, _ = monitor_from_mask(np.zeros((360, 640), np.uint8), CALIBRATION)
    assert not departure.valid and path is None
    assert not compare_to_map(departure, 0.4)["usable"]
    assert not monitor_lane([(10, 0), (10, 1)]).valid


def test_map_look_ahead_offset_sign():
    route = RouteGeometry(((0, 0), (50, 0)))
    assert map_lookahead_offset(route, 5, 0.4, 0.0) == pytest.approx(0.4)
    assert map_lookahead_offset(route, 5, -0.4, 0.0) == pytest.approx(-0.4)
    # Heading 0.05 rad left at the axle puts the look-ahead point further left.
    assert map_lookahead_offset(route, 5, 0.0, 0.05) == pytest.approx(
        REFERENCE_X_M * math.sin(0.05), abs=1e-6
    )


class MaskFromRed:
    """Stub segmenter: the replay frames carry the true lane mask in red."""

    def predict(self, rgb):
        return rgb[:, :, 0]


def test_replay_matches_route_through_the_mission_turn():
    route = RouteGeometry(mission_route().points_xy_m)
    rows, frames = [], []
    for frame_id, s in enumerate(np.arange(20.0, 95.0, 1.5)):
        for lateral in (-0.4, 0.4):
            x, y, yaw = route.at(s, lateral)
            world_to_ego = inverse_rigid(ego_to_world_matrix(x, y, yaw))
            mask = np.zeros((360, 640), np.uint8)
            for side in (-1.75, 1.75):
                boundary = []
                for ahead in np.arange(0.0, 45.0, 0.5):
                    bx, by, _ = route.at(s + ahead, side)
                    local = world_to_ego @ np.array([bx, by, 0.0, 1.0])
                    if local[0] > 3.5:
                        boundary.append((local[0], local[1]))
                draw(mask, boundary)
            rgb = np.zeros((360, 640, 3), np.uint8)
            rgb[:, :, 0] = mask
            rows.append(
                dict(
                    frame_id=len(rows),
                    sim_time_s=0.05 * len(rows),
                    progress_m=float(s),
                    ego=dict(x_m=x, y_m=y, yaw_rad=yaw),
                )
            )
            frames.append(rgb)
    records = replay(rows, frames, MaskFromRed(), CALIBRATION, route, every=1)
    result = report(records, rows=len(rows), frames_read=len(frames))
    summary = result["summary"]
    assert result["aligned"] and not result["control_coupling"]
    assert summary["straight"]["usable_frames"] > 20
    assert summary["straight"]["sign_agreement"] == 1.0
    assert summary["straight"]["median_abs_difference_m"] < 0.1
    assert summary["curve"]["usable_frames"] > 4
    assert summary["curve"]["sign_agreement"] == 1.0
