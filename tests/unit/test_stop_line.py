"""The painted stop line from the front camera (G4 M07, 2 Nov): the detector on rendered
ground, and the tracker's lock (a line stays put as the car moves; an artefact does not)."""

from types import SimpleNamespace

import numpy as np

from mapless.ego_motion import Pose2D
from mapless.stop_line import StopLineTracker, detect

# The live front camera: 1280 x 720, 90 degrees, 2.0 m up, 2.886 m ahead of the rear axle.
CAL = SimpleNamespace(fx_px=640.0, fy_px=640.0, cx_px=640.0, cy_px=360.0, camera_to_ego=((0.0, 0.0, 1.0, 2.886), (-1.0, 0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 2.0), (0.0, 0.0, 0.0, 1.0)))


def render(paint):
    """Grey road, sky above the horizon; ``paint(x, y)`` -> True where the ground is white."""
    v, u = np.mgrid[0:720, 0:1280].astype(float)
    rgb = np.full((720, 1280, 3), 200, dtype=np.uint8)
    below = v > 361
    # A pixel's ray from the camera hits the ground (ego z 0) at:
    zc = 2.0 * 640.0 / (v[below] - 360.0)  # depth along the camera's forward axis
    x = zc + 2.886
    y = -(u[below] - 640.0) * zc / 640.0
    grey = np.where(paint(x, y), 235, 110).astype(np.uint8)
    rgb[below] = grey[:, None]
    return rgb


def lane_marks(x, y):
    return (np.abs(np.abs(y) - 1.75) < 0.07) & ((x % 6.0) < 3.0)  # dashed lane lines either side


def test_a_stop_line_across_the_lane_is_found_at_its_near_edge_and_a_zebra_is_not():
    line = render(lambda x, y: lane_marks(x, y) | ((x >= 15.0) & (x <= 15.4)))
    d = detect(line, CAL)
    assert d is not None and abs(d - 15.0) <= 0.25
    assert detect(render(lane_marks), CAL) is None  # lane markings alone
    zebra = render(lambda x, y: lane_marks(x, y) | ((x >= 12.0) & (x <= 15.0) & ((np.floor(y / 0.5) % 2) == 0)))
    assert detect(zebra, CAL) is None  # bars along the lane cover half its width
    arrow = render(lambda x, y: lane_marks(x, y) | ((x >= 12.0) & (x <= 12.3) & (np.abs(y) < 0.4)))
    assert detect(arrow, CAL) is None  # a third of the lane


def test_the_tracker_locks_a_line_that_stays_put_and_not_one_that_moves_with_the_car():
    tracker, artefact = StopLineTracker(), StopLineTracker()
    for k in range(12):  # the car advances 0.2 m a frame towards a line 25 m on
        car = 0.2 * k
        pose = Pose2D(car, 0.0, 0.0)
        tracker.update(render(lambda x, y, c=car: (x + c >= 25.0) & (x + c <= 25.4)), CAL, pose, 0.1 * k)
        artefact.update(render(lambda x, y: (x >= 10.0) & (x <= 10.4)), CAL, pose, 0.1 * k)  # always 10 m ahead
    assert tracker.locked is not None and abs(tracker.locked[0] - 25.0) <= 0.3
    assert abs(tracker.distance_m(Pose2D(20.0, 0.0, 0.0)) - 5.0) <= 0.3  # kept once out of view
    assert artefact.locked is None
    tracker.reset()
    assert tracker.locked is None and tracker.distance_m(Pose2D(0.0, 0.0, 0.0)) is None
