"""The turn planner's swept check: what "verified" free space means (review of 6773e24).

Within the stopping envelope a pose is verified only when observed road spans the body's
width; beyond it sparse support (far LiDAR rings) still counts, as tentative support.
"""

import numpy as np

from mapless.ego_motion import Pose2D
from mapless.junction import JunctionConfig, _swept_check
from mapless.terrain import TerrainGrid, VehicleBody

BODY = VehicleBody()
CONFIG = JunctionConfig()


def grid_with_road(x_to_m, y_from_m, y_to_m, hole=None):
    """ROAD evidence over x in [-2, x_to], y in [y_from, y_to], optionally minus ``hole``
    (x0, x1, y0, y1), everything seen at t=0."""
    grid = TerrainGrid()
    grid.time_s = 0.0
    xs = np.arange(-2.0, x_to_m, grid.config.cell_m / 2)
    ys = np.arange(y_from_m, y_to_m, grid.config.cell_m / 2)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    if hole is not None:
        x0, x1, y0, y1 = hole
        xy = xy[~((xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1))]
    mark_road(grid, xy)
    return grid


def mark_road(grid, xy):
    from mapless.terrain import PointClass  # noqa: PLC0415

    ij, inside = grid.cells(xy)
    i, j = ij[inside, 0], ij[inside, 1]
    grid.road[i, j] = 5.0
    grid.seen[i, j] = 0.0
    grid.seen_by_class[PointClass.ROAD][i, j] = 0.0


def straight(length_m, step_m=0.5):
    x = np.arange(0.0, length_m + 1e-9, step_m)
    return np.column_stack((x, np.zeros_like(x)))


def check(grid, points):
    return _swept_check(grid, Pose2D(0.0, 0.0, 0.0), points, BODY, CONFIG, 10.0, None)


def test_fully_observed_straight_is_verified_to_its_end():
    feasible, verified, _ = check(grid_with_road(40.0, -2.0, 2.0), straight(30.0))
    assert feasible and verified >= 29.5


def test_unknown_half_of_the_footprint_is_not_verified_near_the_car():
    # Road on the left half only (y > 0): 40% of the cells can still be ROAD, but the
    # right-hand columns of the body hold none, so nothing within the envelope is verified.
    feasible, verified, _ = check(grid_with_road(40.0, 0.0, 2.0), straight(30.0))
    assert feasible
    assert verified <= CONFIG.gap_m + 1e-6


def test_unknown_centre_strip_stops_verification_at_the_strip():
    # A strip the car's width wide is unknown from 4 m to 12 m along the path: 70% of the
    # footprint's cells are still ROAD, but the centre columns hold none.
    feasible, verified, _ = check(grid_with_road(40.0, -2.0, 2.0, hole=(4.0, 12.0, -0.4, 0.4)), straight(30.0))
    assert feasible
    assert verified < 7.0


def test_far_sparse_support_still_counts_beyond_the_envelope():
    # Beyond near_cover_m only a 0.5 m strip of ROAD every metre (rings) along the lane: the
    # share and gap rules apply as before, so a far path is tentative support, not blind.
    grid = grid_with_road(CONFIG.near_cover_m + 0.5, -2.0, 2.0)
    for x in np.arange(CONFIG.near_cover_m + 1.0, 30.0, 1.0):
        xs = np.arange(x, x + 0.5, grid.config.cell_m / 2)
        ys = np.arange(-2.0, 2.0, grid.config.cell_m / 2)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        mark_road(grid, np.column_stack((xx.ravel(), yy.ravel())))
    feasible, verified, _ = check(grid, straight(30.0))
    assert feasible
    assert verified > 20.0
