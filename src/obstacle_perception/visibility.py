"""Fresh depth visibility probes, separate from detection/track expiry."""

import math

import numpy as np

from geometry.transforms import ego_to_world_matrix, inverse_rigid
from obstacle_perception.box_depth import GROUND_CLEARANCE_M

CLEAR = "clear"
OUT_OF_VIEW = "out_of_view"
OCCLUDED = "occluded"
INVALID = "invalid"


class RouteClearanceMemory:
    """Route corridor cells that fresh depth has shown clear earlier this episode.

    A 90-degree camera cannot see the inside edge of a tight junction turn while
    the car is in it, although it saw that edge on the approach. Under the
    baseline's stationary-obstacle model, such a cell stays usable until
    ``max_age_s``. Only the latest observation of a cell counts: CLEAR stores
    it, OCCLUDED or INVALID erases it, so an older clear record can never
    reappear once the cell goes out of view. Only a later CLEAR restores it.
    Memory never clears a formerly blocked obstacle area; that still needs a
    current observation. This is an explicit stationary-scenario assumption.
    """

    def __init__(self, max_age_s=15.0):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("Memory age must be finite and positive")
        self.max_age_s = float(max_age_s)
        self._seen = {}

    def reset(self):
        self._seen.clear()

    def mark(self, key, time_s):
        self._seen[key] = float(time_s)

    def forget(self, key):
        self._seen.pop(key, None)

    def recent(self, key, time_s):
        seen = self._seen.get(key)
        return seen is not None and 0 <= time_s - seen <= self.max_age_s


class DepthVisibility:
    """Check finite, in-image depth behind specified world-space probes.

    This is a sampled conservative visibility test, not a complete occupancy
    grid. Out-of-view, occluded, invalid and missing pixels all mean unknown.
    """

    def __init__(self, frame, clearance_m=0.5):
        self.frame = frame
        self.clearance_m = clearance_m
        state = frame.ego.state
        self.world_to_optical = inverse_rigid(frame.calibration.camera_to_ego) @ inverse_rigid(
            ego_to_world_matrix(state.x_m, state.y_m, state.yaw_rad, frame.ego.z_m)
        )

    def probe_status(self, point_world_m):
        """CLEAR, OUT_OF_VIEW, OCCLUDED (nearer surface) or INVALID depth."""
        return self.probe_statuses([point_world_m])[0]

    def probe_statuses(self, points_world_m):
        """Vectorised :meth:`probe_status` for an (N, 3) array of world points.

        Each probe projects to its nearest pixel; the 3x3 patch around it needs
        8 finite depths (else INVALID) and no surface nearer than the probe
        plus ``clearance_m`` (else OCCLUDED). Off-image or behind-camera probes
        are OUT_OF_VIEW.
        """
        points = np.asarray(points_world_m, dtype=float).reshape(-1, 3)
        status = np.full(len(points), OUT_OF_VIEW, dtype=object)
        depth = self.frame.depth_m
        if depth is None:
            status[:] = INVALID
            return status
        if not len(points):
            return status
        h, w = depth.shape
        cal = self.frame.calibration
        optical = (self.world_to_optical @ np.c_[points, np.ones(len(points))].T).T
        z = optical[:, 2]
        front = z > 0
        safe = np.where(front, z, 1.0)
        # np.rint and Python round() both round half to even.
        u = np.rint(cal.cx_px + cal.fx_px * optical[:, 0] / safe)
        v = np.rint(cal.cy_px + cal.fy_px * optical[:, 1] / safe)
        inside = front & (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
        if not inside.any():
            return status
        ui, vi = u[inside].astype(int), v[inside].astype(int)
        patch = np.stack(
            [depth[vi + dv, ui + du] for dv in (-1, 0, 1) for du in (-1, 0, 1)], axis=1
        )
        finite = np.isfinite(patch)
        nearest = np.where(finite, patch, np.inf).min(axis=1)
        result = np.where(
            finite.sum(axis=1) < 8,
            INVALID,
            np.where(nearest < z[inside] + self.clearance_m, OCCLUDED, CLEAR),
        )
        status[inside] = result
        return status

    def clear_points(self, points_world_m):
        if self.frame.depth_m is None or not len(points_world_m):
            return False
        return bool((self.probe_statuses(points_world_m) == CLEAR).all())

    def footprint_clear(self, footprint):
        corners = np.asarray(footprint.corners_world_m)
        points = []
        # Cover the previously occupied area at two heights; never look at
        # scenario removal events or at a simulator actor list.
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            for ratio in np.linspace(0, 1, max(2, math.ceil(np.linalg.norm(b - a) / 0.4) + 1)):
                xy = a + ratio * (b - a)
                points.extend(
                    (float(xy[0]), float(xy[1]), self.frame.ego.z_m + z) for z in (0.6, 1.0)
                )
        centre = corners.mean(axis=0)
        points.append((float(centre[0]), float(centre[1]), self.frame.ego.z_m + 0.8))
        return self.clear_points(points)

    def corridor_area_clear(self, footprint, route, ego_s, half_width):
        """Observe the formerly blocked part of the route, not roadside corners.

        Conservative inferred boxes can extend outside the camera after a turn.
        Those roadside parts cannot block this fixed swept corridor.
        """
        interval = route.footprint_interval(
            footprint.corners_world_m, half_width_m=half_width, minimum_s=ego_s
        )
        if interval is None:
            return True
        s_values = np.linspace(
            interval[0], interval[1], max(2, math.ceil((interval[1] - interval[0]) / 0.5) + 1)
        )
        points = self._route_points(
            route, s_values, np.linspace(-half_width, half_width, 7), (0.6, 1.0), extend=False
        )
        return self.clear_points(points.reshape(-1, 3))

    def box_may_overlap_route(self, box, route, ego_s, front_offset, half_width, horizon_m):
        """Conservative image overlap for an unknown-depth box.

        A roadside truncated box outside every projected route interval cannot
        stop the entire mission. Include heights from ground to tall vehicle and
        a pixel guard; ambiguous overlap remains unknown and forces stopping.
        """
        cal = self.frame.calibration
        # A rejected mixed-surface median is still useful for EXCLUSION when
        # every pixel has valid depth and the entire measured cloud is outside
        # the swept corridor. Do not discard unknown pixels to make this proof.
        if self.frame.depth_m is not None:
            x0, y0, x1, y1 = map(int, (box.x_min, box.y_min, box.x_max, box.y_max))
            depths = self.frame.depth_m[y0:y1, x0:x1]
            if depths.size >= 9 and np.isfinite(depths).all():
                vv, uu = np.indices(depths.shape)
                optical = np.stack(
                    (
                        (uu + x0 - cal.cx_px) * depths / cal.fx_px,
                        (vv + y0 - cal.cy_px) * depths / cal.fy_px,
                        depths,
                    ),
                    axis=-1,
                ).reshape(-1, 3)
                optical_to_world = np.linalg.inv(self.world_to_optical)
                cloud = optical @ optical_to_world[:3, :3].T + optical_to_world[:3, 3]
                # Measured road surface inside the box is not obstacle surface
                # (every pixel is still required to be valid above). A box with
                # no raised surface at all proves nothing and falls through.
                cloud = cloud[cloud[:, 2] >= self.frame.ego.z_m + GROUND_CLEARANCE_M]
                if len(cloud) >= 9:
                    lo, hi = cloud[:, :2].min(axis=0) - 0.3, cloud[:, :2].max(axis=0) + 0.3
                    polygon = (
                        (lo[0], lo[1]),
                        (hi[0], lo[1]),
                        (hi[0], hi[1]),
                        (lo[0], hi[1]),
                    )
                if len(cloud) >= 9 and (
                    route.footprint_interval(
                        polygon,
                        half_width_m=half_width,
                        minimum_s=ego_s,
                        maximum_s=ego_s + front_offset + horizon_m,
                    )
                    is None
                ):
                    return False
        distances = np.arange(0.5, horizon_m + 0.5, 0.5)
        points = self._route_points(
            route,
            ego_s + front_offset + distances,
            (-half_width - 0.3, half_width + 0.3),
            (-0.3, 2.5),
            extend=False,
        ).reshape(len(distances), 4, 3)
        homogeneous = np.concatenate((points, np.ones((*points.shape[:2], 1))), axis=-1)
        optical = homogeneous @ self.world_to_optical.T
        z = optical[..., 2]
        valid = z > 0.05
        safe = np.where(valid, z, 1.0)
        u = cal.cx_px + cal.fx_px * optical[..., 0] / safe
        v = cal.cy_px + cal.fy_px * optical[..., 1] / safe
        rows = valid.any(axis=1)
        if not rows.any():
            return False
        lo_u = np.where(valid, u, np.inf).min(axis=1) - 3
        hi_u = np.where(valid, u, -np.inf).max(axis=1) + 3
        lo_v = np.where(valid, v, np.inf).min(axis=1) - 3
        hi_v = np.where(valid, v, -np.inf).max(axis=1) + 3
        overlap = (
            rows
            & (box.x_max >= lo_u)
            & (box.x_min <= hi_u)
            & (box.y_max >= lo_v)
            & (box.y_min <= hi_v)
        )
        return bool(overlap.any())

    def _route_points(self, route, s_values, laterals, heights, *, extend=True):
        """World probes, shape (len(s), len(laterals), len(heights), 3).

        Matches ``route.at``: s is clipped to the route; with ``extend`` the
        part beyond the rear-axle goal continues along the final heading, so
        visibility covers the front bumper. The controller route is unchanged.
        """
        s_values = np.asarray(s_values, dtype=float)
        laterals = np.asarray(laterals, dtype=float)
        heights = np.asarray(heights, dtype=float)
        if not (np.isfinite(s_values).all() and np.isfinite(laterals).all()):
            raise ValueError("Route coordinates must be finite")
        clipped = np.clip(s_values, 0, route.length_m)
        index = np.minimum(
            np.searchsorted(route.arc, clipped, side="right") - 1, len(route.lengths) - 1
        )
        along = clipped - route.arc[index]
        if extend:
            along = along + np.maximum(0.0, s_values - route.length_m)
        base = route.points[index] + along[:, None] * route.tangents[index]
        xy = base[:, None, :] + laterals[None, :, None] * route.normals[index][:, None, :]
        shape = (len(s_values), len(laterals), len(heights))
        out = np.empty((*shape, 3))
        out[..., :2] = xy[:, :, None, :]
        out[..., 2] = self.frame.ego.z_m + heights[None, None, :]
        return out

    def route_horizon(
        self, route, ego_s, front_offset, half_width, horizon_m=35.0, *, memory=None, time_s=None
    ):
        """First unverified arc distance from bumper (near hood blind zone excluded).

        The 3 m near-camera blind zone is explicit. Previously seen obstacles in
        it still block resume through footprint_clear; this does not certify all
        possible unseen objects inside that zone.

        Without ``memory`` every probe must be visible and clear now. With a
        :class:`RouteClearanceMemory`, a probe that is merely outside the camera
        frustum may use a recent CLEAR observation of the same route cell. An
        occluded or invalid in-view probe always ends the horizon, and every
        in-view clear cell along the whole horizon refreshes the memory.
        """
        distances = np.arange(3.0, horizon_m + 0.1, 1.0)
        laterals = (-half_width, 0.0, half_width)
        heights = (0.6, 1.0)
        if memory is None:
            points = self._route_points(route, ego_s + front_offset + distances, laterals, heights)
            statuses = self.probe_statuses(points.reshape(-1, 3)).reshape(len(distances), -1)
            blocked = ~(statuses == CLEAR).all(axis=1)
            if not blocked.any():
                return float(distances[-1]) if len(distances) else 0.0
            first = int(np.argmax(blocked))
            return float(distances[first - 1]) if first else 0.0
        if time_s is None or not math.isfinite(time_s):
            raise ValueError("Memory-backed visibility needs the frame simulation time")
        # Cells sit on a fixed 1 m route grid so different frames share keys.
        cells = [math.floor(ego_s + front_offset + distance) for distance in distances]
        points = self._route_points(route, cells, laterals, heights)
        statuses = self.probe_statuses(points.reshape(-1, 3)).reshape(
            len(cells), len(laterals), len(heights)
        )
        covered, blocked = 0.0, False
        for row, (distance, cell) in enumerate(zip(distances, cells)):
            known = True
            for index in range(len(laterals)):
                for column, z in enumerate(heights):
                    key = (cell, index, z)
                    status = statuses[row, index, column]
                    if status == CLEAR:
                        memory.mark(key, time_s)
                    elif status in (OCCLUDED, INVALID):
                        # A newer non-clear view supersedes any older clearance.
                        memory.forget(key)
                        known = False
                    elif not memory.recent(key, time_s):
                        known = False
            if not known:
                blocked = True
            if not blocked:
                covered = float(distance)
        return covered
