"""Fresh depth visibility probes, separate from detection/track expiry."""

import math

import numpy as np

from geometry.transforms import ego_to_world_matrix, inverse_rigid


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

    def clear_points(self, points_world_m):
        depth = self.frame.depth_m
        if depth is None or not points_world_m:
            return False
        h, w = depth.shape
        cal = self.frame.calibration
        for point in points_world_m:
            optical = self.world_to_optical @ np.r_[point, 1.0]
            if optical[2] <= 0:
                return False
            u = int(round(cal.cx_px + cal.fx_px * optical[0] / optical[2]))
            v = int(round(cal.cy_px + cal.fy_px * optical[1] / optical[2]))
            if not (1 <= u < w - 1 and 1 <= v < h - 1):
                return False
            patch = depth[v - 1 : v + 2, u - 1 : u + 2]
            if np.isfinite(patch).sum() < 8:
                return False
            if float(np.nanmin(patch)) < optical[2] + self.clearance_m:
                return False
        return True

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
        points = []
        for s in np.linspace(
            interval[0], interval[1], max(2, math.ceil((interval[1] - interval[0]) / 0.5) + 1)
        ):
            for lateral in np.linspace(-half_width, half_width, 7):
                x, y, _ = route.at(s, lateral)
                points.extend((x, y, self.frame.ego.z_m + z) for z in (0.6, 1.0))
        return self.clear_points(points)

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
                lo, hi = cloud[:, :2].min(axis=0) - 0.3, cloud[:, :2].max(axis=0) + 0.3
                polygon = ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))
                if (
                    route.footprint_interval(
                        polygon,
                        half_width_m=half_width,
                        minimum_s=ego_s,
                        maximum_s=ego_s + front_offset + horizon_m,
                    )
                    is None
                ):
                    return False
        for distance in np.arange(0.5, horizon_m + 0.5, 0.5):
            pixels = []
            for side in (-half_width - 0.3, half_width + 0.3):
                x, y, _ = route.at(ego_s + front_offset + distance, side)
                for z in (-0.3, 2.5):
                    optical = self.world_to_optical @ np.array([x, y, self.frame.ego.z_m + z, 1])
                    if optical[2] > 0.05:
                        pixels.append(
                            (
                                cal.cx_px + cal.fx_px * optical[0] / optical[2],
                                cal.cy_px + cal.fy_px * optical[1] / optical[2],
                            )
                        )
            if pixels:
                lo, hi = np.min(pixels, axis=0) - 3, np.max(pixels, axis=0) + 3
                if (
                    box.x_max >= lo[0]
                    and box.x_min <= hi[0]
                    and box.y_max >= lo[1]
                    and box.y_min <= hi[1]
                ):
                    return True
        return False

    def route_horizon(self, route, ego_s, front_offset, half_width, horizon_m=35.0):
        """First unverified arc distance from bumper (near hood blind zone excluded).

        The 3 m near-camera blind zone is explicit. Previously seen obstacles in
        it still block resume through footprint_clear; this does not certify all
        possible unseen objects inside that zone.
        """
        distances = np.arange(3.0, horizon_m + 0.1, 1.0)
        covered = 0.0
        for distance in distances:
            s = ego_s + front_offset + distance
            points = []
            for lateral in (-half_width, 0.0, half_width):
                x, y, yaw = route.at(s, lateral)
                # Visibility continues past the rear-axle goal to cover the
                # front bumper; this does not extend the controller route.
                if s > route.length_m:
                    x += (s - route.length_m) * math.cos(yaw)
                    y += (s - route.length_m) * math.sin(yaw)
                points.extend((x, y, self.frame.ego.z_m + z) for z in (0.6, 1.0))
            if not self.clear_points(points):
                return covered
            covered = float(distance)
        return covered
