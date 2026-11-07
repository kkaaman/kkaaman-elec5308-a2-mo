"""Conservative depth-buffer visibility of parking swept body prisms.

No detector absence, map geometry or obstacle truth can establish clearance.
This requires an unobstructed calibrated view of each requested prism. A normal
front camera can legitimately produce UNKNOWN near the car or outside its FOV.
"""

from enum import Enum

import numpy as np

from autonomy.interfaces import FrameBundle
from autonomy.parking_contracts import ParkingConfig, ParkingVehicle


class Clearance(str, Enum):
    CLEAR = "clear"
    BLOCKED = "blocked_or_occluded"
    UNKNOWN = "unknown"


class DepthParkingClearance:
    """Per-frame axial depth queries; whole projected pixel rectangles are checked.

    The rectangular image bound and farthest prism depth overapproximate the
    volume. This may stop unnecessarily at ground/occlusion; it never treats
    unobserved pixels as free. The flat-ground excluded strip, finite pixel
    resolution and supplied depth error bound are explicit sensor assumptions.
    """

    def __init__(self, frame: FrameBundle, vehicle: ParkingVehicle, config: ParkingConfig):
        self.frame, self.vehicle, self.config = frame, vehicle, config
        self._cache = {}
        if config.ground_exclusion_m >= vehicle.height_m:
            raise ValueError("Ground exclusion must be below body height")
        calibration = frame.calibration
        camera = np.asarray(calibration.camera_to_ego)
        self._camera_rotation = camera[:3, :3]
        self._camera_origin = camera[:3, 3]
        yaw = frame.ego.state.yaw_rad
        c, s = np.cos(yaw), np.sin(yaw)
        self._world_to_ego = np.array([[c, -s], [s, c]])

    def check(self, footprint_xy):
        frame, config = self.frame, self.config
        if frame.depth_m is None:
            return Clearance.UNKNOWN
        xy = np.asarray(footprint_xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
            raise ValueError("Expected finite footprint polygon")
        ego = frame.ego.state
        local = (xy - (ego.x_m, ego.y_m)) @ self._world_to_ego
        # z is relative to the known local flat ground, not a simulator obstacle.
        prism = np.vstack([
            np.column_stack((local, np.full(len(xy), height)))
            for height in (config.ground_exclusion_m, self.vehicle.height_m)
        ])
        optical = (prism - self._camera_origin) @ self._camera_rotation
        z = optical[:, 2]
        if np.any(z <= 0) or np.any(z >= config.max_depth_m):
            return Clearance.UNKNOWN
        cal = frame.calibration
        u = cal.fx_px * optical[:, 0] / z + cal.cx_px
        v = cal.fy_px * optical[:, 1] / z + cal.cy_px
        # Outward rounding checks even partially intersecting edge pixels.
        x0, x1 = int(np.floor(u.min())), int(np.ceil(u.max()))
        y0, y1 = int(np.floor(v.min())), int(np.ceil(v.max()))
        height, width = frame.depth_m.shape
        if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
            return Clearance.UNKNOWN
        key = x0, x1, y0, y1
        if key not in self._cache:
            patch = frame.depth_m[y0:y1 + 1, x0:x1 + 1]
            valid = np.isfinite(patch).all() and (patch < config.max_depth_m).all()
            self._cache[key] = float(patch.min()) if valid else None
        nearest = self._cache[key]
        if nearest is None:
            return Clearance.UNKNOWN
        if nearest <= float(z.max()) + config.depth_margin_m:
            return Clearance.BLOCKED
        return Clearance.CLEAR
