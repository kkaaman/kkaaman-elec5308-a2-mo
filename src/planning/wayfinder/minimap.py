"""Route mini-map rendered with OpenCV, an existing project dependency (E3-03).

The frame shows the full map lanes in grey, the planned route, the driven
portion, the current rear-axle pose with heading, the next manoeuvre point and
a text panel with distance to the next turn, remaining distance and ETA. The
image is a plain ``HxWx3`` uint8 RGB array; callers save or display it.
"""

from __future__ import annotations

import math

import numpy as np

from planning.lane_graph import LaneTopology
from planning.wayfinder.guidance import GuidanceState, format_distance, format_eta
from planning.wayfinder.route import WayfinderRoute

__all__ = ["MiniMapStyle", "render_minimap", "save_png"]

# RGB colours.
BACKGROUND = (24, 26, 30)
LANE = (78, 82, 90)
ROUTE = (54, 140, 255)
DRIVEN = (120, 120, 120)
EGO = (255, 200, 40)
TURN = (255, 90, 90)
TEXT = (235, 235, 235)
PANEL = (40, 44, 52)
ARROW = {"left": "<- LEFT", "right": "RIGHT ->", "straight": "^ STRAIGHT", "arrive": "ARRIVE"}


class MiniMapStyle:
    def __init__(self, size_px: int = 640, margin_px: int = 24, panel_px: int = 96) -> None:
        if min(size_px, margin_px, panel_px) <= 0 or size_px <= 2 * margin_px + panel_px:
            raise ValueError("mini-map size must leave room for margins and the text panel")
        self.size_px, self.margin_px, self.panel_px = size_px, margin_px, panel_px


def _bounds(topology: LaneTopology | None, route: WayfinderRoute):
    xs, ys = [], []
    sources = [route.points_xyz_m]
    if topology is not None:
        sources.extend(lane.points_xyz_m for lane in topology.lanes)
    for points in sources:
        xs.extend(p[0] for p in points)
        ys.extend(p[1] for p in points)
    return min(xs), max(xs), min(ys), max(ys)


def _projector(bounds, style: MiniMapStyle):
    x0, x1, y0, y1 = bounds
    span = max(x1 - x0, y1 - y0, 1e-6)
    drawable = style.size_px - 2 * style.margin_px - style.panel_px
    scale = drawable / span
    offset_x = style.margin_px + (drawable - (x1 - x0) * scale) / 2
    offset_y = style.panel_px + style.margin_px + (drawable - (y1 - y0) * scale) / 2

    def project(x, y):
        # +Y is left in the project frame, so it points up on the map.
        return int(round(offset_x + (x - x0) * scale)), int(round(offset_y + (y1 - y) * scale))

    return project


def _polyline(cv2, image, points, colour, thickness):
    if len(points) >= 2:
        cv2.polylines(image, [np.array(points, dtype=np.int32).reshape(-1, 1, 2)], False, colour, thickness, cv2.LINE_AA)


def render_minimap(route: WayfinderRoute, state: GuidanceState, yaw_rad: float,
                   topology: LaneTopology | None = None, style: MiniMapStyle | None = None) -> np.ndarray:
    """Return an RGB uint8 frame for one guidance state."""
    import cv2

    style = style or MiniMapStyle()
    if not math.isfinite(yaw_rad):
        raise ValueError("yaw_rad must be finite")
    image = np.empty((style.size_px, style.size_px, 3), dtype=np.uint8)
    image[:] = BACKGROUND
    project = _projector(_bounds(topology, route), style)
    if topology is not None:
        for lane in topology.lanes:
            _polyline(cv2, image, [project(p[0], p[1]) for p in lane.points_xyz_m], LANE, 1)
    route_px = [project(p[0], p[1]) for p in route.points_xyz_m]
    _polyline(cv2, image, route_px, ROUTE, 3)
    # Driven part: route points whose arc length is behind the current progress.
    arc, driven = 0.0, [route_px[0]]
    for a, b, pixel in zip(route.points_xyz_m, route.points_xyz_m[1:], route_px[1:]):
        arc += math.dist(a[:2], b[:2])
        if arc > state.progress_m:
            break
        driven.append(pixel)
    _polyline(cv2, image, driven, DRIVEN, 3)
    next_turn = next((m for m in route.maneuvers if m.route_s_m + 1e-6 >= state.progress_m), route.maneuvers[-1])
    turn_point = _point_at(route, next_turn.route_s_m)
    cv2.circle(image, project(*turn_point), 7, TURN, 2, cv2.LINE_AA)
    cv2.circle(image, project(route.goal_xyz_m[0], route.goal_xyz_m[1]), 5, TEXT, -1, cv2.LINE_AA)
    ego = project(state.x_m, state.y_m)
    head = project(state.x_m + 6.0 * math.cos(yaw_rad), state.y_m + 6.0 * math.sin(yaw_rad))
    cv2.circle(image, ego, 6, EGO, -1, cv2.LINE_AA)
    cv2.arrowedLine(image, ego, head, EGO, 2, cv2.LINE_AA, tipLength=0.4)
    cv2.rectangle(image, (0, 0), (style.size_px, style.panel_px), PANEL, -1)
    lines = (
        f"{route.origin_name} -> {route.destination_name}",
        f"Next: {ARROW[state.next_maneuver]} in {format_distance(state.next_maneuver_m)}",
        f"Remaining {format_distance(state.remaining_m)}   {state.speed_mps:.1f} m/s   {format_eta(state.eta_state, state.eta_s)}",
    )
    for row, text in enumerate(lines):
        cv2.putText(image, text, (12, 26 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 1, cv2.LINE_AA)
    return image


def _point_at(route: WayfinderRoute, s_m: float):
    arc = 0.0
    for a, b in zip(route.points_xyz_m, route.points_xyz_m[1:]):
        step = math.dist(a[:2], b[:2])
        if arc + step >= s_m - 1e-9:
            t = 0.0 if step == 0 else min(1.0, max(0.0, (s_m - arc) / step))
            return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        arc += step
    return route.points_xyz_m[-1][:2]


def save_png(image: np.ndarray, path) -> None:
    import cv2

    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write {path}")
