"""Showcase-quality helpers for the live perception overlay demos.

None of this changes any model's predictions -- it only changes how the
existing outputs (from perception.lane_path.extract_ego_path,
perception.ObjectDetector, traffic_perception.TrafficPerception,
segmentation.SemanticSegmenter) are displayed, so it deliberately doesn't
reach into any teammate's module internals. Two problems this targets,
both purely presentational:

1. Frame-to-frame jitter in the raw lane mask reads as "glitching" even
   when the underlying detection is basically fine -- LaneTemporalSmoother
   fixes this with an EMA over a fixed set of resampled points.
2. A hard miss (extract_ego_path returning None) currently makes the lane
   overlay vanish instantly and puts big red "NO LANE DETECTED" text on
   screen -- both read as "broken" to a live audience. LaneTemporalSmoother
   + the fade-blend helper turn that into a brief, smooth fade-out instead,
   and PerceptionHUD gives a small persistent status badge instead of
   screen-filling text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

_BOUNDARY_KEYS = ("left_boundary", "right_boundary", "centerline")


def _resample_polyline(points, n_points: int) -> Optional[np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return None
    order = np.argsort(pts[:, 1])
    pts = pts[order]
    y_min, y_max = pts[0, 1], pts[-1, 1]
    if y_max - y_min < 1e-6:
        return None
    y_grid = np.linspace(y_min, y_max, n_points)
    x_grid = np.interp(y_grid, pts[:, 1], pts[:, 0])
    return np.stack([x_grid, y_grid], axis=1)


@dataclass
class LaneTemporalSmoother:
    ema_alpha: float = 0.35
    hold_frames: int = 8
    resample_points: int = 25

    _smoothed: Optional[dict] = field(default=None, init=False, repr=False)
    _miss_streak: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        self._smoothed = None
        self._miss_streak = 0

    def update(self, path: Optional[dict]):
        resampled = None
        if path is not None:
            resampled = {}
            for key in _BOUNDARY_KEYS:
                r = _resample_polyline(path.get(key, []), self.resample_points)
                if r is None:
                    resampled = None
                    break
                resampled[key] = r

        if resampled is not None:
            if self._smoothed is None:
                self._smoothed = resampled
            else:
                self._smoothed = {
                    key: self.ema_alpha * resampled[key]
                    + (1 - self.ema_alpha) * self._smoothed[key]
                    for key in _BOUNDARY_KEYS
                }
            self._miss_streak = 0
            return self._smoothed, 1.0, "ok"

        self._miss_streak += 1
        if self._smoothed is None or self._miss_streak > self.hold_frames:
            return None, 0.0, "miss"

        alpha = max(0.0, 1.0 - self._miss_streak / (self.hold_frames + 1))
        return self._smoothed, alpha, "stale"


def blend_overlay(base_bgr: np.ndarray, drawn_bgr: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 1.0:
        return drawn_bgr
    if alpha <= 0.0:
        return base_bgr
    return cv2.addWeighted(drawn_bgr, alpha, base_bgr, 1 - alpha, 0)


def confidence_thickness(confidence: float, min_px: int = 1, max_px: int = 3) -> int:
    confidence = max(0.0, min(1.0, confidence))
    return int(round(min_px + confidence * (max_px - min_px)))


_STATUS_COLOR_BGR = {
    "ok": (60, 200, 60),
    "stale": (0, 200, 255),
    "miss": (60, 60, 220),
    "unknown": (120, 120, 120),
}


@dataclass
class PerceptionHUD:
    modules: tuple = ("segmentation", "detection", "traffic", "lane")
    legend: tuple = ()

    _status: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._status = {m: "unknown" for m in self.modules}

    def mark(self, module: str, status: str) -> None:
        if status not in _STATUS_COLOR_BGR:
            status = "unknown"
        self._status[module] = status

    def draw(self, image_bgr: np.ndarray, fps: Optional[float] = None) -> np.ndarray:
        output = image_bgr.copy()
        x0, y0 = 12, 12
        line_h = 20
        pad = 6

        n_lines = len(self.modules) + (1 if fps is not None else 0) + len(self.legend)
        panel_h = pad * 2 + line_h * n_lines
        panel_w = 190
        sub = output[y0 : y0 + panel_h, x0 : x0 + panel_w]
        if sub.shape[0] == panel_h and sub.shape[1] == panel_w:
            black = np.zeros_like(sub)
            output[y0 : y0 + panel_h, x0 : x0 + panel_w] = cv2.addWeighted(sub, 0.45, black, 0.55, 0)

        ty = y0 + pad + 14
        if fps is not None:
            cv2.putText(
                output, f"{fps:.1f} FPS", (x0 + pad, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )
            ty += line_h

        for module in self.modules:
            color = _STATUS_COLOR_BGR[self._status.get(module, "unknown")]
            cv2.circle(output, (x0 + pad + 5, ty - 5), 5, color, -1, cv2.LINE_AA)
            cv2.putText(
                output, module, (x0 + pad + 18, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
            ty += line_h

        for label, color in self.legend:
            cv2.line(output, (x0 + pad, ty - 8), (x0 + pad + 16, ty - 8), color, 3, cv2.LINE_AA)
            cv2.putText(
                output, label, (x0 + pad + 22, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )
            ty += line_h

        return output
