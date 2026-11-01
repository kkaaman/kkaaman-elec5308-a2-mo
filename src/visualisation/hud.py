"""HUD overlay hook for the lane-departure monitor (RV-06, E3).

Rendering is display-only. ``departure_hud_lines`` is pure text so the HUD
contents can be tested without OpenCV; ``draw_departure_overlay`` imports
``cv2`` lazily and returns a new image, leaving the input untouched.
"""

from __future__ import annotations

import math

import numpy as np

from .lane_departure import DepartureReport

__all__ = ["departure_hud_lines", "draw_departure_overlay"]

_STATUS_COLOR_RGB = {
    "agree": (60, 200, 60),
    "disagree": (255, 160, 0),
    "invalid": (220, 60, 60),
}


def departure_hud_lines(report: DepartureReport | None) -> list[str]:
    """Text lines describing a report: status, offsets (m), confidence and reasons.

    Offsets are printed in metres with sign (``+`` = left). A ``None`` report
    yields a single ``lane monitor: no report`` line.
    """
    if report is None:
        return ["lane monitor: no report"]
    lines = [f"lane monitor: {report.status.upper()}  conf {report.confidence:.2f}"]
    if report.image_offset_m is not None:
        heading = (
            f"  hdg {math.degrees(report.image_heading_rad):+.1f} deg"
            if report.image_heading_rad is not None
            else ""
        )
        lines.append(f"image offset {report.image_offset_m:+.2f} m{heading}")
    else:
        lines.append("image offset --")
    if report.map_cte_m is not None:
        lines.append(f"map CTE      {report.map_cte_m:+.2f} m")
    else:
        lines.append("map CTE      --")
    if report.difference_m is not None:
        lines.append(f"difference   {report.difference_m:+.2f} m")
    if report.reasons:
        lines.append("reasons: " + ", ".join(report.reasons))
    return lines


def draw_departure_overlay(
    image_rgb: np.ndarray,
    report: DepartureReport | None,
    *,
    origin_px: tuple[int, int] = (12, 12),
    alpha: float = 0.55,
) -> np.ndarray:
    """Draw the monitor panel and a signed offset bar on a copy of ``image_rgb``.

    ``image_rgb`` must be a nonempty HxWx3 uint8 RGB image. The bar under the
    text shows the image offset (filled) and map CTE (tick) on a +/-2.5 m
    scale, positive to the left of centre to match the ego +Y convention.
    """
    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.dtype != np.uint8
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or 0 in image_rgb.shape
    ):
        raise ValueError("image_rgb must be a nonempty HxWx3 uint8 RGB image")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    import cv2

    output = image_rgb.copy()
    lines = departure_hud_lines(report)
    status = report.status if report is not None else "invalid"
    color = _STATUS_COLOR_RGB.get(status, _STATUS_COLOR_RGB["invalid"])
    x0, y0 = origin_px
    line_h, pad = 18, 6
    panel_w, panel_h = 300, pad * 2 + line_h * len(lines) + 16
    y1, x1 = min(output.shape[0], y0 + panel_h), min(output.shape[1], x0 + panel_w)
    if y1 > y0 and x1 > x0:
        region = output[y0:y1, x0:x1]
        output[y0:y1, x0:x1] = cv2.addWeighted(
            region, 1 - alpha, np.zeros_like(region), alpha, 0
        )
    ty = y0 + pad + 13
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (x0 + pad, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color if index == 0 else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        ty += line_h
    # Offset bar: centre = lane/route centre; left of centre = positive offset.
    bar_y = ty - 4
    bar_left, bar_right = x0 + pad, x0 + panel_w - pad
    centre = (bar_left + bar_right) // 2
    cv2.line(output, (bar_left, bar_y), (bar_right, bar_y), (160, 160, 160), 1, cv2.LINE_AA)
    cv2.line(output, (centre, bar_y - 6), (centre, bar_y + 6), (160, 160, 160), 1, cv2.LINE_AA)
    scale_px_per_m = (bar_right - bar_left) / 2 / 2.5
    if report is not None and report.image_offset_m is not None:
        end = int(round(centre - report.image_offset_m * scale_px_per_m))
        end = max(bar_left, min(bar_right, end))
        cv2.line(output, (centre, bar_y), (end, bar_y), color, 4, cv2.LINE_AA)
    if report is not None and report.map_cte_m is not None:
        tick = int(round(centre - report.map_cte_m * scale_px_per_m))
        tick = max(bar_left, min(bar_right, tick))
        cv2.line(output, (tick, bar_y - 8), (tick, bar_y + 8), (255, 255, 255), 2, cv2.LINE_AA)
    return output
