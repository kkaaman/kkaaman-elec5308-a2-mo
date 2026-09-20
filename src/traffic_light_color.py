"""Classify a detected traffic light's lit color from the camera crop.

Why this exists as a separate function rather than a change to
object_detection.py or traffic_perception.py: neither module has any
notion of traffic light COLOR -- object_detection's YOLO model detects
the class "traffic light" only, with no state; traffic_perception passes
through whatever category/label its backend gives it, also with no color
field. Rather than add a new field to a teammate's dataclass, this stays
a standalone function operating on the already-detected bounding box, used
only by demo_fused_lane_following.py's own stop-for-red-light logic.

Approach: real traffic lights (including CARLA's rendered ones) are
vertically arranged red-top / yellow-middle / green-bottom. Split the crop
into thirds and check each third for a bright, saturated match to its
expected hue -- the third with the strongest match wins. Requires a
minimum brightness+saturation to return a definite color at all, so a
fully-dark/off light (still detected as an object by YOLO, since shape
alone can trigger that class) correctly returns "unknown" rather than a
false color.
"""

import cv2
import numpy as np

MIN_SATURATION = 80   # 0-255; below this, a "match" is probably just
                        # background/pole colour, not a lit bulb
MIN_VALUE = 100        # 0-255; an unlit/off bulb is dark regardless of hue
MIN_MATCH_FRACTION = 0.08  # at least this fraction of a third's pixels
                              # must match for that colour to count at all

# OpenCV hue is 0-179. Red wraps around 0/179, so it's checked as two ranges.
_HUE_RANGES = {
    "red": [(0, 10), (170, 179)],
    "yellow": [(15, 35)],
    "green": [(40, 90)],
}


def _match_fraction(hsv_region: np.ndarray, hue_ranges) -> float:
    h, s, v = hsv_region[:, :, 0], hsv_region[:, :, 1], hsv_region[:, :, 2]
    bright_saturated = (s >= MIN_SATURATION) & (v >= MIN_VALUE)
    hue_match = np.zeros(h.shape, dtype=bool)
    for lo, hi in hue_ranges:
        hue_match |= (h >= lo) & (h <= hi)
    matched = hue_match & bright_saturated
    total = h.size
    return matched.sum() / total if total > 0 else 0.0


def classify_traffic_light_color(frame_rgb: np.ndarray, box) -> str:
    """Return "red" / "yellow" / "green" / "unknown" for one detected
    traffic-light bounding box.

    Args:
        frame_rgb: the full camera frame, RGB.
        box: an object with x_min, y_min, x_max, y_max (pixel coords) --
            matches both object_detection.BoundingBox and
            traffic_perception.BoundingBox's shape.
    """
    x_min, y_min = int(max(0, box.x_min)), int(max(0, box.y_min))
    x_max = int(min(frame_rgb.shape[1], box.x_max))
    y_max = int(min(frame_rgb.shape[0], box.y_max))
    if x_max <= x_min or y_max <= y_min:
        return "unknown"

    crop_rgb = frame_rgb[y_min:y_max, x_min:x_max]
    crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)

    height = crop_hsv.shape[0]
    thirds = {
        "red": crop_hsv[0 : height // 3],
        "yellow": crop_hsv[height // 3 : 2 * height // 3],
        "green": crop_hsv[2 * height // 3 :],
    }

    best_color, best_fraction = "unknown", 0.0
    for color, region in thirds.items():
        if region.size == 0:
            continue
        fraction = _match_fraction(region, _HUE_RANGES[color])
        if fraction > best_fraction:
            best_fraction = fraction
            best_color = color

    if best_fraction < MIN_MATCH_FRACTION:
        return "unknown"
    return best_color
