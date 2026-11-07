"""DEMO ONLY: image-based connected illuminated-lamp colour candidate.

RGB input, OpenCV HSV hue units. No resize, map state, or control output.
Saved 8x16 green regression: highly saturated yellow housing defeats S*V;
its green lamp is brighter but less saturated. Keep hue regions separate,
select compact bright components, and never infer colour from box thirds.
"""
import cv2
import numpy as np


def classify_traffic_light_color(frame_rgb, box):
    h, w = frame_rgb.shape[:2]
    x0, x1 = [int(np.clip(x, 0, w)) for x in (box.x_min, box.x_max)]
    y0, y1 = [int(np.clip(y, 0, h)) for y in (box.y_min, box.y_max)]
    if x1 <= x0 or y1 <= y0:
        return "unknown"
    hsv = cv2.cvtColor(frame_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    # Keep native pixels: interpolation can merge housing with a tiny lamp.
    # Near-peak brightness includes the pale illuminated core, without
    # rewarding the much more saturated but dimmer painted housing.
    bright = (sat >= 70) & (val >= max(100, 0.90 * float(val.max())))
    masks = {
        "red": (hue <= 10) | (hue >= 170),
        "yellow": (hue >= 15) & (hue <= 35),
        "green": (hue >= 40) & (hue <= 90),
    }
    candidates = []
    for colour, hue_mask in masks.items():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (bright & hue_mask).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            # A connected lamp core is compact; a long sunlit housing edge
            # is not. Two pixels is the existing minimum evidence size.
            if area < 2 or ch > 0.45 * (y1-y0) or not 0.4 <= cw/ch <= 2.5:
                continue
            score = float(np.square(val[labels == i].astype(float)/255).sum())
            candidates.append((score, colour))
    return max(candidates)[1] if candidates else "unknown"
