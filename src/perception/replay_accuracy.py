"""Class-aware one-to-one IoU matching for explicitly annotated replay."""
from __future__ import annotations

from obstacle_perception.types import PixelBox


def match_counts(detections, truth, *, scale_x=1.0, scale_y=1.0):
    """Confidence-ordered greedy matching at IoU >= 0.5; micro TP/FP/FN."""
    boxes = []
    for row in truth:
        name = row["class_name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Truth class must be nonempty")
        x1, y1, x2, y2 = row["xyxy"]
        boxes.append((name, PixelBox(x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y)))
    used = set()
    tp = 0
    for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
        candidates = []
        a = detection.box
        for index, (name, b) in enumerate(boxes):
            if index in used or name != detection.class_name:
                continue
            intersection = max(0, min(a.x_max,b.x_max)-max(a.x_min,b.x_min))*max(
                0, min(a.y_max,b.y_max)-max(a.y_min,b.y_min))
            union = a.width_px*a.height_px+b.width_px*b.height_px-intersection
            candidates.append((intersection/union, index))
        if candidates:
            score, index = max(candidates)
            if score >= .5:
                used.add(index)
                tp += 1
    return tp, len(detections)-tp, len(boxes)-tp
