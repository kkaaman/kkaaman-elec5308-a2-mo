"""Calculate E1-03 detector results for the three approved weathers."""

from collections import defaultdict
from statistics import median

CLASSES = ("car", "person")
WEATHERS = ("ClearNoon", "HardRainNoon", "ClearNight")


def _check_inputs(matrix, observations):
    """Only check that this is the correct experiment and that recordings exist."""
    cells = matrix.get("cells", [])
    if len(cells) != 27 or {cell["condition"] for cell in cells} != set(WEATHERS):
        raise ValueError("Expected the approved 27-run weather matrix")
    if (
        observations.get("specification_sha256") != matrix.get("specification_sha256")
        or not observations.get("frames")
    ):
        raise ValueError("Recordings are missing or belong to a different weather matrix")
    return {cell["cell_id"]: cell for cell in cells}


def _iou(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return overlap / (first_area + second_area - overlap)


def _match(labels, detections, threshold):
    """Pair each visible label with at most one same-class detection."""
    candidates = []
    for label_index, label in enumerate(labels):
        if not label["visible"]:
            continue
        for detection_index, detection in enumerate(detections):
            if label["class_name"] == detection["class_name"]:
                overlap = _iou(label["xyxy"], detection["xyxy"])
                if overlap >= threshold:
                    candidates.append((overlap, label_index, detection_index))

    matched_labels, matched_detections, pairs = set(), set(), []
    for _overlap, label_index, detection_index in sorted(candidates, reverse=True):
        if label_index not in matched_labels and detection_index not in matched_detections:
            matched_labels.add(label_index)
            matched_detections.add(detection_index)
            pairs.append((label_index, detection_index))
    return matched_labels, matched_detections, pairs


def _divide(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate_weather(observations, matrix):
    """Return recall, precision and first-confirmed range for cars and people."""
    cells = _check_inputs(matrix, observations)
    threshold = observations.get("match_iou", 0.5)
    frames_by_cell = defaultdict(list)
    for frame in observations["frames"]:
        frames_by_cell[frame["cell_id"]].append(frame)

    totals = {
        (weather, class_name): {
            "weather": weather,
            "class_name": class_name,
            "frames": 0,
            "invalid_frames": 0,
            "labelled_objects": 0,
            "tp": 0,
            "fn": 0,
            "fp": 0,
            "first_confirmed_ranges": [],
        }
        for weather in WEATHERS
        for class_name in CLASSES
    }

    for cell_id, cell in cells.items():
        first_confirmed = {class_name: None for class_name in CLASSES}
        for frame in frames_by_cell.get(cell_id, []):
            labels = frame["labels"]
            detections = frame["detections"]
            matched_labels, matched_detections, pairs = _match(labels, detections, threshold)

            for class_name in CLASSES:
                result = totals[(cell["condition"], class_name)]
                visible_labels = [
                    index
                    for index, label in enumerate(labels)
                    if label["class_name"] == class_name and label["visible"]
                ]
                class_detections = [
                    index
                    for index, detection in enumerate(detections)
                    if detection["class_name"] == class_name
                ]
                result["frames"] += 1
                result["invalid_frames"] += not frame["valid"]
                result["labelled_objects"] += len(visible_labels)
                result["tp"] += len(set(visible_labels) & matched_labels)
                result["fn"] += len(set(visible_labels) - matched_labels)
                result["fp"] += len(set(class_detections) - matched_detections)

            for label_index, detection_index in pairs:
                label = labels[label_index]
                detection = detections[detection_index]
                class_name = label["class_name"]
                if (
                    first_confirmed[class_name] is None
                    and detection["confirmed"]
                    and label["depth_valid"]
                ):
                    first_confirmed[class_name] = label["distance_m"]

        for class_name, distance in first_confirmed.items():
            if distance is not None:
                totals[(cell["condition"], class_name)]["first_confirmed_ranges"].append(
                    distance
                )

    rows, missing_denominators = [], []
    for key in sorted(totals):
        result = totals[key]
        ranges = result.pop("first_confirmed_ranges")
        result["recall"] = _divide(result["tp"], result["tp"] + result["fn"])
        result["precision"] = _divide(result["tp"], result["tp"] + result["fp"])
        result["first_confirmed_range_m"] = {
            "minimum": min(ranges) if ranges else None,
            "median": median(ranges) if ranges else None,
            "maximum": max(ranges) if ranges else None,
        }
        if not result["labelled_objects"]:
            missing_denominators.append(f"{result['weather']}:{result['class_name']}")
        rows.append(result)

    missing_cells = sorted(set(cells) - set(frames_by_cell))
    complete = not missing_cells and not missing_denominators
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "missing_cells": missing_cells,
        "missing_denominators": missing_denominators,
        "rows": rows,
    }


def markdown_table(report):
    """Turn the calculated results into a table for the report."""
    lines = [
        "| Weather | Class | Labels | TP | FN | FP | Recall | Precision | "
        "First-confirmed range min/median/max (m) | Invalid frames |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in report["rows"]:
        ranges = row["first_confirmed_range_m"]

        def show(value):
            return "N/A" if value is None else f"{value:.3f}"

        lines.append(
            f"| {row['weather']} | {row['class_name']} | {row['labelled_objects']} | "
            f"{row['tp']} | {row['fn']} | {row['fp']} | {show(row['recall'])} | "
            f"{show(row['precision'])} | {show(ranges['minimum'])} / "
            f"{show(ranges['median'])} / {show(ranges['maximum'])} | "
            f"{row['invalid_frames']} |"
        )
    return "\n".join(lines) + "\n"
