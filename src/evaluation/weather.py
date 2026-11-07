"""Calculate E1-03 detector results for the three approved weathers."""

import itertools
import math
import re
from collections import defaultdict
from statistics import median

CLASSES = ("car", "person")
WEATHERS = ("ClearNoon", "HardRainNoon", "ClearNight")


def _finite(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _box(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(_finite(v) for v in value)
        and value[2] > value[0]
        and value[3] > value[1]
    )


def _check_inputs(matrix, observations):
    """Reject malformed/duplicate/unpaired data before counting any results."""
    cells = matrix.get("cells", [])
    expected = set(itertools.product(("S0", "S1", "S2"), WEATHERS, (1, 2, 3)))
    try:
        triples = {(c["scenario"], c["condition"], c["repetition"]) for c in cells}
        ids = [c["cell_id"] for c in cells]
        if len(cells) != 27 or triples != expected or len(set(ids)) != 27:
            raise ValueError("Expected unique approved 27-cell matrix")
        for scenario, rep in itertools.product(("S0", "S1", "S2"), (1, 2, 3)):
            seeds = {
                c["seed"] for c in cells if c["scenario"] == scenario and c["repetition"] == rep
            }
            if len(seeds) != 1 or any(type(seed) is not int or seed < 0 for seed in seeds):
                raise ValueError("Weather scene seeds must be paired")
        digest = matrix["specification_sha256"]
        if (
            not isinstance(digest, str)
            or not re.fullmatch("[0-9a-f]{64}", digest)
            or observations.get("specification_sha256") != digest
        ):
            raise ValueError("Missing/mismatched specification identity")
        if observations.get("origin") not in ("recorded", "synthetic"):
            raise ValueError("Explicit recorded/synthetic origin required")
        frames = observations["frames"]
        if not isinstance(frames, list) or not frames:
            raise ValueError("Nonempty frame list required")
        threshold = observations.get("match_iou", 0.5)
        if not _finite(threshold) or not 0 < threshold <= 1:
            raise ValueError("IoU must be in (0,1]")
        seen = set()
        for frame in frames:
            cell = frame["cell_id"]
            frame_id = frame["frame_id"]
            if cell not in ids or type(frame_id) is not int or frame_id < 0:
                raise ValueError("Unknown cell or invalid frame identity")
            key = (cell, frame_id)
            if key in seen:
                raise ValueError("Duplicate cell/frame identity")
            seen.add(key)
            if not _finite(frame["sim_time_s"]) or type(frame["valid"]) is not bool:
                raise ValueError("Finite frame time and boolean validity required")
            labels, detections = frame["labels"], frame["detections"]
            if not isinstance(labels, list) or not isinstance(detections, list):
                raise ValueError("Explicit label/detection lists required")
            for label in labels:
                if (
                    label["class_name"] not in CLASSES
                    or not _box(label["xyxy"])
                    or type(label["visible"]) is not bool
                    or type(label["depth_valid"]) is not bool
                    or not (_finite(label["distance_m"]) or (label["distance_m"] is None and not label["depth_valid"]))
                ):
                    raise ValueError("Invalid labelled object")
            for detection in detections:
                if (
                    detection["class_name"] not in CLASSES
                    or not _box(detection["xyxy"])
                    or type(detection["confirmed"]) is not bool
                ):
                    raise ValueError("Invalid detection")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed weather input") from error
    return {c["cell_id"]: c for c in cells}


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
            "invalid_depth_labels": 0,
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
        ordered = sorted(frames_by_cell.get(cell_id, []), key=lambda f: f["frame_id"])
        if any(a["sim_time_s"] >= b["sim_time_s"] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Frame IDs and simulation times must strictly agree")
        for frame in ordered:
            labels = frame["labels"]
            detections = frame["detections"] if frame["valid"] else []
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
                result["invalid_depth_labels"] += sum(not labels[i]["depth_valid"] for i in visible_labels)
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
                totals[(cell["condition"], class_name)]["first_confirmed_ranges"].append(distance)

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
    invalid_cells = sorted({f["cell_id"] for f in observations["frames"] if not f["valid"]})
    missing_ranges = [f"{r['weather']}:{r['class_name']}" for r in rows
                      if r["first_confirmed_range_m"]["minimum"] is None]
    complete = not missing_cells and not missing_denominators and not invalid_cells and not missing_ranges
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "origin": observations["origin"],
        "operational_acceptance": False,
        "invalid_cells": invalid_cells,
        "missing_cells": missing_cells,
        "missing_denominators": missing_denominators,
        "missing_range_evidence": missing_ranges,
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
