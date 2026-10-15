"""Simulator-free TEST-01 raw logging and explicitly defined summary metrics."""

import csv
import hashlib
import json
import math
from pathlib import Path

COLUMNS = (
    "sample",
    "wall_epoch_s",
    "elapsed_wall_s",
    "sim_frame",
    "sim_time_s",
    "cycle_wall_before_apply_s",
    "actual_dt_s",
    "mode",
    "lane_available",
    "misses",
    "actor_x_m",
    "actor_y_m",
    "actor_z_m",
    "actor_yaw_rad",
    "speed_mps",
    "nearest_lane_cte_signed_m",
    "nearest_lane_road_id",
    "nearest_lane_id",
    "controller_cte_m",
    "controller_target_mps",
    "controller_diagnostic_current",
    "requested_steering_rad",
    "throttle",
    "brake",
    "applied_carla_steer",
    "applied_throttle",
    "applied_brake",
    "traffic_light_stop",
)


def normalized_hash(path):
    data = Path(path).read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
    return hashlib.sha256(data.encode()).hexdigest()


def verify_sources(root, manifest):
    mismatches = [
        name
        for name, expected in manifest.items()
        if not (Path(root) / name).is_file()
        or normalized_hash(Path(root) / name) != expected
    ]
    if mismatches:
        raise ValueError(
            "Source differs from reviewed baseline: " + ", ".join(mismatches)
        )


def signed_lane_cte(x, y, lane_x, lane_y, lane_yaw_rad):
    """Signed perpendicular offset from nearest lane tangent in core world axes."""
    values = (x, y, lane_x, lane_y, lane_yaw_rad)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("CTE coordinates must be finite")
    return -math.sin(lane_yaw_rad) * (x - lane_x) + math.cos(lane_yaw_rad) * (
        y - lane_y
    )


def metrics(values, absolute=False):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if absolute:
        values = [abs(v) for v in values]
    if not values:
        return {"count": 0, "mean": None, "rms": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "rms": math.sqrt(sum(v * v for v in values) / len(values)),
        "max": max(values),
    }


class CaptureLog:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.stream = (self.folder / "samples.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self.writer = csv.DictWriter(self.stream, fieldnames=COLUMNS)
        self.writer.writeheader()
        self.rows = []

    def append(self, row):
        if set(row) != set(COLUMNS):
            raise ValueError("Raw sample fields do not match the logging schema")
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Nonfinite sample field: {key}")
        self.writer.writerow(row)
        self.stream.flush()
        self.rows.append(dict(row))

    def summary(self):
        return {
            "samples": len(self.rows),
            "nearest_lane_absolute_cte_m": metrics(
                [r["nearest_lane_cte_signed_m"] for r in self.rows], absolute=True
            ),
            "controller_cte_m_current_updates_only": metrics(
                [
                    r["controller_cte_m"]
                    for r in self.rows
                    if r["controller_diagnostic_current"]
                ]
            ),
            "speed_mps": metrics([r["speed_mps"] for r in self.rows]),
            "supplied_prompt_results": {
                "values_m": [0.22, 1.14],
                "status": "UNVERIFIED_SUPPLIED_RESULTS_NOT_MEASURED",
            },
        }

    def close(self):
        self.stream.close()


def write_json(path, record):
    Path(path).write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
