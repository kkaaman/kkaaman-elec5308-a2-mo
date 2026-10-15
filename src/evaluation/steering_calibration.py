"""SI steering-response analysis; no CARLA imports or automatic adapter tuning."""

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


def inferred_angle(wheelbase_m, speed_mps, yaw_rate_rad_s):
    if not all(math.isfinite(v) for v in (wheelbase_m, speed_mps, yaw_rate_rad_s)):
        raise ValueError("Bicycle-model inputs must be finite")
    if wheelbase_m <= 0 or speed_mps <= 0.5:
        raise ValueError("Positive geometry and forward speed above 0.5 m/s required")
    return math.atan(wheelbase_m * yaw_rate_rad_s / speed_mps)


@dataclass
class SpeedServo:
    integral: float = 0.0

    def update(self, target, measured, dt):
        if (
            not all(math.isfinite(v) for v in (target, measured, dt))
            or target <= 0
            or measured < 0
            or dt <= 0
        ):
            raise ValueError("Invalid speed-servo inputs")
        error = target - measured
        proposed = max(-5.0, min(5.0, self.integral + error * dt))
        effort = 0.35 * error + 0.3 * proposed
        if abs(effort) < 1 or self.integral * error < 0:
            self.integral = proposed
        effort = max(-1.0, min(1.0, effort))
        return max(effort, 0.0), max(-effort, 0.0)


def analyze(folder):
    folder = Path(folder)
    with (folder / "samples.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    grouped = {}
    for row in rows:
        if row["phase"] == "sweep":
            grouped.setdefault(int(row["trial"]), []).append(row)
    trials = []
    for trial, values in sorted(grouped.items()):
        requested = float(values[0]["requested_angle_rad"])
        target = float(values[0]["target_speed_mps"])
        selected = [
            r
            for r in values
            if float(r["phase_time_s"]) >= 0.4
            and abs(float(r["speed_mps"]) - target) <= 0.25
        ]
        angles = [float(r["inferred_angle_rad"]) for r in selected]
        mean = statistics.mean(angles) if angles else None
        deviation = statistics.pstdev(angles) if angles else None
        good = (
            (
                len(angles) >= 6
                and mean * requested > 0
                and deviation <= max(0.005, abs(mean) * 0.3)
            )
            if angles
            else False
        )
        trials.append(
            {
                "trial": trial,
                "target_speed_mps": target,
                "requested_angle_rad": requested,
                "accepted_samples": len(angles),
                "inferred_mean_rad": mean,
                "inferred_std_rad": deviation,
                "response_gain": None if mean is None else mean / requested,
                "physics_front_left_mean_deg_raw": statistics.mean(
                    float(r["wheel_fl_deg_raw"]) for r in selected
                )
                if selected
                else None,
                "physics_front_right_mean_deg_raw": statistics.mean(
                    float(r["wheel_fr_deg_raw"]) for r in selected
                )
                if selected
                else None,
                "max_applied_normalized_steer_abs": max(
                    abs(float(r["applied_steer"])) for r in values
                ),
                "quality": "USABLE_SHORT_WINDOW" if good else "REVIEW_REQUIRED",
            }
        )
    report = {
        "status": "MEASURED_REVIEW_REQUIRED",
        "trials": trials,
        "automatic_adapter_update": False,
        "model": "atan(wheelbase * core yaw rate / forward longitudinal speed)",
        "limitations": [
            "Short response window, not proof of steady-state tyre behaviour",
            "Bicycle-equivalent angle may differ from individual Ackermann wheel angles",
            "No extrapolation beyond measured speeds or requested angles",
        ],
    }
    (folder / "calibration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speeds = sorted({t["target_speed_mps"] for t in trials})
    if not speeds:
        raise ValueError("No sweep data available for plotting")
    figure, axes = plt.subplots(
        1, len(speeds), figsize=(5 * len(speeds), 4), squeeze=False
    )
    for axis, speed in zip(axes[0], speeds):
        points = sorted(
            (
                t
                for t in trials
                if t["target_speed_mps"] == speed and t["inferred_mean_rad"] is not None
            ),
            key=lambda t: t["requested_angle_rad"],
        )
        axis.errorbar(
            [math.degrees(t["requested_angle_rad"]) for t in points],
            [math.degrees(t["inferred_mean_rad"]) for t in points],
            yerr=[math.degrees(t["inferred_std_rad"]) for t in points],
            fmt="o",
            label="Yaw-rate estimate ± std",
        )
        axis.plot([-6, 6], [-6, 6], "--", color="gray", label="Requested = inferred")
        axis.set(
            title=f"Target {speed:g} m/s",
            xlabel="Requested core angle (deg)",
            ylabel="Inferred equivalent angle (deg)",
        )
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / "steering_response.png", dpi=160)
    plt.close(figure)
    return report
