"""TEST-03: compare A1 and A2 navigation on one cross-track definition.

A1's TEST-01 recorder logged the signed perpendicular offset of the actor origin
from the nearest driving-lane tangent (CARLA ``get_waypoint``). A2 logs rear-axle
distance to its fixed route. Those differ, so this module recomputes the A1
definition for A2 runs from logged poses and the exported lane topology.

Against the committed TEST-01 capture the offline definition reproduces the
values A1 logged to a 1.3 cm maximum difference outside junctions (3 cm inside),
see ``tests/unit/test_a1_comparison.py``. Evaluation-only; no simulator import.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

REAR_AXLE_TO_ACTOR_ORIGIN_M = 1.38641  # Tesla Model 3, BL-E-01 measured geometry
CTE_DEFINITION = (
    "signed perpendicular offset of the actor origin from the nearest driving-lane"
    " centreline tangent (A1 TEST-01 definition); positive left of lane direction"
)


class LaneCentrelines:
    """Nearest driving-lane tangent over every exported lane section."""

    def __init__(self, topology):
        starts, ends, junction, ids = [], [], [], []
        for lane in topology.lanes:
            points = np.asarray(lane.points_xyz_m, dtype=float)[:, :2]
            starts.append(points[:-1])
            ends.append(points[1:])
            junction.extend([lane.junction_id is not None] * (len(points) - 1))
            ids.extend([lane.segment_id] * (len(points) - 1))
        self.start = np.vstack(starts)
        self.delta = np.vstack(ends) - self.start
        self.length2 = (self.delta**2).sum(axis=1)
        keep = self.length2 > 1e-12
        self.start, self.delta, self.length2 = (
            self.start[keep],
            self.delta[keep],
            self.length2[keep],
        )
        self.junction = np.asarray(junction)[keep]
        self.ids = np.asarray(ids)[keep]

    def signed_offset(self, x_m, y_m):
        """(signed offset m, lane section id, inside junction) at a core XY point."""
        point = np.array([x_m, y_m], dtype=float)
        t = np.clip(((point - self.start) * self.delta).sum(axis=1) / self.length2, 0.0, 1.0)
        nearest = self.start + t[:, None] * self.delta
        index = int(np.argmin(np.linalg.norm(point - nearest, axis=1)))
        unit = self.delta[index] / math.sqrt(self.length2[index])
        rel = point - nearest[index]
        return (
            float(unit[0] * rel[1] - unit[1] * rel[0]),
            str(self.ids[index]),
            bool(self.junction[index]),
        )


def _stats(values):
    values = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if not len(values):
        return dict(count=0, mean=None, rms=None, p95=None, max=None)
    return dict(
        count=int(len(values)),
        mean=float(values.mean()),
        rms=float(np.sqrt((values**2).mean())),
        p95=float(np.percentile(values, 95)),
        max=float(values.max()),
    )


def load_a1_capture(path):
    """TEST-01 recorder output: a folder with samples.csv (+ summary.json), or
    the committed ``docs/evidence/TEST-01-samples.csv`` (+ ``TEST-01-summary.json``)."""
    path = Path(path)
    if path.is_dir():
        samples, summary_path = path / "samples.csv", path / "summary.json"
    else:
        samples = path
        summary_path = path.with_name(path.name.replace("samples.csv", "summary.json"))
    with samples.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    return rows, summary


def a1_metrics(rows, lanes=None):
    moving = [r for r in rows if float(r["speed_mps"]) > 0.1]
    logged = [abs(float(r["nearest_lane_cte_signed_m"])) for r in moving]
    result = dict(
        samples=len(rows),
        moving_samples=len(moving),
        cte_a1_definition_abs_m=_stats(logged),
        cycle_wall_before_apply_ms=_stats(
            [1000 * float(r["cycle_wall_before_apply_s"]) for r in rows]
        ),
        control_period_sim_s=_stats([float(r["actual_dt_s"]) for r in rows[1:]]),
        speed_mps=_stats([float(r["speed_mps"]) for r in moving]),
        real_time_factor=_rtf(
            [float(r["sim_time_s"]) for r in rows], [float(r["elapsed_wall_s"]) for r in rows]
        ),
        trace=[
            dict(
                t_s=float(r["sim_time_s"]) - float(rows[0]["sim_time_s"]),
                cte_a1_m=float(r["nearest_lane_cte_signed_m"]),
                speed_mps=float(r["speed_mps"]),
                cycle_ms=1000 * float(r["cycle_wall_before_apply_s"]),
            )
            for r in rows
        ],
    )
    if lanes is not None:
        recomputed = [
            abs(
                lanes.signed_offset(float(r["actor_x_m"]), float(r["actor_y_m"]))[0]
                - float(r["nearest_lane_cte_signed_m"])
            )
            for r in rows
        ]
        result["offline_definition_check_abs_diff_m"] = _stats(recomputed)
    return result


def _rtf(sim, wall):
    if len(sim) < 2 or wall[-1] - wall[0] <= 0:
        return None
    return (sim[-1] - sim[0]) / (wall[-1] - wall[0])


def a2_metrics(manifest, rows, summary, lanes):
    moving = [r for r in rows if r["speed_mps"] > 0.1]
    a1_cte, junction_flags, trace = [], [], []
    t0 = rows[0]["sim_time_s"] if rows else 0.0
    for r in rows:
        ego = r["ego"]
        x = ego["x_m"] + REAR_AXLE_TO_ACTOR_ORIGIN_M * math.cos(ego["yaw_rad"])
        y = ego["y_m"] + REAR_AXLE_TO_ACTOR_ORIGIN_M * math.sin(ego["yaw_rad"])
        offset, _, junction = lanes.signed_offset(x, y)
        trace.append(
            dict(
                t_s=r["sim_time_s"] - t0,
                cte_a1_m=offset,
                cte_route_m=r.get("signed_cte_m"),
                speed_mps=r["speed_mps"],
                capture_apply_ms=1000 * r["capture_apply_s"],
                simulation_age_s=r.get("simulation_age_s", 0.0),
            )
        )
        if r["speed_mps"] > 0.1:
            a1_cte.append(abs(offset))
            junction_flags.append(junction)
    return dict(
        run=manifest.get("run_folder"),
        engine=manifest.get("engine"),
        scenario=manifest.get("scenario"),
        mode=manifest.get("mode"),
        seed=manifest.get("seed"),
        samples=len(rows),
        moving_samples=len(moving),
        cte_a1_definition_abs_m=_stats(a1_cte),
        cte_a1_definition_abs_m_outside_junctions=_stats(
            [v for v, j in zip(a1_cte, junction_flags) if not j]
        ),
        cte_route_rear_axle_abs_m=_stats([abs(r["cte_m"]) for r in moving]),
        capture_apply_ms=_stats([1000 * r["capture_apply_s"] for r in rows]),
        tick_to_apply_ms=_stats(
            [1000 * r.get("tick_to_apply_s", r["capture_apply_s"]) for r in rows]
        ),
        simulation_age_s=_stats([r.get("simulation_age_s", 0.0) for r in rows]),
        control_period_sim_s=float(manifest.get("dt_s", 0.05)),
        speed_mps=_stats([r["speed_mps"] for r in moving]),
        real_time_factor=summary.get("real_time_factor"),
        trace=trace,
    )


KNOWN_DIFFERENCES = (
    "Route: A1 follows lane estimates with a local junction rule from its own spawn; A2 follows"
    " one fixed legal mission. Different roads are driven, so CTE is compared as a"
    " distribution, not sample by sample.",
    "Timing: A1 runs asynchronously (variable control period); A2 runs synchronously at a fixed"
    " 0.05 s step, reading the frame it just ticked.",
    "Speed: A1 cruises at 3 m/s; A2 at the frozen cruise speed (provisional 5 m/s).",
    "Vehicle/geometry: A1 used its demo's first blueprint match, 2.5 m wheelbase and 0.6 rad"
    " normalisation; A2 uses the measured Tesla Model 3 geometry (BL-E-01).",
    "Latency: A1 cycle_wall_before_apply includes all A1 perception for that cycle; A2"
    " capture_apply is first sensor receipt to apply_control. Both are host wall clock.",
)


def compare(a1, a2_runs):
    def worst(key):
        values = [run[key]["mean"] for run in a2_runs if run[key]["mean"] is not None]
        return max(values) if values else None

    a1_cte = a1["cte_a1_definition_abs_m"]["mean"]
    a2_cte = worst("cte_a1_definition_abs_m")
    a2_p95 = [run["capture_apply_ms"]["p95"] for run in a2_runs if run["capture_apply_ms"]["p95"]]
    return dict(
        status="COMPARABLE_DEFINITION_MEASURED_REVIEW_REQUIRED",
        cte_definition=CTE_DEFINITION,
        a1_mean_abs_cte_m=a1_cte,
        a2_worst_run_mean_abs_cte_m=a2_cte,
        cte_ratio_a2_over_a1=(a2_cte / a1_cte) if a1_cte and a2_cte is not None else None,
        a1_cycle_p95_ms=a1["cycle_wall_before_apply_ms"]["p95"],
        a2_worst_capture_apply_p95_ms=max(a2_p95) if a2_p95 else None,
        a1_real_time_factor=a1["real_time_factor"],
        a2_real_time_factor=[run["real_time_factor"] for run in a2_runs],
        a2_live_carla=bool(a2_runs) and all(run["engine"] == "carla" for run in a2_runs),
        known_differences=list(KNOWN_DIFFERENCES),
        note=(
            "Values are measured, not a claim of improvement; review the listed differences"
            " before quoting a ratio. Harness A2 runs are not live evidence."
        ),
    )
