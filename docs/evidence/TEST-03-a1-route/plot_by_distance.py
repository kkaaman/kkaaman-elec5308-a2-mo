"""Same-route TEST-03 figure: signed CTE (A1 definition) and speed against route distance.

Usage (repository root, project venv):
    python docs/evidence/TEST-03-a1-route/plot_by_distance.py <compare_runs output>/traces.json out.png

A1 positions come from docs/evidence/TEST-01-samples.csv and are projected onto the
route; A2 distance is integrated from logged speed (A2 follows the route).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from autonomy.paths import FIXTURES_DIR, REPO_ROOT  # noqa: E402
from planning.corridor import RouteGeometry  # noqa: E402
from planning.lane_graph import LaneTopology  # noqa: E402
from planning.mission import load_route_definition, mission_from_definition  # noqa: E402

INK, SECOND, MUTED, GRID, SURFACE, BAND = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#fcfcfb",
    "#efeeea",
)
A2_COLOUR, A1_COLOUR = "#2a78d6", "#eb6834"
REAR_AXLE_START_M = 1.38641


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    topology = LaneTopology.load(FIXTURES_DIR / "Town10HD.lane_graph.json")
    mission = mission_from_definition(
        topology, load_route_definition(REPO_ROOT / "configs/missions/a1_test01.json")
    )
    route = RouteGeometry(mission.points_xy_m)
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    junction = (
        route.project(*by_id["515:0:-1"].points_xyz_m[0][:2]).s_m,
        mission.turn_exit_s_m,
    )

    traces = json.loads(args.traces.read_text(encoding="utf-8"))
    rows = list(csv.DictReader((REPO_ROOT / "docs/evidence/TEST-01-samples.csv").open()))
    if len(rows) != len(traces["a1"]):
        raise ValueError("traces.json A1 samples do not match TEST-01-samples.csv")
    a1_s = np.array([route.project(float(r["actor_x_m"]), float(r["actor_y_m"])).s_m for r in rows])
    a1_v = np.array([float(r["speed_mps"]) for r in rows])
    a1_cte = np.array([r["cte_a1_m"] for r in traces["a1"]])

    runs = list(traces["a2"].values())
    t = np.array([r["t_s"] for r in runs[0]])
    v = np.array([r["speed_mps"] for r in runs[0]])
    a2_s = REAR_AXLE_START_M + np.concatenate(
        [[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))]
    )
    a2_cte = np.array([r["cte_a1_m"] for r in runs[0]])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": SECOND,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    )
    fig, (cte_ax, speed_ax) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1])
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in (cte_ax, speed_ax):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axvspan(0, 62, color=BAND, lw=0, zorder=0)
        ax.axvspan(*junction, color=BAND, lw=0, zorder=0)
    cte_ax.text(31, 3.25, "opening bend", ha="center", color=MUTED, fontsize=9)
    cte_ax.text(sum(junction) / 2, 3.25, "junction 468", ha="center", color=MUTED, fontsize=9)
    cte_ax.axhline(0, color=MUTED, lw=0.8)
    cte_ax.plot(
        a1_s,
        a1_cte,
        color=A1_COLOUR,
        lw=2,
        label="A1 (TEST-01 recording): mean |CTE| 0.92 m, max 3.43 m",
    )
    cte_ax.plot(
        a2_s,
        a2_cte,
        color=A2_COLOUR,
        lw=2,
        label="A2 (5 live runs, within 3 mm of each other): mean |CTE| 0.025 m, max 0.085 m",
    )
    cte_ax.set_ylabel("Signed CTE, A1 definition (m)")
    cte_ax.set_ylim(-3.7, 3.6)
    cte_ax.legend(loc="lower left", frameon=False, fontsize=9)
    speed_ax.plot(a1_s, a1_v, color=A1_COLOUR, lw=2)
    speed_ax.plot(a2_s, v, color=A2_COLOUR, lw=2)
    speed_ax.set_ylabel("Speed (m/s)")
    speed_ax.set_ylim(0, 5.6)
    speed_ax.set_xlabel("Distance along the route (m)")
    speed_ax.set_xlim(0, route.length_m)
    fig.suptitle(
        "A1 vs A2 on the same Town10HD roads (A1's TEST-01 route)",
        x=0.06,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=140, facecolor=SURFACE)


if __name__ == "__main__":
    main()
