"""A2 on A1's route: driving on the simulator pose vs on its own sensors (RTK GNSS).

Usage (repository root, project venv):
    python docs/evidence/TEST-03-a1-route-estimate/plot_estimate_vs_truth.py \
        <truth compare_runs output>/traces.json <estimate compare_runs output>/traces.json \
        docs/evidence/TEST-03-a1-route-estimate

Writes ``cte_estimate_vs_truth.png`` and ``checks.json`` into the output folder.

The check: on the estimate, the speed controller holds the *measured* wheel speed,
which carries a seeded per-run scale error. The true speed should then differ from
the simulator-pose runs by exactly that error, seed by seed.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from autonomy.localisation import LocalisationConfig, NoisySensors  # noqa: E402
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
TRUTH_COLOUR, ESTIMATE_COLOUR = "#2a78d6", "#1baf7a"  # categorical slots 1 and 3
REAR_AXLE_START_M = 1.38641
SEEDS = range(5308, 5313)


def profile(run):
    t = np.array([r["t_s"] for r in run])
    v = np.array([r["speed_mps"] for r in run])
    s = REAR_AXLE_START_M + np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    return dict(s=s, v=v, cte=np.array([r["cte_a1_m"] for r in run]), t=t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("truth_traces", type=Path)
    parser.add_argument("estimate_traces", type=Path)
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
    truth = [profile(r) for r in json.loads(args.truth_traces.read_text())["a2"].values()]
    estimate = [profile(r) for r in json.loads(args.estimate_traces.read_text())["a2"].values()]
    if len(truth) != len(SEEDS) or len(estimate) != len(SEEDS):
        raise ValueError("Expected five runs (seeds 5308-5312) in each traces file")

    # Seed fingerprint: drawn wheel-speed scale vs the scale implied by true speed
    # (mean moving speed from each compare_runs comparison.json, keyed by seed).
    def mean_speeds(traces_path):
        document = json.loads((traces_path.parent / "comparison.json").read_text())
        return {run["seed"]: run["speed_mps"]["mean"] for run in document["a2"]}

    truth_speed, estimate_speed = mean_speeds(args.truth_traces), mean_speeds(args.estimate_traces)
    rtk = LocalisationConfig.load(REPO_ROOT / "configs/localisation_rtk_gnss.json")
    checks = []
    for seed, base, run in zip(SEEDS, truth, estimate):
        checks.append(
            dict(
                seed=seed,
                drawn_wheel_scale=NoisySensors(rtk.sensors, seed).wheel_scale,
                implied_wheel_scale=truth_speed[seed] / estimate_speed[seed],
                distance_m=float(run["s"][-1]),
                truth_distance_m=float(base["s"][-1]),
                final_speed_mps=float(run["v"][-1]),
            )
        )
    errors = [abs(c["implied_wheel_scale"] - c["drawn_wheel_scale"]) for c in checks]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checks.json").write_text(
        json.dumps(
            dict(
                definition=(
                    "implied = mean moving true speed on the simulator pose / mean moving"
                    " true speed on the estimate, same seed (compare_runs speed_mps.mean);"
                    " drawn = the seeded wheel-speed scale error (NoisySensors, RTK profile)"
                ),
                max_abs_scale_difference=max(errors),
                runs=checks,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

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
    fig, (cte_ax, scale_ax) = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[2.6, 1])
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in (cte_ax, scale_ax):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    cte_ax.axvspan(0, 62, color=BAND, lw=0, zorder=0)
    cte_ax.axvspan(*junction, color=BAND, lw=0, zorder=0)
    cte_ax.text(31, 0.112, "opening bend", ha="center", color=MUTED, fontsize=9)
    cte_ax.text(sum(junction) / 2, 0.112, "junction", ha="center", color=MUTED, fontsize=9)
    cte_ax.axhline(0, color=MUTED, lw=0.8)
    for i, run in enumerate(estimate):
        cte_ax.plot(
            run["s"],
            run["cte"],
            color=ESTIMATE_COLOUR,
            lw=1.5,
            label="Own sensors, RTK GNSS (5 seeds)" if i == 0 else None,
        )
    cte_ax.plot(
        truth[0]["s"],
        truth[0]["cte"],
        color=TRUTH_COLOUR,
        lw=2,
        label="Simulator pose (5 runs, identical)",
    )
    cte_ax.set(
        xlim=(0, route.length_m),
        ylim=(-0.125, 0.125),
        xlabel="Distance along the route (m)",
        ylabel="Signed CTE, A1 definition (m)",
    )
    cte_ax.legend(loc="lower left", frameon=False, fontsize=9)
    cte_ax.set_title(
        "Lane-keeping on the true pose (A1 on the same roads: mean |CTE| 0.92 m)",
        loc="left",
        fontsize=10,
        color=SECOND,
    )

    drawn = [c["drawn_wheel_scale"] for c in checks]
    implied = [c["implied_wheel_scale"] for c in checks]
    lo, hi = 0.975, 1.025
    scale_ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
    scale_ax.text(1.019, 1.0235, "equal", color=MUTED, fontsize=8, ha="right")
    scale_ax.scatter(
        drawn, implied, s=64, color=ESTIMATE_COLOUR, edgecolor=SURFACE, linewidth=2, zorder=3
    )
    for c in checks:
        scale_ax.annotate(
            str(c["seed"]),
            (c["drawn_wheel_scale"], c["implied_wheel_scale"]),
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=8,
            color=SECOND,
        )
    scale_ax.set(
        xlim=(lo, hi),
        ylim=(lo, hi),
        xlabel="Wheel-speed scale error drawn for the seed",
        ylabel="Scale implied by the true speed",
    )
    scale_ax.set_title("The estimate was in control", loc="left", fontsize=10, color=SECOND)
    fig.suptitle(
        "A2 on A1's route: simulator pose vs own sensors (GNSS + gyro + wheel speed), live CARLA",
        x=0.02,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(args.output / "cte_estimate_vs_truth.png", dpi=140, facecolor=SURFACE)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
