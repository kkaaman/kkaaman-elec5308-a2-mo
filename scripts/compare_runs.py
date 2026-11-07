"""TEST-03: A1 versus A2 navigation on A1's cross-track definition, plus loop cost.

Example (A2 runs from the Navigation stage, A1 from the committed TEST-01 capture):

    python scripts/compare_runs.py --a1-capture docs/evidence/TEST-01-samples.csv \
        --a2-run <navigation>/runs/S0-01 <navigation>/runs/S0-02 \
        --topology <LiveSetup>/mission/topology.json --output <new folder>
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.a1_comparison import (
    LaneCentrelines,
    a1_metrics,
    a2_metrics,
    compare,
    load_a1_capture,
)
from planning.lane_graph import LaneTopology

ROOT = Path(__file__).resolve().parents[1]


def _load_a2_run(folder):
    """Load the retained TEST-02 CARLA CSV format for TEST-03 comparison."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())

    config = manifest["effective_config"]
    dt_s = float(config["runtime"]["fixed_delta_seconds"])

    with (folder / "samples.csv").open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    rows = []
    for row in raw_rows:
        rows.append(
            {
                "sim_time_s": float(row["simulation_time_s"]),
                "speed_mps": float(row["speed_mps"]),
                "ego": {
                    "x_m": float(row["x_m"]),
                    "y_m": float(row["y_m"]),
                    "yaw_rad": float(row["yaw_rad"]),
                },
                "signed_cte_m": float(row["signed_cte_m"]),
                "cte_m": float(row["signed_cte_m"]),
                "capture_apply_s": float(row["capture_to_apply_ms"]) / 1000.0,
                "simulation_age_s": float(row["frame_age"]) * dt_s,
            }
        )

    normalized_manifest = dict(manifest)
    normalized_manifest.update(
        engine="carla",
        scenario=manifest.get("scenario"),
        mode="rgb",
        seed=config["runtime"].get("seed"),
        dt_s=dt_s,
    )

    if len(raw_rows) >= 2:
        sim_elapsed = (
            float(raw_rows[-1]["simulation_time_s"])
            - float(raw_rows[0]["simulation_time_s"])
        )
        wall_elapsed = (
            float(raw_rows[-1]["apply_monotonic_s"])
            - float(raw_rows[0]["apply_monotonic_s"])
        )
        summary["real_time_factor"] = (
            sim_elapsed / wall_elapsed if wall_elapsed > 0 else None
        )
    else:
        summary["real_time_factor"] = None

    return normalized_manifest, rows, summary


def _cdf(ax, values, label):
    values = np.sort(np.asarray(values, dtype=float))
    if len(values):
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a1-capture",
        type=Path,
        default=ROOT / "docs/evidence/TEST-01-samples.csv",
        help="TEST-01 folder (samples.csv) or the committed samples CSV",
    )
    parser.add_argument("--a2-run", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--topology",
        type=Path,
        default=ROOT / "docs/evidence/baseline-batch/junction-topology.json",
        help="Exported Town10HD_Opt lane topology (LiveSetup writes mission/topology.json)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lanes = LaneCentrelines(LaneTopology.load(args.topology))
    a1_rows, a1_summary = load_a1_capture(args.a1_capture)
    a1 = a1_metrics(a1_rows, lanes)
    a2 = []
    for folder in args.a2_run:
        manifest, rows, summary = _load_a2_run(folder)
        a2.append(a2_metrics(dict(manifest, run_folder=str(folder)), rows, summary, lanes))
    args.output.mkdir(parents=True, exist_ok=False)
    report = compare(a1, a2)
    report.update(
        a1=dict(
            {k: v for k, v in a1.items() if k != "trace"},
            capture_status=a1_summary.get("capture_status"),
        ),
        a2=[{k: v for k, v in run.items() if k != "trace"} for run in a2],
        topology=str(args.topology),
    )
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    traces = dict(a1=a1["trace"], a2={run["run"]: run["trace"] for run in a2})
    (args.output / "traces.json").write_text(json.dumps(traces) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(
        [t["t_s"] for t in a1["trace"]], [t["cte_a1_m"] for t in a1["trace"]], label="A1 (TEST-01)"
    )
    for run in a2:
        label = f"A2 {run['scenario']} seed {run['seed']}"
        ax.plot(
            [t["t_s"] for t in run["trace"]], [t["cte_a1_m"] for t in run["trace"]], label=label
        )
    ax.set(
        xlabel="Simulation time (s)",
        ylabel="Signed CTE, A1 definition (m)",
        title="Cross-track error",
    )
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    _cdf(ax, [abs(t["cte_a1_m"]) for t in a1["trace"] if t["speed_mps"] > 0.1], "A1")
    for run in a2:
        _cdf(
            ax,
            [abs(t["cte_a1_m"]) for t in run["trace"] if t["speed_mps"] > 0.1],
            f"A2 seed {run['seed']}",
        )
    ax.set(xlabel="|CTE| (m), moving samples", ylabel="Fraction", title="CTE distribution")
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    _cdf(ax, [t["cycle_ms"] for t in a1["trace"]], "A1 cycle before apply")
    for run in a2:
        _cdf(
            ax,
            [t["capture_apply_ms"] for t in run["trace"]],
            f"A2 receipt→apply seed {run['seed']}",
        )
    ax.axvline(100, color="red", linestyle="--", linewidth=1)
    ax.set(
        xscale="log",
        xlabel="Wall latency (ms)",
        ylabel="Fraction",
        title="Loop cost (100 ms limit)",
    )
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    _cdf(ax, [float(r["actual_dt_s"]) for r in a1_rows[1:]], "A1 control period (sim s)")
    for run in a2:
        _cdf(
            ax,
            [run["control_period_sim_s"] + t["simulation_age_s"] for t in run["trace"]],
            f"A2 period + command age seed {run['seed']}",
        )
    ax.set(xlabel="Simulation seconds", ylabel="Fraction", title="Command age / control period")
    ax.legend(fontsize=7)
    fig.suptitle("TEST-03: A1 vs A2 on A1's CTE definition (see comparison.json for differences)")
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=130)
    plt.close(fig)
    print(
        f"{report['status']} | A1 mean |CTE| {report['a1_mean_abs_cte_m']:.3f} m"
        f" | A2 worst {report['a2_worst_run_mean_abs_cte_m']} m | {args.output}"
    )


if __name__ == "__main__":
    main()
