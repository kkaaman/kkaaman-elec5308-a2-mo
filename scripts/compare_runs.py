"""Plot A1/A2 measurements with their actual definitions and provenance."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.metrics import load_run, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-summary", type=Path, required=True)
    parser.add_argument("--a2-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a1 = json.loads(args.a1_summary.read_text())
    manifest, rows, events = load_run(args.a2_run)
    a2 = summarize(rows, events, manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="DEFINITION_REVIEW_REQUIRED",
        a1=a1,
        a2=a2,
        comparable_cte=False,
        explanation="A1 nearest driving-lane tangent at actor origin differs from A2 fixed-route rear-axle CTE. No percentage improvement is claimed.",
    )
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = [r["sim_time_s"] - rows[0]["sim_time_s"] for r in rows]
    axes[0].plot(t, [r["cte_m"] for r in rows])
    axes[0].set_ylabel("A2 CTE (m)")
    axes[1].plot(t, [r["capture_apply_s"] * 1000 for r in rows])
    axes[1].set_ylabel("Wall age (ms)")
    axes[1].axhline(100, color="red", linestyle="--")
    axes[2].plot(t, [r.get("simulation_age_s", 0) for r in rows])
    axes[2].set_ylabel("Sim age (s)")
    axes[2].set_xlabel("Simulation time (s)")
    fig.suptitle("A2 fixed-route trace; A1 definition differs (see comparison.json)")
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png")
    plt.close(fig)
    print(f"Definition review required | {args.output}")


if __name__ == "__main__":
    main()
