"""Retain every P1 synthetic-depth trial, samples, provenance and trajectory plots."""

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from autonomy.parking_contracts import ParkingConfig
from autonomy.paths import REPO_ROOT
from planning.parking_geometry import bay_polygon, rectangle
from sim_harness.parking import run_parking_case, standard_case


def plot_trial(path, rows, plan, goal, vehicle, name):
    # Optional presentation dependency; the acceptance computation does not use it.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, (ax, speed) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    ax.plot([p.x_m for p in plan.waypoints], [p.y_m for p in plan.waypoints],
            "--", color="#949cab", label="planned rear axle")
    ax.plot([r["x_m"] for r in rows], [r["y_m"] for r in rows], color="#1267af", label="driven rear axle")
    ax.add_patch(Polygon(bay_polygon(goal.bay), fill=False, edgecolor="#14845a", linewidth=2, label="bay"))
    for row in rows[::max(1, len(rows)//10)] + [rows[-1]]:
        body = rectangle(row["x_m"], row["y_m"], row["yaw_rad"], vehicle.front_m,
                         vehicle.rear_m, vehicle.half_width_m)
        ax.add_patch(Polygon(body, fill=False, edgecolor="#1267af", alpha=0.35))
    for box in standard_case(name)[-1]:
        ax.add_patch(Polygon(box.corners, color="#ce5045", alpha=0.7, label="scripted obstacle"))
    ax.set(xlabel="World X (m)", ylabel="World Y (m)", title=name)
    ax.axis("equal")
    ax.legend(fontsize=8)
    speed.plot([r["sim_time_s"] for r in rows], [r["speed_mps"] for r in rows], label="measured")
    speed.plot([r["sim_time_s"] for r in rows], [r["target_speed_mps"] for r in rows], "--", label="target")
    speed.axhline(1, color="#ce5045", linestyle=":", label="P1 limit")
    speed.set(xlabel="Simulation time (s)", ylabel="Speed (m/s)", ylim=(-0.05, 1.1))
    speed.legend()
    fig.suptitle("P1 synthetic depth / kinematic harness — not live CARLA evidence")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/parking_p1.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "runs/P1")
    parser.add_argument("--plots", action="store_true", help="Also create PNG plots (requires matplotlib)")
    args = parser.parse_args()
    document = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if document["scope"] != "synthetic_depth_kinematic_P1":
        raise ValueError("Only the explicit synthetic P1 profile is supported")
    config = ParkingConfig(**document["policy"])
    names = document["cases"]
    if not names or len(set(names)) != len(names):
        raise ValueError("Require distinct P1 cases")
    for name in names:
        standard_case(name)
    batch = args.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    batch.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for directory in ("autonomy", "planning", "navigation_control", "obstacle_perception", "sim_harness"):
        for path in sorted((REPO_ROOT / "src" / directory).glob("*.py")):
            hashes[path.relative_to(REPO_ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (Path(__file__).resolve(), args.config.resolve()):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(
        scope=document["scope"], policy=asdict(config), source_sha256=hashes,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True),
        cases=names, sensor="ideal downward 256x256 axial depth, height 8 m, FOV 120 deg",
        exclusions=["live CARLA", "G1 admission", "reverse", "front-camera coverage certification"],
    )
    (batch / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summaries = []
    for name in names:
        directory = batch / name
        directory.mkdir()
        try:
            with (directory / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = None

                def retain(row):
                    nonlocal writer
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=row.keys())
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()

                summary, rows, plan, goal, vehicle = run_parking_case(name, config=config, sample_sink=retain)
            (directory / "plan.json").write_text(json.dumps(
                dict(plan=asdict(plan), goal=asdict(goal), vehicle=asdict(vehicle)), indent=2
            ) + "\n", encoding="utf-8")
            if args.plots:
                plot_trial(directory / "trajectory.png", rows, plan, goal, vehicle, name)
        except Exception as error:
            summary = dict(case=name, passed=False, error=f"{type(error).__name__}: {error}")
        summaries.append(summary)
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(case=name, passed=summary["passed"])), flush=True)
    passed = all(result["passed"] for result in summaries)
    (batch / "summary.json").write_text(json.dumps(
        dict(passed=passed, scope=document["scope"], cases=summaries), indent=2
    ) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=passed, output=str(batch))))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
