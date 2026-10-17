"""CARLA-free missing/stale depth and invalid-visibility fault evidence."""

import argparse
import json
from pathlib import Path

from autonomy.baseline import BaselineConfig
from evaluation.metrics import load_run
from evaluation.runner import run_episode
from sim_harness.baseline import BaselineWorld, RasterDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for fault in ("missing_depth", "stale", "invalid_depth"):
        world = BaselineWorld("S1", fault=fault)
        folder = args.output / fault
        result = run_episode(
            world,
            RasterDetector(),
            world.points,
            BaselineConfig(),
            folder,
            dict(
                engine="harness",
                scenario="S1",
                seed=5308,
                step_wall_budget_s=10,
                event_sensors_verified=True,
                truth_channel_complete=True,
            ),
            max_sim_s=12,
        )
        manifest, rows, _ = load_run(folder)
        active = [r for r in rows if r["frame_id"] >= 85]
        if fault in {"missing_depth", "stale"}:
            passed = bool(active) and all(
                r["state"] == "FAULT_STOP"
                and r["command"]["throttle"] == 0
                and r["command"]["brake"] == 1
                for r in active
            )
        else:
            passed = (
                bool(active)
                and all(r["target_mps"] == 0 for r in active)
                and rows[-1]["speed_mps"] <= 0.1
            )
        passed = passed and manifest["cleanup"]["ok"] and result["metrics"]["collisions"] == 0
        reports.append(
            dict(
                fault=fault,
                status="PASS" if passed else "FAIL",
                run=str(folder),
                expected_collection="INCOMPLETE; injected fault prevents normal completion",
            )
        )
    report = dict(
        status="HARNESS_FAULTS_PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL",
        runs=reports,
        live_carla=False,
    )
    (args.output / "faults.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status']} | {args.output}")
    return 0 if report["status"] == "HARNESS_FAULTS_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
