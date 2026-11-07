"""Run the frozen Town10HD S0 baseline and retain every run's evidence."""

import argparse
import json
import math
from pathlib import Path

from carla_adapter.s0_runner import run_s0_suite
from planning.baseline_mission import load_frozen_mission


def positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("A finite positive number is required")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", type=Path, default=Path("configs/mission_baseline.json"))
    parser.add_argument("--runs", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--max-sim-seconds", type=positive_number,
                        help="Single-run smoke override; recorded in the effective configuration hash")
    parser.add_argument("--hold-at-progress", type=positive_number,
                        help="Separate temporary-HOLD regression at this route progress in metres")
    parser.add_argument("--hold-duration", type=positive_number, default=2.0)
    parser.add_argument("--navigation-only", action="store_true",
                        help="Single-run diagnostic without RGB; cannot provide S0 latency acceptance evidence")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.runs == 5 and (args.max_sim_seconds is not None or args.hold_at_progress is not None or args.navigation_only):
        parser.error("Five-run acceptance requires the frozen mission without smoke/HOLD overrides")
    if args.navigation_only and args.runs != 1:
        parser.error("Navigation-only diagnostics require exactly one run")
    mission = load_frozen_mission(args.mission)
    summary = run_s0_suite(
        mission, args.output, runs=args.runs, host=args.host, port=args.port,
        max_simulation_seconds=args.max_sim_seconds,
        hold_at_progress=args.hold_at_progress, hold_duration=args.hold_duration,
        mission_path=args.mission,
        navigation_only=args.navigation_only,
    )
    print(json.dumps(summary, indent=2))
    if args.runs < 5:
        # The suite remains insufficient for five-run acceptance. The process
        # exit code describes whether all requested individual trials passed.
        return 0 if summary["passed_runs"] == args.runs else 1
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
