"""TEST-05 / TEST-07: recompute S3, S4 or S5 reporting from persisted run folders.

Each run folder must contain ``samples.csv``, ``events.json`` and
``manifest.json`` in the ``carla_adapter.s0_runner`` layout. Scenario-specific
evidence (``phase == hold`` samples, ``bumper_gap_m``, ``obstacle_*`` events) is
optional: missing evidence is reported as INSUFFICIENT_EVIDENCE, never PASS.

Example::

    python scripts/evaluate_scenario_runs.py S3 runs/S3/trial-*/run-01 ^
        --matched-s0 runs/S0/rgb-20261101/run-01 --output runs/S3/summary.json
"""

import argparse
import csv
import json
import math
from pathlib import Path

from evaluation.run_log import write_json
from evaluation.scenario_metrics import aggregate_scenario_runs, evaluate_scenario_run

_NUMERIC_EXCLUDED = {"phase", "status"}
_BOOLEAN = {"stable_cruise", "in_goal_window"}


def _value(key, text):
    if key in _NUMERIC_EXCLUDED:
        return text
    if text in ("", None):
        return None
    if key in _BOOLEAN:
        return {"True": True, "False": False}.get(text, text)
    try:
        number = float(text)
    except ValueError:
        return text
    if key in ("frame_age", "simulation_frame", "capture_frame") and number.is_integer():
        return int(number)
    return number if math.isfinite(number) else None


def load_run(folder):
    folder = Path(folder)
    with (folder / "samples.csv").open(newline="", encoding="utf-8") as handle:
        samples = [{k: _value(k, v) for k, v in row.items()} for row in csv.DictReader(handle)]
    events = json.loads((folder / "events.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return samples, events, manifest, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", choices=["S3", "S4", "S5"])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--matched-s0", type=Path, help="S0 run folder for the S3 slowdown comparison")
    parser.add_argument("--mission", type=Path, default=Path("configs/mission_baseline.json"),
                        help="Frozen mission; supplies deceleration/stop margin for the S5 envelope")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    matched_samples = load_run(args.matched_s0)[0] if args.matched_s0 else None
    deceleration = stop_margin = None
    if args.mission.is_file():
        freeze = json.loads(args.mission.read_text(encoding="utf-8")).get("baseline_threshold_freeze", {})
        deceleration = freeze.get("frozen_values", {}).get("deceleration_mps2")
        stop_margin = freeze.get("frozen_values", {}).get("stop_margin_m")
    summaries = []
    for folder in args.runs:
        samples, events, manifest, previous = load_run(folder)
        # Missing evidence is reported, never defaulted: an absent summary.json would
        # otherwise read as "not completed" and an absent cleanup list as "clean".
        missing = [
            text for present, text in (
                ("completed" in previous, "summary.json lacks 'completed' (run the endpoint predicate first)"),
                ("cleanup_errors" in manifest, "manifest.json lacks 'cleanup_errors'"),
            ) if not present
        ]
        if missing:
            parser.error(f"{folder}: " + "; ".join(missing))
        summary = evaluate_scenario_run(
            args.scenario,
            samples,
            events,
            completed=bool(previous["completed"]),
            matched_s0_samples=matched_samples,
            deceleration_mps2=deceleration,
            stop_margin_m=stop_margin,
            failure_reason=manifest.get("failure_reason"),
            route_install_count=manifest.get("route_install_count", 1),
            cleanup_ok=not manifest["cleanup_errors"],
        )
        summary.update(
            run_id=folder.name,
            run_folder=str(folder),
            route_sha256=manifest.get("route_sha256"),
            config_sha256=manifest.get("config_sha256"),
        )
        summaries.append(summary)
    result = aggregate_scenario_runs(args.scenario, summaries)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
