"""Run all attempts, preserve failed evidence and recompute scenario criteria."""

import copy
import json

from run_scenario import ROOT, arguments, execute

from evaluation.acceptance import aggregate, find_runs
from evaluation.run_logger import json_text


def main():
    parser = arguments()
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error("Use 1..5 repeats; five are required for CARLA acceptance")
    if args.engine == "carla" and not args.navigation_only:
        from autonomy.configuration import read_config

        _, document, _ = read_config(args.config)
        if document["status"] != "FROZEN_FOR_VALIDATION":
            parser.error(
                "Run calibration/navigation first, then freeze_thresholds.py; provisional experiments use run_scenario.py"
            )
    args.output.mkdir(parents=True, exist_ok=False)
    attempts = []
    for scenario in ("S0",) if args.navigation_only else tuple(f"S{x}" for x in range(6)):
        definition = json.loads((ROOT / f"configs/scenarios/{scenario}.yaml").read_text())
        seeds = definition.get("acceptance_seeds") or [args.seed + i for i in range(5)]
        for repeat in range(args.repeats):
            run = copy.copy(args)
            run.scenario = scenario
            # Seeds come from the versioned scenario definition (BL-B-06).
            run.seed = seeds[repeat]
            run.output = args.output / f"{scenario}-{repeat + 1:02d}"
            try:
                result = execute(run)
                attempts.append(
                    dict(
                        scenario=scenario,
                        seed=run.seed,
                        status=result["collection_status"],
                        error=result["error"],
                    )
                )
            except Exception as exc:
                attempts.append(
                    dict(
                        scenario=scenario,
                        seed=run.seed,
                        status="START_FAILED",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                print(f"{scenario}: START_FAILED: {exc}", flush=True)
            (args.output / "attempts.json").write_text(json_text(attempts) + "\n", encoding="utf-8")
            # A failed cleanup must not expose the following trial to leaked actors.
            manifest = run.output / "manifest.json"
            if manifest.is_file() and not json.loads(manifest.read_text()).get("cleanup", {}).get(
                "ok"
            ):
                print("Cleanup failed; remaining attempts not started.", flush=True)
                break
        else:
            continue
        break
    report = aggregate(
        find_runs(args.output),
        json.loads((ROOT / "tests/fixtures/acceptance_definitions.json").read_text()),
        required_repeats=5,
        navigation_only=args.navigation_only,
    )
    report["attempts"] = attempts
    (args.output / "acceptance.json").write_text(json_text(report) + "\n", encoding="utf-8")
    print(f"{report['status']} | {args.output / 'acceptance.json'}")
    if args.engine == "harness":
        return (
            0
            if len(attempts) == args.repeats * (1 if args.navigation_only else 6)
            and all(x["status"] == "COLLECTED" for x in attempts)
            else 1
        )
    return 0 if report["status"] == "MEASURED_PASS_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
