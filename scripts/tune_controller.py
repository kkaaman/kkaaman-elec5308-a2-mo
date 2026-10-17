"""Small fixed-gain navigation sweep; preserve the controller implementation."""

import copy
import json
from dataclasses import replace

from run_scenario import arguments, execute

from autonomy.configuration import read_config, write_config
from evaluation.run_logger import json_text


def main():
    args = arguments().parse_args()
    if args.scenario != "S0":
        raise ValueError("Tune only on clear S0, not obstacle acceptance runs")
    config, _, _ = read_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for i, (lookahead, kp, ki) in enumerate(
        ((4, 0.45, 0.15), (5, 0.4, 0.1), (6, 0.4, 0.1), (5, 0.5, 0.15))
    ):
        trial = copy.copy(args)
        trial.navigation_only = True
        trial.seed = 5000 + i
        trial.config = args.output / f"candidate-{i}.json"
        trial.output = args.output / f"trial-{i}"
        candidate = replace(
            config, controller=replace(config.controller, lookahead_m=lookahead, kp=kp, ki=ki)
        )
        write_config(trial.config, candidate)
        try:
            result = execute(trial)
        except Exception as exc:
            result = {"error": str(exc), "collection_status": "INCOMPLETE", "metrics": {}}
        metrics = result["metrics"]
        qualifies = (
            result["collection_status"] == "COLLECTED"
            and metrics.get("route_completion") == 1
            and metrics.get("collisions") == 0
            and metrics.get("unplanned_lane_invasions") == 0
            and metrics.get("cte_mean_m") is not None
            and metrics["cte_mean_m"] <= 0.3
            and metrics.get("cruise_speed_rms_mps") is not None
            and metrics["cruise_speed_rms_mps"] <= 0.5
        )
        results.append(
            dict(
                candidate=str(trial.config),
                run=str(trial.output),
                metrics=metrics,
                qualifies=qualifies,
                error=result.get("error"),
            )
        )
        manifest = trial.output / "manifest.json"
        if manifest.is_file() and not json.loads(manifest.read_text()).get("cleanup", {}).get("ok"):
            break
    good = [x for x in results if x["qualifies"]]
    selected = (
        min(good, key=lambda x: x["metrics"]["cte_mean_m"] + x["metrics"]["cruise_speed_rms_mps"])
        if good
        else None
    )
    report = dict(
        status="CANDIDATE_REQUIRES_FIVE_INDEPENDENT_RUNS" if selected else "NO_QUALIFIED_CANDIDATE",
        engine=args.engine,
        selected=selected,
        trials=results,
    )
    (args.output / "tuning.json").write_text(json_text(report) + "\n")
    if selected:
        chosen, _, _ = read_config(selected["candidate"])
        write_config(args.output / "selected.json", chosen)
    print(f"{report['status']} | {args.output}")
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
