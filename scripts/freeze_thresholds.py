"""Derive a measured speed budget, then freeze only after five matching S0 runs."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from autonomy.configuration import content_hash, read_config, write_config
from evaluation.run_logger import file_sha256
from planning.speed_governor import speed_for_distance


def derive(config, braking, perception, depth):
    if braking.get("status") != "MEASURED_PASS" or depth.get("status") != "MEASURED_PASS":
        raise ValueError("Successful braking and depth calibration are required")
    if perception.get("split") != "val" or perception.get("fault") is not None:
        raise ValueError(
            "Tune on unmodified validation sequences; held-out test results cannot tune thresholds"
        )
    ranges = perception["usable_range_lower_bound_m"]
    usable = min(ranges["car"], ranges["person"])
    if usable < 10:
        raise ValueError(
            "Insufficient validated detection/depth range; collect/review missing bins first"
        )
    latency = perception.get("warm_p95_s")
    if latency is None or latency > 0.1:
        raise ValueError(
            "Perception p95 exceeds 100 ms; fix device/performance before timing acceptance"
        )
    decel = min(config.governor.deceleration_mps2, braking["deceleration_mps2"])
    response = max(
        config.governor.response_budget_s, braking["response_budget_s"], 0.2 + latency + 0.15 + 0.05
    )
    # One metre camera-to-bumper allowance and one metre depth/extent reserve.
    speed = min(
        config.governor.cruise_mps,
        speed_for_distance(usable - 2 - config.governor.stop_margin_m, decel, response),
    )
    speed = int(speed * 10) / 10
    if speed < 1:
        raise ValueError("Measured stopping envelope cannot support baseline cruise")
    return replace(
        config,
        governor=replace(
            config.governor, cruise_mps=speed, deceleration_mps2=decel, response_budget_s=response
        ),
    )


def main():
    from dataclasses import asdict

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "braking", "perception", "depth-calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--navigation-report", type=Path)
    args = parser.parse_args()
    config, _, _ = read_config(args.config)
    files = {
        "braking": args.braking,
        "perception": args.perception,
        "depth": args.depth_calibration,
    }
    reports = {k: json.loads(p.read_text(encoding="utf-8-sig")) for k, p in files.items()}
    if reports["perception"].get("depth_calibration_sha256") != file_sha256(args.depth_calibration):
        raise ValueError("Perception was measured with a different depth calibration")
    candidate = derive(config, reports["braking"], reports["perception"], reports["depth"])
    digest = content_hash(asdict(candidate))
    status = "PROVISIONAL"
    evidence = {k: {"path": str(p.resolve()), "sha256": file_sha256(p)} for k, p in files.items()}
    evidence.update(
        usable_range_lower_bound_m=reports["perception"]["usable_range_lower_bound_m"],
        pedestrian_25m_covered=reports["perception"]["pedestrian_25m_covered"],
        model_sha256=reports["perception"]["model_sha256"],
        rationale="Measured brake >=0.4, 30% deceleration reserve, camera/filter/compute lag, 2 m range reserve; lower cruise if necessary.",
    )
    if args.navigation_report:
        report = json.loads(args.navigation_report.read_text())
        if (
            report.get("status") != "MEASURED_PASS_REVIEW_REQUIRED"
            or report.get("mode") != "navigation"
            or report.get("counts") != {"S0": 5}
            or not report.get("live_carla")
            or any(x["manifest"].get("config_sha256") != digest for x in report["runs"])
        ):
            raise ValueError(
                "Five successful live navigation runs with THIS derived configuration are required"
            )
        status = "FROZEN_FOR_VALIDATION"
        evidence["navigation"] = {
            "path": str(args.navigation_report.resolve()),
            "sha256": file_sha256(args.navigation_report),
        }
        evidence["mission_sha256"] = report["runs"][0]["manifest"]["mission_sha256"]
        evidence["source_sha256"] = report["runs"][0]["manifest"]["source_sha256"]
    write_config(args.output, candidate, status=status, evidence=evidence)
    print(f"{status} | cruise {candidate.governor.cruise_mps:.1f} m/s | {args.output}")
    if status == "PROVISIONAL":
        print(
            "Next: five navigation S0 runs with this config, then repeat with --navigation-report."
        )


if __name__ == "__main__":
    main()
