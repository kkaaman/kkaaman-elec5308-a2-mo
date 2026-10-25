"""Reproduce the BL-D-06 evidence review; never populate runtime thresholds."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from planning.speed_governor import speed_for_distance, stopping_distance


def review_cold(evidence_root):
    evidence_root = Path(evidence_root)
    provenance = json.loads((evidence_root / "provenance.json").read_text())
    docs = {}
    for name, record in provenance.items():
        payload = (evidence_root / "sources" / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"Reviewed source hash mismatch: {name}")
        docs[name] = json.loads(payload)
    benchmark = docs["benchmark.json"]
    capture = docs["capture-summary.json"]
    depth = docs["depth-calibration.json"]
    analysis = docs["depth-analysis.json"]
    braking = docs["longitudinal.json"]
    acceptance = docs["acceptance.json"]
    if (
        benchmark["origin"] != "recorded"
        or benchmark["status"] != "MEASURED"
        or capture["capture_status"] != "COMPLETE"
        or not capture["settings_restored"]
        or not capture["no_actor_leaks"]
        or depth["status"] != "MEASURED_PASS"
        or braking["status"] != "MEASURED_PASS"
        or benchmark["input_sha256"] != capture["manifest_sha256"]
    ):
        raise ValueError("Incomplete or mismatched measured evidence")
    a = braking["deceleration_mps2"]
    vehicle_lag = braking["response_budget_s"]
    if a != benchmark["assumptions"]["braking_deceleration_mps2"]:
        raise ValueError("Benchmark braking input differs from reviewed calibration")
    if vehicle_lag != benchmark["assumptions"]["additional_response_s"]:
        raise ValueError("Benchmark vehicle response input differs")
    if benchmark["assumptions"]["stopping_margin_m"] != acceptance["margin_max_m"]:
        raise ValueError("Benchmark stopping margin differs from reviewed configuration")
    if analysis["depth_calibration_sha256"] != provenance["depth-calibration.json"]["sha256"]:
        raise ValueError("Depth analysis calibration provenance differs")
    if depth["convention"] != "axial" or capture["environment"]["rig"] != depth["rig"]:
        raise ValueError("Depth convention/rig mismatch")
    cruise = docs["navigation-baseline.json"]["cruise_speed_mps"]
    responses = [s["first_confirmed_response_s"] for s in benchmark["sequences"]]
    if any(response is None for response in responses):
        raise ValueError("Missing confirmation response")
    # The benchmark response already contains sensor lag, serial backlog,
    # inference and filter time. Do not add sequence-start confirmation latency.
    perception = max(responses)
    response = perception + vehicle_lag
    margin = acceptance["margin_max_m"]
    required = margin + stopping_distance(cruise, a, response)
    # The reviewed claim covers cars and pedestrians through 25 m. Sparse 30 m
    # samples are reported separately, not silently promoted to an operational bound.
    reviewed_range = 25.0
    common = [
        r
        for r in analysis["bins"]
        if r["class_name"] in ("car", "person") and r["range_centre_m"] <= reviewed_range
    ]
    if any(
        r["median_abs_geometry_error_m"] is None or r["invalid_depth_rate"] is None for r in common
    ):
        raise ValueError("Incomplete reviewed depth bins")
    caps = {str(range_m): speed_for_distance(range_m - margin, a, response) for range_m in (25, 30)}
    return dict(
        date="2026-10-20",
        status="BLOCKED_RANGE_LATENCY_COMPOSITION",
        sources=provenance,
        measured_inputs=dict(
            cruise_candidate_mps=cruise,
            braking_deceleration_mps2=a,
            vehicle_response_s=vehicle_lag,
            worst_capture_to_confirmed_result_s=perception,
            optimistic_response_without_controller_s=response,
            reviewed_depth_scope_m=reviewed_range,
            maximum_reported_depth_bin_centre_m=max(r["range_centre_m"] for r in analysis["bins"]),
            upstream_navigation_capture_to_apply_p95_ms=docs["controller-timing.json"][
                "primary_suite"
            ]["worst_run_metrics"]["capture_to_apply_p95_ms"],
        ),
        screening=dict(
            bl_c_reported_cap_mps=benchmark["measured_speed_cap_mps"],
            bl_c_recommended_mps=benchmark["recommended_speed_mps"],
            required_bumper_gap_at_candidate_m=required,
            optimistic_depth_scope_caps_mps=caps,
            candidate_supported_at_reviewed_depth_scope=required <= reviewed_range,
            controller_delay_added_s=0.0,
            controller_delay_note="Zero is an optimistic diagnostic lower bound, not a frozen controller budget. Even this bound fails; the navigation-only p95 is not an integrated perception/control bound.",
            range_note="25 m is the reviewed evidence scope, not a new acceptance ceiling. Treating axial depth as bumper gap is optimistic; no vehicle offsets or error allowance were subtracted.",
        ),
        acceptance=dict(
            stop_margin_min_m=acceptance["margin_min_m"],
            stop_margin_max_m=margin,
            latency_p95_ms=acceptance["latency_p95_ms"],
            changed=False,
        ),
        decision=dict(
            frozen=False,
            reason="BL-C first-confirmation screening alone does not establish a joint depth-range/latency operating envelope for the governor.",
            needed="Validate a freshness-gated warm operational path with complete downstream timing, or obtain sufficient depth-range evidence and revise the supported cruise through review.",
        ),
    )


def review(evidence_root):
    """Retain cold review, then accept only the measured gated operating path."""
    cold = review_cold(evidence_root)
    path = Path(evidence_root) / "warm-operational.json"
    if not path.exists():
        return cold
    warm = json.loads(path.read_text())
    archive_path = Path(evidence_root) / "warm-source.zip"
    with zipfile.ZipFile(archive_path) as archive:
        if archive.read("report.json") != path.read_bytes():
            raise ValueError("Measured report differs from retained source archive")
        for name, digest in warm["source_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Measured runtime source hash mismatch")

    config = warm["stack_config"]
    stack_path = Path(evidence_root).resolve().parents[2] / "configs/stack.yaml"
    if (
        hashlib.sha256(stack_path.read_bytes()).hexdigest()
        != warm["provenance"]["configs/stack.yaml"]
    ):
        raise ValueError("Operational timing budgets differ from measured stack contracts")
    rows = warm["rows"]
    eligible = [r for r in rows if r["warmed"] and not r["injected_stale"]]
    if (
        warm["status"] != "COMPLETE"
        or not warm["settings_restored"]
        or not warm["no_actor_leaks"]
        or not eligible
        or not warm["summary"]["positive_packets"]
    ):
        raise ValueError("No complete positive operational timing evidence")
    if any(r["throttle"] > 0 for r in rows if not r["warmed"] or r["injected_stale"]):
        raise ValueError("Cold or stale result authorised positive motion")
    if any(r["throttle"] > 0 for r in rows if r["index"] == 0):
        raise ValueError("Reset did not begin in HOLD")
    if any(
        not r["packet_current"]
        or r["stack_s"] > config["max_step_wall_s"]
        or r["capture_to_decision_s"] > config["max_observation_age_s"]
        for r in eligible
    ):
        raise ValueError("Operational observation exceeded enforced deadline")
    benchmark = json.loads((Path(evidence_root) / "sources/benchmark.json").read_text())
    for key in ("confidence", "image_size", "model_sha256", "tracker", "tracker_settings"):
        if warm["settings"][key] != benchmark["settings"][key]:
            raise ValueError(f"Operational detector settings changed: {key}")
    if warm["track_policy"] != benchmark["track_policy"]:
        raise ValueError("Confirmation policy changed")
    if (
        warm["rig"]
        != json.loads((Path(evidence_root) / "sources/depth-calibration.json").read_text())["rig"]
    ):
        raise ValueError("Operational rig changed")
    for row in rows:
        partition = sum(
            row[k]
            for k in (
                "sensor_availability_s",
                "bundle_s",
                "detector_s",
                "filter_s",
                "controller_s",
                "decision_and_overhead_s",
            )
        )
        if abs(partition - row["capture_to_decision_s"]) > 1e-8:
            raise ValueError("Timing stages overlap or omit time")
    a = cold["measured_inputs"]["braking_deceleration_mps2"]
    lag = cold["measured_inputs"]["vehicle_response_s"]
    measured_max = max(r["capture_to_decision_s"] for r in eligible)
    # Gate permits ages up to max_observation_age_s, so freezing only the smaller
    # observed maximum would leave admitted observations outside the stop envelope.
    # Delivery reserve follows the existing command TTL and is after availability,
    # not another detector/filter/controller stage.
    response = config["max_observation_age_s"] + config["command_ttl_s"] + lag
    measured_response = measured_max + lag
    margin = cold["acceptance"]["stop_margin_max_m"]
    gap = margin + stopping_distance(4.0, a, response)
    cap = speed_for_distance(25.0 - margin, a, response)
    conditional_supported = gap <= 25.0
    # Depth accuracy through 25 m axial range does not establish usable bumper
    # visibility or first-confirmation range. Missing BL-C read timing cannot be
    # repaired with this parked operational timing ledger.
    dependency_path = (
        Path(evidence_root).resolve().parents[2]
        / "docs/evidence/BL-C-04-06/sensor-wait-timing-review.json"
    )
    dependency = json.loads(dependency_path.read_text())
    dependency_status = dependency["status"]
    if dependency_status != "BLOCKED_INCOMPLETE_TIMING":
        raise ValueError("Fresh BL-C evidence requires explicit range/timing re-review")
    result = dict(
        date="2026-10-20",
        status="BLOCKED_PENDING_FRESH_BL_C_CAPTURE",
        dependency_review=dict(
            bl_c_timing_status=dependency_status,
            timing_review_sha256=hashlib.sha256(dependency_path.read_bytes()).hexdigest(),
            historical_screening_usable=False,
            historical_screening_used_in_operational_budget=False,
            response_budget_independently_justified=True,
            usable_bumper_visibility_established=False,
            depth_accuracy_scope_axial_m=25.0,
            reason="Operational age bound is independent; axial depth accuracy and parked shadow timing do not establish first-confirmation range or 25 m usable bumper visibility. Fresh BL-C capture and explicit range review required.",
        ),
        cold_review_status="HISTORICAL_SUPERSEDED_BL_C_TIMING",
        cold_review=cold,
        warm_measurement=dict(
            path=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            raw_path="runs/BL-D-06/warm-operational-v4/report.json",
            source_archive="warm-source.zip",
            source_archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            source_hashes=warm["source_hashes"],
            summary=warm["summary"],
            cold_first_frames=[r for r in rows if r["index"] == 0],
        ),
        response_composition=dict(
            measured_perception_control_max_s=measured_max,
            measured_vehicle_response_s=lag,
            measured_total_s=measured_response,
            enforced_capture_age_budget_s=config["max_observation_age_s"],
            post_decision_delivery_reserve_s=config["command_ttl_s"],
            candidate_total_response_budget_s=response,
            note="Sensor, detector, filter and control availability appear once in measured end-to-end response. Candidate covers the enforced admission bound, plus distinct command-delivery reserve and vehicle lag once.",
        ),
        screening=dict(
            measured_required_gap_m=margin + stopping_distance(4.0, a, measured_response),
            candidate_budget_required_gap_m=gap,
            speed_cap_mps=cap,
            recommended_cruise_mps=None,
            visibility_m=25.0,
            candidate_supported=False,
            conditional_candidate_supported_at_assumed_visibility=conditional_supported,
            visibility_status="ASSUMPTION_NOT_VALIDATED_BUMPER_VISIBILITY",
            speed_cap_status="CONDITIONAL_DIAGNOSTIC_NOT_FROZEN",
        ),
        candidate_values=dict(
            cruise_speed_mps=4.0,
            deceleration_mps2=a,
            response_budget_s=response,
            stop_margin_m=margin,
        ),
        frozen_values=dict(
            cruise_speed_mps=None,
            deceleration_mps2=a,
            response_budget_s=None,
            stop_margin_m=margin,
        ),
        acceptance=cold["acceptance"],
        limitations=[
            "Parked-ego shadow control timing, not closed-loop stopping acceptance.",
            "Only the gated operational entry point may use this timing composition.",
            "Invalid perception remains HOLD; no empty/expired track proves clearance.",
            "Receipt-to-decision p95 is not a completed capture-to-actuator acceptance run.",
        ],
    )

    return review_fresh_range(result, evidence_root)


def review_fresh_range(result, evidence_root):
    """Replace the missing-capture diagnosis only after checking fresh evidence."""
    evidence_root = Path(evidence_root)
    root = evidence_root.resolve().parents[2]
    fresh = root / "docs/evidence/BL-C-05-refresh"
    capture_path = fresh / "sequences.json"
    audit_path = evidence_root / "operational-range-review.json"
    if not audit_path.exists():
        return result
    capture = json.loads(capture_path.read_text())
    benchmark = json.loads((fresh / "benchmark.json").read_text())
    summary = json.loads((fresh / "capture-summary.json").read_text())
    audit = json.loads(audit_path.read_text())
    capture_hash = hashlib.sha256(capture_path.read_bytes()).hexdigest()
    if audit["capture_sha256"] != capture_hash or benchmark["input_sha256"] != capture_hash:
        raise ValueError("Fresh capture/audit hash mismatch")
    current_hash = hashlib.sha256(
        (root / "scripts/review_operational_range.py").read_bytes()
    ).hexdigest()
    retained_script = evidence_root / "history/review-operational-range-before-sameframe.py.txt"
    if audit["audit_script_sha256"] != current_hash and (
        not retained_script.exists()
        or hashlib.sha256(retained_script.read_bytes()).hexdigest() != audit["audit_script_sha256"]
    ):
        raise ValueError("Operational audit script changed without retained source")
    for name, record in audit["runtime_source_comparison"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != record["current_sha256"]:
            raise ValueError("Audited operational runtime changed")
    profile = json.loads((root / "configs/mission_baseline.json").read_text())[
        "baseline_threshold_freeze"
    ]
    if any(profile[key] != value for key, value in audit["operational_profile"].items()):
        raise ValueError("Audited operational profile changed")
    rows = [r for sequence in capture["sequences"] for r in sequence["frames"]]
    if (
        capture["capture_status"] != "COMPLETE"
        or len(rows) != 600
        or any(r.get("sensor_read_wall_s") is None for r in rows)
    ):
        raise ValueError("Fresh measured-wait capture incomplete")
    if any(r["throttle"] > 0 or r["observation_valid"] for r in audit["rows"]):
        raise ValueError("Operational range admission requires new decision review")
    if sum(s["counts"]["current_invalid_track_frames"] for s in audit["sequences"]) != len(rows):
        raise ValueError("Operational unknown-track blocker changed")
    visibility = summary["conservative_common_range_screen"]["usable_confirmed_bumper_range_m"]
    a = result["candidate_values"]["deceleration_mps2"]
    response = result["response_composition"]["candidate_total_response_budget_s"]
    margin = result["candidate_values"]["stop_margin_m"]
    result["date"] = "2026-10-21"
    result["status"] = "BLOCKED_OPERATIONAL_RANGE_ADMISSION"
    result["dependency_review"].update(
        bl_c_timing_status="FRESH_MEASURED_WAIT_CAPTURE_COMPLETE",
        fresh_capture_sha256=capture_hash,
        fresh_frames=len(rows),
        bl_c_05_task_status="PASS",
        operational_range_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        target_only_sampled_bumper_range_m=visibility,
        fresh_bl_c_conservative_screen_cap_mps=summary["conservative_common_range_screen"][
            "speed_cap_rounded_down_mps"
        ],
        cold_backlog_response_s=summary["conservative_common_range_screen"][
            "perception_response_s"
        ],
        cold_backlog_excluded_from_positive_decisions=True,
        operational_range_valid_frames=0,
        reason=audit["blocker"],
    )
    result["screening"].update(
        speed_cap_mps=speed_for_distance(visibility - margin, a, response),
        visibility_m=visibility,
        conditional_candidate_supported_at_assumed_visibility=True,
        visibility_status="TARGET_ONLY_RANGE_NOT_ADMITTED_ALL_TRACK_OBSERVATION",
        speed_cap_status="CONDITIONAL_OPERATIONAL_ARITHMETIC_NOT_FROZEN",
        candidate_supported=False,
    )
    result["unsatisfied_technical_criterion"] = audit["blocker"]
    result["speed_requirement"] = dict(
        four_mps_mandatory=False,
        source="docs/decisions.md D-003 and configs/acceptance.json; reduce cruise when measured envelope is insufficient, do not relax criteria",
        lower_cruise_freezable_now=False,
        reason="Lower speed changes stopping distance, not invalid obstacle/visibility admission; 3.2 m/s remains a target-only screening recommendation.",
    )
    from scripts.review_current_operation import review_current_operation

    return review_current_operation(result, root, evidence_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=Path("docs/evidence/BL-D-06"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.evidence_root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
