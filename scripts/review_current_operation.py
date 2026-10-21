"""Compose a bounded BL-D candidate from checked same-frame measurements."""

import hashlib
import json
import zipfile

from planning.speed_governor import speed_for_distance, stopping_distance
from scripts.review_same_frame_capture import validate_decision_clocks


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_current_operation(result, root, evidence_root):
    folder = evidence_root / "operational-approach"
    if not (folder / "review.json").exists():
        return result
    capture = json.loads((folder / "capture.json").read_text())
    operation = json.loads((folder / "review.json").read_text())
    manifest = json.loads((folder / "artifact-manifest.json").read_text())
    digest = sha(folder / "capture.json")
    if (
        operation["capture_sha256"] != digest
        or manifest["files"]["report.json"]["sha256"] != digest
    ):
        raise ValueError("Operational capture hash mismatch")
    if operation["review_implementation_sha256"] != sha(
        root / "scripts/review_same_frame_capture.py"
    ) or operation["audit_script_sha256"] != sha(root / "scripts/review_operational_range.py"):
        raise ValueError("Operational reviewer source changed")
    with zipfile.ZipFile(folder / "measured-source.zip") as archive:
        for name, expected in capture["source_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("Operational source archive mismatch")
            if name.startswith("src/") and sha(root / name) != expected:
                raise ValueError("Measured operational runtime changed")
    if (
        capture["status"] != "COMPLETE"
        or not capture["settings_restored"]
        or not capture["no_actor_leaks"]
        or operation["status"] != "PASS"
    ):
        raise ValueError("No complete supported operational evidence")
    if operation["controller_replay_mismatches"] or operation["invalid_positive_packets"]:
        raise ValueError("Operational replay/safety failed")
    profile = capture["operational_profile"]
    first_ranges = []
    for case in capture["cases"]:
        first = None
        for row in case["rows"]:
            validate_decision_clocks(row, profile["operational_gate"])
            ids = {
                row["detections"][i]["track_id"]
                for i in row["evaluation"]["target_detection_indices"]
            }
            if (
                first is None
                and row["admitted"]
                and any(
                    t["identity"]["track_id"] in ids
                    and t["current"]
                    and t["confirmed"]
                    and t["depth_valid"]
                    for t in row["tracks"]
                )
            ):
                first = min(row["evaluation"]["target_gap_m"], row["provider"]["obstacle_gap_m"])
        if first is None:
            raise ValueError("No matched-target first admission")
        first_ranges.append(first)
    visibility = min(first_ranges)
    values = dict(result["candidate_values"])
    response = values["response_budget_s"]
    required = values["stop_margin_m"] + stopping_distance(
        values["cruise_speed_mps"], values["deceleration_mps2"], response
    )
    if required > visibility:
        raise ValueError("Candidate does not fit measured operational range")
    result["status"] = "FROZEN_GATED_OPERATIONAL"
    result["dependency_review"].update(
        usable_bumper_visibility_established=True,
        operational_range_valid_frames=operation["admissible_frames"],
        reason="Same-frame actual-policy approach establishes bounded first-admission range; previous BL-C rejection evidence is preserved.",
    )
    result["operational_measurement"] = dict(
        path="operational-approach/capture.json",
        sha256=digest,
        review_sha256=sha(folder / "review.json"),
        source_archive_sha256=sha(folder / "measured-source.zip"),
        total_frames=operation["total_frames"],
        admissible_frames=operation["admissible_frames"],
        positive_packets=operation["positive_packets"],
        invalid_positive_packets=0,
        cases=operation["cases"],
        timing=operation["timing"],
        controller_replay_mismatches=0,
    )
    measured = operation["timing"]["source_age_at_decision_s"]["max"]
    result["response_composition"].update(
        measured_perception_control_max_s=measured,
        measured_total_s=measured + operation["vehicle_lag_s"],
    )
    result["screening"].update(
        measured_required_gap_m=values["stop_margin_m"]
        + stopping_distance(
            values["cruise_speed_mps"],
            values["deceleration_mps2"],
            measured + operation["vehicle_lag_s"],
        ),
        candidate_budget_required_gap_m=required,
        speed_cap_mps=speed_for_distance(
            visibility - values["stop_margin_m"], values["deceleration_mps2"], response
        ),
        recommended_cruise_mps=values["cruise_speed_mps"],
        visibility_m=visibility,
        candidate_supported=True,
        visibility_status="MEASURED_CONTROLLED_STRAIGHT_FIRST_ADMISSION",
        speed_cap_status="MEASURED_ENVELOPE_BOUND_NOT_DRIVEN_SPEED_CERTIFICATE",
    )
    result["frozen_values"] = values
    result.pop("unsatisfied_technical_criterion", None)
    result["speed_requirement"].update(
        lower_cruise_freezable_now=True,
        reason="Measured operational envelope supports unchanged 4.0 m/s candidate; no lower cruise is needed for this bounded profile.",
    )
    result["publication_status"] = (
        "LOCAL_TECHNICAL_CANDIDATE_PENDING_COMMIT_AND_HUMAN_REVIEW_NOT_G1_ADMISSION"
    )
    return result
