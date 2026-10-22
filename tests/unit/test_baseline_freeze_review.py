"""Joint evidence checks: never advertise an unsupported BL-D-06 freeze."""

import json
import math
import shutil
from pathlib import Path

import pytest

from planning.behaviour import ClearanceGuard
from planning.speed_governor import (
    GovernorConfig,
    SpeedGovernor,
    speed_for_distance,
    stopping_distance,
)
from scripts.review_baseline_freeze import review, review_cold

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/evidence/BL-D-06"


@pytest.fixture
def audited():
    return review_cold(EVIDENCE)


def governor_config(audited):
    values = audited["measured_inputs"]
    # These existing BL-D-07 test policy values are not claimed as frozen calibrations.
    return GovernorConfig(
        cruise_mps=values["cruise_candidate_mps"],
        deceleration_mps2=values["braking_deceleration_mps2"],
        lateral_acceleration_mps2=1.5,
        response_budget_s=values["optimistic_response_without_controller_s"],
        stop_margin_m=6.0,
        acceleration_mps2=1.0,
        emergency_margin_m=2.0,
    )


def test_saved_review_reproduces_from_hashed_sources(audited):
    assert review(EVIDENCE) == json.loads((EVIDENCE / "review.json").read_text())
    assert audited == json.loads((EVIDENCE / "cold-review.json").read_text())
    assert all(
        len(source["source_commit"]) == 40 and len(source["source_blob"]) == 40
        for source in audited["sources"].values()
    )


def test_historical_unsupported_review_does_not_replace_fresh_gated_evidence(audited):
    baseline = json.loads((ROOT / "configs/mission_baseline.json").read_text())
    freeze = baseline["baseline_threshold_freeze"]
    assert audited["status"] == "BLOCKED_RANGE_LATENCY_COMPOSITION"
    assert freeze["status"] == "TECHNICALLY_VALIDATED_CANDIDATE"
    assert baseline["cruise_speed_mps"] == 4.0
    assert (
        freeze["candidate_values"]["response_budget_s"]
        < audited["measured_inputs"]["optimistic_response_without_controller_s"]
    )
    assert not audited["decision"]["frozen"]
    assert not baseline["obstacle_truth_for_control"]


def test_response_composition_counts_capture_to_result_once(audited):
    values = audited["measured_inputs"]
    assert values["braking_deceleration_mps2"] == pytest.approx(1.497566177392398)
    assert values["vehicle_response_s"] == pytest.approx(0.3500000052154064)
    assert values["worst_capture_to_confirmed_result_s"] == pytest.approx(4.802199796118202)
    assert values["optimistic_response_without_controller_s"] == pytest.approx(5.152199801333609)
    # Sequence-start latency is not another response component.
    assert values["optimistic_response_without_controller_s"] < 6.202199816979828


def test_four_mps_does_not_fit_reviewed_depth_scope(audited):
    cfg = governor_config(audited)
    required = cfg.stop_margin_m + stopping_distance(
        4.0, cfg.deceleration_mps2, cfg.response_budget_s
    )
    assert required == pytest.approx(31.950800193851556)
    assert required > 30 > 25
    # Vehicle lag alone would incorrectly suggest enough headroom.
    assert 6 + stopping_distance(4.0, cfg.deceleration_mps2, 0.3500000052154064) < 25


def test_existing_visibility_ceiling_reduces_four_mps(audited):
    governor = SpeedGovernor(governor_config(audited))
    governor.previous_target_mps = 4.0
    speed, emergency = governor.limit(speed_mps=4.0, dt_s=0.05, visible_distance_m=25.0)
    assert speed == pytest.approx(3.074998366509492)
    assert not emergency


def test_original_bl_c_screening_cap_is_preserved_not_treated_as_joint_proof(audited):
    data = json.loads((EVIDENCE / "sources/benchmark.json").read_text())
    a = audited["measured_inputs"]["braking_deceleration_mps2"]
    lag = audited["measured_inputs"]["vehicle_response_s"]
    cap = min(
        speed_for_distance(
            s["first_confirmed"]["range_m"] - 6.0, a, s["first_confirmed_response_s"] + lag
        )
        for s in data["sequences"]
    )
    assert math.floor(cap * 10) / 10 == data["measured_speed_cap_mps"] == 4.9
    assert data["recommended_speed_mps"] == 4.0
    assert not audited["screening"]["candidate_supported_at_reviewed_depth_scope"]


def test_acceptance_constraints_are_not_relaxed(audited):
    baseline = json.loads((ROOT / "configs/mission_baseline.json").read_text())
    constraints = baseline["baseline_threshold_freeze"]["candidate_constraints"]
    assert constraints == dict(
        cruise_speed_mps=5.0,
        minimum_detection_range_m=25.0,
        stop_margin_m=dict(minimum=2.5, maximum=6.0),
        perception_latency_p95_s=0.1,
    )
    assert audited["acceptance"] == dict(
        stop_margin_min_m=2.5,
        stop_margin_max_m=6.0,
        latency_p95_ms=100.0,
        changed=False,
    )


def test_invalid_perception_keeps_hold_and_emergency_precedence(audited):
    guard = ClearanceGuard()
    assert not guard.update(time_s=0.0, obstacle_present=True, observation_valid=True)
    # Missing/expired/stale observations are invalid, never valid empty clearance.
    assert not guard.update(time_s=0.1, obstacle_present=False, observation_valid=False)
    governor = SpeedGovernor(governor_config(audited))
    governor.previous_target_mps = 4.0
    speed, emergency = governor.limit(speed_mps=4.0, dt_s=0.05, visible_distance_m=25.0, hold=True)
    assert speed == 0 and not emergency
    speed, emergency = governor.limit(speed_mps=4.0, dt_s=0.05, obstacle_gap_m=1.0, hold=True)
    assert speed == 0 and emergency
    assert guard.blocked_latch


def test_modified_evidence_is_rejected(tmp_path):
    copied = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, copied)
    (copied / "sources/benchmark.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        review(copied)


def test_warm_freeze_covers_the_full_admitted_age_not_only_the_observed_maximum():
    from autonomy.baseline_operation import frozen_governor_config

    result = review(EVIDENCE)
    baseline = json.loads((ROOT / "configs/mission_baseline.json").read_text())
    with pytest.raises(ValueError, match="not frozen"):
        frozen_governor_config(ROOT / "configs/mission_baseline.json")
    values = baseline["baseline_threshold_freeze"]["candidate_values"]
    composition = result["response_composition"]
    assert composition["measured_perception_control_max_s"] == pytest.approx(0.06097370599854912)
    assert values["response_budget_s"] == pytest.approx(0.5 + 0.1 + 0.3500000052154064)
    assert values["response_budget_s"] > composition["measured_total_s"]
    assert 6 + stopping_distance(
        4.0, values["deceleration_mps2"], values["response_budget_s"]
    ) == pytest.approx(15.142001009378749)
    assert result["screening"]["candidate_supported"]
    assert result["screening"]["conditional_candidate_supported_at_assumed_visibility"]
    assert result["dependency_review"]["response_budget_independently_justified"]
    assert not result["dependency_review"]["historical_screening_used_in_operational_budget"]
    assert result["dependency_review"]["usable_bumper_visibility_established"]
    assert values["cruise_speed_mps"] == 4.0
    assert result["screening"]["speed_cap_mps"] == pytest.approx(
        speed_for_distance(22.348599722608924 - 6, 1.497566177392398, values["response_budget_s"])
    )


def test_baseline_loader_rejects_a_budget_that_omits_admitted_age(tmp_path):
    from autonomy.baseline_operation import frozen_governor_config

    doc = json.loads((ROOT / "configs/mission_baseline.json").read_text())
    freeze = doc["baseline_threshold_freeze"]
    freeze["status"] = "FROZEN_GATED_OPERATIONAL"  # Hypothetical accepted fixture only.
    freeze["frozen_values"] = dict(freeze["candidate_values"], response_budget_s=0.35)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="omits"):
        frozen_governor_config(path)


def test_warm_evidence_has_no_cold_or_expired_positive_packets():
    warm = json.loads((EVIDENCE / "warm-operational.json").read_text())
    assert len(warm["rows"]) == 240
    assert warm["summary"]["positive_packets"] == 122
    assert warm["summary"]["cold_or_stale_positive_packets"] == 0
    assert all(r["throttle"] == 0 for r in warm["rows"] if r["index"] == 0 or r["injected_stale"])


def test_only_calibration_and_envelope_values_are_in_frozen_values():
    freeze = json.loads((ROOT / "configs/mission_baseline.json").read_text())[
        "baseline_threshold_freeze"
    ]
    assert set(freeze["frozen_values"]) == {
        "cruise_speed_mps",
        "deceleration_mps2",
        "response_budget_s",
        "stop_margin_m",
    }
    assert freeze["frozen_values"]["cruise_speed_mps"] is None
    assert freeze["frozen_values"]["response_budget_s"] is None
    assert freeze["implementation_policy"]["lateral_acceleration_mps2"] == 1.5
    assert freeze["implementation_policy"]["emergency_margin_m"] == 2.0


def test_current_review_does_not_promote_historical_bl_c_screen():
    result = review(EVIDENCE)
    assert result["status"] == "TECHNICALLY_VALIDATED_CANDIDATE"
    assert not result["dependency_review"]["historical_screening_usable"]
    assert result["screening"]["recommended_cruise_mps"] == 4.0
    assert result["cold_review_status"] == "HISTORICAL_SUPERSEDED_BL_C_TIMING"


def test_distinct_operational_capture_resolves_the_old_bl_c_admission_blocker():
    result = review(EVIDENCE)
    dependency = result["dependency_review"]
    assert dependency["bl_c_05_task_status"] == "PASS"
    assert dependency["fresh_frames"] == 600
    assert dependency["fresh_bl_c_conservative_screen_cap_mps"] == 3.2
    assert dependency["cold_backlog_response_s"] == pytest.approx(4.199441865980589)
    assert dependency["cold_backlog_excluded_from_positive_decisions"]
    assert dependency["operational_range_valid_frames"] == 345
    assert not result["speed_requirement"]["four_mps_mandatory"]
    assert not result["speed_requirement"]["lower_cruise_freezable_now"]


def test_optimistic_all_track_replay_still_holds_every_captured_frame():
    audit = json.loads((EVIDENCE / "operational-range-review.json").read_text())
    assert len(audit["rows"]) == 600
    for sequence in audit["sequences"]:
        counts = sequence["counts"]
        assert counts["frames"] == counts["current_invalid_track_frames"] == 200
        assert counts["target_depth_valid_frames"] > 0
        assert counts["positive_commands"] == counts["observation_valid_frames"] == 0
    assert all(row["brake"] == 1 and row["throttle"] == 0 for row in audit["rows"])


def test_documented_repository_root_review_command(tmp_path):
    import os
    import subprocess
    import sys

    output = tmp_path / "review.json"
    environment = dict(os.environ, PYTHONPATH="src")
    subprocess.run(
        [sys.executable, "-m", "scripts.review_baseline_freeze", "--output", str(output)],
        cwd=ROOT, env=environment, check=True, capture_output=True,
    )
    result = json.loads(output.read_text())
    assert result["status"] == "TECHNICALLY_VALIDATED_CANDIDATE"
    assert result["screening"]["recommended_cruise_mps"] == 4.0
    assert "pedestrian operational first-admission unestablished" in result["operational_scope"]
