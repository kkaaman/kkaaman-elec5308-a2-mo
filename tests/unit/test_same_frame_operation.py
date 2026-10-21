"""Measured operational clock/admission evidence must fail closed on corruption."""

import copy
import json
from pathlib import Path

import pytest

from scripts.operational_evidence import read_archived_evidence
from scripts.review_same_frame_capture import validate_decision_clocks

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/evidence/BL-D-06/operational-approach"


@pytest.fixture(scope="module")
def captured():
    return json.loads(read_archived_evidence(EVIDENCE))


def test_all_measured_clock_partitions_and_admission_limits(captured):
    config = captured["operational_profile"]["operational_gate"]
    rows = [r for c in captured["cases"] for r in c["rows"]]
    assert len(rows) == 800
    assert sum(r["admitted"] for r in rows) == 345
    assert sum(r["packet"]["command"]["throttle"] > 0 for r in rows) == 22
    assert all(validate_decision_clocks(row, config) < 1e-8 for row in rows)
    assert max(r["source_sim_age_s"] for r in rows) == 0


def test_failure_hold_reset_and_recovery_are_explicit(captured):
    case = next(c for c in captured["cases"] if c["scenario"] == "failure_recovery")
    failure = case["rows"][140]
    assert failure["injected_failure"] and not failure["input_depth_present"]
    assert failure["packet"]["command"]["throttle"] == 0
    assert failure["packet"]["command"]["brake"] == 1
    assert not failure["admitted"]
    assert case["rows"][141]["explicit_reset"] and not case["rows"][141]["admitted"]
    assert any(r["admitted"] for r in case["rows"][142:])


@pytest.mark.parametrize("corruption", ["partition", "source", "expiry", "invalid_positive"])
def test_corrupt_admission_measurements_are_rejected(captured, corruption):
    row = copy.deepcopy(next(r for r in captured["cases"][0]["rows"] if r["admitted"]))
    if corruption == "partition":
        row["sensor_availability_s"] += 0.1
    elif corruption == "source":
        row["packet"]["source_frame_id"] += 1
    elif corruption == "expiry":
        row["packet"]["valid_until_monotonic_s"] += 1
    else:
        row["admitted"] = False
        row["packet"]["command"]["throttle"] = 0.5
    with pytest.raises(ValueError):
        validate_decision_clocks(row, captured["operational_profile"]["operational_gate"])


def test_speed_uses_operational_range_and_enforced_budget_once():
    from planning.speed_governor import speed_for_distance, stopping_distance

    review = json.loads((EVIDENCE / "review.json").read_text())
    a = 1.497566177392398
    budget = 0.5 + 0.1 + 0.3500000052154064
    assert review["response_budget_s"] == pytest.approx(budget)
    assert review["required_gap_at_candidate_m"] == pytest.approx(
        6 + stopping_distance(4, a, budget)
    )
    assert review["exact_speed_cap_mps"] == pytest.approx(
        speed_for_distance(review["usable_operational_range_m"] - 6, a, budget)
    )
    assert review["speed_cap_rounded_down_mps"] == 5.7


def test_kinematic_target_uses_road_reference_not_elevated_spawn():
    from types import SimpleNamespace

    from scripts.capture_operational_approach import pose_along

    api = SimpleNamespace(
        Location=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z),
        Transform=lambda location, rotation: SimpleNamespace(location=location, rotation=rotation),
    )
    road = SimpleNamespace(
        location=SimpleNamespace(x=1, y=2, z=0),
        rotation=SimpleNamespace(yaw=0),
        get_forward_vector=lambda: SimpleNamespace(x=1, y=0, z=0),
    )
    pose = pose_along(api, road, 50)
    assert pose.location.x == 51 and pose.location.y == 2
    assert pose.location.z == 0.05 and road.location.z == 0


def test_archived_evidence_rejects_corruption(tmp_path):
    import shutil

    for name in ("raw-evidence.zip", "raw-evidence.sha256", "raw-evidence.manifest.json"):
        shutil.copyfile(EVIDENCE / name, tmp_path / name)
    with (tmp_path / "raw-evidence.zip").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        read_archived_evidence(tmp_path)
