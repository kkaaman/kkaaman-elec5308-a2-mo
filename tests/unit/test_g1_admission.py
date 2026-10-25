"""Production G1 identity checks, including every previously accepted mutation."""

import json
import shutil
from pathlib import Path

import pytest

from autonomy import admission
from autonomy.baseline_operation import frozen_governor_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    record = admission.read_json(ROOT / admission.RECORD_PATH)
    files = record["source_variants"]["canonical_loader_only"]["source_hashes"]
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    record_path = tmp_path / admission.RECORD_PATH
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    config_path = tmp_path / "configs/mission_baseline.json"
    shutil.copyfile(ROOT / "configs/mission_baseline.json", config_path)
    monkeypatch.setattr(admission, "ADMISSION_ROOT", tmp_path)
    return config_path, record_path


def change_config(path, mutate):
    document = admission.read_json(path)
    mutate(document["baseline_threshold_freeze"])
    path.write_text(json.dumps(document))


def test_admitted_exact_values_load(admitted):
    config = frozen_governor_config(admitted[0])
    assert config.cruise_mps == 4.0
    assert config.deceleration_mps2 == 1.497566177392398
    assert config.response_budget_s == .9500000052154064
    assert config.stop_margin_m == 6.0


def test_candidate_remains_rejected(admitted):
    change_config(admitted[0], lambda f: f.update(status="TECHNICALLY_VALIDATED_CANDIDATE"))
    with pytest.raises(ValueError, match="not frozen"):
        frozen_governor_config(admitted[0])


@pytest.mark.parametrize("field,value", [
    ("cruise_speed_mps", 4.5), ("deceleration_mps2", 1.6),
    ("response_budget_s", 1.0), ("stop_margin_m", 5.9),
])
def test_modified_frozen_values_rejected(admitted, field, value):
    change_config(admitted[0], lambda f: f["frozen_values"].update({field: value}))
    with pytest.raises(ValueError, match="configuration identity"):
        frozen_governor_config(admitted[0])


@pytest.mark.parametrize("field,value", [
    ("issue41_runner_scope_sha256", "0" * 64),
    ("issue41_runner_source_revision", "0" * 40),
    ("operational_scope", "Pedestrian/general obstacle operation"),
    ("admitted_profile", "pedestrian_and_general_obstacles"),
])
def test_changed_binding_or_broadened_scope_rejected(admitted, field, value):
    change_config(admitted[0], lambda f: f.update({field: value}))
    with pytest.raises(ValueError, match="configuration identity"):
        frozen_governor_config(admitted[0])


def test_modified_config_outside_freeze_is_also_bound(admitted):
    document = admission.read_json(admitted[0])
    document["controller"]["max_steering_angle_rad"] += .01
    admitted[0].write_text(json.dumps(document))
    with pytest.raises(ValueError, match="configuration identity"):
        frozen_governor_config(admitted[0])


@pytest.mark.parametrize("kind", ["changed", "missing", "added"])
def test_bound_source_inventory_rejected_on_drift(admitted, kind):
    root = admitted[0].parents[1]
    source = root / "src/autonomy/stack.py"
    if kind == "changed":
        source.write_text(source.read_text() + "\n# unreviewed change\n")
    elif kind == "missing":
        source.unlink()
    else:
        (source.parent / "unreviewed.py").write_text("# extra source\n")
    with pytest.raises(ValueError, match="source inventory"):
        frozen_governor_config(admitted[0])


def test_bound_auxiliary_config_change_rejected(admitted):
    source = admitted[0].parent / "acceptance.json"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="source inventory"):
        frozen_governor_config(admitted[0])


def test_formatting_and_object_order_do_not_change_config_identity(admitted):
    document = admission.read_json(admitted[0])
    admitted[0].write_text(json.dumps(dict(reversed(list(document.items()))), indent=4))
    assert frozen_governor_config(admitted[0]).cruise_mps == 4.0


def test_bad_record_config_checksum_rejected(admitted):
    record = admission.read_json(admitted[1])
    record["config_identity_sha256"] = "0" * 64
    admitted[1].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="configuration identity"):
        frozen_governor_config(admitted[0])


def test_refreshed_checksum_cannot_override_frozen_value_record(admitted):
    document = admission.read_json(admitted[0])
    document["baseline_threshold_freeze"]["frozen_values"]["cruise_speed_mps"] = 4.5
    admitted[0].write_text(json.dumps(document))
    record = admission.read_json(admitted[1])
    record["config_identity_sha256"] = admission.config_identity(document)
    admitted[1].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="numeric values"):
        frozen_governor_config(admitted[0])


def test_wrong_runner_record_hash_rejected(admitted):
    record = admission.read_json(admitted[1])
    record["source_variants"]["issue41_runner"]["scope_sha256"] = "0" * 64
    admitted[1].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="runner record fingerprint"):
        frozen_governor_config(admitted[0])


def test_duplicate_json_keys_rejected(admitted):
    payload = admitted[0].read_text()
    admitted[0].write_text('{"cruise_speed_mps":99,' + payload[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        frozen_governor_config(admitted[0])


def test_current_record_pins_source_and_stationary_scope():
    record = admission.read_json(ROOT / admission.RECORD_PATH)
    baseline = admission.load_admitted_baseline(ROOT / "configs/mission_baseline.json")
    assert record["config_identity_sha256"] == admission.config_identity(baseline)
    assert "pedestrian" in record["operational_scope"]
    assert "deferred" in record["operational_scope"]
    assert record["admitted_profile"] == "stationary_vehicle_g2_only"
    assert record["runner_scope"]["clearance_or_resume_established"] is False


def test_unregistered_config_copy_is_rejected(admitted):
    copied = admitted[0].with_name("unregistered.json")
    copied.write_bytes(admitted[0].read_bytes())
    with pytest.raises(ValueError, match="admitted config path"):
        frozen_governor_config(copied)


def test_pr104_binding_rejects_post_pr98_inventory(admitted):
    old = ROOT / "docs/gates/history/G1-admission-pr104.json"
    admitted[1].write_bytes(old.read_bytes())
    with pytest.raises(ValueError, match="source inventory"):
        frozen_governor_config(admitted[0])


@pytest.mark.parametrize("kind", ["changed", "missing", "added"])
def test_pr98_scenario_inventory_still_rejects_drift(admitted, kind):
    source = admitted[0].parent / "scenarios/S3.json"
    if kind == "changed":
        source.write_text(source.read_text().replace('"lateral_m": -3.5', '"lateral_m": 3.5'))
    elif kind == "missing":
        source.unlink()
    else:
        source.with_name("unreviewed.json").write_text("{}")
    with pytest.raises(ValueError, match="source inventory"):
        frozen_governor_config(admitted[0])


def test_pr98_rebind_preserves_pr104_values_scope_and_runner():
    current = admission.read_json(ROOT / admission.RECORD_PATH)
    old = admission.read_json(ROOT / "docs/gates/history/G1-admission-pr104.json")
    for field in ("frozen_values", "operational_scope", "admitted_profile", "runner_scope",
                  "runner_scope_sha256", "runner_base_revision", "config_identity_sha256",
                  "config_file_sha256"):
        assert current[field] == old[field]
    assert current["source_variants"]["issue41_runner"] == old["source_variants"]["issue41_runner"]
    assert admission.source_inventory(ROOT) == current["source_variants"]["canonical_loader_only"]["source_hashes"]
