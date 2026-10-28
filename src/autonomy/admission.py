"""Verify the versioned G1 configuration and source identity before loading.

The trusted record lives outside configs/src to avoid a source/config hash cycle.
It is a reviewed repository artifact, not a signature against arbitrary repo edits.
"""

import hashlib
import json
from pathlib import Path

ADMISSION_ROOT = Path(__file__).resolve().parents[2]
RECORD_PATH = "docs/gates/G1-admission.json"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate admission/config JSON key")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)


def config_identity(document):
    """All JSON fields/values; ignore only formatting and object-key order."""
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_inventory(root):
    """Exactly the runner's LF-normalized source/config/calibration surface."""
    root = Path(root)
    files = sorted((root / "src").rglob("*.py"))
    files += sorted(p for p in (root / "configs").rglob("*")
                    if p.is_file() and p.name != "mission_baseline.json")
    calibration = root / "docs/evidence/BL-D-06/sources/depth-calibration.json"
    if calibration.is_file():
        files.append(calibration)
    return {p.relative_to(root).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files}


def scope_fingerprint(scope, hashes):
    payload = json.dumps({"scope": scope, "source_hashes": hashes}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def load_admitted_baseline(path):
    """Fail closed on candidate, altered config, broadened scope or source drift."""
    baseline = read_json(path)
    freeze = baseline["baseline_threshold_freeze"]
    if freeze["status"] != "FROZEN_GATED_OPERATIONAL":
        raise ValueError("Baseline is not frozen for operational governance")
    record = read_json(ADMISSION_ROOT / RECORD_PATH)
    if (record["config_path"] != "configs/mission_baseline.json"
            or Path(path).resolve() != (ADMISSION_ROOT / record["config_path"]).resolve()):
        raise ValueError("Configuration path is not the admitted config path")
    if record["decision"] != "PASS" or record["schema_version"] != 1:
        raise ValueError("G1 admission record is not PASS")
    if config_identity(baseline) != record["config_identity_sha256"]:
        raise ValueError("Admitted configuration identity mismatch")
    if (freeze["frozen_values"] != record["frozen_values"]
            or freeze["operational_scope"] != record["operational_scope"]
            or freeze["admitted_profile"] != "stationary_vehicle_g2_only"
            or record["admitted_profile"] != "stationary_vehicle_g2_only"):
        raise ValueError("Admitted numeric values or stationary-vehicle scope mismatch")
    expected = record["runner_scope_sha256"]
    if freeze["issue41_runner_scope_sha256"] != expected:
        raise ValueError("Admitted runner fingerprint mismatch")
    if freeze["issue41_runner_source_revision"] != record["runner_base_revision"]:
        raise ValueError("Admitted source revision mismatch")
    runner_variant = record["source_variants"]["issue41_runner"]
    if (runner_variant["kind"] != "issue41_runner"
            or runner_variant["scope_sha256"] != expected
            or scope_fingerprint(record["runner_scope"], runner_variant["source_hashes"]) != expected):
        raise ValueError("Admitted runner record fingerprint mismatch")
    hashes = source_inventory(ADMISSION_ROOT)
    matching = [v for v in record["source_variants"].values() if v["source_hashes"] == hashes]
    if len(matching) != 1:
        raise ValueError("Admitted source inventory mismatch")
    variant = matching[0]
    if scope_fingerprint(record["runner_scope"], hashes) != variant["scope_sha256"]:
        raise ValueError("Admitted source fingerprint mismatch")
    if (variant["kind"] == "issue41_runner"
            and variant["scope_sha256"] != expected):
        raise ValueError("Admitted runner source differs from binding")
    if variant["kind"] not in ("loader_only", "issue41_runner"):
        raise ValueError("Unsupported admitted source kind")
    # The baseline loader is available on canonical main. Live runner admission
    # additionally requires the integration variant and its own matching hash.
    return baseline
