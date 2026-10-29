"""Fault matrix, diagnostic retention and failure-oracle regression tests."""
import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("offline_fault_suite", ROOT / "scripts/offline_fault_suite.py")
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


@pytest.mark.parametrize("profile", suite.PROFILES[:2])
@pytest.mark.parametrize("behaviour", suite.BEHAVIOURS)
@pytest.mark.parametrize("tracking", (False, True))
@pytest.mark.parametrize("fault", suite.FAULTS)
def test_available_combinations_select_full_brake(profile, behaviour, tracking, fault):
    result = suite.run_case(profile, behaviour, tracking, fault)
    assert result["status"] == "PASS", result
    trace = next(r for r in result["trace"] if r.get("fault") == fault)
    assert trace["command"] == {"steering_angle_rad": 0, "throttle": 0, "brake": 1}
    assert trace["decision_reason"] and trace["session"] and trace["source"]


def test_matrix_cannot_admit_missing_extensions_and_exports_all_rows(tmp_path):
    report = suite.run_suite()
    assert report["status"] == "BLOCKED"
    assert report["counts"] == {"executed": 216, "passed": 216, "failed": 0, "blocked_combinations": 12}
    assert len({(c["profile"], c["behaviour"], c["tracking"], c["fault"]) for c in report["cases"]}) == 216
    suite.write_report(report, tmp_path / "out", tmp_path)
    saved = json.loads((tmp_path / "out/report.json").read_text())
    assert saved["cases"] == json.loads(json.dumps(report["cases"]))
    diagnostics = [json.loads(line) for line in (tmp_path / "out/diagnostics.jsonl").read_text().splitlines()]
    assert len(diagnostics) == sum(len(c["trace"]) for c in report["cases"])
    assert len({r["case_id"] for r in diagnostics}) == 216
    assert all(r["case_fault"] in suite.FAULTS for r in diagnostics)
    with pytest.raises(FileExistsError):
        suite.write_report(report, tmp_path / "out", tmp_path)



def test_failure_oracle_catches_positive_fault_command(monkeypatch):
    original = suite.Rig.step
    def broken(self, fault=None):
        packet, row = original(self, fault)
        if fault:
            packet = suite.replace(packet, command=suite.ControlCommand(0, 1, 0))
        return packet, row
    monkeypatch.setattr(suite.Rig, "step", broken)
    result = suite.run_case("local_fixture", "cruise", True, "stale_capture")
    assert result["status"] == "FAIL"
    assert not result["checks"]["fault_full_brake"]


def test_deadline_miss_resets_clearance_in_actual_policy():
    result = suite.run_case("delayed_fixture", "hold", True, "deadline_miss")
    row = next(r for r in result["trace"] if r.get("fault") == "deadline_miss")
    assert row["clear_count_before"] == 2
    assert row["clear_count_after"] == 0
    assert row["fresh_results"] == 0 and row["blocked_latch"]
    assert row["decision_reason"] == ["processing_budget_hold"]


def test_old_session_result_has_both_identities_and_latches():
    result = suite.run_case("local_fixture", "approach", False, "old_session_result")
    row = next(r for r in result["trace"] if r.get("fault") == "old_session_result")
    assert row["result_metadata"]["episode_id"] == "old-session"
    assert row["session"] == "test18-session"
    assert row["stack_emergency"]


def test_no_network_guard_actually_blocks_connect(monkeypatch):
    def try_network(*args):
        with socket.socket() as connection:
            connection.connect(("127.0.0.1", 1))
    monkeypatch.setattr(suite, "run_case", try_network)
    with pytest.raises(AssertionError, match="Network forbidden"):
        suite.run_suite()


@pytest.mark.parametrize("profile", ("remote", "hybrid", "unknown"))
def test_missing_backend_cannot_be_run_as_fixture(profile):
    with pytest.raises(ValueError, match="remain blocked"):
        suite.run_case(profile, "cruise", True, "missing_depth")


def test_standalone_import_has_no_carla_or_ml_dependency():
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "-c",
                    "import sys; sys.path[:0] = ['src', 'scripts']; "
                    "from offline_fault_suite import run_suite; run_suite(); "
                    "assert not ({'carla', 'torch', 'ultralytics'} & set(sys.modules))"],
                   cwd=root, check=True, timeout=30)
