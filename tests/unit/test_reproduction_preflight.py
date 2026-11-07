"""Fail-closed preflight checks using tiny local fixtures, no real models."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

MODULE = Path(__file__).resolve().parents[2] / "scripts/check_reproduction.py"
spec = importlib.util.spec_from_file_location("doc62_preflight", MODULE)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def reference(payload=b"model"):
    return {"assets": [{"path": "model.pt", "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest()}]}


def test_assets_accept_exact_bytes_and_reject_same_size_corruption(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"model")
    assert preflight.check_reference(tmp_path, reference(), True) == []
    path.write_bytes(b"other")
    assert "SHA-256 mismatch" in preflight.check_reference(tmp_path, reference(), True)[0]


def test_missing_and_pointer_assets_fail(tmp_path):
    assert "Missing model" in preflight.check_reference(tmp_path, reference(), True)[0]
    (tmp_path / "model.pt").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    assert "size mismatch" in preflight.check_reference(tmp_path, reference(), True)[0]


def test_live_check_sets_timeout_and_rejects_server_mismatch(monkeypatch):
    calls = []

    class Client:
        def __init__(self, host, port):
            calls.append((host, port))

        def set_timeout(self, timeout):
            calls.append(timeout)

        def get_client_version(self):
            return "0.9.16"

        def get_server_version(self):
            return "0.9.15"

    monkeypatch.setattr(preflight.importlib, "import_module", lambda name: SimpleNamespace(Client=Client))
    errors = preflight.check_live("localhost", 2000, 2, "0.9.16")
    assert calls == [("localhost", 2000), 2]
    assert errors == ["CARLA version mismatch; use matching client wheel and server"]


def test_assets_only_does_not_inspect_packages(tmp_path, monkeypatch):
    (tmp_path / "model.pt").write_bytes(b"model")

    def forbidden(*args):
        raise AssertionError("package inspection must not run")

    monkeypatch.setattr(preflight.importlib.metadata, "version", forbidden)
    assert preflight.check_reference(tmp_path, reference(), True) == []


def test_required_cuda_absence_fails_without_accepting_cpu(monkeypatch):
    class Client:
        def __init__(self, *args):
            pass

        def set_timeout(self, timeout):
            pass

        def get_client_version(self):
            return "0.9.16"

        def get_server_version(self):
            return "0.9.16"

    modules = {"carla": SimpleNamespace(Client=Client),
               "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))}
    monkeypatch.setattr(preflight.importlib, "import_module", modules.__getitem__)
    assert preflight.check_live("localhost", 2000, 2, "0.9.16", True) == [
        "CUDA unavailable; check NVIDIA hardware, driver and torch build"]


def test_setup_check_only_never_runs_installer(monkeypatch):
    import runpy
    import subprocess
    import sys

    import pytest

    calls = []
    checker = SimpleNamespace(main=lambda args: calls.append(args) or 0)
    monkeypatch.setitem(sys.modules, "scripts.check_reproduction", checker)
    monkeypatch.setattr(sys, "argv", ["setup_environment.py", "--offline", "--check-only", "--assets-only"])

    def forbidden(*args, **kwargs):
        raise AssertionError("check-only must not spawn pip or git")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(MODULE.parents[1] / "setup_environment.py"), run_name="__main__")
    assert exc.value.code == 0
    assert calls == [["--assets-only"]]
