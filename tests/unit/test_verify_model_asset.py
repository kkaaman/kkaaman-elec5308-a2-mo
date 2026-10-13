"""Tests for scripts/verify_model_asset.py (no torch or model weights needed)."""

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from autonomy.paths import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "verify_model_asset.py"

spec = importlib.util.spec_from_file_location("verify_model_asset", SCRIPT)
vma = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = vma  # dataclasses resolve their module through sys.modules
spec.loader.exec_module(vma)


def _entry(path: str, data: bytes, loader=None) -> dict:
    return {
        "name": Path(path).stem,
        "path": path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "loader": loader,
    }


def _pointer(oid: str, size: int) -> bytes:
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n".encode()


def test_matching_file_passes(tmp_path):
    data = b"\x80\x02fake-weights" * 100
    (tmp_path / "w.pth").write_bytes(data)
    result = vma.check_asset(_entry("w.pth", data), tmp_path)
    assert result.status == "ok"
    assert result.passed
    assert result.actual_sha256 == hashlib.sha256(data).hexdigest()


def test_lfs_pointer_is_detected_not_hashed(tmp_path):
    data = b"real weights" * 1000
    entry = _entry("w.pth", data)
    (tmp_path / "w.pth").write_bytes(_pointer(entry["sha256"], entry["size"]))
    result = vma.check_asset(entry, tmp_path)
    assert result.status == "lfs_pointer"
    assert "git lfs pull" in result.message
    assert not result.passed


def test_parse_lfs_pointer_fields(tmp_path):
    oid = "a" * 64
    (tmp_path / "p").write_bytes(_pointer(oid, 123))
    assert vma.parse_lfs_pointer(tmp_path / "p") == {"oid": oid, "size": 123}
    (tmp_path / "q").write_bytes(b"not a pointer")
    assert vma.parse_lfs_pointer(tmp_path / "q") is None


def test_size_mismatch(tmp_path):
    data = b"x" * 2000
    (tmp_path / "w.pth").write_bytes(data[:-1])
    assert vma.check_asset(_entry("w.pth", data), tmp_path).status == "size_mismatch"


def test_hash_mismatch_same_size(tmp_path):
    data = b"x" * 2000
    (tmp_path / "w.pth").write_bytes(b"y" * 2000)
    assert vma.check_asset(_entry("w.pth", data), tmp_path).status == "hash_mismatch"


def test_missing_file(tmp_path):
    assert vma.check_asset(_entry("absent.pth", b"abc"), tmp_path).status == "missing"


def test_loader_failure_is_reported_not_raised(tmp_path, monkeypatch):
    def broken(weights, sample, repo_root):
        raise ImportError("No module named 'torch'")

    monkeypatch.setitem(vma.LOADERS, "segmentation", broken)
    load = vma.run_load({"path": "w.pth", "loader": "segmentation"}, None, tmp_path)
    assert load["status"] == "failed"
    assert "torch" in load["error"]
    assert load["hint"]


def test_asset_without_loader_is_skipped(tmp_path):
    load = vma.run_load({"path": "w.pth", "loader": None}, None, tmp_path)
    assert load["status"] == "skipped"

    data = b"weights" * 300
    (tmp_path / "w.pth").write_bytes(data)
    result = vma.check_asset(_entry("w.pth", data), tmp_path)
    result.load = load
    assert result.passed  # hash-only assets are not failed by a missing loader


def test_main_exit_codes_and_record(tmp_path, capsys):
    good, bad = b"good" * 600, b"bad" * 800
    (tmp_path / "good.pth").write_bytes(good)
    bad_entry = _entry("bad.pth", bad)
    (tmp_path / "bad.pth").write_bytes(_pointer(bad_entry["sha256"], bad_entry["size"]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"assets": [_entry("good.pth", good), bad_entry]}))

    common = ["--manifest", str(manifest), "--repo-root", str(tmp_path)]
    assert vma.main(["good", *common]) == 0
    record_path = tmp_path / "out" / "record.json"
    assert vma.main([*common, "--record", str(record_path)]) == 1

    record = json.loads(record_path.read_text())
    assert record["all_passed"] is False
    assert {a["status"] for a in record["assets"]} == {"ok", "lfs_pointer"}
    assert "[FAIL] bad" in capsys.readouterr().out


def test_unknown_asset_name_is_an_error(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"assets": [_entry("w.pth", b"abc")]}))
    with pytest.raises(SystemExit):
        vma.main(["nope", "--manifest", str(manifest), "--repo-root", str(tmp_path)])


def test_repository_manifest_is_well_formed():
    manifest = json.loads((REPO_ROOT / "assets" / "model_manifest.json").read_text())
    names = [a["name"] for a in manifest["assets"]]
    assert len(names) == len(set(names))
    for asset in manifest["assets"]:
        assert re.fullmatch(r"[0-9a-f]{64}", asset["sha256"])
        assert asset["size"] > vma.LFS_POINTER_MAX_BYTES
        assert not Path(asset["path"]).is_absolute()
        assert asset["loader"] in (None, *vma.LOADERS)


def test_export_copies_verified_files_with_checksums(tmp_path):
    data = b"weights" * 400
    (tmp_path / "w.pth").write_bytes(data)
    result = vma.check_asset(_entry("w.pth", data), tmp_path)
    sums = vma.export_assets([result], tmp_path, tmp_path / "backup")
    assert sums.read_text().split() == [hashlib.sha256(data).hexdigest(), "w.pth"]
    assert (tmp_path / "backup" / "w.pth").read_bytes() == data


def test_export_refuses_unverified_files(tmp_path):
    result = vma.check_asset(_entry("absent.pth", b"abc"), tmp_path)
    with pytest.raises(ValueError, match="not exporting"):
        vma.export_assets([result], tmp_path, tmp_path / "backup")


# --- The repository's own weights (skipped where Git LFS objects are absent, e.g. CI) ---

def _repo_entry(name):
    manifest = json.loads((REPO_ROOT / "assets" / "model_manifest.json").read_text())
    return next(a for a in manifest["assets"] if a["name"] == name)


@pytest.mark.parametrize("name", ["segmentation_bisenetv2", "lane_unet", "detector_yolo26n"])
def test_repository_weights_match_the_manifest(name):
    result = vma.check_asset(_repo_entry(name), REPO_ROOT)
    if result.status in ("lfs_pointer", "missing"):
        pytest.skip(f"{name}: {result.message}")
    assert result.passed, result.message


def test_segmentation_weights_load_and_segment_the_sample():
    entry = _repo_entry("segmentation_bisenetv2")
    if vma.check_asset(entry, REPO_ROOT).status != "ok":
        pytest.skip("real segmentation weights not present (run git lfs pull)")
    pytest.importorskip("torch")
    load = vma.run_load(entry, REPO_ROOT / "src" / "perception" / "street.jpg", REPO_ROOT)
    assert load["status"] == "ok", load
    assert load["top_classes"][0]["class"] == "road"
