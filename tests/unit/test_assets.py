"""RV-01: the segmentation weights are the real checkpoint and load offline.

Skipped (with the reason shown) in clones without Git LFS objects, such as
CI checkouts, where the .pth files are pointer stubs.
"""

import importlib.util

import pytest

from autonomy.paths import SCRIPTS_DIR, model_path

_spec = importlib.util.spec_from_file_location("verify_assets", SCRIPTS_DIR / "verify_assets.py")
verify_assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_assets)


def _require_real(name):
    try:
        model_path(name)
    except FileNotFoundError as exc:
        pytest.skip(f"{name} weights unavailable here: {exc}")


@pytest.mark.parametrize("name", ["segmentation", "lane"])
def test_lfs_weights_match_their_recorded_checksum(name):
    _require_real(name)
    ok, message = verify_assets.check(name)
    assert ok, message


def test_yolo_weights_match_their_recorded_checksum():
    ok, message = verify_assets.check("yolo")  # normal Git file: present in every clone
    assert ok, message


def test_segmentation_model_loads_and_segments_the_sample_offline():
    _require_real("segmentation")
    summary = verify_assets.load_segmentation()
    assert "classes present" in summary


def test_export_writes_checksum_file(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_assets, "EXPECTED", {"yolo": verify_assets.EXPECTED["yolo"]})
    verify_assets.export(tmp_path)
    sums = (tmp_path / "SHA256SUMS").read_text().split()
    assert sums == [verify_assets.EXPECTED["yolo"][2], "yolo26n.pt"]
    assert (tmp_path / "yolo26n.pt").stat().st_size == verify_assets.EXPECTED["yolo"][1]
