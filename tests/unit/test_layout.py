"""The A2 package layout imports cleanly and paths resolve from anywhere."""

import os
import subprocess
import sys

import pytest

from autonomy import paths

CORE_PACKAGES = [
    "autonomy",
    "autonomy.paths",
    "cloud",
    "evaluation",
    "geometry",
    "navigation_control",
    "obstacle_perception",
    "planning",
    "sim_harness",
    "traffic_perception",
    "visualisation",
]


def test_core_packages_import_without_carla():
    # Fresh interpreter: other tests install a fake ``carla`` module.
    code = (
        "import importlib, sys\n"
        f"for name in {CORE_PACKAGES!r}:\n"
        "    importlib.import_module(name)\n"
        "    assert 'carla' not in sys.modules, name\n"
    )
    env = dict(os.environ, PYTHONPATH=str(paths.SRC_DIR))
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_repo_root_is_the_clone():
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()
    assert paths.SRC_DIR == paths.REPO_ROOT / "src"
    assert paths.CONFIG_DIR.is_dir()


def test_config_path_adds_yaml_suffix():
    assert paths.config_path("stack") == paths.CONFIG_DIR / "stack.yaml"
    assert paths.config_path("scenarios/S0") == paths.SCENARIO_DIR / "S0.yaml"
    assert paths.config_path("weather/clear.yml") == paths.CONFIG_DIR / "weather/clear.yml"


def test_config_path_passes_absolute_paths_through(tmp_path):
    custom = tmp_path / "my.yaml"
    assert paths.config_path(str(custom)) == custom


def test_model_path_finds_weights_by_short_name(tmp_path):
    (tmp_path / "segmentation_model.pth").write_bytes(b"\x80\x02weights")
    found = paths.model_path("segmentation", root=tmp_path)
    assert found == tmp_path / "segmentation_model.pth"


def test_model_path_finds_yolo_by_short_name(tmp_path):
    # Path resolution is a unit test. Real model identity/inference is checked
    # by check_detector_offline.py; source-only exports omit model binaries.
    (tmp_path / "yolo26n.pt").write_bytes(b"path-resolution fixture, not model weights")
    assert paths.model_path("yolo", root=tmp_path) == tmp_path / "yolo26n.pt"


def test_model_path_explains_lfs_pointer(tmp_path):
    (tmp_path / "lane_unet_best.pth").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"
    )
    with pytest.raises(FileNotFoundError, match="git lfs pull"):
        paths.model_path("lane", root=tmp_path)


def test_model_path_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="known names"):
        paths.model_path("lane", root=tmp_path)


def test_run_dir_never_reuses_a_run(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    first = paths.run_dir("S0", "run-1")
    assert first == tmp_path / "S0" / "run-1" and first.is_dir()
    with pytest.raises(FileExistsError):
        paths.run_dir("S0", "run-1")
