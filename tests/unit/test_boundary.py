"""RV-05: failure fixtures for Ruff, AST analysis and isolated core imports."""

import os
import subprocess
import sys

import pytest

from scripts.check_simulator_boundary import (
    REPO_ROOT,
    boundary_violations,
    smoke_imports,
    source_modules,
)

SRC_DIR = REPO_ROOT / "src"


def write_source(root, relative_path, code):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "code,target",
    [
        ("import carla", "carla"),
        ("import carla.libcarla as sdk", "carla.libcarla"),
        ("from carla import Vehicle", "carla"),
        ("import carla_adapter as adapter", "carla_adapter"),
        ("from carla_adapter.vehicle import FixedVehicleAdapter", "carla_adapter.vehicle"),
        ("from src.carla_adapter.vehicle import FixedVehicleAdapter", "src.carla_adapter.vehicle"),
        ("from src import carla_adapter", "src.carla_adapter"),
        ("def load():\n    import carla", "carla"),
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import carla", "carla"),
        ("import carla  # noqa: TID251", "carla"),
        ("import demo_fused_lane_following", "demo_fused_lane_following"),
        ("import importlib\nimportlib.import_module('carla')", "carla"),
        ("import importlib as loader\nloader.import_module('carla_adapter.world')", "carla_adapter.world"),
        ("from importlib import import_module as load\nload('car' + 'la')", "carla"),
        ("from importlib import import_module\nimport_module(name='carla')", "carla"),
        ("from importlib import import_module\nimport_module('.world', 'carla_adapter')", "carla_adapter.world"),
        ("__import__('carla')", "carla"),
        ("import builtins as b\nb.__import__('carla_adapter')", "carla_adapter"),
        ("from builtins import __import__ as load\nload('carla')", "carla"),
        ("import importlib\nimportlib.import_module(backend)", "<unresolved dynamic import>"),
        ("__import__('world', globals(), locals(), [], 1)", "<unresolved relative dynamic import>"),
    ],
)
def test_ast_rejects_simulator_dependencies(tmp_path, code, target):
    write_source(tmp_path, "autonomy/leak.py", code + "\n")
    violations = boundary_violations(tmp_path)
    assert any(f"autonomy.leak -> {target}" in violation for violation in violations)
    assert all("leak.py:" in violation for violation in violations)


def test_relative_import_and_indirect_shim_cannot_hide_adapter(tmp_path):
    write_source(tmp_path, "autonomy/stack.py", "from .shim import state\n")
    write_source(tmp_path, "autonomy/shim.py", "from carla_adapter.vehicle import state\n")
    violations = boundary_violations(tmp_path)
    assert any("autonomy.shim -> carla_adapter.vehicle" in violation for violation in violations)


def test_invalid_relative_import_fails_closed(tmp_path):
    write_source(tmp_path, "autonomy/shim.py", "from ..carla_adapter import state\n")
    assert any("<invalid relative import>" in item for item in boundary_violations(tmp_path))


@pytest.mark.parametrize(
    "path,code",
    [
        ("carla_adapter/vehicle.py", "import carla\n"),
        ("demo_combined_perception.py", "import carla\n"),
        ("demo_fused_lane_following.py", "import carla\n"),
        ("autonomy/clean.py", '"""import carla; from carla_adapter import state"""\n# import carla\n'),
        ("autonomy/clean.py", "import carla_tools\nfrom carla_adapter_utils import state\n"),
        ("autonomy/clean.py", "from .interfaces import FrameBundle\n"),
        ("autonomy/clean.py", "import importlib as loader\nloader.import_module('.interfaces', __package__)\n"),
    ],
)
def test_ast_allows_adapter_and_plain_core(tmp_path, path, code):
    write_source(tmp_path, path, code)
    assert boundary_violations(tmp_path) == []


def test_source_tree_obeys_boundary():
    assert boundary_violations(SRC_DIR) == []


def test_new_modules_and_namespace_packages_are_discovered(tmp_path):
    write_source(tmp_path, "new_core/nested.py", "import carla\n")
    write_source(tmp_path, "autonomy/__init__.py", "")
    assert set(source_modules(tmp_path)) == {"new_core.nested", "autonomy"}
    assert any("new_core.nested -> carla" in item for item in boundary_violations(tmp_path))


def test_all_core_modules_import_with_simulator_blocked(tmp_path):
    # A fresh interpreter prevents the old adapter tests' sys.modules fake from
    # masking a leak, and a different cwd tests package imports rather than cwd.
    result = smoke_imports(SRC_DIR, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "core modules with simulator imports blocked" in result.stdout


@pytest.mark.parametrize("target", ["carla", "carla_adapter.world"])
@pytest.mark.parametrize("handler", ["ImportError", "Exception", "BaseException"])
def test_smoke_rejects_even_swallowed_optional_import(tmp_path, target, handler):
    write_source(
        tmp_path,
        "leak.py",
        f"try:\n    __import__({target!r})\nexcept {handler}:\n    pass\n",
    )
    result = smoke_imports(tmp_path)
    assert result.returncode != 0
    assert "Forbidden simulator import" in result.stderr
    assert target.split(".")[0] in result.stderr


def test_smoke_follows_indirect_module_dependency(tmp_path):
    write_source(tmp_path, "core.py", "import shim\n")
    write_source(tmp_path, "shim.py", "import carla_adapter\n")
    result = smoke_imports(tmp_path, modules=["core"])
    assert result.returncode != 0
    assert "Forbidden simulator import: carla_adapter" in result.stderr


@pytest.mark.parametrize("target", ["carla", "carla_adapter.world"])
@pytest.mark.parametrize("filename", ["src/autonomy/leak.py", "tests/unit/test_new_module.py"])
def test_ruff_rule_rejects_banned_import(target, filename):
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--config", str(REPO_ROOT / "pyproject.toml"),
         "--select", "TID251", "--stdin-filename", filename, "-"],
        input=f"import {target}\n",
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "TID251" in result.stdout


@pytest.mark.parametrize(
    "filename,code",
    [
        ("src/carla_adapter/vehicle.py", "import carla\n"),
        ("examples/run_carla_demo.py", "import carla\n"),
        ("tests/unit/test_vehicle_geometry.py", "import carla_adapter.vehicle\n"),
    ],
)
def test_ruff_allowlist_is_limited_to_known_adapter_consumers(filename, code):
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--config", str(REPO_ROOT / "pyproject.toml"),
         "--select", "TID251", "--stdin-filename", filename, "-"],
        input=code,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_returns_failure_for_ast_violation(tmp_path):
    # Load the actual checker against a temporary source tree; do not change
    # repository code to demonstrate a failing CI command.
    write_source(tmp_path, "src/autonomy/leak.py", "import carla\n")
    code = (
        "from pathlib import Path\n"
        "import scripts.check_simulator_boundary as checker\n"
        f"checker.REPO_ROOT = Path({str(tmp_path)!r})\n"
        "raise SystemExit(checker.main())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "autonomy.leak -> carla" in result.stderr
