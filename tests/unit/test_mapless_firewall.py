"""Truth firewall: the mapless car sees sensors and a sat-nav route, nothing else.

These are static checks over every module in ``src/mapless``:
- Imports are limited to an allow-list of the standard library, numerical packages and
  simulator-free core packages. The simulator, its adapter, the evaluator, the harness
  and the map tooling (planning, lane graphs, missions) are all outside it.
- No string literal names a map artefact or a town, so the car cannot load or branch on
  one.
- ``SensorFrame`` has exactly the reviewed fields. Adding one, for example a pose, fails
  here.
"""

import ast
import dataclasses
import sys
from pathlib import Path

import pytest

from mapless.frame import SensorFrame

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "mapless"

ALLOWED_INTERNAL = {
    "mapless",
    "autonomy.interfaces",
    "autonomy.localisation",
    "geometry",
    "navigation_control",
    "obstacle_perception",
    "perception",
    "segmentation",
    "traffic_perception",
}
ALLOWED_EXTERNAL = {"numpy", "cv2", "torch", "torchvision", "ultralytics", "scipy"}
FORBIDDEN_WORDS = (
    "town0",
    "town10",
    "lane_graph",
    "opendrive",
    ".xodr",
    "mission",
    "configs/missions",
    "fixtures",
    "carla",
)
FRAME_FIELDS = {
    "episode_id",
    "frame_id",
    "sim_time_s",
    "received_monotonic_s",
    "cameras",
    "lidars",
    "radars",
    "gnss",
    "imu",
    "speed_mps",
}


def modules():
    files = sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)
    assert files, "mapless package not found"
    return files


def imported(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                yield node.lineno, "mapless." + (node.module or "")
            else:
                yield node.lineno, node.module or ""
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in {"import_module", "__import__", "find_spec", "spec_from_file_location"}:
                yield node.lineno, "<dynamic import>"


def allowed(module):
    root = module.split(".")[0]
    if root in sys.stdlib_module_names or root in ALLOWED_EXTERNAL:
        return True
    return any(module == a or module.startswith(a + ".") for a in ALLOWED_INTERNAL)


@pytest.mark.parametrize("path", modules(), ids=lambda p: p.name)
def test_imports_stay_inside_the_allow_list(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = [(line, name) for line, name in imported(tree) if not allowed(name)]
    assert not bad, f"{path.name} imports outside the mapless allow-list: {bad}"


@pytest.mark.parametrize("path", modules(), ids=lambda p: p.name)
def test_no_map_or_town_names_in_string_literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    docstrings = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            text = node.value.lower()
            found += [(node.lineno, word) for word in FORBIDDEN_WORDS if word in text]
    assert not found, f"{path.name} names map artefacts: {found}"


def test_sensor_frame_fields_are_the_reviewed_set():
    assert {f.name for f in dataclasses.fields(SensorFrame)} == FRAME_FIELDS


def test_the_allow_list_rejects_what_it_should():
    for module in (
        "carla",
        "carla_adapter.sensor_rig",
        "evaluation.satnav_route",
        "sim_harness.baseline",
        "planning.mission",
        "planning.lane_graph",
        "autonomy.lane_localisation",
        "autonomy.baseline",
    ):
        assert not allowed(module), module
    for module in ("numpy", "math", "autonomy.interfaces", "geometry.camera", "mapless.frame"):
        assert allowed(module), module


def test_importing_the_package_pulls_in_nothing_forbidden():
    """Transitive check: import every mapless module in a fresh interpreter."""
    import subprocess

    names = [
        "mapless." + ".".join(p.relative_to(PACKAGE).with_suffix("").parts)
        for p in modules()
        if p.name != "__init__.py"
    ]
    code = (
        "import sys, importlib\n"
        f"for name in {names!r}: importlib.import_module(name)\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'carla', 'carla_adapter', 'evaluation', 'sim_harness', 'planning'})\n"
        "print(','.join(bad))\n"
    )
    env_path = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": env_path},
        check=True,
    )
    assert result.stdout.strip() == "", f"forbidden modules loaded: {result.stdout.strip()}"
