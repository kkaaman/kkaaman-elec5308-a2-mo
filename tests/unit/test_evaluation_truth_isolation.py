"""BL-B-06/07: evaluation truth cannot feed obstacle-control decisions.

Two structural guarantees, checked statically:
1. No perception, planning or control package imports ``evaluation`` (where the
   scenario truth, stop-margin truth and event counting live) or
   ``carla_adapter`` (where the live truth is read). This holds however the
   import is written, including literal ``importlib.import_module`` calls.
2. The only input to ``AutonomyStack.step`` (protocol and skeleton) is a ``FrameBundle``, and no
   ``FrameBundle`` field carries truth, events or scenario data.
"""

import ast
import dataclasses
import inspect
from pathlib import Path

from autonomy.interfaces import AutonomyStack, FrameBundle
from autonomy.stack import SkeletonStack

SRC = Path(__file__).resolve().parents[2] / "src"
CONTROL_PATH = (
    "autonomy",
    "geometry",
    "navigation_control",
    "obstacle_perception",
    "perception",
    "planning",
    "segmentation",
    "traffic_perception",
)
FORBIDDEN = ("evaluation", "carla_adapter", "sim_harness")


def imported_roots(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module.split(".")[0]
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    yield node.lineno, arg.value.split(".")[0]


def test_control_path_never_imports_evaluation_truth():
    offenders = []
    for package in CONTROL_PATH:
        for path in sorted((SRC / package).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            offenders += [
                f"{path.relative_to(SRC)}:{line} imports {root}"
                for line, root in imported_roots(tree)
                if root in FORBIDDEN
            ]
    assert offenders == []


def test_import_scan_sees_every_form():
    tree = ast.parse(
        "import evaluation.stop_truth\n"
        "from evaluation import scenarios\n"
        "import importlib\n"
        "importlib.import_module('carla_adapter.scenario_world')\n"
    )
    assert {root for _, root in imported_roots(tree)} >= {"evaluation", "carla_adapter"}


def test_stack_input_carries_no_truth():
    names = {field.name for field in dataclasses.fields(FrameBundle)}
    assert not {
        n for n in names if any(word in n for word in ("truth", "event", "scenario", "obstacle"))
    }
    for stack in (AutonomyStack, SkeletonStack):
        assert list(inspect.signature(stack.step).parameters) == ["self", "frame"]
    assert inspect.signature(AutonomyStack.step).parameters["frame"].annotation in (
        FrameBundle,
        "FrameBundle",
    )
