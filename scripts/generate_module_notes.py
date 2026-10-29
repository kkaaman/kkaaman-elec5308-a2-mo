"""Generate DOC-05 source inventory without importing models or simulator code."""

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PURPOSE = {
    "autodrive": "Preserve inherited training/import entry points while canonical perception modules evolve.",
    "autonomy": "Keep frame, detector and command contracts independent of the world implementation.",
    "carla_adapter": "Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.",
    "cloud": "Reserve the optional E4 backend namespace; its existence is not evidence of a measured extension.",
    "evaluation": "Keep recorded observations and metric analysis separate from controller inputs.",
    "geometry": "Share explicit optical/ego/world transforms rather than duplicate coordinate assumptions.",
    "navigation_control": "Separate path geometry, lateral control and speed control with physical-unit contracts.",
    "obstacle_perception": "Turn sensor detections/depth into explicit observations without introducing evaluation truth.",
    "perception": "Expose RGB perception and deterministic fixtures through simulator-independent APIs.",
    "planning": "Use explicit graph connectivity and reusable routes rather than simulator objects in control.",
    "segmentation": "Separate preprocessing, learned architecture, inference and display postprocessing.",
    "sim_harness": "Exercise world/stack contracts without CARLA; synthetic fixtures do not prove live performance.",
    "traffic_perception": "Keep traffic detections and calibrated distance calculations independent of simulator APIs.",
    "visualisation": "Reserve the visualisation namespace; existing overlays still live in legacy helper modules.",
}
FALLBACK = {
    "autodrive/perception/lane_dataset.py": "Load paired road images and lane masks for training.",
    "autodrive/perception/losses.py": "Combine segmentation losses for the inherited lane training pipeline.",
    "perception/lane_inference.py": "Load the local lane checkpoint and convert RGB frames into lane masks.",
    "perception/lane_path.py": "Extract an ego-lane path from a predicted lane mask.",
    "perception/lane_segmentation.py": "Define the lane segmentation network used by the checkpoint loader.",
    "traffic_perception/__init__.py": "Export public traffic detection, distance and configuration types.",
    "traffic_perception/_validation.py": "Validate numeric and class inputs for the traffic API.",
    "traffic_perception/detector.py": "Filter and normalise object detections into traffic observations.",
    "traffic_perception/distance.py": "Estimate distance from calibrated image geometry and known object dimensions.",
    "traffic_perception/types.py": "Define traffic classes, bounding boxes, configuration and returned observations.",
}
HEADER = """Module notes and dataflow — DOC-05
==================================

Generated from the current ``src`` tree by ``scripts/generate_module_notes.py``.
Regenerate after adding or changing modules; ``--check`` fails on stale notes.
Roles come from source docstrings with explicit notes for older undocumented
modules. This inventory describes source responsibilities, not execution or
acceptance evidence. Public member documentation is in :doc:`api`; detailed
contract rules and review status are in :doc:`interfaces`.

Public protocol and dataflow
----------------------------

The world owns sensing and time. ``World.reset(seed)`` returns a ``FrameBundle``
and ``LaneGraph``. ``AutonomyStack.reset`` installs episode/graph state;
``AutonomyStack.step(frame)`` produces a source-bound ``ControlPacket``.
``World.step(packet)`` applies control and returns the next frame. Use the exact
signatures in :doc:`api`; protocol method presence alone does not validate
semantics. Close the world idempotently on success or failure.

``FrameBundle`` contains RGB, rear-axle pose, calibration, capture/receipt times
and optional aligned axial depth. The detector contract converts a frame to
source-bound detections; obstacle perception associates depth separately.
Planning creates paths consumed by ``navigation_control``. The current stack is
a validated skeleton; this description does not claim that every planned
perception/planning stage is integrated. Keep episode IDs, frame IDs, units and
clock domains intact. Only the adapter imports CARLA. Evaluation observations
are not perception inputs. The offline world supplies synthetic sensor fixtures,
not a validated model of collisions or live sensor performance.

Module inventory
----------------

"""


def render(root=ROOT):
    sections = [HEADER]
    for path in sorted((root / "src").rglob("*.py")):
        relative = path.relative_to(root / "src").as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        doc = ast.get_docstring(tree)
        role = " ".join(doc.split("\n\n", 1)[0].split()) if doc else FALLBACK.get(relative)
        if role is None:
            raise ValueError(f"Add a source docstring or explicit role for {relative}")
        family = relative.split("/", 1)[0]
        why = PURPOSE.get(family)
        if why is None:
            if family.startswith("demo_"):
                why = "Compose inherited live demonstration components at an explicit simulator entry point."
            else:
                why = "Keep the inherited helper/import interface available while reusable packages own new work."
        symbols = [node.name for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                   and not node.name.startswith("_")]
        sections.extend([relative + "\n", "~" * len(relative) + "\n\n",
                         "Role (source summary)::\n\n    " + role + "\n\n",
                         "Why: " + why + "\n\n"])
        if symbols:
            sections.append("Declared public classes/functions: "
                            + ", ".join(f"``{name}``" for name in symbols) + ".\n\n")
        else:
            sections.append("No public top-level classes/functions declared; see exports/source.\n\n")
    return "".join(sections).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "docs/module_notes.rst"
    content = render()
    if args.check:
        if not destination.exists() or destination.read_text(encoding="utf-8") != content:
            parser.exit(1, "Module notes are stale; run python scripts/generate_module_notes.py\n")
        print("Module notes match the current src tree")
    else:
        destination.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
