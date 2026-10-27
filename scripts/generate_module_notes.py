"""Generate a deterministic source inventory without importing ML or CARLA.

Module docstrings supply purpose; public definitions and dependencies show the
source surface. Missing purpose notes stay visible for owner review.
"""
import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/module_notes.md"


PURPOSE_NOTES = {
    "traffic_perception/__init__.py": "Exports shared traffic contracts and backend entry points so callers use a stable package surface.",
    "traffic_perception/_validation.py": "Validates numeric and image inputs centrally so invalid data is rejected consistently before perception work.",
    "traffic_perception/detector.py": "Converts injected backend predictions into traffic detections; backend injection separates model choice from contract validation.",
    "traffic_perception/distance.py": "Provides legacy pinhole distance and aligned-depth delegation; validity diagnostics distinguish unknown depth from clearance.",
    "traffic_perception/types.py": "Defines traffic categories and validated detection records; the legacy box constructor reuses the canonical XYXY contract.",
    "perception/lane_inference.py": "Loads a lane checkpoint and predicts masks from RGB frames; device selection permits CPU and supported accelerators without changing callers.",
    "perception/lane_segmentation.py": "Defines the ResNet18-based lane encoder/decoder; skip connections combine coarse context with spatial detail.",
    "perception/lane_path.py": "Cleans lane masks and extracts ego-lane geometry; filtering and gap closing reduce isolated mask noise before path construction.",
    "autodrive/perception/losses.py": "Defines focal and soft Dice training losses to address difficult pixels and mask overlap alongside classification loss.",
    "autodrive/perception/lane_dataset.py": "Loads paired lane images and masks for training; coordinated resizing and tensor conversion keep labels aligned with input pixels.",
}


def render(root):
    lines = ["# Module notes and source surface - DOC-05", "",
             "Generated from Python source without importing models or simulator libraries.",
             "Purpose text comes from module docstrings or curated source notes. An inventory is not a review of",
             "every design rationale; missing purpose notes require owner follow-up.", "",
             "## Public protocol and data flow", "",
             "The adapter converts simulator state and sensor data into plain contracts.",
             "Perception supplies frame-associated observations; autonomy checks freshness",
             "and operating limits; planning supplies control targets; the adapter applies",
             "commands. The offline harness provides a separate world implementation.",
             "See [interfaces](interfaces.md), [detector contracts](detector_contracts.md),",
             "[simulator boundary](simulator_boundary.md) and [API](api.rst).", ""]
    for path in sorted((root / "src").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        doc = ast.get_docstring(tree)
        purpose = " ".join(doc.splitlines()[0].split()) if doc else PURPOSE_NOTES.get(
            path.relative_to(root / "src").as_posix(), "PURPOSE NOTE PENDING owner review.")
        public = [node.name for node in tree.body
                  if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                  and not node.name.startswith("_")]
        dependencies = sorted({node.module or "(relative)" for node in tree.body
                               if isinstance(node, ast.ImportFrom)} |
                              {alias.name for node in tree.body if isinstance(node, ast.Import)
                               for alias in node.names})
        lines += [f"## `{path.relative_to(root).as_posix()}`", "", purpose, "",
                  "Public definitions: " + (", ".join(f"`{n}`" for n in public) or "none at module level") + ".", "",
                  "Direct module imports: " + (", ".join(f"`{n}`" for n in dependencies) or "none") + ".", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render(ROOT)
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != expected:
            print("FAIL: module notes are stale; run python scripts/generate_module_notes.py")
            return 1
        print("PASS: module notes match source")
        return 0
    OUTPUT.write_text(expected, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
