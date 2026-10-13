"""Repository locations that do not depend on the current working directory.

Use these instead of relative paths such as ``"../segmentation_model.pth"``,
so code behaves the same when run from the repository root, ``src/``,
``scripts/``, the test suite or a notebook::

    from autonomy.paths import config_path, model_path, run_dir

    weights = model_path("segmentation")
    stack_cfg = config_path("stack")          # configs/stack.yaml
    s0 = config_path("scenarios/S0")          # configs/scenarios/S0.yaml
    out = run_dir("S0", "2026-10-20T10-00-00")  # runs/S0/2026-10-20T10-00-00/

Core modules (perception, planning, ...) should keep taking paths as
arguments. Entry points, config loaders and scripts resolve defaults here.

Environment overrides: ``A2_REPO_ROOT`` (repository root) and
``A2_RUNS_DIR`` (where run outputs go).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "CONFIG_DIR",
    "DOCS_DIR",
    "FIXTURES_DIR",
    "MODEL_FILES",
    "REPO_ROOT",
    "RUNS_DIR",
    "SCENARIO_DIR",
    "SCRIPTS_DIR",
    "SRC_DIR",
    "config_path",
    "model_path",
    "run_dir",
]


def _find_repo_root() -> Path:
    override = os.environ.get("A2_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    raise RuntimeError(
        "Could not locate the repository root from autonomy/paths.py. "
        "Install with `pip install -e .` from a clone, or set A2_REPO_ROOT."
    )


REPO_ROOT = _find_repo_root()
SRC_DIR = REPO_ROOT / "src"
CONFIG_DIR = REPO_ROOT / "configs"
SCENARIO_DIR = CONFIG_DIR / "scenarios"
SCRIPTS_DIR = REPO_ROOT / "scripts"
DOCS_DIR = REPO_ROOT / "docs"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
RUNS_DIR = Path(os.environ.get("A2_RUNS_DIR", REPO_ROOT / "runs")).expanduser()

#: Model weights shipped in the repository, by short name. BL-C-03 replaces
#: this table with ``configs/models.yaml`` (checksums, versions); keep the
#: short names stable so callers do not change.
MODEL_FILES = {
    "lane": "lane_unet_best.pth",
    "segmentation": "segmentation_model.pth",
    "yolo": "yolo26n.pt",
}

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def config_path(name: str) -> Path:
    """Return ``configs/<name>``, adding ``.yaml`` when no suffix is given.

    The file is not required to exist yet. Absolute paths are returned
    unchanged so callers can pass a user-supplied config through.
    """
    path = Path(name)
    if path.is_absolute():
        return path
    if not path.suffix:
        path = path.with_suffix(".yaml")
    return CONFIG_DIR / path


def model_path(name: str, root: Path | None = None) -> Path:
    """Return the path to a model file by short name or file name.

    Args:
        name: A key of :data:`MODEL_FILES` (``"lane"``, ``"segmentation"``,
            ``"yolo"``) or a file name relative to the repository root.
        root: Directory to look in. Defaults to :data:`REPO_ROOT`.

    Raises:
        FileNotFoundError: If the file is missing, or if it is still a Git
            LFS pointer because ``git lfs pull`` has not been run.
    """
    base = REPO_ROOT if root is None else Path(root)
    path = base / MODEL_FILES.get(name, name)
    if not path.is_file():
        known = ", ".join(sorted(MODEL_FILES))
        raise FileNotFoundError(f"Model file not found: {path} (known names: {known})")
    with path.open("rb") as handle:
        head = handle.read(len(_LFS_POINTER_PREFIX))
    if head == _LFS_POINTER_PREFIX:
        raise FileNotFoundError(
            f"{path.name} is a Git LFS pointer, not the weights. "
            "Run `git lfs install` and `git lfs pull` in the repository."
        )
    return path


def run_dir(scenario: str, run_id: str, create: bool = True) -> Path:
    """Return ``runs/<scenario>/<run_id>/`` for one run's outputs.

    With ``create=True`` the directory is created and must not already exist,
    so an earlier run is never overwritten.
    """
    path = RUNS_DIR / scenario / run_id
    if create:
        path.mkdir(parents=True, exist_ok=False)
    return path
