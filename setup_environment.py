"""One-shot dependency installer for the perception + navigation_control
demos (demo_combined_perception.py, demo_fused_lane_following.py,
test_navigation_control_carla.py, etc).

Run this from your ACTIVE Python environment (e.g. with `myenv` already
activated) -- it installs into whichever interpreter runs it (via
`sys.executable -m pip`), it does not create or activate an environment
for you.

Usage:
    python setup_environment.py --offline --check-only
    python setup_environment.py --offline --wheelhouse ./wheels --lock-file ./runtime-lock.txt
    python setup_environment.py --carla-wheel /path/to/carla-0.9.16-cp312-cp312-win_amd64.whl
    python setup_environment.py --carla-version 0.9.15   # falls back to PyPI if no wheel path given
    python setup_environment.py --cuda-tag cu121
    python setup_environment.py --skip-carla --skip-torch   # re-run just the rest

What this does, and why each step exists:
    1. Checks the running Python version. The CARLA demos were tested on
       Python 3.12.
    2. Removes opencv-python-headless if present, then installs the project
       from pyproject.toml in editable mode with the test/docs extras.
    3. Installs torch + torchvision using the requested CUDA build and
       verifies that torch can access the GPU.
    4. Installs the CARLA Python API, preferring a compatible local wheel
       where available.
    5. Runs git lfs pull so the tracked model-weight files are the real
       binaries rather than Git LFS pointer files.
    6. Prints a final checklist for CARLA version matching, model weights,
       repository verification and running the live demos.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PYPROJECT_FILE = Path(__file__).parent / "pyproject.toml"

# CARLA switched from `carla==X.Y.Z` to a `carla-X-Y-Z`-named PyPI package
# for 0.9.16 (the classic name only has wheels through 0.9.15 as of this
# script's writing). Both are tried in order below regardless of version,
# so this list is just which to try FIRST for a known version -- if
# CARLA's PyPI naming changes again after this script was written, the
# fallback attempt still gives a clear error to act on instead of a
# silent wrong guess.
CLASSIC_NAMING_VERSIONS = {"0.9.15", "0.9.14", "0.9.13", "0.9.12", "0.9.11", "0.9.10"}


def run(cmd: list[str], description: str) -> bool:
    """Run a subprocess command, print what's happening, and report
    success/failure without raising -- one failed step should not stop
    every other independent step from still being attempted.
    """
    print(f"\n--- {description} ---")
    print(f"$ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd)
    except OSError as exc:
        print(f"[FAILED] {description}: {exc}")
        return False
    if result.returncode != 0:
        print(f"[FAILED] {description} (exit code {result.returncode})")
        return False
    print(f"[OK] {description}")
    return True


def pip_install(args: list[str], description: str) -> bool:
    return run([sys.executable, "-m", "pip", "install", *args], description)


def check_python_version() -> None:
    major, minor = sys.version_info[:2]
    print(f"Running under Python {major}.{minor} ({sys.executable})")
    if (major, minor) != (3, 12):
        print(
            f"[WARNING] The CARLA demos were tested with Python 3.12; "
            f"you're running {major}.{minor}. CARLA 0.9.16 supports "
            f"3.10/3.11/3.12, so this may still work, but if package "
            f"resolution fails below, that's the first thing to check."
        )


def remove_conflicting_opencv() -> bool:
    """opencv-python and opencv-python-headless both provide `cv2` --
    having both installed is a common cause of cv2.imshow failing with
    a "not implemented" error in the live demos, since pip doesn't
    reliably pick the non-headless one when both are present.
    """
    check = subprocess.run(
        [sys.executable, "-m", "pip", "show", "opencv-python-headless"],
        capture_output=True,
    )
    if check.returncode == 0:
        print("\nFound opencv-python-headless installed -- removing it")
        print("(the live demos need the real GUI build of opencv, not headless).")
        return run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python-headless"],
            "Removing opencv-python-headless",
        )
    else:
        print("\nNo opencv-python-headless found -- nothing to remove.")
    return True


def install_requirements() -> bool:
    """Install the project from pyproject.toml in editable mode."""
    if not PYPROJECT_FILE.exists():
        print(f"[FAILED] {PYPROJECT_FILE} not found next to this script.")
        return False
    
    return pip_install(
        ["-e", f"{PYPROJECT_FILE.parent}[test,docs]"],
        "Installing project (pyproject.toml, editable, with test/docs extras)",
    )


def detect_cuda_tag() -> str | None:
    """Best-effort read of the NVIDIA driver's max supported CUDA version
    via `nvidia-smi`, purely to print for your own reference -- this
    script does NOT parse it into an exact PyTorch index tag, since that
    mapping changes over time and getting it wrong fails silently at
    `import torch` rather than at install time. Use the printed number
    to sanity-check the --cuda-tag you pass in (or the default below).
    """
    if shutil.which("nvidia-smi") is None:
        print(
            "\n[WARNING] nvidia-smi not found on PATH -- can't confirm your "
            "driver's CUDA version. If the GPU torch install below fails or "
            "torch.cuda.is_available() comes back False, check your NVIDIA "
            "driver is installed and reachable."
        )
        return None
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print("\n--- nvidia-smi output (for reference) ---")
    print(result.stdout)
    return None  # intentionally not auto-parsed -- see docstring above


def torch_cuda_available() -> bool | None:
    """Import torch in a fresh subprocess (not this process) and report
    whether it sees CUDA. Returns None if torch isn't importable at all
    yet -- distinct from False, which means torch imported fine but is a
    CPU-only build or the driver isn't reachable.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() == "True"


def install_torch(cuda_tag: str) -> bool:
    """Install torch + torchvision for CUDA and verify GPU access.

    The CUDA-specific torch build is installed before the project
    dependencies because ultralytics depends on torch. Installing the
    project first could allow pip to satisfy that dependency with a
    CPU-only build from the default PyPI index.

    If an incompatible or CPU-only torch installation already exists,
    this function attempts a forced reinstall from the selected CUDA
    index.
    """
    print(
        f"\nInstalling torch + torchvision for CUDA tag '{cuda_tag}'.\n"
        f"If this fails, or torch.cuda.is_available() is False afterwards, "
        f"the CUDA tag likely doesn't match your driver -- check the "
        f"'CUDA Version' line in the nvidia-smi output above against "
        f"https://pytorch.org/get-started/locally/ and re-run with e.g.:\n"
        f"    python setup_environment.py --skip-carla --cuda-tag cu121\n"
    )
    ok = pip_install(
        ["torch", "torchvision", "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}"],
        f"Installing torch + torchvision ({cuda_tag})",
    )
    if not ok:
        return False

    print("\nVerifying torch actually sees the GPU (installing successfully isn't proof of that)...")
    if torch_cuda_available():
        print("[OK] torch.cuda.is_available() -> True")
        return True

    print(
        "[WARNING] torch installed but torch.cuda.is_available() is False. "
        "This usually means a CPU-only torch was already present (e.g. "
        "pulled in as an unpinned dependency by another package) and the "
        "plain install above saw it as 'already satisfied' and skipped "
        "replacing it. Forcing a real reinstall from the CUDA index..."
    )
    ok = pip_install(
        [
            "torch", "torchvision", "--index-url",
            f"https://download.pytorch.org/whl/{cuda_tag}",
            "--force-reinstall", "--no-cache-dir",
        ],
        f"Force-reinstalling torch + torchvision ({cuda_tag})",
    )
    if not ok:
        return False

    if torch_cuda_available():
        print("[OK] torch.cuda.is_available() -> True (after force-reinstall)")
        return True

    print(
        "[FAILED] torch.cuda.is_available() is still False after a forced "
        "CUDA reinstall. At this point it's very likely the --cuda-tag "
        f"('{cuda_tag}') genuinely doesn't match your driver -- re-check "
        "the nvidia-smi 'CUDA Version' line above against "
        "https://pytorch.org/get-started/locally/ and re-run with the "
        "correct --cuda-tag, e.g.:\n"
        "    python setup_environment.py --skip-carla --skip-lfs --cuda-tag cu121"
    )
    return False


WHEEL_FILENAME_RE = re.compile(
    r"^carla-(?P<version>[\d.]+)-cp(?P<pytag>\d+)-cp\d+.*\.whl$"
)


def find_carla_wheel(version: str) -> Path | None:
    """Search common locations for a carla-<version>-*.whl matching the
    requested version, so most people don't need to pass --carla-wheel
    by hand -- CARLA's own packaging always puts it at
    <extracted CARLA folder>/PythonAPI/carla/dist/carla-X.Y.Z-*.whl.
    Only searches shallow, known spots (home dir's PythonAPI folder, and
    the current directory) rather than a full-disk recursive search,
    which would be slow and could match an unrelated stale wheel from
    somewhere unexpected.
    """
    candidates = [
        Path.home() / "PythonAPI" / "carla" / "dist",
        Path.cwd() / "PythonAPI" / "carla" / "dist",
    ]
    for folder in candidates:
        if not folder.is_dir():
            continue
        for wheel in folder.glob(f"carla-{version}-*.whl"):
            return wheel
    return None


def check_wheel_python_tag(wheel_path: Path) -> bool:
    """Parse the wheel filename's cpXY tag and compare it against the
    RUNNING interpreter -- a mismatch here (e.g. a cp312 wheel under a
    cp313 interpreter) fails outright at install time with no useful
    recovery, so it's worth a clear, specific message before even
    attempting it rather than surfacing pip's generic error.
    """
    match = WHEEL_FILENAME_RE.match(wheel_path.name)
    if not match:
        print(
            f"[WARNING] Couldn't parse a Python tag out of '{wheel_path.name}' "
            f"-- proceeding anyway, but if install fails, check this by hand."
        )
        return True

    wheel_py = int(match.group("pytag"))  # e.g. 312 for cp312
    running_py = sys.version_info.major * 100 + sys.version_info.minor
    if wheel_py != running_py:
        wheel_py_str = f"{wheel_py // 100}.{wheel_py % 100}"
        running_py_str = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(
            f"\n[FAILED] '{wheel_path.name}' is built for Python {wheel_py_str}, "
            f"but this script is running under Python {running_py_str} "
            f"({sys.executable}).\n"
            f"This WILL fail if attempted -- a wheel's Python tag has to match "
            f"the installing interpreter exactly. Activate the Python "
            f"{wheel_py_str} environment this wheel needs (e.g. `myenv`, if "
            f"that's your 3.12 environment) and re-run this script from there."
        )
        return False
    return True


def install_carla_from_wheel(wheel_path: Path) -> bool:
    if not wheel_path.exists():
        print(f"[FAILED] Wheel path does not exist: {wheel_path}")
        return False
    if not check_wheel_python_tag(wheel_path):
        return False
    return pip_install([str(wheel_path)], f"Installing CARLA Python API from {wheel_path.name}")


def install_carla(version: str, wheel_path: Path | None) -> bool:
    """Install the carla Python API, preferring a local wheel (CARLA's
    own documented method, and the only one confirmed to work on
    Windows for 0.9.16 -- see this module's docstring). Only falls back
    to PyPI naming-scheme guesses if no wheel is given or found.
    """
    if wheel_path is None:
        wheel_path = find_carla_wheel(version)
        if wheel_path is not None:
            print(f"\nFound a local CARLA wheel: {wheel_path}")

    if wheel_path is not None:
        match = WHEEL_FILENAME_RE.match(wheel_path.name)
        if match is None or match.group("version") != version:
            print(f"[FAILED] Expected a carla-{version} wheel; got {wheel_path.name}")
            return False
        if install_carla_from_wheel(wheel_path):
            return True
        return False

    classic_name = f"carla=={version}"
    hyphen_name = "carla-" + version.replace(".", "-")

    order = (
        [classic_name, hyphen_name]
        if version in CLASSIC_NAMING_VERSIONS
        else [hyphen_name, classic_name]
    )

    for package in order:
        print(f"\nTrying: pip install {package}")
        if pip_install([package], f"Installing CARLA Python API ({package})"):
            return True
        print(f"'{package}' didn't work -- trying the other naming scheme.")

    print(
        f"\n[FAILED] Could not install a carla package for version {version} "
        f"under either naming scheme, and no working local wheel was found/given. "
        f"Locate carla-{version}-*.whl inside your extracted CARLA package's "
        f"PythonAPI/carla/dist/ folder and pass it explicitly with "
        f"--carla-wheel <path>."
    )
    return False


def pull_lfs_weights() -> bool:
    """Run git lfs pull if we're inside a git repo with git-lfs available,
    so *.pth weight files are the real trained weights rather than LFS
    pointer stubs -- installing packages successfully does NOT guarantee
    this; it's a separate, easy-to-miss step.
    """
    if shutil.which("git") is None:
        print(
            "\n[SKIPPED] git not found on PATH -- can't check/pull LFS "
            "weight files automatically. If segmentation_model.pth or "
            "lane_unet_best.pth are small text-looking files (a few hundred "
            "bytes) rather than real weight files, that's an LFS pointer "
            "stub, not the actual weights -- install git and git-lfs, then "
            "run `git lfs pull` yourself from inside the repo."
        )
        return False

    inside_repo = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], capture_output=True
    )
    if inside_repo.returncode != 0:
        print(
            "\n[SKIPPED] Not inside a git repository from this script's "
            "location -- run `git lfs pull` yourself from your repo root "
            "once cloned, to fetch the real .pth weight files."
        )
        return False

    if shutil.which("git-lfs") is None:
        # `git lfs` is a git subcommand, not always a separate `git-lfs`
        # binary on PATH -- try the subcommand directly before giving up.
        lfs_check = subprocess.run(["git", "lfs", "version"], capture_output=True)
        if lfs_check.returncode != 0:
            print(
                "\n[SKIPPED] git-lfs doesn't appear to be installed. Install "
                "it from https://git-lfs.com, then run `git lfs pull` from "
                "your repo root -- otherwise segmentation_model.pth and "
                "lane_unet_best.pth will just be small pointer stubs and "
                "every demo will fail loading weights with a torch error."
            )
            return False

    if not run(["git", "lfs", "install"], "Setting up git-lfs for this repo"):
        return False
    return run(["git", "lfs", "pull"], "Pulling LFS weight files (segmentation_model.pth, lane_unet_best.pth)")


def print_manual_checklist(args: argparse.Namespace) -> None:
    print(
        "\n" + "=" * 70
        + "\nInstallation steps finished; environment validation is still required:"
        + "\n" + "=" * 70
        + "\n1. CARLA CLIENT/SERVER VERSION MATCH\n"
        + "   Confirm the running CarlaUE4.exe matches client version "
        + f"{args.carla_version}.\n"
        + "\n2. MODEL WEIGHTS\n"
        + "   Confirm yolo26n.pt, segmentation_model.pth and "
        + "lane_unet_best.pth are present and contain the real weights.\n"
        + "\n3. VERIFY THE REPOSITORY\n"
        + "   From the repository root, run:\n"
        + "       ruff check .\n"
        + "       python -m pytest\n"
        + "       sphinx-build -W docs docs/_build/html\n"
        + "\n4. LIVE DEMO\n"
        + "   Start CARLA before running either demo from src/.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carla-version", default="0.9.16")
    parser.add_argument(
        "--carla-wheel", type=Path, default=None,
        help="Path to a local carla-X.Y.Z-cpNN-*.whl (CARLA's own documented "
             "install method -- found inside your extracted CARLA package's "
             "PythonAPI/carla/dist/ folder). If omitted, this script looks "
             "for one automatically under your home directory and the "
             "current directory before falling back to PyPI.",
    )
    parser.add_argument(
        "--cuda-tag", default="cu124",
        help="PyTorch CUDA index tag, e.g. cu121/cu124/cu128 -- see nvidia-smi "
             "output this script prints, and https://pytorch.org/get-started/locally/",
    )
    parser.add_argument("--skip-carla", action="store_true")
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--skip-lfs", action="store_true")
    parser.add_argument("--check-only", action="store_true",
                        help="Read-only asset and recorded-runtime checks; installs nothing")
    parser.add_argument("--assets-only", action="store_true",
                        help="With --check-only: verify binaries without checking installed packages")
    parser.add_argument("--offline", action="store_true",
                        help="No network: check-only or install from an explicit local wheelhouse")
    parser.add_argument("--wheelhouse", type=Path,
                        help="Prepared local wheels for the active Python/platform")
    parser.add_argument("--lock-file", type=Path,
                        help="Offline pip requirements: full pinned transitive lock with hashes")
    args = parser.parse_args()
    root = PYPROJECT_FILE.resolve().parent
    if args.assets_only and not args.check_only:
        parser.error("--assets-only requires --check-only")
    if args.check_only:
        command = [sys.executable, str(root / "scripts/check_reproduction.py")]
        if args.assets_only:
            command.append("--assets-only")
        return 0 if run(command, "Read-only reproduction preflight") else 1
    if args.offline:
        if not args.wheelhouse or not args.wheelhouse.is_dir():
            parser.error("--offline installation needs --wheelhouse pointing to local wheels")
        if not args.lock_file or not args.lock_file.is_file():
            parser.error("--offline installation needs --lock-file with pinned packages and hashes")
        # --no-index alone does not block direct URL requirements or nested -r files.
        # Accept a deliberately narrow local hashed-lock format before running pip.
        lock_text = args.lock_file.read_text(encoding="utf-8")
        logical_lines = lock_text.replace("\\\n", " ").splitlines()
        package_line = re.compile(
            r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_,.-]+\])?"
            r"==[A-Za-z0-9_.+!-]+(?:\s+--hash=sha256:[a-fA-F0-9]{64})+$"
        )
        entries = [line.strip() for line in logical_lines
                   if line.strip() and not line.lstrip().startswith("#")]
        if not entries or any(package_line.fullmatch(line) is None for line in entries):
            parser.error("Offline lock must contain only name==version entries with SHA-256 hashes; "
                         "URL/VCS/options/nested requirements are not allowed")
        local = ["--isolated", "--no-index", "--only-binary=:all:",
                 "--find-links", str(args.wheelhouse.resolve())]
        if not pip_install([*local, "--no-deps", "--require-hashes", "-r", str(args.lock_file.resolve())],
                           "Installing the prepared offline runtime"):
            return 1
        # No build isolation: build requirements must also be included in the lock/wheelhouse.
        if not pip_install([*local, "--no-deps", "--no-build-isolation", "-e", str(root)],
                           "Installing local project without dependency resolution"):
            return 1
        if not run([sys.executable, "-m", "pip", "check"], "Checking installed dependency consistency"):
            return 1
        print("Offline install finished; no LFS pull was attempted. Run --offline --check-only.")
        print("CARLA, CUDA and inference require the separate checks in docs/reproduction.md.")
        return 0

    check_python_version()
    if not remove_conflicting_opencv():
        return 1

    # Install torch/torchvision before the project dependencies so
    # ultralytics does not accidentally satisfy the torch dependency
    # with a CPU-only build from the default PyPI index.
    detect_cuda_tag()
    if not args.skip_torch:
        if not install_torch(args.cuda_tag):
            return 1
    else:
        print("\n[SKIPPED] torch/torchvision (--skip-torch)")

    if not install_requirements():
        return 1

    if not args.skip_carla:
        if not install_carla(args.carla_version, args.carla_wheel):
            return 1
    else:
        print("\n[SKIPPED] carla (--skip-carla)")

    if not args.skip_lfs:
        if not pull_lfs_weights():
            return 1
    else:
        print("\n[SKIPPED] git-lfs weight pull (--skip-lfs)")

    print_manual_checklist(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
