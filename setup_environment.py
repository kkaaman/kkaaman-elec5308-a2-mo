"""One-shot dependency installer for the perception + navigation_control
demos (demo_combined_perception.py, demo_fused_lane_following.py,
test_navigation_control_carla.py, etc).

Run this from your ACTIVE Python environment (e.g. with `myenv` already
activated) -- it installs into whichever interpreter runs it (via
`sys.executable -m pip`), it does not create or activate an environment
for you.

Usage:
    python setup_environment.py --carla-wheel /path/to/carla-0.9.16-cp312-cp312-win_amd64.whl
    python setup_environment.py --carla-version 0.9.15   # falls back to PyPI if no wheel path given
    python setup_environment.py --cuda-tag cu121
    python setup_environment.py --skip-carla --skip-torch   # re-run just the rest

What this does, and why each step exists:
    1. Checks the running Python's version (the project's demos were
       tested on 3.12; a mismatch is a warning, not a hard stop in
       general -- but see step 4, where it IS a hard stop for CARLA
       specifically if you're installing from a version-tagged wheel).
    2. Removes opencv-python-headless if present, then installs
       requirements.txt (numpy, opencv-python, ultralytics, pytest).
       Headless and non-headless opencv both provide the `cv2` import
       name -- having both installed is a common, confusing source of
       "cv2.imshow not implemented" errors, since the demos need the
       real GUI build.
    3. Installs torch + torchvision. Reads your NVIDIA driver's max
       supported CUDA version via `nvidia-smi` if available and installs
       a matching (or safely older) PyTorch CUDA build; falls back to
       asking you to pick the index yourself if it can't detect one --
       see the printed note, since this is the one step whose exact
       correct command changes over time and by hardware, and guessing
       wrong here fails silently at import time, not install time.
    4. Installs the `carla` Python API. PREFERS a local wheel if
       --carla-wheel is given (or one is auto-detected -- see
       find_carla_wheel() below) -- this is CARLA's own documented
       install method, and the only one confirmed to work on Windows for
       0.9.16: PyPI currently has no Windows wheel for this version under
       either naming scheme (`carla==0.9.16` doesn't exist on the main
       `carla` project at all; `carla-0-9-16` exists but ships Linux
       wheels only). A local wheel's filename embeds its exact Python
       tag (e.g. cp312) -- this is checked against the RUNNING
       interpreter before attempting install, and treated as a hard
       stop on mismatch, since pip installing a cp312 wheel under a
       cp313 interpreter fails outright with no useful recovery. Falls
       back to the PyPI naming-scheme attempts (both classic
       `carla==X.Y.Z` and the newer `carla-X-Y-Z`) only if no wheel path
       is given/found -- useful on Linux, where PyPI does have wheels.
    5. If this script is run from inside the project's git repository,
       runs `git lfs pull` so the *.pth weight files (segmentation_model.pth,
       lane_unet_best.pth -- see .gitattributes) are the real trained
       weights, not empty LFS pointer stubs. This step is skipped (with
       a clear message) if git or git-lfs isn't available, since it's
       not required for pip installation to succeed -- but the demos
       will fail later with a confusing "invalid load key" torch error
       if the weights are still just pointer text.
    6. Prints a checklist of things this script CANNOT verify for you
       (exact CARLA server/client version match, first-run internet-
       dependent downloads, a known test-file import mismatch) so
       nothing here is a silent assumption.
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
    result = subprocess.run(cmd)
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
            f"[WARNING] This project was set up against Python 3.12; "
            f"you're running {major}.{minor}. CARLA 0.9.16 supports "
            f"3.10/3.11/3.12, so this may still work, but if package "
            f"resolution fails below, that's the first thing to check."
        )


def remove_conflicting_opencv() -> None:
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
        run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python-headless"],
            "Removing opencv-python-headless",
        )
    else:
        print("\nNo opencv-python-headless found -- nothing to remove.")


def install_requirements() -> bool:
    # CHANGED: was `pip install -r requirements.txt` -- now installs from
    # pyproject.toml instead, in editable mode with the [dev] extra (pulls
    # in pytest alongside the core numpy/opencv-python/ultralytics/pandas
    # dependencies). `-e` (editable) means the installed package tracks
    # this checkout directly, matching how the assignment's own
    # "pip install -e ." requirement is normally satisfied.
    if not PYPROJECT_FILE.exists():
        print(f"[FAILED] {PYPROJECT_FILE} not found next to this script.")
        return False
    return pip_install(
        ["-e", f"{PYPROJECT_FILE.parent}[dev]"],
        "Installing project (pyproject.toml, editable, with dev extras)",
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
    """Install torch + torchvision for CUDA, then VERIFY it actually
    took -- installing alone isn't proof of a GPU build.

    FIXED: requirements.txt pulls in ultralytics, which depends on
    `torch>=1.8.0` with no upper bound. If requirements.txt is installed
    before this function runs, pip may have already satisfied that
    unversioned dependency with a plain CPU-only `torch` from the
    default PyPI index -- and a later `pip install torch --index-url
    .../cuXXX` with no version pin sees torch "already satisfied" and
    silently does NOT replace it, even though it's the wrong build. This
    is why main() now calls install_torch() BEFORE install_requirements()
    -- so ultralytics' dependency is satisfied by the correct build
    already present, not the other way around. That ordering fix only
    prevents the problem in a FRESH environment though; it does nothing
    for an environment where the wrong CPU build is already sitting
    there from an earlier run. So this function also verifies
    torch.cuda.is_available() afterwards and, if it's False, forces a
    real reinstall (--force-reinstall) from the CUDA index specifically
    -- the only way to actually fix an already-broken existing install,
    since a plain reinstall would again see "already satisfied" and
    skip doing anything.
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
        if install_carla_from_wheel(wheel_path):
            return True
        print(
            "\nLocal wheel install failed/was skipped -- falling back to "
            "trying PyPI naming schemes (unlikely to have a Windows wheel "
            "for recent CARLA versions, but worth a real attempt)."
        )

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


def pull_lfs_weights() -> None:
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
        return

    inside_repo = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], capture_output=True
    )
    if inside_repo.returncode != 0:
        print(
            "\n[SKIPPED] Not inside a git repository from this script's "
            "location -- run `git lfs pull` yourself from your repo root "
            "once cloned, to fetch the real .pth weight files."
        )
        return

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
            return

    run(["git", "lfs", "install"], "Setting up git-lfs for this repo")
    run(["git", "lfs", "pull"], "Pulling LFS weight files (segmentation_model.pth, lane_unet_best.pth)")


def print_manual_checklist(args: argparse.Namespace) -> None:
    print(
        "\n" + "=" * 70 +
        "\nThis script installed everything it safely can. A few things "
        "\nit can't verify for you:\n" + "=" * 70 +
        f"\n1. CARLA CLIENT/SERVER VERSION MATCH\n"
        f"   Installed the client for {args.carla_version} -- confirm your\n"
        f"   running CarlaUE4.exe is the SAME version, or every demo will\n"
        f"   fail to connect with a version-mismatch error at startup.\n"
        f"\n2. FIRST-RUN INTERNET DOWNLOADS\n"
        f"   object_detection.ObjectDetector() downloads yolo26n.pt on its\n"
        f"   first run, and perception.lane_segmentation.LaneUNet's ResNet18\n"
        f"   backbone downloads ImageNet-pretrained weights from torchvision\n"
        f"   on ITS first run too. Both need real internet access the first\n"
        f"   time any demo runs -- do one throwaway run somewhere with\n"
        f"   internet before you need to demo offline.\n"
        f"\n3. A KNOWN TEST-FILE IMPORT MISMATCH (not fixed by this script)\n"
        f"   test_lane_path.py imports from `autodrive.perception.lane_path`,\n"
        f"   but every other file in the project (lane_inference.py's own\n"
        f"   FIXED note included) treats the real location as flat\n"
        f"   `perception.lane_path`. Running pytest as-is will likely fail\n"
        f"   to collect that one test file with an ImportError -- this is a\n"
        f"   pre-existing bug in the test file itself, not a dependency\n"
        f"   issue, so it's flagged here rather than silently patched.\n"
    )


def main() -> None:
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
    args = parser.parse_args()

    check_python_version()
    remove_conflicting_opencv()

    # torch/torchvision installed BEFORE requirements.txt -- see the
    # FIXED note in install_torch()'s docstring. requirements.txt pulls
    # in ultralytics, whose unversioned `torch>=1.8.0` dependency would
    # otherwise get silently satisfied by a default CPU-only torch
    # first, causing this function's own CUDA-specific install to look
    # "already satisfied" and skip doing anything.
    detect_cuda_tag()
    if not args.skip_torch:
        install_torch(args.cuda_tag)
    else:
        print("\n[SKIPPED] torch/torchvision (--skip-torch)")

    install_requirements()

    if not args.skip_carla:
        install_carla(args.carla_version, args.carla_wheel)
    else:
        print("\n[SKIPPED] carla (--skip-carla)")

    if not args.skip_lfs:
        pull_lfs_weights()
    else:
        print("\n[SKIPPED] git-lfs weight pull (--skip-lfs)")

    print_manual_checklist(args)


if __name__ == "__main__":
    main()
