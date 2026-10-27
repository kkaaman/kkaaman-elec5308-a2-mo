"""Read-only DOC-03 asset/runtime and optional CARLA preflight checks.

No installation, model loading, actor creation or evidence mutation occurs.
Reference hashes come from retained evidence, not the current local files.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_reference(root, manifest, assets_only=False):
    """Return all identity/version errors without changing local files."""
    errors = []
    for asset in manifest["assets"]:
        path = root / asset["path"]
        if not path.is_file():
            errors.append(f"Missing model: {path}; recover the authorized offline copy")
            continue
        if path.stat().st_size != asset["size_bytes"]:
            errors.append(f"Model size mismatch: {path}; check for an LFS pointer")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != asset["sha256"]:
            errors.append(f"Model SHA-256 mismatch: {path}; restore the recorded asset")
    if not assets_only:
        actual = ".".join(platform.python_version_tuple()[:2])
        if actual != manifest["python_major_minor"]:
            errors.append(f"Python {actual}; reference requires {manifest['python_major_minor']}")
        for name, expected in manifest["packages"].items():
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"Missing package: {name}=={expected}; prepare its offline wheel")
                continue
            print(f"Package {name}: {actual} (reference {expected})")
            if actual.split("+", 1)[0] != expected.split("+", 1)[0]:
                errors.append(f"Version mismatch: {name}={actual}, reference={expected}")
    return errors


def check_live(host, port, timeout, expected, require_cuda=False):
    """Read client/server versions; never tick or create simulator actors."""
    errors = []
    try:
        carla = importlib.import_module("carla")
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        local, remote = client.get_client_version(), client.get_server_version()
        print(f"CARLA client={local}, server={remote}, expected={expected}")
        if local != remote or local != expected:
            errors.append("CARLA version mismatch; use matching client wheel and server")
    except Exception as exc:
        errors.append(f"CARLA unavailable: {exc}; start the matching server and check host/port")
    if require_cuda:
        try:
            torch = importlib.import_module("torch")
            if not torch.cuda.is_available():
                errors.append("CUDA unavailable; check NVIDIA hardware, driver and torch build")
            else:
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception as exc:
            errors.append(f"CUDA check failed: {exc}")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/reproduction.json")
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.timeout <= 0 or not 1 <= args.port <= 65535:
        parser.error("timeout must be positive and port must be 1..65535")
    if args.require_cuda and not args.live:
        parser.error("--require-cuda requires --live")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        errors = check_reference(ROOT, manifest, args.assets_only)
        if args.live:
            errors += check_live(args.host, args.port, args.timeout,
                                 manifest["carla_version"], args.require_cuda)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL: invalid reference manifest: {exc}")
        return 1
    for error in errors:
        print(f"FAIL: {error}")
    if not errors:
        print("PASS: requested preflight checks only; not driving or accuracy acceptance")
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
