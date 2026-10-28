"""Read-only DOC-03 preflight. Standard library only; never downloads assets."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_assets(root, manifest):
    """Check every listed binary, rejecting missing, LFS-pointer or corrupt files."""
    results = []
    for asset in manifest["assets"]:
        path = root / asset["path"]
        try:
            with path.open("rb") as stream:
                prefix = stream.read(128)
                if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise ValueError("Git LFS pointer; restore the verified offline binary")
                stream.seek(0)
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if path.stat().st_size != asset["size_bytes"]:
                raise ValueError("size differs from recorded asset")
            if digest.hexdigest() != asset["sha256"]:
                raise ValueError("SHA-256 differs from recorded asset")
            results.append({"asset": asset["path"], "status": "PASS"})
        except (OSError, ValueError) as exc:
            results.append({"asset": asset["path"], "status": "FAIL", "error": str(exc)})
    return results


def check_packages(expected, version=importlib.metadata.version):
    """Compare the recorded subset only; local torch build suffixes are retained."""
    results = []
    for name, target in expected.items():
        try:
            installed = version(name)
            passed = installed.split("+", 1)[0] == target
            results.append({"package": name, "expected": target, "installed": installed,
                            "status": "PASS" if passed else "FAIL"})
        except importlib.metadata.PackageNotFoundError:
            results.append({"package": name, "status": "FAIL", "error": "not installed"})
    return results


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
        if actual != manifest.get("python_major_minor", ".".join(manifest["python_reference"].split(".")[:2])):
            errors.append(f"Python {actual}; reference requires {manifest.get('python_major_minor', manifest.get('python_reference'))}")
        for name, expected in manifest.get("packages", manifest.get("recorded_packages", {})).items():
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
    parser.add_argument("--assets-only", action="store_true",
                        help="Skip package/Python checks; does not validate an environment")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 60 or not 1 <= args.port <= 65535:
        parser.error("Finite timeout in (0,60] and port 1..65535 required")
    if args.require_cuda and not args.live:
        parser.error("--require-cuda requires --live")
    manifest = json.loads((ROOT / "configs/reproduction.json").read_text(encoding="utf-8"))
    results = check_assets(ROOT, manifest)
    if not args.assets_only:
        results += check_packages(manifest["recorded_packages"])
        results.append({"python": sys.version.split()[0], "reference": manifest["python_reference"],
                        "status": "PASS" if sys.version_info[:2] == (3, 12) else "FAIL"})
    if args.live:
        results += [{"status": "FAIL", "error": error} for error in
                    check_live(args.host, args.port, args.timeout,
                               manifest["carla_version"], args.require_cuda)]
    report = {"scope": "asset identity only" if args.assets_only else manifest["scope"],
              "checks": results, "status": "PASS" if all(x["status"] == "PASS" for x in results) else "FAIL"}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
