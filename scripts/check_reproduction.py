"""Read-only DOC-03 preflight. Standard library only; never downloads assets."""

import argparse
import hashlib
import importlib.metadata
import json
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-only", action="store_true",
                        help="Skip package/Python checks; does not validate an environment")
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / "configs/reproduction.json").read_text(encoding="utf-8"))
    results = check_assets(ROOT, manifest)
    if not args.assets_only:
        results += check_packages(manifest["recorded_packages"])
        results.append({"python": sys.version.split()[0], "reference": manifest["python_reference"],
                        "status": "PASS" if sys.version_info[:2] == (3, 12) else "FAIL"})
    report = {"scope": "asset identity only" if args.assets_only else manifest["scope"],
              "checks": results, "status": "PASS" if all(x["status"] == "PASS" for x in results) else "FAIL"}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
