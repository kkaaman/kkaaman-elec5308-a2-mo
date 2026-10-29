"""Run TEST-18 offline fixtures and export diagnostics; blocked scope stays visible."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offline_fault_suite import run_suite, write_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-fixtures", action="store_true",
                        help="Exit 0 for passing available fixtures; overall scope stays BLOCKED")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new directory")
    report = run_suite()
    try:
        report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        report["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        report["git_commit"], report["git_dirty"] = None, None
    write_report(report, args.output, ROOT)
    print(json.dumps({"status": report["status"], **report["counts"]}))
    if report["counts"]["failed"]:
        return 1
    return 0 if args.check_fixtures else 2


if __name__ == "__main__":
    raise SystemExit(main())
