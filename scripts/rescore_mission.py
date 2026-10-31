"""Re-score the mission gates of recorded runs with the current ``score_mission``.

    python scripts/rescore_mission.py "D:\\Mjud2621\\Group7_runs\\r14_g2"

For every ``--scenario mission`` run under the root (a ``drive_report.json`` with an
``acceptance`` record and its ``drive_log.jsonl``), the mission rows (goal,
no_stop_for_roadside_car, obstacle_stop, resume, held) are recomputed from the log with
the scoring code of this checkout and written back; the run's status is recomputed from
its checks. The report as the run wrote it is kept beside it as
``drive_report.original.json`` (once), and the report records ``rescored`` with the
scoring commit, so a batch re-scored by a later evaluator fix (thirteenth review, 2: a
hold judged on its first gap only) keeps its provenance: the driving code is
``source_commit``, the scoring code ``rescored.commit``. Nothing about the drive changes.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_adapter.mapless_drive import DT_S, FAIL, INCOMPLETE, PASS, run_status  # noqa: E402
from carla_adapter.mapless_mission import load_scenario, score_mission  # noqa: E402

MISSION_CHECKS = ("goal", "no_stop_for_roadside_car", "obstacle_stop", "resume", "held")


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def rescore(folder, scenarios_dir=ROOT / "configs" / "scenarios", commit=None):
    """Re-score one run folder; returns (old status, new status) or None when it is not a
    mission run or has no log."""
    report_path = folder / "drive_report.json"
    log_path = folder / "drive_log.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    acceptance = report.get("acceptance")
    if not acceptance or not log_path.exists():
        return None
    log = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not log:
        return None
    scenario = load_scenario(acceptance["scenario"], scenarios_dir)
    old = {c["check"]: c for c in report["checks"]}
    held_s = (old.get("held") or old.get("obstacle_stop") or {}).get("held_s") or 0.0
    removed_at_s = (old.get("resume") or {}).get("removed_at_s")
    rows = score_mission(scenario, log, report.get("end"), int(round(held_s / DT_S)), removed_at_s, DT_S)
    fresh = {name: dict(check=name, status=INCOMPLETE if passed is None else (PASS if passed else FAIL), **values) for name, passed, values in rows}
    checks = [fresh.pop(c["check"], c) if c["check"] in MISSION_CHECKS else c for c in report["checks"]]
    checks.extend(fresh.values())  # a row the original run did not have
    original = folder / "drive_report.original.json"
    if not original.exists():
        shutil.copyfile(report_path, original)
    before = report.get("status")
    report["checks"] = checks
    report["status"] = run_status(checks)
    report["rescored"] = dict(commit=commit, checks=list(MISSION_CHECKS), previous_status=before, previous_report="drive_report.original.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return before, report["status"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path, help="folder holding the mission runs (searched recursively)")
    args = parser.parse_args(argv)
    commit = _commit()
    changed = 0
    for report_path in sorted(args.root.rglob("drive_report.json")):
        result = rescore(report_path.parent, commit=commit)
        if result is None:
            continue
        before, after = result
        mark = "" if before == after else "  <- changed"
        print(f"{report_path.parent}: {before} -> {after}{mark}")
        changed += before != after
    print(f"re-scored with {commit or 'an unknown commit'}; {changed} status change(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
