"""G2's gate: S0–S5 × 5 seeds on the baseline route, from the mapless harness's reports.

    python scripts/aggregate_mapless_acceptance.py "$HOME\\Documents\\Group7_runs\\g2" [--repeats 5] [--output g2_acceptance.json]

Finds every ``drive_report.json`` under the root whose run was ``--scenario mission``
(``report["acceptance"]``), groups them by scenario and seed, and gives the gate's verdict:

- MEASURED_PASS: every scenario has ``repeats`` distinct seeds and every run is
  MEASURED_PASS (or MEASURED_PASS_OVER_BUDGET, which is noted: the behaviour passed, the
  20 Hz deadline did not);
- MEASURED_FAIL: any run failed a gate;
- INCOMPLETE: a scenario short of its repeats, a run INCOMPLETE, or runs on more than one
  commit or mission.

Writes the JSON summary and prints a table. The plan's static gates (zero collisions, the
stop margin, HOLD speed ≤ 0.1 m/s, completion) are the runs' own checks; this only counts.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCENARIOS = tuple(f"S{i}" for i in range(6))
PASSING = ("MEASURED_PASS", "MEASURED_PASS_OVER_BUDGET")


def collect(root):
    runs = []
    for path in sorted(Path(root).rglob("drive_report.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            runs.append(dict(folder=str(path.parent), error=str(error)))
            continue
        acceptance = report.get("acceptance")
        if not acceptance:
            continue
        failed = [c["check"] for c in report.get("checks", []) if c.get("status") == "FAIL"]
        incomplete = [c["check"] for c in report.get("checks", []) if c.get("status") == "INCOMPLETE"]
        runs.append(
            dict(
                folder=str(path.parent),
                scenario=acceptance.get("scenario"),
                seed=acceptance.get("seed"),
                mission=acceptance.get("mission"),
                route_noise_m=acceptance.get("route_noise_m"),
                status=report.get("status"),
                end=report.get("end"),
                code=report.get("source_commit"),
                failed=failed,
                incomplete=incomplete,
                collisions=len(report.get("collisions") or []),
            )
        )
    return runs


def aggregate(runs, repeats=5):
    valid = [r for r in runs if "error" not in r and r.get("scenario") in SCENARIOS]
    by_scenario = defaultdict(list)
    for r in valid:
        by_scenario[r["scenario"]].append(r)
    per_scenario = {}
    for sid in SCENARIOS:
        rows = by_scenario.get(sid, [])
        seeds = sorted({r["seed"] for r in rows})
        per_scenario[sid] = dict(
            runs=len(rows),
            seeds=seeds,
            passed=sum(r["status"] in PASSING for r in rows),
            over_budget=sum(r["status"] == "MEASURED_PASS_OVER_BUDGET" for r in rows),
            failed=[dict(folder=r["folder"], seed=r["seed"], checks=r["failed"], end=r["end"]) for r in rows if r["status"] == "MEASURED_FAIL"],
            incomplete=[dict(folder=r["folder"], seed=r["seed"], checks=r["incomplete"], end=r["end"]) for r in rows if r["status"] not in PASSING and r["status"] != "MEASURED_FAIL"],
        )
    codes = {r["code"] for r in valid if r.get("code")}
    missions = {r["mission"] for r in valid}
    homogeneous = len(codes) <= 1 and len(missions) <= 1
    complete = all(len(per_scenario[sid]["seeds"]) >= repeats and per_scenario[sid]["runs"] >= repeats for sid in SCENARIOS)
    any_fail = any(per_scenario[sid]["failed"] for sid in SCENARIOS)
    any_incomplete = any(per_scenario[sid]["incomplete"] for sid in SCENARIOS)
    if any_fail:
        verdict = "MEASURED_FAIL"
    elif not complete or any_incomplete or not homogeneous or not valid:
        verdict = "INCOMPLETE"
    else:
        verdict = "MEASURED_PASS"
    over = sum(per_scenario[sid]["over_budget"] for sid in SCENARIOS)
    return dict(
        verdict=verdict,
        note=("every behaviour gate passed; " + f"{over} of {len(valid)} runs missed the 20 Hz deadline (MEASURED_PASS_OVER_BUDGET)") if verdict == "MEASURED_PASS" and over else None,
        repeats_required=repeats,
        runs=len(valid),
        codes=sorted(codes),
        missions=sorted(m for m in missions if m),
        homogeneous=homogeneous,
        scenarios=per_scenario,
        unreadable=[r for r in runs if "error" in r],
    )


def table(summary):
    lines = [f"{'scenario':9} {'runs':>4} {'passed':>6} {'over budget':>11} {'failed':>6}  seeds", "-" * 60]
    for sid in SCENARIOS:
        s = summary["scenarios"][sid]
        lines.append(f"{sid:9} {s['runs']:>4} {s['passed']:>6} {s['over_budget']:>11} {len(s['failed']):>6}  {', '.join(str(x) for x in s['seeds'])}")
    lines.append("-" * 60)
    lines.append(f"verdict: {summary['verdict']}" + (f" ({summary['note']})" if summary.get("note") else ""))
    for sid in SCENARIOS:
        for f in summary["scenarios"][sid]["failed"]:
            lines.append(f"  FAIL {sid} seed {f['seed']}: {', '.join(f['checks']) or f['end']}  {f['folder']}")
        for f in summary["scenarios"][sid]["incomplete"]:
            lines.append(f"  INCOMPLETE {sid} seed {f['seed']}: {', '.join(f['checks']) or f['end']}  {f['folder']}")
    if not summary["homogeneous"]:
        lines.append(f"  runs span commits {summary['codes']} / missions {summary['missions']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path, help="folder holding the mission runs (searched recursively)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None, help="JSON summary (default: <root>/g2_acceptance.json)")
    args = parser.parse_args(argv)
    summary = aggregate(collect(args.root), args.repeats)
    out = args.output or args.root / "g2_acceptance.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table(summary))
    print(f"written {out}")
    return 0 if summary["verdict"] == "MEASURED_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
