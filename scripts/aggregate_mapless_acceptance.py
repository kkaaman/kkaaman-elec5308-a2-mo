"""G2's gate: S0–S5 × 5 seeds on the baseline route, from the mapless harness's reports.

    python scripts/aggregate_mapless_acceptance.py "$HOME\\Documents\\Group7_runs\\g2" [--repeats 5] [--output g2_acceptance.json]

Finds every ``drive_report.json`` under the root whose run was ``--scenario mission``
(``report["acceptance"]``), groups them by scenario and seed, and gives the gate's verdict:

- MEASURED_PASS: every scenario was run at exactly the required seeds (the frozen five,
  or ``--seeds``; ``--seeds any`` takes ``--repeats`` distinct ones), every run is
  MEASURED_PASS (or MEASURED_PASS_OVER_BUDGET, which is noted: the behaviour passed, the
  20 Hz deadline did not), every run carries its experiment identity in full (commit,
  mission, route noise, sensor-suite hash, stack-config hash, segmenter hash, admission)
  and all carry the same one, and nothing under the root is unreadable;
- MEASURED_FAIL: any run failed a gate;
- INCOMPLETE: anything else — a seed missing or unexpected, a run INCOMPLETE or
  NOT_EXERCISED (counted and listed apart: it passed its gates without reaching what its
  scenario is for), a run without its provenance, runs of more than one identity, or an
  unreadable report (twelfth review, 4: a missing commit was once "the same commit", and
  mixed route noise went unnoticed).

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


IDENTITY = ("code", "mission", "route_noise_m", "suite_sha256", "stack_config_sha256", "segmenter_sha256", "admission")
FROZEN_SEEDS = (5308, 5309, 5310, 5311, 5312)


def collect(root):
    """Every mission run under ``root``: its scenario, seed, status, gate results and its
    experiment identity (commit, mission, route noise, sensor-suite hash, stack-config
    hash, segmenter hash, admission). A report that cannot be read is listed with its
    error; a report that is not a mission run is ignored."""
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
        stack = report.get("stack") or {}
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
                suite_sha256=report.get("suite_sha256"),
                stack_config_sha256=stack.get("stack_config_sha256"),
                segmenter_sha256=stack.get("segmenter_sha256"),
                admission=stack.get("admission"),
                failed=failed,
                incomplete=incomplete,
                collisions=len(report.get("collisions") or []),
            )
        )
    return runs


def aggregate(runs, repeats=5, seeds=FROZEN_SEEDS):
    """The gate's verdict over ``runs`` (``collect``). MEASURED_PASS needs every scenario run
    at exactly the required ``seeds`` (``repeats`` of them when ``seeds`` is None), every
    run passing, every run carrying its full experiment identity (``IDENTITY``; a missing
    commit is not "the same commit": twelfth review, 4) and all of them the same, and no
    unreadable report under the root (a controlled acceptance folder holds nothing else)."""
    valid = [r for r in runs if "error" not in r and r.get("scenario") in SCENARIOS]
    by_scenario = defaultdict(list)
    for r in valid:
        by_scenario[r["scenario"]].append(r)
    required = tuple(seeds) if seeds else None
    per_scenario = {}
    for sid in SCENARIOS:
        rows = by_scenario.get(sid, [])
        seen = sorted({r["seed"] for r in rows})
        per_scenario[sid] = dict(
            runs=len(rows),
            seeds=seen,
            missing_seeds=[x for x in required if x not in seen] if required else [],
            unexpected_seeds=[x for x in seen if x not in required] if required else [],
            passed=sum(r["status"] in PASSING for r in rows),
            over_budget=sum(r["status"] == "MEASURED_PASS_OVER_BUDGET" for r in rows),
            failed=[dict(folder=r["folder"], seed=r["seed"], checks=r["failed"], end=r["end"]) for r in rows if r["status"] == "MEASURED_FAIL"],
            # Reported apart (external review, 31 Oct, section 6): a run that passed its gates
            # without reaching the transition its scenario is for is not a pass and not a
            # broken run.
            not_exercised=[dict(folder=r["folder"], seed=r["seed"], end=r["end"]) for r in rows if r["status"] == "NOT_EXERCISED"],
            incomplete=[dict(folder=r["folder"], seed=r["seed"], checks=r["incomplete"], end=r["end"]) for r in rows if r["status"] not in PASSING and r["status"] not in ("MEASURED_FAIL", "NOT_EXERCISED")],
        )
    # Experiment identity: every run names it in full, and all name the same one. The
    # segmenter hash may be None for the constant segmenter only when every run says so.
    unprovenanced = [dict(folder=r["folder"], missing=[k for k in IDENTITY if r.get(k) is None and k != "segmenter_sha256"]) for r in valid]
    unprovenanced = [u for u in unprovenanced if u["missing"]]
    identities = {tuple(r.get(k) for k in IDENTITY) for r in valid}
    homogeneous = len(identities) <= 1
    codes = sorted({r["code"] for r in valid if r.get("code")})
    missions = sorted({r["mission"] for r in valid if r.get("mission")})
    if required:
        complete = all(not per_scenario[sid]["missing_seeds"] and not per_scenario[sid]["unexpected_seeds"] and per_scenario[sid]["runs"] == len(required) for sid in SCENARIOS)
    else:
        complete = all(len(per_scenario[sid]["seeds"]) >= repeats and per_scenario[sid]["runs"] >= repeats for sid in SCENARIOS)
    unreadable = [r for r in runs if "error" in r]
    any_fail = any(per_scenario[sid]["failed"] for sid in SCENARIOS)
    any_incomplete = any(per_scenario[sid]["incomplete"] or per_scenario[sid]["not_exercised"] for sid in SCENARIOS)
    if any_fail:
        verdict = "MEASURED_FAIL"
    elif not complete or any_incomplete or not homogeneous or unprovenanced or unreadable or not valid:
        verdict = "INCOMPLETE"
    else:
        verdict = "MEASURED_PASS"
    over = sum(per_scenario[sid]["over_budget"] for sid in SCENARIOS)
    return dict(
        verdict=verdict,
        note=("every behaviour gate passed; " + f"{over} of {len(valid)} runs missed the 20 Hz deadline (MEASURED_PASS_OVER_BUDGET)") if verdict == "MEASURED_PASS" and over else None,
        repeats_required=repeats,
        seeds_required=list(required) if required else None,
        runs=len(valid),
        codes=codes,
        missions=missions,
        identity=dict(zip(IDENTITY, next(iter(identities)))) if len(identities) == 1 else None,
        identities=len(identities),
        homogeneous=homogeneous,
        unprovenanced=unprovenanced,
        scenarios=per_scenario,
        unreadable=unreadable,
    )


def table(summary):
    lines = [f"{'scenario':9} {'runs':>4} {'passed':>6} {'over budget':>11} {'failed':>6} {'not exercised':>13} {'incomplete':>10}  seeds", "-" * 84]
    for sid in SCENARIOS:
        s = summary["scenarios"][sid]
        lines.append(f"{sid:9} {s['runs']:>4} {s['passed']:>6} {s['over_budget']:>11} {len(s['failed']):>6} {len(s['not_exercised']):>13} {len(s['incomplete']):>10}  {', '.join(str(x) for x in s['seeds'])}")
    lines.append("-" * 84)
    lines.append(f"verdict: {summary['verdict']}" + (f" ({summary['note']})" if summary.get("note") else ""))
    for sid in SCENARIOS:
        for f in summary["scenarios"][sid]["failed"]:
            lines.append(f"  FAIL {sid} seed {f['seed']}: {', '.join(f['checks']) or f['end']}  {f['folder']}")
        for f in summary["scenarios"][sid]["not_exercised"]:
            lines.append(f"  NOT_EXERCISED {sid} seed {f['seed']}: {f['end']}  {f['folder']}")
        for f in summary["scenarios"][sid]["incomplete"]:
            lines.append(f"  INCOMPLETE {sid} seed {f['seed']}: {', '.join(f['checks']) or f['end']}  {f['folder']}")
        if summary["scenarios"][sid]["missing_seeds"] or summary["scenarios"][sid]["unexpected_seeds"]:
            lines.append(f"  SEEDS {sid}: missing {summary['scenarios'][sid]['missing_seeds']}, unexpected {summary['scenarios'][sid]['unexpected_seeds']}")
    if not summary["homogeneous"]:
        lines.append(f"  runs span {summary['identities']} experiment identities (commits {summary['codes']}, missions {summary['missions']}; route noise, suite, stack config, segmenter and admission are compared too)")
    for u in summary["unprovenanced"]:
        lines.append(f"  NO PROVENANCE {u['folder']}: missing {', '.join(u['missing'])}")
    for u in summary["unreadable"]:
        lines.append(f"  UNREADABLE {u['folder']}: {u['error']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path, help="folder holding the mission runs (searched recursively)")
    parser.add_argument("--repeats", type=int, default=5, help="runs per scenario when --seeds is 'any'")
    parser.add_argument("--seeds", default=",".join(str(x) for x in FROZEN_SEEDS), help="the required seed set, exactly (default: the frozen five); 'any' for --repeats distinct seeds")
    parser.add_argument("--output", type=Path, default=None, help="JSON summary (default: <root>/g2_acceptance.json)")
    args = parser.parse_args(argv)
    seeds = None if args.seeds.strip().lower() == "any" else tuple(int(x) for x in args.seeds.split(","))
    summary = aggregate(collect(args.root), args.repeats, seeds)
    out = args.output or args.root / "g2_acceptance.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table(summary))
    print(f"written {out}")
    return 0 if summary["verdict"] == "MEASURED_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
