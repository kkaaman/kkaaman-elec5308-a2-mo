"""Achieved braking against the stack's assumed decelerations (third review, section 4).

Usage:
    python scripts/braking_table.py runs/s2b_brake runs/s2b_ped [--calibration runs/bl_e_03/longitudinal.csv] [--json out.json]

Drive-run folders (``drive_log.jsonl``) give closed-loop braking episodes; a BL-E-03 brake
step file (``scripts/calibrate_longitudinal.py --output ...``) gives the open-loop steps,
the cleaner measure. Prints the table by command and starting speed, and each assumed
deceleration against the achieved peak at full brake.
"""

import argparse
import json
from pathlib import Path

from evaluation.braking_table import (
    against_assumptions,
    default_assumptions,
    episodes,
    read_log,
    steps_from_calibration,
    table,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="drive-run folders or drive_log.jsonl files")
    parser.add_argument("--calibration", type=Path, help="BL-E-03 longitudinal.csv")
    parser.add_argument("--json", type=Path, help="write the episodes, table and comparison here")
    args = parser.parse_args()
    eps, recorded = [], {}
    for run in args.runs:
        for e in episodes(read_log(run)):
            eps.append(dict(e, source=str(run)))
        report = (run if run.is_dir() else run.parent) / "drive_report.json"
        rules = json.loads(report.read_text(encoding="utf-8")).get("braking_assumptions") if report.exists() else None
        recorded[str(run)] = rules
    known = [r for r in recorded.values() if r is not None]
    if any(r != known[0] for r in known):
        # The rules are checked against what the runs drove with: runs with different
        # braking settings are not one analysis (fourth review).
        raise SystemExit("the runs record different braking assumptions; analyse them separately: " + json.dumps(recorded, indent=1))
    rules = known[0] if known else default_assumptions()
    if len(known) < len(recorded):
        print(f"{len(recorded) - len(known)} run(s) record no braking assumptions: the code's defaults stand in for them")
    if args.calibration is not None:
        eps += [dict(e, source="BL-E-03") for e in steps_from_calibration(args.calibration)]
    if not eps:
        raise SystemExit("no braking episodes found")
    rows = table(eps)
    print(f"{len(eps)} braking episodes")
    print(f"{'brake':>10} {'speed m/s':>10} {'n':>4} {'mean':>6} {'peak':>6} {'peak p10':>8} {'effective':>9} {'lag':>6} {'lag max':>7}")
    fmt = lambda v, w: f"{'' if v is None else format(v, '.2f'):>{w}}"  # noqa: E731
    for r in rows:
        print(
            f"{r['brake']:>10} {r['speed_mps']:>10} {r['episodes']:>4} {r['mean_decel_median_mps2']:>6.2f} {r['peak_decel_median_mps2']:>6.2f}"
            f" {r['peak_decel_p10_mps2']:>8.2f} {fmt(r['effective_decel_median_mps2'], 9)} {fmt(r['lag_median_s'], 6)} {fmt(r['lag_max_s'], 7)}"
        )
    check = against_assumptions(eps, rules)
    print(json.dumps(check, indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(dict(episodes=eps, table=rows, against_assumptions=check), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
