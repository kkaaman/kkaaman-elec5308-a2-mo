"""Shadow localisation error across runs (reads each run's summary.json and manifest.json).

Usage: python scripts/localisation_report.py <folder containing run folders> [--output report.json]
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for summary_path in sorted(args.runs.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads((summary_path.parent / "manifest.json").read_text(encoding="utf-8"))
        report = summary.get("localisation")
        if not report:
            continue
        sensors = manifest.get("localisation", {}).get("sensors", {})
        rows.append(
            dict(
                run=summary_path.parent.name,
                seed=manifest.get("seed"),
                lateral_mean_m=report["lateral_error_m"]["mean"],
                lateral_p95_m=report["lateral_error_m"]["p95"],
                lateral_max_m=report["lateral_error_m"]["max"],
                along_p95_m=report["along_error_m"]["p95"],
                yaw_p95_deg=report["yaw_error_deg"]["p95"],
                gyro_truth_correlation=report["gyro_truth_correlation"],
                gnss_bias_m=sensors.get("gnss_bias_m"),
                fixes=sensors.get("counts", {}).get("fixes"),
                late_or_missing_fixes=sensors.get("counts", {}).get("late_or_missing_fixes"),
            )
        )
    if not rows:
        raise SystemExit(f"No runs with a localisation summary under {args.runs}")
    for r in rows:
        print(
            f"{r['run']} seed {r['seed']} | lateral mean {r['lateral_mean_m']:.3f} "
            f"p95 {r['lateral_p95_m']:.3f} max {r['lateral_max_m']:.3f} m | along p95 "
            f"{r['along_p95_m']:.3f} m | yaw p95 {r['yaw_p95_deg']:.2f} deg | gyro corr "
            f"{r['gyro_truth_correlation']} | GNSS offset {r['gnss_bias_m']} | fixes {r['fixes']} "
            f"(late/missing {r['late_or_missing_fixes']})"
        )
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
