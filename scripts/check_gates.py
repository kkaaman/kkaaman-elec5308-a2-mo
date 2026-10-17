"""Report unresolved G1/G2/final evidence; never infer completion from filenames."""

import argparse
import json
from pathlib import Path

from autonomy.configuration import read_config
from evaluation.metrics import load_run, summarize
from evaluation.run_logger import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--s1-run", type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--heldout-perception", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    for name in (
        "comparison",
        "depth-analysis",
        "offline-report",
        "mission-validation",
        "lane-monitor",
    ):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    _, config, _ = read_config(args.config)
    missing = []
    g1 = config["status"] == "FROZEN_FOR_VALIDATION"
    for name, record in config.get("evidence", {}).items():
        if isinstance(record, dict) and "path" in record:
            path = Path(record["path"])
            if not path.is_file() or file_sha256(path) != record["sha256"]:
                missing.append(f"Changed/missing {name} evidence")
                g1 = False
    for name in (
        "comparison",
        "depth_analysis",
        "offline_report",
        "mission_validation",
        "lane_monitor",
    ):
        path = getattr(args, name)
        if path is None or not path.is_file():
            missing.append("G1 review evidence not supplied: " + name)
            g1 = False
    g2 = False
    if args.s1_run:
        m, rows, events = load_run(args.s1_run)
        s = summarize(rows, events, m)
        g2 = (
            m["engine"] == "carla"
            and m["scenario"] == "S1"
            and m["mode"] == "full_stack"
            and s["metrics"]["collisions"] == 0
            and s["metrics"]["hold_margin_min_m"] is not None
            and s["metrics"]["hold_margin_min_m"] > 0
            and any(r["state"] == "HOLD" and r.get("obstacles") for r in rows)
            and m["cleanup"]["ok"]
        )
    final = False
    if args.acceptance:
        a = json.loads(args.acceptance.read_text())
        final = a.get("status") == "MEASURED_PASS_REVIEW_REQUIRED" and a.get("counts") == {
            f"S{i}": 5 for i in range(6)
        }
    heldout = False
    if args.heldout_perception:
        p = json.loads(args.heldout_perception.read_text())
        heldout = p.get("split") == "test" and p.get("fault") is None and bool(p.get("sequences"))
    report = dict(
        G1="EVIDENCE_READY_REVIEW_REQUIRED" if g1 else "PENDING",
        G2="PERCEIVED_STOP_REVIEW_REQUIRED" if g2 else "PENDING",
        final_30_runs="MEASURED_PASS_REVIEW_REQUIRED" if final else "PENDING",
        heldout_audit="MEASURED_REVIEW_REQUIRED" if heldout else "PENDING",
        missing=missing,
        review="Named team review and route/overlay inspection remain required; this tool never merges or marks the task board.",
    )
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
