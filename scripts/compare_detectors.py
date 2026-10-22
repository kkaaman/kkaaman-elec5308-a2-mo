"""Compare two perception replays of the same split, pose by pose (baseline vs candidate).

Both reports must come from scripts/benchmark_perception.py on the same dataset index,
split, depth calibration and detector settings, so only the weights differ. The gate
numbers are reported for both, but the useful reading is per in-path pose: which
targets the candidate fixes, which it breaks, and which neither model detects.
"""

import argparse
import json
from pathlib import Path

from evaluation.dataset import save_json
from evaluation.perception_replay import _pose_passes

BINS = (5, 10, 15, 20, 25, 30)


def in_path(report):
    return {
        (s["sequence_id"], s["class_name"]): s
        for s in report["sequences"]
        if s["scenario_target"]
        and s["requested_lateral_m"] is not None
        and abs(s["requested_lateral_m"]) < 0.5
    }


def compare(baseline, candidate):
    for key in ("dataset_index_sha256", "split", "depth_calibration_sha256", "detector"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"Reports differ in {key}; compare like with like")
    a, b = in_path(baseline), in_path(candidate)
    if set(a) != set(b):
        raise ValueError("Reports cover different in-path poses")
    rows = []
    for key in sorted(a, key=lambda k: (k[1], a[k]["distance_m"])):
        old, new = a[key], b[key]
        rows.append(
            dict(
                sequence_id=key[0],
                class_name=key[1],
                distance_m=round(old["distance_m"], 2),
                baseline_pass=_pose_passes(old),
                candidate_pass=_pose_passes(new),
                baseline_recall=old["recall"],
                candidate_recall=new["recall"],
            )
        )
    per_bin = []
    for name in ("car", "person"):
        for centre in BINS:
            chosen = [
                r for r in rows if r["class_name"] == name and abs(r["distance_m"] - centre) <= 2.5
            ]
            if chosen:
                per_bin.append(
                    dict(
                        class_name=name,
                        bin_centre_m=centre,
                        poses=len(chosen),
                        baseline_pass=sum(r["baseline_pass"] for r in chosen),
                        candidate_pass=sum(r["candidate_pass"] for r in chosen),
                    )
                )
    return dict(
        split=baseline["split"],
        detector=baseline["detector"],
        baseline_model_sha256=baseline["model_sha256"],
        candidate_model_sha256=candidate["model_sha256"],
        usable_range_m=dict(
            baseline=baseline["usable_range_lower_bound_m"],
            candidate=candidate["usable_range_lower_bound_m"],
        ),
        warm_p95_ms=dict(
            baseline=round(baseline["warm_p95_s"] * 1000, 1),
            candidate=round(candidate["warm_p95_s"] * 1000, 1),
        ),
        false_positives=dict(
            baseline=baseline.get("false_positives"), candidate=candidate.get("false_positives")
        ),
        in_path_poses=len(rows),
        fixed=[r for r in rows if r["candidate_pass"] and not r["baseline_pass"]],
        broken=[r for r in rows if r["baseline_pass"] and not r["candidate_pass"]],
        never_detected=dict(
            baseline=[r for r in rows if r["baseline_recall"] == 0],
            candidate=[r for r in rows if r["candidate_recall"] == 0],
        ),
        per_bin=per_bin,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))  # noqa: E731
    report = compare(load(args.baseline), load(args.candidate))
    if args.output.exists():
        raise FileExistsError(f"Choose a new output file: {args.output}")
    save_json(args.output, report)
    missed = {k: len(v) for k, v in report["never_detected"].items()}
    print(
        f"{report['split']} | in-path poses {report['in_path_poses']}"
        f" | fixed {len(report['fixed'])} broken {len(report['broken'])}"
        f" | never detected {missed}"
        f" | usable range baseline {report['usable_range_m']['baseline']}"
        f" candidate {report['usable_range_m']['candidate']}"
        f" | p95 ms {report['warm_p95_ms']} | {args.output}"
    )


if __name__ == "__main__":
    main()
