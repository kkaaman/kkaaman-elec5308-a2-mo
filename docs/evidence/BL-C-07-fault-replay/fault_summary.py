"""Fault-aware summary of perception replays (evaluation only, offline).

The range gate counts every frame, so a replay that deliberately drops 2 of
every 5 frames can never reach its 90% recall, whatever the tracker does. For
BL-C-07 the questions are instead, for the placed in-path target at each pose:
  * on the frames the detector was allowed to see, is it still detected,
    confirmed quickly and given valid depth?
  * once confirmed, how long can it go without a confirmed observation?
  * after a tracker-ID reset, how many frames until it is confirmed again?

Usage: python docs/evidence/BL-C-07-fault-replay/fault_summary.py NAME=replay.jsonl [NAME=replay.jsonl ...] [--json out.json]
NAME is clean, flicker or id_reset (decides which frames count as dropped).
"""

import json
import statistics
import sys
from collections import defaultdict

DT_S = 0.05
RESET_INDEX = 10
BINS = (5, 10, 15, 20, 25, 30)


def dropped(fault, index):
    return fault == "flicker" and index % 5 in (0, 1)


def traces(path):
    groups = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for truth in row["truths"]:
                lateral = truth.get("requested_lateral_m")
                if truth.get("scenario_target") and lateral is not None and abs(lateral) < 0.5:
                    key = (row["sequence_id"], truth["class_name"], truth["instance_code"])
                    groups[key].append((row["sequence_index"], truth))
    for key in groups:
        groups[key].sort(key=lambda item: item[0])
    return groups


def pose_metrics(fault, trace):
    start = trace[0][0]
    seen = [(i, t) for i, t in trace if not dropped(fault, i)]
    confirmed = [i for i, t in seen if t["confirmed"]]
    distances = [t["distance_m"] for _, t in trace if t["distance_m"] is not None]
    ids = [t["track_id"] for _, t in seen if t["track_id"] is not None]
    result = dict(
        distance_m=statistics.median(distances) if distances else None,
        frames=len(trace),
        seen_frames=len(seen),
        recall_seen=sum(t["detected"] for _, t in seen) / len(seen) if seen else 0.0,
        valid_depth_seen=sum(t["valid_depth"] for _, t in seen) / len(seen) if seen else 0.0,
        confirmation_s=(confirmed[0] - start) * DT_S if confirmed else None,
        # Longest wait between confirmed observations once first confirmed
        # (including the end of the sequence).
        max_confirmed_interval_s=max(
            (b - a for a, b in zip(confirmed, confirmed[1:] + [trace[-1][0] + 1])),
            default=None,
        ),
        distinct_track_ids=len(set(ids)),
    )
    if result["max_confirmed_interval_s"] is not None:
        result["max_confirmed_interval_s"] *= DT_S
    if fault == "id_reset":
        before = {t["track_id"] for i, t in seen if i < RESET_INDEX and t["track_id"] is not None}
        after = [i for i, t in seen if i >= RESET_INDEX and t["confirmed"]]
        detected_at_reset = any(t["detected"] for i, t in seen if i == RESET_INDEX)
        result.update(
            reset_in_trace=start <= RESET_INDEX <= trace[-1][0],
            detected_at_reset=detected_at_reset,
            new_id_after_reset=bool(
                {t["track_id"] for i, t in seen if i >= RESET_INDEX and t["track_id"] is not None}
                - before
            ),
            reconfirm_frames=after[0] - RESET_INDEX if after else None,
        )
    return result


def summarize(fault, path):
    poses = [
        pose_metrics(fault, trace) | dict(class_name=key[1]) for key, trace in traces(path).items()
    ]
    bins = []
    for name in ("car", "person"):
        for centre in BINS:
            rows = [
                p
                for p in poses
                if p["class_name"] == name
                and p["distance_m"] is not None
                and abs(p["distance_m"] - centre) <= 2.5
            ]
            if not rows:
                continue
            conf = [p["confirmation_s"] for p in rows if p["confirmation_s"] is not None]
            gaps = [
                p["max_confirmed_interval_s"]
                for p in rows
                if p["max_confirmed_interval_s"] is not None
            ]
            item = dict(
                class_name=name,
                bin_centre_m=centre,
                poses=len(rows),
                never_confirmed=sum(p["confirmation_s"] is None for p in rows),
                min_recall_seen=round(min(p["recall_seen"] for p in rows), 3),
                median_recall_seen=round(statistics.median(p["recall_seen"] for p in rows), 3),
                min_valid_depth_seen=round(min(p["valid_depth_seen"] for p in rows), 3),
                median_confirmation_s=round(statistics.median(conf), 3) if conf else None,
                max_confirmation_s=round(max(conf), 3) if conf else None,
                median_max_confirmed_interval_s=round(statistics.median(gaps), 3) if gaps else None,
                max_confirmed_interval_s=round(max(gaps), 3) if gaps else None,
                poses_with_id_change=sum(p["distinct_track_ids"] > 1 for p in rows),
            )
            if fault == "id_reset":
                hit = [p for p in rows if p["reset_in_trace"] and p["detected_at_reset"]]
                re = [p["reconfirm_frames"] for p in hit if p["reconfirm_frames"] is not None]
                item.update(
                    poses_detected_at_reset=len(hit),
                    never_reconfirmed=sum(p["reconfirm_frames"] is None for p in hit),
                    max_reconfirm_frames=max(re) if re else None,
                    median_reconfirm_frames=statistics.median(re) if re else None,
                )
            bins.append(item)
    return dict(fault=fault, path=path, in_path_poses=len(poses), bins=bins)


def main(argv):
    out = None
    if "--json" in argv:
        k = argv.index("--json")
        out = argv[k + 1]
        argv = argv[:k] + argv[k + 2 :]
    reports = []
    for spec in argv:
        fault, path = spec.split("=", 1)
        report = summarize(fault, path)
        reports.append(report)
        print(f"\n== {fault}: {report['in_path_poses']} in-path target poses ({path})")
        cols = [
            "class_name",
            "bin_centre_m",
            "poses",
            "never_confirmed",
            "min_recall_seen",
            "min_valid_depth_seen",
            "max_confirmation_s",
            "max_confirmed_interval_s",
            "poses_with_id_change",
        ]
        if fault == "id_reset":
            cols += ["poses_detected_at_reset", "never_reconfirmed", "max_reconfirm_frames"]
        print(" | ".join(cols))
        for b in report["bins"]:
            print(" | ".join(str(b.get(c)) for c in cols))
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(reports, handle, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
