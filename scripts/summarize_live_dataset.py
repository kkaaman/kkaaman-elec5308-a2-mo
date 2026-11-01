"""Summarise paired capture/replay with spatial-site uncertainty and retained failures.

Reuse retained model outputs; no inference, model tuning or simulator calls.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.dataset import sha256


def close_value(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close_value(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close_value(x, y) for x, y in zip(a, b))
    if type(a) in (int, float) and type(b) in (int, float):
        return math.isclose(a, b, abs_tol=1e-4, rel_tol=1e-7)
    return a == b


def pairing(rows):
    groups = defaultdict(dict)
    failed_targets = set()
    for row in rows:
        scenario, condition, repetition = row["weather_cell_id"].split("-")
        sample = row["sample_id"].split("/")[-1]
        key = (scenario, repetition, sample)
        if condition in groups[key]:
            raise ValueError("Duplicate paired sample")
        target = {
            k: v
            for k, v in row["target_evaluation_only"].items()
            if k not in ("actor_id", "visible_instance_box_count")
        }
        groups[key][condition] = dict(
            road_id=row["road_id"],
            lane_id=row["lane_id"],
            road_s_m=row["road_s_m"],
            ego_world=row["ego_world"],
            camera_world=row["camera_world"],
            target=target,
        )
        if scenario != "S0" and not target["spawned"]:
            failed_targets.add((row["weather_cell_id"], row["sequence_id"]))
    mismatches = []
    for key, conditions in groups.items():
        if set(conditions) != {"ClearNoon", "HardRainNoon", "ClearNight"}:
            mismatches.append(dict(pair=list(key), reason="missing condition"))
        elif not all(close_value(v, conditions["ClearNoon"]) for v in conditions.values()):
            mismatches.append(dict(pair=list(key), reason="pose/target mismatch"))
    return dict(
        status="PAIRED" if not mismatches else "PAIRING_FAILED",
        paired_frame_groups=len(groups),
        mismatches=mismatches,
        failed_target_scenes=[list(v) for v in sorted(failed_targets)],
    )


def clustered_interval(values, seed=5308):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    totals = values.sum(axis=0)
    rng = np.random.default_rng(seed)
    recalls = []
    for _ in range(1000):
        tp, fp, fn = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        if tp + fn:
            recalls.append(tp / (tp + fn))
    return dict(
        spatial_sites=len(values),
        tp=int(totals[0]),
        fp=int(totals[1]),
        fn=int(totals[2]),
        recall=float(totals[0] / (totals[0] + totals[2])) if totals[0] + totals[2] else None,
        recall_site_bootstrap_95=[float(v) for v in np.percentile(recalls, [2.5, 97.5])]
        if recalls
        else None,
        bootstrap_valid_replicates=len(recalls),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--perception", type=Path, required=True)
    parser.add_argument("--weather", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    index = args.dataset / "samples.jsonl"
    if sha256(index) != manifest["samples_sha256"]:
        raise ValueError("Captured index changed")
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    selected = {r["sample_id"]: r for r in rows if r["split"] == "test"}
    records = [
        json.loads(line) for line in (args.perception / "records.jsonl").read_text().splitlines()
    ]
    if len(records) != len(selected) or {r["sample_id"] for r in records} != set(selected):
        raise ValueError("Held-out replay coverage differs from captured test set")
    clusters = defaultdict(lambda: defaultdict(lambda: np.zeros(3, dtype=int)))
    invalid_depth = 0
    for record in records:
        source = selected[record["sample_id"]]
        site = (source["road_id"], source["lane_id"], round(source["road_s_m"], 3))
        for name in ("car", "person"):
            clusters[(source["weather"], name)][site] += record["class_counts"][name]
        invalid_depth += record["invalid_depth_count"]
    summary = json.loads((args.perception / "summary.json").read_text())
    weather = json.loads(args.weather.read_text())
    report = dict(
        pairing=pairing(rows),
        heldout_frames=len(records),
        invalid_depth_detections=invalid_depth,
        per_condition_class=[
            dict(weather=w, class_name=n, **clustered_interval(list(v.values())))
            for (w, n), v in sorted(clusters.items())
        ],
        limitations=[
            "Visible instance masks, not amodal boxes.",
            "Intervals resample spatial sites within this captured town; not cross-town confidence.",
            "Adjacent frames/seeds/weather views at a site are not independent trials.",
            "Stationary first-confirmed range is not moving first-admission or stopping evidence.",
        ],
        weather_status=weather["status"],
        operational_acceptance=False,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Held-out replay failure and correction table",
        "",
        "No detector tuning was applied to test scenes. Canonical filter/freshness/HOLD fixes are evaluated without inventing a new model correction.",
        "",
        "| Sequence | Class | Frames | Recall | Valid depth | Confirm s (simulation) | First confirmed axial m | ID switches |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summary["sequences"]:
        vals = [
            row[k]
            for k in (
                "sequence_id",
                "class_name",
                "frames",
                "recall",
                "valid_depth_fraction",
                "confirmation_s",
                "first_confirmed_distance_m",
                "id_switches",
            )
        ]
        lines.append(
            "| " + " | ".join("NOT CONFIRMED" if v is None else str(v) for v in vals) + " |"
        )
    (args.output / "replay-table.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    weathers = ["ClearNoon", "HardRainNoon", "ClearNight"]
    for index, name in enumerate(("car", "person")):
        selected_rows = [
            next(
                r
                for r in report["per_condition_class"]
                if r["weather"] == w and r["class_name"] == name
            )
            for w in weathers
        ]
        for j, row in enumerate(selected_rows):
            if row["recall"] is not None:
                axes[0].scatter(j + index * 0.15, row["recall"], label=name if j == 0 else None)
                interval = row["recall_site_bootstrap_95"]
                if interval:
                    axes[0].vlines(j + index * 0.15, *interval, linewidth=2)
        for j, w in enumerate(weathers):
            sequences = [
                row
                for row in summary["sequences"]
                if row["class_name"] == name
                and row["first_confirmed_distance_m"] is not None
                and any(
                    r["sequence_id"] == row["sequence_id"] and r["weather"] == w
                    for r in selected.values()
                )
            ]
            distances = [row["first_confirmed_distance_m"] for row in sequences]
            if distances:
                axes[1].scatter(
                    j + index * 0.15, np.median(distances), label=name if j == 0 else None
                )
                axes[1].vlines(j + index * 0.15, min(distances), max(distances), linewidth=2)
    for ax in axes:
        ax.set_xticks(np.arange(3) + 0.075, weathers, rotation=12)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set(ylabel="Visible-object recall (spatial-site bootstrap 95%)", ylim=(-0.05, 1.05))
    axes[1].set(ylabel="First-confirmed axial distance (m): median / min-max")
    figure.savefig(args.output / "perception-range.png", dpi=220)
    plt.close(figure)
    print(
        json.dumps(
            dict(
                pairing=report["pairing"]["status"],
                heldout_frames=len(records),
                weather_status=weather["status"],
            )
        )
    )
    return 0 if report["pairing"]["status"] == "PAIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
