"""Join verified captured labels and retained model replay to the weather evaluator."""

import argparse
import json
from pathlib import Path

from evaluation.dataset import read_dataset, sha256
from evaluation.weather import evaluate_weather, markdown_table


def convert(dataset, records, matrix):
    manifest, rows = read_dataset(dataset)
    if manifest.get("specification_sha256") != matrix["specification_sha256"]:
        raise ValueError("Dataset weather specification mismatch")
    inputs = {row["sample_id"]: row for row in rows if row["split"] == "test"}
    if not inputs or {r["sample_id"] for r in records} != set(inputs):
        raise ValueError("Replay must cover exactly the held-out samples")
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Duplicate replay sample")
    frames = []
    for record in records:
        source = inputs[record["sample_id"]]
        if (record["frame_id"], record["sim_time_s"]) != (source["frame_id"], source["sim_time_s"]):
            raise ValueError("Replay source identity differs from capture")
        labels = [
            dict(
                class_name=t["class_name"],
                xyxy=t["xyxy"],
                visible=True,
                depth_valid=t["distance_m"] is not None,
                distance_m=t["distance_m"],
            )
            for t in record["truths"]
        ]
        detections = [
            dict(class_name=p["class_name"], xyxy=p["xyxy"], confirmed=p["confirmed"])
            for p in record["predictions"]
            if p["class_name"] in ("car", "person")
        ]
        frames.append(
            dict(
                cell_id=source["weather_cell_id"],
                frame_id=record["frame_id"],
                sim_time_s=record["sim_time_s"],
                valid=record["valid"],
                labels=labels,
                detections=detections,
            )
        )
    return dict(
        origin="recorded", specification_sha256=matrix["specification_sha256"], frames=frames
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text().splitlines()]
    matrix = json.loads(args.matrix.read_text())
    observations = convert(args.dataset, records, matrix)
    report = evaluate_weather(observations, matrix)
    report["records_sha256"] = sha256(args.records)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "observations.json").write_text(
        json.dumps(observations, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "table.md").write_text(markdown_table(report))
    print(report["status"])
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
