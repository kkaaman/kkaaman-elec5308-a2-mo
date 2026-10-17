"""Recorded lane overlays only; no vehicle command or controller path updates."""

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.dataset import local_file, read_dataset, save_json
from geometry.camera import CameraIntrinsics, pixel_to_ground
from perception.lane_inference import LaneSegmenter
from perception.lane_path import draw_ego_path, extract_ego_path
from visualisation.lane_monitor import REFERENCE_X_M, draw_monitor, monitor_lane


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.limit <= 200:
        parser.error("Limit must be in 1..200")
    manifest, rows = read_dataset(args.dataset)
    rows = [r for r in rows if r["split"] == "val" and r.get("sequence_index", 0) == 0]
    if not rows:
        raise ValueError("No validation frames")
    width, height = manifest["image_size_wh"]
    rig = manifest["camera"]
    # The Tesla actor origin is at road level, so the camera height above the
    # road is the mount height, consistent with CarlaWorld and the replay tool.
    # Flat-plane projection is diagnostic and is unreliable on grades.
    model = LaneSegmenter(str(args.weights))
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for row in rows[:: max(1, len(rows) // args.limit)][: args.limit]:
        with Image.open(local_file(args.dataset, row["rgb"])) as im:
            rgb = np.asarray(im.convert("RGB"))
        mask = model.predict(rgb)
        path = extract_ego_path(mask)
        points = []
        cal = CameraIntrinsics.from_horizontal_fov(
            width, height, math.radians(rig["fov_deg"])
        ).calibration(
            (
                (0, 0, 1, rig["x_actor_m"] + 1.38641),
                (-1, 0, 0, 0),
                (0, -1, 0, rig["z_actor_m"]),
                (0, 0, 0, 1),
            )
        )
        if path is not None:
            for u, v in path["centerline"]:
                point = pixel_to_ground(float(u), float(v), cal)
                if point is not None and 0 < point[0] < 30:
                    points.append(point[:2])
        points = sorted(points, key=lambda p: p[0])
        departure = monitor_lane(points, ego_x_m=REFERENCE_X_M)
        overlay = draw_monitor(draw_ego_path(rgb, path), departure)
        Image.fromarray(overlay).save(args.output / f"{row['sample_id']}.png")
        records.append(dict(sample_id=row["sample_id"], monitor=asdict(departure)))
    save_json(
        args.output / "monitor.json",
        dict(
            status="VISUAL_REVIEW_REQUIRED",
            records=records,
            control_coupling=False,
            scope="Parked dataset frames, flat-plane projection, offset 8 m ahead of the rear axle; use scripts/replay_lane_monitor.py on recorded runs for the map comparison on curves.",
        ),
    )
    print(f"Review {len(records)} overlays | {args.output}")


if __name__ == "__main__":
    main()
