"""Inspect a mapless capture run (plan v2.1, section 12: the pilot is inspected before scaling up).

Reads one run folder written by ``carla_adapter.mapless_capture`` and writes, under
``<run>/inspection/``:

- ``summary.json``:
  - the run: status, samples, episodes, skipped degraded frames;
  - conditions: weather, speed, traffic-light states, actors nearby;
  - labels: per-camera class shares, and pedestrian blob heights in the front camera
    (small pedestrians are the hard case);
  - two projection checks per camera, which catch a wrong extrinsic or a label image
    that does not belong to its RGB image:
    - **ground:** LiDAR points near the road surface should land on road, pavement,
      lane marking, terrain or ground labels;
    - **vehicles:** LiDAR points inside a truth vehicle box should land on vehicle labels;
  - LiDAR and radar point counts.
- ``contact_sheet.png``: random samples, each as front RGB, semantic overlay, and LiDAR
  projected on the image coloured by range.
- ``cameras_sheet.png``: every camera with its overlay, for two samples.

Truth and labels are read here for inspection only; they never reach the car.
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.rig_calibration import inside_box, project
from mapless.geometry import camera_calibration
from mapless.sensor_suite import suite_from_dict

# CARLA 0.9.16 semantic tags.
TAG_NAMES = {
    0: "unlabeled", 1: "road", 2: "sidewalk", 3: "building", 4: "wall", 5: "fence", 6: "pole",
    7: "traffic_light", 8: "traffic_sign", 9: "vegetation", 10: "terrain", 11: "sky",
    12: "pedestrian", 13: "rider", 14: "car", 15: "truck", 16: "bus", 17: "train",
    18: "motorcycle", 19: "bicycle", 20: "static", 21: "dynamic", 22: "other", 23: "water",
    24: "road_line", 25: "ground", 26: "bridge", 27: "rail_track", 28: "guard_rail",
}  # fmt: skip
GROUND_TAGS = (1, 2, 10, 24, 25)
VEHICLE_TAGS = (14, 15, 16, 17, 18, 19)
PEDESTRIAN_TAG = 12
PALETTE = {
    1: (128, 64, 128), 2: (244, 35, 232), 3: (70, 70, 70), 4: (102, 102, 156), 5: (190, 153, 153),
    6: (153, 153, 153), 7: (250, 170, 30), 8: (220, 220, 0), 9: (107, 142, 35), 10: (152, 251, 152),
    11: (70, 130, 180), 12: (220, 20, 60), 13: (255, 0, 0), 14: (0, 0, 142), 15: (0, 0, 70),
    16: (0, 60, 100), 18: (0, 0, 230), 19: (119, 11, 32), 24: (157, 234, 50), 25: (81, 0, 81),
}  # fmt: skip
PEDESTRIAN_HEIGHT_BINS = (0, 20, 40, 80, 10_000)


def colourise(tags):
    rgb = np.zeros(tags.shape + (3,), np.uint8)
    for tag, colour in PALETTE.items():
        rgb[tags == tag] = colour
    return rgb


def read_rgb(path, cv2):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image[:, :, ::-1]


def read_tags(path, cv2):
    tags = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if tags is None:
        raise FileNotFoundError(path)
    return tags if tags.ndim == 2 else tags[:, :, 2]


def lidar_points(run, sample):
    files = sample["files"]["lidar"]
    if not files:
        return np.zeros((0, 4), np.float32)
    with np.load(run / next(iter(files.values()))) as data:
        return data["points"]


def labels_at(points, calibration, tags):
    """Semantic tag under each point that projects into the image (-1 elsewhere)."""
    out = np.full(len(points), -1, dtype=int)
    if len(points) == 0:
        return out
    pixels, front = project(points[:, :3], calibration)
    u, v = np.round(pixels[:, 0]).astype(int), np.round(pixels[:, 1]).astype(int)
    h, w = tags.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    rows = np.flatnonzero(front)[inside]
    out[rows] = tags[v[inside], u[inside]]
    return out


def pedestrian_heights(tags, cv2):
    mask = (tags == PEDESTRIAN_TAG).astype(np.uint8)
    if not mask.any():
        return []
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [int(stats[i, cv2.CC_STAT_HEIGHT]) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 6]


def rear_axle_offset(manifest):
    records = manifest.get("rig_records") or []
    offsets = [r.get("rear_axle_offset_m") for r in records if r.get("rear_axle_offset_m")]
    return float(np.median(offsets)) if offsets else 1.38641


def share(numerator, denominator):
    return None if denominator == 0 else round(numerator / denominator, 4)


def inspect(run, *, max_samples=200, sheet_samples=8, seed=0, cv2=None):
    if cv2 is None:
        import cv2
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    samples = [json.loads(line) for line in (run / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not samples:
        raise ValueError("No samples in this run")
    suite = suite_from_dict(manifest["suite"])
    offset = rear_axle_offset(manifest)
    specs = {spec.name: spec for spec in suite.cameras}
    calibrations = {spec.name: camera_calibration(spec, offset) for spec in suite.cameras}
    rng = random.Random(seed)
    step = max(1, len(samples) // max_samples)
    subset = samples[::step][:max_samples]

    weather, lights = Counter(), Counter()
    speeds, vehicles_near, pedestrians_near = [], [], []
    for sample in samples:
        truth = sample["truth"]
        weather[truth.get("weather")] += 1
        lights[truth.get("traffic_light_state")] += 1
        speeds.append(sample["speed_mps"])
        actors = truth.get("actors", [])
        near = [a for a in actors if math.hypot(a["x_m"], a["y_m"]) <= 30]
        vehicles_near.append(sum(a["type_id"].startswith("vehicle.") for a in near))
        pedestrians_near.append(sum(a["type_id"].startswith("walker.") for a in near))

    class_pixels = {name: Counter() for name in calibrations}
    ground = {name: [0, 0] for name in calibrations}
    vehicle = {name: [0, 0] for name in calibrations}
    shape_errors = []
    ped_heights = []
    lidar_counts, radar_counts = [], Counter()
    for sample in subset:
        points = lidar_points(run, sample)
        lidar_counts.append(len(points))
        for name, path in sample["files"]["radar"].items():
            radar_counts[name] += int(np.load(run / path).shape[0])
        near_ground = (
            (points[:, 2] < 0.25) & (np.hypot(points[:, 0], points[:, 1]) > 4) & (np.hypot(points[:, 0], points[:, 1]) < 30)
        )
        on_vehicle = np.zeros(len(points), bool)
        for actor in sample["truth"].get("actors", []):
            if actor["type_id"].startswith("vehicle.") and math.hypot(actor["x_m"], actor["y_m"]) < 30:
                on_vehicle |= inside_box(
                    points[:, :3],
                    (actor["x_m"], actor["y_m"], actor["z_m"]),
                    actor["yaw_rad"],
                    actor["extent_m"],
                    0.1,  # LiDAR returns lie on the car's surface, at or just inside its box
                    min_above_bottom_m=0.3,
                )
        for name, calibration in calibrations.items():
            rgb_path = sample["files"]["rgb"].get(name)
            label_path = sample["files"]["semantic"].get(name)
            if rgb_path is None or label_path is None:
                continue
            tags = read_tags(run / label_path, cv2)
            if tags.shape != (specs[name].height, specs[name].width):
                shape_errors.append(dict(frame=sample["frame"], camera=name, shape=list(tags.shape)))
                continue
            values, counts = np.unique(tags, return_counts=True)
            class_pixels[name].update(dict(zip(values.tolist(), counts.tolist())))
            hit = labels_at(points, calibration, tags)
            seen = hit >= 0
            g = seen & near_ground
            ground[name][0] += int(np.isin(hit[g], GROUND_TAGS).sum())
            ground[name][1] += int(g.sum())
            vv = seen & on_vehicle
            vehicle[name][0] += int(np.isin(hit[vv], VEHICLE_TAGS).sum())
            vehicle[name][1] += int(vv.sum())
            if name == "front":
                ped_heights += pedestrian_heights(tags, cv2)

    bins = PEDESTRIAN_HEIGHT_BINS
    ped_hist = {
        f"{bins[i]}-{bins[i + 1]}px" if bins[i + 1] < 10_000 else f">={bins[i]}px": sum(bins[i] <= h < bins[i + 1] for h in ped_heights)
        for i in range(len(bins) - 1)
    }
    classes = {}
    for name, counter in class_pixels.items():
        total = sum(counter.values())
        classes[name] = {TAG_NAMES.get(t, str(t)): round(c / total, 4) for t, c in counter.most_common(12)} if total else {}
    speeds = np.array(speeds)
    summary = dict(
        run=str(run.name),
        map=manifest.get("map"),
        status=manifest.get("status"),
        failure=manifest.get("failure"),
        samples=len(samples),
        episodes=len({s["truth"].get("episode") for s in samples}),
        skipped_degraded_frames=manifest.get("skipped_degraded_frames"),
        inspected_samples=len(subset),
        rear_axle_offset_m=offset,
        weather=dict(weather),
        speed=dict(stationary_share=share(int((speeds < 0.1).sum()), len(speeds)), median_mps=round(float(np.median(speeds)), 2)),
        traffic_light_states=dict(lights),
        actors_within_30m=dict(
            vehicles_mean=round(float(np.mean(vehicles_near)), 2),
            pedestrians_mean=round(float(np.mean(pedestrians_near)), 2),
            samples_with_pedestrian=int(sum(p > 0 for p in pedestrians_near)),
        ),
        front_pedestrian_blob_heights=ped_hist,
        class_shares=classes,
        projection_checks={
            name: dict(
                ground_points_on_ground_labels=share(*ground[name]),
                ground_points=ground[name][1],
                vehicle_points_on_vehicle_labels=share(*vehicle[name]),
                vehicle_points=vehicle[name][1],
            )
            for name in calibrations
        },
        label_shape_errors=shape_errors[:20],
        lidar_points_per_sweep_median=int(np.median(lidar_counts)) if lidar_counts else 0,
        radar_returns_per_frame_mean={k: round(v / len(subset), 1) for k, v in radar_counts.items()},
        notes=(
            "Projection checks: expect ground >= 0.9 and vehicle >= 0.8 on every camera. A camera far"
            " below the others points to a wrong mount or a label image paired with the wrong RGB frame."
        ),
    )
    out = run / "inspection"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _contact_sheet(run, rng.sample(samples, min(sheet_samples, len(samples))), calibrations, out / "contact_sheet.png", cv2)
    _cameras_sheet(run, rng.sample(samples, min(2, len(samples))), calibrations, out / "cameras_sheet.png", cv2)
    return summary


def _tile(image, size, cv2):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _overlay(rgb, tags):
    return (0.5 * rgb + 0.5 * colourise(tags)).astype(np.uint8)


def _lidar_on(rgb, points, calibration, cv2):
    image = rgb.copy()
    if len(points):
        pixels, front = project(points[:, :3], calibration)
        ranges = np.hypot(points[front, 0], points[front, 1])
        for (u, v), r in zip(pixels.astype(int), ranges):
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                t = min(1.0, r / 40.0)
                cv2.circle(image, (int(u), int(v)), 2, (int(255 * (1 - t)), int(255 * t), 60), -1)
    return image


def _contact_sheet(run, samples, calibrations, path, cv2, size=(480, 270)):
    rows = []
    for sample in samples:
        name = "front" if "front" in calibrations else next(iter(calibrations))
        rgb = read_rgb(run / sample["files"]["rgb"][name], cv2)
        tags = read_tags(run / sample["files"]["semantic"][name], cv2)
        points = lidar_points(run, sample)
        tiles = [rgb, _overlay(rgb, tags), _lidar_on(rgb, points, calibrations[name], cv2)]
        row = np.hstack([_tile(t, size, cv2) for t in tiles])
        label = f"frame {sample['frame']}  ep {sample['truth'].get('episode')}  {sample['truth'].get('weather')}  v={sample['speed_mps']:.1f} m/s"
        cv2.putText(row, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(row)
    cv2.imwrite(str(path), np.vstack(rows)[:, :, ::-1])


def _cameras_sheet(run, samples, calibrations, path, cv2, size=(400, 225)):
    rows = []
    for sample in samples:
        tiles = []
        for name in calibrations:
            rgb = read_rgb(run / sample["files"]["rgb"][name], cv2)
            tags = read_tags(run / sample["files"]["semantic"][name], cv2)
            tile = _tile(_overlay(rgb, tags), size, cv2)
            cv2.putText(tile, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(path), np.vstack(rows)[:, :, ::-1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="A capture run folder (has manifest.json)")
    parser.add_argument("--max-samples", type=int, default=200, help="Samples read for the label and projection checks")
    parser.add_argument("--sheet-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    summary = inspect(args.run, max_samples=args.max_samples, sheet_samples=args.sheet_samples, seed=args.seed)
    checks = summary["projection_checks"]
    print(json.dumps(dict(samples=summary["samples"], episodes=summary["episodes"], projection_checks=checks), indent=2))
    print(f"Wrote {Path(args.run) / 'inspection'}")
