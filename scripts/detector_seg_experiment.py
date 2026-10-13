#!/usr/bin/env python3
"""Experiment: does YOLO-seg give better obstacle distances than box-only YOLO?

Compares the current detector (``yolo26n.pt``) with its segmentation version
(``yolo26n-seg.pt``) on CARLA frames with one parked car or one standing
pedestrian placed in the ego lane at known distances.

Two phases, so the models never compete with CARLA for the GPU while timed:

``capture`` (needs a running CARLA server)
    Finds a straight lane in the current map, parks the ego car with
    co-located RGB, depth and instance-segmentation cameras, saves a baseline
    frame, then places each obstacle at each gap (ego front bumper to the
    obstacle's near surface, the plan's stop-margin definition) and saves a
    frame set.
    Everything spawned is destroyed and the world settings are restored.

``analyze`` (needs ultralytics; no CARLA)
    Runs each model on the saved frames and reports, per class and distance:
    whether the obstacle was detected (IoU >= 0.5 with the ground-truth box),
    and the distance error of three ways of reading depth for the detection:
    full box median, inner box median (middle 50%), and segmentation-mask
    median (seg models only). Ground truth is the median depth over the
    obstacle's own pixels from the instance camera (evaluation only).

Examples (from the project folder)::

    python scripts/detector_seg_experiment.py capture --out runs/det_seg
    python scripts/detector_seg_experiment.py analyze runs/det_seg/<run>

    # Try a different camera, e.g. the 1280x720 option from the pixel-size table
    python scripts/detector_seg_experiment.py capture --width 1280 --height 720

This is an experiment and a prototype for BL-B-03 capture, not baseline code.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

CARLA_TAGS = {"car": 14, "pedestrian": 12}
DETECTOR_CLASSES = {"car": {"car", "truck", "bus"}, "pedestrian": {"person"}}
DEFAULT_DISTANCES = (5, 10, 15, 20, 25, 30)
DEFAULT_MODELS = ("yolo26n.pt", "yolo26n-seg.pt")
SKY_DEPTH_M = 999.0
IOU_MATCH = 0.5
NEAR_PERCENTILE = 5  # "nearest surface" = 5th percentile of depth, robust to stray edge pixels
ERROR_KEYS = (
    "err_box_full_m", "err_box_inner_m", "err_mask_m",
    "err_near_box_m", "err_near_inner_m", "err_near_mask_m",
)  # fmt: skip

# ============================================================================ pure helpers


def decode_depth(bgra: np.ndarray) -> np.ndarray:
    """CARLA depth image (BGRA uint8) -> metres (float32), without uint8 overflow."""
    b, g, r = (bgra[..., i].astype(np.float64) for i in range(3))
    return (1000.0 * (r + 256.0 * g + 65536.0 * b) / 16777215.0).astype(np.float32)


def decode_instance(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CARLA instance image -> (semantic tag, instance id). Tag is red; id is green/blue."""
    tag = bgr[..., 2].astype(np.int32)
    instance = bgr[..., 1].astype(np.int32) * 256 + bgr[..., 0].astype(np.int32)
    return tag, instance


def ground_truth_mask(tag, instance, base_tag, class_tag: int) -> np.ndarray | None:
    """Pixels of the placed obstacle.

    Pixels that are ``class_tag`` now but were not in the baseline frame belong
    to the new obstacle; the dominant instance id among them identifies it,
    and all pixels with that id form the mask (so partial overlaps with parked
    cars of the same class do not shrink it). Returns None if not visible.
    """
    new_pixels = (tag == class_tag) & (base_tag != class_tag)
    if new_pixels.sum() < 4:
        return None
    ids, counts = np.unique(instance[new_pixels], return_counts=True)
    obstacle_id = ids[np.argmax(counts)]
    return (instance == obstacle_id) & (tag == class_tag)


def mask_box(mask: np.ndarray) -> tuple[float, float, float, float]:
    """Tight (x1, y1, x2, y2) pixel box around a boolean mask (x2/y2 exclusive)."""
    ys, xs = np.nonzero(mask)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_region(shape, box, fraction: float = 1.0) -> np.ndarray:
    """Boolean mask of a box, optionally shrunk to its central ``fraction``."""
    h, w = shape
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * fraction / 2, (y2 - y1) * fraction / 2
    c0, c1 = int(max(0, math.floor(cx - hw))), int(min(w, math.ceil(cx + hw)))
    r0, r1 = int(max(0, math.floor(cy - hh))), int(min(h, math.ceil(cy + hh)))
    region = np.zeros((h, w), dtype=bool)
    region[r0:r1, c0:c1] = True
    return region


def median_depth(depth: np.ndarray, region: np.ndarray) -> float | None:
    """Median depth over ``region``, ignoring sky; None if nothing valid."""
    values = depth[region & (depth < SKY_DEPTH_M)]
    return float(np.median(values)) if values.size else None


def near_depth(depth: np.ndarray, region: np.ndarray, percentile: float = NEAR_PERCENTILE) -> float | None:
    """Nearest-surface depth over ``region`` (low percentile), ignoring sky."""
    values = depth[region & (depth < SKY_DEPTH_M)]
    return float(np.percentile(values, percentile)) if values.size else None


def evaluate_detection(frame: dict, detections: list[dict]) -> dict:
    """Match detections to the ground-truth obstacle and score the depth readings.

    Two targets: the median visible surface (``err_*``) and the nearest surface
    (``err_near_*``), which is what the stop margin depends on.
    """
    gt_box, gt_dist, depth = frame["gt_box"], frame["gt_distance_m"], frame["depth"]
    gt_near = frame.get("gt_near_m")
    wanted = DETECTOR_CLASSES[frame["class"]]
    best, best_iou = None, 0.0
    for det in detections:
        if det["label"] in wanted:
            iou = box_iou(det["box"], gt_box)
            if iou > best_iou:
                best, best_iou = det, iou
    row = {"detected": best is not None and best_iou >= IOU_MATCH, "iou": round(best_iou, 3)}
    row["confidence"] = round(best["confidence"], 3) if best else None
    for key in ERROR_KEYS:
        row[key] = None
    if not row["detected"]:
        return row
    full = box_region(depth.shape, best["box"])
    regions = {"box": full, "inner": box_region(depth.shape, best["box"], 0.5)}
    if best.get("mask") is not None:
        regions["mask"] = best["mask"] & full
    median_keys = {"box": "err_box_full_m", "inner": "err_box_inner_m", "mask": "err_mask_m"}
    for name, region in regions.items():
        value = median_depth(depth, region)
        row[median_keys[name]] = None if value is None else round(value - gt_dist, 3)
        if gt_near is not None:
            near = near_depth(depth, region)
            row[f"err_near_{name}_m"] = None if near is None else round(near - gt_near, 3)
    return row


def summarize(rows: list[dict]) -> dict:
    """Per model and class: detection rate and median |error| for each depth reading."""
    summary: dict = {}
    for row in rows:
        entry = summary.setdefault(row["model"], {}).setdefault(row["class"], {"frames": 0, "detected": 0})
        entry["frames"] += 1
        entry["detected"] += int(row["detected"])
        for key in ERROR_KEYS:
            if row.get(key) is not None:
                entry.setdefault(key, []).append(abs(row[key]))
    for classes in summary.values():
        for entry in classes.values():
            entry["detection_rate"] = round(entry["detected"] / entry["frames"], 3)
            for key in ERROR_KEYS:
                if key in entry:
                    entry[key.replace("err_", "median_abs_err_")] = round(statistics.median(entry.pop(key)), 3)
    return summary


# ============================================================================ capture (CARLA)


def _wait_for_frame(sensor_queue, frame, timeout_s=5.0):
    deadline = time.perf_counter() + timeout_s
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"sensor did not deliver frame {frame}")
        data = sensor_queue.get(timeout=remaining)
        if data.frame == frame:
            return data
        if data.frame > frame:
            raise RuntimeError(f"sensor skipped to frame {data.frame} (wanted {frame})")


def _bgra(image) -> np.ndarray:
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4).copy()


def _straight_lane_start(carla_map, spawn_points, length_m: float, start_index: int):
    """First spawn point whose lane runs straight with no junction for ``length_m``."""
    order = spawn_points[start_index:] + spawn_points[:start_index]
    for spawn in order:
        wp0 = carla_map.get_waypoint(spawn.location)
        if wp0 is None or wp0.is_junction:
            continue
        ok = True
        for d in np.arange(2.5, length_m + 2.5, 2.5):
            nxt = wp0.next(float(d))
            if not nxt:
                ok = False
                break
            wp = nxt[0]
            yaw_diff = abs((wp.transform.rotation.yaw - wp0.transform.rotation.yaw + 180) % 360 - 180)
            if wp.is_junction or yaw_diff > 3.0:
                ok = False
                break
        if ok:
            return wp0
    raise RuntimeError(f"no straight {length_m:.0f} m lane found from any spawn point")


def capture(args) -> Path:
    import carla

    run_dir = args.out / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    carla_map = world.get_map()
    original = world.get_settings()
    spawned, sensors = [], {}

    def tick():
        frame = world.tick(10.0)
        return frame, {k: _wait_for_frame(q, frame) for k, (_, q) in sensors.items()}

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)
        blueprints = world.get_blueprint_library()

        wp0 = _straight_lane_start(
            carla_map, carla_map.get_spawn_points(), max(args.distances) + 10, args.spawn_index
        )
        ego_tf = carla.Transform(
            carla.Location(wp0.transform.location.x, wp0.transform.location.y, wp0.transform.location.z + 0.3),
            wp0.transform.rotation,
        )
        ego = world.try_spawn_actor(blueprints.filter(args.ego)[0], ego_tf)
        if ego is None:
            raise RuntimeError("could not spawn the ego vehicle on the chosen lane")
        spawned.append(ego)
        ego.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))

        cam_tf = carla.Transform(carla.Location(x=1.5, z=1.7))
        for kind, bp_id in (
            ("rgb", "sensor.camera.rgb"),
            ("depth", "sensor.camera.depth"),
            ("instance", "sensor.camera.instance_segmentation"),
        ):
            bp = blueprints.find(bp_id)
            bp.set_attribute("image_size_x", str(args.width))
            bp.set_attribute("image_size_y", str(args.height))
            bp.set_attribute("fov", str(args.fov))
            actor = world.spawn_actor(bp, cam_tf, attach_to=ego)
            spawned.append(actor)
            q: queue.Queue = queue.Queue()
            actor.listen(q.put)
            sensors[kind] = (actor, q)

        fwd = wp0.transform.get_forward_vector()
        loc = wp0.transform.location
        world.get_spectator().set_transform(
            carla.Transform(
                carla.Location(loc.x - 10 * fwd.x, loc.y - 10 * fwd.y, loc.z + 5),
                carla.Rotation(pitch=-15, yaw=wp0.transform.rotation.yaw),
            )
        )

        for _ in range(15):  # let the ego settle and cameras warm up
            tick()
        _, data = tick()
        _save_frame(run_dir / "baseline", data)
        print(f"Baseline saved. Lane start at ({loc.x:.1f}, {loc.y:.1f}).")

        focal_px = args.width / (2 * math.tan(math.radians(args.fov) / 2))
        meta = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "map": carla_map.name.split("/")[-1],
            "camera": {"width": args.width, "height": args.height, "fov_deg": args.fov,
                       "focal_px": round(focal_px, 2), "mount_xyz_m": [1.5, 0.0, 1.7]},
            "distance_definition": "gap_m = ego front bumper to obstacle near surface, along the lane",
            "frames": [],
        }  # fmt: skip
        base_tag, _ = decode_instance(cv2.imread(str(run_dir / "baseline" / "instance.png")))

        ego_front_m = ego.bounding_box.location.x + ego.bounding_box.extent.x
        meta["camera_to_bumper_m"] = round(ego_front_m - 1.5, 3)  # camera mounted at x = 1.5 m

        def lane_transform(centre_m: float, dz: float, yaw_offset: float):
            target = wp0.next(float(centre_m))[0].transform
            return carla.Transform(
                carla.Location(target.location.x, target.location.y, target.location.z + dz),
                carla.Rotation(yaw=target.rotation.yaw + yaw_offset),
            )

        for cls in args.classes:
            bp = blueprints.filter(args.car if cls == "car" else args.walker)[0]
            yaw_offset = 180.0 if cls == "pedestrian" else 0.0
            for dist in args.distances:
                guess_half = 2.4 if cls == "car" else 0.3
                # A pedestrian's origin is at hip height, so it needs ~1 m clearance;
                # spawned lower it ends up partly underground and falls through.
                heights = (0.3, 0.6, 1.0, 1.5) if cls == "car" else (1.0, 1.2, 1.5)
                obstacle, dz_used = None, heights[0]
                for dz in heights:
                    obstacle = world.try_spawn_actor(bp, lane_transform(dist + ego_front_m + guess_half, dz, yaw_offset))
                    if obstacle is not None:
                        dz_used = dz
                        break
                name = f"{cls}_{dist:02d}m"
                if obstacle is None:
                    print(f"  {name}: could not spawn, skipped")
                    continue
                try:
                    # Re-place using the obstacle's real size so the gap is exact.
                    box = obstacle.bounding_box
                    near_m = box.extent.x - (box.location.x if cls == "car" else 0.0)
                    if cls == "pedestrian":
                        # Physics off: stands exactly where placed, feet on the road.
                        obstacle.set_simulate_physics(False)
                        dz_used = box.extent.z - box.location.z + 0.03
                    obstacle.set_transform(lane_transform(dist + ego_front_m + near_m, dz_used, yaw_offset))
                    if cls == "car":
                        obstacle.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
                    for _ in range(15):  # settle onto the road
                        tick()
                    frame_id, data = tick()
                    frame_dir = run_dir / "frames" / name
                    _save_frame(frame_dir, data)
                    tag, inst = decode_instance(cv2.imread(str(frame_dir / "instance.png")))
                    mask = ground_truth_mask(tag, inst, base_tag, CARLA_TAGS[cls])
                    pixels = 0 if mask is None else int(mask.sum())
                    meta["frames"].append(
                        {"name": name, "class": cls, "gap_m": dist, "frame": frame_id,
                         "obstacle": obstacle.type_id, "gt_pixels": pixels}
                    )  # fmt: skip
                    note = "" if pixels else "  <- not visible in the camera; check the CARLA window"
                    print(f"  {name}: saved ({pixels} obstacle pixels){note}")
                finally:
                    obstacle.destroy()
                    tick()
        (run_dir / "capture.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"\nCaptured {len(meta['frames'])} frame sets in {run_dir}")
        return run_dir
    finally:
        for actor, _ in sensors.values():
            try:
                actor.stop()
            except Exception:  # noqa: BLE001
                pass
        for actor in reversed(spawned):
            try:
                actor.destroy()
            except Exception:  # noqa: BLE001
                pass
        world.apply_settings(original)


def _save_frame(directory: Path, data: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / "rgb.png"), _bgra(data["rgb"])[..., :3])
    np.save(directory / "depth.npy", decode_depth(_bgra(data["depth"])))
    cv2.imwrite(str(directory / "instance.png"), _bgra(data["instance"])[..., :3])


# ============================================================================ analysis


def load_frames(run_dir: Path) -> list[dict]:
    meta = json.loads((run_dir / "capture.json").read_text(encoding="utf-8"))
    cam_to_bumper = meta.get("camera_to_bumper_m", 0.95)
    base_tag, _ = decode_instance(cv2.imread(str(run_dir / "baseline" / "instance.png")))
    frames = []
    for info in meta["frames"]:
        d = run_dir / "frames" / info["name"]
        tag, inst = decode_instance(cv2.imread(str(d / "instance.png")))
        mask = ground_truth_mask(tag, inst, base_tag, CARLA_TAGS[info["class"]])
        if mask is None:
            print(f"  {info['name']}: obstacle not visible in the instance image, skipped")
            continue
        depth = np.load(d / "depth.npy")
        gt_near = near_depth(depth, mask)
        frames.append(
            {**info, "bgr": cv2.imread(str(d / "rgb.png")), "depth": depth, "gt_mask": mask,
             "gt_box": mask_box(mask), "gt_distance_m": median_depth(depth, mask), "gt_near_m": gt_near,
             "gt_gap_m": None if gt_near is None else gt_near - cam_to_bumper}
        )  # fmt: skip
    return frames


def run_model(model, frame_bgr: np.ndarray, args) -> list[dict]:
    """Ultralytics prediction -> list of {label, confidence, box, mask?}."""
    result = model.predict(frame_bgr, conf=args.conf, imgsz=args.imgsz, retina_masks=True, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    names = result.names
    masks = result.masks.data.cpu().numpy() > 0 if result.masks is not None else None
    h, w = frame_bgr.shape[:2]
    out = []
    for i, (cls, conf, box) in enumerate(
        zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.xyxy.cpu().numpy())
    ):
        mask = None
        if masks is not None:
            mask = masks[i]
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        out.append({"label": names[int(cls)], "confidence": float(conf), "box": tuple(map(float, box)), "mask": mask})
    return out


def analyze(args) -> dict:
    from ultralytics import YOLO

    run_dir: Path = args.run_dir
    frames = load_frames(run_dir)
    if not frames:
        raise SystemExit("no usable frames to analyse")
    rows, timing = [], {}
    for weights in args.models:
        model = YOLO(weights)
        for _ in range(5):
            model.predict(frames[0]["bgr"], conf=args.conf, imgsz=args.imgsz, retina_masks=True, verbose=False)
        times = []
        for frame in frames:
            start = time.perf_counter()
            detections = run_model(model, frame["bgr"], args)
            times.append((time.perf_counter() - start) * 1000.0)
            row = {"model": Path(weights).name, "class": frame["class"], "gap_m": frame["gap_m"],
                   "gt_distance_m": round(frame["gt_distance_m"], 2), "gt_pixels": int(frame["gt_mask"].sum()),
                   "gt_gap_m": None if frame["gt_gap_m"] is None else round(frame["gt_gap_m"], 2)}
            row.update(evaluate_detection(frame, detections))
            rows.append(row)
        timing[Path(weights).name] = {"mean_ms": round(statistics.fmean(times), 1), "n": len(times)}  # fmt: skip

    report = {
        "run_dir": str(run_dir),
        "analyzed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"models": list(args.models), "conf": args.conf, "imgsz": args.imgsz, "iou_match": IOU_MATCH},
        "timing": timing,
        "summary": summarize(rows),
        "rows": rows,
    }
    (run_dir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_report(report)
    return report


def _fmt(value) -> str:
    return "-" if value is None else f"{value:+.2f}" if isinstance(value, float) else str(value)


def _print_report(report: dict) -> None:
    print("\nErrors in metres (estimate minus truth). 'Median' = median visible surface;")
    print("'Nearest' = nearest surface, which the stop margin depends on.\n")
    print("| Model | Class | Gap m | Pixels | Detected | IoU | Median: box | inner | mask "
          "| Nearest: box | inner | mask |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in report["rows"]:
        print(
            f"| {r['model']} | {r['class']} | {r['gap_m']} | {r['gt_pixels']} "
            f"| {'yes' if r['detected'] else 'NO'} | {r['iou']:.2f} | {_fmt(r['err_box_full_m'])} "
            f"| {_fmt(r['err_box_inner_m'])} | {_fmt(r['err_mask_m'])} | {_fmt(r.get('err_near_box_m'))} "
            f"| {_fmt(r.get('err_near_inner_m'))} | {_fmt(r.get('err_near_mask_m'))} |"
        )
    print("\nSummary: median absolute error over detected frames, metres")
    print("| Model | Class | Detected | Median: box | inner | mask | Nearest: box | inner | mask | ms/frame |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for model, classes in report["summary"].items():
        for cls, s in classes.items():
            cells = [s.get(f"median_abs_{k}", "-") for k in ERROR_KEYS]
            print(
                f"| {model} | {cls} | {s['detected']}/{s['frames']} | " + " | ".join(str(c) for c in cells)
                + f" | {report['timing'][model]['mean_ms']} |"
            )
    print(f"\nSaved: {Path(report['run_dir']) / 'analysis.json'}")


# ============================================================================ CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="YOLO vs YOLO-seg obstacle-distance experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="place obstacles in CARLA and save frames")
    cap.add_argument("--out", type=Path, default=Path("runs/det_seg"))
    cap.add_argument("--host", default="localhost")
    cap.add_argument("--port", type=int, default=2000)
    cap.add_argument("--classes", nargs="+", choices=sorted(CARLA_TAGS), default=["car", "pedestrian"])
    cap.add_argument("--distances", type=int, nargs="+", default=list(DEFAULT_DISTANCES),
                     help="gaps in metres from the ego front bumper to the obstacle")
    cap.add_argument("--width", type=int, default=800)
    cap.add_argument("--height", type=int, default=450)
    cap.add_argument("--fov", type=float, default=90.0)
    cap.add_argument("--ego", default="vehicle.lincoln.mkz_2020")
    cap.add_argument("--car", default="vehicle.tesla.model3", help="obstacle car blueprint")
    cap.add_argument("--walker", default="walker.pedestrian.0001", help="pedestrian blueprint")
    cap.add_argument("--spawn-index", type=int, default=0, help="where to start searching for a straight lane")
    cap.add_argument("--then-analyze", action="store_true", help="run the analysis right after capture")

    for p in (sub.add_parser("analyze", help="run the models on a captured run"), cap):
        p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
        p.add_argument("--conf", type=float, default=0.20, help="detector confidence (A1 demo uses 0.20)")
        p.add_argument("--imgsz", type=int, default=640)
    sub.choices["analyze"].add_argument("run_dir", type=Path)

    args = parser.parse_args(argv)
    if args.command == "capture":
        run_dir = capture(args)
        if args.then_analyze:
            args.run_dir = run_dir
            analyze(args)
    else:
        analyze(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
