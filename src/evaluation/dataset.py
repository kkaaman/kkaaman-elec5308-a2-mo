"""Dataset contracts and metrics; no simulator or model imports."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

KIT_VERSION = "group7-training-1"
ROADLINE_TAG = 24  # CARLA 0.9.16; the older tag 6 is incorrect for this build.
SPLITS = ("train", "val", "test")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def local_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing file or path outside dataset: {relative}")
    return path


# The baseline demo runs on Town10HD (Town10HD_Opt shares its OpenDRIVE).
DEMO_MAPS = ("Town10HD_Opt", "Town10HD")


def map_short_name(name):
    return str(name).replace("\\", "/").rsplit("/", 1)[-1]


def recorded_on_demo_map(manifest):
    """True unless the dataset manifest shows another town (unknown counts as demo)."""
    if "demo_map" in manifest:
        return bool(manifest["demo_map"])
    name = map_short_name(manifest.get("map", ""))
    return not name or name in DEMO_MAPS


def road_splits(road_ids):
    """Fixed grouping independent of collection seed, weather or frame order."""
    ids = sorted(
        set(int(x) for x in road_ids),
        key=lambda x: hashlib.sha256(f"road-split-v1:{x}".encode()).digest(),
    )
    if len(ids) < 5:
        raise ValueError("At least five non-junction roads are needed for spatial splits")
    n_test = max(1, round(len(ids) * 0.2))
    n_val = max(1, round(len(ids) * 0.2))
    return {
        road: ("test" if i < n_test else "val" if i < n_test + n_val else "train")
        for i, road in enumerate(ids)
    }


def decode_bgra(raw, width, height):
    if width <= 0 or height <= 0 or len(raw) != width * height * 4:
        raise ValueError("Invalid camera image dimensions or BGRA byte count")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)


def decode_depth(bgra):
    encoded = (
        bgra[:, :, 2].astype(np.uint32)
        + 256 * bgra[:, :, 1].astype(np.uint32)
        + 65536 * bgra[:, :, 0].astype(np.uint32)
    )
    depth = encoded.astype(np.float32) * (1000.0 / 16777215.0)
    depth[(encoded == 0) | (encoded == 16777215)] = np.nan
    return depth


def visible_boxes(instance_bgra, min_pixels=16):
    """Visible-mask boxes, NOT full/amodal actor boxes or production detections."""
    tags = instance_bgra[:, :, 2]
    ids = instance_bgra[:, :, 1].astype(np.uint32) * 256 + instance_bgra[:, :, 0]
    boxes = []
    for tag, name in ((12, "person"), (14, "car")):
        for identity in np.unique(ids[tags == tag]):
            ys, xs = np.where((tags == tag) & (ids == identity))
            if len(xs) < min_pixels:
                continue
            boxes.append(
                {
                    "class_name": name,
                    "instance_code": int(identity),
                    "xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                    "visible_pixels": int(len(xs)),
                }
            )
    return boxes


def confusion(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Prediction and target must be equal-size 2-D masks")
    if not np.isin(prediction, (0, 1)).all() or not np.isin(target, (0, 1)).all():
        raise ValueError("Masks must contain only binary labels")
    p, t = prediction.astype(bool), target.astype(bool)
    return np.array(
        [np.sum(p & t), np.sum(p & ~t), np.sum(~p & t), np.sum(~p & ~t)], dtype=np.int64
    )


def metrics(counts):
    tp, fp, fn, tn = map(int, counts)

    def ratio(a, b):
        return a / b if b else None

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "foreground_iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "false_positive_pixel_rate": ratio(fp, fp + tn),
    }


def read_dataset(folder, verify_hashes=True, require_complete=True):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != KIT_VERSION or manifest.get("label_tag") != ROADLINE_TAG:
        raise ValueError("Unsupported dataset schema or semantic label mapping")
    if require_complete and manifest.get("status") != "COLLECTED":
        raise ValueError("Collection is incomplete; inspect manifest.json")
    index = folder / "samples.jsonl"
    if sha256(index) != manifest.get("samples_sha256"):
        raise ValueError("Dataset index hash mismatch")
    rows = [
        json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or len(rows) != manifest.get("sample_count"):
        raise ValueError("Dataset is empty or sample count differs from manifest")
    road_owner, run_owner, image_owner, identities = {}, {}, {}, set()
    for row in rows:
        split = row["split"]
        if split not in SPLITS:
            raise ValueError("Unknown split")
        if row["sample_id"] in identities:
            raise ValueError("Duplicate sample ID")
        identities.add(row["sample_id"])
        if (
            len(set(row["sensor_frames"].values())) != 1
            or row["frame_id"] not in row["sensor_frames"].values()
        ):
            raise ValueError("Sensor frame mismatch")
        if not math.isfinite(row["sim_time_s"]):
            raise ValueError("Invalid simulation time")
        for owner, key in (
            (road_owner, str(row["road_id"])),
            (run_owner, row["run_id"]),
            (image_owner, row["sha256"]["rgb"]),
        ):
            if key in owner and owner[key] != split:
                raise ValueError("Train/validation/test leakage across road, run or image")
            owner[key] = split
        for kind in ("rgb", "lane", "instance", "depth"):
            path = local_file(folder, row[kind])
            if verify_hashes and sha256(path) != row["sha256"][kind]:
                raise ValueError(f"File hash mismatch: {path}")
        with (
            Image.open(local_file(folder, row["rgb"])) as image,
            Image.open(local_file(folder, row["lane"])) as mask,
        ):
            if (
                image.mode != "RGB"
                or image.size != mask.size
                or image.size != tuple(manifest["image_size_wh"])
            ):
                raise ValueError("Image mode, size or label alignment mismatch")
            if not np.isin(np.asarray(mask), (0, 1)).all():
                raise ValueError("Road-line mask must use binary 0/1 labels")
            target = np.asarray(mask).copy()
        with Image.open(local_file(folder, row["instance"])) as image:
            instance = np.asarray(image)
            if image.mode != "RGB" or instance.shape[:2] != target.shape:
                raise ValueError("Instance labels are not aligned")
            if not np.array_equal(instance[:, :, 0] == ROADLINE_TAG, target.astype(bool)):
                raise ValueError("Road-line mask disagrees with raw instance tags")
        depth = np.load(local_file(folder, row["depth"]), allow_pickle=False)
        if depth.shape != target.shape or depth.dtype.kind != "f":
            raise ValueError("Raw depth image is not aligned floating-point data")
    return manifest, rows
