"""Label definitions shared by the segmentation backends and their evaluation.

Every segmenter in this package returns Cityscapes trainIds (0-18) with 255 for
pixels it does not label. CARLA's semantic camera uses tags 1-19 for the same 19
classes in the same order (tag = trainId + 1), so ground truth captured in CARLA
can be converted without retraining any model.
"""

from __future__ import annotations

import numpy as np

from .postprocessing import CITYSCAPES_PALETTE

IGNORE_ID = 255

CITYSCAPES_CLASSES: tuple[str, ...] = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)  # fmt: skip

#: trainIds that matter for obstacle stopping: person, rider, car, truck, bus,
#: motorcycle, bicycle (train is excluded; there are no trains on the route).
OBSTACLE_TRAIN_IDS: tuple[int, ...] = (11, 12, 13, 14, 15, 17, 18)

CARLA_ROADLINE_TAG = 24
#: CARLA CityScapesPalette colour for RoadLine (tag 24), labelled as road.
CARLA_ROADLINE_RGB = (157, 234, 50)


def _carla_tag_lut() -> np.ndarray:
    lut = np.full(256, IGNORE_ID, dtype=np.uint8)
    lut[1:20] = np.arange(19, dtype=np.uint8)  # tags 1..19 -> trainIds 0..18
    lut[CARLA_ROADLINE_TAG] = 0  # lane markings are part of the road in Cityscapes
    return lut


_CARLA_TAG_LUT = _carla_tag_lut()


def carla_tags_to_trainid(tags: np.ndarray) -> np.ndarray:
    """Convert CARLA semantic tags (HxW uint8) to Cityscapes trainIds.

    Tags 1-19 map to trainIds 0-18, RoadLine (24) maps to road, and every other
    tag (Unlabeled, Static, Dynamic, Water, GuardRail, ...) becomes 255.

    Args:
        tags: HxW integer array of CARLA tags, e.g. the red channel of a raw
            ``sensor.camera.semantic_segmentation`` image.

    Returns:
        HxW uint8 array of trainIds with 255 for ignored pixels.
    """
    tags = np.asarray(tags)
    if tags.ndim != 2:
        raise ValueError(f"expected an HxW tag array, got shape {tags.shape}")
    return _CARLA_TAG_LUT[np.clip(tags, 0, 255).astype(np.uint8)]


def carla_palette_to_trainid(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert an image saved with CARLA's ``CityScapesPalette`` back to trainIds.

    Useful for frames saved as colour PNGs (for example by the smoke test). Raw
    tag images are preferable for datasets because no colour lookup is needed.

    Args:
        rgb: HxWx3 uint8 RGB image.

    Returns:
        ``(mask, matched_fraction)``: the HxW trainId mask (255 where the colour
        is not one of the 19 classes or RoadLine) and the fraction of pixels whose
        colour matched. A low fraction means the image is not a CityScapes-palette
        label image.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected an HxWx3 uint8 RGB image")
    colours = np.vstack([CITYSCAPES_PALETTE, np.array([CARLA_ROADLINE_RGB], dtype=np.uint8)])
    train_ids = np.append(np.arange(19, dtype=np.uint8), np.uint8(0))
    keys = (colours[:, 0].astype(np.int64) << 16) | (colours[:, 1].astype(np.int64) << 8) | colours[:, 2]
    order = np.argsort(keys)
    keys, train_ids = keys[order], train_ids[order]

    pixels = (rgb[..., 0].astype(np.int64) << 16) | (rgb[..., 1].astype(np.int64) << 8) | rgb[..., 2]
    index = np.clip(np.searchsorted(keys, pixels), 0, len(keys) - 1)
    matched = keys[index] == pixels
    mask = np.where(matched, train_ids[index], IGNORE_ID).astype(np.uint8)
    return mask, float(matched.mean())
