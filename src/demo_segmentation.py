"""Segmentation-model options shared by the live CARLA demos.

ADDED (A2, BL-B-05 prep): lets demo_combined_perception.py and
demo_fused_lane_following.py swap the layer-1 segmentation model from the
command line (``--seg-backend bisenetv2|segformer_b0|yolo_seg``) and report
its per-frame time, without changing their default A1 behaviour.

Run the demos from src/, so the relative weight paths below resolve.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import numpy as np

from segmentation import BACKENDS, create_segmenter
from segmentation.backends import DEFAULT_WEIGHTS

SEGMENTATION_WEIGHTS = "../segmentation_model.pth"
SEG_TIMING_WINDOW = 20


def add_segmentation_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--seg-backend``, ``--seg-weights`` and ``--segformer-size``."""
    parser.add_argument(
        "--seg-backend", choices=BACKENDS, default="bisenetv2",
        help="segmentation model for the segmentation layer (default: the A1 BiSeNetV2)",
    )  # fmt: skip
    parser.add_argument(
        "--seg-weights", default=None,
        help="weights path or model id for --seg-backend (default per backend)",
    )  # fmt: skip
    parser.add_argument(
        "--segformer-size", type=int, nargs=2, default=(512, 256), metavar=("W", "H"),
        help="SegFormer input size; 512 256 matches BiSeNetV2's (default)",
    )  # fmt: skip


def require_real_weights(path: str) -> None:
    """Exit with a clear message if ``path`` is missing or a Git LFS pointer.

    A GitHub ZIP download contains ~130-byte LFS pointer text in place of .pth
    files, which otherwise fails deep inside torch.load with an unhelpful
    "Unsupported operand 118" error.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(64)
    except OSError:
        sys.exit(f"Model weights not found: {path}")
    if head.startswith(b"version https://git-lfs"):
        sys.exit(
            f"{path} is a Git LFS pointer, not the weights. Download the real file "
            "(GitHub: open the file -> 'Download raw file', or `git lfs pull`), then "
            "check it with: python scripts/verify_model_asset.py"
        )


def required_weight_files(args: argparse.Namespace, *always: str) -> list[str]:
    """Local weight files the demo needs before it loads anything."""
    files = list(always)
    if args.seg_backend == "bisenetv2":
        files.append(args.seg_weights or SEGMENTATION_WEIGHTS)
    return files


class TimedSegmenter:
    """Wraps any segmenter and records how long each predict() call takes."""

    def __init__(self, segmenter, window: int = SEG_TIMING_WINDOW):
        self.segmenter = segmenter
        self.name = getattr(segmenter, "name", "bisenetv2")
        self.recent_ms: deque[float] = deque(maxlen=window)

    def predict(self, frame_rgb: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        mask = self.segmenter.predict(frame_rgb)
        self.recent_ms.append((time.perf_counter() - start) * 1000.0)
        return mask

    def visualize(self, frame_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        return self.segmenter.visualize(frame_rgb, mask, alpha=alpha)

    @property
    def mean_ms(self) -> float | None:
        return sum(self.recent_ms) / len(self.recent_ms) if self.recent_ms else None

    def timing_text(self) -> str:
        """Short "name 12.3 ms/frame" text for console logs ("" before the first call)."""
        return f"{self.name} {self.mean_ms:.1f} ms/frame" if self.mean_ms else ""


def build_segmenter(backend: str, weights: str | None, segformer_size) -> object:
    """Create the segmenter chosen on the command line.

    BiSeNetV2 keeps the A1 weights path and 512x256 input. YOLO-seg prefers
    weights already downloaded to the project root (one level up from src/)
    so running from src/ does not download them a second time.
    """
    if backend == "bisenetv2":
        return create_segmenter("bisenetv2", weights or SEGMENTATION_WEIGHTS, input_size=(512, 256))
    if backend == "segformer_b0":
        return create_segmenter("segformer_b0", weights, input_size=tuple(segformer_size))
    if weights is None:
        candidate = f"../{DEFAULT_WEIGHTS['yolo_seg']}"
        try:
            with open(candidate, "rb"):
                weights = candidate
        except OSError:
            weights = None  # Ultralytics downloads the default on first use
    return create_segmenter("yolo_seg", weights)


def build_demo_segmenter(args: argparse.Namespace) -> TimedSegmenter:
    """Build the segmenter from parsed demo arguments, wrapped for timing."""
    segmenter = TimedSegmenter(build_segmenter(args.seg_backend, args.seg_weights, args.segformer_size))
    print(f"Segmentation layer: {segmenter.name}")
    return segmenter
