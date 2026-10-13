"""Simulator-agnostic semantic segmentation (BiSeNetV2, plus SegFormer-B0 and YOLO-seg backends)."""
from .backends import BACKENDS, create_segmenter
from .segmentation import SemanticSegmenter

__all__ = ["BACKENDS", "SemanticSegmenter", "create_segmenter"]
