"""Legacy BGR interface. New callers should use perception.lane_inference."""

import cv2
import numpy as np

from perception.lane_inference import LaneSegmenter as RGBLaneSegmenter


class LaneSegmenter(RGBLaneSegmenter):
    """Compatibility wrapper preserving the legacy BGR input convention."""

    def predict(self, image):
        if image is None:
            raise ValueError("Input image cannot be None.")
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a three-channel BGR image.")
        if image.dtype != np.uint8 or 0 in image.shape:
            raise ValueError("Expected a nonempty uint8 BGR image.")
        return super().predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
