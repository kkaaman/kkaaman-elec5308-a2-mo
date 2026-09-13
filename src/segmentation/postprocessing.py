"""Postprocessing utilities: Allocates every pixel an integer 0-18 indicating which pixel belongs 
to which object, for example a car. Then converts it into a real RGB image by referring to the 
class palette. The script also blends the coloured mask into the original photo with 50% 
transparancy Allowing the segmentation results to be visualised."""


import cv2
import numpy as np

# Cityscapes 19-class palette (trainId -> RGB). Index 255 (ignore) is not drawn.
CITYSCAPES_PALETTE = np.array(
    [
        [128, 64, 128],   # 0 road
        [244, 35, 232],   # 1 sidewalk
        [70, 70, 70],     # 2 building
        [102, 102, 156],  # 3 wall
        [190, 153, 153],  # 4 fence
        [153, 153, 153],  # 5 pole
        [250, 170, 30],   # 6 traffic light
        [220, 220, 0],    # 7 traffic sign
        [107, 142, 35],   # 8 vegetation
        [152, 251, 152],  # 9 terrain
        [70, 130, 180],   # 10 sky
        [220, 20, 60],    # 11 person
        [255, 0, 0],      # 12 rider
        [0, 0, 142],      # 13 car
        [0, 0, 70],       # 14 truck
        [0, 60, 100],     # 15 bus
        [0, 80, 100],     # 16 train
        [0, 0, 230],      # 17 motorcycle
        [119, 11, 32],    # 18 bicycle
    ],
    dtype=np.uint8,
)


def colorize_mask(mask: np.ndarray, palette: np.ndarray | None = None) -> np.ndarray:
    """Map a class-index mask to an RGB image using a palette.

    Args:
        mask: HxW array of class indices.
        palette: Optional Nx3 uint8 array mapping class index -> RGB.
            Defaults to the Cityscapes 19-class palette.

    Returns:
        HxWx3 uint8 RGB array.
    """
    palette = CITYSCAPES_PALETTE if palette is None else palette
    safe_mask = np.clip(mask, 0, len(palette) - 1)
    return palette[safe_mask]


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a colorized mask over the source image for visualization.

    Args:
        image: HxWx3 uint8 RGB array, same resolution as ``mask``.
        mask: HxW array of class indices.
        alpha: Blend weight for the mask (0 = only image, 1 = only mask colors).

    Returns:
        HxWx3 uint8 RGB array.
    """
    color_mask = colorize_mask(mask)
    if color_mask.shape[:2] != image.shape[:2]:
        color_mask = cv2.resize(
            color_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
