"""Preprocessing pipeline for the semantic segmentation input frames."""

import cv2
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], dtype = np.float32)
STANDARD_DEV = np.array([0.229, 0.224, 0.225], dtype = np.float32)



def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an image to ``size`` with bilinear interpolation.

    Args:
        image: HxWx3 uint8 RGB array.
        size: Target (width, height) in pixels, both positive.

    Returns:
        Resized HxWx3 uint8 RGB array.
    """
    width, height = size
    return cv2.resize(image, (width, height), interpolation = cv2.INTER_LINEAR)



def normalize_image(image: np.ndarray) -> np.ndarray:
    """Scale an image to [0, 1] and apply ImageNet mean/std normalisation.

    Args:
        image: HxWx3 uint8 RGB array.

    Returns:
        Normalised HxWx3 float32 array (dimensionless, roughly zero mean).
    """

    norm_arr = image.astype(np.float32) / 255.0
    norm_arr = (norm_arr - MEAN) / STANDARD_DEV


    return norm_arr




def preprocess_image(image: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """Resize, normalise and convert an RGB frame into a batched tensor.

    Args:
        image: HxWx3 uint8 RGB array at input resolution.
        size: Model input (width, height) in pixels.

    Returns:
        1x3xHxW float32 torch tensor (batch, channel, row, column).
    """

    resized = resize_image(image, size)
    normalized = normalize_image(resized)
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor
