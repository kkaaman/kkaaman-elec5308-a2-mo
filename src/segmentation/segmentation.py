"""Simulator-agnostic semantic segmentation:
        Combines the preprocessing, bisenetv2 as well as postprocessing 
        scripts along with a path to the trained weights to produce the coloured 
        overlay of the semantic segmenter"""

import os

import cv2
import numpy as np
import torch

from .bisenetv2 import BiSeNetV2
from .postprocessing import overlay_mask
from .preprocessing import preprocess_image

DEFAULT_INPUT_SIZE: tuple[int, int] = (512, 256)  # (width, height)
DEFAULT_NUM_CLASSES = 19  # Cityscapes trainId classes


class SemanticSegmenter:
    """Loads a trained segmentation model and exposes a simulator-agnostic API.



    Args:
        weights_path: Path to trained model weights.
        input_size: (width, height) the model expects. Frames are resized
            to this before inference and the output mask is resized back
            up to the original input resolution.
        num_classes: Number of output classes (default: Cityscapes 19).
        device: "cuda" or "cpu". Auto-detected if not given.
    """

    def __init__(
        self,
        weights_path: str,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        num_classes: int = DEFAULT_NUM_CLASSES,
        device: str | None = None,
    ):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found at: {weights_path}")

        self.input_size = input_size
        self.num_classes = num_classes
        self.device = device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu")

        self.model = self._build_model(num_classes)
        self._load_weights(weights_path)
        self.model.eval()
        self.model.to(self.device)

    def _build_model(self, num_classes: int) -> torch.nn.Module:
        """Instantiate the architecture (weights not loaded yet).
        """
        return BiSeNetV2(num_classes=num_classes)

    def _load_weights(self, weights_path: str) -> None:
        """Load a trained state dict into ``self.model``.

        Accepts either a bare state dict or a checkpoint dict containing one
        under the key "model_state_dict".
        """
        checkpoint = torch.load(weights_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run semantic segmentation on a single RGB frame.

        Args:
            image: HxWx3 uint8 RGB array, any input resolution.

        Returns:
            HxW uint8 array of per-pixel class indices, at the *original*
            input resolution (not the model's internal input_size).

        Raises:
            ValueError: If ``image`` is not an HxWx3 uint8 RGB array.
        """
        self._validate_input(image)
        original_shape = image.shape[:2]

        tensor = preprocess_image(image, self.input_size).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        mask = self._resize_mask(mask, original_shape)
        return mask.astype(np.uint8)

    def visualize(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay a predicted mask on the source image for display.

        Args:
            image: HxWx3 uint8 RGB array matching ``mask``'s resolution.
            mask: HxW class-index array, as returned by :meth:`predict`.
            alpha: Blend weight for the mask overlay.

        Returns:
            HxWx3 uint8 RGB array.
        """
        return overlay_mask(image, mask, alpha=alpha)

    @staticmethod
    def _validate_input(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected a numpy array, got {type(image)}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 RGB array, got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"Expected dtype uint8, got {image.dtype}")

    @staticmethod
    def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        height, width = target_shape
        return cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        )
