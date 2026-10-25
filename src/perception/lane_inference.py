# FIXED (Mohammad): predict() was converting BGR->RGB internally, assuming
# callers pass BGR. lane_dataset.py trains on RGB tensors and the shared
# CARLA adapter hands out RGB frames, so this was silently swapping channels
# on already-RGB input. Removed the conversion; see inline comment below.
#
# FIXED (Mohammad): import path was 'autodrive.perception.lane_segmentation',
# which does not exist anywhere in the repo -- this file (and lane_segmentation.py)
# actually live flat at src/perception/, not nested under autodrive. Corrected
# to match the real location; this was breaking the module on any fresh clone.
import cv2
import numpy as np
import torch

from perception.lane_segmentation import LaneUNet


def _get_default_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


class LaneSegmenter:
    """ResNet18 LaneUNet lane-marking segmenter on RGB frames.

    ``checkpoint_path`` must be a complete tensor-only state dict (loaded with
    ``weights_only=True``, strict); ``image_size`` is the model's
    ``(height, width)`` in pixels (default 360x640). Frames are resized to it
    for inference and the mask is resized back with nearest-neighbour
    interpolation. No pretrained download is attempted.
    """

    def __init__(
        self,
        checkpoint_path,
        image_size=(360, 640),
        device=None,
    ):
        self.image_size = image_size

        if device is None:
            device = _get_default_device()

        self.device = torch.device(device)

        self.model = LaneUNet(
            num_classes=2, pretrained=False
        ).to(self.device)

        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, image):
        """Return an ``HxW`` uint8 lane mask (1 = lane marking) at input resolution.

        ``image`` must be a nonempty ``HxWx3`` uint8 RGB array (CARLA BGRA is
        converted at the adapter boundary; the legacy BGR wrapper lives in
        ``autodrive.perception.lane_inference``). The mask belongs to this frame
        only; it carries no timestamp, so callers attach frame identity before
        passing it to ``perception.lane_path.extract_ego_path`` or the RV-06
        departure monitor. Raises ``ValueError`` / ``TypeError`` on bad input.
        """
        if image is None:
            raise ValueError("Input image cannot be None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Input image must be a NumPy array."
            )

        if image.ndim != 3 or image.shape[2] != 3 or 0 in image.shape:
            raise ValueError("Expected a nonempty HxWx3 RGB image.")
        if image.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB image.")

        original_height, original_width = image.shape[:2]

        resized = cv2.resize(
            image,
            (
                self.image_size[1],
                self.image_size[0],
            ),
        )

        # FIXED (Mohammad): removed cv2.cvtColor(resized, COLOR_BGR2RGB) here.
        # Input is already RGB; that call was flipping it to BGR instead.
        tensor = torch.from_numpy(resized).float() / 255.0

        tensor = tensor.permute(
            2,
            0,
            1,
        )

        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

            prediction = torch.argmax(
                logits,
                dim=1,
            )

        mask = (
            prediction[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        mask = cv2.resize(
            mask,
            (
                original_width,
                original_height,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        return mask
