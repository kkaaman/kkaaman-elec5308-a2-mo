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
            num_classes=2
        ).to(self.device)

        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, image):
        if image is None:
            raise ValueError("Input image cannot be None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Input image must be a NumPy array."
            )

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
