from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class LaneDataset(Dataset):
    def __init__(
        self,
        dataframe,
        data_dir,
        image_size=(360, 640),
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.image_size = image_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = self.data_dir / row["image_path"]
        mask_path = self.data_dir / row["mask_path"]

        image = cv2.imread(str(image_path))
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not read image at {image_path}"
            )

        if mask is None:
            raise FileNotFoundError(
                f"Could not read mask at {mask_path}"
            )

        height, width = self.image_size

        image = cv2.resize(
            image,
            (width, height),
        )

        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = image.astype("float32") / 255.0

        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        mask = (mask > 0).astype("int64")
        mask = torch.from_numpy(mask)

        return image, mask