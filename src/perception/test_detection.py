from pathlib import Path
import sys

import numpy as np
from PIL import Image


# Add the src folder to Python's import path.
CURRENT_FILE = Path(__file__).resolve()
SRC_FOLDER = CURRENT_FILE.parent.parent

sys.path.insert(0, str(SRC_FOLDER))


from perception import ObjectDetector


# street.jpg is in the same folder as this test file.
IMAGE_PATH = CURRENT_FILE.parent / "street.jpg"


image = Image.open(IMAGE_PATH).convert("RGB")
image = np.array(image)

detector = ObjectDetector()

detections = detector.detect(image)

print(f"Detected {len(detections)} objects:")

for detection in detections:
    print(
        f"{detection.class_name} "
        f"confidence={detection.confidence:.2f} "
        f"box={detection.bounding_box}"
    )