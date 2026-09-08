from pathlib import Path
import sys

import cv2

CURRENT_FILE = Path(__file__).resolve()
SRC_FOLDER = CURRENT_FILE.parent.parent

sys.path.insert(0, str(SRC_FOLDER))

from perception import ObjectDetector


VIDEO_PATH = CURRENT_FILE.parent / "test_video.mp4"

detector = ObjectDetector()
video = cv2.VideoCapture(str(VIDEO_PATH))

DISPLAY_WIDTH = 600


while True:
    success, frame_bgr = video.read()

    if not success:
        break

    # OpenCV gives BGR, our detector expects RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    detections = detector.track(frame_rgb)

    for detection in detections:
        box = detection.bounding_box

        x1 = int(box.x_min)
        y1 = int(box.y_min)
        x2 = int(box.x_max)
        y2 = int(box.y_max)

        label = (
            f"{detection.class_name} "
            f"{detection.confidence:.2f}"
        )

        if detection.track_id is not None:
            label += f" ID:{detection.track_id}"

        cv2.rectangle(
            frame_bgr,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame_bgr,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Resize only for displaying on screen
    height, width = frame_bgr.shape[:2]

    scale = DISPLAY_WIDTH / width
    display_height = int(height * scale)

    display_frame = cv2.resize(
        frame_bgr,
        (DISPLAY_WIDTH, display_height),
    )

    cv2.imshow("Object Tracking", display_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()