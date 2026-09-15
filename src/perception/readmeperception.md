# Perception Module

This module provides object detection and multi-object tracking for RGB images.

It uses:
- YOLO for object detection
- BoT-SORT for tracking objects across frames
- Custom `Detection` objects so the rest of the project does not need to deal directly with Ultralytics outputs

## Files

- `types.py` — defines `Detection`, `BoundingBox`, and `PerceptionConfig`
- `object_detection.py` — contains the `ObjectDetector` class
- `__init__.py` — exposes the main public classes

## Input

The detector expects an RGB NumPy image with shape:

    (height, width, 3)

Example:

    (720, 1280, 3)

The module is simulator-independent. CARLA images should first be converted into a normal RGB NumPy image before being passed to the detector.

## Single-Image Detection

Use `detect()` when processing one independent image.

    from PIL import Image
    import numpy as np
    from perception import ObjectDetector

    detector = ObjectDetector()

    image = Image.open("street.jpg").convert("RGB")
    image = np.array(image)

    detections = detector.detect(image)

    for detection in detections:
        print(
            detection.class_name,
            detection.confidence,
            detection.bounding_box,
        )

Example output:

    car 0.86 BoundingBox(...)
    bus 0.85 BoundingBox(...)
    person 0.44 BoundingBox(...)

For normal detection, `track_id` will usually be `None`.

## Video / Camera Tracking

Use `track()` for consecutive frames from a video, camera, or CARLA simulation.

Create the detector once and reuse it for every frame:

    from perception import ObjectDetector

    detector = ObjectDetector()

    for frame in video_frames:
        detections = detector.track(frame)

        for detection in detections:
            print(
                detection.class_name,
                detection.confidence,
                detection.track_id,
                detection.bounding_box,
            )

Example:

    Frame 1: car ID:3
    Frame 2: car ID:3
    Frame 3: car ID:3

The same ID means BoT-SORT believes it is the same physical object across frames.

Do not create a new `ObjectDetector` inside the frame loop, because this resets tracking history.

## Detection Output

Each object is returned as a `Detection`:

    Detection(
        class_name="car",
        confidence=0.92,
        bounding_box=BoundingBox(
            x_min=255,
            y_min=316,
            x_max=353,
            y_max=399,
        ),
        track_id=4,
    )

Fields:

- `class_name` — object type, such as `car`, `person`, or `truck`
- `confidence` — YOLO confidence from 0 to 1
- `bounding_box` — location of the object in image pixels
- `track_id` — persistent ID when tracking, otherwise `None`

## Bounding Boxes

Bounding boxes contain:

    x_min = left edge
    y_min = top edge
    x_max = right edge
    y_max = bottom edge

They describe the rectangle around the detected object.

## Default Classes

The detector currently focuses on road-relevant COCO classes:

- person
- bicycle
- car
- motorcycle
- bus
- truck
- traffic light
- stop sign

These can be changed using `PerceptionConfig`.

Example:

    from perception import ObjectDetector, PerceptionConfig

    config = PerceptionConfig(
        allowed_classes=("car", "person", "truck"),
    )

    detector = ObjectDetector(config)

## Confidence Threshold

The confidence threshold controls how confident YOLO must be before returning a detection.

Example:

    config = PerceptionConfig(
        confidence_threshold=0.4,
    )

Higher threshold:
- fewer detections
- generally fewer false positives
- may miss difficult or distant objects

Lower threshold:
- more detections
- may detect small or distant objects
- may also produce more false positives

## Detection vs Tracking

Use:

    detector.detect(image)

when asking:

    "What objects are in this image?"

Use:

    detector.track(frame)

when asking:

    "What objects are in this frame, and which ones are the same objects seen in previous frames?"

## Visualisation

The green boxes shown in the video test are only for visualisation.

The real output is:

    detections = detector.track(frame)

Other modules can directly use the returned data.

Example:

    for detection in detections:
        if detection.class_name == "person":
            print("Pedestrian detected")

The perception module therefore outputs structured information that can later be used by control, navigation, evaluation, or CARLA integration code.