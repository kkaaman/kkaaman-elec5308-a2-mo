import unittest

import numpy as np

from traffic_perception import (
    BoundingBox,
    ModelPrediction,
    TrafficCategory,
    TrafficPerception,
    estimate_distance_m,
)


class TrafficPerceptionTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_distance_estimation(self):
        distance_m = estimate_distance_m(
            observed_width_px=50.0,
            real_width_m=0.6,
            focal_length_px=1000.0,
        )

        self.assertAlmostEqual(distance_m, 12.0)

    def test_invalid_distance_input_is_rejected(self):
        with self.assertRaises(ValueError):
            estimate_distance_m(
                observed_width_px=0.0,
                real_width_m=0.6,
                focal_length_px=1000.0,
            )

    def test_low_confidence_prediction_is_filtered(self):
        def backend(_image):
            return (
                ModelPrediction(
                    category=TrafficCategory.SIGN,
                    label="stop",
                    confidence=0.4,
                    bounding_box=BoundingBox(10, 20, 50, 60),
                ),
            )

        perception = TrafficPerception(
            backend,
            confidence_threshold=0.5,
        )

        self.assertEqual(perception.detect(self.image), ())

    def test_detection_preserves_model_output(self):
        box = BoundingBox(10, 20, 50, 60)

        def backend(_image):
            return (
                ModelPrediction(
                    category=TrafficCategory.SIGN,
                    label="stop",
                    confidence=0.9,
                    bounding_box=box,
                ),
            )

        perception = TrafficPerception(backend)

        detection = perception.detect(self.image)[0]

        self.assertEqual(detection.category, TrafficCategory.SIGN)
        self.assertEqual(detection.label, "stop")
        self.assertEqual(detection.bounding_box, box)
        self.assertAlmostEqual(detection.confidence, 0.9)

    def test_distance_is_added_when_calibration_is_available(self):
        def backend(_image):
            return (
                ModelPrediction(
                    category=TrafficCategory.SIGN,
                    label="stop",
                    confidence=0.9,
                    bounding_box=BoundingBox(10, 20, 60, 70),
                ),
            )

        perception = TrafficPerception(
            backend,
            focal_length_px=1000.0,
            object_widths_m={"stop": 0.6},
        )

        detection = perception.detect(self.image)[0]

        self.assertAlmostEqual(detection.distance_m, 12.0)

    def test_unknown_object_width_leaves_distance_unset(self):
        def backend(_image):
            return (
                ModelPrediction(
                    category=TrafficCategory.LIGHT,
                    label="red",
                    confidence=0.9,
                    bounding_box=BoundingBox(10, 20, 30, 60),
                ),
            )

        perception = TrafficPerception(
            backend,
            focal_length_px=1000.0,
            object_widths_m={"stop": 0.6},
        )

        detection = perception.detect(self.image)[0]

        self.assertIsNone(detection.distance_m)

    def test_invalid_image_shape_is_rejected(self):
        perception = TrafficPerception(lambda _image: ())

        image = np.zeros((480, 640), dtype=np.uint8)

        with self.assertRaises(ValueError):
            perception.detect(image)

    def test_model_prediction_rejects_nan_confidence(self):
        with self.assertRaises(ValueError):
            ModelPrediction(
                category=TrafficCategory.SIGN,
                label="stop",
                confidence=float("nan"),
                bounding_box=BoundingBox(
                    x_min=0,
                    y_min=0,
                    x_max=10,
                    y_max=10,
                ),
            )


if __name__ == "__main__":
    unittest.main()
