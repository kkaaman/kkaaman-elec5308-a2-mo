"""Numerical BL-D-01 examples; production transforms arrive in BL-D-02."""
import json
import math
import unittest
from pathlib import Path

import numpy as np


class RGBDSpecificationTests(unittest.TestCase):
    def test_raw_bgra_rgb_order(self):
        pixel = np.array([10, 20, 200, 255], dtype=np.uint8)
        self.assertEqual(pixel[[2, 1, 0]].tolist(), [200, 20, 10])

    def test_depth_bytes_and_full_scale(self):
        # R least significant, B most significant; promote before multiplying.
        bgra = np.array([[0, 0, 1, 255], [0, 1, 0, 255], [1, 0, 0, 255], [255, 255, 255, 255]], dtype=np.uint8).astype(np.uint32)
        encoded = bgra[:, 2] + 256 * bgra[:, 1] + 65536 * bgra[:, 0]
        self.assertEqual(encoded.tolist(), [1, 256, 65536, 16777215])
        self.assertEqual(float(1000 * (encoded[-1] / 16777215)), 1000)
        half_scale = 16777215 // 2
        half_depth_m = 1000 * half_scale / 16777215
        self.assertAlmostEqual(half_depth_m, 500.0, places=3)

    def test_off_axis_axial_and_range_are_distinct(self):
        ray_range = math.sqrt(5**2 + 10**2)
        self.assertAlmostEqual(ray_range, 11.1803398875, places=9)
        self.assertAlmostEqual(ray_range / math.sqrt(1 + 0.5**2), 10)
        self.assertGreater(abs(ray_range - 10), 1)

    def test_worked_optical_ego_world_coordinate(self):
        z = 10
        optical = np.array([(680 - 480) / 400 * z, (310 - 270) / 400 * z, z])
        np.testing.assert_allclose(optical, [5, 1, 10])
        optical_to_ego = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        ego = optical_to_ego @ optical + [1.5, 0, 1.7]
        np.testing.assert_allclose(ego, [11.5, -5, 0.7])
        yaw_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        world = yaw_rotation @ ego + [100, 20, 0]
        np.testing.assert_allclose(world, [105, 31.5, 0.7])
        self.assertAlmostEqual(float(np.linalg.det(optical_to_ego)), 1)

    def test_rotated_footprint_not_only_centre(self):
        # Centre Y=2.2 lies outside corridor [-1,1], but a 90-degree rotated
        # length-4 vehicle reaches Y=0.2 and overlaps it.
        center_y, half_length = 2.2, 2
        self.assertGreater(center_y, 1)
        self.assertLess(center_y - half_length, 1)

    def test_s4_completion_not_applicable_and_results_not_run(self):
        path = Path(__file__).resolve().parents[1] / "fixtures/acceptance_definitions.json"
        definitions = json.loads(path.read_text())
        self.assertTrue(any(r["scenario"] == "S4" for r in definitions["not_applicable"]))
        for row in definitions["metrics"]:
            self.assertEqual(row["result_status"], "NOT RUN" if row["applicable"] else "NOT APPLICABLE")
            self.assertFalse(row["scenario"] == "S4" and row["metric"].startswith("Route completion") and row["applicable"])

    def test_s0_thresholds_match_plan(self):
        path = Path(__file__).resolve().parents[1] / "fixtures/acceptance_definitions.json"
        rows = {r["metric"]: r for r in json.loads(path.read_text())["metrics"] if r["scenario"] == "S0"}
        self.assertEqual(rows["Route completion (minimum per run)"]["lower"], 1)
        self.assertEqual(rows["Cross-track mean (worst run)"]["upper"], 0.3)
        self.assertEqual(rows["Capture-to-apply p95 (worst run)"]["upper"], 100)
        self.assertTrue(all(row["required_runs"] == 5 for row in rows.values()))
