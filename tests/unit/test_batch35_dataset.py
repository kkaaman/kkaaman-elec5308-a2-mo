"""Offline data-contract, split, metric, queue and source-export regressions."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from carla_adapter.dataset_capture import FrameBuffer
from evaluation.dataset import (
    KIT_VERSION,
    confusion,
    decode_bgra,
    decode_depth,
    metrics,
    read_dataset,
    road_splits,
    save_json,
    sha256,
    visible_boxes,
)
from scripts.benchmark_yolo import iou, match_boxes


class DataTests(unittest.TestCase):
    def test_road_split_independent_of_input_order(self):
        roads = list(range(20))
        self.assertEqual(road_splits(roads), road_splits(reversed(roads)))
        self.assertEqual(set(road_splits(roads).values()), {"train", "val", "test"})

    def test_insufficient_roads_rejected(self):
        with self.assertRaises(ValueError):
            road_splits([1, 2, 3, 4])

    def test_red_channel_is_carla_label(self):
        raw = bytes([7, 9, 24, 255, 24, 0, 1, 255])
        image = decode_bgra(raw, 2, 1)
        self.assertEqual((image[:, :, 2] == 24).tolist(), [[True, False]])

    def test_bad_bgra_buffer_rejected(self):
        with self.assertRaises(ValueError):
            decode_bgra(b"bad", 2, 1)

    def test_depth_encoding_and_invalid_saturation(self):
        encoded = int(round(50.0 / 1000 * 16777215))
        image = np.array(
            [
                [
                    [0, 0, 0, 255],
                    [encoded >> 16, (encoded >> 8) & 255, encoded & 255, 255],
                    [255, 255, 255, 255],
                ]
            ],
            dtype=np.uint8,
        )
        decoded = decode_depth(image)
        self.assertTrue(np.isnan(decoded[0, 0]))
        self.assertTrue(np.isnan(decoded[0, 2]))
        self.assertAlmostEqual(float(decoded[0, 1]), 50, places=3)

    def test_instance_boxes_keep_class_and_half_open_extent(self):
        image = np.zeros((10, 12, 4), dtype=np.uint8)
        image[1:5, 2:6, :3] = [8, 9, 14]
        image[5:9, 7:11, :3] = [8, 9, 12]
        boxes = visible_boxes(image)
        self.assertEqual(len(boxes), 2)
        car = next(b for b in boxes if b["class_name"] == "car")
        self.assertEqual(car["xyxy"], [2, 1, 6, 5])
        self.assertEqual(car["visible_pixels"], 16)

    def test_metrics_known_confusion(self):
        counts = confusion(np.array([[1, 1], [0, 0]]), np.array([[1, 0], [1, 0]]))
        self.assertEqual(counts.tolist(), [1, 1, 1, 1])
        self.assertAlmostEqual(metrics(counts)["foreground_iou"], 1 / 3)

    def test_empty_foreground_not_perfect_score(self):
        self.assertIsNone(metrics([0, 0, 0, 30])["foreground_iou"])

    def test_bad_labels_rejected(self):
        with self.assertRaises(ValueError):
            confusion(np.array([[2]]), np.array([[1]]))

    def test_duplicate_detections_count_false_positive(self):
        predictions = [
            {"xyxy": [0, 0, 4, 4], "confidence": 0.8},
            {"xyxy": [0, 0, 4, 4], "confidence": 0.6},
        ]
        self.assertEqual(match_boxes(predictions, [{"xyxy": [0, 0, 4, 4]}]), (1, 1, 0))
        self.assertEqual(iou([0, 0, 1, 1], [2, 2, 3, 3]), 0)


class QueueTests(unittest.TestCase):
    def image(self, frame, time=1.0, shift=0):
        matrix = np.eye(4)
        matrix[0, 3] = shift
        return SimpleNamespace(
            frame=frame, timestamp=time, transform=SimpleNamespace(get_matrix=lambda: matrix)
        )

    def test_out_of_order_exact_join(self):
        frames = FrameBuffer(("rgb", "depth"))
        frames.put("rgb", self.image(10))
        frames.put("rgb", self.image(11))
        frames.put("depth", self.image(10))
        joined = frames.take(10, timeout=0)
        self.assertEqual({im.frame for im in joined.values()}, {10})

    def test_unmatched_frame_times_out(self):
        frames = FrameBuffer(("rgb", "depth"))
        frames.put("rgb", self.image(10))
        frames.put("depth", self.image(11))
        with self.assertRaises(TimeoutError):
            frames.take(10, timeout=0)

    def test_queue_is_bounded(self):
        frames = FrameBuffer(("rgb",), capacity=2)
        for frame in range(10):
            frames.put("rgb", self.image(frame))
        self.assertEqual(len(frames.buffers["rgb"]), 2)
        self.assertEqual(frames.evicted["rgb"], 8)

    def test_timestamp_mismatch_rejected(self):
        frames = FrameBuffer(("rgb", "depth"))
        frames.put("rgb", self.image(10, time=1))
        frames.put("depth", self.image(10, time=2))
        with self.assertRaises(ValueError):
            frames.take(10, timeout=0)

    def test_camera_pose_mismatch_rejected(self):
        frames = FrameBuffer(("rgb", "depth"))
        frames.put("rgb", self.image(10))
        frames.put("depth", self.image(10, shift=0.1))
        with self.assertRaises(ValueError):
            frames.take(10, timeout=0)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows = []
        for i, split in enumerate(("train", "val", "test")):
            paths = {
                kind: self.root / f"{i}_{kind}.{('npy' if kind == 'depth' else 'png')}"
                for kind in ("rgb", "lane", "instance", "depth")
            }
            Image.fromarray(np.full((6, 8, 3), i * 40, dtype=np.uint8)).save(paths["rgb"])
            Image.fromarray(np.ones((6, 8), dtype=np.uint8)).save(paths["lane"])
            instance = np.zeros((6, 8, 3), dtype=np.uint8)
            instance[:, :, 0] = 24
            Image.fromarray(instance).save(paths["instance"])
            np.save(paths["depth"], np.ones((6, 8), dtype=np.float32))
            self.rows.append(
                {
                    "sample_id": str(i),
                    "split": split,
                    "run_id": f"run{i}",
                    "road_id": i,
                    "frame_id": i,
                    "sensor_frames": {"rgb": i, "instance": i, "depth": i},
                    "sim_time_s": i * 0.05,
                    "sha256": {kind: sha256(path) for kind, path in paths.items()},
                    **{kind: path.name for kind, path in paths.items()},
                }
            )
        self.manifest = {
            "schema": KIT_VERSION,
            "label_tag": 24,
            "status": "COLLECTED",
            "sample_count": 3,
            "image_size_wh": [8, 6],
        }
        self.write()

    def tearDown(self):
        self.temporary.cleanup()

    def write(self):
        index = self.root / "samples.jsonl"
        index.write_text("\n".join(json.dumps(r) for r in self.rows) + "\n")
        self.manifest["samples_sha256"] = sha256(index)
        save_json(self.root / "manifest.json", self.manifest)

    def test_valid_dataset(self):
        _, rows = read_dataset(self.root)
        self.assertEqual(len(rows), 3)

    def test_road_leakage_rejected(self):
        self.rows[1]["road_id"] = self.rows[0]["road_id"]
        self.write()
        with self.assertRaisesRegex(ValueError, "leakage"):
            read_dataset(self.root)

    def test_run_leakage_rejected(self):
        self.rows[1]["run_id"] = self.rows[0]["run_id"]
        self.write()
        with self.assertRaisesRegex(ValueError, "leakage"):
            read_dataset(self.root)

    def test_image_leakage_rejected(self):
        self.rows[1]["rgb"] = self.rows[0]["rgb"]
        self.rows[1]["sha256"]["rgb"] = self.rows[0]["sha256"]["rgb"]
        self.write()
        with self.assertRaisesRegex(ValueError, "leakage"):
            read_dataset(self.root)

    def test_tampered_file_rejected(self):
        (self.root / self.rows[0]["rgb"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            read_dataset(self.root)

    def test_incomplete_dataset_rejected(self):
        self.manifest["status"] = "INCOMPLETE"
        self.write()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            read_dataset(self.root)

    def test_frame_mismatch_rejected(self):
        self.rows[0]["sensor_frames"]["depth"] = 100
        self.write()
        with self.assertRaisesRegex(ValueError, "frame mismatch"):
            read_dataset(self.root)

    def test_duplicate_sample_rejected(self):
        self.rows[1]["sample_id"] = self.rows[0]["sample_id"]
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_dataset(self.root)


if __name__ == "__main__":
    unittest.main()
