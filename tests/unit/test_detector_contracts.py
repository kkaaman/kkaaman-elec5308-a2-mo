"""BL-C-01 conformance and invalid-observation tests; no learned model."""
import unittest
from dataclasses import replace

import numpy as np

from autonomy.detector_contracts import Detector
from autonomy.interfaces import CameraCalibration, EgoPose, FrameBundle, VehicleState
from obstacle_perception.types import (
    CallTiming,
    DetectionBatch,
    DetectionObservation,
    ObservationMetadata,
    Obstacle,
    PixelBox,
)


def metadata():
    return ObservationMetadata("episode", 4, 0.2, "camera-front", "fixture-v1")


def detection(track_id=None):
    return DetectionObservation("car", 0.9, PixelBox(1, 1, 3, 3), track_id)


def batch():
    return DetectionBatch(metadata(), (detection(),), 4, 4, CallTiming(0.01))


class FakeLocalDetector:
    def reset(self, *, episode_id):
        self.episode = episode_id
        self.frames = []

    def detect(self, frame):
        if frame.episode_id != self.episode:
            raise ValueError("episode mismatch")
        info = ObservationMetadata(frame.episode_id, frame.frame_id, frame.sim_time_s, "camera-front", "fixture-v1")
        return DetectionBatch(info, (detection(),), 4, 4, CallTiming(0.01))

    def track(self, frame):
        result = self.detect(frame)
        if self.frames and frame.frame_id <= self.frames[-1]:
            raise ValueError("non-increasing frame")
        self.frames.append(frame.frame_id)
        return replace(result, detections=(detection(0),))

    def close(self):
        self.frames = []


class FakeRemoteDetector(FakeLocalDetector):
    """Simulated response boundary; does not make network requests."""
    def detect(self, frame):
        return replace(super().detect(frame), timing=CallTiming(0.03, 0.01, 0.02))


class DetectorContractTests(unittest.TestCase):
    def test_optional_track_id_and_namespaced_identity(self):
        self.assertIsNone(detection().track_id)
        info = metadata()
        self.assertNotEqual(info.track_identity(0), replace(info, episode_id="next").track_identity(0))
        self.assertNotEqual(info.track_identity(0), replace(info, source_id="rear").track_identity(0))
        self.assertNotEqual(info.track_identity(0), replace(info, model_version="v2").track_identity(0))

    def test_metadata_rejects_missing_model_and_bad_frame(self):
        for changes in ({"frame_id": True}, {"frame_id": -1}, {"model_version": ""}, {"sim_time_s": float("nan")}):
            with self.assertRaises(ValueError):
                replace(metadata(), **changes)

    def test_boxes_reject_bad_bounds(self):
        for box in ((1,1,1,3), (-1,1,3,3), (1,1,float("nan"),3)):
            with self.assertRaises(ValueError):
                PixelBox(*box)
        with self.assertRaises(ValueError):
            replace(batch(), image_width=2)

    def test_confidence_rejects_nan_and_out_of_range(self):
        for confidence in (-0.1, 1.1, float("nan"), True):
            with self.assertRaises(ValueError):
                replace(detection(), confidence=confidence)

    def test_duplicate_tracks_rejected(self):
        with self.assertRaises(ValueError):
            replace(batch(), detections=(detection(1), detection(1)))

    def test_invalid_is_distinct_from_valid_empty(self):
        self.assertTrue(replace(batch(), detections=()).valid)
        invalid = replace(batch(), detections=(), valid=False, invalid_reason="timeout")
        self.assertFalse(invalid.valid)
        with self.assertRaises(ValueError):
            replace(batch(), valid=False)
        with self.assertRaises(ValueError):
            replace(batch(), valid=False, invalid_reason="timeout")

    def test_timing_rejects_negative_and_nan(self):
        for value in (-1, float("nan")):
            with self.assertRaises(ValueError):
                CallTiming(value)

    def test_obstacle_extent_and_identity(self):
        obstacle = Obstacle(metadata(), "car", 0.8, [10, 1, 0], [4, 2, 1.5], track_id=0)
        self.assertEqual(obstacle.center_ego_m, (10,1,0))
        self.assertEqual(obstacle.identity, metadata().track_identity(0))
        with self.assertRaises(ValueError):
            replace(obstacle, extent_m=(4,0,1))
        with self.assertRaises(ValueError):
            replace(obstacle, center_ego_m=(1,2))

    def test_local_remote_protocol_and_episode_reset(self):
        calib = CameraCalibration(100,100,2,2, ((0,0,1,0),(-1,0,0,0),(0,-1,0,0),(0,0,0,1)))
        frame = FrameBundle("episode",4,0.2,10,EgoPose(VehicleState(0,0,0,0)),np.zeros((4,4,3),dtype=np.uint8),calib)
        local, remote = FakeLocalDetector(), FakeRemoteDetector()
        for model in (local, remote):
            self.assertIsInstance(model, Detector)
            model.reset(episode_id="episode")
            detected = model.detect(frame)
            self.assertEqual(detected.metadata.frame_id, frame.frame_id)
            self.assertIsNone(detected.detections[0].track_id)
            self.assertEqual(model.track(frame).detections[0].track_id, 0)
            with self.assertRaises(ValueError):
                model.track(frame)
            model.reset(episode_id="next")
            with self.assertRaises(ValueError):
                model.detect(frame)
            self.assertEqual(model.track(replace(frame, episode_id="next", frame_id=0)).detections[0].track_id, 0)
            model.close()
        local.reset(episode_id="episode")
        remote.reset(episode_id="episode")
        self.assertEqual(local.detect(frame).detections, remote.detect(frame).detections)

    def test_obstacle_source_rejects_wrong_frame_and_invalid_depth(self):
        from autonomy.detector_contracts import ObstacleSource
        class FakeSource:
            def reset(self, *, episode_id):
                self.episode = episode_id
            def observe(self, frame, detections):
                info = detections.metadata
                if not detections.valid or frame.depth_m is None:
                    raise ValueError("invalid sensor observation")
                if (info.episode_id, info.frame_id, info.sim_time_s) != (frame.episode_id, frame.frame_id, frame.sim_time_s):
                    raise ValueError("frame mismatch")
                return ()
            def close(self):
                pass
        calib = CameraCalibration(100,100,2,2, ((0,0,1,0),(-1,0,0,0),(0,-1,0,0),(0,0,0,1)))
        frame = FrameBundle("episode",4,0.2,10,EgoPose(VehicleState(0,0,0,0)),np.zeros((4,4,3),dtype=np.uint8),calib)
        source = FakeSource()
        self.assertIsInstance(source, ObstacleSource)
        source.reset(episode_id="episode")
        with self.assertRaises(ValueError):
            source.observe(frame, batch())
        frame = replace(frame, depth_m=np.ones((4,4)), depth_frame_id=4)
        self.assertEqual(source.observe(frame, replace(batch(), detections=())), ())
        with self.assertRaises(ValueError):
            source.observe(frame, replace(batch(), metadata=replace(metadata(), frame_id=3)))
        with self.assertRaises(ValueError):
            source.observe(frame, replace(batch(), detections=(), valid=False, invalid_reason="timeout"))
