"""BL-F-03 sensor fixture checks, independent of obstacle scene truth."""

from dataclasses import replace

import numpy as np
import pytest

from autonomy.detector_contracts import Detector, ObstacleSource
from autonomy.stack import SkeletonStack, StackConfig
from navigation_control.types import ControlCommand
from obstacle_perception.types import DetectionObservation, PixelBox
from sim_harness import (
    KinematicWorld,
    SyntheticDetection,
    SyntheticDetector,
    SyntheticFrame,
    SyntheticObstacleSource,
    SyntheticScenario,
)


def observation(depth=12.0, box=None, track_id=3):
    return SyntheticDetection(
        DetectionObservation("car", 0.9, box or PixelBox(60, 30, 80, 50), track_id),
        depth,
    )


def setup(script=None):
    scenario = SyntheticScenario((script or SyntheticFrame(0, (observation(),)),))
    world = KinematicWorld(scenario=scenario, clock=lambda: 10.0)
    frame, graph = world.reset(seed=7)
    detector = SyntheticDetector(scenario)
    source = SyntheticObstacleSource(extent_prior_m=(4, 2, 1.5))
    detector.reset(episode_id=frame.episode_id)
    source.reset(episode_id=frame.episode_id)
    return world, frame, graph, detector, source


def test_protocols_depth_alignment_projection_and_namespaced_track():
    _, frame, _, detector, source = setup()
    assert isinstance(detector, Detector) and isinstance(source, ObstacleSource)
    batch = detector.detect(frame)
    assert batch == detector.track(frame)
    assert batch.metadata.frame_id == frame.depth_frame_id == frame.frame_id
    assert batch.metadata.sim_time_s == frame.sim_time_s
    assert np.isnan(frame.depth_m[0, 0])
    assert frame.depth_m[40, 70] == 12
    np.testing.assert_equal(frame.rgb[40, 70], (90, 150, 210))
    (obstacle,) = source.observe(frame, batch)
    # fx=80; optical u=70 is to the left of cx=80 -> ego Y positive.
    assert obstacle.center_ego_m == pytest.approx((12, 1.5, 2.25))
    assert obstacle.extent_m == (4, 2, 1.5)
    assert obstacle.identity == batch.metadata.track_identity(3)
    assert obstacle.confidence == 0.9 and obstacle.class_name == "car"


def test_missing_track_id_and_two_disjoint_observations():
    a = observation(track_id=None)
    b = observation(20, PixelBox(90, 30, 110, 50), 9)
    _, frame, _, detector, source = setup(SyntheticFrame(0, (a, b)))
    obstacles = source.observe(frame, detector.detect(frame))
    assert obstacles[0].identity is None
    assert obstacles[1].center_ego_m[0] == pytest.approx(20)
    assert obstacles[1].center_ego_m[1] < 0


@pytest.mark.parametrize(
    "script",
    [
        SyntheticFrame(0, (observation(None),)),
        SyntheticFrame(0, (observation(),), depth_available=False),
        SyntheticFrame(0, depth_available=False),
        SyntheticFrame(0, valid=False, invalid_reason="camera_unavailable"),
    ],
)
def test_unknown_depth_and_invalid_observations_are_not_clear_space(script):
    _, frame, _, detector, source = setup(script)
    batch = detector.detect(frame)
    with pytest.raises(ValueError):
        source.observe(frame, batch)


@pytest.mark.parametrize("kind", ["episode", "frame", "time", "dimensions"])
def test_source_rejects_mismatched_batch(kind):
    _, frame, _, detector, source = setup()
    batch = detector.detect(frame)
    if kind == "episode":
        batch = replace(batch, metadata=replace(batch.metadata, episode_id="other"))
    elif kind == "frame":
        batch = replace(batch, metadata=replace(batch.metadata, frame_id=1))
    elif kind == "time":
        batch = replace(batch, metadata=replace(batch.metadata, sim_time_s=0.05))
    else:
        batch = replace(batch, image_width=200)
    with pytest.raises(ValueError):
        source.observe(frame, batch)


def test_script_replays_on_reset_without_old_episode_identity_or_state():
    world, old, _, detector, source = setup()
    old_batch = detector.detect(old)
    (old_obstacle,) = source.observe(old, old_batch)
    frame, _ = world.reset(seed=7)
    with pytest.raises(ValueError):
        detector.detect(frame)
    detector.reset(episode_id=frame.episode_id)
    source.reset(episode_id=frame.episode_id)
    with pytest.raises(ValueError):
        detector.detect(old)
    with pytest.raises(ValueError):
        source.observe(old, old_batch)
    (obstacle,) = source.observe(frame, detector.detect(frame))
    assert obstacle.identity != old_obstacle.identity
    assert obstacle.center_ego_m == old_obstacle.center_ego_m
    np.testing.assert_equal(old.depth_m, frame.depth_m)


def test_unspecified_frames_are_valid_empty_and_arrays_are_independent():
    scenario = SyntheticScenario()
    detector = SyntheticDetector(scenario)
    world = KinematicWorld(scenario=scenario)
    frame, _ = world.reset(seed=0)
    detector.reset(episode_id=frame.episode_id)
    source = SyntheticObstacleSource()
    source.reset(episode_id=frame.episode_id)
    batch = detector.detect(frame)
    assert batch.valid and not batch.detections
    assert source.observe(frame, batch) == ()
    assert np.isnan(frame.depth_m).all()
    rgb, depth = scenario.sensor_arrays(0)
    rgb[:] = 255
    depth[:] = 5
    assert not frame.rgb.any() and np.isnan(frame.depth_m).all()


def test_script_validation_rejects_ambiguous_rois_ids_and_invalid_depth():
    for depth in (0, -1, True, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            observation(depth)
    for frames in (
        (SyntheticFrame(0), SyntheticFrame(0)),
        (SyntheticFrame(0, (observation(), observation(track_id=4))),),
        (SyntheticFrame(0, (observation(), observation(box=PixelBox(90, 30, 110, 50)))),),
        (SyntheticFrame(0, (observation(box=PixelBox(150, 30, 170, 50)),)),),
    ):
        with pytest.raises(ValueError):
            SyntheticScenario(frames)
    with pytest.raises(ValueError):
        SyntheticScenario(image_width=True)
    with pytest.raises(ValueError):
        SyntheticScenario(horizontal_fov_rad=True)
    with pytest.raises(ValueError):
        SyntheticScenario(camera_height_m=-1)
    with pytest.raises(ValueError):
        SyntheticFrame(0, valid=False)
    with pytest.raises(ValueError):
        SyntheticFrame(0, (observation(),), valid=False, invalid_reason="bad")


def test_detector_checks_frame_geometry_and_both_components_close_idempotently():
    _, frame, _, detector, source = setup()
    with pytest.raises(ValueError):
        detector.detect(replace(frame, calibration=replace(frame.calibration, fx_px=100)))
    with pytest.raises(ValueError):
        detector.detect(
            replace(
                frame, rgb=np.zeros((90, 170, 3), dtype=np.uint8), depth_m=None, depth_frame_id=None
            )
        )
    for component in (detector, source):
        component.close()
        component.close()
        with pytest.raises(RuntimeError):
            component.reset(episode_id=frame.episode_id)
    with pytest.raises(RuntimeError):
        detector.detect(frame)
    with pytest.raises(RuntimeError):
        source.observe(frame, None)


def test_unreset_components_and_extent_validation():
    _, frame, _, _, _ = setup()
    with pytest.raises(RuntimeError):
        SyntheticDetector().detect(frame)
    with pytest.raises(RuntimeError):
        SyntheticObstacleSource().observe(frame, None)
    with pytest.raises(ValueError):
        SyntheticObstacleSource(extent_prior_m=(1, 0, 1))


def test_stack_policy_consumes_scripted_distance_then_fault_latches_brake():
    scenario = SyntheticScenario(
        (
            SyntheticFrame(0, (observation(12),)),
            SyntheticFrame(1, (observation(8),)),
            SyntheticFrame(2, (observation(None),)),
            SyntheticFrame(3, (observation(20),)),
        )
    )
    world = KinematicWorld(scenario=scenario, clock=lambda: 10.0)
    detector = SyntheticDetector(scenario)
    source = SyntheticObstacleSource()
    observed_distances = []

    class ObservationConsumer:
        def reset(self, *, episode_id, graph):
            source.reset(episode_id=episode_id)

        def __call__(self, frame, batch):
            obstacles = source.observe(frame, batch)
            observed_distances.append(obstacles[0].center_ego_m[0])
            # A fixed motion command exercises the data path; this fixture is
            # deliberately not a production distance governor (BL-F-06).
            return ControlCommand(0, 0.25, 0)

        def close(self):
            source.close()

    stack = SkeletonStack(
        detector,
        StackConfig(max_step_wall_s=5, require_depth=True),
        ObservationConsumer(),
        clock=lambda: 10.0,
    )
    try:
        frame, graph = world.reset(seed=0)
        stack.reset(episode_id=frame.episode_id, graph=graph)
        for index in range(4):
            packet = stack.step(frame)
            if index < 2:
                assert packet.diagnostics.observation_valid
                assert packet.command.throttle == 0.25
            else:
                assert not packet.diagnostics.observation_valid
                assert packet.command == ControlCommand(0, 0, 1)
            frame = world.step(packet)
        assert observed_distances == pytest.approx([12, 8])
    finally:
        stack.close()
        world.close()
