"""E2-04: conservative fusion of LiDAR and RGB-D corridor evidence for the governor."""

from types import SimpleNamespace

import numpy as np
import pytest

from autonomy.interfaces import EgoPose, FrameBundle, VehicleState
from autonomy.lidar_fusion import FusedObservationProvider, LidarCorridorSource, rgbd_report
from autonomy.obstacle_policy import CorridorObservation, TrackedGovernancePolicy
from geometry.camera import CameraIntrinsics
from navigation_control.types import ControlCommand
from obstacle_perception.fusion import SourceCorridorReport, fuse_corridor_reports
from obstacle_perception.lidar_obstacles import LidarObstaclePolicy
from obstacle_perception.lidar_synthetic import Scene, SceneBox, synthetic_sweep
from obstacle_perception.tracking import TrackDepthFilter, TrackPolicy
from obstacle_perception.types import (
    CallTiming,
    DetectionBatch,
    DetectionObservation,
    ObservationMetadata,
    PixelBox,
    TrackIdentity,
)
from planning.corridor import RouteGeometry
from planning.speed_governor import GovernorConfig, SpeedGovernor

ROUTE = RouteGeometry(((0, 0), (60, 0)))


def identity(source, track_id):
    return TrackIdentity("e", source, "v", track_id)


def test_nearer_source_wins_and_disagreement_is_recorded_not_averaged():
    depth = SourceCorridorReport("rgb", True, 25.0, 25.0, False, (identity("rgb", 3),))
    lidar = SourceCorridorReport("lidar", True, 10.0, 12.0, False, (identity("lidar", 3),))
    fused = fuse_corridor_reports((depth, lidar), required_sources=("rgb", "lidar"))
    assert fused.valid and fused.obstacle_gap_m == 10.0 and fused.nearest_source_id == "lidar"
    assert fused.visible_distance_m == 12.0 and fused.disagreement_m == 15.0
    assert len(fused.track_identities) == 2  # same integer ID, different namespaces
    assert fused.to_dict()["source_gaps_m"] == {"rgb": 25.0, "lidar": 10.0}
    swapped = fuse_corridor_reports(
        (
            SourceCorridorReport("rgb", True, 8.0, 8.0),
            SourceCorridorReport("lidar", True, 9.5, 20.0),
        ),
        required_sources=("rgb", "lidar"),
    )
    assert swapped.obstacle_gap_m == 8.0 and swapped.nearest_source_id == "rgb"


def test_loss_of_either_sensor_is_invalid_never_clear():
    depth = SourceCorridorReport("rgb", True, None, 30.0, True)
    dead = SourceCorridorReport("lidar", False, invalid_reason="sparse_sweep_points")
    fused = fuse_corridor_reports((depth, dead), required_sources=("rgb", "lidar"))
    assert not fused.valid and fused.obstacle_gap_m is None and not fused.clearance_valid
    assert fused.invalid_reasons == ("lidar:sparse_sweep_points",)
    missing = fuse_corridor_reports((depth,), required_sources=("rgb", "lidar"))
    assert not missing.valid and missing.invalid_reasons == ("lidar:missing",)
    lidar_only = fuse_corridor_reports(
        (SourceCorridorReport("lidar", True, None, 20.0, True),), required_sources=("rgb", "lidar")
    )
    assert not lidar_only.valid and lidar_only.obstacle_gap_m is None
    # A valid LiDAR gap is still withheld while the depth channel is down: HOLD, not clear.
    withheld = fuse_corridor_reports(
        (
            SourceCorridorReport("rgb", False, invalid_reason="dropped"),
            SourceCorridorReport("lidar", True, 9.0, 9.0),
        ),
        required_sources=("rgb", "lidar"),
    )
    assert (
        not withheld.valid
        and withheld.obstacle_gap_m is None
        and withheld.source_gaps_m == (("rgb", None), ("lidar", 9.0))
    )


def test_clearance_requires_every_source_and_a_valid_empty_source_adds_nothing():
    both = fuse_corridor_reports(
        (
            SourceCorridorReport("rgb", True, None, 30.0, True),
            SourceCorridorReport("lidar", True, None, 18.0, True),
        ),
        required_sources=("rgb", "lidar"),
    )
    assert (
        both.valid
        and both.clearance_valid
        and both.obstacle_gap_m is None
        and both.visible_distance_m == 18.0
    )
    one = fuse_corridor_reports(
        (
            SourceCorridorReport("rgb", True, None, 30.0, True),
            SourceCorridorReport("lidar", True, None, 18.0, False),
        ),
        required_sources=("rgb", "lidar"),
    )
    assert one.valid and not one.clearance_valid and one.obstacle_gap_m is None


def test_double_counting_is_rejected():
    with pytest.raises(ValueError, match="duplicate source"):
        fuse_corridor_reports(
            (SourceCorridorReport("lidar", True, 5.0), SourceCorridorReport("lidar", True, 6.0)),
            required_sources=("lidar",),
        )
    with pytest.raises(ValueError, match="namespaced"):
        SourceCorridorReport("lidar", True, 5.0, track_identities=(identity("rgb", 1),))
    with pytest.raises(ValueError, match="duplicate track"):
        SourceCorridorReport(
            "lidar", True, 5.0, track_identities=(identity("lidar", 1), identity("lidar", 1))
        )
    # A second source carrying another source's identity is a forged namespace;
    # bypass the report validator to prove the fusion step rejects it too.
    report_a = SourceCorridorReport("a", True, 5.0, track_identities=(identity("a", 1),))
    shared = TrackIdentity("e", "a", "v", 1)
    report_b = SourceCorridorReport.__new__(SourceCorridorReport)
    object.__setattr__(report_b, "source_id", "b")
    object.__setattr__(report_b, "valid", True)
    object.__setattr__(report_b, "nearest_gap_m", 6.0)
    object.__setattr__(report_b, "visible_distance_m", None)
    object.__setattr__(report_b, "clearance_valid", False)
    object.__setattr__(report_b, "track_identities", (shared,))
    object.__setattr__(report_b, "invalid_reason", None)
    with pytest.raises(ValueError, match="repeated across"):
        fuse_corridor_reports((report_a, report_b), required_sources=("a", "b"))
    with pytest.raises(ValueError):
        fuse_corridor_reports((), required_sources=())
    with pytest.raises(ValueError):
        SourceCorridorReport("x", False)
    with pytest.raises(ValueError):
        SourceCorridorReport("x", False, 3.0, invalid_reason="r")
    with pytest.raises(ValueError):
        SourceCorridorReport("x", True, float("nan"))


def frame(index, depth=20.0):
    return FrameBundle(
        "e",
        index,
        index * 0.05,
        0,
        EgoPose(VehicleState(0, 0, 0, 0)),
        np.zeros((40, 40, 3), np.uint8),
        CameraIntrinsics(40, 40, 20, 20, 20, 20).calibration(np.eye(4)),
        None if depth is None else np.full((40, 40), depth),
        None if depth is None else index,
    )


def detections(index, ids=(0,)):
    return DetectionBatch(
        ObservationMetadata("e", index, index * 0.05, "rgb", "v1"),
        tuple(DetectionObservation("car", 0.9, PixelBox(0, 0, 40, 40), x) for x in ids),
        40,
        40,
        CallTiming(0),
    )


def rgbd_provider(frame, batch, tracks):
    valid = bool(tracks.tracks and all(t.depth_valid for t in tracks.tracks))
    gap = 25.0 if valid else None
    return CorridorObservation(batch.metadata, valid, gap, gap, False)


def lidar_sweep(index, target_x=20.0):
    return synthetic_sweep(
        Scene(boxes=(SceneBox(target_x, 0.0, 4.0, 2.0, 1.5),)),
        episode_id="e",
        frame_id=index,
        sim_time_s=index * 0.05,
    )


def setup_policy():
    source = LidarCorridorSource(ROUTE, policy=LidarObstaclePolicy(), track_policy=TrackPolicy())
    provider = FusedObservationProvider(rgbd_provider, source)
    policy = TrackedGovernancePolicy(
        TrackDepthFilter(source_id="rgb", model_version="v1"),
        SpeedGovernor(GovernorConfig(4.0, 1.497566177392398, 1.5, 0.95, 6.0, 1.0, 2.0)),
        provider,
        lambda frame, target, dt: ControlCommand(0, 0.5 if target > 0 else 0, 0),
        dt_s=0.05,
    )
    policy.reset(episode_id="e", graph=None)
    source.reset(episode_id="e")
    return source, provider, policy


def test_fused_provider_feeds_the_nearer_lidar_gap_to_the_governor():
    source, provider, policy = setup_policy()
    # Frame 0: both unconfirmed -> HOLD. Frame 1: RGB-D at 25 m, LiDAR car face at 20 m.
    for index in (0, 1):
        source.observe(lidar_sweep(index))
        command = policy.evaluate(frame(index), detections(index), fresh=True, ready=True)
    assert policy.observation_valid and command.throttle > 0 and command.brake == 0
    fused = provider.last_fused
    assert fused.nearest_source_id == "lidar" and fused.obstacle_gap_m == pytest.approx(
        18.0 - 3.8, abs=0.05
    )
    assert fused.disagreement_m == pytest.approx(25.0 - fused.obstacle_gap_m)
    assert {i.source_id for i in fused.track_identities} == {"rgb", "lidar"}
    batch = detections(1)
    rgb = rgbd_report(
        batch, SimpleNamespace(tracks=()), CorridorObservation(batch.metadata, True, 25.0, 25.0)
    )
    assert rgb.source_id == "rgb" and rgb.nearest_gap_m == 25.0 and rgb.valid
    invalid = rgbd_report(
        batch, SimpleNamespace(tracks=()), CorridorObservation(batch.metadata, False, None, None)
    )
    assert not invalid.valid and invalid.invalid_reason == "rgbd_observation_invalid"
    with pytest.raises(ValueError):
        rgbd_report(
            detections(2),
            SimpleNamespace(tracks=()),
            CorridorObservation(batch.metadata, True, 25.0, 25.0),
        )


def test_missing_or_sparse_lidar_holds_instead_of_clearing():
    source, provider, policy = setup_policy()
    for index in (0, 1):
        source.observe(lidar_sweep(index))
        policy.evaluate(frame(index), detections(index), fresh=True, ready=True)
    assert policy.observation_valid
    # No sweep for frame 2: stale LiDAR -> invalid observation -> brake, even though RGB-D is fine.
    command = policy.evaluate(frame(2), detections(2), fresh=True, ready=True)
    assert not policy.observation_valid and command.brake == 1 and command.throttle == 0
    assert provider.last_fused.invalid_reasons == ("lidar:stale_sweep",)
    # Sparse sweep for frame 3: reported invalid, not clear.
    source.observe(
        synthetic_sweep(Scene(), episode_id="e", frame_id=3, sim_time_s=0.15, dropout=0.98)
    )
    command = policy.evaluate(frame(3), detections(3), fresh=True, ready=True)
    assert command.brake == 1 and provider.last_fused.invalid_reasons == (
        "lidar:sparse_sweep_points",
    )
    assert provider.last_fused.obstacle_gap_m is None


def test_lost_depth_holds_even_when_lidar_sees_the_road():
    source, provider, policy = setup_policy()
    for index in (0, 1):
        source.observe(lidar_sweep(index))
        policy.evaluate(frame(index), detections(index), fresh=True, ready=True)
    source.observe(lidar_sweep(2))
    command = policy.evaluate(frame(2, depth=None), detections(2), fresh=True, ready=True)
    assert command.brake == 1 and not policy.observation_valid


def test_lidar_source_rejects_wrong_episode_and_reports_frame_mismatch():
    source = LidarCorridorSource(ROUTE)
    source.reset(episode_id="e")
    assert source.report(frame(0)).invalid_reason == "missing_sweep"
    with pytest.raises(ValueError):
        source.observe(synthetic_sweep(Scene(), episode_id="other", frame_id=0, sim_time_s=0.0))
    with pytest.raises(TypeError):
        source.observe(object())
    source.observe(lidar_sweep(5))
    assert source.report(frame(4)).invalid_reason == "stale_sweep"  # sweep is from a later frame
    assert source.report(frame(6)).invalid_reason == "stale_sweep"  # sweep is older than the frame
    report = source.report(frame(5))
    assert not report.valid and report.invalid_reason == "unconfirmed_tracks"
    source.observe(lidar_sweep(6))
    report = source.report(frame(6))
    assert report.valid and report.source_id == "lidar" and report.nearest_gap_m is not None
    assert all(i.source_id == "lidar" for i in report.track_identities)
    with pytest.raises(ValueError):
        LidarCorridorSource(ROUTE, max_age_s=-1)


def test_lidar_clearance_needs_clear_bins_out_to_clear_start_not_one_bin():
    """An open road sweep asserts clearance; a sweep whose clear bins stop short does not."""
    source = LidarCorridorSource(ROUTE)
    source.reset(episode_id="e")
    for index in range(3):
        source.observe(synthetic_sweep(Scene(), episode_id="e", frame_id=index, sim_time_s=index * 0.05))
    report = source.report(frame(2))
    assert report.valid and report.nearest_gap_m is None
    tracks = source.last_tracks
    assert tracks.seen_clear_m >= source.policy.clear_start_m and report.clearance_valid
    # The same tracks with the clear distance clipped to one bin: no clearance.
    object.__setattr__(tracks, "seen_clear_m", source.policy.clear_bin_m)
    assert not source.report(frame(2)).clearance_valid
    object.__setattr__(tracks, "seen_clear_m", 0.0)
    assert not source.report(frame(2)).clearance_valid
