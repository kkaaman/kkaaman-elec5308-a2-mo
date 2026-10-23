"""G3 framework: radar measurements, the tracker, predictions and lead-car following,
on synthetic measurements. These pin the plumbing and the plan's numbers (gates,
lifecycle, limits); the behaviour against CARLA is the G3 debug pass."""

import math

import numpy as np
import pytest

from mapless.contracts import (
    CLASSES,
    DecisionRecord,
    LocalRoadModel,
    ObjectTrack,
    Prediction,
    TrackStatus,
)
from mapless.ego_motion import Pose2D
from mapless.following import FollowingConfig, FollowingController, worst_case_speed_cap
from mapless.frame import RadarReturns
from mapless.prediction import PredictionConfig, first_conflict, predict
from mapless.radar import RadarConfig, cluster_returns, ego_radial_speed, measure
from mapless.tracking import CHI2_2D, LidarMeasurement, Tracker, TrackerConfig

ORIGIN = Pose2D(0.0, 0.0, 0.0)


def uniform():
    return {k: 1.0 / len(CLASSES) for k in CLASSES}


# -- contracts ---------------------------------------------------------------------------
def test_contracts_validate_their_fields():
    track = ObjectTrack(1, 0.0, [1, 2, 3, 4], np.eye(4), uniform(), TrackStatus.TENTATIVE, 1, 0)
    assert track.speed_mps == pytest.approx(5.0) and track.best_class in CLASSES
    with pytest.raises(ValueError):
        ObjectTrack(1, 0.0, [1, 2, 3, 4], np.eye(4), {"car": 2.0}, TrackStatus.TENTATIVE, 1, 0)
    with pytest.raises(ValueError):
        LocalRoadModel(0.0, (0, 0), 0.25, {"occupancy": np.zeros((2, 2))})
    with pytest.raises(ValueError):
        DecisionRecord(0.0, "FOLLOW", "STOP")  # a veto needs its reason
    pred = Prediction(1, [0.2, 0.4], [[10, 0], [11, 0]], [[2.5, 1.3], [2.6, 1.4]], 0.0)
    assert pred.occupies([[10.5, 0.5]], 0.2)[0] and not pred.occupies([[10.5, 3.0]], 0.2)[0]


# -- radar -------------------------------------------------------------------------------
def test_static_world_shows_the_ego_speed_as_approach():
    # A point straight ahead of a sensor on a car driving at 5 m/s closes at 5 m/s.
    rate = ego_radial_speed([[30.0, 0.0]], (2.5, 0.0), 5.0, 0.0)
    assert rate[0] == pytest.approx(-5.0)
    # Turning: the lever arm adds omega x r to the sensor's velocity.
    rate = ego_radial_speed([[2.5, 20.0]], (2.5, 0.0), 0.0, 0.5)
    assert rate[0] == pytest.approx(-0.5 * 2.5)


def test_radar_clusters_split_two_cars_and_remove_ego_motion():
    sensor_to_ego = np.eye(4)
    sensor_to_ego[0, 3] = 2.5
    # Two static cars side by side 3.5 m apart at 20 m, three returns each, seen at 4 m/s.
    returns = []
    for y in (0.0, 3.5):
        for dx in (-0.4, 0.0, 0.4):
            x = 20.0 + dx
            r = math.hypot(x - 2.5, y)
            az = math.atan2(y, x - 2.5)
            returns.append([r, az, 0.0, -4.0 * (x - 2.5) / r])
    radar = RadarReturns("radar_front", np.array(returns, dtype=np.float32), sensor_to_ego)
    objects = measure(radar, 4.0, 0.0, RadarConfig(cluster_m=1.5))
    assert len(objects) == 2
    assert all(abs(o.radial_speed_mps) < 0.2 for o in objects)  # static once the ego's speed is removed
    assert sorted(round(o.xy[1], 1) for o in objects) == [0.0, 3.5]
    assert measure(RadarReturns("radar_front", np.zeros((0, 4), dtype=np.float32), sensor_to_ego), 4.0, 0.0) == []


def test_wide_radar_clusters_are_split():
    xy = np.column_stack((np.full(20, 15.0), np.linspace(-6, 6, 20)))
    data = np.column_stack((xy, np.zeros(20)))
    parts = cluster_returns(data, np.array([2.5, 0.0]), "r", RadarConfig(cluster_m=1.0, max_cluster_span_m=6.0))
    assert len(parts) >= 2


# -- tracker -----------------------------------------------------------------------------
def test_tracker_confirms_estimates_velocity_coasts_and_deletes():
    tracker = Tracker(TrackerConfig())
    tracks = []
    for k in range(40):
        t = k * 0.05
        xy = np.array([20.0 + 3.0 * t, 1.0])  # a car ahead moving away at 3 m/s
        tracks = tracker.update(t, ORIGIN, lidar=[LidarMeasurement(xy, (4.5, 2.0, 0.0), {"car": 0.9, "unknown": 0.1})])
    assert len(tracks) == 1
    track = tracks[0]
    assert track.status is TrackStatus.CONFIRMED and track.best_class == "car"
    assert abs(track.velocity[0] - 3.0) < 0.6 and abs(track.velocity[1]) < 0.3
    assert track.footprint is not None and "lidar" in track.support
    # Three hits in five frames confirmed it; before that it was tentative.
    early = Tracker(TrackerConfig())
    early.update(0.0, ORIGIN, lidar=[LidarMeasurement(np.array([20.0, 1.0]))])
    assert early.snapshot()[0].status is TrackStatus.TENTATIVE
    # No more hits: coasting, then deleted after delete_s.
    coasting = tracker.update(2.0 + 0.2, ORIGIN)
    assert coasting[0].status is TrackStatus.COASTING
    assert tracker.update(2.0 + 0.8, ORIGIN) == []


def test_tracker_gates_by_class_and_by_distance():
    tracker = Tracker(TrackerConfig())
    for k in range(5):
        tracker.update(k * 0.05, ORIGIN, lidar=[LidarMeasurement(np.array([20.0, 0.0]), None, {"pedestrian": 0.95})])
    # A confident car measurement at the same spot does not join the pedestrian track.
    tracks = tracker.update(0.25, ORIGIN, lidar=[LidarMeasurement(np.array([20.1, 0.0]), None, {"car": 0.95})])
    assert len(tracks) == 2
    # A measurement 10 m away starts its own track too.
    tracks = tracker.update(0.30, ORIGIN, lidar=[LidarMeasurement(np.array([30.0, 0.0]), None, {"pedestrian": 0.95})])
    assert len(tracks) == 3


def test_radar_measurements_update_and_start_tracks():
    sensor_to_ego = np.eye(4)
    sensor_to_ego[0, 3] = 2.5
    tracker = Tracker(TrackerConfig())
    for k in range(30):
        t = k * 0.05
        x = 25.0 - 2.0 * t  # approaching at 2 m/s, the ego standing
        r = x - 2.5
        radar = RadarReturns("radar_front", np.array([[r, 0.0, 0.0, -2.0], [r + 0.3, 0.01, 0.0, -2.0]], dtype=np.float32), sensor_to_ego)
        tracks = tracker.update(t, ORIGIN, radar=measure(radar, 0.0, 0.0))
    assert len(tracks) == 1 and "radar" in tracks[0].support
    assert tracks[0].velocity[0] == pytest.approx(-2.0, abs=0.5)
    assert tracks[0].footprint is None  # a radar return has no extent


def test_hazards_include_tentative_tracks_close_ahead():
    tracker = Tracker(TrackerConfig(hazard_m=12.0))
    tracker.update(0.0, ORIGIN, lidar=[LidarMeasurement(np.array([8.0, 0.0])), LidarMeasurement(np.array([40.0, 0.0]))])
    near = tracker.hazards(ORIGIN)
    assert [round(t.position[0]) for t in near] == [8]
    assert CHI2_2D == 9.21


# -- prediction --------------------------------------------------------------------------
def test_predictions_grow_with_time_and_pedestrians_widen():
    car = ObjectTrack(1, 0.0, [20, 0, 5, 0], np.diag([0.1, 0.1, 0.5, 0.5]), {"car": 1.0}, TrackStatus.CONFIRMED, 5, 0, footprint=(4.5, 2.0, 0.0))
    pred = predict(car, PredictionConfig())
    assert len(pred.times_s) == 20 and pred.times_s[-1] == pytest.approx(4.0)
    assert pred.centres[-1, 0] == pytest.approx(40.0)
    assert (np.diff(pred.half_extents[:, 0]) >= 0).all()
    walker = ObjectTrack(2, 0.0, [15, 3, 0, -1], np.diag([0.1, 0.1, 0.2, 0.2]), {"pedestrian": 1.0}, TrackStatus.CONFIRMED, 5, 0)
    wide = predict(walker)
    assert wide.half_extents[-1, 1] > wide.half_extents[0, 1] + 3.0
    assert predict(car, lane_change=True).times_s[-1] == pytest.approx(6.0)


def test_first_conflict_finds_the_crossing_pedestrian():
    walker = ObjectTrack(2, 0.0, [15, 4, 0, -1.2], np.diag([0.05, 0.05, 0.05, 0.05]), {"pedestrian": 1.0}, TrackStatus.CONFIRMED, 5, 0)
    path = np.column_stack((np.arange(0, 40.0, 0.5), np.zeros(80)))
    times = np.arange(0, 40.0, 0.5) / 2.0  # driving at 2 m/s
    hit = first_conflict([predict(walker)], path, times, body_radius_m=1.05)
    assert hit is not None and hit[1] == 2 and hit[0] <= 4.0
    far = ObjectTrack(3, 0.0, [15, 30, 0, 0], np.diag([0.05] * 4), {"car": 1.0}, TrackStatus.CONFIRMED, 5, 0)
    assert first_conflict([predict(far)], path, times, 1.05) is None


# -- following ---------------------------------------------------------------------------
def test_worst_case_cap_and_rate_limits():
    c = FollowingConfig()
    assert worst_case_speed_cap(2.0, 0.0, c) == 0.0  # inside the standstill margin: stop
    assert worst_case_speed_cap(30.0, 0.0, c) > worst_case_speed_cap(10.0, 0.0, c) > 0.0
    assert worst_case_speed_cap(10.0, 5.0, c) > worst_case_speed_cap(10.0, 0.0, c)  # a moving lead leaves more room
    follow = FollowingController(c)
    speeds = [follow.target_speed(4.0, 0.0, 40.0, 4.0, 0.05) for _ in range(40)]
    accel = np.diff([0.0] + speeds) / 0.05
    assert accel.max() <= c.accel_mps2 + 1e-6
    assert np.abs(np.diff(accel)).max() / 0.05 <= c.jerk_mps3 + 1e-6
    # A standing lead 6 m ahead: the target settles near zero.
    follow.reset()
    for _ in range(60):
        v = follow.target_speed(4.0, 2.0, 6.0, 0.0, 0.05)
    assert v < 0.5


def test_lead_ahead_picks_the_nearest_track_in_the_lane():
    follow = FollowingController()
    path = np.column_stack((np.arange(0, 50.0), np.zeros(50)))
    in_lane = ObjectTrack(1, 0.0, [25, 0.5, 0, 0], np.eye(4), {"car": 1.0}, TrackStatus.CONFIRMED, 5, 0)
    nearer_beside = ObjectTrack(2, 0.0, [10, 4.0, 0, 0], np.eye(4), {"car": 1.0}, TrackStatus.CONFIRMED, 5, 0)
    behind = ObjectTrack(3, 0.0, [-5, 0.0, 0, 0], np.eye(4), {"car": 1.0}, TrackStatus.CONFIRMED, 5, 0)
    lead = follow.lead_ahead([in_lane, nearer_beside, behind], ORIGIN, path)
    assert lead is not None and lead[1].track_id == 1 and lead[0] == pytest.approx(25.0)
