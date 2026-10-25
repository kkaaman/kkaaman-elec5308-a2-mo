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
from mapless.prediction import PredictionConfig, first_conflict, first_footprint_conflict, predict
from mapless.radar import RadarConfig, cluster_returns, ego_radial_speed, measure
from mapless.terrain import VehicleBody
from mapless.tracking import CHI2_2D, LidarMeasurement, Tracker, TrackerConfig

ORIGIN = Pose2D(0.0, 0.0, 0.0)
BODY = VehicleBody()  # front 3.78 m ahead of the rear axle, 1.01 m behind, 1.05 m half width


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


# -- regressions from the review of 36ba0ca ------------------------------------------------
def test_radar_bearing_innovation_wraps_across_pi():
    # A track behind the car in the odometry frame (bearing near +-pi): the update must not
    # read a 2*pi bearing error (the first version jumped the track 60 m in one update).
    sensor_to_ego = np.eye(4)
    sensor_to_ego[0, 3] = -2.45
    tracker = Tracker(TrackerConfig())
    pose = Pose2D(0.0, 0.0, math.pi)  # the car faces -x in the odometry frame
    for k in range(20):
        t = k * 0.05
        # Object 10 m straight behind the sensor in the vehicle frame (bearing pi in the car),
        # i.e. at odometry (+12.45, 0): bearing 0 there, pi in the body: wrapping matters.
        radar = RadarReturns("radar_rear", np.array([[10.0, math.pi - 0.001, 0.0, 0.0], [10.2, math.pi - 0.002, 0.0, 0.0]], dtype=np.float32), sensor_to_ego)
        tracks = tracker.update(t, pose, radar=measure(radar, 0.0, 0.0))
    assert len(tracks) == 1
    assert np.hypot(*(tracks[0].position - np.array([12.45, 0.0]))) < 1.0
    assert tracks[0].speed_mps < 1.0


def test_coasting_tracks_leave_the_live_set_after_coast_s():
    tracker = Tracker(TrackerConfig(coast_s=0.3, delete_s=0.75))
    for k in range(5):
        tracker.update(k * 0.05, ORIGIN, lidar=[LidarMeasurement(np.array([10.0, 0.0]))])
    tracker.update(0.4, ORIGIN)  # 0.2 s unseen: still live
    assert len(tracker.live()) == 1 and len(tracker.hazards(ORIGIN)) == 1
    tracker.update(0.7, ORIGIN)  # 0.5 s unseen: kept for re-association, not for planning
    assert tracker.snapshot() and tracker.live() == [] and tracker.hazards(ORIGIN) == []


# -- regressions from the external review of the G3 framework ------------------------------
def test_a_tentative_track_that_misses_a_frame_is_no_lead():
    # One detection, then one missed frame: the track used to read COASTING (a status the
    # following stage accepts), so a 3-point cluster seen once led the car.
    tracker = Tracker(TrackerConfig())
    tracker.update(0.0, ORIGIN, lidar=[LidarMeasurement(np.array([20.0, 0.0]), None, {"car": 0.9, "unknown": 0.1})])
    tracks = tracker.update(0.05, ORIGIN)
    assert len(tracks) == 1 and tracks[0].status is TrackStatus.TENTATIVE
    path = np.column_stack((np.arange(0, 50.0), np.zeros(50)))
    assert FollowingController().lead_ahead(tracks, ORIGIN, path) is None
    # Still a hazard when close (acts before confirmation), still deleted by the timer.
    assert tracker.hazards(ORIGIN) == [] and tracker.update(0.1, ORIGIN, lidar=[LidarMeasurement(np.array([8.0, 0.0]))])
    assert [round(t.position[0]) for t in tracker.hazards(ORIGIN)] == [8]
    assert all(t.status is TrackStatus.TENTATIVE for t in tracker.snapshot())
    assert tracker.update(1.0, ORIGIN) == []
    # Confirmed first, then a miss: coasting, and a lead.
    confirmed = Tracker(TrackerConfig())
    for k in range(3):
        confirmed.update(k * 0.05, ORIGIN, lidar=[LidarMeasurement(np.array([20.0, 0.0]))])
    tracks = confirmed.update(0.15, ORIGIN)
    assert tracks[0].status is TrackStatus.COASTING
    assert FollowingController().lead_ahead(tracks, ORIGIN, path) is not None


def test_prediction_uncertainty_follows_the_object_heading():
    cfg = PredictionConfig(process_noise_m2ps3=0.0)
    # Northbound car with a 3 m along-track (odometry y) sigma: ~6 m longitudinal inflation,
    # none lateral. The first version added the y variance to the width.
    north = ObjectTrack(1, 0.0, [0, 0, 0, 5], np.diag([0.01, 9.0, 0.0, 0.0]), {"car": 1.0}, TrackStatus.CONFIRMED, 5, 0, footprint=(4.5, 2.0, math.pi / 2))
    pred = predict(north, cfg)
    assert pred.heading_rad == pytest.approx(math.pi / 2)
    assert pred.half_extents[0, 0] == pytest.approx(4.5 / 2 + 0.3 + 6.0, abs=0.05)
    assert pred.half_extents[0, 1] == pytest.approx(2.0 / 2 + 0.3 + 0.2, abs=0.05)
    # The same scene rotated by 90 degrees gives the same footprints (frame independence).
    P = np.array([[0.4, 0.1, 0.05, 0.0], [0.1, 0.2, 0.0, 0.02], [0.05, 0.0, 0.3, 0.05], [0.0, 0.02, 0.05, 0.1]])
    east = ObjectTrack(2, 0.0, [10, 2, 4, 1], P, {"car": 1.0}, TrackStatus.CONFIRMED, 5, 0, footprint=(4.5, 2.0, 0.2))
    R = np.array([[0, -1], [1, 0]])  # +90 degrees
    T = np.block([[R, np.zeros((2, 2))], [np.zeros((2, 2)), R]])
    rotated = ObjectTrack(2, 0.0, T @ east.state, T @ P @ T.T, {"car": 1.0}, TrackStatus.CONFIRMED, 5, 0, footprint=(4.5, 2.0, 0.2 + math.pi / 2))
    a, b = predict(east, PredictionConfig()), predict(rotated, PredictionConfig())
    assert np.allclose(a.half_extents, b.half_extents, atol=1e-6)
    assert np.allclose((R @ a.centres.T).T, b.centres, atol=1e-9)
    # Position-velocity cross terms propagate: sigma_long(t)^2 = P_pp + 2 t P_pv + t^2 P_vv.
    cross = ObjectTrack(3, 0.0, [0, 0, 5, 0], np.array([[1.0, 0, 0.5, 0], [0, 0.1, 0, 0], [0.5, 0, 0.25, 0], [0, 0, 0, 0.0]]), {"car": 1.0}, TrackStatus.CONFIRMED, 5, 0, footprint=(4.5, 2.0, 0.0))
    p3 = predict(cross, cfg)
    t = p3.times_s
    assert np.allclose(p3.half_extents[:, 0], 2.25 + 0.3 + 2 * np.sqrt(1.0 + 2 * 0.5 * t + 0.25 * t**2))
    # Process noise grows the region over the horizon even with a perfect estimate.
    sure = ObjectTrack(4, 0.0, [0, 0, 5, 0], np.zeros((4, 4)), {"car": 1.0}, TrackStatus.CONFIRMED, 5, 0, footprint=(4.5, 2.0, 0.0))
    grown = predict(sure, PredictionConfig()).half_extents
    assert grown[-1, 0] > grown[0, 0] + 1.0 and (np.diff(grown[:, 0]) >= 0).all()
    # Velocity uncertainty counts only for a settled, moving track (first live stack run:
    # every roadside cluster grew into a region metres wide and held the car at the start
    # line). A settled stationary track and an unsettled one are predicted on their
    # measured velocity alone; a settled moving one on its (bounded) velocity sigma.
    settled_still = ObjectTrack(5, 0.0, [12, 3, 0.05, 0.0], np.diag([0.05, 0.05, 0.25, 0.25]), {"car": 1.0}, TrackStatus.CONFIRMED, 20, 0, footprint=(4.5, 2.0, 0.0))
    unsettled = ObjectTrack(6, 0.0, [12, 3, 0.05, 0.0], np.diag([0.05, 0.05, 16.0, 16.0]), {"car": 1.0}, TrackStatus.CONFIRMED, 3, 0, footprint=(4.5, 2.0, 0.0))
    moving = ObjectTrack(7, 0.0, [12, 3, 3.0, 0.0], np.diag([0.05, 0.05, 0.25, 0.25]), {"car": 1.0}, TrackStatus.CONFIRMED, 20, 0, footprint=(4.5, 2.0, 0.0))
    still_ext, unsettled_ext, moving_ext = (predict(tr, cfg).half_extents for tr in (settled_still, unsettled, moving))
    assert np.allclose(still_ext, unsettled_ext)
    assert still_ext[-1, 1] == pytest.approx(2.0 / 2 + 0.3 + 2 * math.sqrt(0.05), abs=0.02)  # position sigma only, at 4 s
    assert moving_ext[-1, 0] > still_ext[-1, 0] + 3.0  # 2 * 0.5 m/s * 4 s along the heading
    # Process noise is not charged to a stationary track either (second live stack run:
    # the kerb cluster 4.8 m to the right grew to 3.2 m half width at 3.6 s under the
    # default noise and capped the speed at the start line). Its region stays its
    # measured extent plus the position sigma over the whole horizon.
    kerb = ObjectTrack(8, 0.0, [6.3, -4.8, 0.0, 0.0], np.diag([0.05, 0.05, 0.25, 0.25]), {"unknown": 1.0}, TrackStatus.CONFIRMED, 20, 0, footprint=(1.2, 0.6, 0.0))
    kerb_ext = predict(kerb, PredictionConfig()).half_extents
    assert np.allclose(kerb_ext[-1], kerb_ext[0]) and kerb_ext[-1, 1] == pytest.approx(0.3 + 0.3 + 2 * math.sqrt(0.05), abs=0.01)


def small_obstacle(track_id, xy):
    # A stationary 0.3 m object with a sharp estimate: its region is ~0.47 m half extent.
    return ObjectTrack(track_id, 0.0, [xy[0], xy[1], 0, 0], np.eye(4) * 1e-4, {"pedestrian": 1.0}, TrackStatus.CONFIRMED, 5, 0, footprint=(0.3, 0.3, 0.0))


def test_footprint_conflicts_cover_the_overhangs_not_a_disc():
    cfg = PredictionConfig(process_noise_m2ps3=0.0, pedestrian_lateral_mps=0.0)
    # Straight path along +x at 5 m/s, time zero at the rear axle.
    path = np.column_stack((np.arange(-5.0, 40.0, 0.5), np.zeros(90)))
    times = path[:, 0] / 5.0
    # Under the bonnet now: 3 m ahead of the axle, 0.9 m off centre. The body covers it at
    # t = 0, so the first prediction step already conflicts (the disc about the axle only
    # caught it once the axle had driven up to it, a step or two later).
    front = predict(small_obstacle(1, (3.0, 0.9)), cfg)
    hit = first_footprint_conflict([front], path, times, BODY)
    assert hit is not None and hit[1] == 1 and hit[0] == pytest.approx(0.2)
    disc = first_conflict([front], path, times, BODY.half_width_m)
    assert disc is not None and disc[0] >= 0.4
    # Just beside the car (2 m to the left): no conflict ever.
    beside = predict(small_obstacle(2, (1.0, 2.0)), cfg)
    assert first_footprint_conflict([beside], path, times, BODY) is None
    # A left turn of radius 20 m: an object the rear-right corner touches 2 s in, 1.9 m from
    # the axle there (outside the old 1.05 m disc + its 0.47 m region). The oriented body
    # finds it no later than that (the front-right corner, swinging wide, reaches the spot
    # about a second earlier, which is right); the pose-sampled headings carry the turn.
    R = 20.0
    s = np.arange(0.0, 30.0, 0.25)
    arc = np.column_stack((R * np.sin(s / R), R * (1 - np.cos(s / R))))
    arc_times = s / 5.0
    theta = 10.0 / R
    axle = np.array([R * math.sin(theta), R * (1 - math.cos(theta))])
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    corner = axle + rot @ np.array([BODY.rear_m - 0.3, -BODY.half_width_m - 0.3])
    rear = predict(small_obstacle(3, corner), cfg)
    hit = first_footprint_conflict([rear], arc, arc_times, BODY)
    assert hit is not None and hit[1] == 3 and 0.5 <= hit[0] <= 2.1
    # The same object mirrored to the inside of the turn, 2.2 m from the axle path: beside
    # the car throughout, no conflict (the check is not merely a wider disc).
    inside = axle + rot @ np.array([0.0, BODY.half_width_m + 1.15])
    assert first_footprint_conflict([predict(small_obstacle(4, inside), cfg)], arc, arc_times, BODY) is None
