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
    sensor_to_ego[0, 3], sensor_to_ego[2, 3] = 2.5, 0.7
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


def test_radar_drops_road_surface_and_overhead_returns():
    """A bumper radar's 15 degree vertical field hits the tarmac from 5 m out (first full
    live run: those returns became a lead 5 m ahead); overhead structure is no road user."""
    sensor_to_ego = np.eye(4)
    sensor_to_ego[:3, 3] = (2.5, 0.0, 0.7)
    returns = []
    for x in (5.0, 8.0, 12.0):  # road surface ahead: z = 0 from a sensor at 0.7 m
        r = math.hypot(x - 2.5, 0.7)
        returns.append([r, 0.0, -math.asin(0.7 / r), 0.0])
    returns.append([20.0, 0.0, 0.0, 0.0])  # a car face at the sensor's height
    returns.append([15.0, 0.0, math.asin(4.0 / 15.0), 0.0])  # a gantry 4.7 m up
    radar = RadarReturns("radar_front", np.array(returns, dtype=np.float32), sensor_to_ego)
    objects = measure(radar, 0.0, 0.0)
    assert len(objects) == 1 and objects[0].xy[0] == pytest.approx(22.5)
    # Flat-mounted test fixtures keep working: a return level with a sensor at z = 0.7 is kept.
    assert len(measure(radar, 0.0, 0.0, RadarConfig(min_height_m=-1.0))) == 4  # gantry still out


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
    assert track.travelled_m == pytest.approx(3.0 * 39 * 0.05, abs=0.3)  # net displacement since first seen
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
    sensor_to_ego[0, 3], sensor_to_ego[2, 3] = 2.5, 0.7
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


def test_radar_only_tracks_start_beyond_lidar_range_or_moving_and_rejections_are_counted():
    """Town05, 26 Oct: the front radar's unclaimed static returns from pavement, kerbs and
    buildings close to the car made 155 tracks against 61 without it (45 ms a tick). A
    radar-only track now starts beyond the LiDAR's object range or for a moving return.
    A radial speed the track cannot explain is counted, consecutively, and logged."""
    from mapless.radar import RadarMeasurement

    sensor_xy = np.array([2.5, 0.0])

    def ret(xy, radial):
        rel = np.asarray(xy, dtype=float) - sensor_xy
        return RadarMeasurement("radar_front", np.asarray(xy, dtype=float), float(np.hypot(*rel)), float(math.atan2(rel[1], rel[0])), radial, 3, sensor_xy)

    tracker = Tracker(TrackerConfig())
    tracks = tracker.update(0.0, ORIGIN, radar=[ret((15.0, -6.0), 0.0), ret((45.0, 0.0), 0.0), ret((15.0, 3.0), -2.0), ret((45.0, 15.0), 0.0)])
    assert len(tracks) == 2  # the far static one ahead and the near moving one
    assert tracker.radar_stats["clutter_skipped"] == 2 and tracker.radar_stats["spawned"] == 2
    # A settled static LiDAR track; the radar reads it moving, alternately 3 m/s away and
    # towards the sensor: isolated outliers, each rejected and the count restarted.
    car = LidarMeasurement(np.array([20.0, 0.0]), (4.5, 2.0, 0.0))
    tracker = Tracker(TrackerConfig())
    for k in range(20):
        tracks = tracker.update(k * 0.05, ORIGIN, lidar=[car])
    for k in range(20, 24):
        tracks = tracker.update(k * 0.05, ORIGIN, lidar=[car], radar=[ret((20.0, 0.0), 3.0 if k % 2 else -3.0)])
        assert tracks[0].doppler_rejects == 1 and not tracker.radar_stats["doppler_manoeuvres"]
    assert abs(tracks[0].velocity[0]) < 0.3
    # Then 3 m/s away again (the last outlier was too): two running is still rejected,
    # three running is a manoeuvre and the radial speed is taken.
    tracks = tracker.update(24 * 0.05, ORIGIN, lidar=[car], radar=[ret((20.0, 0.0), 3.0)])
    assert tracks[0].doppler_rejects == 2 and not tracker.radar_stats["doppler_manoeuvres"]
    tracks = tracker.update(25 * 0.05, ORIGIN, lidar=[car], radar=[ret((20.0, 0.0), 3.0)])
    assert tracker.radar_stats["doppler_manoeuvres"] and tracks[0].doppler_rejects == 0
    assert tracks[0].velocity[0] > 2.0


def test_a_braking_lead_is_seen_by_its_radial_speed_within_three_radar_updates():
    """B9 (26 Oct): the lead stopped within 0.2 s; the radar read it 1.7 m/s slower than
    the track nine ticks running and every reading was rejected, so the track slowed only
    as fast as the LiDAR positions allowed (0.6 s). A one-sided run of rejections is a
    manoeuvre. The tangential speed is untouched: a crossing walker keeps its speed."""
    from mapless.radar import RadarMeasurement

    sensor_xy = np.array([2.5, 0.0])

    def ret(xy, radial):
        rel = np.asarray(xy, dtype=float) - sensor_xy
        return RadarMeasurement("radar_front", np.asarray(xy, dtype=float), float(np.hypot(*rel)), float(math.atan2(rel[1], rel[0])), radial, 3, sensor_xy)

    tracker = Tracker(TrackerConfig())
    x = 15.0
    for k in range(60):  # 2 m/s ahead, LiDAR and radar agreeing
        x += 2.0 * 0.05
        tracks = tracker.update(k * 0.05, ORIGIN, lidar=[LidarMeasurement(np.array([x, 0.0]), (0.4, 1.8, 0.0))], radar=[ret((x, 0.0), 2.0)])
    assert tracks[0].velocity[0] == pytest.approx(2.0, abs=0.2)
    speeds = []
    for k in range(60, 66):  # stopped dead
        tracks = tracker.update(k * 0.05, ORIGIN, lidar=[LidarMeasurement(np.array([x, 0.0]), (0.4, 1.8, 0.0))], radar=[ret((x, 0.0), 0.0)])
        speeds.append(tracks[0].velocity[0])
    assert speeds[2] < 0.6  # by the third radar update (0.15 s)
    # A walker crossing square on: the radar reads ~0 radial, the LiDAR sees it cross.
    tracker = Tracker(TrackerConfig())
    for k in range(40):
        y = -3.0 + 1.2 * k * 0.05
        tracks = tracker.update(k * 0.05, ORIGIN, lidar=[LidarMeasurement(np.array([20.0, y]), (0.5, 0.5, 0.0))], radar=[ret((20.0, y), 0.0)])
    assert tracks[0].velocity[1] == pytest.approx(1.2, abs=0.3)


def test_radar_twin_of_a_lidar_track_is_merged_and_near_static_radar_clutter_dropped():
    """Braking lead B8 (26 Oct): the radar picked the lead up at 32 m (track 166); when
    the LiDAR clustered it, the cluster started its own track and both lived on, the
    radar one followed to 12 m, the LiDAR one (1.7 x 1.0 m, sideways jitter) read as a
    crossing walker. Static radar-only tracks carried in from beyond 30 m made 148 tracks."""
    from mapless.radar import RadarMeasurement

    sensor_xy = np.array([2.5, 0.0])

    def ret(xy, radial):
        rel = np.asarray(xy, dtype=float) - sensor_xy
        return RadarMeasurement("radar_front", np.asarray(xy, dtype=float), float(np.hypot(*rel)), float(math.atan2(rel[1], rel[0])), radial, 3, sensor_xy)

    tracker = Tracker(TrackerConfig())
    t = 0.0
    for k in range(40):  # a lead 36 m out at 2 m/s, the car standing: radar only
        t = k * 0.05
        tracks = tracker.update(t, ORIGIN, radar=[ret((36.0 + 2.0 * t, 0.0), 2.0)])
    assert len(tracks) == 1 and tracks[0].support == ("radar",)
    radar_id = tracks[0].track_id
    merged = 0
    for k in range(40, 60):  # the LiDAR clusters the car 2.5 m nearer than the radar's point: outside the gate
        t = k * 0.05
        x = 36.0 + 2.0 * t
        tracks = tracker.update(t, ORIGIN, lidar=[LidarMeasurement(np.array([x - 2.5, 0.0]), (4.5, 1.8, 0.0))], radar=[ret((x, 0.0), 2.0)])
        merged += tracker.radar_stats.get("merged", 0)
    assert merged >= 1
    assert [tr.track_id for tr in tracks] == [radar_id]  # one track, the older id
    assert set(tracks[0].support) == {"lidar", "radar"} and tracks[0].footprint is not None
    assert tracks[0].velocity[0] == pytest.approx(2.0, abs=0.3)
    # A static return carried in from 40 m by the car driving at 4 m/s: dropped inside 25 m.
    tracker = Tracker(TrackerConfig())
    for k in range(120):
        t = k * 0.05
        pose = Pose2D(4.0 * t, 0.0, 0.0)
        tracks = tracker.update(t, pose, radar=[ret((40.0 - 4.0 * t, 5.0), 0.0)])
        if 40.0 - 4.0 * t > 30.0:
            assert len(tracks) == 1
    assert tracks == [] and tracker.radar_stats["clutter_skipped"] == 1  # dropped, and no new one near


def test_only_lidar_supported_vouched_tracks_are_predicted():
    from mapless.stack import predictable_tracks

    def track(tid, status, support):
        return ObjectTrack(tid, 0.0, [20, -4, 0, 0], np.eye(4) * 0.1, {"unknown": 1.0}, status, 5, 0, support=support)

    tracks = [
        track(1, TrackStatus.CONFIRMED, ("lidar",)),
        track(2, TrackStatus.COASTING, ("lidar", "radar")),
        track(3, TrackStatus.CONFIRMED, ("radar",)),  # radar-only: 1 m bearing jitter at 30 m, no extent
        track(4, TrackStatus.TENTATIVE, ("lidar",)),
    ]
    assert [t.track_id for t in predictable_tracks(tracks)] == [1, 2]


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
    sensor_to_ego[0, 3], sensor_to_ego[2, 3] = -2.45, 0.7
    tracker = Tracker(TrackerConfig(radar_spawn_min_range_m=0.0))  # a static radar-only track, on purpose
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
    # Motion states (external review of 405b056). STATIC (recent mean speed under 0.3 m/s):
    # predicted where it stands with no velocity or process noise (first and second live
    # stack runs: roadside clusters grew metres wide and held the car at the start line).
    still = ObjectTrack(5, 0.0, [12, 3, 0.05, 0.0], np.diag([0.05, 0.05, 0.25, 0.25]), {"car": 1.0}, TrackStatus.CONFIRMED, 20, 0, footprint=(4.5, 2.0, 0.0), recent_speed_mps=0.05)
    still_ext = predict(still, cfg).half_extents
    assert np.allclose(still_ext[-1], still_ext[0])
    assert still_ext[-1, 1] == pytest.approx(2.0 / 2 + 0.3 + 2 * math.sqrt(0.05), abs=0.02)
    # MOVING: along its velocity, with velocity sigma capped at 1.5 m/s.
    moving = ObjectTrack(7, 0.0, [12, 3, 3.0, 0.0], np.diag([0.05, 0.05, 0.25, 0.25]), {"car": 1.0}, TrackStatus.CONFIRMED, 20, 0, footprint=(4.5, 2.0, 0.0))
    moving_ext = predict(moving, cfg).half_extents
    assert moving_ext[-1, 0] > still_ext[-1, 0] + 3.0
    # UNCERTAIN (no history yet, or slow): where it stands, but growing at up to walking
    # pace; a less certain filter never gives a smaller envelope (no cliff at a threshold).
    sizes = []
    for sigma in (0.5, 0.99, 1.01, 2.0, 4.0):
        young = ObjectTrack(6, 0.0, [12, 3, 0.05, 0.0], np.diag([0.05, 0.05, sigma**2, sigma**2]), {"car": 1.0}, TrackStatus.CONFIRMED, 3, 0, footprint=(4.5, 2.0, 0.0))
        pred = predict(young, PredictionConfig())
        assert np.allclose(pred.centres, pred.centres[0])
        sizes.append(pred.half_extents[14, 1])  # at 3 s
    assert all(b >= a - 1e-9 for a, b in zip(sizes, sizes[1:])), sizes
    assert sizes[2] - sizes[1] < 0.1  # 0.99 -> 1.01 m/s: continuous (the old rule dropped 2.75 m here)
    assert sizes[-1] < 2.0 / 2 + 0.3 + 2 * math.sqrt(0.05 + (1.0 * 3) ** 2 + PredictionConfig().process_noise_m2ps3 * 27 / 3) + 0.01  # capped at walking pace
    # A static kerb cluster: no process noise either (second live stack run, 25 Oct).
    kerb = ObjectTrack(8, 0.0, [6.3, -4.8, 0.0, 0.0], np.diag([0.05, 0.05, 0.25, 0.25]), {"unknown": 1.0}, TrackStatus.CONFIRMED, 20, 0, footprint=(1.2, 0.6, 0.0), recent_speed_mps=0.0)
    kerb_ext = predict(kerb, PredictionConfig()).half_extents
    assert np.allclose(kerb_ext[-1], kerb_ext[0]) and kerb_ext[-1, 1] == pytest.approx(0.3 + 0.3 + 2 * math.sqrt(0.05), abs=0.01)
    # Centroid jitter on a track that has hardly moved is not motion: not carried along.
    jitter = ObjectTrack(9, 0.0, [14.8, -4.5, 0.44, 0.05], np.diag([0.05, 0.05, 0.25, 0.25]), {"unknown": 1.0}, TrackStatus.CONFIRMED, 20, 0, footprint=(6.0, 0.6, 0.0), travelled_m=0.2)
    held = predict(jitter, PredictionConfig())
    assert np.allclose(held.centres, held.centres[0])
    gone = predict(ObjectTrack(9, 0.0, [14.8, -4.5, 0.44, 0.05], np.diag([0.05, 0.05, 0.25, 0.25]), {"unknown": 1.0}, TrackStatus.CONFIRMED, 20, 0, footprint=(6.0, 0.6, 0.0), travelled_m=1.0), PredictionConfig())
    assert gone.centres[-1, 0] > held.centres[-1, 0] + 1.5


def test_too_little_motion_evidence_is_uncertain_not_static():
    """External review of 405b056: a 0.4 m/s walker, one seen after only 0.5 m of travel,
    and one that has turned back near its start were all predicted stationary and left
    out of the conflict check. They are now uncertain: tested, with a bounded envelope."""
    from mapless.prediction import motion_state
    from mapless.stack import StackConfig, conflict_candidates

    c = StackConfig()
    small = (0.5, 0.5, 0.0)
    u = {k: 0.2 for k in ("car", "truck", "bike", "pedestrian", "unknown")}
    slow = ObjectTrack(1, 0.0, [12, -3, 0.0, 0.4], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 20, 0, footprint=small, travelled_m=1.0, recent_speed_mps=0.4)
    new = ObjectTrack(2, 0.0, [12, -3, 0.0, 1.2], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 4, 0, footprint=small, travelled_m=0.5)
    back = ObjectTrack(3, 0.0, [12, -3, 0.0, -1.2], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 40, 0, footprint=small, travelled_m=0.2, recent_speed_mps=0.35)
    pole = ObjectTrack(4, 0.0, [12, -3, 0.0, 0.0], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 40, 0, footprint=small, travelled_m=0.1, recent_speed_mps=0.05)
    assert [motion_state(t, c.prediction) for t in (slow, new, back, pole)] == ["uncertain", "uncertain", "uncertain", "static"]
    tracks = [slow, new, back, pole]
    preds = [predict(t, c.prediction) for t in tracks]
    assert [p.track_id for p in conflict_candidates(tracks, preds, None, c.prediction)] == [1, 2, 3]


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


def test_conflict_check_tests_moving_road_users_and_pedestrians_only():
    """First live stack cut-in run (26 Oct): a car parked at the kerb beside the lane, which
    the driver's corridor check cleared, held the car through its predicted box."""
    from mapless.prediction import predict
    from mapless.stack import conflict_candidates

    cfg = PredictionConfig()
    parked = ObjectTrack(1, 0.0, [12, -2.6, 0.0, 0.0], np.diag([0.05] * 4), {"car": 1.0}, TrackStatus.CONFIRMED, 20, 0, support=("lidar",), footprint=(4.5, 2.0, 0.0), travelled_m=0.0, recent_speed_mps=0.0)
    crossing = ObjectTrack(2, 0.0, [15, 4, 0.0, -1.2], np.diag([0.05] * 4), {"pedestrian": 1.0}, TrackStatus.CONFIRMED, 20, 0, support=("lidar",), travelled_m=2.0)
    standing = ObjectTrack(3, 0.0, [15, 3, 0.0, 0.0], np.diag([0.05] * 4), {"pedestrian": 1.0}, TrackStatus.CONFIRMED, 20, 0, support=("lidar",), travelled_m=0.0)
    lead = ObjectTrack(4, 0.0, [20, 0, 2.0, 0.0], np.diag([0.05] * 4), {"car": 1.0}, TrackStatus.CONFIRMED, 20, 0, support=("lidar",), travelled_m=5.0)
    tracks = [parked, crossing, standing, lead]
    preds = [predict(t, cfg) for t in tracks]
    assert [p.track_id for p in conflict_candidates(tracks, preds, lead_id=4, config=cfg)] == [2, 3]


def test_crossing_walker_ahead_is_yielded_to_at_walking_pace():
    """First live stack M04 (26 Oct): a walker crossing 12-16 m ahead cleared before the car
    would have reached it, so the timed conflict check never fired and the car did not slow.
    A vulnerable road user predicted anywhere on the path ahead is now yielded to."""
    from mapless.prediction import first_occupied_arc, predict
    from mapless.stack import StackConfig, crossing_yield_cap

    c = StackConfig()
    path = np.column_stack((np.arange(0.0, 40.0, 0.5), np.zeros(80)))
    # No detector: uniform class probabilities (best_class breaks the tie on "car"; the
    # second live M04 run took the walker for a car that way and never yielded).
    uniform = {k: 0.2 for k in ("car", "truck", "bike", "pedestrian", "unknown")}
    walker = ObjectTrack(1, 0.0, [18.0, -3.0, 0.0, 1.2], np.diag([0.05] * 4), uniform, TrackStatus.CONFIRMED, 20, 0, support=("lidar",), footprint=(0.6, 0.6, 0.0), travelled_m=1.5)
    assert walker.best_class == "car"
    pred = predict(walker, c.prediction)
    arc = first_occupied_arc([pred], path, BODY)
    # Between the car and the walker: the front overhang reaches the walker's predicted
    # region (which grows with its uncertainty over the 4 s horizon) before the walker.
    assert arc is not None and 0.0 < arc < 18.0 - BODY.front_m
    cap = crossing_yield_cap([walker], [pred], path, 0.0, BODY, 1.0, c)
    assert cap == pytest.approx(c.yield_mps)  # far away: walking pace, not a stop
    close = crossing_yield_cap([walker], [pred], path, arc - c.yield_margin_m, BODY, 1.0, c)
    assert close == pytest.approx(0.0)  # at the margin: stop
    # A car-sized track moving across (a cut-in) is the conflict check's, not a yield.
    car = ObjectTrack(2, 0.0, [18.0, -3.0, 0.0, 1.2], np.diag([0.05] * 4), uniform, TrackStatus.CONFIRMED, 20, 0, support=("lidar",), footprint=(4.5, 1.9, 0.0), travelled_m=1.5)
    assert crossing_yield_cap([car], [predict(car, c.prediction)], path, 0.0, BODY, 1.0, c) is None
    # A walker standing on the pavement is not yielded to until it moves.
    still = ObjectTrack(3, 0.0, [18.0, -4.5, 0.0, 0.0], np.diag([0.05] * 4), uniform, TrackStatus.CONFIRMED, 20, 0, support=("lidar",), footprint=(0.6, 0.6, 0.0), travelled_m=0.0)
    # A detector's decided class still wins over size: a classified pedestrian is yielded to.
    tagged = ObjectTrack(4, 0.0, [18.0, -3.0, 0.0, 1.2], np.diag([0.05] * 4), {"pedestrian": 0.9, "unknown": 0.1}, TrackStatus.CONFIRMED, 20, 0, support=("lidar",), footprint=None, travelled_m=1.5)
    assert crossing_yield_cap([tagged], [predict(tagged, c.prediction)], path, 0.0, BODY, 1.0, c) == pytest.approx(c.yield_mps)
    assert crossing_yield_cap([still], [predict(still, c.prediction)], path, 0.0, BODY, 1.0, c) is None


def test_moving_needs_sustained_recent_motion_not_a_one_off_centroid_shift():
    """Third live stack M04 (26 Oct): small static things beside the road made the car yield
    when their centroids shifted ~0.8 m as other faces came into view. Motion now also
    needs a mean speed of 0.5 m/s over the last second; a walker does 1.2 m/s."""
    from mapless.prediction import is_moving

    cfg = PredictionConfig()
    pole, walker = Tracker(TrackerConfig()), Tracker(TrackerConfig())
    for k in range(60):
        t = k * 0.05
        shift = 0.8 if k >= 20 else 0.0  # one shift at 1 s, then still
        pole.update(t, ORIGIN, lidar=[LidarMeasurement(np.array([20.0 + shift, -4.0]), (0.4, 0.4, 0.0))])
        walker.update(t, ORIGIN, lidar=[LidarMeasurement(np.array([20.0, -4.0 + 1.2 * t]), (0.5, 0.5, 0.0))])
    (p,), (w,) = pole.snapshot(), walker.snapshot()
    assert p.travelled_m > 0.75 and p.recent_speed_mps < 0.5 and not is_moving(p, cfg)
    assert w.recent_speed_mps == pytest.approx(1.2, abs=0.15) and is_moving(w, cfg)


def test_only_plausible_crossers_count_the_tracks_from_the_fourth_live_m04_run():
    """Fourth live stack M04 (26 Oct): the cap_track log named the false triggers: thin
    tracks 4.7-15 m right of the path (walls and fences whose visible stretch slides
    past at about the car's speed), or ones heading away. Track 529 was the walker."""
    from mapless.prediction import predict
    from mapless.stack import StackConfig, plausible_crossers

    c = StackConfig()
    path = np.column_stack((np.arange(-2.0, 40.0, 0.5), np.zeros(84)))
    logged = {  # track id: (x, y, vx, vy, footprint length, width, travelled, recent speed), vehicle frame
        271: (6.33, -7.18, 1.42, 0.95, 0.73, 0.3, 1.53, 1.61),
        334: (2.08, -15.0, 1.6, 2.03, 2.19, 0.31, 1.88, 2.68),
        360: (2.5, -7.89, 3.56, 0.19, 0.67, 0.56, 1.55, 2.82),
        715: (-2.45, -15.37, -0.18, 3.48, 1.04, 0.32, 0.87, None),
        771: (14.2, -4.66, -0.11, -0.35, 1.51, 0.49, 0.9, 1.38),
        842: (11.93, -6.55, 2.9, 1.26, 0.3, 0.3, 2.41, 2.84),
        529: (14.99, -5.07, 1.42, 2.73, 1.17, 0.3, 1.94, None),
    }
    uniform = {k: 0.2 for k in ("car", "truck", "bike", "pedestrian", "unknown")}
    tracks = [
        ObjectTrack(tid, 0.0, [x, y, vx, vy], np.diag([0.05] * 4), uniform, TrackStatus.CONFIRMED, 20, 0, support=("lidar",), footprint=(fl, fw, 0.0), travelled_m=tr, recent_speed_mps=rs)
        for tid, (x, y, vx, vy, fl, fw, tr, rs) in logged.items()
    ]
    preds = [predict(t, c.prediction) for t in tracks]

    class Walls:
        # The structures these tracks were: walls and fences 12 m and more to the right,
        # seen steadily (persistent tall evidence), as on the live run. The car drove at 2 m/s.
        def tall_evidence(self, xy, max_age_s=None):
            return np.where(np.asarray(xy)[:, 1] < -12.0, 60.0, 0.0)

    kept = plausible_crossers(tracks, preds, path, keep_right_m=4.197, config=c, ego_speed_mps=2.0, grid=Walls())
    assert [p.track_id for p in kept] == [529]
    # Off the road but heading across it on open ground, with no slide: a cyclist out of a
    # side road is kept (review of 3e9083a, H3: the reach alone dropped it until 3.75 m out).
    cyclist = ObjectTrack(5, 0.0, [7.0, -8.0, 0.0, 5.0], np.diag([0.05] * 4), uniform, TrackStatus.CONFIRMED, 20, 0, support=("lidar",), footprint=(1.8, 0.6, math.pi / 2), travelled_m=4.0, recent_speed_mps=5.0)
    assert [p.track_id for p in plausible_crossers([cyclist], [predict(cyclist, c.prediction)], path, 4.197, c, ego_speed_mps=2.0, grid=Walls())] == [5]
    # A car-sized mover is left to the timed conflict check, wherever it is.
    car = ObjectTrack(9, 0.0, [20.0, -12.0, 0.0, 3.0], np.diag([0.05] * 4), uniform, TrackStatus.CONFIRMED, 20, 0, support=("lidar",), footprint=(4.5, 1.9, 0.0), travelled_m=5.0)
    assert [p.track_id for p in plausible_crossers([car], [predict(car, c.prediction)], path, 4.197, c)] == [9]


def test_lead_role_keeps_a_diagonal_walker_crossing_and_gap_is_to_the_rear():
    """External review of 405b056: a walker ahead moving 0.4 m/s along and 1.2 m/s across
    was followed and dropped from the crossing checks; the following gap ran to the lead's
    centre (2.25 m generous for a car) and its speed along the car's own axis."""
    from mapless.stack import StackConfig, _rear_extent, lead_role

    c = StackConfig()
    u = {k: 0.2 for k in ("car", "truck", "bike", "pedestrian", "unknown")}
    walker = ObjectTrack(1, 0.0, [15, 0, 0.4, 1.2], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 20, 0, footprint=(0.6, 0.5, 0.0), travelled_m=2.0, recent_speed_mps=1.25)
    assert lead_role(walker, 0.4, 1.2, c) == "crossing"
    car = ObjectTrack(2, 0.0, [15, 0, 3.0, 0.0], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 20, 0, footprint=(4.5, 1.9, 0.0), travelled_m=9.0, recent_speed_mps=3.0)
    assert lead_role(car, 3.0, 0.0, c) == "follow"
    parked = ObjectTrack(3, 0.0, [15, 0, 0.0, 0.0], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 20, 0, footprint=(4.5, 1.9, 0.0), travelled_m=0.0, recent_speed_mps=0.0)
    assert lead_role(parked, 0.0, 0.0, c) == "obstacle_stop"
    newcomer = ObjectTrack(4, 0.0, [15, 0, 0.1, 0.1], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 4, 0, footprint=(0.5, 0.5, 0.0))
    assert lead_role(newcomer, 0.1, 0.1, c) == "uncertain"
    # Rear extent along the path: half the length for an aligned car, half the width across.
    assert _rear_extent(car, 0.0) == pytest.approx(2.25)
    assert _rear_extent(car, math.pi / 2) == pytest.approx(0.95)
    assert _rear_extent(ObjectTrack(5, 0.0, [0, 0, 0, 0], np.eye(4), u, TrackStatus.CONFIRMED, 5, 0), 0.0) == pytest.approx(1.0)


def test_pieces_of_the_lead_are_not_conflicts_but_a_walker_before_it_is():
    """Braking-lead runs on 6b1f705 (26 Oct): 0.3 m tracks 1.5-4 m beyond the lead's rear
    face (its roof and wheels, split from its cluster) jittered sideways and stopped the
    car on every approach; once the lead itself came back as a fresh LiDAR-only track.
    Positions from the logs (vehicle frame); the lead's rear face is 2.13 m ahead of the
    bumper (B6 at 29.8 s)."""
    from mapless.prediction import predict
    from mapless.stack import StackConfig, outside_lead_body

    c = StackConfig()
    u = {k: 0.2 for k in ("car", "truck", "bike", "pedestrian", "unknown")}

    def track(tid, x, y, vx, vy, fl, fw):
        return ObjectTrack(tid, 0.0, [x, y, vx, vy], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 20, 0, support=("lidar",), footprint=(fl, fw, 0.0), travelled_m=1.0, recent_speed_mps=1.0)

    path = np.column_stack((np.arange(-2.0, 40.0, 0.5), np.zeros(84)))
    s_car = 2.0  # the corridor line starts 2 m behind the rear axle
    lead = track(365, BODY.front_m + 2.13 + 0.15, 0.0, 0.0, 0.0, 0.3, 1.8)  # its rear face, across the lane
    pieces = [track(1147, 9.83, 0.48, 0.17, 2.09, 0.46, 0.3), track(1197, 9.74, -0.59, -0.2, -2.59, 0.3, 0.3), track(1223, 9.73, -0.89, 0.05, -0.01, 0.3, 0.3), track(584, 7.2, -0.1, -2.42, -1.05, 2.59, 2.2)]
    keep = [
        track(1, 4.6, 0.5, 0.0, 1.2, 0.5, 0.5),  # a walker between the bumper and the lead
        track(2, 8.0, 3.5, 3.0, 0.0, 4.5, 1.9),  # a car in the next lane
        track(3, 9.0, -2.6, 0.0, 1.2, 0.5, 0.5),  # a walker beside the lead
        track(4, 18.0, 0.0, 0.0, 0.0, 0.5, 0.5),  # beyond the lead's body
    ]
    tracks = [lead, *pieces, *keep]
    preds = [predict(t, c.prediction) for t in tracks[1:]]
    ahead = lead.position[0] - 0.0  # lead_ahead's distance is from the rear axle
    kept = outside_lead_body(tracks, preds, lead, ahead, ORIGIN, path, s_car, c)
    assert sorted(p.track_id for p in kept) == [1, 2, 3, 4]
    # A narrow lead (a walker ahead) shadows only its own depth.
    walker_lead = track(9, 10.0, 0.0, 1.0, 0.0, 0.5, 0.5)
    behind = track(10, 12.0, 0.2, 0.0, 1.0, 0.5, 0.5)
    kept = outside_lead_body([walker_lead, behind], [predict(behind, c.prediction)], walker_lead, 10.0, ORIGIN, path, s_car, c)
    assert [p.track_id for p in kept] == [10]
    # A lead followed along the lane is a body however narrow its measured footprint (B10).
    kept = outside_lead_body([walker_lead, behind], [predict(behind, c.prediction)], walker_lead, 10.0, ORIGIN, path, s_car, c, followed=True)
    assert kept == []


def test_vehicles_beyond_a_kerb_or_railing_cannot_reach_the_path():
    """B8/A9/B9 (26 Oct): walls 13-23 m right of the path, their visible stretch sliding
    at 4-6 m/s, read as cars heading for the path and stopped the car (out of the
    radar's view). A vehicle's run towards the path must not meet a wall (persistent tall
    evidence across a car's width); walkers cross kerbs and are left alone."""
    from mapless.prediction import predict
    from mapless.stack import StackConfig, road_reachable

    class Grid:
        def __init__(self, railing):
            self.railing = railing

        def tall_evidence(self, xy):
            xy = np.asarray(xy, dtype=float)
            out = np.zeros(len(xy), dtype=np.float32)
            if self.railing:
                out[(xy[:, 1] < -5.5) & (xy[:, 1] > -6.5)] = 50.0  # seen standing for seconds
            return out

    c = StackConfig()
    u = {k: 0.2 for k in ("car", "truck", "bike", "pedestrian", "unknown")}

    def track(tid, x, y, vx, vy, fl, fw):
        return ObjectTrack(tid, 0.0, [x, y, vx, vy], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 20, 0, support=("lidar",), footprint=(fl, fw, 0.0), travelled_m=3.0, recent_speed_mps=4.0)

    path = np.column_stack((np.arange(-2.0, 60.0, 0.5), np.zeros(124)))
    wall = track(561, 14.23, -15.1, -0.28, 4.1, 5.51, 3.18)  # logged in B9
    walker = track(2, 14.0, -12.0, 0.0, 1.3, 0.5, 0.5)
    near = track(3, 20.0, -6.8, 0.0, 2.0, 4.5, 1.9)  # within kerb + reach: the timed check's
    tracks = [wall, walker, near]
    preds = [predict(t, c.prediction) for t in tracks]
    kept, dropped = road_reachable(tracks, preds, path, Grid(railing=True), keep_right_m=5.7, config=c)
    assert dropped == [561] and sorted(p.track_id for p in kept) == [2, 3]
    # The same car on an open side road reaches the lane: kept.
    kept, dropped = road_reachable(tracks, preds, path, Grid(railing=False), keep_right_m=5.7, config=c)
    assert dropped == [] and len(kept) == 3


def test_a_known_lead_coming_to_rest_is_a_stopped_obstacle_not_uncertain():
    """B9 (26 Oct): the lead's rear face (1.7 x 0.8 m, no class) read "uncertain" while it
    came to rest, its own walking-pace envelope became a "predicted conflict" and the car
    stopped 10 m short, twice. As wide as a car's rear, or seen for a second, and not
    moving along or across: the driver's obstacle stop."""
    from mapless.stack import StackConfig, lead_role

    c = StackConfig()
    u = {k: 0.2 for k in ("car", "truck", "bike", "pedestrian", "unknown")}
    rear = ObjectTrack(57, 0.0, [15, 0, 0.1, 0.0], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 200, 0, footprint=(0.84, 1.74, 0.0), travelled_m=20.0, recent_speed_mps=1.0)
    assert lead_role(rear, 0.1, 0.0, c) == "obstacle_stop"  # known (200 hits)
    young = ObjectTrack(58, 0.0, [15, 0, 0.1, 0.0], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 5, 0, footprint=(0.84, 1.74, 0.0), travelled_m=0.2, recent_speed_mps=None)
    assert lead_role(young, 0.1, 0.0, c) == "uncertain"
    assert lead_role(young, 0.1, 0.0, c, wide=True) == "obstacle_stop"
    # A walker crossing is still crossing, however long it has been seen.
    walker = ObjectTrack(59, 0.0, [15, 0, 0.1, 1.2], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 200, 0, footprint=(0.5, 0.5, 0.0), travelled_m=3.0, recent_speed_mps=1.2)
    assert lead_role(walker, 0.1, 1.2, c) == "crossing"


def test_a_predicted_conflict_caps_to_a_stop_short_of_it_not_to_rest_at_once():
    """A10 (26 Oct): a 0.3 m piece of the stopped lead, 9 m ahead, was a predicted conflict
    for three ticks and its zero cap braked the car from 2 m/s to rest. The cap is now the
    speed the car can stop from 2 m short of where that road user may be: zero up close."""
    from mapless.prediction import predict
    from mapless.stack import StackConfig, conflict_speed_cap

    c = StackConfig()
    path = np.column_stack((np.arange(-2.0, 40.0, 0.5), np.zeros(84)))
    far = predict(small_obstacle(1, (16.0, 0.0)), c.prediction)
    near = predict(small_obstacle(2, (6.0, 0.0)), c.prediction)
    s_now = 2.0  # the rear axle, on a line starting 2 m behind it
    far_cap = conflict_speed_cap([far], path, s_now, BODY, c)
    # The whole predicted envelope counts (timing ignored, as for yielding), so the stop
    # point is short of its widest step; still enough to carry on at 2 m/s for now.
    assert 2.0 < far_cap < math.sqrt(2 * c.conflict_decel_mps2 * (16.0 - BODY.front_m))
    assert conflict_speed_cap([near], path, s_now, BODY, c) == 0.0
    assert conflict_speed_cap([], path, s_now, BODY, c) == 0.0  # nothing found: stop


def test_radar_updates_only_the_speed_of_a_track_the_lidar_placed():
    """B10 (26 Oct): the radar return off one corner of a 3.3 x 2.2 m roadside object and
    the LiDAR's centroid pulled its position back and forth every tick; it read 0.4-0.5 m/s
    of motion and became a conflict. With the LiDAR this tick, the radar gives speed only."""
    from mapless.radar import RadarMeasurement

    sensor_xy = np.array([2.5, 0.0])

    def ret(xy, radial):
        rel = np.asarray(xy, dtype=float) - sensor_xy
        return RadarMeasurement("radar_front", np.asarray(xy, dtype=float), float(np.hypot(*rel)), float(math.atan2(rel[1], rel[0])), radial, 3, sensor_xy)

    tracker = Tracker(TrackerConfig())
    centre = np.array([10.5, -6.5])
    for k in range(60):
        corner = centre + (np.array([-1.2, 0.9]) if k % 2 else np.array([-1.0, -0.6]))
        tracks = tracker.update(k * 0.05, ORIGIN, lidar=[LidarMeasurement(centre, (3.3, 2.2, 0.0))], radar=[ret(corner, 0.0)])
    assert np.hypot(*(tracks[0].position - centre)) < 0.1
    assert tracks[0].speed_mps < 0.15


def test_sensor_support_expires_when_a_sensor_stops_updating_the_track():
    """External review of 405b056: one LiDAR hit long ago kept a radar-held track
    "LiDAR-supported" for good. Support now lists sensors seen within 0.5 s."""
    from mapless.radar import RadarMeasurement

    tracker = Tracker(TrackerConfig())
    sensor_xy = np.array([2.5, 0.0])
    for k in range(10):  # LiDAR and radar together
        t = k * 0.05
        xy = np.array([20.0, 0.0])
        rel = xy - sensor_xy
        radar = RadarMeasurement("radar_front", xy, float(np.hypot(*rel)), float(math.atan2(rel[1], rel[0])), 0.0, 3, sensor_xy)
        tracks = tracker.update(t, ORIGIN, lidar=[LidarMeasurement(xy, (4.5, 2.0, 0.0))], radar=[radar])
    assert set(tracks[0].support) == {"lidar", "radar"}
    for k in range(10, 30):  # radar only for a second
        t = k * 0.05
        xy = np.array([20.0, 0.0])
        rel = xy - sensor_xy
        radar = RadarMeasurement("radar_front", xy, float(np.hypot(*rel)), float(math.atan2(rel[1], rel[0])), 0.0, 3, sensor_xy)
        tracks = tracker.update(t, ORIGIN, radar=[radar])
    assert tracks[0].support == ("radar",)


def test_yield_only_to_pedestrian_speed_crossers_that_persist():
    """Live cut-in B3 (26 Oct): the cut-in car driving ahead in the lane (its rear face
    under 2 m across) and a one-tick 4.8 m/s fragment were yielded to, braking the car
    to 0.1 m/s. Only road users crossing the path at pedestrian speed, for 0.3 s, count."""
    from mapless.stack import StackConfig, crossing_yield

    c = StackConfig()
    u = {k: 0.2 for k in ("car", "truck", "bike", "pedestrian", "unknown")}
    path = np.column_stack((np.arange(0.0, 40.0, 0.5), np.zeros(80)))
    rear_face = ObjectTrack(1, 0.0, [18.0, 0.0, 2.9, 0.0], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 20, 0, footprint=(1.83, 1.62, 0.0), travelled_m=5.0, recent_speed_mps=2.9)
    fragment = ObjectTrack(2, 0.0, [14.5, -5.5, 1.5, 2.45], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 20, 0, footprint=(0.49, 0.3, 0.0), travelled_m=2.0, recent_speed_mps=4.78)
    walker = ObjectTrack(3, 0.0, [18.0, -3.0, 0.0, 1.2], np.diag([0.05] * 4), u, TrackStatus.CONFIRMED, 20, 0, footprint=(0.6, 0.6, 0.0), travelled_m=1.5, recent_speed_mps=1.2)
    assert crossing_yield([rear_face], [predict(rear_face, c.prediction)], path, 0.0, BODY, 1.0, c) == (None, None)
    # Cyclist-speed crossers are yielded to now (review of 3e9083a, H3: 2.5 m/s left a
    # runner at 2.6 m/s), so a one-tick fragment is kept out by the persistence alone.
    assert crossing_yield([fragment], [predict(fragment, c.prediction)], path, 0.0, BODY, 1.0, c, {}, 5.0) == (None, None)
    since = {}
    pred = [predict(walker, c.prediction)]
    assert crossing_yield([walker], pred, path, 0.0, BODY, 1.0, c, since, 10.0) == (None, None)  # just seen
    assert crossing_yield([walker], pred, path, 0.0, BODY, 1.0, c, since, 10.2) == (None, None)
    cap, tid = crossing_yield([walker], pred, path, 0.0, BODY, 1.0, c, since, 10.3)
    assert tid == 3 and cap == pytest.approx(c.yield_mps)
    crossing_yield([], [], path, 0.0, BODY, 1.0, c, since, 10.35)  # gone for a tick: restarts
    assert since == {}


def test_following_a_slower_car_settles_without_stop_go_and_still_brakes_for_the_worst_case():
    """Live slower cut-in (6f2aec3): behind a car at 1.4 m/s the target went 1.9 -> 0.4 ->
    1.9 m/s every 3 s; with only the jerk limit between the target and the wanted speed it
    overshot. The target now closes the error over ``speed_tau_s``. A car following it with
    a 0.3 s lag settles at the lead's speed and gap, and a lead braking at 6 m/s^2 is still
    stopped behind with the standstill gap."""
    def run(lead, v, gap, seconds=40.0, dt=0.05, lag=0.3):
        follow = FollowingController(FollowingConfig())
        speeds, gaps = [], []
        for k in range(int(seconds / dt)):
            u = lead(k * dt)
            target = follow.target_speed(6.0, v, gap, u, dt)
            v = max(0.0, v + (target - v) * dt / lag)
            gap += (u - v) * dt
            speeds.append(v)
            gaps.append(gap)
        return np.array(speeds), np.array(gaps)

    speeds, gaps = run(lambda t: 1.4, 1.9, 6.5)
    late = speeds[len(speeds) // 2 :]
    assert late.max() - late.min() < 0.05 and abs(late.mean() - 1.4) < 0.05
    assert abs(gaps[-1] - (FollowingConfig().gap0_m + FollowingConfig().headway_s * 1.4)) < 0.2
    speeds, gaps = run(lambda t: 3.0 if t < 10 else max(0.0, 3.0 - 6.0 * (t - 10)), 3.0, 10.0)
    assert gaps.min() >= FollowingConfig().gap0_m - 0.1 and speeds[-1] < 0.05
