"""Tenth technical review (30 Oct): the invariants it asks of the stack, as tests.

2B, junction clearance: loss of supporting perception must not turn a blocked turn into a
cleared one. 2C, safety restrictions continuous across transitions: the car's permitted
speed is the lowest of the valid caps, and no change of who or what restricts it (a
crossing car becoming the followed lead, the lead's track id changing, the lead lost for a
moment, a radar-only track handed to the LiDAR, a standing car moving off, a stop hold
released while another conflict stands) lifts a restriction the hazard still calls for.
The 2C tests run the stack's speed-cap logic closed loop on the cap (``_closed_loop``,
the driver's obstacle stop emulated for a lead handed to it) and check the permitted speed
tick to tick across the transition."""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from mapless.contracts import CLASSES, ObjectTrack, TrackStatus
from mapless.ego_motion import Pose2D
from mapless.following import FollowingController
from mapless.junction import JunctionConfig, OncomingGate
from mapless.obstacles import ObstacleConfig
from mapless.planner import PlannerConfig
from mapless.prediction import predict_all
from mapless.stack import MaplessStack, StackConfig, oncoming_verdict, predictable_tracks
from mapless.terrain import CellLabel, VehicleBody
from navigation_control.local import TrajectorySample

BODY = VehicleBody()
UNIFORM = {k: 1.0 / len(CLASSES) for k in CLASSES}


def track(tid, x, y, vx, vy, fp=(0.6, 0.6, 0.0), hits=40, travelled=None, recent=None, status=TrackStatus.CONFIRMED):
    speed = math.hypot(vx, vy)
    travelled = (3.0 if speed > 0.3 else 0.0) if travelled is None else travelled
    recent = speed if recent is None else recent
    return ObjectTrack(tid, 0.0, [x, y, vx, vy], np.diag([0.05] * 4), UNIFORM, status, hits, 0, support=("lidar",), footprint=fp, travelled_m=travelled, recent_speed_mps=recent)


class _Grid:
    def labels(self, xy, max_age_s=None):
        return np.full(len(np.asarray(xy).reshape(-1, 2)), CellLabel.ROAD, dtype=np.int8)

    def tall_evidence(self, xy, max_age_s=None):
        return np.zeros(len(np.asarray(xy).reshape(-1, 2)), dtype=np.float32)


class _Tracker:
    def live(self, tracks):
        return list(tracks)


def make_stack(cruise=2.0):
    """A stack shell whose driver planned along a straight lane (as the stage 3 stack tests')."""
    c = StackConfig()
    path = np.column_stack((np.arange(-2.0, 40.0, 0.5), np.zeros(84)))
    samples = tuple(TrajectorySample(float(x), float(y), cruise) for x, y in path)
    stack = MaplessStack.__new__(MaplessStack)
    stack.config = c
    stack.driver = SimpleNamespace(
        last_plan=SimpleNamespace(trajectory=SimpleNamespace(samples=samples), curve_speed_mps=50.0),
        last_corridor=SimpleNamespace(centre_body=path, ok=True, pose=Pose2D(0, 0, 0)),
        corridor=SimpleNamespace(keep_right_m=1.75),
        config=SimpleNamespace(body=BODY, planner=PlannerConfig(cruise_mps=cruise), obstacles=ObstacleConfig(), junction=JunctionConfig()),
        dt_s=0.05,
        grid=_Grid(),
        speed_cap_mps=None,
        clearing_opposing_lane=False,
    )
    stack.tracker = _Tracker()
    stack.following = FollowingController(c.following)
    stack._yield_since, stack._outside_lead_s, stack._blocked_ticks = {}, {}, {}
    stack._lead = stack._cause = stack._clearing = stack._oncoming = None
    stack._unreachable = []
    stack._cap_reason = ""
    stack._followed = stack._oncoming_block = None
    return stack

DT = 0.05
RISE = 1.0  # m/s^2: how fast the permitted speed may rise while its hazard stands (the car's own acceleration)


def _closed_loop(stack, scene, seconds=6.0, ego=4.0, cruise=6.0):
    """The car holds the origin and the scene moves past it at the car's speed, which follows
    the permitted speed (4 m/s^2 down, 1 m/s^2 up). ``scene(t, travelled)`` gives the tracks
    at time ``t`` with the car ``travelled`` metres on. A lead handed to the driver's obstacle
    stop is stopped for on the planner's ramp. One row per tick: (t, ego speed, permitted
    speed, cap, lead id, cause id, reason)."""
    rows, travelled, v = [], 0.0, ego
    planner = stack.driver.config.planner
    for k in range(int(seconds / DT)):
        t = 10.0 + k * DT
        tracks = scene(k * DT, travelled)
        unseen = lambda xy: np.zeros(len(np.asarray(xy).reshape(-1, 2)), dtype=bool)  # noqa: E731  (the LiDAR's view: nothing there seen free)
        preds = tuple(predict_all(predictable_tracks(tracks, unseen, stack.config), stack.config.prediction))
        cap, reason = stack._speed_cap(SimpleNamespace(speed_mps=v), Pose2D(0, 0, 0), tracks, preds, t)
        stack.driver.speed_cap_mps = cap
        permitted = cruise if cap is None else min(cruise, cap)
        if stack._lead is not None and stack._lead.get("checked_by") == "obstacle stop":
            permitted = min(permitted, math.sqrt(max(0.0, 2.0 * planner.decel_mps2 * (stack._lead["gap_m"] - planner.stop_gap_m))))
        rows.append((k * DT, v, permitted, cap, None if stack._lead is None else stack._lead["track_id"], None if stack._cause is None else stack._cause["track_id"], reason))
        v = max(0.0, v - 4.0 * DT) if permitted < v else min(permitted, v + 1.0 * DT)
        travelled += v * DT
    return rows


def _lifts(rows, during):
    """Ticks within ``during`` (t0, t1) where the permitted speed rose faster than RISE."""
    return [(round(b[0], 2), round(a[2], 2), round(b[2], 2), b[6]) for a, b in zip(rows, rows[1:]) if during[0] <= b[0] <= during[1] and b[2] > a[2] + RISE * DT + 1e-6]


def _lead(tid, x, v, status=TrackStatus.CONFIRMED, hits=40, travelled=None, recent=None):
    return track(tid, x, 0.0, v, 0.0, fp=(4.5, 1.9, 0.0), hits=hits, travelled=travelled, recent=recent, status=status)


def test_t10_2c_a_crossing_car_that_becomes_the_followed_lead_keeps_its_restriction():
    """A car coming diagonally into the lane ahead at 2 m/s along it and 1 m/s across: the
    conflict check restricts the car, then the car is the lead and following takes over."""
    def scene(t, travelled):
        return [track(7, 16.0 + 2.0 * t - travelled, 4.0 - 1.0 * t if t < 4.0 else 0.0, 2.0, -1.0 if t < 4.0 else 0.0, fp=(4.5, 1.9, math.atan2(-1.0, 2.0) if t < 4.0 else 0.0))]

    rows = _closed_loop(make_stack(cruise=6.0), scene, seconds=7.0)
    assert any(r[5] == 7 for r in rows) and any(r[4] == 7 for r in rows)  # a conflict, then a lead
    assert _lifts(rows, (0.0, 7.0)) == []


def test_t10_2c_the_followed_lead_changing_track_id_keeps_its_restriction():
    """A lead 2 m/s ahead of a car at 4 m/s, followed. At 2 s its track is replaced: the old
    one coasts 0.3 s, a new one is tentative for two ticks, then confirmed, with no history."""
    def scene(t, travelled):
        x = 14.0 + 2.0 * t - travelled
        out = []
        if t < 2.3:
            out.append(_lead(1, x, 2.0, status=TrackStatus.CONFIRMED if t < 2.0 else TrackStatus.COASTING))
        if t >= 2.0:
            age = t - 2.0
            out.append(_lead(2, x, 2.0 if age >= 0.5 else 2.0 * age / 0.5, status=TrackStatus.TENTATIVE if age < 0.1 else TrackStatus.CONFIRMED, hits=1 + int(age / DT), travelled=2.0 * age, recent=None if age < 1.0 else 2.0))
        return out

    rows = _closed_loop(make_stack(cruise=6.0), scene, seconds=5.0)
    assert any(r[4] == 1 for r in rows) and any(r[4] == 2 for r in rows)
    assert _lifts(rows, (1.0, 5.0)) == []


def test_t10_2c_a_lead_lost_for_half_a_second_keeps_its_restriction():
    """The same followed lead, unseen from 2.0 to 2.5 s (coasting 0.3 s, then gone), then
    found again as a new track. Before the fix the cap went to cruise for 0.2 s."""
    def scene(t, travelled):
        x = 14.0 + 2.0 * t - travelled
        if t < 2.0:
            return [_lead(1, x, 2.0)]
        if t < 2.3:
            return [_lead(1, x, 2.0, status=TrackStatus.COASTING)]
        if t < 2.5:
            return []
        age = t - 2.5
        return [_lead(3, x, 2.0, hits=1 + int(age / DT), travelled=2.0 * age, recent=None if age < 1.0 else 2.0)]

    rows = _closed_loop(make_stack(cruise=6.0), scene, seconds=5.0)
    assert _lifts(rows, (1.0, 5.0)) == []
    unseen = [r for r in rows if 2.3 <= r[0] < 2.5]
    assert all(r[3] is not None and "unseen" in r[6] for r in unseen)
    # Not seen again: the memory lasts lead_memory_s, then lets go.
    gone = _closed_loop(make_stack(cruise=6.0), lambda t, d: scene(t, d) if t < 2.3 else [], seconds=5.0)
    assert all(r[3] is None for r in gone if r[0] >= 2.3 + StackConfig().lead_memory_s + 0.1)  # last followed while coasting


def test_t10_2c_a_radar_only_crosser_handed_to_the_lidar_keeps_its_restriction():
    """A car crossing towards the path at 3 m/s, first by the radar alone (no footprint),
    then by the LiDAR too (a young twin), then as one LiDAR-held track with the older id.
    Before the fix the radar track was dropped as the twin's on the LiDAR's first fix,
    whose motion was not known yet: the cap went from 2.8 m/s to none for three ticks."""
    def scene(t, travelled):
        x, y = 16.0 - travelled, 9.0 - 3.0 * t
        radar = replace(track(5, x, y, 0.0, -3.0, fp=None, hits=30, travelled=3.0 * t + 2.0), support=("radar",))
        if t < 1.0:
            return [radar]
        lidar = track(9, x, y + 0.3, 0.0, -2.5, fp=(4.4, 1.9, -math.pi / 2), hits=1 + int((t - 1.0) / DT), travelled=3.0 * (t - 1.0), recent=None)
        if t < 1.15:
            return [radar, lidar]
        return [replace(track(5, x, y, 0.0, -3.0, fp=(4.4, 1.9, -math.pi / 2), hits=40, travelled=3.0 * t + 2.0), support=("lidar", "radar"))]

    rows = _closed_loop(make_stack(cruise=6.0), scene, seconds=2.6)
    assert any(r[3] is not None for r in rows if r[0] < 1.0)  # restricted on the radar alone
    # Across both hand-overs (later, the graded cap follows the car clearing the path).
    assert _lifts(rows, (0.5, 1.6)) == []


def test_t10_2c_a_standing_car_moving_off_into_the_path_is_restricted_in_time():
    """A car standing beside the path, 3.9 m left of it, 18 m ahead, measured by the tracker
    from its cluster's points (its leading edge included), moves off across the path at
    1.5 m/s after 1 s. Standing, it does not restrict the car; moving, it does within 0.5 s,
    and the car stops short of it (the leading-edge rule does not hold it still)."""
    from mapless.tracking import LidarMeasurement, Tracker, TrackerConfig  # noqa: PLC0415

    tracker = Tracker(TrackerConfig())
    state = {}

    def scene(t, travelled):
        xe = travelled
        y_near = 3.9 - max(0.0, t - 1.0) * 1.5
        x0 = 18.0
        end = [(x, y_near) for x in np.arange(x0 - 1.0, x0 + 1.01, 0.1)]
        side = [(x0 - 1.0, y) for y in np.arange(y_near, y_near + 4.2, 0.1)]
        pts = np.array(end + side) - np.array([xe, 0.0])
        centre = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
        along, across = (pts - centre) @ vt[0], (pts - centre) @ vt[1]
        extent = (float(np.ptp(along)), max(float(np.ptp(across)), 0.3), float(math.atan2(vt[0][1], vt[0][0])))
        out = tracker.update(10.0 + t, Pose2D(xe, 0.0, 0.0), lidar=[LidarMeasurement(centre, extent, {"car": 1.0}, len(pts), 0.3, plan_xy=pts)])
        # The tracker works in its odometry frame (the car moves); the stack scaffolding in
        # the car's frame at the origin: shift the tracks by the car's travel.
        state["edge"] = [tr.edge_speed_mps for tr in out]
        return [replace(tr, state=np.array([tr.state[0] - xe, tr.state[1], tr.state[2], tr.state[3]])) for tr in tracker.live(out) if tr.status in (TrackStatus.CONFIRMED, TrackStatus.COASTING)]

    rows = _closed_loop(make_stack(cruise=6.0), scene, seconds=5.0, ego=3.0)
    assert all(r[3] is None for r in rows if 0.5 <= r[0] < 1.0)  # standing: no restriction
    first = next(r[0] for r in rows if r[0] >= 1.0 and r[3] is not None)
    assert first <= 1.5
    assert min(r[1] for r in rows if r[0] > 3.0) <= 0.5  # stopped for it


def test_t10_2c_a_stop_hold_released_while_another_conflict_stands_keeps_the_car_stopped():
    """The driver's obstacle hold is released (its object gone, its region and the tube seen
    clear) while the stack's cap is zero for another road user: the plan's speed stays zero
    throughout, and rises once the cap lets go."""
    from mapless.corridor import Corridor  # noqa: PLC0415
    from mapless.obstacles import PathObservation, PathObstacle  # noqa: PLC0415
    from mapless.planner import LanePlanner, PlannerConfig  # noqa: PLC0415
    from mapless.terrain import VehicleBody  # noqa: PLC0415

    body = VehicleBody()
    planner = LanePlanner(PlannerConfig(), body, tube_half_m=body.half_width_m + 0.3)
    xs = np.arange(-1.0, 30.0, 0.5)
    line = np.column_stack((xs, np.zeros_like(xs)))
    corridor = Corridor(time_s=0.0, pose=Pose2D(0, 0, 0), centre_body=line, centre_odometry=line, valid_length_m=29.0, coefficients=(0.0, 0.0), station_x_m=xs, reference_y_m=np.zeros_like(xs), right_edge_m=np.full(len(xs), np.nan), left_edge_m=np.full(len(xs), np.nan), blocked=np.zeros(len(xs), dtype=bool), evidence_body=np.zeros((0, 2)), evidence_time_s=0.0, support_m=29.0)
    ga, gl = np.meshgrid(np.arange(0.0, 29.0, 0.25), np.arange(-1.6, 1.61, 0.2))
    ground = np.column_stack((ga.ravel(), gl.ravel()))
    blocked = PathObservation(PathObstacle(distance_m=8.0, lateral_m=0.0, points=40, lateral_min_m=-0.5, lateral_max_m=0.5, height_m=1.2), clear_to_m=8.0, ground=ground[ground[:, 0] < 8.0], ground_body=ground[ground[:, 0] < 8.0])
    clear = PathObservation(None, clear_to_m=29.0, ground=ground, ground_body=ground)
    speeds, states = [], []
    for k in range(140):
        obs = blocked if k < 20 else clear
        cap = 0.0 if k < 100 else None
        plan = planner.plan(corridor, obs, None, k * DT, speed_cap_mps=cap)
        states.append(plan.hold_state)
        speeds.append(max(s.speed_mps for s in plan.trajectory.samples) if plan.trajectory is not None else 0.0)
    assert states[10] and states[99] in ("", "clear")  # held, then released before the cap lets go
    assert max(speeds[:100]) <= 1e-6
    assert max(speeds[100:]) > 0.5


def _gate():
    return OncomingGate(Pose2D(0.0, 0.0, 0.0), (1.75, 5.25), (6.0, 14.0), 3.0, 12.0, 8.0, 40.0, 0, True, "")


def test_t10_2b_losing_the_radar_does_not_clear_a_blocked_turn():
    """A radar-only car 41 m from the crossing at 30 km/h blocks the turn. The radar drops
    out: the track coasts, then is deleted, and the stack's verdict alone would read clear.
    It stays "not clear" for oncoming_lost_hold_s after it last blocked; the LiDAR's view
    then has the car (40 m). A blocker seen to leave (driving away) releases it at once."""
    c, j = StackConfig(), JunctionConfig()
    stack = make_stack()
    stack._oncoming_block = None

    def verdict(tracks, t):
        return stack._hold_lost_blockers(oncoming_verdict(tracks, _gate(), j, c), tracks, t)

    coming = replace(track(5, 55.0, 3.5, -8.5, 0.0, fp=None), support=("radar",))
    assert not verdict([coming], 10.0)["clear"]
    coasting = replace(coming, status=TrackStatus.COASTING, support=())
    assert not verdict([coasting], 10.2)["clear"]  # still in the band: blocks on its own
    held = verdict([], 10.5)  # deleted
    assert not held["clear"] and held["held_lost"] == [5] and "no longer observed" in held["note"]
    assert not verdict([], 13.1)["clear"]
    assert verdict([], 13.3)["clear"]  # oncoming_lost_hold_s after it last blocked (10.2 s, coasting)
    # Seen leaving: excused on the evidence, released at once.
    assert not verdict([coming], 20.0)["clear"]
    away = replace(track(5, 30.0, 3.5, 6.0, 0.0, fp=(4.5, 1.9, 0.0)), support=("lidar", "radar"))
    assert verdict([away], 20.5)["clear"]
    # Without the stack's tracks for a tick the driver's gate keeps the last "not clear".
    from mapless.driver import resolve_oncoming_override  # noqa: PLC0415

    last = dict(verdict([coming], 30.0), time_s=30.0)
    assert resolve_oncoming_override(last, 30.2, j.oncoming_override_max_age_s)["clear"] is False


def test_t10_3_the_predicted_regions_are_logged_for_the_evaluator():
    """Tenth review, 3: the evaluator checks each prediction's region at 1 and 2 s against
    where the actor went; the stack logs them in the vehicle frame (``predictions_near``)."""
    from mapless.contracts import Prediction  # noqa: PLC0415
    from mapless.stack import near_predictions  # noqa: PLC0415

    times = np.arange(0.0, 3.01, 0.25)
    pose = Pose2D(100.0, 50.0, math.pi / 2)  # heading +y in odometry
    p = Prediction(7, times, np.column_stack((np.full(len(times), 100.0), 60.0 + 2.0 * times)), np.column_stack((2.4 + 0.1 * times, 1.1 + 0.1 * times)), math.pi / 2)
    rows = near_predictions([p], pose)
    assert [r[:2] for r in rows] == [[7, 1.0], [7, 2.0]]
    assert rows[0][2:4] == [12.0, 0.0] and rows[1][2:4] == [14.0, 0.0]  # 10 m ahead, then 2 m/s on
    assert rows[1][4:6] == [2.6, 1.3] and rows[0][6] == 0.0  # heading along the car's
    far = Prediction(8, times, np.column_stack((np.full(len(times), 100.0), 200.0 + times)), np.ones((len(times), 2)), 0.0)
    assert near_predictions([far], pose) == [] and near_predictions([p], None) == []


def _tracked(points_at, seconds=5.0, ego=4.0):
    """Closed loop on the cap with the tracker in it: ``points_at(t, travelled)`` gives the
    LiDAR cluster's plan points (odometry frame); the tracks are measured from them (leading
    edges included) and handed to the stack's cap logic in the car's frame."""
    from mapless.tracking import LidarMeasurement, Tracker, TrackerConfig  # noqa: PLC0415

    tracker = Tracker(TrackerConfig())

    def scene(t, travelled):
        pts_odo = points_at(t, travelled)
        pts = pts_odo - np.array([travelled, 0.0])
        centre = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
        along, across = (pts - centre) @ vt[0], (pts - centre) @ vt[1]
        extent = (float(np.ptp(along)), max(float(np.ptp(across)), 0.3), float(math.atan2(vt[0][1], vt[0][0])))
        out = tracker.update(10.0 + t, Pose2D(travelled, 0.0, 0.0), lidar=[LidarMeasurement(centre, extent, {"unknown": 1.0}, len(pts), 0.3, plan_xy=pts)])
        return [replace(tr, state=np.array([tr.state[0] - travelled, tr.state[1], tr.state[2], tr.state[3]])) for tr in tracker.live(out) if tr.status in (TrackStatus.CONFIRMED, TrackStatus.COASTING)]

    return _closed_loop(make_stack(cruise=6.0), scene, seconds=seconds, ego=ego)


def test_t10_3_paired_a_roadside_fence_and_a_car_out_of_the_same_driveway():
    """Tenth review, 3: the paired comparison. The same roadside place, 5 m left of the path
    and 22 m ahead: a fence 9 m long whose visible stretch slides with the car as it passes
    (its far end out of view), and a car pulling out of a driveway there towards the path at
    2 m/s. The fence never restricts the car; the car does within 0.7 s of moving, and the
    car is stopped short of it."""
    def fence(t, travelled):
        start = min(25.0, max(18.0, 18.0 + travelled - 6.0))  # its near part drops out of view as the car comes level
        return np.array([(x, 5.0 + 0.15 * (i % 2)) for i, x in enumerate(np.arange(start, 27.0, 0.1))])

    def driveway(t, travelled):
        y_near = 4.9 - max(0.0, t - 1.0) * 2.0
        end = [(x, y_near) for x in np.arange(21.0, 23.01, 0.1)]
        side = [(21.0, y) for y in np.arange(y_near, y_near + 4.4, 0.1)]
        return np.array(end + side)

    rows = _tracked(fence)
    assert all(r[3] is None for r in rows), [r for r in rows if r[3] is not None][:3]
    rows = _tracked(driveway)
    assert all(r[3] is None for r in rows if r[0] < 1.0)
    first = next(r[0] for r in rows if r[0] >= 1.0 and r[3] is not None)
    assert first <= 1.7  # 1.3 m moved, 3.6 m from the path (the same with the leading-edge rule off)
    assert min(r[1] for r in rows if r[0] > 3.0) <= 0.5
