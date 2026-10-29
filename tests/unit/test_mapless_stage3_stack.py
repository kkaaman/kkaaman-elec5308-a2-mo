"""Stage 3, the stack's share: the review of 3e9083a's H2, H3, H5, M4, M5, M7, M8 and M9,
and the stack side of S5 (oncoming traffic on a left turn). Each test states the safe
behaviour on the stack's speed-cap logic alone, with a stand-in driver and grid."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

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


class Grid:
    """Road everywhere except (predicate, label) rules; OCCUPIED cells are trails (a little tall evidence)."""

    def __init__(self, rules=()):
        self.rules = rules

    def labels(self, xy, max_age_s=None):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        out = np.full(len(xy), CellLabel.ROAD, dtype=np.int8)
        for pred, label in self.rules:
            out[pred(xy)] = label
        return out

    def tall_evidence(self, xy, max_age_s=None):
        return np.where(self.labels(xy) == CellLabel.OCCUPIED, 5.0, 0.0).astype(np.float32)


class Tracker:
    def live(self, tracks):
        return list(tracks)


def make_stack(pose=Pose2D(0, 0, 0), path_odo=None, grid=None, cruise=2.0, clearing=False):
    """A stack shell whose driver planned along ``path_odo`` (odometry; a straight lane by default)."""
    c = StackConfig()
    if path_odo is None:
        path_odo = pose.to_world(np.column_stack((np.arange(-2.0, 40.0, 0.5), np.zeros(84))))
    samples = tuple(TrajectorySample(float(x), float(y), cruise) for x, y in path_odo)
    stack = MaplessStack.__new__(MaplessStack)
    stack.config = c
    stack.driver = SimpleNamespace(
        last_plan=SimpleNamespace(trajectory=SimpleNamespace(samples=samples), curve_speed_mps=50.0),
        last_corridor=SimpleNamespace(centre_body=pose.to_body(np.asarray(path_odo)), ok=True, pose=pose),
        corridor=SimpleNamespace(keep_right_m=1.75),
        config=SimpleNamespace(body=BODY, planner=PlannerConfig(cruise_mps=cruise), obstacles=ObstacleConfig(), junction=JunctionConfig()),
        dt_s=0.05,
        grid=grid or Grid(),
        speed_cap_mps=None,
        clearing_opposing_lane=clearing,
    )
    stack.tracker = Tracker()
    stack.following = FollowingController(c.following)
    stack._yield_since, stack._outside_lead_s, stack._blocked_ticks = {}, {}, {}
    stack._lead = stack._cause = stack._clearing = stack._oncoming = None
    stack._unreachable = []
    stack._cap_reason = ""
    return stack


def cap_for(stack, tracks, speed=2.0, pose=Pose2D(0, 0, 0), t=10.0, persist=True):
    """The stack's cap; with ``persist`` evaluated twice 0.35 s apart (the yield's persistence)."""
    motion = SimpleNamespace(speed_mps=speed)
    preds = tuple(predict_all(predictable_tracks(tracks), stack.config.prediction))
    if persist:
        stack._speed_cap(motion, pose, tracks, preds, t - 0.35)
    stack.following.reset()
    return stack._speed_cap(motion, pose, tracks, preds, t)


def cap(tracks, **kw):
    stack_kw = {k: v for k, v in kw.items() if k in ("path_odo", "grid", "cruise", "clearing")}
    if "pose" in kw:
        stack_kw["pose"] = kw["pose"]
    return cap_for(make_stack(**stack_kw), tracks, **{k: v for k, v in kw.items() if k in ("speed", "pose")})[0]


def test_h2_the_graded_cap_is_the_lowest_over_every_conflicting_road_user():
    pulling_out = track(31, 6.0, -9.0, 0.0, 1.5, fp=(4.5, 1.9, math.pi / 2))
    crossing_first = track(30, 10.0, -6.0, 0.0, 6.0, fp=(4.5, 1.9, math.pi / 2))
    assert cap([pulling_out]) == 0.0
    assert cap([crossing_first, pulling_out]) == 0.0  # was 2.83: only the earliest set the cap


def test_h3_a_fast_cyclist_out_of_a_side_road_counts_before_it_is_near_the_kerb():
    assert cap([track(1, 7.0, -8.0, 0.0, 5.0, fp=(1.8, 0.6, math.pi / 2))]) is not None


def test_h3_a_runner_is_yielded_to():
    assert cap([track(1, 10.0, -3.0, 0.0, 2.6, fp=(0.6, 0.6, 0.0))], cruise=4.0, speed=4.0) <= 0.8 + 1e-9


def test_h5_the_lead_is_placed_on_the_planned_path_in_a_turn():
    """Mid right turn, a car crossing 9 m ahead: on the kerb corridor's stale line it was
    the "lead" and handed to the obstacle stop; on the turn's path it is a conflict."""
    radius = 8.0
    th = np.linspace(0, math.pi / 2, 60)
    turn = np.column_stack((radius * np.sin(th), -radius * (1 - np.cos(th))))
    exit_ = np.column_stack((np.full(30, radius), -radius - np.arange(0.5, 15.5, 0.5)))
    path = np.vstack((np.column_stack((np.arange(-2.0, 0.0, 0.5), np.zeros(4))), turn, exit_))
    pose = Pose2D(float(turn[25, 0]), float(turn[25, 1]), -float(th[25]))
    p = pose.to_world(np.array([[9.0, 0.0]]))[0]
    car = track(50, float(p[0]), float(p[1]), -4.0, 0.0, fp=(4.5, 1.9, math.pi))
    assert cap([car], path_odo=path, pose=pose) == 0.0


def test_m2_a_car_behind_another_crossing_car_is_still_reachable():
    trail = (lambda xy: (np.abs(xy[:, 0] - 12.0) < 1.0) & (xy[:, 1] > -6.0) & (xy[:, 1] < -3.0), CellLabel.OCCUPIED)
    assert cap([track(8, 12.0, -12.0, 0.0, 5.0, fp=(4.5, 1.9, math.pi / 2))], grid=Grid([trail])) is not None


def test_m4_an_oncoming_or_reversing_vehicle_in_the_lane_stays_a_conflict():
    assert cap([track(1, 20.0, 0.0, -4.0, 0.0, fp=(4.5, 1.9, 0.0))]) == 0.0
    assert cap([track(2, 9.5, 0.0, -1.5, 0.0, fp=(4.5, 1.9, 0.0))]) == 0.0


def test_m5_a_hesitant_walker_at_the_lane_edge_is_not_the_lead():
    assert cap([track(21, 10.0, -1.72, 0.0, 0.2, fp=(0.5, 0.5, 0.0), travelled=0.3, recent=0.35)]) is not None


def test_m7_a_group_of_walkers_crossing_is_yielded_to():
    assert cap([track(5, 12.0, -2.0, 0.0, 1.3, fp=(2.4, 1.0, math.pi / 2))]) <= 0.8 + 1e-9


def test_m8_a_tick_without_a_planned_path_holds_the_cap():
    walker = track(70, 7.0, -1.0, 0.0, 1.2, fp=(0.6, 0.6, 0.0))
    stack = make_stack()
    first = cap_for(stack, [walker])[0]
    assert first is not None
    stack.driver.speed_cap_mps = first
    stack.driver.last_plan = stack.driver.last_corridor = None
    held, reason = cap_for(stack, [walker], persist=False)
    assert held == first and "held" in reason


def test_m9_pieces_of_a_followed_narrow_lead_are_not_yielded_to():
    lead = track(10, 15.0, 0.0, 2.0, 0.0, fp=(1.6, 1.0, 0.0))
    piece = track(11, 16.9, 0.3, 0.1, 1.5, fp=(0.3, 0.3, 0.0), travelled=1.0, recent=1.0)
    stack = make_stack()
    cap_for(stack, [lead, piece])
    assert stack._cause is None or stack._cause["track_id"] != 11


# -- S5: the stack's view of a left turn's opposing lane -----------------------------------


def _gate():
    frame = Pose2D(0.0, 0.0, 0.0)
    return OncomingGate(frame, (1.75, 5.25), (6.0, 14.0), 3.0, 12.0, 8.0, 40.0, 0, True, "")


def test_s5_an_oncoming_car_that_can_reach_the_crossing_in_time_blocks_the_turn():
    c, j = StackConfig(), JunctionConfig()
    coming = track(1, 50.0, 3.5, -6.0, 0.0, fp=(4.5, 1.9, 0.0))  # 36 m out at 6 m/s: 5.1 s < 8 s
    verdict = oncoming_verdict([coming], _gate(), j, c)
    assert not verdict["clear"] and "track 1" in verdict["note"]
    far_slow = track(2, 80.0, 3.5, -2.0, 0.0, fp=(4.5, 1.9, 0.0))  # 64 m out at 2 m/s: 8.4 s speeding up at 2 m/s^2 to 8.5
    parked = _history(track(3, 25.0, 4.6, 0.0, 0.0, fp=(4.5, 1.9, 0.0), travelled=0.0), 0.0, 10.0)  # at the far kerb, never moved (F5-07)
    leaving = track(4, 20.0, 3.5, 6.0, 0.0, fp=(4.5, 1.9, 0.0))  # the car's way: away from it
    verdict = oncoming_verdict([far_slow, parked, leaving], _gate(), j, c)
    assert verdict["clear"] and len(verdict["excused"]) == 3  # the LiDAR check ignores their cells
    beyond_lidar = track(5, 95.0, 3.5, -8.5, 0.0, fp=None)  # a radar track 80 m out at 30 km/h: 9.4 s
    assert oncoming_verdict([beyond_lidar], _gate(), j, c)["clear"]
    nearer = track(6, 75.0, 3.5, -8.5, 0.0, fp=None)  # 7 s
    assert not oncoming_verdict([nearer], _gate(), j, c)["clear"]


def test_s5_in_the_opposing_lane_the_car_clears_it_rather_than_stopping_in_it():
    """A vehicle predicted where the car already stands: a zero cap would keep the car in
    its way. Outside the opposing lane the same conflict stops the car."""
    oncoming = track(1, 6.0, 1.0, -6.0, 0.0, fp=(4.5, 1.9, 0.0))
    assert cap([oncoming]) == 0.0
    stack = make_stack(clearing=True)
    clear_cap = cap_for(stack, [oncoming])[0]
    assert clear_cap is None or clear_cap > 0.0
    assert stack._clearing is not None and stack._clearing["track_id"] == 1
    walker = track(2, 4.5, -1.0, 0.0, 1.0, fp=(0.6, 0.6, 0.0))  # people on foot are never driven at
    assert cap([walker], clearing=True) <= 0.8 + 1e-9


# -- the first live stack turn (ff1910b, Town10HD site 0) ---------------------------------


def _path():
    return np.column_stack((np.arange(-2.0, 40.0, 0.5), np.zeros(84)))


def test_a_parked_car_beside_the_path_drifting_as_the_car_passes_does_not_stop_it():
    """Live: a parked car 2.4 m to the right, its centroid drifting 0.3-0.46 m/s the car's
    way as its visible part changed, read "uncertain" and stopped the car every 2 s."""
    from mapless.stack import parked_beside  # noqa: PLC0415

    parked = track(30, 3.5, -2.4, 0.35, 0.1, fp=(4.4, 1.8, 0.0), travelled=1.0, recent=0.35)
    assert cap([parked]) is None
    kept, dropped = parked_beside([parked], predict_all(predictable_tracks([parked]), StackConfig().prediction), _path(), BODY, StackConfig())
    assert not kept and dropped == [30]
    pulling_out = track(31, 6.0, -2.4, 0.3, 0.8, fp=(4.4, 1.8, 0.0), travelled=1.0, recent=0.85)
    assert cap([pulling_out]) == 0.0  # coming across: a conflict at once
    alongside = track(32, 6.0, -1.7, 0.3, 0.0, fp=(4.4, 1.8, 0.0), travelled=1.0, recent=0.35)
    kept, _ = parked_beside([alongside], predict_all(predictable_tracks([alongside]), StackConfig().prediction), _path(), BODY, StackConfig())
    assert kept  # its side within 0.1 m of the swept body: left to the conflict check


def test_a_thin_stretch_of_railing_is_not_a_group_on_foot():
    from mapless.stack import is_group  # noqa: PLC0415

    c = StackConfig()
    assert not is_group(track(15, 3.4, -8.0, -0.1, 0.4, fp=(3.4, 0.37, 0.0), travelled=1.1, recent=0.8), c)
    assert is_group(track(5, 12.0, -2.0, 0.0, 1.3, fp=(2.4, 1.0, math.pi / 2)), c)


def test_a_young_fragment_far_off_the_road_is_not_a_crosser_yet():
    """Live: a 0.55 m fragment 18 m right of the path, a few ticks old, "heading for the
    path" at 4.7 m/s, capped the car to 0. Off the road the motion must be sustained first."""
    from dataclasses import replace  # noqa: PLC0415

    from mapless.stack import plausible_crossers  # noqa: PLC0415

    c = StackConfig()
    young = replace(track(162, 1.0, -18.5, -0.2, 4.7, fp=(0.55, 0.3, 0.0), travelled=1.2), recent_speed_mps=None)
    preds = predict_all(predictable_tracks([young]), c.prediction)
    assert plausible_crossers([young], preds, _path(), 1.75, c, ego_speed_mps=2.0) == []
    sustained = replace(young, recent_speed_mps=4.6)
    preds = predict_all(predictable_tracks([sustained]), c.prediction)
    assert [p.track_id for p in plausible_crossers([sustained], preds, _path(), 1.75, c, ego_speed_mps=2.0)] == [162]


def test_the_second_live_stack_turn_false_triggers_are_dropped():
    """7eaca3d, Town10HD site 0: shoulder-parked cars whose tracked centre sits near their
    near face (centre less half-width 1.03-1.12 m from the path), one-cell clusters at
    6-10 m/s 13-23 m out at the junction, and an 8.8 x 0.32 m "vehicle" sliding 20 m out
    on a wall stopped the car 104 ticks. Recorded frames through the fixed stack: 40, of
    which 33 are an unclassified blob standing on the junction's corner (kept: it may be
    people)."""
    from dataclasses import replace  # noqa: PLC0415

    from mapless.stack import parked_beside, plausible_crossers  # noqa: PLC0415

    c = StackConfig()
    parked = track(60, 1.8, -2.1, 0.43, -0.03, fp=(4.8, 2.0, 0.0), travelled=2.3, recent=0.46)
    assert parked_beside([parked], predict_all(predictable_tracks([parked]), c.prediction), _path(), BODY, c)[1] == [60]
    blip = track(551, 2.0, -19.3, 0.0, 7.2, fp=(0.3, 0.3, 0.0), travelled=6.0, recent=6.5)
    slow_blip = replace(blip, footprint=(0.3, 0.3, 0.0), state=np.array([2.0, -12.0, 0.0, 3.0]), recent_speed_mps=2.8)
    for t in (blip, slow_blip):
        assert plausible_crossers([t], predict_all(predictable_tracks([t]), c.prediction), _path(), 1.75, c, ego_speed_mps=2.0) == []

    class Wall:
        def tall_evidence(self, xy, max_age_s=None):
            return np.full(len(np.asarray(xy).reshape(-1, 2)), 30.0)

        def labels(self, xy, max_age_s=None):
            return np.full(len(np.asarray(xy).reshape(-1, 2)), CellLabel.ROAD, dtype=np.int8)  # a road beside the wall

    slide = track(98, 9.0, -20.0, 0.0, 5.5, fp=(8.8, 0.32, math.pi / 2), travelled=4.0, recent=5.2)
    assert plausible_crossers([slide], predict_all(predictable_tracks([slide]), c.prediction), _path(), 1.75, c, ego_speed_mps=2.0, grid=Wall()) == []
    crossing_car = track(99, 9.0, -20.0, 0.0, 5.5, fp=(4.5, 1.9, math.pi / 2), travelled=4.0, recent=5.2)
    assert len(plausible_crossers([crossing_car], predict_all(predictable_tracks([crossing_car]), c.prediction), _path(), 1.75, c, ego_speed_mps=2.0, grid=Wall())) == 1


# -- third review (e93154f) ---------------------------------------------------------------
def _drive_past(stack, make_track, seconds=6.0, ego=6.0, cruise=6.0, dt=0.05, obstacle_stop=False):
    """Closed loop on the cap alone: the car holds the origin and the scene moves past it at
    the car's speed, which follows the cap (4 m/s^2 down, 1 m/s^2 up). ``make_track(t,
    travelled)`` gives the track at time ``t`` with the car ``travelled`` metres on. Returns
    one row per tick: (t, ego speed, cap, the stack's lead note, the cause's track id).
    ``obstacle_stop``: a lead handed to the driver's stop is stopped for on the planner's
    ramp (1 m/s^2 to ``stop_gap_m`` short), as the driver does."""
    rows, travelled, v = [], 0.0, ego
    planner = stack.driver.config.planner
    for k in range(int(seconds / dt)):
        t = 10.0 + k * dt
        tracks = [make_track(k * dt, travelled)]
        preds = tuple(predict_all(predictable_tracks(tracks), stack.config.prediction))
        cap, _ = stack._speed_cap(SimpleNamespace(speed_mps=v), Pose2D(0, 0, 0), tracks, preds, t)
        stack.driver.speed_cap_mps = cap
        rows.append((t, v, cap, None if stack._lead is None else dict(stack._lead), None if stack._cause is None else stack._cause["track_id"]))
        want = cruise if cap is None else min(cruise, cap)
        if obstacle_stop and stack._lead is not None and stack._lead.get("checked_by") == "obstacle stop":
            want = min(want, math.sqrt(max(0.0, 2.0 * planner.decel_mps2 * (stack._lead["gap_m"] - planner.stop_gap_m))))
        v = max(0.0, v - 4.0 * dt) if want < v else min(want, v + 1.0 * dt)
        travelled += v * dt
    return rows


def test_r1_a_vehicle_moving_diagonally_into_the_path_stays_covered_through_the_lead_hand_over():
    """Third review, R1 (F3): a confirmed car crossing diagonally into the lane, swept
    through the point where it becomes the lead. While it is the lead it may leave the
    conflict check only for a rule that covers it, and the hand-over never lifts the
    restriction in force (it did: a 1.4 m/s conflict cap became a 6 m/s following target
    on the tick the car reached the tube), and the car never closes to the car's body."""
    for along in (0.5, 2.0, 4.0):
        for across in (0.5, 1.0, 2.0):
            heading = math.atan2(-across, along)

            def car(t, travelled, along=along, across=across, heading=heading):
                return track(7, 14.0 + along * t - travelled, 4.0 - across * t, along, -across, fp=(4.5, 1.9, heading))

            rows = _drive_past(make_stack(cruise=6.0), car)
            before = None
            for t, v, cap_now, lead, cause in rows:
                case = f"along {along}, across {across}, t {t - 10:.2f}: cap {cap_now}, lead {lead}"
                if lead is not None and lead["track_id"] == 7:
                    if lead["checked_by"] == "following":
                        assert lead["speed_mps"] > 0.3, case
                    elif lead["checked_by"] == "obstacle stop":
                        assert math.hypot(lead["speed_mps"], lead["across_mps"]) < 0.5, case
                    if before is not None and before[1] == 7 and before[0] is not None:
                        assert cap_now is not None and cap_now <= before[0] + 1.0 * 0.05 + 1e-6, case
                    gap = lead["gap_m"]
                    assert gap > 1.0, case
                before = (cap_now, lead["track_id"] if lead is not None else cause)


def test_r4_a_lead_at_a_steady_speed_is_followed_without_braking_below_it():
    """Third review, R4: a lead holding 3 or 5 m/s, its velocity jittering across the lane
    as a LiDAR track's does (0.15 m/s), with the car at the lead's speed and about the
    following gap: no cap well below the lead's speed, and no conflict cap at all."""
    for lead_mps, start_m in ((3.0, 16.0), (5.0, 20.0)):

        def lead(t, travelled, v=lead_mps, x0=start_m):
            return track(9, x0 + v * t - travelled, 0.1 * math.sin(3 * t), v, 0.15 * math.cos(5 * t), fp=(4.5, 1.9, 0.0))

        rows = _drive_past(make_stack(cruise=6.0), lead, seconds=15.0, ego=lead_mps, cruise=6.0)
        assert min(v for _, v, _, _, _ in rows) >= lead_mps - 2.0
        assert all(cause is None for *_, cause in rows)
        assert abs(rows[-1][1] - lead_mps) <= 0.3


def test_r5_a_pedestrian_standing_on_the_pavement_does_not_bring_the_car_to_a_stop():
    """Third review, R5: a walker (a decided pedestrian) standing on the pavement, either
    side, 2.5 to 3.5 m off the lane's centre. Its walking-pace envelope reached the path
    and the car stopped short of it for good; it is now passed at ``roadside_mps``. One
    standing at the lane's edge is still stopped for, and one that walks out yielded to."""
    pedestrian = {k: (0.9 if k == "pedestrian" else 0.025) for k in CLASSES}

    def walker(y, vy=0.0):
        def make(t, travelled):
            return ObjectTrack(5, 0.0, [25.0 - travelled, y + vy * t, 0.0, vy], np.diag([0.05] * 4), pedestrian, TrackStatus.CONFIRMED, 60, 0,
                               support=("lidar",), footprint=(0.5, 0.5, 0.0), travelled_m=abs(vy) * t, recent_speed_mps=abs(vy))
        return make

    roadside = StackConfig().roadside_mps
    for y in (-2.5, -3.0, -3.5, 2.5):
        for cruise in (3.0, 6.0):
            rows = _drive_past(make_stack(cruise=cruise), walker(y), seconds=8.0, ego=cruise, cruise=cruise)
            assert min(v for _, v, _, _, _ in rows) >= min(cruise, roadside) - 0.2, (y, cruise)
    # At the lane's edge: the stop stays.
    rows = _drive_past(make_stack(cruise=6.0), walker(-1.8), seconds=8.0)
    assert min(v for _, v, _, _, _ in rows) < 0.5
    # Walking out of the pavement into the lane: yielded to.
    rows = _drive_past(make_stack(cruise=6.0), walker(-3.0, vy=1.4), seconds=8.0)
    assert min(v for _, v, _, _, _ in rows) <= StackConfig().yield_mps + 1e-6


# -- fourth review (6f2aec3) ----------------------------------------------------------------
def test_t01_t02_a_vehicle_length_track_off_the_road_counts_only_where_there_is_road():
    """Fourth review, T01/T02 (Town05 on 6370b98, recorded frames): building faces 14-22 m
    right of the street, 3.5-9 m long and 0.3-2.4 m across, "approached" at 4-7 m/s and
    capped the car to 0, 38 m short of the lead; the grid had no road within 3 m of any of
    them. The same track with road around it (a car coming out of a side road) still sets
    the cap, and so does one where the grid has not seen the area."""
    face = track(30, 20.0, -20.0, -3.0, 6.0, fp=(5.5, 0.8, math.pi / 2), travelled=4.0, recent=6.5)
    pavement_and_buildings = (lambda xy: xy[:, 1] < -12.0, CellLabel.BARRIER)
    assert cap([face], grid=Grid([pavement_and_buildings]), cruise=4.0, speed=4.0) is None
    side_road = (lambda xy: (xy[:, 1] < -12.0) & (np.abs(xy[:, 0] - 20.0) > 4.0), CellLabel.BARRIER)
    assert cap([face], grid=Grid([side_road]), cruise=4.0, speed=4.0) == 0.0
    unseen = (lambda xy: xy[:, 1] < -12.0, CellLabel.UNKNOWN)
    assert cap([face], grid=Grid([unseen]), cruise=4.0, speed=4.0) == 0.0
    # Within the kerb's reach the rule does not apply: a car pulling out from the kerb.
    pulling_out = track(31, 9.0, -3.5, 0.0, 1.0, fp=(4.5, 1.9, 0.0), travelled=1.0, recent=1.0)
    assert cap([pulling_out], grid=Grid([(lambda xy: xy[:, 1] < -2.5, CellLabel.BARRIER)])) is not None


def test_a_slow_car_cutting_in_is_not_a_group_on_foot():
    """Live slower cut-in (6f2aec3): the car, seen as 3.9 x 1.6 m at 1.4 m/s, was yielded to
    as a group of walkers at 0.8 m/s and the car slowed to 0.25 m/s. A group is no wider
    than ``group_max_width_m``; the car is the conflict check's. A 2.4 x 1.0 m group is
    still yielded to (M7)."""
    from mapless.stack import is_group  # noqa: PLC0415

    c = StackConfig()
    assert not is_group(track(62, 12.0, 2.4, 1.3, -0.45, fp=(3.9, 1.6, -0.3), travelled=3.0, recent=1.4), c)
    assert is_group(track(5, 12.0, -2.0, 0.0, 1.3, fp=(2.4, 1.0, math.pi / 2)), c)


def test_a_vehicle_length_track_beyond_reach_moving_along_the_road_is_not_a_crosser():
    """Town05 recorded frames (6370b98): fence stretches 5-6 m right of the path, 4-8 m long
    and 0.3 m across, "oncoming" at 5 m/s along the road with no motion towards the path,
    reached it only through their envelopes and capped the car to 0. Beyond the kerb's
    reach a vehicle must be heading for the path; the same track turning towards it is
    kept at once, and one within reach (the opposing lane) is kept as before (M4)."""
    fence = track(119, 38.0, -5.5, -5.2, -0.2, fp=(6.2, 0.3, 0.0), travelled=1.0, recent=None)
    assert cap([fence], cruise=4.0, speed=4.0) is None
    turning = track(120, 20.0, -5.5, -3.0, 2.0, fp=(4.5, 1.9, -0.6), travelled=2.0, recent=3.6)
    assert cap([turning], cruise=4.0, speed=4.0) is not None
    oncoming = track(1, 20.0, 3.3, -4.0, 0.0, fp=(4.5, 1.9, 0.0))
    assert cap([oncoming], cruise=4.0, speed=4.0) is not None


# -- fifth review (fcac32a) -----------------------------------------------------------------
LONG_LANE = np.column_stack((np.arange(-2.0, 120.0, 0.5), np.zeros(244)))


def _lead_profile(x0, v_max, accel=1.5, brake_at=None, brake=6.0):
    """A lead starting from rest ``x0`` m ahead: up to ``v_max`` at ``accel``, then, from
    ``brake_at`` s, braking at ``brake`` to a stop. Returns ``make_track(t, travelled)``."""
    def state(t):
        t_up = v_max / accel if accel > 0 else 0.0
        if brake_at is None or t <= brake_at:
            if t <= t_up:
                return 0.5 * accel * t * t, accel * t
            return 0.5 * accel * t_up**2 + v_max * (t - t_up), v_max
        s0, v0 = state(brake_at)
        tb = min(t - brake_at, v0 / brake)
        return s0 + v0 * tb - 0.5 * brake * tb * tb, max(0.0, v0 - brake * tb)

    def make(t, travelled):
        s, v = state(t)
        return track(9, x0 + s - travelled, 0.0, v, 0.0, fp=(4.5, 1.9, 0.0), travelled=s if s > 0.3 else 0.0, recent=v)
    return make


def test_f5_01_a_lead_far_enough_ahead_does_not_slow_the_start():
    """Fifth review, F5-01 (live braking lead, fcac32a): the lead, 17 m ahead and moving
    off, was taken on at the start, the following target began at the car's 0.4 m/s and
    the car was braked from 2.0 to 0.1 m/s (0.26 m/s at 2 s, 2.15 before). A lead standing
    or moving off 20, 40 or 60 m ahead now leaves the start as it is without one, at the
    live cruise (2 m/s) and at 6 m/s, until the gap or the worst-case cap is below cruise;
    from then following restricts the car, never above the cap last in force."""
    for cruise in (2.0, 6.0):
        free = _drive_past(make_stack(cruise=cruise, path_odo=LONG_LANE), lambda t, s: track(9, 200.0, 0.0, 0.0, 0.0, fp=(4.5, 1.9, 0.0)), seconds=8.0, ego=0.0, cruise=cruise)
        for x0 in (20.0, 40.0, 60.0):
            for v_max in (0.0, 1.0, 3.0):
                stack = make_stack(cruise=cruise, path_odo=LONG_LANE)
                rows = _drive_past(stack, _lead_profile(x0, v_max), seconds=8.0, ego=0.0, cruise=cruise)
                case = f"cruise {cruise}, lead at {x0} m to {v_max} m/s"
                restricted = [k for k, r in enumerate(rows) if r[2] is not None and r[2] < cruise]
                first = restricted[0] if restricted else len(rows)
                assert all(abs(a[1] - b[1]) < 1e-9 for a, b in zip(rows[:first], free[:first], strict=True)), case
                if restricted:
                    # It restricts only once the lead's gap or worst case needs it.
                    lead = rows[first][3]["following"]
                    assert min(lead["gap_speed_mps"], lead["worst_case_cap_mps"]) < cruise, case
                    # The hand-over never lifts what is in force (R1).
                    for (_, _, before, *_), (_, _, now, *_) in zip(rows[first:], rows[first + 1 :], strict=False):
                        assert (cruise if now is None else now) <= (cruise if before is None else before) + 1.0 * 0.05 + 1e-6, case
                if cruise == 2.0 and x0 >= 20.0 and v_max >= 1.0:
                    # The live case: nothing to restrict at the start.
                    assert rows[int(2.0 / 0.05)][1] >= 1.9, case


def test_f5_01_the_same_lead_braking_in_the_approach_is_still_stopped_behind():
    """Fifth review, F5-01, the pair: the same leads, moving off and braking at 6 m/s^2 once
    the car is on its way. Following takes over the tick the gap or the worst case calls for
    it and the car stops at least ``standstill_m`` behind (the driver's obstacle stop once
    the lead stands)."""
    from mapless.following import FollowingConfig  # noqa: PLC0415

    f = FollowingConfig()
    for cruise in (2.0, 6.0):
        for x0 in (20.0, 40.0, 60.0):
            for brake_at in (4.0, 7.0):
                stack = make_stack(cruise=cruise, path_odo=LONG_LANE)
                rows = _drive_past(stack, _lead_profile(x0, 3.0, brake_at=brake_at), seconds=40.0, ego=0.0, cruise=cruise, obstacle_stop=True)
                case = f"cruise {cruise}, lead at {x0} m braking at {brake_at} s"
                gaps = [r[3]["gap_m"] for r in rows if r[3] is not None]
                assert min(gaps) >= f.standstill_m, case
                assert rows[-1][1] < 0.05, case
                for t, v, cap_now, lead, _ in rows:
                    if lead is None or "following" not in lead or t - 10.0 < brake_at:
                        continue
                    if min(lead["following"]["gap_speed_mps"], lead["following"]["worst_case_cap_mps"]) < cruise - 1e-6:
                        assert cap_now is not None, (case, t)  # restricting at once
                    if lead["following"]["worst_case_cap_mps"] < v - 1e-6:
                        assert cap_now <= v + 1e-6, (case, t)  # braking at once for the worst case


def test_f5_03_an_uncertain_lead_past_the_lines_end_is_followed_for_the_worst_case_only():
    """Fifth review, F5-03: a car moving off at 0.5 m/s whose body begins 3.75 m past the
    path line's end may not be on the car's way. It is the lead and its worst case still
    caps the car, but it does not slow the car for the following gap; the same car where
    the line reaches it does."""
    def run(line_end):
        lane = np.column_stack((np.arange(-2.0, line_end + 0.25, 0.5), np.zeros(int((line_end + 2.0) / 0.5) + 1)))
        stack = make_stack(cruise=4.0, path_odo=lane)
        lead = track(9, 24.0, 0.0, 0.5, 0.0, fp=(4.5, 1.9, 0.0), travelled=1.0, recent=0.5)
        return cap_for(stack, [lead], speed=4.0, persist=False), stack._lead
    (uncertain_cap, _), lead = run(18.0)
    assert lead["continuation"]["uncertain"] and lead["following"]["keep_gap"] is False
    assert uncertain_cap is None
    (certain_cap, _), lead = run(22.0)
    assert "continuation" in lead and not lead["continuation"]["uncertain"]
    assert certain_cap is not None and certain_cap < 4.0
    # Close up, the worst case caps it even uncertain.
    def close(line_end):
        lane = np.column_stack((np.arange(-2.0, line_end + 0.25, 0.5), np.zeros(int((line_end + 2.0) / 0.5) + 1)))
        stack = make_stack(cruise=4.0, path_odo=lane)
        lead = track(9, 13.5, 0.0, 0.5, 0.0, fp=(4.5, 1.9, 0.0), travelled=1.0, recent=0.5)
        return cap_for(stack, [lead], speed=4.0, persist=False)[0], stack._lead
    near_cap, lead = close(7.5)
    assert lead["continuation"]["uncertain"] and near_cap is not None and near_cap < 4.0


def _history(t, excursion, still):
    from dataclasses import replace  # noqa: PLC0415

    return replace(t, max_excursion_m=excursion, still_s=still)


def test_f5_07_a_stopped_oncoming_car_blocks_the_turn_while_it_could_move_off_in_time():
    """Fifth review, F5-07: a static track in the opposing lane was excused, so a car
    stopped at the junction could move off as the car crossed. A car is judged by the time
    it needs to reach the crossing accelerating at ``oncoming_accel_mps2`` (from rest when
    static), and excused as parked only when it has never moved over a long view and stands
    at the lane's far side. The pairs: parked at the far kerb vs stopped mid-lane; stopped
    near vs too far to arrive in time; creeping off slowly (excused at its own speed before)
    vs driving away."""
    c, j = StackConfig(), JunctionConfig()
    gate = _gate()

    def verdict(*tracks):
        return oncoming_verdict(list(tracks), gate, j, c)

    parked = _history(track(1, 25.0, 4.6, 0.0, 0.0, fp=(4.5, 1.9, 0.0), travelled=0.1), 0.1, 10.0)
    stopped = _history(track(2, 25.0, 3.5, 0.0, 0.0, fp=(4.5, 1.9, 0.0), travelled=0.1), 0.1, 10.0)  # mid-lane
    arrived = _history(track(3, 25.0, 4.6, 0.0, 0.0, fp=(4.5, 1.9, 0.0), travelled=18.0), 18.0, 10.0)  # came to a stop there
    just_seen = _history(track(4, 25.0, 4.6, 0.0, 0.0, fp=(4.5, 1.9, 0.0), travelled=0.1), 0.1, 1.5)
    assert verdict(parked)["clear"]
    for t in (stopped, arrived, just_seen):
        v = verdict(t)
        assert not v["clear"] and "may move off" in v["note"], t.track_id
    # Too far to get there in time from rest: 0.5 x 2 m/s^2 x 8 s^2 would be 64 m, capped at 8.5 m/s.
    far = track(5, 75.0, 3.5, 0.0, 0.0, fp=(4.5, 1.9, 0.0), travelled=0.1, hits=200)
    assert verdict(far)["clear"]
    near_far = track(6, 55.0, 3.5, 0.0, 0.0, fp=(4.5, 1.9, 0.0), travelled=0.1, hits=200)
    assert not verdict(near_far)["clear"]
    # Moving off: 1 m/s, 24 m from the crossing; at its own speed plus 1 m/s, 12 s (excused before).
    creeping = track(7, 40.0, 3.5, -1.0, 0.0, fp=(4.5, 1.9, 0.0), travelled=2.0, recent=1.0, hits=200)
    assert not verdict(creeping)["clear"]
    away = track(8, 40.0, 3.5, 1.0, 0.0, fp=(4.5, 1.9, 0.0), travelled=2.0, recent=1.0, hits=200)
    assert verdict(away)["clear"]


def _coherent(t, value=True, coherence=(1.0, 0.03, 0.1)):
    from dataclasses import replace  # noqa: PLC0415

    return replace(t, motion_coherent=value, coherence=coherence if value else (0.3, 0.4, 1.2))


def test_f5_02_off_the_road_a_car_moving_as_one_still_counts_and_a_buildings_face_does_not():
    """Fifth review, F5-02: beyond the kerb's reach with no road around it in a well-seen
    grid, a vehicle-sized track was dropped as scenery whatever it did. Paired: a car
    coming out of a driveway or across a verge (coherent motion, 3.5 m/s, one size for 40
    updates) counts there; a building's face (the Town05 tracks: incoherent, 6.7 m/s, or
    a fragment that merged into a large cluster and jumped) does not. Where the grid
    shows road (a car park at road level, an access road, a side road at an odd angle) or
    has seen too little (an entrance partly hidden), the track counts as before."""
    pavement = (lambda xy: xy[:, 1] < -6.0, CellLabel.BARRIER)
    driveway_car = track(32, 14.0, -12.0, 0.0, 3.5, fp=(4.5, 1.9, math.pi / 2), travelled=4.0, recent=3.5)
    assert cap([_coherent(driveway_car)], grid=Grid([pavement]), cruise=4.0, speed=4.0) is not None
    assert cap([_coherent(driveway_car, False)], grid=Grid([pavement]), cruise=4.0, speed=4.0) is None
    assert cap([driveway_car], grid=Grid([pavement]), cruise=4.0, speed=4.0) is None  # not yet shown to be coherent
    from dataclasses import replace  # noqa: PLC0415

    young = replace(_coherent(driveway_car), hits=12)
    assert cap([young], grid=Grid([pavement]), cruise=4.0, speed=4.0) is None  # not for long enough
    grew = _coherent(driveway_car, coherence=(1.0, 0.03, 4.0))
    assert cap([grew], grid=Grid([pavement]), cruise=4.0, speed=4.0) is None  # a fragment merged into a big cluster
    # The Town05 faces "approached" at 4-7 m/s incoherently (F6-03 judges them by that, not by their speed).
    face = track(30, 20.0, -20.0, -3.0, 6.0, fp=(5.5, 0.8, math.pi / 2), travelled=4.0, recent=6.5)
    assert cap([_coherent(face, False)], grid=Grid([pavement]), cruise=4.0, speed=4.0) is None
    # Road around it: counted whatever its coherence is not yet known.
    car_park = (lambda xy: (xy[:, 1] < -6.0) & ~((xy[:, 1] < -9.0) & (np.abs(xy[:, 0] - 14.0) < 6.0)), CellLabel.BARRIER)
    assert cap([driveway_car], grid=Grid([car_park]), cruise=4.0, speed=4.0) is not None
    access_road = (lambda xy: (xy[:, 1] < -6.0) & (np.abs(xy[:, 0] - 14.0) > 2.0), CellLabel.BARRIER)
    assert cap([driveway_car], grid=Grid([access_road]), cruise=4.0, speed=4.0) is not None
    skew = (lambda xy: (xy[:, 1] < -6.0) & (np.abs((xy[:, 0] - 14.0) - 0.6 * (xy[:, 1] + 12.0)) > 2.0), CellLabel.BARRIER)
    assert cap([driveway_car], grid=Grid([skew]), cruise=4.0, speed=4.0) is not None
    hidden = (lambda xy: xy[:, 1] < -6.0, CellLabel.UNKNOWN)
    assert cap([driveway_car], grid=Grid([hidden]), cruise=4.0, speed=4.0) is not None


def test_4_3_an_incoherent_vehicle_sized_track_within_reach_is_held_where_it_stands():
    """Fifth review, 4.3 (Town05 recorded frames, c882389): fence stretches 6.2 m right,
    within the 5.7 m kerb's reach, "moving" at 5.5 m/s, measured one update in twenty with
    a 0.27 m residual, capped the car. An incoherent vehicle-sized track is predicted
    standing still (its occupancy is the obstacle stop's): no cap. The same track moving
    coherently, a car coming in, is predicted on its motion and caps the car; so is an
    incoherent one in the path itself, standing there. A walker-sized track keeps its
    motion whatever its coherence."""
    def run(t):
        stack = make_stack(cruise=4.0)
        stack.driver.corridor.keep_right_m = 5.7
        return cap_for(stack, [t], speed=4.0)[0]

    fence = track(119, 18.0, -6.2, -5.0, 1.5, fp=(6.2, 0.3, 0.0), travelled=3.0, recent=5.0)
    assert run(_coherent(fence, False)) is None
    assert run(_coherent(fence)) is not None
    in_path = track(120, 12.0, 0.0, -2.0, 0.5, fp=(4.5, 1.9, 0.0), travelled=3.0, recent=2.0)
    assert run(_coherent(in_path, False)) is not None  # in the lane band its motion stands
    walker = track(5, 10.0, -4.0, 0.0, 1.4, fp=(0.6, 0.6, 0.0), travelled=2.0, recent=1.4)
    assert run(_coherent(walker, False)) == run(_coherent(walker))


def test_4_3_the_tracker_marks_a_jumping_intermittent_track_incoherent_and_a_car_coherent():
    from mapless.tracking import LidarMeasurement, Tracker, TrackerConfig  # noqa: PLC0415

    pose = Pose2D(0.0, 0.0, 0.0)
    tracker = Tracker(TrackerConfig())
    rng = np.random.default_rng(3)
    for k in range(30):
        t = k * 0.05
        car = LidarMeasurement(np.array([20.0 - 2.0 * t + rng.normal(0, 0.03), 8.0 + rng.normal(0, 0.03)]), (4.5, 1.9, 0.0), {"car": 0.9, "unknown": 0.1})
        face = [LidarMeasurement(np.array([25.0 + 5.0 * t + rng.normal(0, 0.5), -15.0 + rng.normal(0, 0.4)]), (6.0, 0.4, 0.0), {"unknown": 1.0})] if k % 3 == 0 else []
        tracks = tracker.update(t, pose, lidar=[car] + face)
    by_pos = {round(float(tr.position[1])): tr for tr in tracks}
    assert by_pos[8].motion_coherent is True and by_pos[8].coherence[0] == 1.0
    assert by_pos[-15].motion_coherent is False


def test_f5_06_a_radar_only_car_coming_across_where_the_lidar_sees_nothing_caps_the_car():
    """Fifth review, F5-06: radar-only tracks were never predicted onto the path, so a car
    coming across out of the LiDAR's reach (or before it clustered it) was protected only
    by the obstacle stop once in the path. A radar-only track moving on corroborated
    motion, its velocity explaining its radial speed, where the LiDAR has seen nothing, is
    now predicted as a car-sized vehicle. Paired: the same track where the LiDAR sees the
    place (a twin or a ghost), standing, barely travelled, or with its radial speed
    rejected, is not."""
    from dataclasses import replace  # noqa: PLC0415

    c = StackConfig()

    def radar_car(**kw):
        t = track(77, 10.0, -12.0, 0.0, 6.0, fp=None, travelled=3.0, recent=6.0)
        return replace(t, support=("radar",), **kw)

    def run(t, saw):
        # ``saw``: the LiDAR saw free ground where the track stands (F6-02's ``_lidar_free``).
        stack = make_stack(cruise=4.0)
        preds = tuple(predict_all(predictable_tracks([t], lambda xy: np.full(len(xy), saw), c), c.prediction))
        return stack._speed_cap(SimpleNamespace(speed_mps=4.0), Pose2D(0, 0, 0), [t], preds, 10.0)[0], preds

    capped, preds = run(radar_car(), saw=False)
    assert len(preds) == 1 and capped is not None and capped < 4.0
    assert run(radar_car(), saw=True)[1] == ()
    standing = replace(radar_car(), state=np.array([10.0, -12.0, 0.0, 0.0]), recent_speed_mps=0.0, travelled_m=0.0)
    assert run(standing, saw=False)[1] == ()
    assert run(radar_car(travelled_m=0.3), saw=False)[1] == ()
    assert run(radar_car(doppler_rejects=2), saw=False)[1] == ()
    # Without the policy's inputs (lidar view, config) only LiDAR-supported tracks are predicted.
    assert predictable_tracks([radar_car()]) == []


def test_4_1_a_crosser_the_car_meets_only_because_it_is_accelerating_or_slowing_counts():
    """Fifth review, 4.1: the conflict check timed the path at one speed (the cruise speed
    or the car's own, whichever is higher). A car coming across at 1.5 m/s, 10 m right of
    a point 6 m ahead, has crossed in front of a car holding 4 m/s, but meets one moving
    off from rest, one at 1 m/s speeding up, or one slowing under a cap (to 1 m/s or to a
    stop there). Those now count; for the car holding 4 m/s it still does not."""
    from mapless.stack import arrival_profiles  # noqa: PLC0415

    def run(v0, held):
        stack = make_stack(cruise=4.0)
        stack.driver.speed_cap_mps = held
        car = track(40, 6.0, -10.0, 0.0, 1.5, fp=(4.5, 1.9, math.pi / 2), travelled=3.0, recent=1.5)
        preds = tuple(predict_all(predictable_tracks([car]), stack.config.prediction))
        return stack._speed_cap(SimpleNamespace(speed_mps=v0), Pose2D(0, 0, 0), [car], preds, 10.0)[0]

    assert run(4.0, None) is None
    for v0, held in ((0.0, None), (1.0, None), (4.0, 1.0), (4.0, 0.0)):
        assert run(v0, held) is not None, (v0, held)
    # The profiles themselves: arrival no earlier than the nominal, and a stop standing.
    xy = np.column_stack((np.arange(-2.0, 40.0, 0.5), np.zeros(84)))
    d = xy[:, 0]
    c = StackConfig()
    names = {n: (pxy, t) for n, pxy, t in arrival_profiles(xy, d, 0.0, 4.0, None, c)}
    assert set(names) == {"nominal", "accelerating"}
    ahead = d > 0
    assert np.all(names["accelerating"][1][ahead] >= names["nominal"][1][ahead] - 1e-9)
    assert names["accelerating"][1][np.searchsorted(d, 8.0)] == pytest.approx(4.0, abs=0.05)  # 8 m from rest at 1 m/s^2
    stop_xy, stop_t = {n: (pxy, t) for n, pxy, t in arrival_profiles(xy, d, 4.0, 4.0, 0.0, c)}["slowing"]
    assert stop_xy[-1][0] == pytest.approx(4.0, abs=0.01) and stop_t[-1] >= c.conflict_horizon_s
    assert {n for n, *_ in arrival_profiles(xy, d, 4.0, 4.0, None, c)} == {"nominal"}


def test_f6_05_a_far_lead_that_replaces_a_near_slow_one_does_not_inherit_its_target():
    """Sixth review, F6-05: following kept one target whatever the lead. The car follows a
    slow lead 9 m ahead; it leaves the lane and a lead 35 m ahead at 4 m/s takes over on the
    next tick. The cap is then the one a car meeting only the far lead would get, not the
    slow lead's low target ramping up at 1 m/s^2."""
    def slow(x):
        return track(1, x, 0.0, 1.0, 0.0, fp=(4.5, 1.9, 0.0), travelled=3.0, recent=1.0)

    far = track(2, 35.0, 0.0, 4.0, 0.0, fp=(4.5, 1.9, 0.0), travelled=3.0, recent=4.0)
    lane = np.column_stack((np.arange(-2.0, 60.0, 0.5), np.zeros(124)))
    stack = make_stack(cruise=6.0, path_odo=lane)
    motion = SimpleNamespace(speed_mps=3.0)
    for _ in range(20):
        held, _ = stack._speed_cap(motion, Pose2D(0, 0, 0), [slow(9.0)], (), 10.0)
        stack.driver.speed_cap_mps = held
    assert held is not None and held < 3.0  # following the slow lead
    switched, _ = stack._speed_cap(motion, Pose2D(0, 0, 0), [far], (), 10.05)
    alone = make_stack(cruise=6.0, path_odo=lane)
    fresh, _ = alone._speed_cap(motion, Pose2D(0, 0, 0), [far], (), 10.05)
    assert switched == fresh


def test_f6_04_a_car_that_moved_and_came_back_is_not_parked():
    """Sixth review, F6-04: parked evidence was net displacement and an update count. A car
    that drove 2 m and reversed to within 0.5 m of where it was first seen, then stopped,
    has moved: it is judged as stopped traffic. The tracker keeps its farthest excursion
    and how long it has stood still."""
    from mapless.tracking import LidarMeasurement, Tracker, TrackerConfig  # noqa: PLC0415

    c, j = StackConfig(), JunctionConfig()
    tracker = Tracker(TrackerConfig())
    pose = Pose2D(0.0, 0.0, 0.0)
    # Out 2 m over 2 s, back 1.5 m over 1.5 s, then 8 s still, at the far kerb of the opposing lane.
    for k in range(int(11.5 / 0.05)):
        t = k * 0.05
        x = 25.0 + (min(t, 2.0) * 1.0) - (min(max(t - 2.0, 0.0), 1.5) * 1.0)
        tracks = tracker.update(t, pose, lidar=[LidarMeasurement(np.array([x, 4.6]), (4.5, 1.9, 0.0), {"car": 0.9, "unknown": 0.1})])
    car = next(tr for tr in tracks if abs(tr.position[1] - 4.6) < 0.5)
    assert car.travelled_m < 1.0 and car.max_excursion_m > 1.5 and car.still_s > 5.0
    v = oncoming_verdict([car], _gate(), j, c)
    assert not v["clear"]  # not excused as parked: it moved
    # The same car that never moved is parked.
    tracker = Tracker(TrackerConfig())
    for k in range(int(11.5 / 0.05)):
        tracks = tracker.update(k * 0.05, pose, lidar=[LidarMeasurement(np.array([25.0, 4.6]), (4.5, 1.9, 0.0), {"car": 0.9, "unknown": 0.1})])
    car = tracks[0]
    assert car.max_excursion_m < 0.3 and car.still_s > 10.0
    assert oncoming_verdict([car], _gate(), j, c)["clear"]


def test_f6_02_a_radar_only_car_is_a_provisional_threat_after_half_a_metre_and_a_full_one_after_two():
    """Sixth review, F6-02: at 2 m/s the 2 m of travel a full radar threat needs take a
    second, a second the car crossing out of a driveway has already used. From 0.5 m of
    travel at 1 m/s or faster (velocity explaining its radial speed) it is a provisional
    threat: predicted along its velocity, it slows the car to no less than
    ``radar_provisional_cap_mps``; from 2 m, a full one, which may stop it. Paired: not
    where the LiDAR saw free ground there (the old test, "the LiDAR saw the place", let a
    hit on the car itself count), nor within ``radar_twin_m`` of a LiDAR-supported track
    (its twin)."""
    from dataclasses import replace  # noqa: PLC0415

    from mapless.stack import provisional_radar, radar_threat  # noqa: PLC0415

    c = StackConfig()

    def radar_car(travelled, recent=2.0):
        t = track(77, 6.0, -4.0, 0.0, 2.0, fp=None, travelled=travelled, recent=recent)
        return replace(t, support=("radar",))

    def run(t, free=False, others=()):
        stack = make_stack(cruise=4.0)
        tracks = [t, *others]
        preds = tuple(predict_all([p for p in predictable_tracks(tracks, lambda xy: np.full(len(xy), free), c) if p.track_id == t.track_id], c.prediction))
        return stack._speed_cap(SimpleNamespace(speed_mps=4.0), Pose2D(0, 0, 0), tracks, preds, 10.0), preds

    never = lambda xy: np.zeros(len(xy), dtype=bool)  # noqa: E731
    young, sure = radar_car(0.6, recent=0.0), radar_car(3.0)
    assert radar_threat(radar_car(0.3), never, c) is None
    assert radar_threat(young, never, c) == "provisional" and provisional_radar(young, c)
    assert radar_threat(sure, never, c) == "full" and not provisional_radar(sure, c)
    (cap, reason), preds = run(young)
    assert len(preds) == 1 and cap == pytest.approx(c.radar_provisional_cap_mps) and "provisional" in reason
    (cap, reason), _ = run(sure)
    assert cap is not None and cap < c.radar_provisional_cap_mps
    # Free ground seen where it stands: a ghost. A LiDAR track beside it: its twin.
    assert run(young, free=True)[1] == ()
    twin = replace(track(5, 6.5, -3.0, 0.0, 2.0, fp=(4.5, 1.9, math.pi / 2), travelled=0.6), support=("lidar",))
    assert radar_threat(young, never, c, lidar_tracks=[twin]) is None
    far = replace(twin, state=np.array([30.0, 10.0, 0.0, 0.0]))
    assert radar_threat(young, never, c, lidar_tracks=[far]) == "provisional"
    # Across the line of sight its motion is its bearing's noise: 46 m out, 2 m of travel is
    # 1.2 sigma (the live crossing run's parked car); 7 m out, 0.6 m is 2.4.
    drift = replace(radar_car(2.1), state=np.array([46.0, -5.0, 0.0, -2.5]))
    assert radar_threat(drift, never, c, ego_xy=(0.0, 0.0)) is None and radar_threat(drift, never, c) == "full"
    assert radar_threat(young, never, c, ego_xy=(0.0, 0.0)) == "provisional"  # 7.2 m out: floor 0.5 m


def test_f6_03_off_the_road_a_car_is_judged_by_its_track_not_its_speed():
    """Sixth review, F6-03: off the road a vehicle-sized track was admitted only at 4 m/s or
    less, so a car leaving a driveway at 5-7 m/s was scenery. Now its track decides:
    coherent, its speed borne out by its sustained motion, one object of one size, and able
    to reach the lane band within the conflict horizon. Paired: cars coming out at 2, 3.5,
    5 and 7 m/s count; the same track at 7 m/s whose sustained motion is 2 m/s (a jump, not
    a car), incoherent, or too far out to reach the lane in time, does not (yet)."""
    from dataclasses import replace  # noqa: PLC0415

    c = StackConfig()
    pavement = (lambda xy: xy[:, 1] < -6.0, CellLabel.BARRIER)

    def car(speed, y=-10.0, recent=None):  # 4 m beyond the pavement's edge: no road within 3 m
        t = track(33, 14.0, y, 0.0, speed, fp=(4.5, 1.9, math.pi / 2), travelled=4.0, recent=speed if recent is None else recent)
        return _coherent(t)

    for speed in (2.0, 3.5, 5.0, 7.0):
        assert cap([car(speed)], grid=Grid([pavement]), cruise=4.0, speed=4.0) is not None, speed
    assert cap([car(7.0, recent=2.0)], grid=Grid([pavement]), cruise=4.0, speed=4.0) is None  # speed not borne out
    assert cap([_coherent(car(5.0), False)], grid=Grid([pavement]), cruise=4.0, speed=4.0) is None
    # 20 m out at 2 m/s: 8.9 s from the lane band, past the 4 s the check looks ahead.
    assert cap([car(2.0, y=-20.0)], grid=Grid([pavement]), cruise=4.0, speed=4.0) is None
    assert (20.0 - (c.following.lane_half_width_m + 0.5)) / 2.0 > c.conflict_horizon_s + 1.0
    # The rule itself, on its inputs: speed only as a physical bound.
    from mapless.stack import moves_as_a_car_off_the_road  # noqa: PLC0415

    assert moves_as_a_car_off_the_road(car(7.0), c)
    assert not moves_as_a_car_off_the_road(car(c.offroad_vehicle_max_mps + 1.0), c)
    assert not moves_as_a_car_off_the_road(replace(car(5.0), hits=c.offroad_vehicle_min_hits - 1), c)
    assert not moves_as_a_car_off_the_road(_coherent(car(5.0), coherence=(1.0, 0.03, 0.9)), c)
