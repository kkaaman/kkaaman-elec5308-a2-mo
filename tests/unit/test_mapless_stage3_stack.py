"""Stage 3, the stack's share: the review of 3e9083a's H2, H3, H5, M4, M5, M7, M8 and M9,
and the stack side of S5 (oncoming traffic on a left turn). Each test states the safe
behaviour on the stack's speed-cap logic alone, with a stand-in driver and grid."""

import math
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
    far_slow = track(2, 50.0, 3.5, -2.0, 0.0, fp=(4.5, 1.9, 0.0))  # 36 m out at 2 m/s: 12 s
    parked = track(3, 25.0, 3.5, 0.0, 0.0, fp=(4.5, 1.9, 0.0), travelled=0.0)
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

    slide = track(98, 9.0, -20.0, 0.0, 5.5, fp=(8.8, 0.32, math.pi / 2), travelled=4.0, recent=5.2)
    assert plausible_crossers([slide], predict_all(predictable_tracks([slide]), c.prediction), _path(), 1.75, c, ego_speed_mps=2.0, grid=Wall()) == []
    crossing_car = track(99, 9.0, -20.0, 0.0, 5.5, fp=(4.5, 1.9, math.pi / 2), travelled=4.0, recent=5.2)
    assert len(plausible_crossers([crossing_car], predict_all(predictable_tracks([crossing_car]), c.prediction), _path(), 1.75, c, ego_speed_mps=2.0, grid=Wall())) == 1


# -- third review (e93154f) ---------------------------------------------------------------
def _drive_past(stack, make_track, seconds=6.0, ego=6.0, cruise=6.0, dt=0.05):
    """Closed loop on the cap alone: the car holds the origin and the scene moves past it at
    the car's speed, which follows the cap (4 m/s^2 down, 1 m/s^2 up). ``make_track(t,
    travelled)`` gives the track at time ``t`` with the car ``travelled`` metres on. Returns
    one row per tick: (t, ego speed, cap, the stack's lead note, the cause's track id)."""
    rows, travelled, v = [], 0.0, ego
    for k in range(int(seconds / dt)):
        t = 10.0 + k * dt
        tracks = [make_track(k * dt, travelled)]
        preds = tuple(predict_all(predictable_tracks(tracks), stack.config.prediction))
        cap, _ = stack._speed_cap(SimpleNamespace(speed_mps=v), Pose2D(0, 0, 0), tracks, preds, t)
        stack.driver.speed_cap_mps = cap
        rows.append((t, v, cap, None if stack._lead is None else dict(stack._lead), None if stack._cause is None else stack._cause["track_id"]))
        want = cruise if cap is None else min(cruise, cap)
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
