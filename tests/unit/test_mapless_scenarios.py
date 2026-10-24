"""Moving-actor evaluator (M04–M06): box separation and per-scenario completion.

Fixes from the external review: separation as the oriented-polygon distance between
the ego box and the actor box (overlap reads 0), INCOMPLETE without actor evidence, and
a completion requirement per scenario instead of accepting a hold everywhere.
"""

import math
from types import SimpleNamespace

import pytest

from carla_adapter import mapless_scenarios as sc
from mapless.terrain import VehicleBody

BODY = VehicleBody(front_m=3.78, rear_m=-1.01, half_width_m=1.05)
GOAL = 60.0


def core_xy(location):
    return location.x, -location.y


def ego_row(tick, s, speed, offset=0.0, goal_distance=None, dt=0.05):
    return dict(
        tick=tick,
        time_s=tick * dt,
        truth=dict(rear_axle_xy=[s, offset], yaw_rad=0.0, speed_mps=speed, goal_distance_m=GOAL - s if goal_distance is None else goal_distance),
    )


def actor(x, y, yaw_rad=0.0, extent=(2.4, 1.0), speed=0.0):
    return dict(centre_xy=[x, y], yaw_rad=yaw_rad, extent_m=list(extent), speed_mps=speed)


# -- separation ------------------------------------------------------------------------------
def test_overlapping_boxes_read_zero_and_fail():
    log = [ego_row(0, 0.0, 2.0)]
    # Lead box 2.4 m long centred 4 m ahead of the rear axle: its rear face at 1.6 m is
    # inside the ego body (front at 3.78 m). The old centre-less-smallest-extent metric
    # could report a positive gap and PASS for overlapping boxes.
    moving = sc.score_moving(log, [actor(4.0, 0.0)], BODY)
    assert moving["min_separation_m"] == 0.0
    assert moving["separation_ok"] is False
    assert moving["ticks"] == 1


def test_long_vehicle_ahead_is_measured_from_its_rear_face():
    gap = sc.polygon_distance(sc.ego_vertices(ego_row(0, 0.0, 0.0)["truth"], BODY), sc.box_vertices((BODY.front_m + 2.4 + 0.5, 0.0), 0.0, (2.4, 1.0)))
    assert gap == pytest.approx(0.5)


def test_long_vehicle_beside_the_ego_is_measured_from_its_side():
    # A car across the lane (yaw 90 deg) with its long side facing the ego's left flank.
    a = actor(1.0, BODY.half_width_m + 2.4 + 0.5, yaw_rad=math.pi / 2)
    moving = sc.score_moving([ego_row(0, 0.0, 0.0)], [a], BODY)
    assert moving["min_separation_m"] == pytest.approx(0.5)
    assert moving["separation_ok"] is False
    # Rotated 90 deg the other way the box is the same; a half metre further and it passes.
    a = actor(1.0, BODY.half_width_m + 2.4 + 1.0, yaw_rad=-math.pi / 2)
    assert sc.score_moving([ego_row(0, 0.0, 0.0)], [a], BODY)["separation_ok"] is True


def test_separation_follows_the_ego_heading():
    row = dict(tick=0, time_s=0.0, truth=dict(rear_axle_xy=[0.0, 0.0], yaw_rad=math.pi / 2, speed_mps=0.0, goal_distance_m=GOAL))
    ahead = actor(0.0, BODY.front_m + 2.4 + 0.7, yaw_rad=math.pi / 2)
    assert sc.score_moving([row], [ahead], BODY)["min_separation_m"] == pytest.approx(0.7)


def test_no_actor_measurements_is_not_a_pass():
    moving = sc.score_moving([ego_row(0, 0.0, 1.0)], [None], BODY)
    assert moving == dict(min_separation_m=None, separation_ok=None, ticks=0)
    assert sc.score_moving([], [], BODY)["separation_ok"] is None


def test_box_truth_carries_the_bounding_box_offset():
    # CARLA frame: origin at (10, 5), yaw 90 deg (facing +y), box centre 1.2 m ahead and 0.1 m right.
    fake = SimpleNamespace(
        get_transform=lambda: SimpleNamespace(location=SimpleNamespace(x=10.0, y=5.0, z=0.0), rotation=SimpleNamespace(yaw=90.0)),
        bounding_box=SimpleNamespace(location=SimpleNamespace(x=1.2, y=0.1, z=0.7), extent=SimpleNamespace(x=2.4, y=1.0, z=0.7)),
        get_velocity=lambda: SimpleNamespace(x=0.0, y=3.0, z=0.0),
    )
    t = sc._box_truth(fake, core_xy)
    assert t["origin_xy"] == [10.0, -5.0]
    # Forward (+y CARLA) 1.2 m and right (-x CARLA) 0.1 m, then y flipped into the core frame.
    assert t["centre_xy"] == pytest.approx([9.9, -6.2])
    assert t["box_offset_m"] == [1.2, 0.1]
    assert t["yaw_rad"] == pytest.approx(-math.pi / 2)
    assert t["speed_mps"] == pytest.approx(3.0)


# -- completion ------------------------------------------------------------------------------
def crossing_run(yield_speed, resume=True, finish=True):
    """Ego cruises at 4 m/s; the walker 30 m ahead crosses right-to-left (7.2 m) over ticks 20..140."""
    log, truths = [], []
    s, speed = 0.0, 4.0
    for tick in range(240):
        if 20 <= tick < 120:
            speed = yield_speed
        elif tick >= 120:
            speed = 4.0 if resume else 0.0
        s += speed * 0.05
        log.append(ego_row(tick, s, speed))
        walking = 20 <= tick < 140
        y = -4.0 + 1.2 * 0.05 * max(0, min(tick, 140) - 20)  # core frame: right of the lane is -y
        truths.append(actor(30.0, y, yaw_rad=math.pi / 2, extent=(0.2, 0.2), speed=1.2 if walking else 0.0))
    end = "finished" if finish else "time limit"
    if finish:
        log[-1]["truth"]["goal_distance_m"] = 0.5
    return log, truths, end


def test_crossing_pedestrian_requires_a_yield_and_a_resumed_goal():
    log, truths, end = crossing_run(0.0)
    done = sc.score_completion("crossing_pedestrian", log, truths, BODY, end, 0.0)
    assert done["passed"] is True and done["yielded"] and done["walker_cleared"] and done["resumed"]
    assert done["required"].startswith("yield")


def test_crossing_pedestrian_fails_without_a_yield():
    log, truths, end = crossing_run(4.0)
    done = sc.score_completion("crossing_pedestrian", log, truths, BODY, end, 0.0)
    assert done["passed"] is False and done["yielded"] is False


def test_crossing_pedestrian_fails_when_held_after_the_walker_clears():
    log, truths, _ = crossing_run(0.0, resume=False, finish=False)
    done = sc.score_completion("crossing_pedestrian", log, truths, BODY, "held at the obstacle", 5.0)
    assert done["passed"] is False and done["resumed"] is False and done["walker_cleared"]


def test_crossing_pedestrian_without_a_crossing_is_incomplete():
    log = [ego_row(t, t * 0.2, 4.0) for t in range(50)]
    truths = [actor(30.0, -4.0, yaw_rad=math.pi / 2, extent=(0.2, 0.2))] * 50  # never stepped off the pavement
    assert sc.score_completion("crossing_pedestrian", log, truths, BODY, "finished", 0.0)["passed"] is None
    assert sc.score_completion("crossing_pedestrian", log, [None] * 50, BODY, "finished", 0.0) == pytest.approx(
        dict(required=sc.COMPLETION_REQUIRED["crossing_pedestrian"], end="finished", goal_distance_m=GOAL - 49 * 0.2, passed=None, note="no actor measurements")
    )


def cut_in_run(gap_after_m, finish=True):
    """A lead level with the ego in the left lane moves over by tick 40 and holds ``gap_after_m``."""
    log, truths = [], []
    for tick in range(200):
        s = tick * 0.2
        log.append(ego_row(tick, s, 4.0))
        y = max(0.0, 3.5 - 3.5 * tick / 40)
        truths.append(actor(s + BODY.front_m + 2.4 + gap_after_m, y, speed=4.0))
    if finish:
        log[-1]["truth"]["goal_distance_m"] = 1.0
    return log, truths, "finished" if finish else "time limit"


def test_cut_in_requires_a_safe_gap_then_the_goal():
    log, truths, end = cut_in_run(2.0)
    done = sc.score_completion("cut_in", log, truths, BODY, end, 0.0)
    assert done["passed"] is True and done["min_gap_after_cut_in_m"] == pytest.approx(2.0)
    assert done["cut_in_time_s"] == pytest.approx(23 * 0.05)  # centre within half a metre of the body's side
    log, truths, end = cut_in_run(0.6)
    assert sc.score_completion("cut_in", log, truths, BODY, end, 0.0)["passed"] is False
    log, truths, end = cut_in_run(2.0, finish=False)
    assert sc.score_completion("cut_in", log, truths, BODY, end, 0.0)["passed"] is False


def test_cut_in_that_never_happens_is_incomplete():
    log = [ego_row(t, t * 0.2, 4.0) for t in range(100)]
    truths = [actor(t * 0.2 + 5.0, 3.5, speed=4.0) for t in range(100)]  # stays in the left lane
    assert sc.score_completion("cut_in", log, truths, BODY, "finished", 0.0)["passed"] is None


def braking_run(final_gap_m, lead_stopped=True):
    log = [ego_row(t, min(t * 0.2, 30.0), 4.0 if t < 150 else 0.0) for t in range(250)]
    truths = [actor(30.0 + BODY.front_m + 2.4 + final_gap_m, 0.0, speed=0.0 if lead_stopped else 3.0) for _ in range(250)]
    return log, truths


def test_braking_lead_accepts_a_sustained_hold_behind_the_stopped_lead():
    log, truths = braking_run(2.0)
    done = sc.score_completion("braking_lead", log, truths, BODY, "held at the obstacle", 5.0)
    assert done["passed"] is True and done["lead_stopped"] and done["held_s"] == 5.0
    assert done["final_gap_m"] == pytest.approx(2.0)


def test_braking_lead_hold_too_close_or_against_a_moving_lead_fails():
    log, truths = braking_run(0.4)
    assert sc.score_completion("braking_lead", log, truths, BODY, "held at the obstacle", 5.0)["passed"] is False
    log, truths = braking_run(2.0, lead_stopped=False)
    assert sc.score_completion("braking_lead", log, truths, BODY, "held at the obstacle", 5.0)["passed"] is False


def test_braking_lead_may_also_reach_the_goal():
    log = [ego_row(t, min(t * 0.3, GOAL), 4.0) for t in range(250)]
    truths = [actor(GOAL + 20.0, 0.0, speed=3.0) for _ in range(250)]  # the lead never stopped within the route
    assert sc.score_completion("braking_lead", log, truths, BODY, "finished", 0.0)["passed"] is True
    assert sc.score_completion("braking_lead", log, truths, BODY, "time limit", 0.0)["passed"] is False


# -- the live loop's hold endpoint -----------------------------------------------------------
class _Walker:
    def __init__(self, location):
        self.location = location

    def get_transform(self):
        return SimpleNamespace(location=self.location)


def test_only_a_cleared_crossing_lets_a_hold_end_the_run():
    scenario = sc.CrossingPedestrian(None, sc.ScenarioParams("crossing_pedestrian"), core_xy)
    # Lane along +x (CARLA), crossing point at (45, 0): right of the lane is +y in CARLA.
    scenario.cross_xy, scenario.right_core, scenario.lane_width_m = (45.0, 0.0), (0.0, -1.0), 3.5
    scenario.actor = _Walker(SimpleNamespace(x=45.0, y=4.0, z=1.0))
    assert scenario.hold_ends_run() is False  # not started: the walker is still on the pavement
    scenario.started = True
    assert scenario.hold_ends_run() is False
    scenario.actor.location.y = -2.0  # in the lane's left half
    assert scenario.hold_ends_run() is False
    scenario.actor.location.y = -2.8  # a metre past the left edge (1.75 + 1.0)
    assert scenario.hold_ends_run() is True
    assert sc.BrakingLeadCar(None, sc.ScenarioParams("braking_lead"), core_xy).hold_ends_run() is True


# -- the harness's gates ---------------------------------------------------------------------
def test_harness_reports_incomplete_gates_without_actor_evidence():
    import carla_adapter.mapless_drive as drive

    harness = drive.MaplessDrive.__new__(drive.MaplessDrive)
    harness.args = SimpleNamespace(scenario="cut_in", speed=4.0)
    harness.report = dict(checks=[], end="time limit", collisions=[])
    harness.body, harness.turn_site, harness.actor = BODY, None, object()
    harness.site = dict(lane_width_m=3.5, kerb_distance_m=2.0, right_lanes=0)
    harness.lane = [[0.0, 0.0]] * (drive.LANE_BACK_M + 1)
    log = []
    for tick in range(3):
        row = ego_row(tick, tick * 0.2, 4.0)
        row |= dict(mode=drive.DriveMode.DRIVING.value, reason="", driver=dict(route_pose=[tick * 0.2, 0.0]))
        row["truth"] |= dict(s_m=tick * 0.2, lane_offset_m=0.0, corner_offset_max_m=1.05, heading_error_rad=0.0, gap_m=None)
        log.append(row)
    modes = [drive.DriveMode.DRIVING] * 3
    harness.score(log, [0.001] * 3, modes, 0, [None, None, None])
    by_name = {c["check"]: c for c in harness.report["checks"]}
    assert by_name["separation"]["status"] == "INCOMPLETE" and by_name["separation"]["ticks"] == 0
    assert by_name["completion"]["status"] == "INCOMPLETE"
    assert by_name["completion"]["required"] == sc.COMPLETION_REQUIRED["cut_in"]
    assert "passed" not in by_name["completion"]
