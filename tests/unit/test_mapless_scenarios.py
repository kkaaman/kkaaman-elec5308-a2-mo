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


def test_crossing_pedestrian_yield_measured_against_the_speed_before_the_walker_entered():
    """Live M04 B6 (26 Oct): the car slowed from 2 m/s to 0.78 m/s before the walker entered
    the path; the gate took 0.78 as the entry speed and wanted half of it again. The
    reference is now the fastest speed in the 3 s before the walker entered."""
    log, truths, end = crossing_run(1.5)  # slows from 4 m/s to 1.5 m/s as the walker steps off
    done = sc.score_completion("crossing_pedestrian", log, truths, BODY, end, 0.0)
    assert done["entry_speed_mps"] == pytest.approx(4.0) and done["yielded"] and done["passed"] is True


def test_lead_tracking_reports_speed_error_brake_delay_and_false_stops():
    """G3's radar comparison needs, from every braking-lead run: the lead's speed error,
    how long the stack took to see it brake, and stops while the lead still moved."""
    log, truths = [], []
    ego_s, lead_s, lead_v = 0.0, 40.0, 2.0
    for tick in range(300):
        t = tick * 0.05
        if t >= 10.0:
            lead_v = max(0.0, lead_v - 3.0 * 0.05)  # brakes at 10 s
        lead_s += lead_v * 0.05
        ego_v = 0.0 if 3.0 <= t < 4.0 else 3.0  # a one-second false stop at 3 s
        ego_s += ego_v * 0.05
        row = ego_row(tick, ego_s, ego_v)
        row["mode"] = "driving"
        truths.append(actor(lead_s + 2.4, 0.0, speed=lead_v))
        seen_v = truths[max(0, tick - 6)]["speed_mps"] + 0.1  # 0.3 s late, 0.1 m/s high
        gap = lead_s - ego_s - BODY.front_m
        row["driver"] = dict(lead=dict(track_id=1, gap_m=round(gap - 0.5, 2), speed_mps=seen_v, support=["lidar", "radar"]) if tick >= 20 else None)
        log.append(row)
    out = sc.score_lead_tracking(log, truths, BODY)
    assert out["first_lead_s"] == pytest.approx(1.0)
    assert out["brake_s"] == pytest.approx(10.0, abs=0.01)
    assert out["brake_delay_s"] == pytest.approx(0.5, abs=0.01)  # 0.3 s late, then 0.5 m/s below cruise
    assert out["stopped_before_lead_s"] == pytest.approx(1.0, abs=0.06)
    assert out["gap_error_median_m"] == pytest.approx(-0.5, abs=0.05)
    assert out["support"] == {"lidar+radar": 280} and out["speed_error_rmse_mps"] < 0.5


# -- third review (e93154f): actor matching and driving quality ------------------------------
def following_run(false_lead=range(100, 141), missed=range(160, 170), switch_at=200, fragments=range(220, 230)):
    """Ego at 3 m/s 15 m behind a lead at 3 m/s; the stack's tracks around the car as logged.
    ``false_lead``: ticks whose lead was a roadside structure (5 m right, standing);
    ``missed``: ticks with no track on the actor; ``switch_at``: the actor's track id changes
    from 1 to 2; ``fragments``: a second track on the actor's rear."""
    log, truths = [], []
    for tick in range(300):
        ego_s, lead_s = 3.0 * 0.05 * tick, 15.0 + 3.0 * 0.05 * tick
        row = ego_row(tick, ego_s, 3.0)
        row["mode"] = "driving"
        truths.append(actor(lead_s, 0.0, speed=3.0))
        rel = lead_s - 2.0 - ego_s  # the track sits on the actor's rear face
        tid = 1 if tick < switch_at else 2
        near = [] if tick in missed else [[tid, rel, 0.0, 3.0, 0.0, 4.4, 1.9]]
        if tick in fragments:
            near.append([9, rel + 0.6, 0.4, 2.0, 0.5, 0.4, 0.4])
        near.append([50, 8.0, -5.0, 0.0, 0.0, 0.5, 0.5])  # the structure
        if tick in false_lead:
            lead = dict(track_id=50, gap_m=4.0, speed_mps=0.0, x_m=8.0, y_m=-5.0, support=["lidar"])
        elif tick in missed:
            lead = None
        else:
            lead = dict(track_id=tid, gap_m=round(rel - BODY.front_m, 2), speed_mps=3.05, x_m=rel, y_m=0.0, support=["lidar"])
        row["driver"] = dict(lead=lead, tracks_near=near)
        log.append(row)
    return log, truths


def test_actor_tracking_counts_misses_fragments_switches_and_false_leads():
    """Third review, F1: the lead's errors were scored against the actor without asking
    whether the lead was the actor. Tracks are matched to the actor's true box on the
    evaluator's side, from the tracks the stack logs."""
    log, truths = following_run()
    out = sc.score_actor_tracking(log, truths, BODY)
    assert out["available"] and out["in_view_ticks"] == 300
    assert out["missed_ticks"] == 10 and out["longest_missed_s"] == pytest.approx(0.5)
    assert out["fragmented_ticks"] == 10
    assert out["track_ids"] == [1, 2] and out["id_switches"] == 1
    assert out["false_lead_ticks"] == 41 and out["lead_is_actor_ticks"] == 300 - 41 - 10
    assert out["actor_in_lane_not_lead_ticks"] == 41 + 10


def test_lead_tracking_scores_only_ticks_whose_lead_was_the_actor():
    log, truths = following_run()
    out = sc.score_lead_tracking(log, truths, BODY)
    assert out["unmatched_lead_ticks"] == 41 and out["ticks"] == 300 - 41 - 10
    assert out["speed_error_rmse_mps"] == pytest.approx(0.05, abs=1e-6)  # the structure's 0 m/s is not the actor's error
    assert out["matching"].startswith("lead matched")
    # A log from before the lead's position was recorded is scored on every lead tick, and says so.
    for row in log:
        if row["driver"]["lead"]:
            row["driver"]["lead"].pop("x_m")
    assert sc.score_lead_tracking(log, truths, BODY)["matching"].startswith("unavailable")
    assert sc.score_actor_tracking([dict(r, driver={"lead": None}) for r in log], truths, BODY)["available"] is False


def test_behaviour_reports_slowing_before_the_lead_brakes_that_was_not_the_lead():
    """Third review, F4: the braking-lead gate passed a run whose car slowed before the
    lead braked. A cap from a track that was not the actor, with the gap ample, is the
    unnecessary restriction, the time it cost and its cause."""
    log, truths = [], []
    ego_s, lead_s, lead_v = 0.0, 25.0, 3.0
    for tick in range(300):
        t = tick * 0.05
        if t >= 10.0:
            lead_v = max(0.0, lead_v - 3.0 * 0.05)
        lead_s += lead_v * 0.05
        phantom = 3.0 <= t < 4.0
        ego_v = 1.0 if phantom else min(3.0, lead_v + 0.2)
        ego_s += ego_v * 0.05
        row = ego_row(tick, ego_s, ego_v)
        row["mode"] = "driving"
        row["command"] = [0.0, 0.0, 0.3 if phantom else 0.0]
        truths.append(actor(lead_s, 0.0, speed=lead_v))
        cause = dict(track_id=7, x_m=9.0, y_m=-4.0)
        row["driver"] = dict(
            speed_cap_mps=1.0 if phantom else None,
            speed_cap_reason="predicted conflict on the path" if 2.95 <= t < 3.95 else "",
            cap_track=cause if 2.95 <= t < 3.95 else None,
            lead=None,
        )
        log.append(row)
    out = sc.score_behaviour("braking_lead", log, truths, BODY, cruise_mps=3.0)
    assert out["lead_brake_s"] == pytest.approx(10.0, abs=0.06)
    assert out["pre_brake_restricted_other_s"] == pytest.approx(1.0, abs=0.06)
    assert out["pre_brake_restricted_by_actor_s"] == 0.0
    assert out["pre_brake_other_causes_s"] == {"cap: predicted conflict on the path": pytest.approx(1.0, abs=0.06)}
    assert out["pre_brake_time_lost_other_s"] == pytest.approx(1.0 * (1 - 1.0 / 3.0), abs=0.05)
    assert out["pre_brake_unjustified_slow_s"] == pytest.approx(1.0, abs=0.06)
    assert out["brake_ticks"] == 20 and out["brake_ticks_actor_share"] == 0.0


def test_behaviour_counts_a_second_near_stop_after_the_walker_has_cleared():
    """Third review, F4: the pedestrian gate passed a run whose car nearly stopped again after
    the walker had cleared and it had resumed."""
    log, truths, end = crossing_run(0.0)
    for tick, row in enumerate(log):
        row["mode"] = "driving"
        late = 180 <= tick < 200
        row["driver"] = dict(speed_cap_mps=0.2 if late else None, speed_cap_reason="obstacle", stop_reason="")
        if late:
            row["truth"]["speed_mps"] = 0.2
    out = sc.score_behaviour("crossing_pedestrian", log, truths, BODY, cruise_mps=4.0)
    assert out["post_clear_near_stops"] == 1
    assert out["post_clear_restricted_other_s"] == pytest.approx(1.0, abs=0.06)
    assert 0.0 < out["post_clear_progress_mps"] < 4.0


def test_behaviour_times_the_response_to_a_cut_in_and_the_least_time_to_collision():
    log, truths = [], []
    ego_s = 0.0
    for tick in range(200):
        t = tick * 0.05
        y = 3.5 if t < 4.0 else max(0.0, 3.5 - 3.5 * (t - 4.0) / 2.0)  # into the lane over 2 s from 4 s
        lead_s = 20.0 + 2.0 * t
        braking = t >= 4.5
        ego_v = 4.0 if not braking else max(2.0, 4.0 - 2.0 * (t - 4.5))
        ego_s += ego_v * 0.05
        row = ego_row(tick, ego_s, ego_v)
        row["mode"] = "driving"
        row["command"] = [0.0, 0.0, 0.4 if braking else 0.0]
        truths.append(actor(lead_s, y, speed=2.0))
        rel = lead_s - ego_s
        row["driver"] = dict(speed_cap_mps=2.0 if braking else None, speed_cap_reason="following the lead", lead=dict(track_id=3, gap_m=rel - 6.0, speed_mps=2.0, x_m=rel - 2.0, y_m=y) if t >= 4.4 else None)
        log.append(row)
    out = sc.score_behaviour("cut_in", log, truths, BODY, cruise_mps=4.0)
    cut = out["cut_in_s"]
    assert 4.0 < cut < 6.0
    assert out["first_brake_from_cut_in_s"] == pytest.approx(4.5 - cut, abs=0.06)
    assert out["first_actor_restriction_from_cut_in_s"] == pytest.approx(4.5 - cut, abs=0.06)
    assert out["min_ttc_s"] is not None and out["min_ttc_s"] > 0


def test_behaviour_reads_old_logs_and_leaves_the_goal_stop_out():
    """On a log without the lead's position (before the third review), following the lead
    is the actor's when the actor is ahead in the band; below a cruise slower than the
    lead, the car is not "unjustifiably" slow; and the stop at the goal is no near stop."""
    log, truths = following_run(false_lead=(), missed=(), fragments=())
    for row in log:
        row["driver"]["lead"].pop("x_m")
        row["driver"].update(speed_cap_mps=2.0, speed_cap_reason="following the lead")
    out = sc.score_behaviour("cut_in", log, truths, BODY, cruise_mps=3.0)
    assert out["restricted_s"] > 14.0 and out["restricted_other_s"] == 0.0
    log, truths = [], []
    ego_s, lead_s, lead_v = 0.0, 30.0, 3.0
    for tick in range(300):
        if tick * 0.05 >= 10.0:
            lead_v = max(0.0, lead_v - 3.0 * 0.05)
        lead_s += lead_v * 0.05
        ego_s += 2.0 * 0.05
        row = ego_row(tick, ego_s, 2.0)
        row["mode"], row["driver"] = "driving", {}
        truths.append(actor(lead_s, 0.0, speed=lead_v))
        log.append(row)
    assert sc.score_behaviour("braking_lead", log, truths, BODY, cruise_mps=2.2)["pre_brake_unjustified_slow_s"] == 0.0
    assert sc.score_behaviour("braking_lead", log, truths, BODY, cruise_mps=6.0)["pre_brake_unjustified_slow_s"] > 5.0
    log, truths, end = crossing_run(0.0)
    for row in log:
        row["mode"], row["driver"] = "driving", {}
    for row in log[-10:]:
        row["truth"].update(speed_mps=0.1, goal_distance_m=0.5)
    assert sc.score_behaviour("crossing_pedestrian", log, truths, BODY, cruise_mps=4.0)["post_clear_near_stops"] == 0


def test_f5_04_lead_speed_nees_shows_an_overconfident_filter():
    """Fifth review, F5-04: is the lead's speed as certain as the filter says? Scored on the
    evaluator's side against the truth, by the sensors behind the lead: an error of 0.05
    m/s with a 0.05 m/s sigma is consistent (NEES 1), the same error claimed at 0.01 m/s is
    overconfident (25)."""
    log, truths = following_run(false_lead=(), missed=(), fragments=())
    for k, row in enumerate(log):
        lead = row["driver"]["lead"]
        lead.update(speed_sigma_mps=0.05 if k < 150 else 0.01, lidar_nis=2.1)
        if k >= 150:
            lead["support"] = ["lidar", "radar"]
    out = sc.score_lead_tracking(log, truths, BODY)
    assert out["speed_nees"]["lidar"]["mean"] == pytest.approx(1.0, abs=0.01) and out["speed_nees"]["lidar"]["within_95"] == 1.0
    assert out["speed_nees"]["lidar+radar"]["mean"] == pytest.approx(25.0, abs=0.1) and out["speed_nees"]["lidar+radar"]["within_95"] == 0.0
    assert out["lidar_nis_mean"] == pytest.approx(2.1)


def crossing_car_run(yield_speed, cross_to=8.0):
    """Ego cruises at 4 m/s; a car 30 m ahead, 6 m right, crosses right-to-left at 2 m/s
    over ticks 20 on (fifth review: a driveway exit, M07), its 4.2 m length across the lane."""
    log, truths = [], []
    s, speed = 0.0, 4.0
    for tick in range(320):
        y = min(cross_to, -6.0 + 2.0 * 0.05 * max(0, tick - 20))
        in_way = -6.0 + 2.1 < y + 2.1 and y - 2.1 < 3.0
        if in_way and 30.0 - s > 6.0:
            speed = yield_speed
        elif not in_way and tick > 20:
            speed = 4.0
        s += speed * 0.05
        log.append(ego_row(tick, s, speed))
        truths.append(actor(30.0, y, yaw_rad=math.pi / 2, extent=(2.1, 0.9), speed=2.0 if y < cross_to else 0.0))
    log[-1]["truth"]["goal_distance_m"] = 0.5
    return log, truths


def test_f5_crossing_vehicle_is_scored_on_its_whole_body_clearing_the_path():
    """The crossing car (M07) is exercised once it enters the band within 30 m, the car
    must yield while it is in the path, and the path is clear only when its whole body (not
    its centre) is past the band."""
    log, truths = crossing_car_run(0.3)
    ex = sc.score_exercised("crossing_vehicle", log, truths, BODY, 4.0)
    assert ex["exercised"] is True
    done = sc.score_completion("crossing_vehicle", log, truths, BODY, "finished", 0.0)
    assert done["yielded"] and done["walker_cleared"] and done["passed"] is True
    # Cleared by its body: the centre past the band by only 1 m is not clear yet.
    k = next(i for i, a in enumerate(truths) if a["centre_xy"][1] > BODY.half_width_m + 1.0)
    assert not sc._cleared_left(log[k]["truth"], truths[k], BODY, vehicle=True)
    assert sc._cleared_left(log[k]["truth"], truths[k], BODY, vehicle=False)
    # Not yielding to it fails.
    log, truths = crossing_car_run(4.0)
    assert sc.score_completion("crossing_vehicle", log, truths, BODY, "finished", 0.0)["passed"] is False
