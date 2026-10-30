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
        truth=dict(rear_axle_xy=[s, offset], yaw_rad=0.0, speed_mps=speed, s_m=s, goal_distance_m=GOAL - s if goal_distance is None else goal_distance),
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
        row["driver"] = dict(lead=dict(track_id=1, gap_m=round(gap - 0.5, 2), speed_mps=seen_v, speed_sigma_mps=0.2, lidar_nis=0.05, support=["lidar", "radar"]) if tick >= 20 else None)
        log.append(row)
    out = sc.score_lead_tracking(log, truths, BODY)
    # Ninth review, N9-03: consistency by phase. Cruising 0.1 m/s high against sigma 0.2:
    # NEES 0.25; braking, 0.3 s late at 3 m/s^2 (0.9 m/s): overconfident; a NIS far under 2.
    phases = out["speed_nees_by_phase"]
    assert set(phases) == {"acquired", "cruise", "braking", "stopped"}
    assert phases["cruise"]["mean"] == pytest.approx(0.25, abs=0.05) and phases["braking"]["mean"] > 3.84
    assert out["lidar_nis_verdict"] == "conservative"
    assert out["brake_achieved_mps2"] == pytest.approx(3.0, rel=0.1)  # 2 m/s to rest at 3 m/s^2
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


def test_a_cut_in_car_that_brakes_to_a_stop_may_be_held_behind():
    """Fifth review's matrix: a slow cut-in, then the car brakes (``cut_brake_after_s``).
    Holding behind it, stopped, at a safe gap completes the run; holding behind a car still
    moving does not."""
    log, truths, _ = cut_in_run(3.0, finish=False)
    for k in range(120, 200):
        truths[k]["speed_mps"] = 0.0
    done = sc.score_completion("cut_in", log, truths, BODY, "held at the obstacle", 5.0)
    assert done["passed"] is True and done["lead_stopped"] is True
    log, truths, _ = cut_in_run(3.0, finish=False)
    assert sc.score_completion("cut_in", log, truths, BODY, "held at the obstacle", 5.0)["passed"] is False


def _radar_tracks(log, truths, radar_from, lidar_from, provisional_at=None, full_at=None, cap_at=None):
    """Log, on each tick, the stack's tracks on the crossing car: radar alone from
    ``radar_from``, LiDAR and radar from ``lidar_from``; its radar threats and cap."""
    for k, (row, a) in enumerate(zip(log, truths)):
        s = row["truth"]["rear_axle_xy"][0]
        near = []
        if k >= radar_from:
            near.append([9, a["centre_xy"][0] - s, a["centre_xy"][1], 0.0, 2.0, None, None, "C", "R" if k < lidar_from else "LR"])
        threats = []
        if provisional_at is not None and k >= provisional_at and (full_at is None or k < full_at):
            threats = [[9, "provisional"]]
        elif full_at is not None and k >= full_at:
            threats = [[9, "full"]]
        capped = cap_at is not None and k >= cap_at
        row["driver"] = dict(
            tracks_near=near, radar_threats=threats, cap_track=dict(track_id=9) if capped else None,
            speed_cap_mps=1.5 if capped else None, speed_cap_reason="provisional radar threat on the path" if capped else "",
        )
        row["command"] = [0.0, 0.0, 0.3 if capped else 0.0]


def test_f6_02_the_radar_timeline_says_when_each_sensor_and_the_stack_had_the_crossing_car():
    """Sixth review, F6-02: a radar-first crossing run records when the radar, the LiDAR and
    the stack's restriction each had the car, and how long the radar had it alone."""
    log, truths = crossing_car_run(0.3)
    truths = [dict(a, speed_mps=0.0) if k <= 20 else a for k, a in enumerate(truths)]  # standing until it moves off
    _radar_tracks(log, truths, radar_from=22, lidar_from=30, provisional_at=24, full_at=28, cap_at=25)
    out = sc.score_radar_timeline(log, truths, BODY, 4.0)
    assert out["radar_first"] is True and out["radar_only_s"] == pytest.approx(8 * 0.05)
    assert out["first_radar_only_s"] == pytest.approx(22 * 0.05) and out["first_lidar_s"] == pytest.approx(30 * 0.05)
    assert out["first_provisional_threat_s"] == pytest.approx(24 * 0.05) and out["first_full_threat_s"] == pytest.approx(28 * 0.05)
    assert out["first_restriction_s"] == pytest.approx(25 * 0.05) and out["first_brake_s"] == pytest.approx(25 * 0.05)
    assert out["from_actor_moving_s"]["first_radar_only"] == pytest.approx(0.05)  # the car moved off at tick 21
    # The room left at the first restriction (F7-03): the car's front to the crossing car.
    room = 30.0 - log[25]["truth"]["rear_axle_xy"][0] - BODY.front_m - 0.9
    assert out["restriction_room_m"] == pytest.approx(room, abs=0.01)
    assert out["restriction_decel_needed_mps2"] == pytest.approx(log[25]["truth"]["speed_mps"] ** 2 / (2 * room), abs=0.01)
    # The LiDAR first: no radar-only interval.
    log, truths = crossing_car_run(0.3)
    _radar_tracks(log, truths, radar_from=30, lidar_from=30)
    out = sc.score_radar_timeline(log, truths, BODY, 4.0)
    assert out["radar_first"] is False and out["radar_only_s"] == 0.0 and out["first_radar_only_s"] is None


def test_a_braking_lead_is_exercised_only_when_a_track_had_it_as_it_braked():
    """Review of 519b8fe, F-11: the hand-over this scenario is for needs the lead in the
    stack's tracks as it brakes. A lead that braked 45 m out, beyond the LiDAR's objects,
    with no track on it is not exercised, whatever the speed trace says."""
    log = [dict(ego_row(t, t * 0.2, 4.0), driver=dict(tracks_near=[])) for t in range(100)]
    truths = [actor(t * 0.2 + 45.0, 0.0, speed=max(0.0, 6.0 - 0.1 * t)) for t in range(100)]  # brakes from the first tick
    ex = sc.score_exercised("braking_lead", log, truths, BODY, 4.0)
    assert ex["exercised"] is False and "no track" in ex["note"]
    for row, a in zip(log, truths):
        row["driver"]["tracks_near"] = [[9, a["centre_xy"][0] - row["truth"]["rear_axle_xy"][0], 0.0, a["speed_mps"], 0.0, 4.4, 1.9, "C", "L"]]
    ex = sc.score_exercised("braking_lead", log, truths, BODY, 4.0)
    assert ex["exercised"] is True


def test_f6_06_a_radar_first_variant_is_exercised_only_with_a_radar_only_interval():
    """Sixth review, F6-06: the radar-first crossing (``radar_first`` 1) is exercised only
    if the radar alone had the car for ``radar_first_min_s`` before the LiDAR; the plain
    crossing needs only the band entry, as before."""
    params = sc.ScenarioParams("crossing_vehicle", radar_first=1.0).record()
    log, truths = crossing_car_run(0.3)
    _radar_tracks(log, truths, radar_from=22, lidar_from=30, provisional_at=24)
    ex = sc.score_exercised("crossing_vehicle", log, truths, BODY, 4.0, params=params)
    assert ex["exercised"] is True and ex["variant"] == "radar_first" and ex["radar_only_s"] == pytest.approx(0.4)
    _radar_tracks(log, truths, radar_from=28, lidar_from=30, provisional_at=28)  # 0.1 s alone
    ex = sc.score_exercised("crossing_vehicle", log, truths, BODY, 4.0, params=params)
    assert ex["exercised"] is False and "radar alone" in ex["note"]
    _radar_tracks(log, truths, radar_from=22, lidar_from=30)  # alone long enough, but no threat to plan with
    ex = sc.score_exercised("crossing_vehicle", log, truths, BODY, 4.0, params=params)
    assert ex["exercised"] is False and "no radar threat" in ex["note"]
    assert sc.score_exercised("crossing_vehicle", log, truths, BODY, 4.0, params=sc.ScenarioParams("crossing_vehicle").record())["exercised"] is True


def test_f6_06_a_cut_in_that_brakes_is_exercised_only_if_it_brakes_after_the_hand_over():
    """Sixth review, F6-06: the cut-in that brakes (``cut_brake_after_s`` 0 or more) is the
    hand-over and then the braking: braking before the car became the lead, or never, is
    not that variant."""
    def run(brake_at):
        log, truths = [], []
        for tick in range(200):
            s = tick * 0.2
            row = ego_row(tick, s, 4.0)
            speed = 3.0 if brake_at is None or tick < brake_at else 1.5
            a = actor(s + 12.0, 0.0 if tick >= 50 else 3.5, speed=speed)
            capped = 40 <= tick < 50
            lead = dict(track_id=4, x_m=12.0, y_m=0.0) if tick >= 50 else None
            row["driver"] = dict(lead=lead, speed_cap_mps=1.0 if capped else None, speed_cap_reason="predicted conflict on the path" if capped else "")
            log.append(row)
            truths.append(a)
        return log, truths

    params = sc.ScenarioParams("cut_in", cut_brake_after_s=1.0).record()
    log, truths = run(brake_at=None)
    assert sc.score_exercised("cut_in", log, truths, BODY, 4.0)["exercised"] is True  # the plain cut-in
    ex = sc.score_exercised("cut_in", log, truths, BODY, 4.0, params=params)
    assert ex["exercised"] is False and ex["variant"] == "cut_brake"
    log, truths = run(brake_at=80)
    ex = sc.score_exercised("cut_in", log, truths, BODY, 4.0, params=params)
    assert ex["exercised"] is True and ex["braked_after_hand_over_s"] == pytest.approx(80 * 0.05)


def test_a_crossing_car_gone_before_the_car_gets_there_needs_no_yield():
    """Live r7 (dd9344c): a car crossing at 5 m/s 13 m ahead of a car at 2 m/s cleared the
    path in 1.25 s, the car would have needed 9 s to get there, and the run failed for want
    of a yield. A yield is needed only if the car, at the speed it was making, would reach
    the crossing car before it cleared; slowing so that it arrives after it has gone is
    giving way. Paired: holding 4 m/s into a 2 m/s crossing 30 m ahead still fails."""
    log, truths = [], []
    s = 0.0
    for tick in range(400):
        y = min(8.0, -6.0 + 6.0 * 0.05 * max(0, tick - 20))
        s += 2.0 * 0.05
        log.append(ego_row(tick, s, 2.0))
        truths.append(actor(40.0, y, yaw_rad=math.pi / 2, extent=(2.1, 0.9), speed=6.0 if y < 8.0 else 0.0))
    log[-1]["truth"]["goal_distance_m"] = 0.5
    done = sc.score_completion("crossing_vehicle", log, truths, BODY, "finished", 0.0)
    assert done["yield_needed"] is False and done["passed"] is True and not done["yielded"]
    log, truths = crossing_car_run(4.0)
    done = sc.score_completion("crossing_vehicle", log, truths, BODY, "finished", 0.0)
    assert done["yield_needed"] is True and done["gave_way"] is False and done["passed"] is False
    # Slowing to 1.2 m/s (not a yield by the walker's rule) so as to arrive once it has gone: gave way.
    log, truths = crossing_car_run(1.2)
    done = sc.score_completion("crossing_vehicle", log, truths, BODY, "finished", 0.0)
    assert done["gave_way"] is True and done["passed"] is True


def test_f7_01_the_radar_only_interval_counts_the_ticks_the_radar_actually_had_the_car():
    """Seventh review, F7-01: the radar-only time was the span from the first radar-only
    tick to the first LiDAR tick, so a radar that had the car for one tick and lost it was
    credited the whole span. Now the run of ticks with a radar-only track on the car is
    what counts (``radar_only_s`` is the longest unbroken run), and the span, the ticks
    observed and the gaps are reported beside it."""
    log, truths = crossing_car_run(0.3)
    truths = [dict(a, speed_mps=0.0) if k <= 20 else a for k, a in enumerate(truths)]
    _radar_tracks(log, truths, radar_from=22, lidar_from=30, provisional_at=22)
    for k in range(23, 30):  # the radar has it on tick 22 only
        log[k]["driver"]["tracks_near"] = []
    out = sc.score_radar_timeline(log, truths, BODY, 4.0)
    assert out["radar_first"] is True and out["radar_only_span_s"] == pytest.approx(0.4)
    assert out["radar_only_s"] == pytest.approx(0.05) and out["radar_only_observed_s"] == pytest.approx(0.05)
    assert out["radar_only_gaps"] == 1 and out["radar_only_longest_gap_s"] == pytest.approx(0.35)
    ex = sc.score_exercised("crossing_vehicle", log, truths, BODY, 4.0, params=sc.ScenarioParams("crossing_vehicle", radar_first=1.0).record())
    assert ex["exercised"] is False
    # Broken once: two runs of 3 and 4 ticks; the longest counts.
    _radar_tracks(log, truths, radar_from=22, lidar_from=30, provisional_at=22)
    log[25]["driver"]["tracks_near"] = []
    out = sc.score_radar_timeline(log, truths, BODY, 4.0)
    assert out["radar_only_s"] == pytest.approx(0.2) and out["radar_only_observed_s"] == pytest.approx(0.35) and out["radar_only_gaps"] == 1


def test_f8_04_a_radar_restriction_counts_for_the_actor_only_through_a_track_on_it():
    """Eighth review, F8-04: "radar" in the cap's reason was enough for the first
    restriction, so an unrelated radar track's cap set it, and the room and deceleration
    after it. Now the responsible track must be on the actor; other radar caps are
    reported apart as possible false interventions."""
    log, truths = crossing_car_run(0.3)
    truths = [dict(a, speed_mps=0.0) if k <= 20 else a for k, a in enumerate(truths)]
    _radar_tracks(log, truths, radar_from=22, lidar_from=30, provisional_at=24, full_at=28, cap_at=27)
    # Ticks 23-26: a radar track elsewhere (id 41, not on the car) caps the car first.
    for k in range(23, 27):
        log[k]["driver"].update(cap_track=dict(track_id=41), speed_cap_mps=1.5, speed_cap_reason="provisional radar threat on the path")
    out = sc.score_radar_timeline(log, truths, BODY, 4.0)
    assert out["first_restriction_s"] == pytest.approx(27 * 0.05)  # the car's own track's cap
    unmatched = out["unmatched_radar_restrictions"]
    assert unmatched["ticks"] == 4 and unmatched["first_s"] == pytest.approx(23 * 0.05) and unmatched["tracks"] == [41]
    # A radar cap with no responsible track logged is not the actor's either.
    log, truths = crossing_car_run(0.3)
    truths = [dict(a, speed_mps=0.0) if k <= 20 else a for k, a in enumerate(truths)]
    _radar_tracks(log, truths, radar_from=22, lidar_from=30)
    log[24]["driver"].update(cap_track=None, speed_cap_mps=1.0, speed_cap_reason="provisional radar threat on the path")
    out = sc.score_radar_timeline(log, truths, BODY, 4.0)
    assert out["first_restriction_s"] is None and out["unmatched_radar_restrictions"]["ticks"] == 1


def test_f8_05_every_scripted_actor_is_recorded_and_the_tracks_kept_apart():
    """Eighth review, F8-05: the braking lead's second car was driven but only the lead was
    recorded, so nothing could say whether a track carried from one car to the other. Each
    actor's truth now carries its role; the multi-actor score tracks each and finds two cars
    taken as one, an identity carried across, and the lead handed from one to the other."""

    class FakeCar:
        is_alive = True

        def __init__(self, x):
            self.x = x

        def get_transform(self):
            return SimpleNamespace(location=SimpleNamespace(x=self.x, y=0.0, z=0.0), rotation=SimpleNamespace(yaw=0.0))

        bounding_box = SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0), extent=SimpleNamespace(x=2.2, y=0.9))

        def get_velocity(self):
            return SimpleNamespace(x=3.0, y=0.0)

    lead = sc.BrakingLeadCar(None, sc.ScenarioParams("braking_lead"), core_xy)
    lead.actor, lead.second = FakeCar(20.0), FakeCar(30.0)
    truths = lead.truths()
    assert [a["role"] for a in truths] == ["lead", "second"] and truths[1]["centre_xy"][0] == pytest.approx(30.0)
    lead.second = None
    assert [a["role"] for a in lead.truths()] == ["lead"]
    single = sc.ScriptedActor(None, sc.ScenarioParams("stop"), core_xy)
    single.actor = FakeCar(10.0)
    assert [a["role"] for a in single.truths()] == ["primary"]

    # Two cars ahead, 10 m apart; track 5 on the lead, 6 on the second, the lead track
    # followed. Ticks 20-24: the tracker takes both as one (track 5 nearest to both).
    # From tick 40: track 6 is on the lead (an identity carried across) and leads.
    log, sets = [], []
    for k in range(60):
        row = ego_row(k, 0.0, 0.0)
        a_lead, a_second = dict(actor(20.0, 0.0), role="lead"), dict(actor(30.0, 0.0), role="second")
        on_lead = 6 if k >= 40 else 5
        near = [[on_lead, 20.0, 0.0, 0.0, 0.0, 4.4, 1.8, "C", "L"]]
        near.append([5 if 20 <= k < 25 else 6 if k < 40 else 7, 30.0 if not 20 <= k < 25 else 30.0, 0.0, 0.0, 0.0, 4.4, 1.8, "C", "L"])
        row["driver"] = dict(tracks_near=near, lead=dict(track_id=on_lead, x_m=20.0, y_m=0.0))
        log.append(row)
        sets.append([a_lead, a_second])
    out = sc.score_multi_actor_tracking(log, sets, BODY)
    assert out["available"] and out["roles"] == ["lead", "second"]
    assert out["shared_track_ticks"] == 5
    assert out["tracks_on_two_actors"] == {"5": ["lead", "second"], "6": ["lead", "second"]}
    assert out["lead_by_role"] == {"lead": 60} and out["lead_hand_overs"] == 0
    assert out["per_role"]["lead"]["id_switches"] == 1 and out["per_role"]["second"]["matched_share"] == 1.0
    # The lead handed to the second car: counted, with the time.
    for row in log[50:]:
        row["driver"]["lead"] = dict(track_id=7, x_m=30.0, y_m=0.0)
    out = sc.score_multi_actor_tracking(log, sets, BODY)
    assert out["lead_by_role"] == {"lead": 50, "second": 10} and out["lead_hand_overs"] == 1
    assert out["lead_hand_over_times"][0]["to_role"] == "second"
    assert sc.score_multi_actor_tracking(log, [[a] for a in [dict(actor(20.0, 0.0), role="lead")] * 60], BODY)["available"] is False


def test_n9_scripted_braking_follows_the_asked_deceleration():
    """PC 2, r10_lead_pair: the fixed pedal stopped the lead from 2.8 m/s in 0.15 s (about
    18 m/s^2) where 3 m/s^2 was asked. Against a plant whose brake is as strong as CARLA's
    at low speed (25 m/s^2 at full pedal), the profile control keeps the asked rate within
    15% at 1, 3 and 4 m/s^2, and holds the car once stopped."""
    for decel in (1.0, 3.0, 4.0):
        v, t, stopped_at = 6.0, 0.0, None
        while t < 10.0:
            brake = sc.braking_profile_control(v, 6.0, t, decel)
            v = max(0.0, v - 25.0 * brake * 0.05)
            t += 0.05
            if v == 0.0 and stopped_at is None:
                stopped_at = t
        assert stopped_at is not None and abs(6.0 / stopped_at - decel) <= 0.15 * decel, (decel, stopped_at)
        assert sc.braking_profile_control(0.0, 6.0, 20.0, decel) == 1.0


def test_r11_scripted_braking_sets_the_profile_velocity_and_holds_at_the_stop():
    """PC 2, r11_lead_b3 and b4 (84fb071): the pedal profile achieved 4.6 and 5.0 m/s^2 for 3
    and 4 asked, the lead's speed jumping 2.8 -> 1.0 -> 1.6 m/s in 0.2 s. A CARLA actor now
    has its velocity set along its heading to the profile's each tick, the pedals released;
    held on the brake once the profile reaches zero. A stand-in without
    ``set_target_velocity`` keeps the pedal profile."""
    from types import SimpleNamespace

    class Actor:
        def __init__(self, yaw_deg, with_velocity=True):
            self.yaw, self.v, self.controls, self.velocities = yaw_deg, SimpleNamespace(x=0.0, y=2.8, z=0.0), [], []
            if with_velocity:
                self.set_target_velocity = lambda vec: self.velocities.append((vec.x, vec.y, vec.z))

        def get_velocity(self):
            return self.v

        def get_transform(self):
            return SimpleNamespace(rotation=SimpleNamespace(yaw=self.yaw))

        def apply_control(self, control):
            self.controls.append(control)

    carla = SimpleNamespace(VehicleControl=lambda **kw: SimpleNamespace(**kw), Vector3D=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z))
    actor = Actor(90.0)
    speeds = []
    for k in range(25):
        mode = sc.scripted_braking(carla, actor, 2.8, (k + 1) * 0.05, 3.0, steer=0.1)
        if mode == "velocity":
            vx, vy, _ = actor.velocities[-1]
            speeds.append(math.hypot(vx, vy))
            assert abs(vx) < 1e-9 and vy > 0  # along the heading (yaw 90 degrees: +y)
            assert actor.controls[-1].brake == 0.0 and actor.controls[-1].throttle == 0.0 and actor.controls[-1].steer == 0.1
    assert len(speeds) == 18  # 2.8 m/s at 3 m/s^2: 0.93 s, 18 ticks before the stop
    assert all(abs((a - b) / 0.05 - 3.0) < 1e-6 for a, b in zip(speeds, speeds[1:]))  # exactly 3 m/s^2
    assert mode == "hold" and actor.controls[-1].brake == 1.0 and actor.velocities[-1] == (0.0, 0.0, 0.0)
    pedal = Actor(0.0, with_velocity=False)
    assert sc.scripted_braking(carla, pedal, 2.8, 0.05, 3.0) == "pedal"
    assert pedal.controls[-1].brake == pytest.approx(sc.braking_profile_control(2.8, 2.8, 0.05, 3.0))


def test_t10_prediction_coverage_checks_the_region_against_where_the_actor_went():
    """Tenth review, 3: does the predicted region hold where the actor went, and how much
    bigger than the actor is it? An actor driving on at 2 m/s, matched to track 4: a
    prediction that moves with it covers its centre 1 and 2 s on; one that stands still,
    and no larger than the car, misses."""
    def log_for(moving):
        log, truths = [], []
        for k in range(100):
            ax = 10.0 + 2.0 * k * 0.05
            actor = dict(centre_xy=[ax, 0.0], yaw_rad=0.0, extent_m=[2.2, 0.95], speed_mps=2.0)
            preds = []
            for h in (1.0, 2.0):
                cx = ax + (2.0 * h if moving else 0.0)
                preds.append([4, h, cx, 0.0, 2.6 + 0.2 * h, 1.2 + 0.1 * h, 0.0])
            log.append(dict(time_s=k * 0.05, truth=dict(rear_axle_xy=[0.0, 0.0], yaw_rad=0.0, speed_mps=0.0), driver=dict(tracks_near=[[4, ax, 0.0, 2.0, 0.0, 4.4, 1.9, "C", "L"]], predictions_near=preds)))
            truths.append(actor)
        return log, truths

    log, truths = log_for(True)
    out = sc.score_prediction_coverage(log, truths, BODY)
    assert out["available"]
    for h in ("1s", "2s"):
        r = out["horizons"][h]
        assert r["judged_ticks"] == 100 - int(float(h[:-1]) / 0.05) and r["covered_share"] == 1.0 and r["worst_miss_m"] is None
        assert r["area_over_actor_median"] > 1.0  # conservatism: the region against the actor's own footprint
    # The whole body, not only its centre (review of 519b8fe, F-06): at 1 s the region's
    # half width 1.3 m covers the 0.95 m half-width car; at 2 s the half length 3.0 m covers
    # the 2.2 m half length. Both whole.
    assert out["horizons"]["1s"]["body_covered_share"] == 1.0 and out["horizons"]["2s"]["body_covered_share"] == 1.0
    narrow = [dict(r, driver=dict(r["driver"], predictions_near=[p[:5] + [0.9, 0.0] for p in r["driver"]["predictions_near"]])) for r in log]
    thin = sc.score_prediction_coverage(narrow, truths, BODY)
    assert thin["horizons"]["1s"]["covered_share"] == 1.0 and thin["horizons"]["1s"]["body_covered_share"] == 0.0  # centre in, the near face out
    still = sc.score_prediction_coverage(*log_for(False), BODY)
    assert still["horizons"]["2s"]["covered_share"] == 0.0 and still["horizons"]["2s"]["worst_miss_m"] == pytest.approx(4.0 - 3.0, abs=0.01)
    assert still["horizons"]["1s"]["covered_share"] == 1.0  # 2 m on is still within a 2.8 m half length
    old = [dict(r, driver={k: v for k, v in r["driver"].items() if k != "predictions_near"}) for r in log]
    assert sc.score_prediction_coverage(old, truths, BODY)["available"] is False
