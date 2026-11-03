"""The scripted showcase lights and the red_light check (evaluator side, 2 Nov)."""

from types import SimpleNamespace

from carla_adapter.mapless_drive import red_light_check
from carla_adapter.showcase_lights import JunctionLights, LightScript

CARLA = SimpleNamespace(TrafficLightState=SimpleNamespace(Red="Red", Yellow="Yellow", Green="Green"))


class Light:
    def __init__(self):
        self.state = "Green"

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state


def drive(script, profile, decide_red_at=None):
    """Front positions and speeds per tick; the car's own light state is Green (it is
    outside the trigger volume), the junction's light is the script's."""
    vehicle = SimpleNamespace(get_traffic_light_state=lambda: "Green")
    previous = profile[0][0]
    for k, (s, v) in enumerate(profile):
        decision = "red" if decide_red_at is not None and s >= decide_red_at and script.junctions[0].lights[0].state == "Red" else None
        script.observe(vehicle, s, previous, 0.05 * k, None, decision, v)
        script.step(CARLA, s, v, 0.05 * k, 0.05)
        previous = s


def test_a_stop_short_of_the_line_and_a_start_on_green_pass_and_a_run_fails():
    good = LightScript("scripted", hold_s=1.0)
    good.junctions.append(JunctionLights(100.0, 468, [Light()]))
    approach = [(60.0 + 0.2 * k, 4.0) for k in range(190)]  # to 97.8
    stand = [(97.8, 0.0)] * 30  # 1.5 s at the line
    leave = [(97.8 + 0.2 * k, 3.0) for k in range(1, 40)]
    drive(good, approach + stand + leave, decide_red_at=70.0)
    check = red_light_check(good, True)
    entry = check["junctions"][0]
    assert check["status"] == "PASS" and check["red_line"] == 0 and entry["state_at_entry"] == "Green"
    assert entry["red_at_front_s_m"] == 65.2 and entry["decided_red_at_front_s_m"] == 70.0 and entry["stood_short_of_line_m"] == 2.2
    bad = LightScript("scripted")
    bad.junctions.append(JunctionLights(100.0, 468, [Light()]))
    drive(bad, [(60.0 + 0.2 * k, 4.0) for k in range(300)])
    check = red_light_check(bad, False)
    assert check["status"] == "FAIL" and check["red_entries"] == 1 and check["red_line"] == 0  # entered on red (judged at the line: the entry)


def test_a_lit_junction_not_reached_is_incomplete_only_when_lights_are_tested():
    for mode, classifier, status in (("unchanged", False, "REPORT"), ("scripted", False, "INCOMPLETE"), ("unchanged", True, "INCOMPLETE")):
        script = LightScript(mode)
        script.junctions.append(JunctionLights(156.2, 468, [Light()]))
        drive(script, [(100.0, 0.0)] * 5)  # held for good short of it (S4)
        assert red_light_check(script, classifier)["status"] == status
    assert red_light_check(LightScript("unchanged"), False) is None  # no lit junction, no classifier: no check


def test_creep_at_the_light_is_not_moving_off_on_red_but_half_a_metre_is():
    """showcase_c2 (PC 2): the car stood 1.3 m short of the line and crept at 0.05-0.06 m/s
    without changing place; that was read as moving off on red. Only half a metre past
    where it first stood counts."""
    for creep, moved in ((0.02, False), (0.6, True)):
        script = LightScript("scripted", hold_s=10.0)
        script.junctions.append(JunctionLights(100.0, 468, [Light()]))
        approach = [(70.0 + 0.2 * k, 3.0) for k in range(140)]  # to 97.8, red from 65
        rocking = [(97.8 + (creep if k % 2 else 0.0), 0.06 if k % 2 else 0.0) for k in range(40)]
        drive(script, approach + rocking)
        assert script.report()["junctions"][0]["moved_on_red"] is moved


def test_a_dip_to_rest_while_drawing_up_to_the_line_is_not_a_stand():
    """final/heldout clear_5317 (PC 3): drawing up to the line the car touched rest for a
    tick 3.2 m short, rolled 1 m on to its hold point and stood there 4 s, never past the
    paint; the tick at rest was taken as the stand and the roll as moving off on red. A
    stand is stand_s still; a move after a real stand still fails."""
    script = LightScript("scripted", hold_s=4.0)
    script.junctions.append(JunctionLights(100.0, 468, [Light()]))
    approach = [(70.0 + 0.2 * k, 3.0) for k in range(134)]  # to 96.6, red from 65
    dip = [(96.8, 0.03)] + [(96.8 + 0.05 * k, 0.6) for k in range(1, 21)]  # one tick at rest, then 1 m on
    stand = [(97.8, 0.0)] * 100
    leave = [(97.8 + 0.2 * k, 3.0) for k in range(1, 40)]
    drive(script, approach + dip + stand + leave)
    entry = script.report()["junctions"][0]
    assert entry["moved_on_red"] is False and entry["stood_front_s_m"] == 97.8 and entry["state_at_entry"] == "Green"
    script = LightScript("scripted", hold_s=10.0)
    script.junctions.append(JunctionLights(100.0, 468, [Light()]))
    drive(script, approach + [(96.8, 0.0)] * 20 + [(96.8 + 0.05 * k, 0.6) for k in range(1, 21)] + [(97.8, 0.0)] * 20)
    assert script.report()["junctions"][0]["moved_on_red"] is True  # stood 1 s, then 1 m on, on red


def test_the_light_the_car_drives_towards_is_logged_as_governing():
    """``governing``: the next lit junction's own light each tick (the car's own state reads
    Green outside the trigger volume), for the S3 gate's red-light excuse."""
    script = LightScript("scripted")
    script.junctions.append(JunctionLights(100.0, 468, [Light()]))
    seen = []
    vehicle = SimpleNamespace(get_traffic_light_state=lambda: "Green")
    for k, s in enumerate((60.0, 70.0, 80.0)):
        script.observe(vehicle, s, s - 1.0, 0.05 * k, None, None, 3.0)
        script.step(CARLA, s, 3.0, 0.05 * k, 0.05)
        seen.append(script.governing)
    assert seen == ["Green", "Green", "Red"]  # red from 35 m before the entry (set by the script's step, after the tick's observe)

