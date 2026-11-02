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
