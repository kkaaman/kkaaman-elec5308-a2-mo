"""Scripted traffic lights for the showcase and their evaluation (evaluator side, 2 Nov).

The car reads lights from its front camera only (``mapless.traffic_light``); this module
is the simulator side. Modes (``--lights``):

- ``unchanged``: CARLA's own cycle, as every run before this one (the default); entries
  are still judged.
- ``green``: every light frozen green: a run that does not test lights says so.
- ``scripted``: every light frozen green, and at each route junction with lights the ones
  governing the car's approach lane turn red when the car comes within ``red_at_m`` of
  the junction entry; they turn green once the car has stood within ``stop_zone_m`` of
  the entry for ``hold_s``, or after ``max_red_s`` whatever the car did. A stop on red
  and a start on green are then exercised on every such junction, every run.

Each tick the car's light (``vehicle.get_traffic_light_state``) is logged; it reads Green
outside a light's trigger volume, which ends a few metres before the stop line (showcase
trial, 2 Nov: Red from rear-axle s 145.4 to 151.1 at 468, then Green with the light still
red), so each junction is judged by its governing light's own state
(``TrafficLight.get_state``) against the stop line at the junction's entry: the painted
line at 468 is at the junction boundary (s 156.4-156.6 from the camera; the chain's entry
156.2), and CARLA's ``get_stop_waypoints`` gave s 149.4, the middle of the trigger volume
with nothing painted there (showcase_t2, 2 Nov), so that is recorded, not judged. The
car's front past the line while the light is red is a failure
(``red_line``), as is an entry on red; an entry on yellow is a report. Each encounter
records where the light went red, where the car first decided red, where it stood, and
whether it moved off only after green. Positions are the car's
front along the route (``s_m`` arguments). With a light classifier on the car, each fresh
reading within ``read_within_m`` of a lit junction's entry is counted against the truth
(``readings``: "truth->read" counts), the classifier's live score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

LIGHT_MODES = ("unchanged", "green", "scripted")


@dataclass
class JunctionLights:
    entry_s_m: float
    junction_id: int
    lights: list  # the carla.TrafficLight actors governing the approach lane
    phase: str = "waiting"  # waiting -> red -> green
    red_since_s: float | None = None
    stood_s: float = 0.0
    entered_state: str | None = None
    stopped_on_red: bool = False
    readings: dict = field(default_factory=dict)  # "truth->read" -> fresh readings on the approach
    carla_stop_s_m: float | None = None  # CARLA's stop waypoint along the route (recorded: the trigger volume's middle)
    red_at_front_s_m: float | None = None  # the car's front when the light turned red
    decided_red_at_front_s_m: float | None = None  # ... when the car first decided red
    stood_front_s_m: float | None = None  # ... standing on red (the furthest)
    past_line_on_red_m: float = 0.0  # the furthest the front went past the line while red
    moved_on_red: bool = False  # moved off from standing while still red
    _standing: bool = False


@dataclass
class LightScript:
    mode: str = "unchanged"
    red_at_m: float = 35.0
    stop_zone_m: float = 12.0
    hold_s: float = 4.0
    max_red_s: float = 20.0
    read_within_m: float = 30.0
    junctions: list = field(default_factory=list)
    log: list = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in LIGHT_MODES:
            raise ValueError(f"lights mode must be one of {LIGHT_MODES}")

    def setup(self, carla, world, carla_map, site):
        """Freeze the lights green (green and scripted modes), and find each route
        junction's lights for the approach lane (every mode: entries are judged in all)."""
        if self.mode != "unchanged":
            for light in world.get_actors().filter("traffic.traffic_light"):
                light.set_state(carla.TrafficLightState.Green)
                light.freeze(True)
        finder = getattr(world, "get_traffic_lights_from_waypoint", None)
        if site is None or finder is None:
            return
        for entry_s, _exit_s, junction_id in site.junctions:
            x, y, _ = site.at(max(0.0, entry_s - 6.0))
            wp = carla_map.get_waypoint(carla.Location(x=x, y=-y, z=0.0), project_to_road=True)
            governing = list(finder(wp, 30.0)) if wp is not None else []
            if governing:
                self.junctions.append(JunctionLights(entry_s, junction_id, governing, carla_stop_s_m=stop_line_s(governing, site, entry_s)))

    def step(self, carla, s_m, speed_mps, time_s, dt_s):
        """Advance the script with the car's truth (route s, speed)."""
        if self.mode != "scripted":
            return
        for j in self.junctions:
            to_entry = j.entry_s_m - s_m
            if j.phase == "waiting" and 0.0 < to_entry <= self.red_at_m:
                for light in j.lights:
                    light.set_state(carla.TrafficLightState.Red)
                j.phase, j.red_since_s = "red", time_s
            elif j.phase == "red":
                if 0.0 <= to_entry <= self.stop_zone_m and speed_mps < 0.1:
                    j.stood_s += dt_s
                    j.stopped_on_red = True
                if j.stood_s >= self.hold_s or time_s - j.red_since_s >= self.max_red_s or to_entry < 0.0:
                    for light in j.lights:
                        light.set_state(carla.TrafficLightState.Green)
                    j.phase = "green"

    def observe(self, vehicle, s_m, previous_s_m, time_s, reading=None, decision=None, speed_mps=None):
        """Log the car's light, judge each lit junction by its own light (``s_m`` is the
        car's front along the route), count a fresh ``reading`` (the car's, or None) on an
        approach, and record the encounter (``decision``: the car's light decision this
        tick). Returns the car's light state."""
        state = _state(vehicle.get_traffic_light_state())
        self.log.append((round(time_s, 2), round(s_m, 2), state, reading))
        for j in self.junctions:
            light = _state(j.lights[0].get_state())
            if j.entered_state is not None:
                continue  # entered on an earlier tick: the encounter is over
            if previous_s_m < j.entry_s_m <= s_m:
                j.entered_state = light
            ahead = j.entry_s_m - s_m
            if reading is not None and 0.0 < ahead <= self.read_within_m:
                key = f"{light.lower()}->{reading}"
                j.readings[key] = j.readings.get(key, 0) + 1
            red = light == "Red"
            if red and j.red_at_front_s_m is None:
                j.red_at_front_s_m = round(s_m, 2)
            if decision == "red" and j.red_at_front_s_m is not None and j.decided_red_at_front_s_m is None:
                j.decided_red_at_front_s_m = round(s_m, 2)
            standing = speed_mps is not None and speed_mps < 0.05
            if red and standing:
                j.stood_front_s_m = round(s_m, 2) if j.stood_front_s_m is None else max(j.stood_front_s_m, round(s_m, 2))
            line = j.entry_s_m  # the painted line, at the junction's boundary
            # Moving off on red counts from a stop at the light (within stop_zone_m of the
            # line), not from the start of the run with the light already red ahead.
            if red and j._standing and not standing:
                j.moved_on_red = True
            j._standing = standing and red and line - s_m <= self.stop_zone_m
            if red:
                j.past_line_on_red_m = max(j.past_line_on_red_m, round(s_m - line, 2))
        return state

    def report(self):
        """The ``red_light`` check's values: entries per junction, failures, exercise."""
        entries = [
            dict(
                junction_id=j.junction_id, entry_s_m=round(j.entry_s_m, 1), stop_line_s_m=round(j.entry_s_m, 1), carla_stop_waypoint_s_m=j.carla_stop_s_m, state_at_entry=j.entered_state, stopped_on_red=j.stopped_on_red,
                stood_s=round(j.stood_s, 2), red_at_front_s_m=j.red_at_front_s_m, decided_red_at_front_s_m=j.decided_red_at_front_s_m, stood_front_s_m=j.stood_front_s_m,
                stood_short_of_line_m=None if j.stood_front_s_m is None else round(j.entry_s_m - j.stood_front_s_m, 2),
                past_line_on_red_m=j.past_line_on_red_m if j.red_at_front_s_m is not None else None, moved_on_red=j.moved_on_red, readings=dict(sorted(j.readings.items())),
            )
            for j in self.junctions
        ]
        red_entries = [e for e in entries if e["state_at_entry"] == "Red"]
        line_crossings = [e for e in entries if (e["past_line_on_red_m"] or 0.0) > 0.0 or e["moved_on_red"]]
        exercised = bool(entries) and all(e["stopped_on_red"] for e in entries)
        return dict(mode=self.mode, junctions=entries, red_entries=len(red_entries), red_line=len(line_crossings), exercised=exercised)


def _state(value):
    return str(value).split(".")[-1]


def stop_line_s(lights, site, entry_s, lane_half_m=1.75, window_m=15.0):
    """The governing lights' stop line along the route: their stop waypoints
    (``TrafficLight.get_stop_waypoints``) projected onto the chain, the one in the car's
    lane (within ``lane_half_m``) and within ``window_m`` of the junction entry; None when
    the API or such a waypoint is missing."""
    import numpy as np  # noqa: PLC0415

    best = None
    for light in lights:
        getter = getattr(light, "get_stop_waypoints", None)
        if getter is None:
            continue
        for wp in getter():
            loc = wp.transform.location
            p = np.array([loc.x, -loc.y])
            d = np.hypot(*(site.points_xy - p).T)
            i = int(np.argmin(d))
            s_line = float(site.arcs[i])
            if d[i] <= lane_half_m and abs(s_line - entry_s) <= window_m and (best is None or abs(s_line - entry_s) < abs(best - entry_s)):
                best = s_line
    return None if best is None else round(best, 2)
