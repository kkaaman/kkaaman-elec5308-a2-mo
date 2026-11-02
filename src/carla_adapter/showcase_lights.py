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

Each tick the car's governing light (``vehicle.get_traffic_light_state``, evaluator
truth) is logged; at each junction entry the state is recorded, and an entry on red is a
failure (``red_light`` check), an entry on yellow a report.
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


@dataclass
class LightScript:
    mode: str = "unchanged"
    red_at_m: float = 35.0
    stop_zone_m: float = 12.0
    hold_s: float = 4.0
    max_red_s: float = 20.0
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
        if site is None:
            return
        for entry_s, _exit_s, junction_id in site.junctions:
            x, y, _ = site.at(max(0.0, entry_s - 6.0))
            wp = carla_map.get_waypoint(carla.Location(x=x, y=-y, z=0.0), project_to_road=True)
            governing = list(world.get_traffic_lights_from_waypoint(wp, 30.0)) if wp is not None else []
            if governing:
                self.junctions.append(JunctionLights(entry_s, junction_id, governing))

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

    def observe(self, vehicle, s_m, previous_s_m, time_s):
        """Log the governing light and record its state at each junction entry crossed."""
        state = str(vehicle.get_traffic_light_state()).split(".")[-1]
        self.log.append((round(time_s, 2), round(s_m, 2), state))
        for j in self.junctions:
            if j.entered_state is None and previous_s_m < j.entry_s_m <= s_m:
                j.entered_state = state

    def report(self):
        """The ``red_light`` check's values: entries per junction, failures, exercise."""
        entries = [dict(junction_id=j.junction_id, entry_s_m=round(j.entry_s_m, 1), state_at_entry=j.entered_state, stopped_on_red=j.stopped_on_red, stood_s=round(j.stood_s, 2)) for j in self.junctions]
        red_entries = [e for e in entries if e["state_at_entry"] == "Red"]
        exercised = bool(entries) and all(e["stopped_on_red"] for e in entries)
        return dict(mode=self.mode, junctions=entries, red_entries=len(red_entries), exercised=exercised)
