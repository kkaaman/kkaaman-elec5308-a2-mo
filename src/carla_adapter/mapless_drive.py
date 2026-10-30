"""Live G1b: the mapless car drives from its sensors and a sat-nav route only.

The car is ``mapless.driver.MaplessDriver``. It receives a ``SensorFrame`` per tick and the
route; it never sees the map, the simulator's poses or any actor. This module plays
everything else:

- **Site** (map, truth side): searched along the map's junction-free lane runs
  (topology segments), from 8 m past a junction. The rightmost driving lane, with the
  whole run either straight (``straight``, ``stop``) or one bend of 30 to 120 degrees with
  radius 12 m or more (``curve``), starting on a straight stretch. Sites where the map has
  a pavement lane straight after the driving lane come first. If none fits, the error
  gives a census of why, and the longest junction-free runs on the map.
- **Route** (sat-nav role): LANEFOLLOW points every 25 m along that lane to the goal,
  placed with the map's own geolocation.
- **Obstacle** (``stop``): a parked car in the ego lane, spawned before the car starts.
- **Turns** (``turn_left``, ``turn_right``, G1c): a rightmost approach lane that ends at a
  junction with a connecting lane leaving at about 90 degrees to the commanded side and
  an exit lane of ``--exit-m``. The route carries the entry point (the lane's last
  waypoint before the junction) with the command and the exit point 5 m into the exit
  lane, as CARLA's routes do. Truth and gates are junction-aware
  (``evaluation.junction_truth``): a permitted area of approach, connecting and exit
  lanes, each with its own width.
- **Stack** (``scripts/run_mapless_stack.py``): the same harness drives ``MaplessStack``
  (G2/G3 stages around the driver) with ``--admission A|B|C`` and the segmenter and
  detector by configuration; the log carries track counts and decision records.
- **Moving actors** (``crossing_pedestrian``, ``braking_lead``, ``cut_in``,
  ``crossing_vehicle``; M04–M07, framework): scripted by ``carla_adapter.mapless_scenarios``,
  scored on separation.
- **Scoring** (truth): lane-centre offset of the rear axle, heading error, bumper gap to
  the parked car, collisions (collision sensor, harness only), goal distance, and the
  route-pose error of the car's GNSS/compass filter.

**Gates** (``drive_report.json``):

| Check | Pass |
|---|---|
| Start-up | The car localised itself from GNSS and compass within 3 s, with no spawn pose |
| No collision | No collision event |
| Lane keeping | After the first 15 m: offset p95 ≤ 0.35 m and max ≤ 0.70 m |
| Lane footprint | Whole run, calibration and start included: no corner of the car's body crosses the lane's edges |
| Calibration | The kerb distance was measured at a standstill and accepted, within 0.3 m of the map's |
| Goal (straight, curve) | FINISHED, rear axle within 3 m of the goal |
| Obstacle stop (stop) | Standing still, bumper gap 1.5 to 5.0 m, held for ``--hold-s`` |
| Turn (turn_left, turn_right) | TURNING entered; no corner outside the permitted area; ends on the commanded exit lane; exit-leg p95 ≤ 0.35 m |
| Separation (moving actors) | Ego box to actor box (oriented polygons) never under 1 m; INCOMPLETE without actor measurements |
| Completion (moving actors) | Per scenario: walker yielded to then goal reached; cut-in gap ≥ 1 m then goal; braking lead held behind for ``--hold-s`` or goal |
| Speed | Never above the cruise speed + 0.5 m/s |
| Faults | Under 2% of driving ticks in FAULT |
| Exercised (moving actors) | The run reached the transition its scenario is for (``score_exercised``), and its variant's too (a radar-first crossing, a cut-in that brakes: ``EXERCISE_VARIANT``); NOT_EXERCISED otherwise, and the run's status is NOT_EXERCISED rather than MEASURED_PASS |

Reported, not gated: driver step time (the whole step, the control path, and which of
them governs actuation: ``--semantic``), the start-up route-pose error, the kerb bias
(where the LiDAR found the kerb against where the map has it), the segmenter's shadow
(coverage, disagreement, its right road edge against the kerb) and, for the crossing car,
the radar timeline (when the radar, the LiDAR and the stack's restriction each had it).

Outputs: ``drive_report.json``, ``drive_log.jsonl`` (one line per tick), ``route.json``,
``drive_plot.png``, every ``--snapshot-every`` ticks a top-down image of the terrain grid
and corridor in ``snapshots/`` and, until the car first moves, the raw classified sweeps
in ``sweeps/`` (``npz``: xyz, cls), so a calibration can be replayed offline.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from carla_adapter.frame_record import FrameRecorder
from carla_adapter.mapless_capture import source_commit
from carla_adapter.mapless_scenarios import SCENARIOS as ACTOR_SCENARIOS
from carla_adapter.mapless_scenarios import (
    make_oncoming,
    make_scenario,
    score_actor_tracking,
    score_behaviour,
    score_completion,
    score_exercised,
    score_lead_tracking,
    score_moving,
    score_multi_actor_tracking,
    score_prediction_coverage,
    score_radar_timeline,
)
from carla_adapter.sensor_rig import SensorRig, true_forward_speed
from carla_adapter.sensor_rig_check import rear_axle_pose
from carla_adapter.vehicle import FixedVehicleAdapter
from carla_adapter.world import SynchronousWorldSession, WorldTiming, _settings_record, ensure_map
from evaluation.braking_table import assumptions as braking_assumptions
from evaluation.junction_truth import TurnSite, score_turn, truth_sample
from evaluation.observability import observability
from mapless.corridor import CorridorConfig
from mapless.driver import DriveMode, DriverConfig, MaplessDriver
from mapless.frame import SensorState, SensorStatus
from mapless.obstacles import project_onto_polyline
from mapless.planner import PlannerConfig
from mapless.satnav import Command, RoutePoint, SatNavRoute
from mapless.sensor_suite import load_sensor_suite, parse_overrides
from mapless.stack import Admission, MaplessStack, StackConfig
from mapless.terrain import CellLabel, VehicleBody
from navigation_control.local import LocalControlConfig
from navigation_control.types import ControllerConfig

DT_S = 0.05
PARKED_CAR = "vehicle.audi.a2"
SETTLE_TICKS = 20  # brake and let the rig settle (IMU spike, spawn delay) before the car starts
LANE_BACK_M = 15  # truth lane extended this far behind the start
PASS, FAIL, INCOMPLETE, REPORT = "PASS", "FAIL", "INCOMPLETE", "REPORT"
# A check that the run never reached what it is for (fifth review, testing: PASS, FAIL or
# NOT_EXERCISED): a cut-in run whose car drove off passed every gate and showed nothing
# about the hand-over it was meant to test.
NOT_EXERCISED = "NOT_EXERCISED"


def _params_record(actor):
    """The scripted actor's parameters as recorded, or None (an actor without them)."""
    params = getattr(actor, "params", None)
    return params.record() if hasattr(params, "record") else None


def run_status(checks):
    """The run's status from its checks: MEASURED_FAIL on any failed gate; INCOMPLETE when
    a gate could not be measured (or there is none); NOT_EXERCISED when every gate passed
    but the run did not reach the transition its scenario targets; else MEASURED_PASS.
    REPORT checks are not gates."""
    gated = [c for c in checks if c["status"] not in (REPORT, NOT_EXERCISED)]
    if any(c["status"] == FAIL for c in gated):
        return "MEASURED_FAIL"
    if any(c["status"] == INCOMPLETE for c in gated) or not gated:
        return "INCOMPLETE"
    if any(c["status"] == NOT_EXERCISED for c in checks):
        return NOT_EXERCISED
    return "MEASURED_PASS"
TURN_SCENARIOS = {"turn_left": "LEFT", "turn_right": "RIGHT"}
LANE_SCENARIOS = ("straight", "curve", "stop")


# -- site and route (map side) ---------------------------------------------------------
def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def core_xy(location):
    return location.x, -location.y


def lane_heading(wp):
    return -math.radians(wp.transform.rotation.yaw)


def follow_lane(wp, length_m, step_m=1.0):
    """Waypoints every ``step_m`` along one lane (first branch), up to ``length_m`` or the next junction."""
    points = [wp]
    while (len(points) - 1) * step_m < length_m:
        nxt = points[-1].next(step_m)
        if not nxt or nxt[0].is_junction:
            break
        points.append(nxt[0])
    return points


def right_kerb(carla, wp):
    """Distance from lane centre to the kerb on the right, the lanes crossed, and the evidence.

    The kerb is where the map's Sidewalk lane starts. Maps that model no Sidewalk lane give
    the outer edge of the last lane instead (``source`` "road edge"); a Driving lane on the
    right means this is not the rightmost lane.
    """
    distance, crossed = wp.lane_width / 2, []
    right = wp.get_right_lane()
    for _ in range(6):
        if right is None:
            return distance, crossed, "road edge"
        if right.lane_type == carla.LaneType.Sidewalk:
            return distance, crossed, "sidewalk"
        if right.lane_type == carla.LaneType.Driving:
            return None, crossed + ["Driving"], "not rightmost"
        crossed.append(str(right.lane_type))
        distance += right.lane_width
        right = right.get_right_lane()
    return None, crossed, "too many side lanes"


def site_shape(points):
    headings = np.unwrap([lane_heading(p) for p in points])
    total = float(abs(headings[-1] - headings[0]))
    window = 4
    rates = np.abs(headings[window:] - headings[:-window]) / window if len(headings) > window else np.zeros(1)
    start_turn = float(abs(headings[min(12, len(headings) - 1)] - headings[0]))
    return total, float(rates.max()), start_turn


def shape_ok(scenario, points):
    total, max_rate, start_turn = site_shape(points)
    if start_turn > math.radians(5):
        return False, total, max_rate
    if scenario in ("straight", "stop"):
        return total <= math.radians(5), total, max_rate
    return math.radians(30) <= total <= math.radians(120) and max_rate <= 1 / 12, total, max_rate


def find_sites(carla, carla_map, scenario, length_m, junction_clearance_m=8.0, slide_m=10.0, max_slope_deg=3.0, max_grade_change_deg=2.0):
    """Sites on junction-free lane runs from the map topology, and a census of why others failed.

    Each topology segment between junctions is walked from ``junction_clearance_m`` past its
    start (away from the kerb corners of the junction mouth). A window of ``length_m`` is
    slid along the run in ``slide_m`` steps until one has the scenario's shape.
    """
    census = dict(segments=0, not_driving_or_junction=0, not_rightmost=0, too_short=0, wrong_shape=0, sloped=0, kerb_gaps=0, sites=0)
    longest = []  # (run length, road, lane) of rightmost runs, for the message when nothing fits
    seen = set()
    sites = []
    for start, _ in carla_map.get_topology():
        census["segments"] += 1
        if start.lane_type != carla.LaneType.Driving or start.is_junction:
            census["not_driving_or_junction"] += 1
            continue
        key = (start.road_id, start.section_id if hasattr(start, "section_id") else 0, start.lane_id)
        if key in seen:
            continue
        seen.add(key)
        kerb, crossed, source = right_kerb(carla, start)
        if kerb is None:
            census["not_rightmost"] += 1
            continue
        run = follow_lane(start, 400.0)
        usable = run[int(junction_clearance_m) :]
        longest.append((len(usable) - 1, start.road_id, start.lane_id))
        if len(usable) - 1 < length_m:
            census["too_short"] += 1
            continue
        found, shaped = None, False
        for offset in range(0, len(usable) - int(length_m), int(slide_m)):
            window = usable[offset : offset + int(length_m) + 1]
            ok, total, max_rate = shape_ok(scenario, window)
            if not ok:
                continue
            shaped = True
            # Flat enough for the planar adapter (it refuses 5 degrees of tilt) and for the
            # terrain model: on Town05's first site the rising road a few metres ahead read
            # as an obstacle moving with the car, and the run ended on a tilt refusal.
            pitches = [math.degrees(wrap(math.radians(w.transform.rotation.pitch))) for w in window]
            if max(abs(p) for p in pitches) > max_slope_deg or max(pitches) - min(pitches) > max_grade_change_deg:
                continue
            found = (offset, window, total, max_rate)
            break
        if found is None:
            census["sloped" if shaped else "wrong_shape"] += 1
            continue
        offset, window, total, max_rate = found
        kerbs = [right_kerb(carla, p)[0] for p in window[::10]]
        if sum(k is not None for k in kerbs) < 0.8 * len(kerbs):
            census["kerb_gaps"] += 1
            continue
        first_kerb, crossed, source = right_kerb(carla, window[0])
        census["sites"] += 1
        sites.append(
            dict(
                waypoints=window,
                kerb_distance_m=first_kerb,
                kerb_source=source,
                right_lanes=crossed,
                heading_change_deg=math.degrees(total),
                min_radius_m=(1 / max_rate) if max_rate > 0 else math.inf,
                lane_width_m=window[0].lane_width,
                road_id=window[0].road_id,
                lane_id=window[0].lane_id,
                junction_free_run_m=len(usable) - 1,
            )
        )
    # A mapped pavement first, then no side lanes before the kerb, then the longest runs.
    sites.sort(key=lambda site: (site["kerb_source"] != "sidewalk", len(site["right_lanes"]), -site["junction_free_run_m"]))
    longest.sort(reverse=True)
    census["longest_rightmost_runs_m"] = [dict(length_m=n, road_id=r, lane_id=lane) for n, r, lane in longest[:5]]
    return sites, census


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_route(carla_map, waypoints, goal_m, spacing_m=25.0, name="g1b"):
    stations = list(np.arange(0.0, goal_m, spacing_m)) + [goal_m]
    points = []
    for s in stations:
        wp = waypoints[min(len(waypoints) - 1, int(round(s)))]
        fix = carla_map.transform_to_geolocation(wp.transform.location)
        points.append(RoutePoint(fix.latitude, fix.longitude, Command.LANEFOLLOW))
    return SatNavRoute(name, tuple(points), provenance=dict(source="map lane centre", spacing_m=spacing_m))


def follow_junction(wp, max_m=80.0):
    """Waypoints every 1 m through a junction's connecting lane, from its first waypoint
    to the first non-junction waypoint (the exit lane's start); None if it never leaves."""
    points = [wp]
    while len(points) < max_m:
        nxt = points[-1].next(1.0)
        if not nxt:
            return None
        points.append(nxt[0])
        if not nxt[0].is_junction:
            return points
    return None


def find_turn_sites(carla, carla_map, command, approach_m, exit_m, junction_clearance_m=8.0, angle_tolerance_deg=10.0, max_kerb_mismatch_m=0.3, max_approach_bend_deg=5.0, max_slope_deg=3.0):
    """Sites for a commanded turn: a rightmost lane run ending at a junction whose connecting
    lane turns about 90 degrees to ``command``'s side into a rightmost exit lane.

    Two map-truth filters keep the first live turns inside what the LiDAR-only car can do
    (the harness may choose the site; the car never sees these numbers):
    - the exit lane's kerb distance must match the approach lane's within
      ``max_kerb_mismatch_m``: the car places the exit lane ``keep_right`` (measured on the
      approach) left of the exit kerb, so a shoulder lane turning into a plain lane would
      put it 2.6 m into the opposing lane (review of 1931070), and nothing in the car can
      tell until lane markings are read (G2);
    - the approach must be straight within ``max_approach_bend_deg`` over its length: the
      exit heading is a right angle to the approach lane's fitted direction, and a bend
      skews it;
    - for the same reason the junction must turn within ``angle_tolerance_deg`` of a right
      angle: the car keeps the right-angle prior unless the route disagrees by 35 degrees,
      and a 100 degree junction ended 1.85 m towards the opposing lane (full review of
      73ae759; was 30 degrees). Skewed junctions are not supported yet.
    """
    want = -math.pi / 2 if command == "RIGHT" else math.pi / 2
    census = dict(segments=0, no_junction_end=0, not_rightmost=0, too_short=0, approach_bent=0, sloped=0, no_matching_exit=0, exit_too_short=0, kerb_mismatch=0, sites=0)
    sites, seen = [], set()
    for start, _ in carla_map.get_topology():
        census["segments"] += 1
        if start.lane_type != carla.LaneType.Driving or start.is_junction:
            continue
        key = (start.road_id, start.section_id if hasattr(start, "section_id") else 0, start.lane_id)
        if key in seen:
            continue
        seen.add(key)
        kerb, crossed, source = right_kerb(carla, start)
        if kerb is None:
            census["not_rightmost"] += 1
            continue
        run = follow_lane(start, 400.0)
        end_next = run[-1].next(1.0)
        if not end_next or not end_next[0].is_junction:
            census["no_junction_end"] += 1
            continue
        if len(run) - 1 < approach_m + junction_clearance_m:
            census["too_short"] += 1
            continue
        approach = run[-(int(approach_m) + 1) :]
        bend, _, _ = site_shape(approach)
        if bend > math.radians(max_approach_bend_deg):
            census["approach_bent"] += 1
            continue
        entry_heading = lane_heading(approach[-1])
        match = None
        for first in end_next:
            through = follow_junction(first)
            if through is None:
                continue
            exit_wp = through[-1]
            change = wrap(lane_heading(exit_wp) - entry_heading)
            if abs(wrap(change - want)) > math.radians(angle_tolerance_deg):
                continue
            exit_kerb, _, exit_source = right_kerb(carla, exit_wp)
            if exit_kerb is None:
                continue
            if abs(exit_kerb - kerb) > max_kerb_mismatch_m:
                census["kerb_mismatch"] += 1
                continue
            exit_run = follow_lane(exit_wp, exit_m + 5.0)
            if len(exit_run) - 1 < exit_m:
                census["exit_too_short"] += 1
                continue
            match = (through, exit_run, change, exit_kerb, exit_source)
            break
        if match is None:
            census["no_matching_exit"] += 1
            continue
        through, exit_run, change, exit_kerb, exit_source = match
        # The planar adapter refuses more than 5 degrees of pitch or roll, and Town03's hills
        # and bridge gave a left-turn site that aborted the run mid-box: every waypoint of
        # the approach, junction and exit must lie within max_slope_deg.
        pitches = [abs(wrap(math.radians(w.transform.rotation.pitch))) for w in (*approach, *through, *exit_run)]
        if max(pitches) > math.radians(max_slope_deg):
            census["sloped"] += 1
            continue
        census["sites"] += 1
        sites.append(
            dict(
                approach=approach,
                junction=through,
                exit=exit_run,
                kerb_distance_m=kerb,
                kerb_source=source,
                exit_kerb_distance_m=exit_kerb,
                exit_kerb_source=exit_source,
                right_lanes=crossed,
                heading_change_deg=math.degrees(change),
                lane_width_m=approach[-1].lane_width,
                junction_width_m=float(np.median([w.lane_width for w in through])),
                exit_width_m=exit_run[0].lane_width,
                approach_bend_deg=math.degrees(bend),
                kerb_mismatch_m=abs(exit_kerb - kerb),
                road_id=approach[0].road_id,
                lane_id=approach[0].lane_id,
                junction_id=through[0].junction_id if hasattr(through[0], "junction_id") else None,
                junction_length_m=len(through) - 1,
            )
        )
    sites.sort(key=lambda site: (site["kerb_source"] != "sidewalk", len(site["right_lanes"]), abs(abs(site["heading_change_deg"]) - 90)))
    return sites, census


def make_turn_route(carla_map, site, command, exit_in_m=5.0, spacing_m=25.0, name="g1c"):
    """LANEFOLLOW points along the approach, the entry point with the command at the lane's
    last waypoint before the junction, the exit point ``exit_in_m`` into the exit lane,
    then LANEFOLLOW to the goal at the exit run's end."""
    approach, exit_run = site["approach"], site["exit"]

    def point(wp, cmd):
        fix = carla_map.transform_to_geolocation(wp.transform.location)
        return RoutePoint(fix.latitude, fix.longitude, cmd)

    points = [point(approach[int(round(s))], Command.LANEFOLLOW) for s in np.arange(0.0, len(approach) - 1 - 10.0, spacing_m)]
    points.append(point(approach[-1], Command.RIGHT if command == "RIGHT" else Command.LEFT))
    points.append(point(exit_run[min(len(exit_run) - 1, int(round(exit_in_m)))], Command.LANEFOLLOW))
    points.append(point(exit_run[-1], Command.LANEFOLLOW))
    return SatNavRoute(name, tuple(points), provenance=dict(source="map lane centre", spacing_m=spacing_m, junction="entry at the lane's last waypoint, exit 5 m in"))


# -- live helpers ------------------------------------------------------------------------
class HoldingVehicleAdapter(FixedVehicleAdapter):
    """The verified adapter, tolerating the roll-back of a car held on the brake.

    The baseline adapter refuses reverse motion beyond 0.05 m/s; a car stopped at a junction
    hold on Town10HD's slight slope rolled back a few cm/s and the first live turn run
    crashed on it. Up to ``roll_back_mps`` is read as standing still; the harness counts
    the ticks. Forward motion and everything else are the baseline's.
    """

    roll_back_mps = 0.2  # ten times the roll-back seen live (a few cm/s); a real reverse beyond it still raises

    def __init__(self, vehicle):
        super().__init__(vehicle)
        self.roll_back_ticks = 0

    def state(self):
        from carla_adapter.vehicle_geometry import forward_state  # noqa: PLC0415

        if not self.vehicle.is_alive:
            raise RuntimeError("Vehicle is no longer alive")
        pose = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        if abs(pose.rotation.pitch) > 5 or abs(pose.rotation.roll) > 5:
            # Where and how much (external review, 26 Oct: two runs ended at the same spot
            # and the report could not say whether it was pitch or roll).
            loc = pose.location
            raise ValueError(
                "Vehicle tilt exceeds planar adapter scope: "
                f"pitch {pose.rotation.pitch:+.1f} deg, roll {pose.rotation.roll:+.1f} deg "
                f"at ({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f}), {math.hypot(velocity.x, velocity.y):.2f} m/s"
            )
        yaw = math.radians(pose.rotation.yaw)
        if velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw) < -0.05:
            self.roll_back_ticks += 1
        return forward_state(pose.location.x, pose.location.y, pose.rotation.yaw, velocity.x, velocity.y, self.geometry, reverse_tolerance_mps=self.roll_back_mps)


class CollisionMonitor:
    def __init__(self, owner, vehicle):
        self.owner, self.vehicle = owner, vehicle
        self.events = []
        self.sensor = None

    def __enter__(self):
        import carla

        world = self.owner.world
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.owner._actors.append(self.sensor)
        self.sensor.listen(lambda e: self.events.append(dict(frame=e.frame, other=e.other_actor.type_id)))
        return self

    def __exit__(self, *exc):
        try:
            if self.sensor is not None and self.sensor.is_alive:
                self.sensor.stop()
        except Exception:
            pass
        return False


def vehicle_body(ego, rear_axle_offset_m):
    box = ego.bounding_box
    return VehicleBody(
        front_m=rear_axle_offset_m + box.location.x + box.extent.x,
        rear_m=rear_axle_offset_m + box.location.x - box.extent.x,
        half_width_m=box.extent.y,
        height_m=box.location.z + box.extent.z,
    )


def render_snapshot(driver, path, half_width_m=20.0, ahead_m=45.0, behind_m=10.0, cell_m=0.25):
    """Top-down terrain labels around the car, with the corridor centreline and evidence."""
    import cv2

    corridor = driver.corridor.previous
    pose = driver.ego.odometry.pose
    xs = np.arange(-behind_m, ahead_m, cell_m)
    ys = np.arange(-half_width_m, half_width_m, cell_m)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    body = np.column_stack((gx.ravel(), gy.ravel()))
    labels = driver.grid.labels(pose.to_world(body)).reshape(gx.shape)
    image = np.full(labels.shape + (3,), 235, np.uint8)
    image[labels == CellLabel.ROAD] = (150, 150, 150)
    image[labels == CellLabel.BARRIER] = (40, 90, 200)
    image[labels == CellLabel.OCCUPIED] = (200, 60, 160)
    image = np.ascontiguousarray(image[::-1, ::-1])  # forward up, left on the left

    def px(x, y):
        return int((half_width_m - y) / cell_m), int((ahead_m - x) / cell_m)

    if corridor is not None and corridor.ok:
        centre = pose.to_body(corridor.centre_odometry)
        for a, b in zip(centre[:-1], centre[1:]):
            cv2.line(image, px(*a), px(*b), (0, 170, 0), 2)
        for x, y in pose.to_body(corridor.pose.to_world(corridor.evidence_body)):
            cv2.circle(image, px(x, y), 2, (0, 0, 0), -1)
    b = driver.config.body
    cv2.rectangle(image, px(b.front_m, b.half_width_m), px(b.rear_m, -b.half_width_m), (0, 0, 255), 1)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), image)


def plot(log, path, cruise_mps):
    import cv2

    width, height = 1000, 520
    img = np.full((height, width, 3), 255, np.uint8)
    rows = [r for r in log if r["truth"]["s_m"] is not None]
    if not rows:
        cv2.imwrite(str(path), img)
        return
    s = np.array([r["truth"]["s_m"] for r in rows])
    lat = np.array([r["truth"]["lane_offset_m"] for r in rows])
    t = np.array([r["time_s"] for r in rows])
    v = np.array([r["truth"]["speed_mps"] for r in rows])
    s_max, t_max = max(1.0, s.max()), max(1.0, t.max())

    def panel(top, values, xs, x_max, y_lo, y_hi, colour, label, lines=()):
        h = 220
        cv2.rectangle(img, (60, top), (width - 20, top + h), (0, 0, 0), 1)
        for y in lines:
            py = int(top + h * (y_hi - y) / (y_hi - y_lo))
            cv2.line(img, (60, py), (width - 20, py), (180, 180, 180), 1)
        pts = [
            (int(60 + (width - 80) * x / x_max), int(top + h * (y_hi - np.clip(y, y_lo, y_hi)) / (y_hi - y_lo)))
            for x, y in zip(xs, values)
        ]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, colour, 1)
        cv2.putText(img, label, (65, top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    panel(20, lat, s, s_max, -1.0, 1.0, (200, 60, 0), "lane-centre offset (m) vs distance; grey: 0, +-0.35, +-0.7", (0, 0.35, -0.35, 0.7, -0.7))
    panel(280, v, t, t_max, 0.0, cruise_mps + 1.0, (0, 140, 0), "speed (m/s) vs time; grey: cruise", (cruise_mps,))
    cv2.imwrite(str(path), img)


def _fmt(value):
    return "-" if value is None else (f"{value:.2f}" if isinstance(value, (int, float)) else str(value))


def driver_config(body, wheelbase_m, steering_limit_rad, speed_mps, keep_right_m):
    """The driver configuration the harness runs with (shared with the offline replay)."""
    controller = ControllerConfig(
        wheelbase_m=wheelbase_m,
        max_steering_angle_rad=steering_limit_rad,
        lookahead_m=4.0,
        kp=0.45,
        ki=0.15,
        stopping_deceleration_mps2=1.0,
        hold_brake=0.5,
    )
    return DriverConfig(
        body=body,
        control=LocalControlConfig(controller),
        corridor=CorridorConfig(keep_right_m=keep_right_m),
        planner=PlannerConfig(cruise_mps=speed_mps),
    )


def rounded(value, digits=3):
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [rounded(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, np.generic):
        return rounded(value.item(), digits)
    return value


def check(name, status, **values):
    return dict(check=name, status=status, **rounded(values))


# -- the run -------------------------------------------------------------------------------
STEP_TARGET_MS = 50.0  # the declared real-time budget: one 20 Hz tick
BRAKE_LAG_S = 0.25  # the longest brake lag measured open loop (BL-E-03: 0.1-0.25 s)


def stop_obstacle_m(speed_mps, settled_ticks=100, dt_s=0.05, start_m=15.0, stop_short_m=8.0, base_m=35.0):
    """How far along the lane the stop scenario's parked car must stand for lane keeping to
    collect its ``settled_ticks`` (seventh review, F7-05: at 4 m/s the 35 m default gave 94
    of the 100): ``start_m`` before they count, ``settled_ticks`` at ``speed_mps``, and the
    ``stop_short_m`` the car holds short of the car's centre. Never under ``base_m`` (the
    sample-size requirement is kept, the route is lengthened), rounded up to 5 m."""
    need = start_m + settled_ticks * dt_s * speed_mps + stop_short_m
    return float(max(base_m, 5.0 * math.ceil(need / 5.0)))


def gate_view_report(log):
    """How far a left turn's oncoming gate could see against what the crossing needed
    (r9, r10: 40 m is 4.7 s at 30 km/h; the crossing at 2 m/s needs about 11 s, so beyond the
    LiDAR the stack's radar tracks must cover the rest, and on Town03 the front radar saw the
    oncoming car only from 33 m). Ticks the gate was evaluated, read clear, and read clear
    with its view short of the need; the view and the need when the car entered the
    opposing lane. None when no gate ran."""
    rows = [r["driver"] for r in log if (r.get("driver") or {}).get("turn_gate_required_s") is not None]
    if not rows:
        return None
    clear = [d for d in rows if d.get("turn_gate_view_s") is not None and not str(d.get("turn_gate") or "").startswith(("waiting", "in the opposing lane"))]
    short = [d for d in clear if d["turn_gate_view_s"] < d["turn_gate_required_s"]]
    last = rows[-1]
    # The stack's verdict kept "not clear" for a blocker no longer observed (tenth review,
    # 2B): how often, so a run that never lost one says the hold was not exercised.
    held = [d for d in ((r.get("driver") or {}) for r in log) if "no longer observed" in str(d.get("oncoming") or "")]
    return dict(
        evaluated_ticks=len(rows),
        clear_ticks=len(clear),
        clear_with_view_short_ticks=len(short),
        lost_blocker_held_ticks=len(held),
        worst_shortfall_s=round(max((d["turn_gate_required_s"] - d["turn_gate_view_s"] for d in short), default=0.0), 1),
        at_entry=dict(view_s=last.get("turn_gate_view_s"), required_s=last.get("turn_gate_required_s"), note=last.get("turn_gate")),
        note="beyond the gate's view the stack's tracks must cover the crossing's need",
    )


def parse_outages(items):
    """``--drop-sensor NAME:START:END`` (seconds from the run's first frame) as a list of
    (name, start_s, end_s)."""
    out = []
    for item in items:
        name, start, end = item.rsplit(":", 2)
        start, end = float(start), float(end)
        if not name or not 0.0 <= start < end:
            raise ValueError(f"--drop-sensor {item!r}: NAME:START:END with 0 <= START < END")
        out.append((name, start, end))
    return out


def drop_sensors(frame, outages, elapsed_s):
    """``frame`` with the sensors of an outage active at ``elapsed_s`` removed and their
    status MISSING (ninth review, T4: a sensor lost as the car decides, e.g. the LiDAR while
    a left turn's gate reads the oncoming lane). The recording keeps the frame as dropped,
    so a replay sees the same outage."""
    gone = {name for name, start, end in outages if start <= elapsed_s < end}
    if not gone:
        return frame
    status = tuple(SensorStatus(s.name, s.kind, SensorState.MISSING, None, "injected outage") if s.name in gone else s for s in frame.status)
    return dataclasses.replace(
        frame,
        cameras=tuple(c for c in frame.cameras if c.name not in gone),
        lidars=tuple(sweep for sweep in frame.lidars if sweep.name not in gone),
        radars=tuple(r for r in frame.radars if r.name not in gone),
        imu=None if "imu" in gone else frame.imu,
        gnss=None if "gnss" in gone else frame.gnss,
        status=status,
    )


def _burn(stop):
    """A busy loop until ``stop`` is set: one core's worth of contention (``LoadGenerator``)."""
    x = 0
    while not stop.is_set():
        for _ in range(20000):
            x = (x * 31 + 7) % 1000003


class LoadGenerator:
    """Deliberate resource contention for a run (tenth review, 4: the semantic worker under
    combined CPU and GPU load). ``cpu``: that many processes each spinning a core; ``gpu``:
    a thread multiplying 2048 x 2048 matrices on the CUDA device back to back (needs torch
    with CUDA). Started before the run, stopped after; what ran goes in the report."""

    def __init__(self, cpu=0, gpu=False):
        self.cpu, self.gpu = int(cpu), bool(gpu)
        self._procs, self._stop, self._gpu_thread, self._gpu_stop = [], None, None, None
        self.note = {}

    def __enter__(self):
        import multiprocessing  # noqa: PLC0415
        import threading  # noqa: PLC0415

        if self.cpu > 0:
            ctx = multiprocessing.get_context("spawn")
            self._stop = ctx.Event()
            self._procs = [ctx.Process(target=_burn, args=(self._stop,), daemon=True) for _ in range(self.cpu)]
            for proc in self._procs:
                proc.start()
        self.note = dict(cpu_processes=self.cpu, gpu="off")
        if self.gpu:
            try:
                import torch  # noqa: PLC0415

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA not available")
                self._gpu_stop = threading.Event()
                a = torch.randn(2048, 2048, device="cuda")

                def spin():
                    while not self._gpu_stop.is_set():
                        (a @ a).sum().item()

                self._gpu_thread = threading.Thread(target=spin, name="gpu-load", daemon=True)
                self._gpu_thread.start()
                self.note["gpu"] = "matmul 2048x2048 on " + torch.cuda.get_device_name(0)
            except Exception as error:  # the run goes on without it, and says so
                self.note["gpu"] = f"not started: {type(error).__name__}: {error}"
        return self

    def __exit__(self, *exc):
        if self._stop is not None:
            self._stop.set()
            for proc in self._procs:
                proc.join(timeout=2.0)
                if proc.is_alive():
                    proc.terminate()
        if self._gpu_stop is not None:
            self._gpu_stop.set()
            self._gpu_thread.join(timeout=2.0)
        return False


def sight_bound_report(log, tolerance_mps=0.3):
    """Whether the car kept to the observability bound (``sight_speed_mps``, the speed it can
    stop from within the path seen clear; eighth review, P1) on every tick it applied:
    the ticks it was over by more than ``tolerance_mps`` (the speed loop's lag), the largest
    excess, and the lowest bound met (tenth review, 5). None when the log has no bound."""
    rows = [(r["truth"]["speed_mps"], (r.get("driver") or {}).get("sight_speed_mps")) for r in log]
    rows = [(v, b) for v, b in rows if b is not None]
    if not rows:
        return None
    over = [v - b for v, b in rows if v > b + tolerance_mps]
    return dict(ticks=len(rows), over_ticks=len(over), max_excess_mps=round(max(over), 2) if over else 0.0, min_bound_mps=round(min(b for _, b in rows), 2), tolerance_mps=tolerance_mps)


def sensor_to_actuator_report(marks, target_ms=STEP_TARGET_MS):
    """The whole path from sensing to actuation on the wall clock, per tick (eighth review,
    section 5: the step's time is the autonomy's computation only, so it is no measured
    sensor-to-actuator deadline). ``marks``: per tick, the wall-clock (perf_counter, s)
    marks ``ticked`` (the synchronous world tick returned: every sensor has produced the
    frame), ``assembled`` (the rig's frame read), ``begun`` and ``ended`` (``step``) and
    ``applied`` (the command handed to the vehicle). Reported, per interval: ``delivery``
    (ticked to assembled: the data's hand-over to the client), ``record`` (assembled to
    begun: the frame recorder, when on), ``step``, ``apply`` (ended to applied) and
    ``sensor_to_actuator`` (ticked to applied), each p50, p95, p99 and max in ms, and the
    share of ticks whose whole path exceeded ``target_ms``. In synchronous CARLA the command
    takes effect at the next world tick whatever these are: they are what real time would
    cost, not what the simulated car saw. None without marks."""
    if not marks:
        return None
    spans = dict(delivery=("ticked", "assembled"), record=("assembled", "begun"), step=("begun", "ended"), apply=("ended", "applied"), sensor_to_actuator=("ticked", "applied"))
    out = {}
    for name, (a, b) in spans.items():
        ms = np.array([1000.0 * (m[b] - m[a]) for m in marks], dtype=float)
        out[f"{name}_ms"] = dict(p50=round(float(np.median(ms)), 2), p95=round(float(np.percentile(ms, 95)), 2), p99=round(float(np.percentile(ms, 99)), 2), max=round(float(ms.max()), 2))
    whole = np.array([1000.0 * (m["applied"] - m["ticked"]) for m in marks], dtype=float)
    out |= dict(target_ms=target_ms, over_target_share=round(float(np.mean(whole > target_ms)), 3), ticks=len(marks), note="wall clock; synchronous CARLA applies the command at the next world tick regardless")
    return out


def step_time_report(step_times_s, log, target_ms=STEP_TARGET_MS, response_s=None, brake_lag_s=BRAKE_LAG_S, dt_s=0.05, semantic_mode=None):
    """The driver step's wall-clock time against the 50 ms tick (third review: p95 and the
    worst case are first-class, so a passing drive cannot hide a missed deadline): p50,
    p95, p99 and max, the share of ticks over the target, and each stage's p50, p95 and
    max where they were logged (the stack's ``stack_timing_ms``, and the driver's own
    ``driver_timing_ms`` as ``driver.<stage>``). Wall-clock in synchronous mode:
    a slow tick slows the simulation's real-time rate, not the simulated car.

    Which time governs actuation (``actuation_governed_by``): the command reaches the
    vehicle when the step returns, so in both modes the whole step's measured wall time
    (seventh review, F7-02: in "thread" mode the sixth review's version took the control
    stages' sum, which leaves out the job hand-off's copies, locking, the diagnostics and
    the record). The control stages' sum stays as a decomposition (``control_path_ms``),
    with what the stage timers do not see (``uninstrumented_ms``, the step's time less
    every stage's, where the two line up tick for tick). The response budget is taken on
    the whole step."""
    times = np.array(step_times_s, dtype=float) * 1000
    if not len(times):
        return dict(note="no timed steps")
    out = dict(
        p50_ms=float(np.median(times)),
        p95_ms=float(np.percentile(times, 95)),
        p99_ms=float(np.percentile(times, 99)),
        max_ms=float(times.max()),
        target_ms=target_ms,
        over_target_share=round(float(np.mean(times > target_ms)), 3),
        ticks=len(times),
    )
    stages = {}
    for row in log:
        d = row.get("driver") or {}
        for key, prefix in (("stack_timing_ms", ""), ("driver_timing_ms", "driver.")):
            for name, ms in (d.get(key) or {}).items():
                stages.setdefault(prefix + name, []).append(ms)
    if stages:
        out["stages_ms"] = {name: dict(p50=round(float(np.median(v)), 1), p95=round(float(np.percentile(v, 95)), 1), max=round(float(max(v)), 1), ticks=len(v)) for name, v in stages.items()}
    # The control path apart from the optional semantic work (fifth review, F5-05): the
    # driver (health, obstacle stop, plan, control), detection, tracking and the speed
    # rules run every tick and must meet the deadline; segmentation, painting and the road
    # model inform no command yet (``optional``).
    control, optional = [], []
    for row in log:
        timing = (row.get("driver") or {}).get("stack_timing_ms")
        if timing:
            control.append(sum(timing.get(k, 0.0) for k in CONTROL_STAGES))
            optional.append(sum(timing.get(k, 0.0) for k in OPTIONAL_STAGES))
    if control:
        c, o = np.array(control), np.array(optional)
        inline = semantic_mode != "thread"
        out["semantic_mode"] = semantic_mode or "inline"
        out["actuation_governed_by"] = "the whole step: " + ("the semantic work runs inside it" if inline else "the semantic work runs on its worker thread, outside it")
        out["control_path_ms"] = dict(p50=round(float(np.median(c)), 1), p95=round(float(np.percentile(c, 95)), 1), max=round(float(c.max()), 1), over_target_share=round(float(np.mean(c > target_ms)), 3), stages=list(CONTROL_STAGES))
        out["optional_ms"] = dict(p50=round(float(np.median(o)), 1), p95=round(float(np.percentile(o, 95)), 1), max=round(float(o.max()), 1), stages=list(OPTIONAL_STAGES))
        if len(log) == len(times):
            staged = np.array([sum((row.get("driver") or {}).get("stack_timing_ms", {}).values()) for row in log])
            rest = times - staged
            out["uninstrumented_ms"] = dict(p50=round(float(np.median(rest)), 1), p95=round(float(np.percentile(rest, 95)), 1), max=round(float(rest.max()), 1))
        if response_s:
            # What the 50 ms is for (fifth review, F5-05): in real time, a road user's change
            # reaches the brakes after the tick that sees it (the cap applies on the next),
            # the control path's own time and the brake's lag. The following rule's worst
            # case allows response_s for all of it. Synchronous CARLA hides the processing
            # time; this says whether the budget would hold if it did not.
            governing = np.array(times)
            worst = dt_s + float(np.percentile(governing, 99)) / 1000.0 + brake_lag_s
            out["response_budget"] = dict(
                tick_s=dt_s,
                governing="whole step",
                governing_p99_s=round(float(np.percentile(governing, 99)) / 1000.0, 3),
                control_path_p99_s=round(float(np.percentile(c, 99)) / 1000.0, 3),
                brake_lag_s=brake_lag_s,
                worst_s=round(worst, 3),
                allowed_s=response_s,
                within=bool(worst <= response_s),
            )
    return out


def semantic_worker_report(log):
    """The semantic worker over the run (seventh review, F7-04), or None inline: its final
    counters and shares (jobs dropped and failed, results stale or late on the wall
    clock), and over the results the stack took, their simulation age, wall-clock age (from
    the job's submission to its use), queue wait and run time (p50, p95, max)."""
    stats = [s for s in ((r.get("driver") or {}).get("semantic_worker") for r in log) if s]
    if not stats:
        return None
    takes = [t for t in ((r.get("driver") or {}).get("semantic_take") for r in log) if t]

    def summary(key):
        v = np.array([t[key] for t in takes if t.get(key) is not None], dtype=float)
        return None if not len(v) else dict(p50=round(float(np.median(v)), 3), p95=round(float(np.percentile(v, 95)), 3), max=round(float(v.max()), 3))

    last = stats[-1]
    keys = ("submitted", "dropped", "segmented", "built", "taken", "stale_epoch", "stale_age", "stale_wall", "late_wall", "errors", "dropped_share", "error_share", "stale_share", "late_share")
    return {k: last.get(k) for k in keys} | dict(
        takes=len(takes),
        sim_age_s=summary("sim_age_s"),
        wall_age_ms=summary("wall_age_ms"),
        queue_wait_ms=summary("queue_wait_ms"),
        run_ms=summary("run_ms"),
        last_error=last.get("last_error"),
    )


def semantic_shadow_report(log, kerb_distance_m=None):
    """The segmenter in shadow (fifth review, ML stage 2), over the road-model ticks: how
    often its evidence covered the lane band along the path, the median share of road in
    it, and the ticks where most of it read pavement (where it disagrees with the kerb
    corridor). Sixth review, ML: the share of the band its evidence covers (``coverage``),
    the mean pavement share in the band (road/pavement disagreement), and, given the site's
    lane-centre-to-kerb distance (``kerb_distance_m``, the evaluator's truth), the error of
    the road's right edge it shows (``right_edge_m`` minus the kerb distance: positive, the
    edge put beyond the kerb). None without a stack."""
    pairs = [(r.get("mode"), s) for r, s in ((r, (r.get("driver") or {}).get("semantic_shadow")) for r in log) if s]
    rows = [s for _, s in pairs]
    if not rows:
        return None
    shares = [s["road_share"] for s in rows if s.get("road_share") is not None]
    pavement = [s["pavement_share"] for s in rows if s.get("pavement_share") is not None]
    coverage = [s["coverage"] for s in rows if s.get("coverage") is not None]
    edges = [s["right_edge_m"] for s in rows if s.get("right_edge_m") is not None]
    out = dict(
        ticks=len(rows),
        with_evidence_ticks=len(shares),
        coverage_median=round(float(np.median(coverage)), 3) if coverage else None,
        coverage_p10=round(float(np.percentile(coverage, 10)), 3) if coverage else None,
        road_share_median=round(float(np.median(shares)), 2) if shares else None,
        road_share_p10=round(float(np.percentile(shares, 10)), 2) if shares else None,
        pavement_share_mean=round(float(np.mean(pavement)), 3) if pavement else None,
        mostly_pavement_ticks=sum(1 for v in pavement if v > 0.5),
        right_edge_ticks=len(edges),
        note="logged only: no rule reads the segmenter's output",
    )
    # By section of the path ahead and by the driver's mode, the visibility conditions
    # differ (seventh review, F7-06): coverage, the edge's signed error and the share of
    # stations where road was seen right of the line but no edge.
    def group(items, key=None):
        cov = [x.get("coverage") for x in items if x.get("coverage") is not None]
        edge = [x.get("right_edge_m") for x in items if x.get("right_edge_m") is not None]
        found = sum(x.get("edge_stations", x.get("right_edge_stations", 0)) or 0 for x in items)
        missed = sum(x.get("missed", 0) or 0 for x in items)
        res = dict(ticks=len(items), coverage_median=round(float(np.median(cov)), 3) if cov else None, edge_ticks=len(edge), missed_share=round(missed / (missed + found), 3) if missed + found else None)
        if edge and kerb_distance_m is not None:
            err = np.array(edge) - float(kerb_distance_m)
            res |= dict(edge_error_median_m=round(float(np.median(err)), 2), edge_error_mae_m=round(float(np.mean(np.abs(err))), 2))
        return res

    sections = sorted({name for s in rows for name in (s.get("sections") or {})})
    if sections:
        out["by_section"] = {name: group([s["sections"][name] for s in rows if name in (s.get("sections") or {})]) for name in sections}
    out["by_mode"] = {mode: group([s for m, s in pairs if m == mode]) for mode in sorted({m for m, _ in pairs if m})}
    if edges and kerb_distance_m is not None:
        err = np.array(edges) - float(kerb_distance_m)
        out["right_edge_error_m"] = dict(
            kerb_distance_m=round(float(kerb_distance_m), 2),
            median=round(float(np.median(err)), 2),
            mae=round(float(np.mean(np.abs(err))), 2),
            p95_abs=round(float(np.percentile(np.abs(err), 95)), 2),
            note="from the path line, which keeps within the lane-keeping error of the lane centre",
        )
    return out


CONTROL_STAGES = ("drive", "detect", "track", "plan", "semantic_submit")
OPTIONAL_STAGES = ("paint", "road_model")


class MaplessDrive:
    def __init__(self, carla, owner, suite, args, report):
        self.carla, self.owner, self.world = carla, owner, owner.world
        self.suite, self.args, self.report = suite, args, report
        self.library = self.world.get_blueprint_library()

    @property
    def turn_command(self):
        return TURN_SCENARIOS.get(self.args.scenario)

    @property
    def actor_scenario(self):
        return self.args.scenario if self.args.scenario in ACTOR_SCENARIOS else None

    @property
    def actor_name(self):
        """The scripted actor's scenario name for scoring: an actor scenario, or the
        oncoming car of a left turn (``oncoming_stopped`` / ``oncoming_parked``)."""
        if self.actor_scenario:
            return self.actor_scenario
        mode = getattr(self.args, "oncoming", "none")
        return f"oncoming_{mode}" if mode != "none" else None

    def setup(self):
        carla, args = self.carla, self.args
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        carla_map = self.world.get_map()
        if self.turn_command:
            return self.setup_turn(carla_map)
        length = args.goal_m + 15.0
        lane_kind = "straight" if self.actor_scenario else args.scenario
        sites, census = find_sites(carla, carla_map, lane_kind, length)
        self.report["site_census"] = census
        if len(sites) <= args.site:
            runs = ", ".join(f"{r['length_m']} m" for r in census["longest_rightmost_runs_m"])
            raise RuntimeError(
                f"Only {len(sites)} {args.scenario} sites on this map for --goal-m {args.goal_m:g} "
                f"(needs {length:g} m without a junction); --site {args.site} not available. "
                f"Census: {json.dumps({k: v for k, v in census.items() if k != 'longest_rightmost_runs_m'})}. "
                f"Longest junction-free rightmost lanes: {runs}. A shorter --goal-m or another --map may fit."
            )
        self.ego = None
        for index in range(args.site, min(len(sites), args.site + 10)):
            spawn = sites[index]["waypoints"][0].transform
            spawn.location.z += 0.3
            self.ego = self.owner.try_spawn(self.library.find("vehicle.tesla.model3"), spawn)
            if self.ego is not None:
                break
        if self.ego is None:
            raise RuntimeError("Could not spawn the car at any of the next 10 sites; try another --site")
        self.site = site = sites[index] | dict(site_index=index)
        wps = site["waypoints"]
        lane = np.array([core_xy(w.transform.location) for w in wps])
        # Extend the truth lane 15 m behind the start: the rear axle starts 1.4 m behind the
        # first waypoint, and the first live run scored that as a 1.39 m lane offset.
        back = (lane[0] - lane[1]) / np.linalg.norm(lane[0] - lane[1])
        self.lane = np.vstack([lane[0] + back * k for k in range(LANE_BACK_M, 0, -1)] + [lane])
        # Wheel positions are only valid once physics has run: the first live attempt read
        # misaligned axles straight after the spawn. The calibration gate also waits 30 ticks.
        self.ego.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(30):
            self.owner.tick()
        self.adapter = HoldingVehicleAdapter(self.ego)
        self.offset = self.adapter.geometry.rear_axle_offset_m
        self.parked = None
        self.actor = None
        self.turn_site = None
        if self.actor_scenario:
            self.actor = make_scenario(carla, self.actor_scenario, core_xy, **(args.actor_params or {}))
            self.actor.spawn(self.owner, self.library, wps)
            self.report["actor"] = self.actor.params.record()
        if args.scenario == "stop":
            wp = wps[int(round(args.obstacle_m))]
            t = wp.transform
            t.location.z += 0.1
            self.parked = self.owner.try_spawn(self.library.find(args.obstacle_blueprint), t)
            if self.parked is None:
                raise RuntimeError("Could not place the parked car")
            self.parked.set_simulate_physics(False)
        self.route = make_route(carla_map, wps, args.goal_m)
        self.route.save(args.output / "route.json")
        self.goal_xy = self.lane[LANE_BACK_M + int(round(args.goal_m))]
        self.lane_arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(self.lane, axis=0).T))))
        self.lane_end_s = float(self.lane_arcs[-1]) - LANE_BACK_M
        self.report["site"] = rounded(
            {k: v for k, v in site.items() if k != "waypoints"} | dict(map=carla_map.name, start_xy_m=self.lane[LANE_BACK_M].tolist())
        )

    def setup_turn(self, carla_map):
        carla, args, command = self.carla, self.args, self.turn_command
        sites, census = find_turn_sites(carla, carla_map, command, args.approach_m, args.exit_m, max_kerb_mismatch_m=args.max_kerb_mismatch)
        self.report["site_census"] = census
        if len(sites) <= args.site:
            raise RuntimeError(
                f"Only {len(sites)} {args.scenario} sites on this map for --approach-m {args.approach_m:g} "
                f"--exit-m {args.exit_m:g}; --site {args.site} not available. Census: {json.dumps(census)}."
            )
        self.ego = None
        for index in range(args.site, min(len(sites), args.site + 10)):
            spawn = sites[index]["approach"][0].transform
            spawn.location.z += 0.3
            self.ego = self.owner.try_spawn(self.library.find("vehicle.tesla.model3"), spawn)
            if self.ego is not None:
                break
        if self.ego is None:
            raise RuntimeError("Could not spawn the car at any of the next 10 turn sites; try another --site")
        site = sites[index] | dict(site_index=index)
        self.site = site
        approach = np.array([core_xy(w.transform.location) for w in site["approach"]])
        back = (approach[0] - approach[1]) / np.linalg.norm(approach[0] - approach[1])
        approach = np.vstack([approach[0] + back * k for k in range(LANE_BACK_M, 0, -1)] + [approach])
        self.turn_site = TurnSite(
            command,
            approach,
            np.array([core_xy(w.transform.location) for w in site["junction"]]),
            np.array([core_xy(w.transform.location) for w in site["exit"]]),
            site["lane_width_m"],
            site["junction_width_m"],
            site["exit_width_m"],
        )
        self.lane = self.turn_site.path
        self.ego.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(30):
            self.owner.tick()
        self.adapter = HoldingVehicleAdapter(self.ego)
        self.offset = self.adapter.geometry.rear_axle_offset_m
        self.parked = None
        self.actor = None
        if getattr(args, "oncoming", "none") != "none":
            # An oncoming car across the junction for the left-turn gate (fifth review, F5-07).
            self.actor = make_oncoming(carla, args.oncoming, core_xy, **(args.actor_params or {}))
            self.actor.spawn_at_turn(self.owner, self.library, site)
            self.report["actor"] = self.actor.params.record() | dict(mode=args.oncoming)
        self.route = make_turn_route(carla_map, site, command)
        self.route.save(args.output / "route.json")
        self.goal_xy = self.turn_site.exit[-1]
        self.lane_arcs = self.turn_site.arcs
        self.lane_end_s = float(self.lane_arcs[-1]) - LANE_BACK_M
        self.report["site"] = rounded(
            {k: v for k, v in site.items() if k not in ("approach", "junction", "exit")}
            | dict(map=carla_map.name, start_xy_m=approach[LANE_BACK_M].tolist(), junction_bounds_m=[b - LANE_BACK_M for b in self.turn_site.segment_bounds()])
        )

    def _follow_with_spectator(self):
        """Park the server's spectator camera behind the ego (a view only; no effect on the run)."""
        carla = self.carla
        t = self.ego.get_transform()
        yaw = math.radians(t.rotation.yaw)
        self.world.get_spectator().set_transform(
            carla.Transform(
                carla.Location(x=t.location.x - 8 * math.cos(yaw), y=t.location.y - 8 * math.sin(yaw), z=t.location.z + 5),
                carla.Rotation(pitch=-22, yaw=t.rotation.yaw),
            )
        )

    def truth(self):
        t = self.ego.get_transform()
        x, y, _, yaw = rear_axle_pose(t, self.offset)
        if self.turn_site is not None:
            sample = truth_sample(self.turn_site, (x, y), yaw, self.body)
            v = self.ego.get_velocity()
            return dict(
                rear_axle_xy=[x, y],
                yaw_rad=yaw,
                s_m=sample["s_m"] - LANE_BACK_M,
                lane_offset_m=sample["lane_offset_m"],
                segment=sample["segment"],
                corner_excess_m=sample["corner_excess_m"],
                corner_offset_max_m=sample["corner_excess_m"],
                heading_error_rad=sample["heading_error_rad"],
                speed_mps=math.hypot(v.x, v.y),
                goal_distance_m=float(math.dist((x, y), self.goal_xy)),
                gap_m=None,
            )
        s, lateral, outside = project_onto_polyline(np.array([[x, y]]), self.lane)
        # The lane heading at the car's arc length (the waypoints are not exactly 1 m
        # apart in bends: reading the segment by index drifted up to 10 degrees on the
        # 39 m curve site and failed the footprint gate on a run at -0.06 m).
        i = int(np.clip(np.searchsorted(self.lane_arcs, float(s[0])) - 1, 0, len(self.lane) - 2))
        heading = math.atan2(*(self.lane[i + 1] - self.lane[i])[::-1])
        # Each body corner's own offset from the lane centre, for the footprint gate.
        b, c, sn = self.body, math.cos(yaw), math.sin(yaw)
        corners = np.array(
            [(x + c * u - sn * v, y + sn * u + c * v) for u, v in ((b.front_m, b.half_width_m), (b.front_m, -b.half_width_m), (b.rear_m, -b.half_width_m), (b.rear_m, b.half_width_m))]
        )
        _, corner_lateral, _ = project_onto_polyline(corners, self.lane)
        gap = None
        if self.parked is not None:
            p = core_xy(self.parked.get_transform().location)
            ps, _, _ = project_onto_polyline(np.array([p]), self.lane)
            gap = float(ps[0]) - self.parked.bounding_box.extent.x - (float(s[0]) + self.body.front_m)
        v = self.ego.get_velocity()
        return dict(
            rear_axle_xy=[x, y],
            yaw_rad=yaw,
            s_m=float(s[0]) - LANE_BACK_M,  # from the route start
            lane_offset_m=float(lateral[0]),
            corner_offset_max_m=float(np.abs(corner_lateral).max()),
            heading_error_rad=wrap(yaw - heading),
            speed_mps=math.hypot(v.x, v.y),
            goal_distance_m=float(math.dist((x, y), self.goal_xy)),
            gap_m=gap,
        )

    def run(self):
        carla, args = self.carla, self.args
        self.setup()
        self.body = vehicle_body(self.ego, self.offset)
        config = driver_config(self.body, self.adapter.geometry.wheelbase_m, self.adapter.geometry.controller_steering_limit_rad, args.speed, args.keep_right)
        self.report["driver_config"] = rounded(_config_dict(config))
        rig = SensorRig(
            self.owner,
            self.ego,
            self.suite,
            episode_id=f"g1b-{uuid.uuid4().hex[:8]}",
            rear_axle_offset_m=self.offset,
            dt_s=DT_S,
            seed=args.seed,
        )
        snapshots = args.output / "snapshots"
        if args.snapshot_every:
            snapshots.mkdir()
        sweeps = args.output / "sweeps"
        sweeps.mkdir()
        record_frames = args.record_frames == "all" or (args.record_frames == "turns" and self.turn_site is not None)
        recorder = FrameRecorder(args.output / "frames", self.suite, rig.calibration_id, rig.episode_id, rig.extrinsics) if record_frames else None
        if self.turn_site is not None:
            ts = self.turn_site
            (args.output / "turn_truth.json").write_text(
                json.dumps(
                    dict(
                        command=ts.command,
                        approach=np.round(ts.approach, 3).tolist(),
                        junction=np.round(ts.junction, 3).tolist(),
                        exit=np.round(ts.exit, 3).tolist(),
                        approach_width_m=ts.approach_width_m,
                        junction_width_m=ts.junction_width_m,
                        exit_width_m=ts.exit_width_m,
                        kerb_distance_m=self.site.get("kerb_distance_m"),
                        exit_kerb_distance_m=self.site.get("exit_kerb_distance_m"),
                        lane_back_m=LANE_BACK_M,
                        frame="evaluator core frame (CARLA x, -y); rear-axle truth in drive_log.jsonl is in the same frame",
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
        moved = False
        log, step_times, modes = [], [], []
        hold_ticks = 0
        junction_hold_ticks = 0
        finished_ticks = 0
        end = "time limit"
        with rig, CollisionMonitor(self.owner, self.ego) as collisions, open(
            args.output / "drive_log.jsonl", "w", encoding="utf-8"
        ) as out:
            self.ego.apply_control(carla.VehicleControl(brake=1.0))
            for _ in range(SETTLE_TICKS):
                frame = self.owner.tick()
                snap = self.world.get_snapshot()
                rig.read(frame, true_forward_speed(snap, self.ego.id), sim_time_s=snap.timestamp.elapsed_seconds)
            if args.stack:
                stack_config = StackConfig(admission=Admission.preset(args.admission), segmenter=args.segmenter, detector=args.detector, semantic_mode=getattr(args, "semantic", "thread"))
                driver = MaplessStack(self.suite, self.route, config, stack_config, DT_S)
                self.report["stack"] = dict(
                    admission=args.admission,
                    segmenter=args.segmenter or "constant",
                    segmenter_sha256=_file_sha256(args.segmenter) if args.segmenter not in (None, "constant") else None,
                    detector=args.detector or "none",
                    stack_config_sha256=hashlib.sha256(json.dumps(dataclasses.asdict(stack_config), sort_keys=True, default=str).encode()).hexdigest(),
                    segmenter_load=getattr(driver.segmenter, "load_mode", None),
                    # Where the semantic work ran and on which ticks (sixth review, F6-01, F6-07).
                    semantic_mode=stack_config.semantic_mode,
                    optional_schedule=driver.schedule,
                )
                f = stack_config.following
                self.report["braking_assumptions"] = braking_assumptions(config.planner.decel_mps2, stack_config.conflict_decel_mps2, f.ego_brake_mps2, f.response_s)
            else:
                driver = MaplessDriver(self.suite, self.route, config, DT_S)
                self.report["braking_assumptions"] = braking_assumptions(config.planner.decel_mps2)
            start_time = None
            outages = parse_outages(args.drop_sensor)  # injected sensor outages (ninth review, T4)
            actor_truths = []
            actor_sets = []  # every scripted actor's truth (F8-05), when there are several
            latencies = []  # the per-tick wall-clock marks (sensor_to_actuator_report)
            for tick in range(int(args.max_s / DT_S)):
                # Wall-clock marks per tick (eighth review, section 5): the world tick
                # returned (every sensor has produced this frame's data), the frame
                # assembled, the step begun and ended, the command handed to the vehicle.
                frame = self.owner.tick()
                ticked = time.perf_counter()
                snap = self.world.get_snapshot()
                if args.follow:
                    self._follow_with_spectator()
                sf = rig.read(frame, true_forward_speed(snap, self.ego.id), sim_time_s=snap.timestamp.elapsed_seconds)
                start_time = sf.sim_time_s if start_time is None else start_time
                if outages:
                    sf = drop_sensors(sf, outages, sf.sim_time_s - start_time)
                assembled = time.perf_counter()
                if recorder is not None:
                    recorder.record(tick, sf)
                begun = time.perf_counter()
                step = driver.step(sf)
                ended = time.perf_counter()
                step_times.append(ended - begun)
                try:
                    self.adapter.apply(step.command)
                    applied = time.perf_counter()
                    latencies.append(dict(ticked=ticked, assembled=assembled, begun=begun, ended=ended, applied=applied))
                except ValueError as error:
                    # The planar adapter's scope (tilt, reverse motion): a result to score,
                    # not a traceback — the gates and the recording are still written.
                    self.adapter.stop()
                    end = f"adapter refused the vehicle state: {error}"
                    break
                truth = self.truth()
                if self.actor is not None:
                    self.actor.step(tick, truth)
                    actor_truths.append(self.actor.truth())
                    actor_sets.append(self.actor.truths())
                modes.append(step.mode)
                row = dict(
                    tick=tick,
                    time_s=sf.sim_time_s - start_time,
                    mode=step.mode.value,
                    reason=step.reason,
                    command=[step.command.steering_angle_rad, step.command.throttle, step.command.brake],
                    truth=truth,
                    driver=step.diagnostics,
                    degraded=[s.name for s in sf.degraded()],
                    actor=actor_truths[-1] if self.actor is not None else None,  # evaluator truth, log only
                )
                if self.actor is not None and len(actor_sets[-1]) > 1:
                    row["actors"] = actor_sets[-1]  # all of them, by role (evaluator truth, log only)
                out.write(json.dumps(rounded(row)) + "\n")
                log.append(row)
                if args.snapshot_every and tick % args.snapshot_every == 0 and step.mode is not DriveMode.STARTUP:
                    render_snapshot(driver.driver if args.stack else driver, snapshots / f"t{tick:05d}.png")
                if truth["speed_mps"] < 0.05 and not moved and sf.lidars:
                    sweep = sf.lidar(self.suite.lidars[0].name)
                    np.savez_compressed(sweeps / f"t{tick:05d}.npz", points=sweep.points, mode=step.mode.value)
                moved = moved or truth["speed_mps"] >= 0.05
                if tick % 20 == 0:
                    d = step.diagnostics
                    line = (
                        f"t={row['time_s']:6.1f}s s={truth['s_m']:6.1f}m off={truth['lane_offset_m']:+.2f}m"
                        f" v={truth['speed_mps']:.2f} {step.mode.value:9} {step.reason[:40]:40}"
                        f" corridor={d.get('corridor_valid_m')} obstacle={d.get('obstacle_m')}"
                    )
                    if self.turn_site is not None:
                        fit = d.get("turn_exit_fit") or {}
                        line += (
                            f" | {truth['segment'][:4]} excess={truth['corner_excess_m']:+.2f} entry={_fmt(d.get('junction_entry_m'))}"
                            f" open={_fmt(d.get('kerb_opening_m'))} R={_fmt(d.get('turn_radius_m'))} exit={d.get('turn_exit')}"
                            f" kerb@{_fmt(fit.get('kerb_from_probe_m'))} ver={_fmt(d.get('turn_verified_m'))}"
                            f" clr={_fmt(d.get('turn_clearance_m'))} {d.get('turn') or ''}"
                            + (f" [{d['turn_refine']}]" if d.get("turn_refine") else "")
                        )
                    print(line, flush=True)
                if collisions.events:
                    end = "collision"
                    break
                if self.turn_site is None and truth["s_m"] > args.goal_m + 10.0:
                    end = "drove past the goal"
                    break
                if truth["s_m"] >= self.lane_end_s - 1.0:
                    end = "reached the end of the truth lane"
                    break
                if self.turn_site is None and abs(truth["lane_offset_m"]) > 3.0:
                    end = "left the lane (more than 3 m off centre)"
                    break
                if self.turn_site is not None and truth["corner_excess_m"] > 1.0:
                    # A corner a metre outside the permitted area: the exit lane was placed
                    # wrongly (opposing lane or pavement). Stop before it gets worse.
                    end = "left the permitted area (corner more than 1 m outside)"
                    break
                if step.mode is DriveMode.FINISHED and truth["speed_mps"] < 0.05:
                    finished_ticks += 1
                    if finished_ticks >= 20:
                        end = "finished"
                        break
                held = step.mode is DriveMode.STOPPING and step.reason.startswith("holding") and truth["speed_mps"] < 0.05
                # Hold ticks count only while a hold may end the run: a crossing walker's yield
                # counted towards --hold-s, so the run ended the tick the walker cleared, with
                # no ticks left for the car to resume (fresh-eyes review of 58adaf3).
                endable = args.scenario == "stop" or (self.actor is not None and self.actor.hold_ends_run())
                hold_ticks = hold_ticks + 1 if held and endable else 0
                # A turn that waits at the junction mouth (no confirmed opening, no clearing
                # radius, exit kerb not measured) is a result, not a time-out: end it after
                # --junction-hold-s and say why, instead of sitting out --max-s.
                waiting = self.turn_site is not None and truth["speed_mps"] < 0.05 and step.diagnostics.get("stop_reason") == "junction"
                junction_hold_ticks = junction_hold_ticks + 1 if waiting else 0
                if junction_hold_ticks * DT_S >= args.junction_hold_s:
                    end = f"held at the junction for {args.junction_hold_s:g} s: {step.diagnostics.get('turn') or 'exit lane not measured'}"
                    break
                # A scripted actor decides whether a hold is an endpoint for its scenario
                # (the crossing walker must be waited out, not held against; review).
                if endable and hold_ticks * DT_S >= args.hold_s:
                    end = "held at the obstacle"
                    break
            self.adapter.stop()
            self.owner.tick()
            if self.actor is not None:
                self.actor.destroy()
        self.report["end"] = end
        if recorder is not None:
            recorder.close()
            self.report["frames_recorded"] = recorder.ticks
        self.report["roll_back_ticks"] = self.adapter.roll_back_ticks
        self.report["collisions"] = collisions.events
        self.report["rig"] = rig.record()
        if args.stack:
            driver.close()
            (args.output / "decisions.jsonl").write_text("\n".join(json.dumps(rounded(r.as_dict())) for r in driver.records), encoding="utf-8")
        plot(log, args.output / "drive_plot.png", args.speed)
        self.latencies = latencies
        self.score(log, step_times, modes, hold_ticks, actor_truths, actor_sets)

    def score(self, log, step_times, modes, hold_ticks, actor_truths=(), actor_sets=()):
        args, checks = self.args, self.report["checks"]
        startup = [r for r in log if r["mode"] != DriveMode.STARTUP.value]
        if not startup:
            checks.append(check("startup", FAIL, note="never left STARTUP"))
            return
        first = startup[0]
        start_s = first["time_s"]
        pose = first["driver"].get("route_pose")
        start_error = None
        if pose:
            true = np.array(first["truth"]["rear_axle_xy"]) - self.lane[LANE_BACK_M]
            start_error = float(np.hypot(*(np.array(pose[:2]) - true)))
        checks.append(check("startup", PASS if start_s <= 3.0 else FAIL, seconds=start_s, route_pose_error_m=start_error))
        checks.append(check("no_collision", PASS if not self.report["collisions"] else FAIL, events=self.report["collisions"]))
        s0 = first["truth"]["s_m"]
        if self.turn_site is not None:
            # Junction-aware gates (review of 697d53c, point F): the car must leave the approach
            # lane's corridor, so the permitted area replaces the lane footprint gate.
            turning = sum(1 for m in modes if m is DriveMode.TURNING)
            for name, passed, values in score_turn(log, self.turn_site, turning, end=self.report["end"], s_offset_m=LANE_BACK_M):
                status = INCOMPLETE if passed is None else (PASS if passed else FAIL)
                checks.append(check(name, status, **values))
            gate = gate_view_report(log)
            if gate is not None:
                checks.append(check("oncoming_gate_view", REPORT, **gate))
        else:
            settled = [r for r in startup if r["truth"]["s_m"] > s0 + 15.0 and r["truth"]["speed_mps"] > 0.3]
            if len(settled) < 100:
                note = {}
                if args.scenario == "stop" and args.obstacle_m < stop_obstacle_m(args.speed):
                    note = dict(note=f"--obstacle-m {args.obstacle_m:g} is short of the {stop_obstacle_m(args.speed):g} m lane keeping needs at {args.speed:g} m/s")
                checks.append(check("lane_keeping", INCOMPLETE, settled_ticks=len(settled), **note))
            else:
                off = np.abs([r["truth"]["lane_offset_m"] for r in settled])
                p95, worst = float(np.percentile(off, 95)), float(off.max())
                checks.append(
                    check("lane_keeping", PASS if p95 <= 0.35 and worst <= 0.70 else FAIL, p95_m=p95, max_m=worst, ticks=len(settled))
                )
            # The whole run, calibration and start included: no corner of the body outside the
            # lane (each corner projected onto the lane centreline itself, so bends are exact).
            half_lane = self.site["lane_width_m"] / 2
            worst_corner = max(r["truth"]["corner_offset_max_m"] for r in log)
            checks.append(
                check("lane_footprint", PASS if worst_corner <= half_lane else FAIL, max_corner_offset_m=worst_corner, half_lane_m=half_lane, ticks=len(log))
            )
        if self.actor is not None:
            moving = score_moving(log, actor_truths, self.body)
            ok = moving["separation_ok"]
            checks.append(check("separation", INCOMPLETE if ok is None else (PASS if ok else FAIL), **moving))
            if getattr(args, "stack", False):
                # Which tracks were the actor, from the logged tracks (third review, F1);
                # the lead's errors are scored only on ticks it was the actor.
                checks.append(check("actor_tracking", REPORT, **score_actor_tracking(log, actor_truths, self.body, DT_S)))
                # Did the predicted regions hold where the actor went (tenth review, 3)?
                checks.append(check("prediction_coverage", REPORT, **score_prediction_coverage(log, actor_truths, self.body, DT_S)))
                if self.actor_scenario in ("braking_lead", "cut_in"):
                    checks.append(check("lead_tracking", REPORT, **score_lead_tracking(log, actor_truths, self.body, DT_S)))
                if self.actor_scenario == "crossing_vehicle":
                    # When the radar, the LiDAR and the stack's restriction each had the car (F6-02).
                    checks.append(check("radar_timeline", REPORT, **score_radar_timeline(log, actor_truths, self.body, args.speed, DT_S)))
                if any(len(actors) > 1 for actors in actor_sets):
                    # Every scripted actor, and whether the tracks kept them apart (F8-05).
                    checks.append(check("multi_actor_tracking", REPORT, **score_multi_actor_tracking(log, actor_sets, self.body, DT_S)))
            # Driving quality beside the gates: what restricted the car, whether it was the
            # actor, and the time it cost (third review, F4).
            checks.append(check("behaviour", REPORT, **score_behaviour(self.actor_name, log, actor_truths, self.body, args.speed, DT_S)))
            # Whether the run reached the transition its scenario is for (fourth review): a
            # pass on a cut-in car that drove off is no evidence for the hand-over.
            exercised = score_exercised(self.actor_name, log, actor_truths, self.body, args.speed, DT_S, params=_params_record(self.actor))
            checks.append(check("exercised", {True: PASS, False: NOT_EXERCISED}.get(exercised.get("exercised"), REPORT), **exercised))
        last = log[-1]
        summary = last["driver"].get("calibration")
        source = last["driver"].get("keep_right_source") or ""
        used = last["driver"].get("keep_right_m")
        if source == "configured":
            checks.append(check("calibration", REPORT, keep_right_used_m=used, source=source, note="--keep-right given"))
        else:
            within = used is not None and abs(used - self.site["kerb_distance_m"]) <= 0.3
            checks.append(
                check(
                    "calibration",
                    PASS if source.startswith("measured") and within else FAIL,
                    keep_right_used_m=used,
                    map_lane_centre_to_kerb_m=self.site["kerb_distance_m"],
                    source=source,
                    summary=summary,
                    calibrating_ticks=sum(1 for r in log if r["mode"] == DriveMode.CALIBRATING.value),
                )
            )
        if self.actor_scenario is not None:
            # What completes the run depends on the scenario (review: a hold is only an
            # endpoint behind a lead that has stopped); the required outcome is reported.
            completion = score_completion(self.actor_scenario, log, actor_truths, self.body, self.report["end"], hold_ticks * DT_S)
            passed = completion.pop("passed")
            checks.append(check("completion", INCOMPLETE if passed is None else (PASS if passed else FAIL), **completion))
        elif args.scenario != "stop":
            done = self.report["end"] == "finished"
            goal = last["truth"]["goal_distance_m"]
            checks.append(check("goal", PASS if done and goal <= 3.0 else FAIL, end=self.report["end"], goal_distance_m=goal))
        else:
            gap = last["truth"]["gap_m"]
            ok = self.report["end"] == "held at the obstacle" and gap is not None and 1.5 <= gap <= 5.0
            checks.append(check("obstacle_stop", PASS if ok else FAIL, end=self.report["end"], gap_m=gap, held_s=hold_ticks * DT_S))
            # How far ahead the obstacle was first seen, and the speed that supports (fifth
            # review, 4.2): the envelope per obstacle size comes from these runs.
            planner = (self.report.get("driver_config") or {}).get("planner") or {}
            seen = observability(log, self.body.front_m, args.speed, decel_mps2=planner.get("decel_mps2", 1.0), margin_m=planner.get("stop_gap_m", 2.5))
            checks.append(check("observability", REPORT, obstacle=args.obstacle_blueprint, **seen))
        top = max(r["truth"]["speed_mps"] for r in log)
        checks.append(check("speed", PASS if top <= args.speed + 0.5 else FAIL, max_mps=top))
        driving = [m for m in modes if m not in (DriveMode.STARTUP, DriveMode.CALIBRATING)]
        faults = [r["reason"] for r in startup if r["mode"] == DriveMode.FAULT.value]
        share = len(faults) / max(1, len(driving))
        reasons = {}
        for reason in faults:
            key = reason.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        checks.append(check("faults", PASS if share <= 0.02 else FAIL, share=share, reasons=reasons))
        following = next((r for name, r in (self.report.get("braking_assumptions") or {}).items() if name.startswith("following")), None)
        semantic_mode = (self.report.get("stack") or {}).get("semantic_mode")
        checks.append(check("driver_step_time", REPORT, **step_time_report(step_times, log, response_s=None if following is None else following["response_s"], semantic_mode=semantic_mode)))
        latency = sensor_to_actuator_report(getattr(self, "latencies", ()))
        if latency is not None:
            checks.append(check("sensor_to_actuator", REPORT, **latency))
            # The response times the safety margins assume against the measured path from
            # sensor to actuator (tenth review, 4: use the measured latency distribution):
            # its p99 plus a tick (a hazard arrives just after a sweep).
            planner_cfg = (self.report.get("driver_config") or {}).get("planner") or {}
            needed_s = round((latency["sensor_to_actuator_ms"]["p99"] + DT_S * 1000.0) / 1000.0, 3)
            assumed = {name: r["response_s"] for name, r in (self.report.get("braking_assumptions") or {}).items()}
            assumed["observability bound (PlannerConfig.sight_response_s)"] = planner_cfg.get("sight_response_s")
            checks.append(check(
                "latency_budget", REPORT, needed_s=needed_s, max_s=round((latency["sensor_to_actuator_ms"]["max"] + DT_S * 1000.0) / 1000.0, 3),
                response_s=assumed, covered={k: bool(v is not None and v >= needed_s) for k, v in assumed.items()},
                # A rule assuming no response time carries the lag in its margin: this much of it at cruise.
                lag_at_cruise_m=round(args.speed * needed_s, 2), stop_gap_m=planner_cfg.get("stop_gap_m"),
            ))
        sight = sight_bound_report(log)
        if sight is not None:
            checks.append(check("sight_bound", REPORT, **sight))
        worker = semantic_worker_report(log)
        if worker is not None:
            checks.append(check("semantic_worker", REPORT, **worker))
        shadow = semantic_shadow_report(log, kerb_distance_m=self.site.get("kerb_distance_m") if isinstance(self.site, dict) else None)
        if shadow is not None:
            checks.append(check("semantic_shadow", REPORT, **shadow))
        edge_errors = []
        for r in startup:
            kerb = r["driver"].get("right_kerb_body")
            if kerb is None:
                continue
            x, detected = kerb
            # Where the map's kerb is, seen from the car: its offset from the lane centre,
            # the car's own offset, and the car's heading relative to the lane over x metres.
            truth = r["truth"]
            true = -(self.site["kerb_distance_m"] + truth["lane_offset_m"]) - x * math.sin(truth["heading_error_rad"])
            edge_errors.append(detected - true)
        checks.append(
            check(
                "kerb_bias",
                REPORT,
                lanes_between=self.site["right_lanes"],
                detected_kerb_bias_m=float(np.median(edge_errors)) if edge_errors else None,
                note="bias > 0: the LiDAR kerb is found closer to the car than the map's.",
            )
        )


def _config_dict(config):
    from dataclasses import asdict

    return asdict(config)


def main(argv=None, stack=False):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenario", choices=LANE_SCENARIOS + tuple(TURN_SCENARIOS) + tuple(ACTOR_SCENARIOS), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New folder for the run")
    parser.add_argument("--map", default=None, help="Load this map first (default: the server's current map)")
    parser.add_argument("--site", type=int, default=0, help="Which suitable site on the map (0 = first)")
    parser.add_argument("--suite", type=Path, default=Path("configs/sensor_suites/mapless_v2.json"))
    parser.add_argument("--set", action="append", default=[], metavar="SENSOR.FIELD=VALUE")
    parser.add_argument("--no-noise", action="store_true", help="Run the suite without its sensor noise")
    parser.add_argument("--speed", type=float, default=2.0, help="Cruise speed (m/s)")
    parser.add_argument(
        "--keep-right",
        type=float,
        default=None,
        help="Lane centre's distance left of the kerb (m); default: measured at start-up",
    )
    parser.add_argument("--goal-m", type=float, default=60.0, help="Route length (m); the site needs this + 15 m without a junction")
    parser.add_argument(
        "--obstacle-m",
        type=float,
        default=None,
        help="stop: parked car this far along the lane (default: 35 m, or as far as lane keeping needs for its 100 settled ticks at --speed)",
    )
    parser.add_argument(
        "--obstacle-blueprint",
        default=PARKED_CAR,
        help="stop: what stands in the lane (a smaller one, e.g. static.prop.trafficcone01 or static.prop.box01, for the observability envelope)",
    )
    parser.add_argument("--approach-m", type=float, default=40.0, help="turns: approach lane length before the junction")
    parser.add_argument("--exit-m", type=float, default=30.0, help="turns: exit lane length after the junction (the goal)")
    parser.add_argument("--max-kerb-mismatch", type=float, default=0.3, help="turns: exit kerb distance may differ from the approach's by this (m)")
    parser.add_argument("--junction-hold-s", type=float, default=25.0, help="turns: end the run after waiting this long at the junction mouth")
    parser.add_argument("--actor-param", action="append", default=[], metavar="NAME=VALUE", help="moving actors: scenario parameter override")
    parser.add_argument(
        "--oncoming",
        choices=("none", "stopped", "parked"),
        default="none",
        help="turn_left: an oncoming car across the junction, stopped mid-lane and pulling away, or parked at its kerb (M08)",
    )
    parser.add_argument("--admission", choices=("A", "B", "C"), default="A", help="stack: sensor admission preset (A camera+LiDAR, B +front radar, C +rear radars)")
    parser.add_argument("--segmenter", default=None, help="stack: road segmenter checkpoint (default: constant)")
    parser.add_argument("--detector", default=None, help="stack: 'yolo' or none")
    parser.add_argument(
        "--semantic",
        choices=("thread", "inline", "off"),
        default="thread",
        help="stack: segmentation, painting and the road model on their own worker thread, off the control path (default), inline in each step, or off (the geometric stack alone)",
    )
    parser.add_argument("--hold-s", type=float, default=5.0)
    parser.add_argument("--max-s", type=float, default=150.0)
    parser.add_argument("--snapshot-every", type=int, default=40, help="Ticks between terrain snapshots (0: none)")
    parser.add_argument("--follow", action="store_true", help="Keep the CARLA spectator camera 8 m behind and 5 m above the car")
    parser.add_argument("--seed", type=int, default=5308)
    parser.add_argument(
        "--drop-sensor",
        action="append",
        default=[],
        metavar="NAME:START:END",
        help="inject an outage: sensor NAME (roof, radar_front, gnss, imu, ...) missing from START to END s after the first frame",
    )
    parser.add_argument("--cpu-load", type=int, default=0, metavar="N", help="deliberate contention: N processes each spinning a core during the run (tenth review, 4)")
    parser.add_argument("--gpu-load", action="store_true", help="deliberate contention: back-to-back 2048x2048 matrix products on the CUDA device during the run")
    parser.add_argument(
        "--record-frames",
        choices=("none", "turns", "all"),
        default="turns",
        help="Record the sensor frames (LiDAR as int16 cm, GNSS, IMU, speed, status) under <output>/frames for an offline replay: ~90 KB a tick",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args(argv)
    args.stack = stack
    args.actor_params = {}
    for item in args.actor_param:
        name, _, value = item.partition("=")
        args.actor_params[name] = float(value)
    if args.output.exists():
        parser.error("Output folder exists; choose a new one")
    if args.segmenter not in (None, "constant") and not Path(args.segmenter).is_file():
        parser.error(f"--segmenter {args.segmenter} not found (checkpoints are not in git; copy it from the PC that trained it, or omit --segmenter)")
    if args.obstacle_m is None:
        args.obstacle_m = stop_obstacle_m(args.speed)
    if args.scenario == "stop" and not 15 <= args.obstacle_m <= args.goal_m - 10:
        parser.error("--obstacle-m must be at least 15 m and 10 m short of the goal")
    if args.oncoming != "none" and args.scenario != "turn_left":
        parser.error("--oncoming is for turn_left only (the oncoming lane a left turn crosses)")
    configured = load_sensor_suite(args.suite).with_overrides(parse_overrides(args.set))
    suite = configured.without_noise() if args.no_noise else configured
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    if client.get_server_version() != "0.9.16" or client.get_client_version() != "0.9.16":
        raise RuntimeError("CARLA 0.9.16 client and server required")
    world = client.get_world()
    if world.get_settings().synchronous_mode:
        raise RuntimeError("Another client owns synchronous ticks; close it first")
    if args.map:
        world = ensure_map(client, args.map, log=lambda text: print(text, flush=True))
    if any(a.type_id.startswith(("vehicle.", "walker.", "sensor.")) for a in world.get_actors()):
        raise RuntimeError("Use an otherwise empty CARLA world")
    args.output.mkdir(parents=True)
    initial_settings = _settings_record(world.get_settings())
    report = dict(
        purpose=("mapless stack (G2/G3)" if stack else "G1 mapless drive") + ": sensors and a sat-nav route only",
        scenario=args.scenario,
        server=client.get_server_version(),
        map=world.get_map().name,
        suite_name=configured.name,
        suite_sha256=suite.sha256,
        # The code and the args that produced this result (review of 405b056: record the
        # commit, configuration and checkpoint with every run).
        source_commit=source_commit(),
        argv=list(argv) if argv is not None else sys.argv[1:],
        source_root=str(Path(__file__).resolve().parents[2]),  # the checkout actually imported
        overrides=parse_overrides(args.set),
        noise="disabled" if args.no_noise else "as configured",
        seed=args.seed,  # the rig's noise seed, default or not (ninth review: map, site, seed and parameters with every run)
        injected_outages=[dict(sensor=n, start_s=a, end_s=b) for n, a, b in parse_outages(args.drop_sensor)],
        status="INCOMPLETE",
        checks=[],
    )
    # On the console before anything else: which map the server is on (it keeps the last
    # one loaded) and which code is running (26 Oct: PC 2's runs were on Town05, and runs
    # started the minute a fix was pushed could not show whether they had it).
    print(f"map {report['map']}  code {report['source_commit'] or 'unknown commit'} at {report['source_root']}", flush=True)
    try:
        with LoadGenerator(args.cpu_load, args.gpu_load) as load, SynchronousWorldSession(client, WorldTiming(fixed_delta_seconds=DT_S)) as owner:
            report["load"] = load.note
            MaplessDrive(carla, owner, suite, args, report).run()
        report["settings_restored"] = _settings_record(client.get_world().get_settings()) == initial_settings
        report["status"] = run_status(report["checks"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if not report["checks"] and not any(args.output.iterdir()):
            # Setup failed before anything was written (no site, map not loaded): leave no
            # empty folder behind, so the same --output can be tried again.
            args.output.rmdir()
            print(f"status: {report['status']}")
            raise
        raise
    finally:
        if args.output.exists():
            (args.output / "drive_report.json").write_text(json.dumps(rounded(report), indent=2, default=str), encoding="utf-8")
            print("\n".join([f"{c['status']:10}  {c['check']}" for c in report["checks"]] + [f"status: {report['status']}"]))
