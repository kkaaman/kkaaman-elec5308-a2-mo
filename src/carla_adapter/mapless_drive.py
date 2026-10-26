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
- **Moving actors** (``crossing_pedestrian``, ``braking_lead``, ``cut_in``; M04–M06,
  framework): scripted by ``carla_adapter.mapless_scenarios``, scored on separation.
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

Reported, not gated: driver step time, the start-up route-pose error, and the kerb
bias (where the LiDAR found the kerb against where the map has it).

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
from carla_adapter.mapless_scenarios import make_scenario, score_completion, score_moving
from carla_adapter.sensor_rig import SensorRig, true_forward_speed
from carla_adapter.sensor_rig_check import rear_axle_pose
from carla_adapter.vehicle import FixedVehicleAdapter
from carla_adapter.world import SynchronousWorldSession, WorldTiming, _settings_record, ensure_map
from evaluation.junction_truth import TurnSite, score_turn, truth_sample
from mapless.corridor import CorridorConfig
from mapless.driver import DriveMode, DriverConfig, MaplessDriver
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


def find_turn_sites(carla, carla_map, command, approach_m, exit_m, junction_clearance_m=8.0, angle_tolerance_deg=30.0, max_kerb_mismatch_m=0.3, max_approach_bend_deg=5.0, max_slope_deg=3.0):
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
      skews it.
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
            raise ValueError("Vehicle tilt exceeds planar adapter scope")
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
            self.parked = self.owner.try_spawn(self.library.find(PARKED_CAR), t)
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
                stack_config = StackConfig(admission=Admission.preset(args.admission), segmenter=args.segmenter, detector=args.detector)
                driver = MaplessStack(self.suite, self.route, config, stack_config, DT_S)
                self.report["stack"] = dict(
                    admission=args.admission,
                    segmenter=args.segmenter or "constant",
                    segmenter_sha256=_file_sha256(args.segmenter) if args.segmenter not in (None, "constant") else None,
                    detector=args.detector or "none",
                    stack_config_sha256=hashlib.sha256(json.dumps(dataclasses.asdict(stack_config), sort_keys=True, default=str).encode()).hexdigest(),
                )
            else:
                driver = MaplessDriver(self.suite, self.route, config, DT_S)
            start_time = None
            actor_truths = []
            for tick in range(int(args.max_s / DT_S)):
                frame = self.owner.tick()
                snap = self.world.get_snapshot()
                if args.follow:
                    self._follow_with_spectator()
                sf = rig.read(frame, true_forward_speed(snap, self.ego.id), sim_time_s=snap.timestamp.elapsed_seconds)
                start_time = sf.sim_time_s if start_time is None else start_time
                if recorder is not None:
                    recorder.record(tick, sf)
                begun = time.perf_counter()
                step = driver.step(sf)
                step_times.append(time.perf_counter() - begun)
                try:
                    self.adapter.apply(step.command)
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
                )
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
            (args.output / "decisions.jsonl").write_text("\n".join(json.dumps(rounded(r.as_dict())) for r in driver.records), encoding="utf-8")
        plot(log, args.output / "drive_plot.png", args.speed)
        self.score(log, step_times, modes, hold_ticks, actor_truths)

    def score(self, log, step_times, modes, hold_ticks, actor_truths=()):
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
        else:
            settled = [r for r in startup if r["truth"]["s_m"] > s0 + 15.0 and r["truth"]["speed_mps"] > 0.3]
            if len(settled) < 100:
                checks.append(check("lane_keeping", INCOMPLETE, settled_ticks=len(settled)))
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
        if self.actor is not None:
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
        times = np.array(step_times) * 1000
        checks.append(check("driver_step_time", REPORT, p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95))))
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
    parser.add_argument("--obstacle-m", type=float, default=35.0, help="stop: parked car this far along the lane")
    parser.add_argument("--approach-m", type=float, default=40.0, help="turns: approach lane length before the junction")
    parser.add_argument("--exit-m", type=float, default=30.0, help="turns: exit lane length after the junction (the goal)")
    parser.add_argument("--max-kerb-mismatch", type=float, default=0.3, help="turns: exit kerb distance may differ from the approach's by this (m)")
    parser.add_argument("--junction-hold-s", type=float, default=25.0, help="turns: end the run after waiting this long at the junction mouth")
    parser.add_argument("--actor-param", action="append", default=[], metavar="NAME=VALUE", help="moving actors: scenario parameter override")
    parser.add_argument("--admission", choices=("A", "B", "C"), default="A", help="stack: sensor admission preset (A camera+LiDAR, B +front radar, C +rear radars)")
    parser.add_argument("--segmenter", default=None, help="stack: road segmenter checkpoint (default: constant)")
    parser.add_argument("--detector", default=None, help="stack: 'yolo' or none")
    parser.add_argument("--hold-s", type=float, default=5.0)
    parser.add_argument("--max-s", type=float, default=150.0)
    parser.add_argument("--snapshot-every", type=int, default=40, help="Ticks between terrain snapshots (0: none)")
    parser.add_argument("--follow", action="store_true", help="Keep the CARLA spectator camera 8 m behind and 5 m above the car")
    parser.add_argument("--seed", type=int, default=5308)
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
    if args.scenario == "stop" and not 15 <= args.obstacle_m <= args.goal_m - 10:
        parser.error("--obstacle-m must be at least 15 m and 10 m short of the goal")
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
        overrides=parse_overrides(args.set),
        noise="disabled" if args.no_noise else "as configured",
        status="INCOMPLETE",
        checks=[],
    )
    try:
        with SynchronousWorldSession(client, WorldTiming(fixed_delta_seconds=DT_S)) as owner:
            MaplessDrive(carla, owner, suite, args, report).run()
        report["settings_restored"] = _settings_record(client.get_world().get_settings()) == initial_settings
        gated = [c for c in report["checks"] if c["status"] != REPORT]
        if any(c["status"] == FAIL for c in gated):
            report["status"] = "MEASURED_FAIL"
        elif any(c["status"] == INCOMPLETE for c in gated) or not gated:
            report["status"] = "INCOMPLETE"
        else:
            report["status"] = "MEASURED_PASS"
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
