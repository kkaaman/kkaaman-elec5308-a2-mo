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
- **Scoring** (truth): lane-centre offset of the rear axle, heading error, bumper gap to
  the parked car, collisions (collision sensor, harness only), goal distance, and the
  route-pose error of the car's GNSS/compass filter.

**Gates** (``drive_report.json``):

| Check | Pass |
|---|---|
| Start-up | The car localised itself from GNSS and compass within 3 s, with no spawn pose |
| No collision | No collision event |
| Lane keeping | After the first 15 m: offset p95 ≤ 0.35 m and max ≤ 0.70 m |
| Goal (straight, curve) | FINISHED, rear axle within 3 m of the goal |
| Obstacle stop (stop) | Standing still, bumper gap 1.5 to 5.0 m, held for ``--hold-s`` |
| Speed | Never above the cruise speed + 0.5 m/s |
| Faults | Under 2% of driving ticks in FAULT |

Reported, not gated: driver step time, the start-up route-pose error, and the kerb
calibration (map distance from lane centre to kerb, against what the LiDAR found). The
calibration says what ``--keep-right`` should be.

Outputs: ``drive_report.json``, ``drive_log.jsonl`` (one line per tick), ``route.json``,
``drive_plot.png`` and, every ``--snapshot-every`` ticks, a top-down image of the terrain
grid and corridor in ``snapshots/``.
"""

import argparse
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np

from carla_adapter.sensor_rig import SensorRig, true_forward_speed
from carla_adapter.sensor_rig_check import rear_axle_pose
from carla_adapter.vehicle import FixedVehicleAdapter
from carla_adapter.world import SynchronousWorldSession, WorldTiming, _settings_record, ensure_map
from mapless.corridor import CorridorConfig
from mapless.driver import DriveMode, DriverConfig, MaplessDriver
from mapless.obstacles import project_onto_polyline
from mapless.planner import PlannerConfig
from mapless.satnav import Command, RoutePoint, SatNavRoute
from mapless.sensor_suite import load_sensor_suite, parse_overrides
from mapless.terrain import CellLabel, VehicleBody
from navigation_control.local import LocalControlConfig
from navigation_control.types import ControllerConfig

DT_S = 0.05
PARKED_CAR = "vehicle.audi.a2"
SETTLE_TICKS = 20  # brake and let the rig settle (IMU spike, spawn delay) before the car starts
PASS, FAIL, INCOMPLETE, REPORT = "PASS", "FAIL", "INCOMPLETE", "REPORT"


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


def find_sites(carla, carla_map, scenario, length_m, junction_clearance_m=8.0, slide_m=10.0):
    """Sites on junction-free lane runs from the map topology, and a census of why others failed.

    Each topology segment between junctions is walked from ``junction_clearance_m`` past its
    start (away from the kerb corners of the junction mouth). A window of ``length_m`` is
    slid along the run in ``slide_m`` steps until one has the scenario's shape.
    """
    census = dict(segments=0, not_driving_or_junction=0, not_rightmost=0, too_short=0, wrong_shape=0, kerb_gaps=0, sites=0)
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
        found = None
        for offset in range(0, len(usable) - int(length_m), int(slide_m)):
            window = usable[offset : offset + int(length_m) + 1]
            ok, total, max_rate = shape_ok(scenario, window)
            if ok:
                found = (offset, window, total, max_rate)
                break
        if found is None:
            census["wrong_shape"] += 1
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


def make_route(carla_map, waypoints, goal_m, spacing_m=25.0, name="g1b"):
    stations = list(np.arange(0.0, goal_m, spacing_m)) + [goal_m]
    points = []
    for s in stations:
        wp = waypoints[min(len(waypoints) - 1, int(round(s)))]
        fix = carla_map.transform_to_geolocation(wp.transform.location)
        points.append(RoutePoint(fix.latitude, fix.longitude, Command.LANEFOLLOW))
    return SatNavRoute(name, tuple(points), provenance=dict(source="map lane centre", spacing_m=spacing_m))


# -- live helpers ------------------------------------------------------------------------
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

    def setup(self):
        carla, args = self.carla, self.args
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        carla_map = self.world.get_map()
        length = args.goal_m + 15.0
        sites, census = find_sites(carla, carla_map, args.scenario, length)
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
        self.lane = np.array([core_xy(w.transform.location) for w in wps])
        self.adapter = FixedVehicleAdapter(self.ego)
        self.offset = self.adapter.geometry.rear_axle_offset_m
        self.parked = None
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
        self.goal_xy = self.lane[int(round(args.goal_m))]
        self.report["site"] = rounded(
            {k: v for k, v in site.items() if k != "waypoints"} | dict(map=carla_map.name, start_xy_m=self.lane[0].tolist())
        )

    def truth(self):
        t = self.ego.get_transform()
        x, y, _, yaw = rear_axle_pose(t, self.offset)
        s, lateral, outside = project_onto_polyline(np.array([[x, y]]), self.lane)
        i = int(np.clip(round(float(s[0])), 0, len(self.lane) - 2))
        heading = math.atan2(*(self.lane[i + 1] - self.lane[i])[::-1])
        gap = None
        if self.parked is not None:
            p = core_xy(self.parked.get_transform().location)
            ps, _, _ = project_onto_polyline(np.array([p]), self.lane)
            gap = float(ps[0]) - self.parked.bounding_box.extent.x - (float(s[0]) + self.body.front_m)
        v = self.ego.get_velocity()
        return dict(
            rear_axle_xy=[x, y],
            yaw_rad=yaw,
            s_m=float(s[0]),
            lane_offset_m=float(lateral[0]),
            heading_error_rad=wrap(yaw - heading),
            speed_mps=math.hypot(v.x, v.y),
            goal_distance_m=float(math.dist((x, y), self.goal_xy)),
            gap_m=gap,
        )

    def run(self):
        carla, args = self.carla, self.args
        self.setup()
        self.body = vehicle_body(self.ego, self.offset)
        controller = ControllerConfig(
            wheelbase_m=self.adapter.geometry.wheelbase_m,
            max_steering_angle_rad=self.adapter.geometry.controller_steering_limit_rad,
            lookahead_m=4.0,
            kp=0.45,
            ki=0.15,
            stopping_deceleration_mps2=1.0,
            hold_brake=0.5,
        )
        config = DriverConfig(
            body=self.body,
            control=LocalControlConfig(controller),
            corridor=CorridorConfig(keep_right_m=args.keep_right),
            planner=PlannerConfig(cruise_mps=args.speed),
        )
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
        log, step_times, modes = [], [], []
        hold_ticks = 0
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
            driver = MaplessDriver(self.suite, self.route, config, DT_S)
            start_time = None
            for tick in range(int(args.max_s / DT_S)):
                frame = self.owner.tick()
                snap = self.world.get_snapshot()
                sf = rig.read(frame, true_forward_speed(snap, self.ego.id), sim_time_s=snap.timestamp.elapsed_seconds)
                start_time = sf.sim_time_s if start_time is None else start_time
                begun = time.perf_counter()
                step = driver.step(sf)
                step_times.append(time.perf_counter() - begun)
                self.adapter.apply(step.command)
                truth = self.truth()
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
                    render_snapshot(driver, snapshots / f"t{tick:05d}.png")
                if tick % 20 == 0:
                    d = step.diagnostics
                    print(
                        f"t={row['time_s']:6.1f}s s={truth['s_m']:6.1f}m off={truth['lane_offset_m']:+.2f}m"
                        f" v={truth['speed_mps']:.2f} {step.mode.value:9} {step.reason[:40]:40}"
                        f" corridor={d.get('corridor_valid_m')} obstacle={d.get('obstacle_m')}",
                        flush=True,
                    )
                if collisions.events:
                    end = "collision"
                    break
                if abs(truth["lane_offset_m"]) > 3.0:
                    end = "left the lane (more than 3 m off centre)"
                    break
                if step.mode is DriveMode.FINISHED and truth["speed_mps"] < 0.05:
                    finished_ticks += 1
                    if finished_ticks >= 20:
                        end = "finished"
                        break
                held = step.mode is DriveMode.STOPPING and step.reason.startswith("holding") and truth["speed_mps"] < 0.05
                hold_ticks = hold_ticks + 1 if held else 0
                if args.scenario == "stop" and hold_ticks * DT_S >= args.hold_s:
                    end = "held at the obstacle"
                    break
            self.adapter.stop()
            self.owner.tick()
        self.report["end"] = end
        self.report["collisions"] = collisions.events
        self.report["rig"] = rig.record()
        plot(log, args.output / "drive_plot.png", args.speed)
        self.score(log, step_times, modes, hold_ticks)

    def score(self, log, step_times, modes, hold_ticks):
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
            true = np.array(first["truth"]["rear_axle_xy"]) - self.lane[0]
            start_error = float(np.hypot(*(np.array(pose[:2]) - true)))
        checks.append(check("startup", PASS if start_s <= 3.0 else FAIL, seconds=start_s, route_pose_error_m=start_error))
        checks.append(check("no_collision", PASS if not self.report["collisions"] else FAIL, events=self.report["collisions"]))
        s0 = first["truth"]["s_m"]
        settled = [r for r in startup if r["truth"]["s_m"] > s0 + 15.0 and r["truth"]["speed_mps"] > 0.3]
        if len(settled) < 100:
            checks.append(check("lane_keeping", INCOMPLETE, settled_ticks=len(settled)))
        else:
            off = np.abs([r["truth"]["lane_offset_m"] for r in settled])
            p95, worst = float(np.percentile(off, 95)), float(off.max())
            checks.append(
                check("lane_keeping", PASS if p95 <= 0.35 and worst <= 0.70 else FAIL, p95_m=p95, max_m=worst, ticks=len(settled))
            )
        last = log[-1]
        if args.scenario in ("straight", "curve"):
            done = self.report["end"] == "finished"
            goal = last["truth"]["goal_distance_m"]
            checks.append(check("goal", PASS if done and goal <= 3.0 else FAIL, end=self.report["end"], goal_distance_m=goal))
        else:
            gap = last["truth"]["gap_m"]
            ok = self.report["end"] == "held at the obstacle" and gap is not None and 1.5 <= gap <= 5.0
            checks.append(check("obstacle_stop", PASS if ok else FAIL, end=self.report["end"], gap_m=gap, held_s=hold_ticks * DT_S))
        top = max(r["truth"]["speed_mps"] for r in log)
        checks.append(check("speed", PASS if top <= args.speed + 0.5 else FAIL, max_mps=top))
        driving = [m for m in modes if m is not DriveMode.STARTUP]
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
        for r in settled:
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
                "kerb_calibration",
                REPORT,
                map_lane_centre_to_kerb_m=self.site["kerb_distance_m"],
                lanes_between=self.site["right_lanes"],
                keep_right_used_m=args.keep_right,
                detected_kerb_bias_m=float(np.median(edge_errors)) if edge_errors else None,
                note="bias > 0: the LiDAR kerb is found closer to the car than the map's. Suggested --keep-right: the map distance minus the bias.",
            )
        )


def _config_dict(config):
    from dataclasses import asdict

    return asdict(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenario", choices=("straight", "curve", "stop"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New folder for the run")
    parser.add_argument("--map", default=None, help="Load this map first (default: the server's current map)")
    parser.add_argument("--site", type=int, default=0, help="Which suitable site on the map (0 = first)")
    parser.add_argument("--suite", type=Path, default=Path("configs/sensor_suites/mapless_v2.json"))
    parser.add_argument("--set", action="append", default=[], metavar="SENSOR.FIELD=VALUE")
    parser.add_argument("--no-noise", action="store_true", help="Run the suite without its sensor noise")
    parser.add_argument("--speed", type=float, default=2.0, help="Cruise speed (m/s)")
    parser.add_argument("--keep-right", type=float, default=1.75, help="Lane centre's distance left of the kerb (m)")
    parser.add_argument("--goal-m", type=float, default=60.0, help="Route length (m); the site needs this + 15 m without a junction")
    parser.add_argument("--obstacle-m", type=float, default=35.0, help="stop: parked car this far along the lane")
    parser.add_argument("--hold-s", type=float, default=5.0)
    parser.add_argument("--max-s", type=float, default=150.0)
    parser.add_argument("--snapshot-every", type=int, default=40, help="Ticks between terrain snapshots (0: none)")
    parser.add_argument("--seed", type=int, default=5308)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output folder exists; choose a new one")
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
        purpose="G1b first mapless drive: sensors and a sat-nav route only",
        scenario=args.scenario,
        server=client.get_server_version(),
        map=world.get_map().name,
        suite_name=configured.name,
        suite_sha256=suite.sha256,
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
        raise
    finally:
        (args.output / "drive_report.json").write_text(json.dumps(rounded(report), indent=2, default=str), encoding="utf-8")
        print("\n".join([f"{c['status']:10}  {c['check']}" for c in report["checks"]] + [f"status: {report['status']}"]))
