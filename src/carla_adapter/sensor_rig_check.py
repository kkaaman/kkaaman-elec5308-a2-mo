"""Live calibration gate for the mapless sensor rig (plan v2.1, section 5).

The rig runs noise-free by default, so every check has a tight tolerance; ``--keep-noise``
uses the suite's noise. ``--set sensor.field=value`` varies the suite, for example in the
radar density-against-FOV experiment.

Truth (actor poses, boxes and velocities) and the instance-segmentation cameras are read
here only to score the rig. They never reach the ``SensorFrame``.

**Site.** A straight road at least 45 m long, with driving lanes on both sides of the
ego lane where the map has them. The ego stays in its lane throughout.

**Phase 1, stationary, no targets:**
- every sensor delivers every tick;
- each LiDAR sweep covers 360°, with no points on the ego body;
- no radar returns from the car itself;
- the IMU reads gravity at rest, the compass agrees with the true heading, and GNSS
  fixes arrive at the configured rate.

**Phase 2, coverage sweep.** A parked car is placed alone at each placement: ego lane
ahead and behind, and each side lane alongside, ahead and behind. Box props follow for
pixel accuracy. For each placement the check records:
- LiDAR points on the true box, against its mirror image (catches sign errors);
- per camera: whether the target is seen, the share of its LiDAR points that land on its
  instance mask, and the silhouette edge error in pixels;
- per radar: hits, and range error along each return's ray.

This gives the coverage table and blind-zone plot.

**Phase 3, radar motion:**
- stationary ego with a target driving away;
- ego and a lead car at the same speed (range rate about 0).

**Phase 4, driving** (straight, gentle left curve, stop):
- GNSS displacement in the car's own tangent plane;
- speedometer;
- gyro yaw rate;
- accelerometer sign;
- radar sign and range rate on static-world returns, with and without the lever arm
  ``omega x r``.

**Gates:**

| Check | Pass |
|---|---|
| LiDAR extrinsics | Every measured placement has 30+ box points, and under 20% of that count in the mirrored box |
| Mask consistency | Each camera ≥ 0.85 |
| Prop reprojection | Edge error p95 ≤ 3 px over at least 8 edges |
| Radar range | Error p95 ≤ 0.5 m at 5–40 m |
| Radar range rate | Error p95 ≤ 0.5 m/s |
| Radar sign | Correct in every case |
| Cleanup | No leaked actors; settings restored |

Car silhouettes are reported, not gated, because a car is not a box.

``rig_check.json`` status:
- MEASURED_PASS: every gated check passes;
- MEASURED_FAIL: any gated check fails;
- INCOMPLETE: there was too little evidence to decide a check.
"""

import argparse
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np

from carla_adapter.sensor_rig import (
    RADAR_VELOCITY_IS_RANGE_RATE,
    SensorRig,
    carla_mount,
    lidar_azimuth_coverage_deg,
    true_forward_speed,
)
from carla_adapter.vehicle import FixedVehicleAdapter
from carla_adapter.world import SynchronousWorldSession, WorldTiming, _settings_record
from evaluation import rig_calibration as cal
from evaluation.dataset import decode_bgra
from mapless.frame import SensorState
from mapless.geometry import wrap, yaw_from_compass
from mapless.satnav import LocalTangentPlane
from mapless.sensor_suite import load_sensor_suite, parse_overrides

DT_S = 0.05
TARGET_CAR = "vehicle.audi.a2"
TARGET_PROP = "static.prop.box01"
PASS, FAIL, INCOMPLETE, REPORT = "PASS", "FAIL", "INCOMPLETE", "REPORT"
GROUND_CLEARANCE_M = 0.2  # box points must sit this far above the box bottom (not road)
IMU_SETTLE_TICKS = 5  # CARLA's accelerometer spikes for a few ticks after the IMU spawns


# -- truth helpers (evaluator side) ---------------------------------------------------
def rear_axle_pose(transform, rear_axle_offset_m):
    loc, yaw = transform.location, -math.radians(transform.rotation.yaw)
    return (
        loc.x - rear_axle_offset_m * math.cos(yaw),
        -loc.y - rear_axle_offset_m * math.sin(yaw),
        loc.z,
        yaw,
    )


def world_to_ego(points_core, pose):
    """World (right-handed) points to the vehicle frame of ``pose`` (x, y, z, yaw)."""
    x, y, z, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(points_core, dtype=float) - (x, y, z)
    return np.column_stack((c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1], p[:, 2]))


def ego_to_world_xy(pose, x_ego, y_ego):
    x, y, _, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * x_ego - s * y_ego, y + s * x_ego + c * y_ego


def to_ego_velocity(vector, pose):
    """A CARLA world vector (left-handed) as vehicle-frame (x, y)."""
    vx, vy = vector.x, -vector.y
    c, s = math.cos(pose[3]), math.sin(pose[3])
    return c * vx + s * vy, -s * vx + c * vy


def target_box_ego(actor, ego_pose):
    corners = [(v.x, -v.y, v.z) for v in actor.bounding_box.get_world_vertices(actor.get_transform())]
    centre = world_to_ego([np.mean(corners, axis=0)], ego_pose)[0]
    yaw = wrap(-math.radians(actor.get_transform().rotation.yaw) - ego_pose[3])
    e = actor.bounding_box.extent
    return centre, yaw, np.array([e.x, e.y, e.z])


def percentiles(values_s):
    if not values_s:
        return None
    ms = np.asarray(values_s) * 1000.0
    return dict(p50_ms=float(np.percentile(ms, 50)), p95_ms=float(np.percentile(ms, 95)), max_ms=float(ms.max()))


def check(name, status, **values):
    if isinstance(status, (bool, np.bool_)):
        status = PASS if status else FAIL
    if status not in (PASS, FAIL, INCOMPLETE, REPORT):
        raise ValueError(f"Unknown check status {status!r}")
    return dict(check=name, status=status, **values)


def rounded(values, digits=3):
    return [round(float(v), digits) for v in values]


class InstanceCameras:
    """Evaluator-only instance-segmentation cameras on the RGB cameras' exact mounts."""

    def __init__(self, owner, vehicle, suite):
        self.owner, self.vehicle, self.suite = owner, vehicle, suite
        self.sensors, self.errors, self._data = [], [], {}

    def __enter__(self):
        import carla

        library = self.owner.world.get_blueprint_library()
        for spec in self.suite.cameras:
            bp = library.find("sensor.camera.instance_segmentation")
            for key, value in dict(
                image_size_x=spec.width, image_size_y=spec.height, fov=spec.fov_deg, sensor_tick=0.0, lens_k=0.0, lens_kcube=0.0
            ).items():
                if bp.has_attribute(key):
                    bp.set_attribute(key, str(value))
            m = carla_mount(spec.mount)
            sensor = self.owner.world.spawn_actor(
                bp,
                carla.Transform(carla.Location(x=m["x"], y=m["y"], z=m["z"]), carla.Rotation(pitch=m["pitch"], yaw=m["yaw"], roll=0.0)),
                attach_to=self.vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
            self.sensors.append(sensor)
            self.owner._actors.append(sensor)
            sensor.listen(lambda image, name=spec.name: self._receive(name, image))
        return self

    def _receive(self, name, image):
        try:
            bgra = decode_bgra(image.raw_data, image.width, image.height)
            ids = bgra[:, :, 1].astype(np.int32) * 256 + bgra[:, :, 0]
            tags = bgra[:, :, 2].astype(np.int32)  # CARLA semantic tag in the red channel
            self._data.setdefault(image.frame, {})[name] = (ids, tags)
            for old in [f for f in self._data if f < image.frame - 8]:
                del self._data[old]
        except Exception as error:
            self.errors.append(f"{name}: {error}")

    def read(self, frame, timeout_s=5.0):
        names = {s.name for s in self.suite.cameras}
        deadline = time.monotonic() + timeout_s
        while not names <= set(self._data.get(frame, {})):
            if time.monotonic() > deadline:
                raise TimeoutError(f"Missing instance images for frame {frame}")
            time.sleep(0.002)
        return self._data.pop(frame)

    def __exit__(self, *exc):
        for sensor in reversed(self.sensors):
            try:
                if sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            except Exception as error:
                self.errors.append(str(error))
        self.sensors.clear()
        return False


# -- site selection -------------------------------------------------------------------
def candidate_sites(carla, carla_map, spawn_points, length_m=45.0):
    """Straight spawn sites, best first: driving lanes on both sides, then on one side."""
    both, one = [], []
    for spawn in spawn_points:
        wp = carla_map.get_waypoint(spawn.location, project_to_road=True)
        if wp is None or wp.is_junction:
            continue
        sides = []
        for side in (wp.get_left_lane(), wp.get_right_lane()):
            sides.append(side if side is not None and side.lane_type == carla.LaneType.Driving else None)
        if not any(sides):
            continue
        ahead, ok, heading = wp, True, wp.transform.rotation.yaw
        for _ in range(int(length_m / 5)):
            nxt = ahead.next(5.0)
            if not nxt or nxt[0].is_junction:
                ok = False
                break
            ahead = nxt[0]
            if abs(wrap(math.radians(ahead.transform.rotation.yaw - heading))) > math.radians(3):
                ok = False
                break
        behind = wp.previous(15.0)
        if not ok or not behind or behind[0].is_junction:
            continue
        (both if all(sides) else one).append((wp, sides[0], sides[1]))
    yield from both
    yield from one


def lane_offset(wp, side):
    """Lateral distance (m) from the ego lane centre to a side lane's centre."""
    a, b = wp.transform.location, side.transform.location
    return math.hypot(a.x - b.x, a.y - b.y)


# -- the check -------------------------------------------------------------------------
class RigCheck:
    def __init__(self, carla, owner, suite, args, report):
        self.carla, self.owner, self.world = carla, owner, owner.world
        self.suite, self.args, self.report = suite, args, report
        self.checks = report["checks"]
        self.timing = dict(tick=[], wait=[], last_arrival=[])
        self.library = self.world.get_blueprint_library()

    # setup
    def setup(self):
        carla = self.carla
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        carla_map = self.world.get_map()
        for wp, left, right in candidate_sites(carla, carla_map, carla_map.get_spawn_points()):
            spawn = wp.transform
            spawn.location.z += 0.3
            ego = self.owner.try_spawn(self.library.find("vehicle.tesla.model3"), spawn)
            if ego is not None:
                break
        else:
            raise RuntimeError("No free straight site with a driving side lane on this map")
        self.ego, self.lane = ego, wp
        self.side = {
            "left": None if left is None else lane_offset(wp, left),
            "right": None if right is None else lane_offset(wp, right),
        }
        ego.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(30):
            self.owner.tick()
        self.offset = FixedVehicleAdapter(ego).geometry.rear_axle_offset_m
        box = ego.bounding_box
        self.ego_centre = np.array([box.location.x + self.offset, -box.location.y, box.location.z])
        self.ego_extent = np.array([box.extent.x, box.extent.y, box.extent.z])
        self.front_x = self.offset + box.location.x + box.extent.x
        self.rear_x = self.offset + box.location.x - box.extent.x
        self.centre_x = self.offset + box.location.x
        self.report["site"] = dict(
            road_id=wp.road_id, lane_id=wp.lane_id, side_lane_offsets_m=self.side, rear_axle_offset_m=self.offset
        )

    def step(self, control=None):
        if control is not None:
            self.ego.apply_control(control)
        start = time.perf_counter()  # sub-millisecond; time.monotonic ticks at 15.6 ms on Windows
        frame = self.owner.tick()
        ticked = time.perf_counter()
        snapshot = self.world.get_snapshot()
        if snapshot.frame != frame:
            raise RuntimeError("Snapshot does not match the ticked frame")
        speed = true_forward_speed(snapshot, self.ego.id)
        sf = self.rig.read(frame, speed, sim_time_s=snapshot.timestamp.elapsed_seconds)
        ids = self.labels.read(frame)
        done = time.perf_counter()
        self.timing["tick"].append(ticked - start)
        self.timing["wait"].append(done - ticked)
        arrivals = self.rig.receipts.get(frame, {})
        if arrivals:
            self.timing["last_arrival"].append(max(arrivals.values()) - start)
        t = self.ego.get_transform()
        return dict(
            frame=sf,
            ids=ids,
            speed=speed,
            transform=t,
            pose=rear_axle_pose(t, self.offset),
            angular=self.ego.get_angular_velocity(),
            accel=self.ego.get_acceleration(),
            velocity=self.ego.get_velocity(),
        )

    def place(self, blueprint_id, x_ego, y_ego, pose):
        carla = self.carla
        wx, wy = ego_to_world_xy(pose, x_ego, y_ego)
        z = self.ego.get_transform().location.z
        try:
            ground = self.world.ground_projection(carla.Location(x=wx, y=-wy, z=z + 2.0), 6.0)
            if ground is not None:
                z = ground.location.z
        except Exception:
            pass
        transform = carla.Transform(carla.Location(x=wx, y=-wy, z=z + 0.05), carla.Rotation(yaw=-math.degrees(pose[3])))
        actor = self.owner.try_spawn(self.library.find(blueprint_id), transform)
        if actor is None:
            return None
        try:
            actor.set_simulate_physics(False)
            actor.set_transform(transform)
        except Exception:
            pass
        return actor

    def remove(self, actor):
        actor.destroy()
        self.owner._actors.remove(actor)

    # phase 1
    def phase_static(self):
        ticks = [self.step() for _ in range(self.args.static_ticks)]
        # CARLA's accelerometer spikes for the first ticks after the IMU spawns (first live
        # run: mean |a| = 1448 m/s^2 at rest). Record those ticks, then judge the rest.
        self.report["imu_first_ticks_accel_norm_mps2"] = [
            round(float(np.linalg.norm(t["frame"].imu.accel_mps2)), 2) if t["frame"].imu else None
            for t in ticks[: 2 * IMU_SETTLE_TICKS]
        ]
        states = {}
        for t in ticks:
            for s in t["frame"].status:
                states.setdefault(s.name, {}).setdefault(s.state.value, 0)
                states[s.name][s.state.value] += 1
        bad = {n: v for n, v in states.items() if v.get("missing") or v.get("invalid")}
        self.checks.append(check("every_sensor_every_tick", not bad, frames=len(ticks), states=states))
        last = ticks[-1]["frame"]
        for sweep in last.lidars:
            coverage = [lidar_azimuth_coverage_deg(t["frame"].lidar(sweep.name).points, sweep.sensor_to_ego) for t in ticks if t["frame"].status_of(sweep.name).state is SensorState.OK]
            counts = [len(t["frame"].lidar(sweep.name).points) for t in ticks if t["frame"].status_of(sweep.name).state is SensorState.OK]
            self_hits = int(cal.inside_box(sweep.points, self.ego_centre, 0.0, self.ego_extent, 0.05).sum())
            self.checks.append(
                check(
                    f"lidar_full_sweep_{sweep.name}",
                    bool(coverage) and min(coverage) >= 350.0,
                    min_coverage_deg=min(coverage) if coverage else None,
                    points_per_sweep=dict(min=min(counts), median=float(np.median(counts))) if counts else None,
                )
            )
            self.checks.append(
                check(
                    f"lidar_body_hits_{sweep.name}",
                    REPORT,
                    self_hits=self_hits,
                    note="Points on the ego body; the perception pipeline masks them. Reported, not gated.",
                )
            )
        for radar in last.radars:
            near = int((radar.returns[:, 0] < 0.5).sum()) if len(radar.returns) else 0
            self.checks.append(check(f"radar_no_self_returns_{radar.name}", near == 0, returns_under_0_5m=near, returns=len(radar.returns)))
        imu_ticks = [t for t in ticks[IMU_SETTLE_TICKS:] if t["frame"].imu is not None]
        if imu_ticks:
            accel = np.array([t["frame"].imu.accel_mps2 for t in imu_ticks])
            gyro = np.array([t["frame"].imu.gyro_rps for t in imu_ticks])
            compass_err = [
                math.degrees(abs(wrap(yaw_from_compass(t["frame"].imu.compass_rad) + math.radians(t["transform"].rotation.yaw))))
                for t in imu_ticks
            ]
            g = float(np.linalg.norm(accel.mean(axis=0)))
            tol = 1.0 if self.args.keep_noise else 0.3
            self.checks.append(check("imu_gravity_at_rest", abs(g - 9.81) <= tol, accel_norm_mps2=g, mean=rounded(accel.mean(axis=0))))
            self.checks.append(check("imu_still_at_rest", float(np.abs(gyro.mean(axis=0)).max()) < 0.01, gyro_mean_rps=rounded(gyro.mean(axis=0), 5)))
            self.checks.append(check("compass_heading", max(compass_err) < (5.0 if self.args.keep_noise else 1.0), max_error_deg=max(compass_err)))
        else:
            self.checks.append(check("imu_present", False))
        if self.suite.gnss is not None:
            fixes = sum(t["frame"].gnss is not None for t in ticks)
            expected = len(ticks) * DT_S / self.suite.gnss.period_s
            self.checks.append(check("gnss_rate", abs(fixes - expected) <= 2, fixes=fixes, expected=expected))
        self.save_samples(ticks[-1])

    def save_samples(self, tick, targets=()):
        import cv2

        out = self.args.output / "samples"
        out.mkdir(exist_ok=True)
        sf = tick["frame"]
        for image in sf.cameras:
            cv2.imwrite(str(out / f"{image.name}.jpg"), image.rgb[:, :, ::-1])
        bev = np.zeros((800, 800, 3), np.uint8)
        for sweep in sf.lidars:
            p = sweep.points
            u, v = (400 - p[:, 1] * 10).astype(int), (400 - p[:, 0] * 10).astype(int)
            keep = (u >= 0) & (u < 800) & (v >= 0) & (v < 800)
            bev[v[keep], u[keep]] = (200, 200, 200)
        for radar in sf.radars:
            for x, y, _ in radar.points_ego():
                cv2.circle(bev, (int(400 - y * 10), int(400 - x * 10)), 3, (0, 165, 255), -1)
        cv2.imwrite(str(out / "lidar_top_down.png"), bev)

    # phase 2
    def measure_target(self, name, actor, tick, vehicle=True):
        sf, pose = tick["frame"], tick["pose"]
        centre, yaw, extent = target_box_ego(actor, pose)
        row = dict(placement=name, centre_ego_m=rounded(centre), lidar={}, cameras={}, radars={})
        mirror = centre * ((1, -1, 1) if abs(centre[1]) > 1.0 else (-1, 1, 1))
        # Road returns under and beside the target are not the target (first live run).
        above = min(GROUND_CLEARANCE_M, extent[2])
        hits_all = []
        for sweep in sf.lidars:
            hit = cal.inside_box(sweep.points, centre, yaw, extent, 0.15, above)
            mirrored = cal.inside_box(sweep.points, mirror, -yaw if abs(centre[1]) > 1.0 else yaw, extent, 0.15, above)
            bearing, rng = cal.bearing_range(sweep.sensor_to_ego, centre)
            row["lidar"][sweep.name] = dict(points=int(hit.sum()), mirrored=int(mirrored.sum()), bearing_deg=round(bearing, 1), range_m=round(rng, 2))
            hits_all.append(sweep.points[hit])
        lidar_points = np.vstack(hits_all) if hits_all else np.zeros((0, 4))
        corners = cal.box_corners(centre, yaw, extent)
        for image in sf.cameras:
            h, w = image.rgb.shape[:2]
            px, front = cal.project(corners, image.calibration)
            entry = dict(visible=False)
            if front.all():
                rect = cal.rect_of(px)
                if rect[2] > 0 and rect[0] < w and rect[3] > 0 and rect[1] < h:
                    ids, tags = tick["ids"][image.name]
                    in_rect = cal.rect_contents(ids, tags, rect)
                    if self.args.save_images:
                        self.save_placement_image(name, image, tags, rect, lidar_points)
                    key = cal.target_instance(ids, rect, tags, cal.VEHICLE_TAGS if vehicle else None)
                    if key is not None:
                        mask = cal.instance_mask(ids, tags, key)
                        measured = cal.mask_rect(mask)
                        lpx, _ = cal.project(lidar_points, image.calibration)
                        entry = dict(
                            visible=True,
                            instance_code=int(key[0]),
                            instance_tag=int(key[1]),
                            mask_pixels=int(mask.sum()),
                            projected_rect_px=rounded(rect, 1),
                            mask_rect_px=rounded(measured, 1),
                            edge_errors_px={k: round(float(v), 2) for k, v in cal.edge_errors(rect, measured, w, h).items()},
                            lidar_points_in_view=int(len(lpx)),
                            mask_consistency=cal.mask_consistency(lpx, mask) if len(lpx) >= 20 else None,
                        )
                    entry["rect_contents"] = in_rect
            bearing, rng = cal.bearing_range(np.asarray(image.calibration.camera_to_ego) @ np.array(
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float), centre)
            entry.update(bearing_deg=round(bearing, 1), range_m=round(rng, 2))
            row["cameras"][image.name] = entry
        for radar in sf.radars:
            points = radar.points_ego()
            hit = cal.inside_box(points, centre, yaw, extent, 0.3, above) if len(points) else np.zeros(0, bool)
            origins, directions = cal.radar_rays(radar.returns[hit], radar.sensor_to_ego) if hit.any() else (None, None)
            errors = []
            for i, (r_meas, *_rest) in enumerate(radar.returns[hit]):
                truth = cal.ray_box_distance(origins[i], directions[i], centre, yaw, extent)
                if truth is not None:
                    errors.append(float(r_meas) - truth)
            bearing, rng = cal.bearing_range(radar.sensor_to_ego, centre)
            row["radars"][radar.name] = dict(hits=int(hit.sum()), range_errors_m=rounded(errors, 3), bearing_deg=round(bearing, 1), range_m=round(rng, 2))
        return row

    def phase_placements(self):
        pose = self.step()["pose"]
        lane = 3.5
        rows, props = [], []
        sweep = [(n, x, y) for n, x, y in cal.placements(self.front_x, self.rear_x, self.centre_x, lane)]
        prop_sweep = cal.prop_placements(self.front_x, self.rear_x, self.centre_x, lane)
        for blueprint, plan, out in ((TARGET_CAR, sweep, rows), (TARGET_PROP, prop_sweep, props)):
            if not list(self.library.filter(blueprint)):
                out += [dict(placement=name, skipped=f"{blueprint} unavailable") for name, _, _ in plan]
                continue
            for name, x, y in plan:
                side = "left" if y > 1 else "right" if y < -1 else None
                if side is not None:
                    if self.side[side] is None:
                        out.append(dict(placement=name, skipped="no driving lane on this side"))
                        continue
                    y = math.copysign(self.side[side], y)
                actor = self.place(blueprint, x, y, pose)
                if actor is None:
                    out.append(dict(placement=name, skipped="spawn blocked"))
                    continue
                first_seen = {}
                for k in range(1, self.args.settle_ticks + 1):
                    tick = self.step()
                    for cam in self.cameras_showing(actor, tick, blueprint == TARGET_CAR):
                        first_seen.setdefault(cam, k)
                row = self.measure_target(name, actor, tick, vehicle=blueprint == TARGET_CAR)
                row["first_visible_tick"] = first_seen
                out.append(row)
                self.remove(actor)
                self.step()
        self.report["coverage"] = dict(cars=rows, props=props)
        measured = [r for r in rows if "lidar" in r]
        lidar_bad = [
            r["placement"]
            for r in measured
            for v in r["lidar"].values()
            if v["range_m"] <= 45
            and not (v["points"] >= (30 if v["range_m"] <= 20 else 8) and v["mirrored"] < 0.2 * v["points"])
        ]
        self.checks.append(
            check(
                "lidar_extrinsics",
                INCOMPLETE if len(measured) < 6 else (not lidar_bad),
                placements_measured=len(measured),
                failing_placements=lidar_bad,
                note=(
                    "Within 45 m: at least 30 points (8 beyond 20 m) on the box at least"
                    f" {GROUND_CLEARANCE_M} m above its bottom, and fewer than 20% of that in the mirror-image box."
                ),
            )
        )
        per_camera = {}
        for r in measured:
            for cam, v in r["cameras"].items():
                if v.get("mask_consistency") is not None:
                    per_camera.setdefault(cam, []).append(v["mask_consistency"])
        cams = {s.name for s in self.suite.cameras}
        self.checks.append(
            check(
                "camera_mask_consistency",
                INCOMPLETE if set(per_camera) != cams else all(min(v) >= 0.85 for v in per_camera.values()),
                min_per_camera={k: round(min(v), 3) for k, v in per_camera.items()},
                cameras_without_evidence=sorted(cams - set(per_camera)),
            )
        )
        def edges(rows_, names):
            return [
                e
                for r in rows_
                if "cameras" in r
                for v in r["cameras"].values()
                if v.get("visible")
                for edge, e in v["edge_errors_px"].items()
                if edge in names
            ]

        gated = edges(props, ("left", "right", "bottom"))
        tops = edges(props, ("top",))
        self.checks.append(
            check(
                "camera_reprojection_props",
                INCOMPLETE if len(gated) < 8 else cal.p95(gated) <= 3.0,
                edges=len(gated),
                p95_px=cal.p95(gated),
                max_px=max(gated) if gated else None,
                top_edges_px=rounded(tops, 2),
                note=(
                    "Left, right and bottom silhouette edges of box props. Top edges are reported,"
                    " not gated: box01's mesh sits a few centimetres below its bounding box, which"
                    " shows as about 4 px at 3 m in the first live run while the other edges agreed"
                    " to under 0.5 px."
                ),
            )
        )
        car_edges = edges(measured, cal.EDGES)
        self.checks.append(
            check(
                "camera_silhouette_cars",
                REPORT,
                edges=len(car_edges),
                p95_px=cal.p95(car_edges),
                median_px=float(np.median(car_edges)) if car_edges else None,
                note="A car is not a box; its edges are approximate.",
            )
        )

        def radar_errors(rows_):
            return [
                e
                for r in rows_
                if "radars" in r
                for v in r["radars"].values()
                if 3.0 <= v["range_m"] <= 40.0
                for e in v["range_errors_m"]
            ]

        prop_range = radar_errors(props)
        car_range = radar_errors(measured)
        self.checks.append(
            check(
                "radar_range_accuracy",
                INCOMPLETE if len(prop_range) < 10 else cal.p95(prop_range) <= 0.5,
                returns=len(prop_range),
                p95_m=cal.p95(prop_range),
                median_m=float(np.median(prop_range)) if prop_range else None,
                note="Box props only: a car's body is not its bounding box (cars are reported separately).",
            )
        )
        self.checks.append(
            check(
                "radar_range_cars",
                REPORT,
                returns=len(car_range),
                p95_m=cal.p95(car_range),
                median_m=float(np.median(car_range)) if car_range else None,
            )
        )
        detection = {}
        for r in measured:
            for name, v in r["radars"].items():
                spec = next(s for s in self.suite.radars if s.name == name)
                if abs(v["bearing_deg"]) <= spec.horizontal_fov_deg / 2 and v["range_m"] <= spec.range_m:
                    d = detection.setdefault(name, dict(in_field=0, detected=0))
                    d["in_field"] += 1
                    d["detected"] += v["hits"] > 0
        self.checks.append(check("radar_detection_coverage", REPORT, per_radar=detection))
        self.plot_coverage(rows)

    def cameras_showing(self, actor, tick, vehicle):
        """Cameras whose segmentation already shows ``actor`` overlapping its projected box."""
        centre, yaw, extent = target_box_ego(actor, tick["pose"])
        corners = cal.box_corners(centre, yaw, extent)
        seen = []
        for image in tick["frame"].cameras:
            h, w = image.rgb.shape[:2]
            px, front = cal.project(corners, image.calibration)
            if not front.all():
                continue
            rect = cal.rect_of(px)
            if rect[2] <= 0 or rect[0] >= w or rect[3] <= 0 or rect[1] >= h:
                continue
            ids, tags = tick["ids"][image.name]
            key = cal.target_instance(ids, rect, tags, cal.VEHICLE_TAGS if vehicle else None)
            if key is None:
                continue
            measured = cal.mask_rect(cal.instance_mask(ids, tags, key))
            if measured is not None and cal.rect_iou(rect, measured) >= 0.3:
                seen.append(image.name)
        return seen

    def phase_render_lag(self):
        """Ticks until the cameras show a settled car that has jumped sideways.

        LiDAR is checked on the same ticks; it ray-casts the physics state, so it should
        show the new position on the first tick. A camera that shows the old position on
        the first tick is rendering the world late, which a moving car would feel as a
        position error of lag x speed.
        """
        pose = self.step()["pose"]
        car = self.place(TARGET_CAR, self.front_x + 10.0, 0.0, pose)
        if car is None:
            self.checks.append(check("camera_render_lag", INCOMPLETE, reason="target spawn blocked"))
            return
        for _ in range(15):
            tick = self.step()
        trials = []
        for shift in (1.5, -1.5, 1.5, -1.5):
            old_box = target_box_ego(car, tick["pose"])
            t = car.get_transform()
            wx, wy = ego_to_world_xy(tick["pose"], old_box[0][0], old_box[0][1] + shift)
            t.location.x, t.location.y = wx, -wy
            car.set_transform(t)
            trial = dict(shift_m=shift, camera_lag_ticks=None, lidar_lag_ticks=None)
            for k in range(1, 9):
                tick = self.step()
                new_box = target_box_ego(car, tick["pose"])
                image = tick["frame"].camera("front")
                ids, tags = tick["ids"]["front"]
                rects = {}
                for label, box in (("old", old_box), ("new", new_box)):
                    px, _ = cal.project(cal.box_corners(*box), image.calibration)
                    rects[label] = cal.rect_of(px)
                union = (min(rects["old"][0], rects["new"][0]), min(rects["old"][1], rects["new"][1]),
                         max(rects["old"][2], rects["new"][2]), max(rects["old"][3], rects["new"][3]))
                key = cal.target_instance(ids, union, tags, cal.VEHICLE_TAGS)
                if key is not None and trial["camera_lag_ticks"] is None:
                    measured = cal.mask_rect(cal.instance_mask(ids, tags, key))
                    if measured is not None and cal.rect_iou(measured, rects["new"]) > cal.rect_iou(measured, rects["old"]):
                        trial["camera_lag_ticks"] = k - 1
                if trial["lidar_lag_ticks"] is None:
                    sweep = tick["frame"].lidars[0]
                    above = min(GROUND_CLEARANCE_M, new_box[2][2])
                    new_hits = cal.inside_box(sweep.points, *new_box, 0.1, above).sum()
                    old_hits = cal.inside_box(sweep.points, *old_box, 0.1, above).sum()
                    if new_hits > old_hits:
                        trial["lidar_lag_ticks"] = k - 1
                if trial["camera_lag_ticks"] is not None and trial["lidar_lag_ticks"] is not None:
                    break
            trials.append(trial)
            for _ in range(3):
                tick = self.step()
        self.remove(car)
        self.step()
        lags = [t["camera_lag_ticks"] for t in trials]
        lidar = [t["lidar_lag_ticks"] for t in trials]
        relative = [None if c is None or li is None else c - li for c, li in zip(lags, lidar)]
        self.report["render_lag_trials"] = trials
        self.checks.append(
            check(
                "camera_render_lag",
                INCOMPLETE if None in relative else max(abs(r) for r in relative) == 0,
                camera_lag_ticks=lags,
                lidar_lag_ticks=lidar,
                camera_minus_lidar_ticks=relative,
                note=(
                    "Ticks between a settled car jumping 1.5 m sideways and the front camera showing it,"
                    " judged against LiDAR on the same ticks. LiDAR ray-casts the physics state, so camera"
                    " minus LiDAR is the camera's own lag. Live, the first jump after spawn took effect a"
                    " tick late for both sensors (a teleport delay, not render lag). A non-zero"
                    " difference means images lag the simulation, and perception must allow lag x speed."
                ),
            )
        )

    def save_placement_image(self, name, image, tags, rect, lidar_points):
        """Camera image beside its segmentation tags, with the projected true box and LiDAR points."""
        import cv2

        out = self.args.output / "placements"
        out.mkdir(exist_ok=True)
        rgb = np.ascontiguousarray(image.rgb[:, :, ::-1])
        seg = cv2.applyColorMap(np.clip(tags * 9, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        for canvas in (rgb, seg):
            r = [int(round(v)) for v in rect]
            cv2.rectangle(canvas, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
            px, _ = cal.project(lidar_points, image.calibration)
            for u, v in px.astype(int):
                if 0 <= u < canvas.shape[1] and 0 <= v < canvas.shape[0]:
                    cv2.circle(canvas, (int(u), int(v)), 2, (0, 0, 255), -1)
        cv2.imwrite(str(out / f"{name}_{image.name}.jpg"), np.hstack((rgb, seg)))

    def plot_coverage(self, rows):
        import cv2

        img = np.full((700, 500, 3), 255, np.uint8)

        def px(x, y):
            return int(250 - y * 20), int(450 - x * 7)

        cv2.rectangle(img, px(self.front_x, 1.0), px(self.rear_x, -1.0), (0, 0, 0), 2)
        for r in rows:
            if "lidar" not in r:
                continue
            x, y, _ = r["centre_ego_m"]
            lidar = any(v["points"] >= 30 for v in r["lidar"].values())
            camera = any(v.get("visible") for v in r["cameras"].values())
            radar = any(v["hits"] > 0 for v in r["radars"].values())
            colour = (0, 160, 0) if (lidar and camera and radar) else (0, 165, 255) if (lidar and camera) else (0, 0, 220)
            cv2.circle(img, px(x, y), 7, colour, -1)
            cv2.putText(img, r["placement"], (px(x, y)[0] + 9, px(x, y)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (40, 40, 40), 1)
        cv2.putText(img, "green: LiDAR+camera+radar  orange: no radar  red: LiDAR or camera missing", (8, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.imwrite(str(self.args.output / "coverage_top_down.png"), img)

    # phase 3
    def radar_case(self, name, target, ticks, front_only=True):
        rows = []
        for _ in range(ticks):
            t = self.step()
            pose = t["pose"]
            centre, yaw, extent = target_box_ego(target, pose)
            v_ego = to_ego_velocity(t["velocity"], pose)
            v_tgt = to_ego_velocity(target.get_velocity(), pose)
            omega = -math.radians(t["angular"].z)
            for radar in t["frame"].radars:
                if front_only and not radar.name.startswith("radar_front"):
                    continue
                points = radar.points_ego()
                if not len(points):
                    continue
                hit = cal.inside_box(points, centre, yaw, extent, 0.3, min(GROUND_CLEARANCE_M, extent[2]))
                if not hit.any():
                    continue
                _, directions = cal.radar_rays(radar.returns[hit], radar.sensor_to_ego)
                expected = cal.expected_range_rate(directions, radar.sensor_to_ego, v_ego, omega, v_tgt, True, self.offset)
                for measured, exp in zip(radar.returns[hit][:, 3], expected):
                    rows.append((float(measured), float(exp)))
        errors = [m - e for m, e in rows]
        signs = [m / e for m, e in rows if abs(e) > 0.5]
        return dict(
            case=name,
            returns=len(rows),
            p95_error_mps=cal.p95(errors),
            median_sign_ratio=float(np.median(signs)) if signs else None,
            median_abs_measured_mps=float(np.median([abs(m) for m, _ in rows])) if rows else None,
        )

    def phase_radar_motion(self):
        carla = self.carla
        cases = []
        side = "left" if self.side["left"] else "right" if self.side["right"] else None
        pose = self.step()["pose"]
        if side is not None:
            y = self.side[side] * (1 if side == "left" else -1)
            target = self.place(TARGET_CAR, self.centre_x + 12.0, y, pose)
            if target is not None:
                target.set_simulate_physics(True)
                target.enable_constant_velocity(carla.Vector3D(3.0, 0.0, 0.0))
                for _ in range(5):
                    self.step()
                cases.append(self.radar_case("stationary_ego_target_receding", target, 30))
                target.disable_constant_velocity()
                self.remove(target)
        lead = self.place(TARGET_CAR, self.front_x + 12.0, 0.0, pose)
        if lead is not None:
            lead.set_simulate_physics(True)
            self.ego.apply_control(carla.VehicleControl(brake=0.0))
            lead.enable_constant_velocity(carla.Vector3D(3.0, 0.0, 0.0))
            self.ego.enable_constant_velocity(carla.Vector3D(3.0, 0.0, 0.0))
            for _ in range(10):
                self.step()
            cases.append(self.radar_case("equal_speed_lead", lead, 25))
            self.ego.disable_constant_velocity()
            lead.disable_constant_velocity()
            self.remove(lead)
        for _ in range(80):
            if self.step(carla.VehicleControl(brake=1.0))["speed"] < 0.05:
                break
        self.report["radar_motion_cases"] = cases
        by_name = {c["case"]: c for c in cases}
        moving = by_name.get("stationary_ego_target_receding")
        self.checks.append(
            check(
                "radar_sign_moving_target",
                INCOMPLETE if not moving or moving["median_sign_ratio"] is None or moving["returns"] < 5 else moving["median_sign_ratio"] > 0.8,
                **(moving or {}),
            )
        )
        lead_case = by_name.get("equal_speed_lead")
        self.checks.append(
            check(
                "radar_equal_speed_lead",
                INCOMPLETE if not lead_case or lead_case["returns"] < 5 else lead_case["median_abs_measured_mps"] <= 0.3,
                **(lead_case or {}),
            )
        )
        self._rate_errors = [c["p95_error_mps"] for c in cases if c["p95_error_mps"] is not None]

    # phase 4
    def phase_drive(self):
        carla = self.carla
        # Straight 3 s, then a gentle left curve (radius about 40 m), then a stop.
        drive = [self.step(carla.VehicleControl(throttle=0.4)) for _ in range(60)]
        drive += [self.step(carla.VehicleControl(throttle=0.2, steer=-0.06)) for _ in range(30)]
        drive += [self.step(carla.VehicleControl(brake=1.0)) for _ in range(40)]
        straight, turn, keep_noise = drive[:60], drive[60:90], self.args.keep_noise
        fixes = [(t["frame"].gnss, t["transform"]) for t in straight if t["frame"].gnss is not None]
        if len(fixes) >= 2:
            plane = LocalTangentPlane(fixes[0][0].latitude_deg, fixes[0][0].longitude_deg)
            (gx0, gy0), (gx1, gy1) = (plane.to_local(f.latitude_deg, f.longitude_deg) for f, _ in (fixes[0], fixes[-1]))
            p0, p1 = (rear_axle_pose(t, self.offset) for _, t in (fixes[0], fixes[-1]))
            measured, true = np.array([gx1 - gx0, gy1 - gy0]), np.array([p1[0] - p0[0], p1[1] - p0[1]])
            angle = math.degrees(abs(wrap(math.atan2(measured[1], measured[0]) - math.atan2(true[1], true[0]))))
            ratio = float(np.linalg.norm(measured) / max(np.linalg.norm(true), 1e-6))
            ok = angle < (5 if keep_noise else 1) and abs(ratio - 1) < (0.1 if keep_noise else 0.01)
            self.checks.append(check("gnss_displacement", ok and np.linalg.norm(true) > 5, angle_error_deg=angle, length_ratio=ratio, true_displacement_m=float(np.linalg.norm(true))))
        else:
            self.checks.append(check("gnss_displacement", INCOMPLETE, fixes=len(fixes)))
        speed_err = [abs(t["frame"].speed_mps - t["speed"]) for t in drive]
        self.checks.append(
            check(
                "speedometer",
                REPORT if keep_noise else max(speed_err) < 0.02,
                max_error_mps=max(speed_err),
                encoder_scale=self.rig.encoder.scale,
            )
        )
        with_imu = [t for t in drive if t["frame"].imu is not None]
        yaw_err = [abs(t["frame"].imu.gyro_rps[2] + math.radians(t["angular"].z)) for t in with_imu]
        turning = [(t["frame"].imu.gyro_rps[2], -math.radians(t["angular"].z)) for t in turn if t["frame"].imu is not None]
        left = float(np.mean([b for _, b in turning])) if turning else 0.0
        measured_left = float(np.mean([a for a, _ in turning])) if turning else 0.0
        self.checks.append(
            check(
                "gyro_yaw_rate",
                max(yaw_err) < (0.05 if keep_noise else 0.01) and left > 0.02 and measured_left > 0.02,
                max_error_rps=max(yaw_err),
                mean_true_left_turn_rps=left,
                mean_measured_left_turn_rps=measured_left,
            )
        )
        rows = []
        for t in with_imu:
            wx, wy = -math.radians(t["angular"].x), math.radians(t["angular"].y)
            yaw = t["pose"][3]
            c, s = math.cos(yaw), math.sin(yaw)
            rows.append((*t["frame"].imu.gyro_rps[:2], c * wx + s * wy, -s * wx + c * wy))
        g = np.array(rows)
        self.report["gyro_xy_vs_truth_rms_rps"] = dict(
            x=float(np.sqrt(np.mean((g[:, 0] - g[:, 2]) ** 2))),
            y=float(np.sqrt(np.mean((g[:, 1] - g[:, 3]) ** 2))),
            note="Roll and pitch rates are small on a flat road; reported, not gated.",
        )
        pairs = []
        for t in with_imu[5:40]:
            fwd = t["transform"].get_forward_vector()
            pairs.append((t["frame"].imu.accel_mps2[0], t["accel"].x * fwd.x + t["accel"].y * fwd.y + t["accel"].z * fwd.z))
        imu_ax, true_ax = float(np.mean([a for a, _ in pairs])), float(np.mean([b for _, b in pairs]))
        self.checks.append(check("accelerometer_x_sign", imu_ax > 0.3 and true_ax > 0.3, imu_mean_mps2=imu_ax, true_mean_mps2=true_ax))

        def static_world(ticks, lever_arm):
            out = []
            for t in ticks:
                pose = t["pose"]
                v_ego = to_ego_velocity(t["velocity"], pose)
                omega = -math.radians(t["angular"].z)
                if math.hypot(*v_ego) < 2.0:
                    continue
                for radar in t["frame"].radars:
                    if not radar.name.startswith("radar_front") or not len(radar.returns):
                        continue
                    _, directions = cal.radar_rays(radar.returns, radar.sensor_to_ego)
                    expected = cal.expected_range_rate(directions, radar.sensor_to_ego, v_ego, omega, (0.0, 0.0), lever_arm, self.offset)
                    out += [(float(m), float(e)) for m, e in zip(radar.returns[:, 3], expected)]
            return out

        straight_rows = static_world(straight[20:], True)
        ratios = [m / e for m, e in straight_rows if abs(e) > 1.0]
        sign = float(np.median(ratios)) if ratios else None
        self.checks.append(
            check(
                "radar_velocity_sign",
                INCOMPLETE if sign is None else sign > 0.8,
                median_ratio=sign,
                returns_used=len(ratios),
                note="About +1 when RADAR_VELOCITY_IS_RANGE_RATE is right; about -1 means flip it.",
            )
        )
        with_arm = [m - e for m, e in static_world(turn, True)]
        without = [m - e for m, e in static_world(turn, False)]
        straight_err = [m - e for m, e in straight_rows]
        rate_p95 = [v for v in (cal.p95(straight_err), cal.p95(with_arm), *getattr(self, "_rate_errors", [])) if v is not None]
        self.checks.append(
            check(
                "radar_range_rate_accuracy",
                INCOMPLETE if not rate_p95 else max(rate_p95) <= 0.5,
                p95_straight_mps=cal.p95(straight_err),
                p95_turn_with_lever_arm_mps=cal.p95(with_arm),
                p95_turn_without_lever_arm_mps=cal.p95(without),
                p95_motion_cases_mps=getattr(self, "_rate_errors", []),
            )
        )

    def run(self):
        self.setup()
        self.rig = SensorRig(
            self.owner,
            self.ego,
            self.suite,
            episode_id=f"rigcheck-{uuid.uuid4().hex[:8]}",
            rear_axle_offset_m=self.offset,
            dt_s=DT_S,
            seed=self.args.seed,
        )
        self.labels = InstanceCameras(self.owner, self.ego, self.suite)
        with self.rig, self.labels:
            self.phase_static()
            if not self.args.skip_placements:
                self.phase_placements()
            self.phase_render_lag()
            if not self.args.skip_radar_motion:
                self.phase_radar_motion()
            self.phase_drive()
        self.report["rig"] = self.rig.record()
        self.report["timing"] = {key: percentiles(values) for key, values in self.timing.items()}
        self.report["timing_note"] = (
            "tick = world.tick(); wait = tick return to a complete SensorFrame plus instance images;"
            " last_arrival = tick start to the last sensor callback (host monotonic). These are"
            " rig timings, not end-to-end agent latency; report the real-time factor separately."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite", type=Path, default=Path("configs/sensor_suites/mapless_v2.json"))
    parser.add_argument("--set", action="append", default=[], metavar="SENSOR.FIELD=VALUE", help="Suite override (repeatable)")
    parser.add_argument("--output", type=Path, required=True, help="New folder for the report")
    parser.add_argument("--static-ticks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=5308)
    parser.add_argument("--keep-noise", action="store_true", help="Run the suite with its noise")
    parser.add_argument("--skip-placements", action="store_true")
    parser.add_argument("--skip-radar-motion", action="store_true")
    parser.add_argument(
        "--settle-ticks",
        type=int,
        default=12,
        help="Ticks between placing a target and measuring it (live: spawned cars took over 3 ticks to appear in camera images)",
    )
    parser.add_argument("--save-images", action="store_true", help="Save camera + segmentation images per placement")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output folder exists; choose a new one")
    configured = load_sensor_suite(args.suite).with_overrides(parse_overrides(args.set))
    suite = configured if args.keep_noise else configured.without_noise()
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    if client.get_server_version() != "0.9.16" or client.get_client_version() != "0.9.16":
        raise RuntimeError("CARLA 0.9.16 client and server required")
    world = client.get_world()
    if world.get_settings().synchronous_mode:
        raise RuntimeError("Another client owns synchronous ticks; close it first")
    if any(a.type_id.startswith(("vehicle.", "walker.", "sensor.")) for a in world.get_actors()):
        raise RuntimeError("Use an otherwise empty CARLA world")
    args.output.mkdir(parents=True)
    initial_settings = _settings_record(world.get_settings())
    initial_ids = {a.id for a in world.get_actors()}
    report = dict(
        purpose="Mapless sensor-rig calibration gate (plan v2.1, section 5)",
        server=client.get_server_version(),
        map=world.get_map().name,
        suite_file=str(args.suite),
        overrides=parse_overrides(args.set),
        suite_name=configured.name,
        suite_sha256=configured.sha256,
        noise="as configured" if args.keep_noise else "disabled for the check",
        radar_velocity_is_range_rate_constant=RADAR_VELOCITY_IS_RANGE_RATE,
        status="INCOMPLETE",
        checks=[],
    )
    checks = report["checks"]
    try:
        with SynchronousWorldSession(client, WorldTiming(fixed_delta_seconds=DT_S)) as owner:
            RigCheck(carla, owner, suite, args, report).run()
        # Destroying actors in synchronous mode only completes once the server ticks again.
        # After the session hands ticking back, give the server a moment before
        # judging leaks (the first live run listed the ego and its sensors as leaked).
        deadline = time.perf_counter() + 5.0
        while True:
            leaks = sorted({a.id for a in world.get_actors()} - initial_ids)
            if not leaks or time.perf_counter() > deadline:
                break
            try:
                world.wait_for_tick(1.0)
            except Exception:
                time.sleep(0.2)
        restored = _settings_record(client.get_world().get_settings()) == initial_settings
        checks.append(check("cleanup", not leaks and restored, leaked_actor_ids=leaks, settings_restored=restored))
        gated = [c for c in checks if c["status"] != REPORT]
        if any(c["status"] == FAIL for c in gated):
            report["status"] = "MEASURED_FAIL"
        elif any(c["status"] == INCOMPLETE for c in gated):
            report["status"] = "INCOMPLETE"
        else:
            report["status"] = "MEASURED_PASS"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "rig_check.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        print("\n".join([f"{c['status']:10}  {c['check']}" for c in checks] + [f"status: {report['status']}"]))
