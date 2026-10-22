"""Live check of the mapless sensor rig: conversions, signs, extrinsics and timing.

By default the rig runs noise-free, so every check has a tight tolerance. ``--keep-noise``
runs the suite as configured and reports the same numbers. Truth (actor poses and boxes)
is read here only to score the rig. It never reaches the ``SensorFrame``.

The check runs in three parts.

**Setup.** It picks a spawn point with a straight road at least 45 m long and a
same-direction neighbour lane. Two parked target cars go in the neighbour lane, one 20 m
ahead (inside the front radar's field of view) and one 10 m behind.

**Stationary phase:**
- every sensor delivers on every tick, with one shared timestamp;
- each LiDAR sweep covers 360°, with no points on the ego body;
- target extrinsics: LiDAR points fall inside the true box and not inside its
  mirror-image box (catches a sign error);
- the camera agrees with the LiDAR: the projected true box overlaps the projected
  LiDAR points;
- the radars see the targets on the correct side;
- the IMU reads gravity at rest, and the compass agrees with the true heading;
- GNSS fixes arrive at the configured period.

**Driving phase** (targets removed; straight, then a gentle left curve, then a stop):
- GNSS displacement in the car's own tangent plane matches the true displacement;
- the speedometer matches the true speed;
- the gyro z-rate has the correct sign and size, and the accelerometer x has the sign of
  the true acceleration;
- the radar velocity sign: static-world returns must show the closing speed the car's
  own motion predicts. This settles ``RADAR_VELOCITY_IS_RANGE_RATE``.

**Report:** ``rig_check.json`` with per-check values and pass/fail, and timing (tick,
sensor arrival and conversion; p50/p95/max). It also saves camera samples and a
top-down LiDAR image. The status is MEASURED_PASS only if every check passes and no
actors leak.
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
    lidar_azimuth_coverage_deg,
    true_forward_speed,
)
from carla_adapter.vehicle import FixedVehicleAdapter
from carla_adapter.world import SynchronousWorldSession, WorldTiming, _settings_record
from mapless.geometry import wrap, yaw_from_compass
from mapless.satnav import LocalTangentPlane
from mapless.sensor_suite import load_sensor_suite

DT_S = 0.05


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


def target_box_ego(actor, ego_pose):
    """(centre xyz, yaw, extent xyz) of an actor's box in the vehicle frame."""
    corners = [(v.x, -v.y, v.z) for v in actor.bounding_box.get_world_vertices(actor.get_transform())]
    centre = world_to_ego([np.mean(corners, axis=0)], ego_pose)[0]
    yaw = wrap(-math.radians(actor.get_transform().rotation.yaw) - ego_pose[3])
    e = actor.bounding_box.extent
    return centre, yaw, np.array([e.x, e.y, e.z])


def inside_box(points, centre, yaw, extent, margin):
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(points, dtype=float)[:, :3] - centre
    local = np.column_stack((c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1], p[:, 2]))
    return np.all(np.abs(local) <= extent + margin, axis=1)


def box_corners(centre, yaw, extent):
    c, s = math.cos(yaw), math.sin(yaw)
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)
    local = signs * extent
    return np.column_stack(
        (c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1], local[:, 2])
    ) + centre


def project(points_ego, calibration):
    """Vehicle-frame points to pixels; rows behind the camera are dropped."""
    m = np.asarray(calibration.camera_to_ego)
    optical = (np.asarray(points_ego, dtype=float)[:, :3] - m[:3, 3]) @ m[:3, :3]
    front = optical[:, 2] > 0.1
    optical = optical[front]
    u = calibration.cx_px + calibration.fx_px * optical[:, 0] / optical[:, 2]
    v = calibration.cy_px + calibration.fy_px * optical[:, 1] / optical[:, 2]
    return np.column_stack((u, v))


def rect_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def percentiles(values_s):
    if not values_s:
        return None
    ms = np.asarray(values_s) * 1000.0
    return dict(p50_ms=float(np.percentile(ms, 50)), p95_ms=float(np.percentile(ms, 95)), max_ms=float(ms.max()))


def check(name, passed, **values):
    return dict(check=name, status="PASS" if passed else "FAIL", **values)


# -- site selection -------------------------------------------------------------------
def candidate_sites(carla, carla_map, spawn_points, length_m=45.0):
    """Spawn points with a straight road ahead and a same-direction neighbour lane."""
    for spawn in spawn_points:
        wp = carla_map.get_waypoint(spawn.location, project_to_road=True)
        if wp is None or wp.is_junction:
            continue
        neighbour = None
        for side in (wp.get_left_lane(), wp.get_right_lane()):
            if side is not None and side.lane_type == carla.LaneType.Driving and side.lane_id * wp.lane_id > 0:
                neighbour = side
                break
        if neighbour is None:
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
        behind = neighbour.previous(10.0)
        if ok and behind and not behind[0].is_junction:
            yield wp, neighbour


# -- main -----------------------------------------------------------------------------
def main(argv=None):
    import carla
    import cv2

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite", type=Path, default=Path("configs/sensor_suites/mapless_v1.json"))
    parser.add_argument("--output", type=Path, required=True, help="New folder for the report")
    parser.add_argument("--static-ticks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=5308)
    parser.add_argument("--keep-noise", action="store_true", help="Run the suite with its noise")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output folder exists; choose a new one")
    configured = load_sensor_suite(args.suite)
    suite = configured if args.keep_noise else configured.without_noise()
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
        purpose="Mapless sensor-rig check (conversions, signs, extrinsics, timing)",
        server=client.get_server_version(),
        map=world.get_map().name,
        suite_file=str(args.suite),
        suite_sha256=configured.sha256,
        noise="as configured" if args.keep_noise else "disabled for the check",
        radar_velocity_is_range_rate_constant=RADAR_VELOCITY_IS_RANGE_RATE,
        status="INCOMPLETE",
        checks=[],
    )
    checks = report["checks"]
    timing = dict(tick=[], wait=[], last_arrival=[])
    try:
        with SynchronousWorldSession(client, WorldTiming(fixed_delta_seconds=DT_S)) as owner:
            world = owner.world
            carla_map = world.get_map()
            world.set_weather(carla.WeatherParameters.ClearNoon)
            library = world.get_blueprint_library()
            ego, targets, site = None, {}, None
            for lane, neighbour in candidate_sites(carla, carla_map, carla_map.get_spawn_points()):
                spawn = lane.transform
                spawn.location.z += 0.3
                ego = owner.try_spawn(library.find("vehicle.tesla.model3"), spawn)
                if ego is None:
                    continue
                spawned = [ego]
                for name, wp in (("ahead", neighbour.next(20.0)[0]), ("behind", neighbour.previous(10.0)[0])):
                    t = wp.transform
                    t.location.z += 0.4
                    actor = owner.try_spawn(library.find("vehicle.audi.a2"), t)
                    if actor is None:
                        break
                    targets[name] = actor
                    spawned.append(actor)
                if len(targets) == 2:
                    site = (lane, neighbour)
                    break
                for actor in spawned:  # blocked site: clear it and try the next one
                    actor.destroy()
                    owner._actors.remove(actor)
                ego, targets = None, {}
            if site is None:
                raise RuntimeError("No free straight site with a same-direction neighbour lane")
            lane, neighbour = site
            ego.apply_control(carla.VehicleControl(brake=1.0))
            for actor in targets.values():
                actor.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            for _ in range(30):
                owner.tick()
            for actor in targets.values():
                actor.set_simulate_physics(False)
            offset = FixedVehicleAdapter(ego).geometry.rear_axle_offset_m
            report["site"] = dict(
                road_id=lane.road_id, lane_id=lane.lane_id, neighbour_lane_id=neighbour.lane_id
            )
            ego_box = ego.bounding_box
            ego_centre = np.array([ego_box.location.x + offset, -ego_box.location.y, ego_box.location.z])
            ego_extent = np.array([ego_box.extent.x, ego_box.extent.y, ego_box.extent.z])
            rig = SensorRig(
                owner,
                ego,
                suite,
                episode_id=f"rigcheck-{uuid.uuid4().hex[:8]}",
                rear_axle_offset_m=offset,
                dt_s=DT_S,
                seed=args.seed,
            )
            with rig:

                def step(control=None):
                    if control is not None:
                        ego.apply_control(control)
                    start = time.monotonic()
                    frame = owner.tick()
                    ticked = time.monotonic()
                    snapshot = world.get_snapshot()
                    if snapshot.frame != frame:
                        raise RuntimeError("Snapshot does not match the ticked frame")
                    speed = true_forward_speed(snapshot, ego.id)
                    sf = rig.read(frame, speed)
                    done = time.monotonic()
                    timing["tick"].append(ticked - start)
                    timing["wait"].append(done - ticked)
                    arrivals = rig.receipts.get(frame, {})
                    if arrivals:
                        timing["last_arrival"].append(max(arrivals.values()) - start)
                    return sf, speed, ego.get_transform(), ego.get_angular_velocity(), ego.get_acceleration()

                static = [step() for _ in range(args.static_ticks)]
                checks.append(
                    check(
                        "every_sensor_every_tick",
                        len(static) == args.static_ticks
                        and not rig.callback_errors
                        and rig.counts.get("timeout", 0) == 0,
                        frames=len(static),
                        callback_errors=list(rig.callback_errors),
                    )
                )
                sf, _, ego_t, _, _ = static[-1]
                pose = rear_axle_pose(ego_t, offset)
                for sweep in sf.lidars:
                    coverage = [lidar_azimuth_coverage_deg(f[0].lidar(sweep.name).points, sweep.sensor_to_ego) for f in static]
                    counts = [len(f[0].lidar(sweep.name).points) for f in static]
                    self_hits = int(inside_box(sweep.points, ego_centre, 0.0, ego_extent, 0.05).sum())
                    checks.append(
                        check(
                            f"lidar_full_sweep_{sweep.name}",
                            min(coverage) >= 350.0,
                            min_coverage_deg=min(coverage),
                            points_per_sweep=dict(min=min(counts), median=float(np.median(counts))),
                        )
                    )
                    checks.append(check(f"lidar_no_self_hits_{sweep.name}", self_hits == 0, self_hits=self_hits))
                    for name, actor in targets.items():
                        centre, yaw, extent = target_box_ego(actor, pose)
                        hit = inside_box(sweep.points, centre, yaw, extent, 0.15)
                        mirrored = inside_box(sweep.points, centre * (1, -1, 1), -yaw, extent, 0.15)
                        checks.append(
                            check(
                                f"lidar_extrinsics_{sweep.name}_{name}",
                                hit.sum() >= 30 and mirrored.sum() < 0.2 * hit.sum(),
                                points_in_box=int(hit.sum()),
                                points_in_mirrored_box=int(mirrored.sum()),
                                box_centre_ego_m=[round(float(v), 3) for v in centre],
                            )
                        )
                        if name != "ahead":
                            continue
                        points = sweep.points[hit]
                        for image in sf.cameras:
                            box_px = project(box_corners(centre, yaw, extent), image.calibration)
                            lidar_px = project(points, image.calibration)
                            h, w = image.rgb.shape[:2]
                            if len(box_px) < 8 or len(lidar_px) < 10:
                                continue
                            a = (*box_px.min(axis=0), *box_px.max(axis=0))
                            b = (*lidar_px.min(axis=0), *lidar_px.max(axis=0))
                            if a[2] < 0 or a[0] > w or a[3] < 0 or a[1] > h:
                                continue
                            iou = rect_iou(a, b)
                            checks.append(
                                check(
                                    f"camera_lidar_agree_{image.name}",
                                    iou >= 0.5,
                                    iou=round(iou, 3),
                                    box_rect_px=[round(float(v), 1) for v in a],
                                    lidar_rect_px=[round(float(v), 1) for v in b],
                                )
                            )
                for radar in sf.radars:
                    pts = radar.points_ego()
                    seen = {}
                    for name, actor in targets.items():
                        centre, yaw, extent = target_box_ego(actor, pose)
                        hits = int(inside_box(pts, centre, yaw, extent, 0.75).sum()) if len(pts) else 0
                        mirrored = (
                            int(inside_box(pts, centre * (1, -1, 1), -yaw, extent, 0.75).sum()) if len(pts) else 0
                        )
                        seen[name] = dict(hits=hits, mirrored_hits=mirrored)
                    report.setdefault("radar_static", {})[radar.name] = dict(returns=len(pts), targets=seen)
                for side, name in (("ahead", "radar_front"), ("behind", "radar_rear")):
                    radars = [r for r in sf.radars if r.name.startswith(name)]
                    if not radars:
                        continue
                    stats = [report["radar_static"][r.name]["targets"][side] for r in radars]
                    hits = sum(s["hits"] for s in stats)
                    mirrored = sum(s["mirrored_hits"] for s in stats)
                    checks.append(
                        check(f"radar_sees_target_{side}", hits >= 1 and hits > mirrored, hits=hits, mirrored_hits=mirrored)
                    )
                accel = np.array([f[0].imu.accel_mps2 for f in static])
                gyro = np.array([f[0].imu.gyro_rps for f in static])
                compass_err = [
                    math.degrees(abs(wrap(yaw_from_compass(f[0].imu.compass_rad) + math.radians(f[2].rotation.yaw))))
                    for f in static
                ]
                g = float(np.linalg.norm(accel.mean(axis=0)))
                tol = 1.0 if args.keep_noise else 0.3
                checks.append(check("imu_gravity_at_rest", abs(g - 9.81) <= tol, accel_norm_mps2=g, mean=accel.mean(axis=0).round(3).tolist()))
                checks.append(check("imu_still_at_rest", float(np.abs(gyro.mean(axis=0)).max()) < 0.01, gyro_mean_rps=gyro.mean(axis=0).round(5).tolist()))
                checks.append(check("compass_heading", max(compass_err) < (5.0 if args.keep_noise else 1.0), max_error_deg=max(compass_err)))
                if suite.gnss is not None:
                    fixes = sum(f[0].gnss is not None for f in static)
                    expected = args.static_ticks * DT_S / suite.gnss.period_s
                    checks.append(check("gnss_rate", abs(fixes - expected) <= 2, fixes=fixes, expected=expected))
                sample_dir = args.output / "samples"
                sample_dir.mkdir()
                for image in sf.cameras:
                    cv2.imwrite(str(sample_dir / f"{image.name}.jpg"), image.rgb[:, :, ::-1])
                bev = np.zeros((800, 800, 3), np.uint8)
                for sweep in sf.lidars:
                    p = sweep.points
                    u = (400 - p[:, 1] * 10).astype(int)
                    v = (400 - p[:, 0] * 10).astype(int)
                    keep = (u >= 0) & (u < 800) & (v >= 0) & (v < 800)
                    bev[v[keep], u[keep]] = (200, 200, 200)
                for radar in sf.radars:
                    for x, y, _ in radar.points_ego():
                        cv2.circle(bev, (int(400 - y * 10), int(400 - x * 10)), 3, (0, 165, 255), -1)
                for actor in targets.values():
                    corners = box_corners(*target_box_ego(actor, pose))[[0, 2, 6, 4], :2]
                    poly = np.column_stack((400 - corners[:, 1] * 10, 400 - corners[:, 0] * 10)).astype(np.int32)
                    cv2.polylines(bev, [poly], True, (0, 255, 0), 1)
                cv2.imwrite(str(sample_dir / "lidar_top_down.png"), bev)

                for actor in targets.values():
                    actor.destroy()
                    owner._actors.remove(actor)
                ego.apply_control(carla.VehicleControl(brake=0.0))
                drive = [step(carla.VehicleControl(throttle=0.5)) for _ in range(80)]
                drive += [step(carla.VehicleControl(throttle=0.3, steer=-0.2)) for _ in range(30)]
                drive += [step(carla.VehicleControl(brake=1.0)) for _ in range(30)]

            straight = drive[:80]
            fixes = [(f[0].gnss, f[2]) for f in straight if f[0].gnss is not None]
            if len(fixes) >= 2:
                plane = LocalTangentPlane(fixes[0][0].latitude_deg, fixes[0][0].longitude_deg)
                (gx0, gy0), (gx1, gy1) = (plane.to_local(f.latitude_deg, f.longitude_deg) for f, _ in (fixes[0], fixes[-1]))
                p0, p1 = (rear_axle_pose(t, offset) for _, t in (fixes[0], fixes[-1]))
                measured = np.array([gx1 - gx0, gy1 - gy0])
                true = np.array([p1[0] - p0[0], p1[1] - p0[1]])
                angle = math.degrees(abs(wrap(math.atan2(measured[1], measured[0]) - math.atan2(true[1], true[0]))))
                ratio = float(np.linalg.norm(measured) / max(np.linalg.norm(true), 1e-6))
                ok = angle < (5 if args.keep_noise else 1) and abs(ratio - 1) < (0.1 if args.keep_noise else 0.01)
                checks.append(
                    check("gnss_displacement", ok and np.linalg.norm(true) > 5, angle_error_deg=angle, length_ratio=ratio, true_displacement_m=float(np.linalg.norm(true)))
                )
            speed_err = [abs(f[0].speed_mps - f[1]) for f in drive]
            checks.append(
                check(
                    "speedometer",
                    True if args.keep_noise else max(speed_err) < 0.02,
                    max_error_mps=max(speed_err),
                    encoder_scale=rig.encoder.scale,
                    note="Gated only when noise-free; with noise the error is the configured model.",
                )
            )
            yaw_err = [abs(f[0].imu.gyro_rps[2] - (-math.radians(f[3].z))) for f in drive]
            turning = [(f[0].imu.gyro_rps[2], -math.radians(f[3].z)) for f in drive[80:110]]
            left = float(np.mean([t for _, t in turning]))
            measured_left = float(np.mean([m for m, _ in turning]))
            checks.append(
                check(
                    "gyro_yaw_rate",
                    max(yaw_err) < (0.05 if args.keep_noise else 0.01) and left > 0.02 and measured_left > 0.02,
                    max_error_rps=max(yaw_err),
                    mean_true_left_turn_rps=left,
                    mean_measured_left_turn_rps=measured_left,
                )
            )
            rows = []
            for f in drive:
                # True angular velocity: CARLA world frame (deg/s, left-handed) -> vehicle frame.
                wx, wy = -math.radians(f[3].x), math.radians(f[3].y)
                yaw = -math.radians(f[2].rotation.yaw)
                c, s = math.cos(yaw), math.sin(yaw)
                rows.append((*f[0].imu.gyro_rps[:2], c * wx + s * wy, -s * wx + c * wy))
            gyro_xy = np.array(rows)
            report["gyro_xy_vs_truth_rms_rps"] = dict(
                x=float(np.sqrt(np.mean((gyro_xy[:, 0] - gyro_xy[:, 2]) ** 2))),
                y=float(np.sqrt(np.mean((gyro_xy[:, 1] - gyro_xy[:, 3]) ** 2))),
                note="Roll and pitch rates are small on a flat road; reported, not gated.",
            )
            accel_pairs = []
            for f in drive[5:40]:
                fwd = f[2].get_forward_vector()
                accel_pairs.append((f[0].imu.accel_mps2[0], f[4].x * fwd.x + f[4].y * fwd.y + f[4].z * fwd.z))
            imu_ax = float(np.mean([a for a, _ in accel_pairs]))
            true_ax = float(np.mean([t for _, t in accel_pairs]))
            checks.append(check("accelerometer_x_sign", imu_ax > 0.3 and true_ax > 0.3, imu_mean_mps2=imu_ax, true_mean_mps2=true_ax))
            ratios = []
            for f in straight[20:]:
                v = f[1]
                if v < 2.0:
                    continue
                for radar in f[0].radars:
                    if not radar.name.startswith("radar_front"):
                        continue
                    for rng_m, az, el, rate in radar.returns:
                        expected = -v * math.cos(az) * math.cos(el)
                        if abs(expected) > 1.0:
                            ratios.append(rate / expected)
            sign = float(np.median(ratios)) if ratios else None
            checks.append(
                check(
                    "radar_velocity_sign",
                    sign is not None and sign > 0.8,
                    median_ratio=sign,
                    returns_used=len(ratios),
                    note="About +1 when RADAR_VELOCITY_IS_RANGE_RATE is right; about -1 means flip it.",
                )
            )
            report["rig"] = rig.record()
            report["timing"] = {key: percentiles(values) for key, values in timing.items()}
            report["timing_note"] = (
                "tick = world.tick(); wait = from tick return to a complete SensorFrame;"
                " last_arrival = tick start to the last sensor callback (host monotonic)."
            )
        final_ids = {a.id for a in world.get_actors()}
        leaks = sorted(final_ids - initial_ids)
        restored = _settings_record(client.get_world().get_settings()) == initial_settings
        checks.append(check("cleanup", not leaks and restored, leaked_actor_ids=leaks, settings_restored=restored))
        report["status"] = "MEASURED_PASS" if all(c["status"] == "PASS" for c in checks) else "MEASURED_FAIL"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "rig_check.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        summary = [f"{c['status']:4}  {c['check']}" for c in checks]
        print("\n".join(summary + [f"status: {report['status']}"]))
