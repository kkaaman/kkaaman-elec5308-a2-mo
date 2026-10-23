"""Training-data capture for the mapless stack (Town01–Town06 only; never Town07 or Town10HD).

Town07 is the held-out final test town (plan, section 12) and Town10HD the development
benchmark the baseline was tuned on: neither may appear in training data.

The ego drives on CARLA's Traffic Manager autopilot through background traffic and
pedestrians. Autopilot is used here for data collection only; it never drives a
measured run. Each saved sample holds:
- **What the car sees:** every sensor in the suite, converted exactly as the live
  ``SensorRig`` converts it (RGB, LiDAR, radar, GNSS, IMU, speedometer).
- **Training labels:** a semantic-segmentation image for every RGB camera, from a
  camera with the same mount and lens.
- **Truth:** ego pose, velocity and control; traffic-light state; nearby vehicles and
  pedestrians as boxes in the vehicle frame (centre, yaw, extent and the four plan-view
  corners, from CARLA's world vertices, so an off-centre bounding box is placed right);
  the map lane and its marking types.

Labels and truth are for training and evaluation only and are never inputs to the car.
``manifest.json`` says so and records the suite hash, map, seed and weather.

Layout of one run folder:

    manifest.json
    samples.jsonl                       one line per saved sample
    rgb/<camera>/<frame>.jpg
    labels/semantic/<camera>/<frame>.png   CARLA 0.9.16 semantic tags (uint8)
    lidar/<name>/<frame>.npz            N x 4 float32 (vehicle frame) under key "points"
    radar/<name>/<frame>.npy            N x 4 float32

Splits are by whole run and by town, never by neighbouring frames.
"""

import argparse
import json
import math
import random
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from carla_adapter.sensor_rig import SensorRig, carla_mount, true_forward_speed
from carla_adapter.vehicle import FixedVehicleAdapter
from carla_adapter.world import SynchronousWorldSession, WorldTiming, ensure_map
from evaluation.dataset import decode_bgra, map_short_name
from mapless.sensor_suite import load_sensor_suite

TEST_ONLY_MAPS = ("Town07", "Town07_Opt", "Town10HD", "Town10HD_Opt")  # held-out final test; dev benchmark
CAPTURE_MAPS = tuple(f"Town0{i}" for i in range(1, 7))
WEATHERS = (
    "ClearNoon",
    "CloudyNoon",
    "WetNoon",
    "ClearSunset",
    "CloudySunset",
    "WetCloudyNoon",
    "SoftRainNoon",
)
SCHEMA = "group7-mapless-capture-2"  # 2: truth boxes placed from world vertices, with corners


def capture_map_allowed(name):
    short = map_short_name(name)
    base = short[:-4] if short.endswith("_Opt") else short
    return base in CAPTURE_MAPS


class LabelCameras:
    """Semantic cameras matching each RGB camera (training labels only)."""

    def __init__(self, owner, vehicle, suite):
        self.owner, self.vehicle, self.suite = owner, vehicle, suite
        self.sensors, self.errors = [], []
        self._data, self._condition = {}, threading.Condition()

    def __enter__(self):
        import carla

        library = self.owner.world.get_blueprint_library()
        for spec in self.suite.cameras:
            bp = library.find("sensor.camera.semantic_segmentation")
            for key, value in dict(
                image_size_x=spec.width,
                image_size_y=spec.height,
                fov=spec.fov_deg,
                sensor_tick=0.0,
                lens_k=0.0,
                lens_kcube=0.0,
            ).items():
                if bp.has_attribute(key):
                    bp.set_attribute(key, str(value))
            m = carla_mount(spec.mount)
            sensor = self.owner.world.spawn_actor(
                bp,
                carla.Transform(
                    carla.Location(x=m["x"], y=m["y"], z=m["z"]),
                    carla.Rotation(pitch=m["pitch"], yaw=m["yaw"], roll=0.0),
                ),
                attach_to=self.vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
            self.sensors.append(sensor)
            self.owner._actors.append(sensor)
            sensor.listen(lambda image, name=spec.name: self._receive(name, image))
        return self

    def _receive(self, name, image):
        try:
            tags = decode_bgra(image.raw_data, image.width, image.height)[:, :, 2].copy()
            with self._condition:
                self._data.setdefault(image.frame, {})[name] = tags
                for old in [f for f in self._data if f < image.frame - 8]:
                    del self._data[old]
                self._condition.notify_all()
        except Exception as error:
            self.errors.append(f"{name}: {error}")
            del self.errors[:-8]

    def read(self, frame, timeout_s=5.0):
        names = {s.name for s in self.suite.cameras}
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not names <= set(self._data.get(frame, {})):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Missing label images for frame {frame}")
                self._condition.wait(remaining)
            result = self._data.pop(frame)
            for old in [f for f in self._data if f <= frame]:
                del self._data[old]
            return result

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


def core_pose(transform):
    """CARLA transform to (x, y, z, yaw_rad) in the right-handed frame."""
    loc, rot = transform.location, transform.rotation
    return (loc.x, -loc.y, loc.z, -math.radians(rot.yaw))


def to_ego_frame(ego_pose, point_xy, yaw):
    ex, ey, _, eyaw = ego_pose
    dx, dy = point_xy[0] - ex, point_xy[1] - ey
    c, s = math.cos(eyaw), math.sin(eyaw)
    return (c * dx + s * dy, -s * dx + c * dy, (yaw - eyaw + math.pi) % (2 * math.pi) - math.pi)


def truth_record(vehicle, carla_map, actors, rear_axle_offset_m, radius_m=80.0):
    """Training/evaluation truth for one sample. Never an input to the car."""
    ego_t = vehicle.get_transform()
    pose = core_pose(ego_t)
    # Rear-axle pose, matching the vehicle frame the sensors use.
    rear = (
        pose[0] - rear_axle_offset_m * math.cos(pose[3]),
        pose[1] - rear_axle_offset_m * math.sin(pose[3]),
        pose[2],
        pose[3],
    )
    v, w, control = vehicle.get_velocity(), vehicle.get_angular_velocity(), vehicle.get_control()
    waypoint = carla_map.get_waypoint(ego_t.location)
    others = []
    for actor in actors:
        if actor.id == vehicle.id or not actor.is_alive:
            continue
        t = actor.get_transform()
        p = core_pose(t)
        if math.dist(p[:2], rear[:2]) > radius_m:
            continue
        av = actor.get_velocity()
        c, s = math.cos(rear[3]), math.sin(rear[3])
        vx, vy = av.x, -av.y
        box = actor.bounding_box
        # The box centre is the actor's transform plus box.location (off the actor origin
        # for most CARLA vehicles). Place it from the box's own world vertices: the first
        # capture pilots put the centre on the actor origin (review of 94b5caa).
        vertices = [(v.x, -v.y, v.z) for v in box.get_world_vertices(t)]
        centre = tuple(float(np.mean([v[k] for v in vertices])) for k in range(3))
        x, y, yaw = to_ego_frame(rear, centre[:2], p[3])
        bottom = sorted(vertices, key=lambda v: v[2])[:4]  # the four lower corners, plan view
        corners = [list(to_ego_frame(rear, v[:2], p[3])[:2]) for v in bottom]
        others.append(
            dict(
                id=actor.id,
                type_id=actor.type_id,
                x_m=x,
                y_m=y,
                z_m=centre[2] - rear[2],
                yaw_rad=yaw,
                extent_m=[box.extent.x, box.extent.y, box.extent.z],
                corners_m=corners,
                velocity_mps=[c * vx + s * vy, -s * vx + c * vy],
            )
        )
    light = vehicle.get_traffic_light_state()
    return dict(
        ego_rear_axle=dict(x_m=rear[0], y_m=rear[1], z_m=rear[2], yaw_rad=rear[3]),
        ego_velocity_mps=[v.x, -v.y, v.z],
        ego_yaw_rate_rps=-math.radians(w.z),
        ego_control=dict(throttle=control.throttle, steer=-control.steer, brake=control.brake),
        at_traffic_light=bool(vehicle.is_at_traffic_light()),
        traffic_light_state=str(light).split(".")[-1],
        lane=dict(
            road_id=waypoint.road_id,
            section_id=waypoint.section_id,
            lane_id=waypoint.lane_id,
            is_junction=bool(waypoint.is_junction),
            lane_width_m=waypoint.lane_width,
            left_marking=str(waypoint.left_lane_marking.type),
            right_marking=str(waypoint.right_lane_marking.type),
            lane_change=str(waypoint.lane_change),
        ),
        actors=others,
    )


def source_commit():
    """The repository commit this capture ran from, or None outside a checkout."""
    import subprocess

    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def _write_sample(root, frame, sensor_frame, labels, truth, cv2):
    rel = {"rgb": {}, "semantic": {}, "lidar": {}, "radar": {}}
    for image in sensor_frame.cameras:
        path = Path("rgb") / image.name / f"{frame:08d}.jpg"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(root / path), image.rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write {path}")
        rel["rgb"][image.name] = path.as_posix()
        label = Path("labels") / "semantic" / image.name / f"{frame:08d}.png"
        (root / label).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(root / label), labels[image.name]):
            raise RuntimeError(f"Could not write {label}")
        rel["semantic"][image.name] = label.as_posix()
    for sweep in sensor_frame.lidars:
        path = Path("lidar") / sweep.name / f"{frame:08d}.npz"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / path, points=sweep.points)
        rel["lidar"][sweep.name] = path.as_posix()
    for radar in sensor_frame.radars:
        path = Path("radar") / radar.name / f"{frame:08d}.npy"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        np.save(root / path, radar.returns)
        rel["radar"][radar.name] = path.as_posix()
    gnss = sensor_frame.gnss
    return dict(
        frame=frame,
        sim_time_s=sensor_frame.sim_time_s,
        files=rel,
        gnss=None
        if gnss is None
        else dict(latitude_deg=gnss.latitude_deg, longitude_deg=gnss.longitude_deg, altitude_m=gnss.altitude_m),
        imu=dict(
            accel_mps2=list(sensor_frame.imu.accel_mps2),
            gyro_rps=list(sensor_frame.imu.gyro_rps),
            compass_rad=sensor_frame.imu.compass_rad,
        ),
        speed_mps=sensor_frame.speed_mps,
        calibration_id=sensor_frame.calibration_id,
        status=[dict(name=s.name, state=s.state.value) for s in sensor_frame.status],
        truth=truth,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--map", required=True, help="Town01-Town06 (optionally _Opt)")
    parser.add_argument("--suite", type=Path, default=Path("configs/sensor_suites/mapless_v2.json"))
    parser.add_argument("--output", type=Path, required=True, help="New folder for this run")
    parser.add_argument("--samples", type=int, default=1000, help="Samples to save")
    parser.add_argument("--save-every", type=int, default=10, help="Ticks between samples while moving")
    parser.add_argument("--episode-s", type=float, default=90.0, help="Respawn the ego after this long")
    # The Town03 pilot (40 cars, 40 walkers) averaged 0.8 cars and 0.2 pedestrians within
    # 30 m, and only 12 front-camera pedestrians in 200 samples were 40 px or taller.
    parser.add_argument("--vehicles", type=int, default=80)
    parser.add_argument("--walkers", type=int, default=150)
    parser.add_argument(
        "--cross-factor",
        type=float,
        default=0.3,
        help="Share of walkers that cross roads (CARLA set_pedestrians_cross_factor)",
    )
    parser.add_argument("--weather", default="cycle", help="A CARLA preset name, or 'cycle'")
    parser.add_argument("--seed", type=int, default=5308)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    args = parser.parse_args(argv)

    if map_short_name(args.map) in TEST_ONLY_MAPS or not capture_map_allowed(args.map):
        parser.error("Capture is allowed on Town01-Town06 only; Town07 (held-out test) and Town10HD (dev benchmark) are refused")
    if args.weather != "cycle" and args.weather not in WEATHERS:
        parser.error(f"--weather must be 'cycle' or one of {', '.join(WEATHERS)}")
    if args.output.exists():
        parser.error("Output folder exists; choose a new one")
    if args.samples < 1 or args.save_every < 1 or args.episode_s < 5:
        parser.error("Need --samples >= 1, --save-every >= 1, --episode-s >= 5")

    suite = load_sensor_suite(args.suite)
    import carla
    import cv2

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    if client.get_server_version() != "0.9.16" or client.get_client_version() != "0.9.16":
        raise RuntimeError("CARLA 0.9.16 client and server required")
    world = client.get_world()
    if world.get_settings().synchronous_mode:
        raise RuntimeError("Another client owns synchronous ticks; close it first")
    world = ensure_map(client, args.map, log=lambda text: print(text, flush=True))
    carla_map = world.get_map()
    if not capture_map_allowed(carla_map.name):
        raise RuntimeError(f"Server map {carla_map.name} is not allowed for capture")
    if any(a.type_id.startswith(("vehicle.", "walker.", "sensor.")) for a in world.get_actors()):
        raise RuntimeError("Use an otherwise empty CARLA world")

    run_id = f"{map_short_name(carla_map.name)}-{args.seed}-{uuid.uuid4().hex[:8]}"
    root = args.output
    root.mkdir(parents=True)
    manifest = dict(
        schema=SCHEMA,
        run_id=run_id,
        created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="TRAINING_DATA. labels/ and truth are training labels and evaluation data; never car inputs.",
        map=carla_map.name,
        seed=args.seed,
        suite=suite.as_dict(),
        suite_sha256=suite.sha256,
        server=client.get_server_version(),
        weather=args.weather,
        vehicles=args.vehicles,
        walkers=args.walkers,
        walker_cross_factor=args.cross_factor,
        save_every_ticks=args.save_every,
        episode_s=args.episode_s,
        driver="CARLA Traffic Manager autopilot (data collection only)",
        source_commit=source_commit(),
        box_geometry="centre and corners from bounding_box.get_world_vertices (schema 2)",
        rear_axle_offset="measured per episode from the wheels (FixedVehicleAdapter)",
        split_policy="Whole runs and whole towns; never neighbouring frames.",
        status="INCOMPLETE",
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rng = random.Random(args.seed)
    saved, episodes, skipped, failure = 0, 0, 0, None
    tm = client.get_trafficmanager(args.tm_port)
    walker_controllers = []
    session = SynchronousWorldSession(client, WorldTiming(fixed_delta_seconds=0.05))
    try:
        with session as owner:
            tm.set_synchronous_mode(True)
            tm.set_random_device_seed(args.seed)
            world = owner.world
            world.set_pedestrians_seed(args.seed)
            world.set_pedestrians_cross_factor(args.cross_factor)
            library = world.get_blueprint_library()
            spawn_points = list(carla_map.get_spawn_points())
            rng.shuffle(spawn_points)
            car_blueprints = [
                bp
                for bp in library.filter("vehicle.*")
                if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) == 4
            ]
            spawned_vehicles = 0
            for spawn in spawn_points[: args.vehicles]:
                actor = owner.try_spawn(rng.choice(car_blueprints), spawn)
                if actor is not None:
                    actor.set_autopilot(True, args.tm_port)
                    spawned_vehicles += 1
            walker_blueprints = list(library.filter("walker.pedestrian.*"))
            controller_bp = library.find("controller.ai.walker")
            for _ in range(args.walkers):
                location = world.get_random_location_from_navigation()
                if location is None:
                    continue
                walker = owner.try_spawn(rng.choice(walker_blueprints), carla.Transform(location))
                if walker is None:
                    continue
                controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                owner._actors.append(controller)
                walker_controllers.append(controller)
            manifest.update(spawned_vehicles=spawned_vehicles, spawned_walkers=len(walker_controllers))
            owner.tick()
            for controller in walker_controllers:
                controller.start()
                controller.go_to_location(world.get_random_location_from_navigation())
                controller.set_max_speed(rng.uniform(0.9, 1.8))
            ego_bp = library.find("vehicle.tesla.model3")
            ego_bp.set_attribute("role_name", "hero")
            weathers = list(WEATHERS) if args.weather == "cycle" else [args.weather]
            with (root / "samples.jsonl").open("w", encoding="utf-8") as index:
                while saved < args.samples:
                    weather = weathers[episodes % len(weathers)]
                    world.set_weather(getattr(carla.WeatherParameters, weather))
                    ego = None
                    for spawn in rng.sample(spawn_points, len(spawn_points)):
                        ego = owner.try_spawn(ego_bp, spawn)
                        if ego is not None:
                            break
                    if ego is None:
                        raise RuntimeError("No free spawn point for the ego")
                    owner.tick()
                    offset = FixedVehicleAdapter(ego).geometry.rear_axle_offset_m
                    ego.set_autopilot(True, args.tm_port)
                    tm.ignore_lights_percentage(ego, 0.0)
                    episode_seed = args.seed * 1000 + episodes
                    with SensorRig(
                        owner,
                        ego,
                        suite,
                        episode_id=f"{run_id}-e{episodes}",
                        rear_axle_offset_m=offset,
                        dt_s=0.05,
                        seed=episode_seed,
                    ) as rig, LabelCameras(owner, ego, suite) as labels:
                        ticks, stopped_ticks = 0, 0
                        while ticks * 0.05 < args.episode_s and saved < args.samples:
                            frame = owner.tick()
                            ticks += 1
                            snapshot = world.get_snapshot()
                            if snapshot.frame != frame:
                                raise RuntimeError("Snapshot does not match the ticked frame")
                            speed = true_forward_speed(snapshot, ego.id)
                            sensor_frame = rig.read(frame, speed, sim_time_s=snapshot.timestamp.elapsed_seconds)
                            label_images = labels.read(frame)
                            if sensor_frame.degraded():
                                skipped += 1  # keep the dataset clean; the rig record counts why
                                continue
                            stopped_ticks = stopped_ticks + 1 if abs(speed) < 0.1 else 0
                            if stopped_ticks * 0.05 > 30.0:
                                break  # stuck: next episode
                            every = args.save_every if abs(speed) > 0.5 else 4 * args.save_every
                            if ticks < 20 or ticks % every:
                                continue
                            actors = world.get_actors()
                            truth = truth_record(
                                ego,
                                carla_map,
                                list(actors.filter("vehicle.*")) + list(actors.filter("walker.pedestrian.*")),
                                offset,
                            )
                            truth.update(weather=weather, episode=episodes)
                            sample = _write_sample(root, frame, sensor_frame, label_images, truth, cv2)
                            index.write(json.dumps(sample) + "\n")
                            saved += 1
                            if saved % 50 == 0:
                                print(f"{saved}/{args.samples} samples", flush=True)
                        manifest.setdefault("rig_records", []).append(rig.record())
                    ego.set_autopilot(False, args.tm_port)
                    ego.destroy()
                    owner._actors.remove(ego)
                    episodes += 1
            for controller in walker_controllers:
                controller.stop()
            tm.set_synchronous_mode(False)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        raise
    finally:
        manifest.update(
            samples=saved,
            episodes=episodes,
            skipped_degraded_frames=skipped,
            status="COMPLETE" if failure is None and saved >= args.samples else "INCOMPLETE",
            failure=failure,
            actors_removed_by_server=len(session.vanished_actor_ids),
        )
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(dict(samples=saved, episodes=episodes, status=manifest["status"])), file=sys.stderr)
