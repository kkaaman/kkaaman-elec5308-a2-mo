"""BL-E-04: single-owner RGB-D world, same public stack as the offline harness."""

import math
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from autonomy.interfaces import EgoPose
from carla_adapter.ground_truth import truth_record
from carla_adapter.mission import assert_current_map
from carla_adapter.sensors import RGBDCapture
from carla_adapter.vehicle import FixedVehicleAdapter
from carla_adapter.world import SynchronousWorldSession, WorldTiming, _settings_record
from evaluation.event_checks import annotate_invasion
from geometry.camera import CameraIntrinsics
from planning.corridor import RouteGeometry
from planning.mission import route_graph, scenario_obstacle_s


class CarlaWorld:
    """Own actors/ticks; evaluation observations never enter FrameBundle.

    Camera pipeline delay is explicit. Commands expire against current world
    simulation time, not the older camera timestamp. No autopilot is enabled.

    The default reads the frame just ticked (``sensor_lag_frames=0``). The
    world is synchronous, so the simulator is paused while the client waits
    for that frame's callbacks; a fixed pipeline lag would only make control
    act on older pose/images and inflate capture-to-apply latency.
    """

    def __init__(
        self,
        client,
        mission,
        scenario,
        config,
        *,
        depth_convention,
        output,
        follow=False,
        sensor_lag_frames=0,
        event_verification=None,
        declared_crossings=(),
    ):
        if depth_convention not in {"axial", "radial"}:
            raise ValueError("Calibrated depth convention required")
        if type(sensor_lag_frames) is not int or not 0 <= sensor_lag_frames <= 4:
            raise ValueError("Invalid camera pipeline lag")
        self.client, self.mission, self.scenario, self.config = client, mission, scenario, config
        self.depth_convention, self.follow, self.sensor_lag = (
            depth_convention,
            follow,
            sensor_lag_frames,
        )
        self.output = Path(output)
        # Collision/lane-invasion metrics are scored only after a live BL-B-07
        # verification for this map build has passed; otherwise they stay None.
        self.event_verification = event_verification
        self.declared_crossings = tuple(declared_crossings)
        self.route = RouteGeometry(mission.points_xy_m)
        self.owner = self.capture = self.adapter = self.vehicle = self.obstacle = None
        self._world = None
        self._settings = None
        self._weather = None
        self._spectator = None
        self._lights = []
        self._sensors = []
        # Shadow localisation (GNSS + IMU); off unless enable_localisation() is called.
        self.localisation = None
        self._localisation_seed = None
        self._loc = None
        self._truth = {}
        self._ego_box = self._obstacle_box = self._obstacle_corners = None
        self._spectator_actor = None
        self._capture_wall = {}
        self._frames = []
        self._hold_since = None
        self.events = []
        self._event_lock = threading.Lock()
        self._video = None
        self.manifest_details = {}
        self.cleanup = {"ok": False}
        self._closed = False

    def reset(self, *, seed):
        import carla

        if self.owner is not None:
            raise RuntimeError("Create a fresh CarlaWorld per episode")
        if type(seed) is not int or seed < 0:
            raise ValueError("Nonnegative seed required")
        world = self.client.get_world()
        self._world = world
        if (
            self.client.get_server_version() != "0.9.16"
            or self.client.get_client_version() != "0.9.16"
        ):
            raise ValueError("CARLA 0.9.16 required")
        assert_current_map(world.get_map(), self.mission)
        self._initial_ids = {a.id for a in world.get_actors()}
        if any(
            a.type_id.startswith(("vehicle.", "walker.", "sensor.")) for a in world.get_actors()
        ):
            raise RuntimeError(
                "Close other driving clients; world must have no vehicles, walkers or sensors"
            )
        self._settings = _settings_record(world.get_settings())
        self._weather = world.get_weather()
        self._spectator_actor = world.get_spectator()
        self._spectator = self._spectator_actor.get_transform()
        self._episode = f"carla-{seed}-{uuid.uuid4().hex}"
        self.owner = SynchronousWorldSession(
            self.client, WorldTiming(fixed_delta_seconds=self.config.dt_s)
        )
        self.owner.open()
        settings = world.get_settings()
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        for light in world.get_actors().filter("traffic.traffic_light*"):
            self._lights.append((light, light.get_state(), light.is_frozen()))
            light.set_state(carla.TrafficLightState.Green)
            light.freeze(True)
        spawn = self.mission.map_validation["spawn_actor_pose"]
        pose = carla.Transform(
            carla.Location(x=spawn["x"], y=spawn["y"], z=spawn["z"]),
            carla.Rotation(yaw=spawn["yaw_deg"]),
        )
        self.vehicle = self.owner.try_spawn(
            world.get_blueprint_library().find("vehicle.tesla.model3"), pose
        )
        if self.vehicle is None:
            raise RuntimeError("Fixed mission spawn occupied")
        # Bounding boxes are server calls; read them before cameras stream.
        self._ego_box = self.vehicle.bounding_box
        self.vehicle.apply_control(carla.VehicleControl(brake=1))
        for _ in range(20):
            self.owner.tick()
        self.adapter = FixedVehicleAdapter(
            self.vehicle, self.config.controller.max_steering_angle_rad
        )
        self._spawn_obstacle()
        for _ in range(10):
            self.owner.tick()
        self._install_event_sensors()
        if self.localisation is not None:
            self._install_localisation_sensors()
        self.capture = RGBDCapture(self.owner, self.vehicle, self._episode, capacity=16)
        self.capture.__enter__()
        rig = self.capture.rig
        self.calibration = CameraIntrinsics.from_horizontal_fov(
            rig.width, rig.height, math.radians(rig.fov_deg)
        ).calibration(
            (
                (0, 0, 1, rig.x_actor_m + self.adapter.geometry.rear_axle_offset_m),
                (-1, 0, 0, 0),
                (0, -1, 0, rig.z_actor_m),
                (0, 0, 0, 1),
            )
        )
        self.manifest_details = dict(
            server=self.client.get_server_version(),
            map=world.get_map().name,
            mission_sha256=self.mission.sha256,
            opendrive_sha256=self.mission.opendrive_sha256,
            event_sensors_verified=self._events_verified(),
            event_verification_status=(self.event_verification or {}).get("status"),
            declared_crossings=list(self.declared_crossings),
            truth_channel_complete=True,
            capture_timing=(
                "capture_apply_s: host-monotonic first sensor callback receipt for the"
                " frame to the apply_control call (board definition). tick_to_apply_s:"
                " tick submission to the confirmed application tick (report-only)."
            ),
            depth_convention=self.depth_convention,
            sensor_lag_frames=self.sensor_lag,
            # Rear-axle camera calibration, so recorded frames can be replayed
            # (RV-06 lane monitor) with exactly the rig used live.
            camera_calibration=asdict(self.calibration),
            collision_metric="contact episodes per actor; >=1 s gap splits contact",
            lane_metric=(
                "all recorded crossings count unless inside a predeclared legal"
                " (route s interval, marking types) crossing; no junction exemptions"
            ),
            margin_metric="front bumper edge to obstacle bounding-box perimeter, metres",
        )
        return self.refresh(), route_graph(self.mission.points_xy_m)

    def _spawn_obstacle(self):
        import carla

        kind = self.scenario.get("obstacle_class")
        if kind is None:
            return
        s = scenario_obstacle_s(self.mission, self.scenario)
        if s > self.route.length_m - 15:
            raise ValueError("Obstacle too near mission end")
        x, y, yaw = self.route.at(s, self.scenario.get("lateral_m", 0))
        # Search for the ground just above the road surface at this s. Searching
        # down from well above the road can land on whatever overhangs it (tree
        # canopy, a structure), and the obstacle then spawns inside that.
        road_z = float(np.interp(s, self.route.arc, [p[2] for p in self.mission.points_xyz_m]))
        site = f"s={s:.1f} m, CARLA x={x:.2f} y={-y:.2f}, road z={road_z:.2f}"
        ground = self._world.ground_projection(carla.Location(x=x, y=-y, z=road_z + 1.5), 4)
        if ground is None:
            raise RuntimeError(f"No ground at configured obstacle site ({site})")
        bp = self._world.get_blueprint_library().find(
            "walker.pedestrian.0001" if kind == "person" else "vehicle.tesla.model3"
        )
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        transform = carla.Transform(
            carla.Location(x=x, y=-y, z=ground.location.z + 1),
            carla.Rotation(yaw=-math.degrees(yaw)),
        )
        self.obstacle = self.owner.try_spawn(bp, transform)
        if self.obstacle is None:
            raise RuntimeError(
                f"Scenario obstacle spawn failed ({site}, ground z={ground.location.z:.2f}):"
                " something already occupies that spot"
            )
        self.obstacle.set_simulate_physics(False)
        self._obstacle_box = self.obstacle.bounding_box
        bottom = min(p.z for p in self._obstacle_box.get_world_vertices(transform))
        transform.location.z += ground.location.z + 0.01 - bottom
        self.obstacle.set_transform(transform)
        placement = self._placement_record(kind, s, transform)
        # Static obstacle (physics off): its footprint is fixed until removal.
        self._obstacle_corners = tuple(tuple(p) for p in placement["corners_world_m"])
        self.events.append(
            dict(
                type="obstacle_spawned",
                sim_time_s=self._world.get_snapshot().timestamp.elapsed_seconds,
                actor_id=self.obstacle.id,
                route_s_m=s,
                lateral_m=self.scenario.get("lateral_m", 0),
                **placement,
            )
        )
        if placement["in_corridor"] != placement["expected_in_corridor"]:
            raise RuntimeError(
                "Scenario obstacle placement contradicts its definition: " + str(placement)
            )

    def _placement_record(self, kind, s, transform):
        """BL-B-06: record where the actor landed and check corridor intent."""
        import carla

        from carla_adapter.ground_truth import footprint_xy

        half_width = self.config.ego_half_width_m + self.config.lateral_padding_m
        # The actor was placed this tick; get_transform() is not updated until
        # the next tick in synchronous mode, so use the pose that was applied.
        corners = footprint_xy(self.obstacle, transform, self._obstacle_box)
        interval = self.route.footprint_interval(corners, half_width_m=half_width)
        waypoint = self._world.get_map().get_waypoint(
            transform.location, project_to_road=False, lane_type=carla.LaneType.Any
        )
        lane = (
            dict(
                road_id=waypoint.road_id,
                lane_id=waypoint.lane_id,
                lane_type=str(waypoint.lane_type),
                is_junction=waypoint.is_junction,
            )
            if waypoint is not None
            else None
        )
        return dict(
            obstacle_class=kind,
            blueprint=self.obstacle.type_id,
            corners_world_m=[list(p) for p in corners],
            in_corridor=interval is not None,
            expected_in_corridor=abs(self.scenario.get("lateral_m", 0)) < half_width,
            corridor_interval_s_m=list(interval) if interval else None,
            map_lane=lane,
        )

    def _events_verified(self):
        report = self.event_verification
        return bool(
            report
            and report.get("status") == "MEASURED_PASS"
            and report.get("opendrive_sha256") == self.mission.opendrive_sha256
        )

    def enable_localisation(self, config, seed):
        """Install GNSS + IMU at reset; estimates are logged, never driven on."""
        if self.owner is not None:
            raise RuntimeError("Enable localisation before reset")
        self.localisation, self._localisation_seed = config, int(seed)

    def _install_localisation_sensors(self):
        import carla

        from carla_adapter.localisation_sensors import LiveLocalisationSensors

        carla_map = self._world.get_map()

        def geolocate(x, y):
            fix = carla_map.transform_to_geolocation(carla.Location(x=x, y=y, z=0.0))
            return fix.latitude, fix.longitude

        def spawn(blueprint_id, attributes, mount, callback):
            bp = self._world.get_blueprint_library().find(blueprint_id)
            for key, value in attributes.items():
                if not bp.has_attribute(key):
                    raise ValueError(f"{blueprint_id} has no attribute {key}")
                bp.set_attribute(key, str(value))
            sensor = self._world.spawn_actor(
                bp, carla.Transform(carla.Location(*mount)), attach_to=self.vehicle
            )
            self.owner._actors.append(sensor)
            self._sensors.append(sensor)
            sensor.listen(callback)

        self._loc = LiveLocalisationSensors(
            self.localisation,
            self._localisation_seed,
            geolocate,
            spawn,
            self.adapter.geometry.rear_axle_offset_m,
            self.config.dt_s,
        )

    def localisation_inputs(self, frame_id):
        return self._loc.inputs(frame_id)

    def localisation_record(self):
        return self._loc.record() if self._loc else dict(error="localisation sensors not installed")

    def _install_event_sensors(self):
        import carla

        for kind in ("collision", "lane_invasion"):
            bp = self._world.get_blueprint_library().find("sensor.other." + kind)
            sensor = self._world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
            self.owner._actors.append(sensor)
            self._sensors.append(sensor)

            def receive(event, channel=kind):
                row = dict(type=channel, frame_id=event.frame, sim_time_s=event.timestamp)
                if channel == "collision":
                    row.update(
                        other_actor_id=event.other_actor.id,
                        impulse=[
                            event.normal_impulse.x,
                            event.normal_impulse.y,
                            event.normal_impulse.z,
                        ],
                    )
                else:
                    row.update(
                        markings=[str(x.type) for x in event.crossed_lane_markings],
                        declared_legal=False,
                    )
                with self._event_lock:
                    self.events.append(row)

            sensor.listen(receive)

    def _tick(self):
        tick_started = time.monotonic()
        frame = self.owner.tick()
        self._capture_wall[frame] = tick_started
        state = self.adapter.state()
        pose = self.vehicle.get_transform()
        self.capture.record_snapshot(frame, EgoPose(state, pose.location.z))
        if self._loc is not None:
            self._loc.record_speed(frame, state.speed_mps)
        route_s = self.route.project(state.x_m, state.y_m).s_m
        with self._event_lock:
            for index, event in enumerate(self.events):
                if event.get("type") == "lane_invasion" and "route_s_m" not in event:
                    # Callback happened during the tick that just completed.
                    self.events[index] = annotate_invasion(event, route_s, self.declared_crossings)
        self._truth[frame] = self._truth_now(state)
        self._frames.append(frame)
        if len(self._frames) > 40:
            old = self._frames.pop(0)
            self._truth.pop(old, None)
            self._capture_wall.pop(old, None)
        if self.follow:
            import carla

            yaw = math.radians(pose.rotation.yaw)
            if self._spectator_actor is None:
                self._spectator_actor = self._world.get_spectator()
            self._spectator_actor.set_transform(
                carla.Transform(
                    carla.Location(
                        x=pose.location.x - 8 * math.cos(yaw),
                        y=pose.location.y - 8 * math.sin(yaw),
                        z=pose.location.z + 5,
                    ),
                    carla.Rotation(pitch=-22, yaw=pose.rotation.yaw),
                )
            )
        return frame

    def refresh(self):
        self.adapter.stop()
        for _ in range(self.sensor_lag + 8):
            self._tick()
        return self._read()

    def _read(self):
        frame = self._frames[-1 - self.sensor_lag]
        if self.capture.callback_errors:
            raise RuntimeError("Camera callback failure: " + str(self.capture.callback_errors))
        capture = self.capture.read(frame, 10)
        self._last_bundle = capture.to_bundle(
            self.calibration, established_depth_convention=self.depth_convention
        )
        return self._last_bundle

    def step(self, packet):
        now = self._world.get_snapshot().timestamp.elapsed_seconds
        if (
            packet.source_frame_id != self._last_bundle.frame_id
            or packet.source_sim_time_s != self._last_bundle.sim_time_s
            or not packet.is_current(self._episode, now)
        ):
            self.adapter.stop()
            raise ValueError("Foreign, expired or wrong-frame control packet")
        self.adapter.apply(packet.command)
        self.last_command_sent_monotonic_s = time.monotonic()
        self.last_simulation_age_s = now - packet.source_sim_time_s
        self._scenario_events(now)
        self._tick()
        self.last_applied_monotonic_s = time.monotonic()
        return self._read()

    def _scenario_events(self, now):
        if (
            self.obstacle is None
            or not self.obstacle.is_alive
            or self.scenario.get("remove_after_hold_s") is None
        ):
            return
        state = self.adapter.state()
        truth = self._truth_now(state)
        held = state.speed_mps <= 0.1 and truth["truth_in_corridor"] and truth["truth_gap_m"] < 8
        self._hold_since = (now if self._hold_since is None else self._hold_since) if held else None
        if (
            self._hold_since is not None
            and now - self._hold_since >= self.scenario["remove_after_hold_s"]
        ):
            if not self.obstacle.destroy():
                raise RuntimeError("Scripted removal failed")
            self.events.append(
                dict(type="obstacle_removed", sim_time_s=now, actor_id=self.obstacle.id)
            )
            self.obstacle = None

    def _truth_now(self, state):
        return truth_record(
            self.vehicle,
            self.obstacle,
            self.route,
            state,
            ego_box=self._ego_box,
            obstacle_corners=self._obstacle_corners if self.obstacle is not None else None,
        )

    def capture_monotonic(self, frame_id):
        return self._capture_wall[frame_id]

    def evaluation(self, frame_id):
        if frame_id not in self._truth:
            raise ValueError("Missing matched evaluation frame")
        return self._truth.pop(frame_id)

    def record_visual(self, frame, record):
        image = frame.rgb[:, :, ::-1].copy()
        text = f"{record['state']}  v={record['speed_mps']:.2f}  target={record['target_mps']:.2f}  CTE={record['cte_m']:.2f}"
        cv2.putText(image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        if self._video is None:
            self._video = cv2.VideoWriter(
                str(self.output / "video.avi"),
                cv2.VideoWriter_fourcc(*"MJPG"),
                1 / self.config.dt_s,
                (image.shape[1], image.shape[0]),
            )
            if not self._video.isOpened():
                raise RuntimeError("Video encoder failed")
        self._video.write(image)
        if frame.frame_id % 20 == 0:
            cv2.imwrite(str(self.output / "latest.png"), image)

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        if self.adapter:
            try:
                self.adapter.stop()
            except Exception as exc:
                errors.append(str(exc))
        if self.capture:
            try:
                self.capture.close()
            except Exception as exc:
                errors.append(str(exc))
        for sensor in self._sensors:
            try:
                if sensor.is_alive:
                    sensor.stop()
            except Exception as exc:
                errors.append(str(exc))
        for light, state, frozen in self._lights:
            try:
                light.set_state(state)
                light.freeze(frozen)
            except Exception as exc:
                errors.append(str(exc))
        if self._world is not None:
            for action in (
                lambda: (
                    self._world.set_weather(self._weather) if self._weather is not None else None
                ),
                lambda: (
                    self._world.get_spectator().set_transform(self._spectator)
                    if self._spectator is not None
                    else None
                ),
            ):
                try:
                    action()
                except Exception as exc:
                    errors.append(str(exc))
        if self.owner:
            try:
                self.owner.close()
            except Exception as exc:
                errors.append(str(exc))
        if self._video:
            self._video.release()
        restored = (
            self._settings is None or _settings_record(self._world.get_settings()) == self._settings
        )
        no_leaks = (
            not hasattr(self, "_initial_ids")
            or {a.id for a in self._world.get_actors()} == self._initial_ids
        )
        self.cleanup = dict(
            ok=not errors and restored and no_leaks,
            settings_restored=restored,
            no_actor_leaks=no_leaks,
            errors=errors,
        )
        if not self.cleanup["ok"]:
            raise RuntimeError("World cleanup incomplete: " + str(self.cleanup))
