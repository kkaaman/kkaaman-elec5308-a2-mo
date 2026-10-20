"""Run one frozen, obstacle-free S0 mission and retain auditable evidence.

The camera observes the run; navigation uses course-authorized ego truth. RGB
callbacks timestamp receipt before any image access. Bounded GPU pipelining
joins each observation to its exact ego snapshot, while commands use the most
recent world state. Frame age is reported independently from host latency.
"""

import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from autonomy.mission import MissionEpisode
from evaluation.s0_metrics import aggregate_s0_runs, evaluate_s0_run
from navigation_control import ControlStatus
from navigation_control.path import ReferencePath

from .vehicle import FixedVehicleAdapter
from .world import SynchronousWorldSession, WorldTiming, _settings_record

SAMPLE_FIELDS = (
    "simulation_time_s", "simulation_frame", "capture_frame", "capture_time_s",
    "capture_receipt_monotonic_s", "apply_monotonic_s", "capture_to_apply_ms",
    "frame_age", "real_time_factor", "x_m", "y_m", "yaw_rad", "captured_x_m",
    "captured_y_m", "speed_mps", "target_speed_mps", "signed_cte_m", "phase",
    "stable_cruise", "in_goal_window", "progress_m", "remaining_distance_m", "steering_angle_rad",
    "throttle", "brake", "status",
)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def configuration_digest(config):
    """Hash the effective configuration, including any explicit smoke override."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def execution_provenance(mission, mission_path=None):
    """Fingerprint loaded execution sources and inputs without machine identifiers."""
    repository = Path(__file__).resolve().parents[2]
    modules = (
        "carla_adapter.s0_runner", "carla_adapter.world", "carla_adapter.vehicle",
        "carla_adapter.vehicle_geometry", "evaluation.s0_metrics", "autonomy.mission",
        "navigation_control.controller", "navigation_control.path", "navigation_control.speed",
        "navigation_control.steering", "navigation_control.types", "planning.baseline_mission",
        "planning.route_manager", "planning.lane_graph",
    )
    paths = {name: Path(sys.modules[name].__file__) for name in modules if name in sys.modules}
    paths["scripts.run_scenario"] = repository / "scripts" / "run_scenario.py"
    if mission_path is not None:
        config_path = Path(mission_path).resolve()
        paths["input.mission_config"] = config_path
        for key in ("route_file", "topology_file"):
            if key in mission.config:
                paths[f"input.{key}"] = config_path.parent / mission.config[key]
    hashes, errors = {}, []
    for name, path in paths.items():
        try:
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            errors.append(f"Cannot fingerprint {name}: {type(error).__name__}")
    versions = {}
    for distribution in ("carla", "numpy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    git_head = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True,
            text=True, timeout=5, check=False,
        )
        candidate = result.stdout.strip()
        if result.returncode == 0 and len(candidate) in (40, 64) and all(
            character in "0123456789abcdef" for character in candidate
        ):
            git_head = candidate
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(), "architecture": platform.machine(),
        "installed_distributions": versions, "git_head": git_head,
        "source_sha256": hashes, "source_bundle_sha256": configuration_digest(hashes),
        "fingerprint_errors": errors,
    }


def signed_cross_track(path, x_m, y_m):
    """Measure rear-axle distance to the nearest continuous route segment."""
    projection = path.project(x_m, y_m)
    for index, end_s in enumerate(path.arc_lengths[1:]):
        if projection.progress_m <= end_s or index == len(path.points) - 2:
            a, b = path.points[index:index + 2]
            cross = (b.x_m - a.x_m) * (y_m - projection.point.y_m) - (
                b.y_m - a.y_m
            ) * (x_m - projection.point.x_m)
            return math.copysign(projection.distance_m, cross if cross else 1.0)
    raise RuntimeError("Reference path has no segment")


def route_speed_target(path, progress_m, config):
    """Apply a frozen curve ceiling from the route's forward heading change."""
    cruise = float(config["cruise_speed_mps"])
    preview = float(config.get("curve_preview_m", 15.0))
    probe = min(1.0, preview * 0.25)
    a = path.point_at(progress_m)
    b = path.point_at(progress_m + probe)
    c = path.point_at(progress_m + preview)
    d = path.point_at(progress_m + preview + probe)
    first = math.atan2(b.y_m - a.y_m, b.x_m - a.x_m)
    second = math.atan2(d.y_m - c.y_m, d.x_m - c.x_m)
    change = abs(math.atan2(math.sin(second - first), math.cos(second - first)))
    if change >= math.radians(float(config.get("curve_heading_threshold_deg", 15.0))):
        return min(cruise, float(config.get("curve_speed_mps", cruise)))
    return cruise


@dataclass(frozen=True)
class CameraReceipt:
    frame: int
    simulation_time_s: float
    received_monotonic_s: float


class RGBReceiptQueue:
    """Retain bounded real callback receipts; never reuse an observation."""

    def __init__(self, width, height, capacity=16):
        self.width, self.height, self.capacity = width, height, capacity
        self._condition = threading.Condition()
        self._receipts = deque(maxlen=capacity)
        self._retired = -1
        self._closed = False
        self.received_count = 0
        self.discarded_count = 0
        self.callback_errors = []

    def receive(self, image):
        received = time.monotonic()
        with self._condition:
            if self._closed:
                return
            try:
                if (image.width, image.height) != (self.width, self.height):
                    raise ValueError("RGB camera dimensions changed")
                if len(image.raw_data) != self.width * self.height * 4:
                    raise ValueError("RGB camera buffer is incomplete")
                if image.frame <= self._retired:
                    self.discarded_count += 1
                    return
                self._receipts.append(CameraReceipt(image.frame, image.timestamp, received))
                self.received_count += 1
            except Exception as error:
                self.callback_errors.append(f"{type(error).__name__}: {error}")
            self._condition.notify_all()

    def read_latest(self, min_frame, max_frame, timeout_s):
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self.callback_errors:
                    raise RuntimeError("RGB callback failed: " + self.callback_errors[-1])
                if self._closed:
                    raise RuntimeError("RGB receipt queue is closed")
                usable = [r for r in self._receipts if min_frame <= r.frame <= max_frame]
                if usable:
                    result = max(usable, key=lambda value: value.frame)
                    self._retired = result.frame
                    self.discarded_count += sum(r.frame < result.frame for r in self._receipts)
                    self._receipts = deque(
                        (r for r in self._receipts if r.frame > result.frame), maxlen=self.capacity
                    )
                    return result
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"No fresh RGB observation for frames {min_frame}..{max_frame}")
                self._condition.wait(remaining)

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class RunSensors:
    """Own only this episode's RGB, collision and lane-invasion sensors."""

    def __init__(self, owner, vehicle, runtime, events):
        self.owner, self.vehicle, self.events = owner, vehicle, events
        self.runtime = runtime
        self.camera = RGBReceiptQueue(runtime["camera_width"], runtime["camera_height"])
        self.actors = []
        self.callback_errors = []
        self._lock = threading.Lock()
        self._callback_condition = threading.Condition(self._lock)
        self._contacts = {}
        self._lane_sequence = 0
        self.opened = False
        self._admission_open = True
        self._finalized = False
        self._last_callback_s = time.monotonic()
        self.late_callback_records = []

    def open(self, carla):
        library = self.owner.world.get_blueprint_library()
        names = ["sensor.other.collision", "sensor.other.lane_invasion"]
        if self.runtime.get("rgb_enabled", True):
            names.insert(0, "sensor.camera.rgb")
        for name in names:
            blueprint = library.find(name)
            transform = carla.Transform()
            if name == "sensor.camera.rgb":
                for key, value in {
                    "image_size_x": self.runtime["camera_width"],
                    "image_size_y": self.runtime["camera_height"],
                    "fov": 90, "sensor_tick": 0.0,
                    "enable_postprocess_effects": "false",
                }.items():
                    blueprint.set_attribute(key, str(value))
                transform = carla.Transform(carla.Location(x=1.5, z=2.0))
            sensor = self.owner.world.spawn_actor(
                blueprint, transform, attach_to=self.vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
            self.actors.append(sensor)
            self.owner._actors.append(sensor)
            callback = {
                "sensor.camera.rgb": self.camera.receive,
                "sensor.other.collision": self._collision,
                "sensor.other.lane_invasion": self._lane,
            }[name]
            sensor.listen(callback)
        self.opened = True
        return self

    def _collision(self, event):
        with self._lock:
            if not self._admit_event("collision", event):
                return
            try:
                actor_id = str(event.other_actor.id)
                old = self._contacts.get(actor_id)
                # Separate re-contact after >0.5 simulated seconds; preserve all callbacks.
                start = event.frame if old is None or event.timestamp - old[1] > 0.5 else old[0]
                self._contacts[actor_id] = (start, event.timestamp)
                impulse = event.normal_impulse
                self.events.append({
                    "kind": "collision", "frame": event.frame,
                    "simulation_time_s": event.timestamp,
                    "contact_id": f"{actor_id}:{start}", "other_actor_id": actor_id,
                    "other_actor_type": event.other_actor.type_id,
                    "normal_impulse": {"x": impulse.x, "y": impulse.y, "z": impulse.z},
                })
            except Exception as error:
                self.callback_errors.append(f"Collision callback: {error}")

    def _lane(self, event):
        with self._lock:
            if not self._admit_event("lane_invasion", event):
                return
            try:
                self._lane_sequence += 1
                self.events.append({
                    "kind": "lane_invasion", "frame": event.frame,
                    "simulation_time_s": event.timestamp,
                    "event_id": f"{event.frame}:{self._lane_sequence}",
                    "crossed_lane_markings": [str(mark.type) for mark in event.crossed_lane_markings],
                    "predeclared_legal": False,
                })
            except Exception as error:
                self.callback_errors.append(f"Lane invasion callback: {error}")

    def _admit_event(self, kind, event):
        """Called under the callback lock; closed callbacks never alter event history."""
        self._last_callback_s = time.monotonic()
        self._callback_condition.notify_all()
        if self._admission_open:
            return True
        fence = "finalization" if self._finalized else "admission fence"
        self.callback_errors.append(f"Late {kind} callback after {fence}")
        self.late_callback_records.append({
            "kind": "late_sensor_callback", "original_kind": kind,
            "frame": event.frame, "simulation_time_s": event.timestamp,
        })
        return False

    def close(self, *, reachable=True):
        errors = []
        actors = self.actors if reachable else []
        if not reachable:
            errors.append("Sensor cleanup unknown: server unreachable; no further RPC attempted")
        for actor in reversed(actors):
            try:
                if actor.is_alive:
                    actor.stop()
                    if not actor.destroy():
                        errors.append(f"Sensor destroy returned false: {actor.id}")
            except Exception as error:
                errors.append(f"Sensor cleanup {actor.id}: {error}")
        # Stop/destroy acknowledgement, callback-lock barrier, then at most 100ms
        # of callback quiescence. This is the declared stream-admission fence;
        # it is not a claim about undelivered data from an unreachable server.
        with self._callback_condition:
            deadline = time.monotonic() + (0.1 if reachable else 0)
            while reachable and time.monotonic() - self._last_callback_s < 0.05:
                remaining = min(deadline - time.monotonic(), 0.05)
                if remaining <= 0:
                    errors.append("Sensor callbacks did not quiesce before the bounded fence")
                    break
                self._callback_condition.wait(remaining)
            self._admission_open = False
            self.callback_errors.extend(self.camera.callback_errors)
        self.camera.close()
        self.actors.clear()
        return errors

    def snapshot_events(self):
        """Freeze a deep event/error snapshot after all cleanup and the callback fence."""
        with self._lock:
            self._finalized = True
            return copy.deepcopy(self.events + self.late_callback_records), list(self.callback_errors)


class FrozenEnvironment:
    """Temporarily set clear weather and green lights, then restore both."""

    def __init__(self, world):
        self.world = world
        self.weather = world.get_weather()
        self.weather_record = _settings_record(self.weather)
        self.lights = []
        self.restore_evidence = {}

    def apply(self, carla):
        # CARLA freeze() affects every traffic light. Capture all initial
        # states/flags before the first global mutation.
        self.lights = [(light, light.get_state(), light.is_frozen())
                       for light in self.world.get_actors().filter("traffic.traffic_light*")]
        if len({frozen for _, _, frozen in self.lights}) > 1:
            raise RuntimeError("Initial global traffic-light freeze flags are inconsistent")
        if self.lights:
            self.lights[0][0].freeze(True)
        for light, _, _ in self.lights:
            light.set_state(carla.TrafficLightState.Green)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def restore(self, advance=None):
        """Verify restored states while frozen, then resume the original light cycle.

        State getters reflect the latest world snapshot. The active tick owner
        supplies ``advance`` so each verification observes acknowledged changes.
        Unfrozen lights may legitimately change state after cycles are resumed.
        """
        errors = []
        evidence = {"frozen_state_checks": [], "freeze_flag_checks": [], "refresh_frames": []}
        self.restore_evidence = evidence
        consistent = len({frozen for _, _, frozen in self.lights}) <= 1
        evidence["initial_freeze_flags_consistent"] = consistent
        if not consistent:
            return ["Cannot restore inconsistent global traffic-light freeze flags"]
        try:
            if self.lights:
                self.lights[0][0].freeze(True)
            for light, state, _ in self.lights:
                light.set_state(state)
            self.world.set_weather(self.weather)
            if advance is not None:
                evidence["refresh_frames"].append(advance())
            for light, state, _ in self.lights:
                actual, frozen = light.get_state(), light.is_frozen()
                evidence["frozen_state_checks"].append({
                    "actor_id": light.id, "expected_state": str(state),
                    "actual_state": str(actual), "frozen": frozen,
                })
                if actual != state or not frozen:
                    errors.append(f"Frozen traffic-light state did not restore: {light.id}")
        except Exception as error:
            errors.append(f"Frozen environment restoration: {error}")
        try:
            if self.lights:
                self.lights[0][0].freeze(self.lights[0][2])
            if advance is not None:
                evidence["refresh_frames"].append(advance())
            for light, state, frozen in self.lights:
                actual_frozen = light.is_frozen()
                evidence["freeze_flag_checks"].append({
                    "actor_id": light.id, "expected_frozen": frozen,
                    "actual_frozen": actual_frozen,
                })
                if actual_frozen != frozen:
                    errors.append(f"Traffic-light freeze flag did not restore: {light.id}")
                if frozen and light.get_state() != state:
                    errors.append(f"Originally frozen traffic-light state changed: {light.id}")
            if _settings_record(self.world.get_weather()) != self.weather_record:
                errors.append("Restored weather differs from the original parameters")
            evidence["weather_restored"] = _settings_record(self.world.get_weather()) == self.weather_record
        except Exception as error:
            errors.append(f"Environment restore verification: {error}")
        return errors


def _foreign_actors(world):
    return [a for a in world.get_actors() if a.type_id.startswith(("vehicle.", "walker.", "sensor."))]


def run_s0_episode(mission, output, *, host="127.0.0.1", port=2000,
                   max_simulation_seconds=None, hold_at_progress=None, hold_duration=2.0,
                   mission_path=None, navigation_only=False):
    """Run and save one episode, returning a summary even on connection failure."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(mission.config)
    runtime = config["runtime"]
    if navigation_only:
        runtime["rgb_enabled"] = False
    if max_simulation_seconds is not None:
        runtime["max_simulation_seconds"] = max_simulation_seconds
    if hold_at_progress is not None:
        runtime["injected_hold"] = {"progress_m": hold_at_progress, "duration_s": hold_duration}
    stable_target_duration = float(config.get("measurement", {}).get("stable_target_duration_s", 2.0))
    config_sha256 = mission.config_sha256 if config == mission.config else configuration_digest(
        {"config": config, "route_sha256": mission.route_sha256}
    )
    manifest = {
        "scenario": "S0", "run_id": output.name, "status": "RUNNING",
        "route_sha256": mission.route_sha256, "config_sha256": config_sha256,
        "effective_config": config, "host": host, "port": port,
        "formal_acceptance": hold_at_progress is None and max_simulation_seconds is None and runtime.get("rgb_enabled", True),
        "navigation_only_diagnostic": not runtime.get("rgb_enabled", True),
        "navigation_source": "course-authorized rear-axle ego truth",
        "latency_definition": "RGB callback receipt before buffer access to checked apply_batch_sync acknowledgement (do_tick=False)",
        "camera_pipeline_max_frames": 4,
        "cruise_window": f"cruise target unchanged for >={stable_target_duration:g} simulation seconds; never filter measured error",
        "collision_deduplication": "Same other actor with callbacks <=0.5 simulation seconds apart is one contact; later re-contact is new",
        "legal_lane_manoeuvres": [], "errors": [],
        "execution_provenance": execution_provenance(mission, mission_path),
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "events.json", [])
    samples, events = [], []
    client = owner = sensors = environment = world = episode = None
    initial_settings = initial_ids = None
    failure = None
    cleanup_errors = []
    completed = False
    with (output / "samples.csv").open("w", newline="", encoding="utf-8") as sample_file:
        writer = csv.DictWriter(sample_file, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        sample_file.flush()
        try:
            import carla

            client = carla.Client(host, port)
            client.set_timeout(float(runtime.get("capture_timeout_s", 10.0)))
            world = client.get_world()
            initial_settings = _settings_record(world.get_settings())
            initial_ids = {a.id for a in world.get_actors()}
            manifest.update(server_version=client.get_server_version(), client_version=client.get_client_version())
            actual_map = world.get_map()
            manifest["actual_map"] = actual_map.name
            actual_digest = hashlib.sha256(actual_map.to_opendrive().encode("utf-8")).hexdigest()
            manifest["actual_opendrive_sha256"] = actual_digest
            actual_basename = actual_map.name.rsplit("/", 1)[-1].removesuffix("_Opt")
            if actual_basename != mission.map_name.removesuffix("_Opt"):
                raise RuntimeError(f"Expected map {mission.map_name}, received {actual_map.name}")
            if actual_digest != mission.opendrive_sha256:
                raise RuntimeError("Installed map OpenDRIVE hash differs from the frozen route")
            if runtime.get("rgb_enabled", True) and world.get_settings().no_rendering_mode:
                raise RuntimeError("RGB evidence requires rendering; no_rendering_mode is enabled")
            foreign = _foreign_actors(world)
            if foreign:
                raise RuntimeError("Stop other clients first; existing vehicles/walkers/sensors: " + str([a.id for a in foreign]))
            owner = SynchronousWorldSession(
                client, WorldTiming(fixed_delta_seconds=float(runtime["fixed_delta_seconds"])),
                ownership_key=f"{host}:{port}",
            )
            owner.open()
            map_layers = runtime.get("map_layers", "ALL")
            if map_layers not in ("NONE", "ALL"):
                raise ValueError("runtime.map_layers must be NONE or ALL")
            manifest.update(
                world_episode_id_before=world.id, declared_map_layers=map_layers,
                world_reload_policy="Enable synchronous fixed-step settings before load_world(same_map_basename, reset_settings=False, declared_map_layers)",
            )
            layers = carla.MapLayer.NONE if map_layers == "NONE" else carla.MapLayer.All
            world = client.load_world(actual_map.name.rsplit("/", 1)[-1], reset_settings=False, map_layers=layers)
            owner.world = world
            initial_ids = {a.id for a in world.get_actors()}
            manifest["world_episode_id_after"] = world.id
            environment = FrozenEnvironment(world)
            environment.apply(carla)
            manifest["environment"] = {
                "weather": "ClearNoon", "traffic_light_policy": "frozen green",
                "traffic_light_count": len(environment.lights),
                "original_weather": environment.weather_record,
                "seed": runtime["seed"],
                "seed_scope": "No dynamic actors, random route choices or Traffic Manager are used",
            }
            spawn = mission.spawn_pose
            transform = carla.Transform(
                carla.Location(x=spawn["x_m"], y=-spawn["y_m"], z=spawn["z_m"]),
                carla.Rotation(yaw=-math.degrees(spawn["yaw_rad"])),
            )
            blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
            blueprint.set_attribute("role_name", "baseline-s0")
            vehicle = owner.try_spawn(blueprint, transform)
            if vehicle is None:
                raise RuntimeError("The frozen mission spawn is occupied")
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
            for _ in range(int(runtime["settle_ticks"])):
                owner.tick()
            adapter = FixedVehicleAdapter(vehicle, mission.controller_config.max_steering_angle_rad)
            manifest["vehicle_geometry"] = asdict(adapter.geometry)
            episode = MissionEpisode(mission.controller_config, mission.waypoints, mission.route_sha256)
            path = ReferencePath(mission.waypoints)
            sensors = RunSensors(owner, vehicle, runtime, events)
            sensors.open(carla)
            snapshots = {}
            for _ in range(6):
                frame = owner.tick()
                snapshot = world.get_snapshot()
                snapshots[frame] = (snapshot.timestamp.elapsed_seconds, adapter.state())
            started = world.get_snapshot().timestamp.elapsed_seconds
            wall_started = time.monotonic()
            target_started = started
            previous_target = None
            hold_started = None
            hold_done = False
            while True:
                frame = owner.tick()
                snapshot = world.get_snapshot()
                if snapshot.frame != frame:
                    raise RuntimeError("World snapshot frame differs from the owned tick")
                simulation_time = snapshot.timestamp.elapsed_seconds
                state = adapter.state()
                snapshots[frame] = (simulation_time, state)
                receipt = captured_state = capture_time = None
                if runtime.get("rgb_enabled", True):
                    receipt = sensors.camera.read_latest(frame - 4, frame, float(runtime["capture_timeout_s"]))
                    if receipt.frame not in snapshots:
                        raise RuntimeError("RGB observation has no exact ego snapshot")
                    capture_time, captured_state = snapshots[receipt.frame]
                    if not math.isclose(capture_time, receipt.simulation_time_s, abs_tol=1e-6):
                        raise RuntimeError("RGB observation and ego snapshot timestamps differ")
                target = route_speed_target(path, episode.progress_m, config)
                if hold_at_progress is not None and not hold_done and hold_started is None and episode.progress_m >= hold_at_progress:
                    hold_started = simulation_time
                    events.append({"kind": "injected_hold_started", "simulation_time_s": simulation_time, "progress_m": episode.progress_m})
                hold = hold_started is not None and simulation_time - hold_started < hold_duration
                if hold_started is not None and not hold and not hold_done:
                    hold_done = True
                    events.append({"kind": "injected_hold_released", "simulation_time_s": simulation_time, "progress_m": episode.progress_m})
                result = episode.step(state, target, float(runtime["fixed_delta_seconds"]), hold=hold)
                adapter.apply(result.command, client=client)
                applied = time.monotonic()
                internal_target = result.diagnostics.internal_target_speed_mps
                if result.status == ControlStatus.FINISHED:
                    phase = "goal"
                elif hold:
                    phase = "hold"
                elif target < config["cruise_speed_mps"] - 1e-9 or internal_target < target - 1e-9:
                    phase = "approach"
                else:
                    phase = "cruise"
                target_key = (phase, internal_target)
                if target_key != previous_target:
                    target_started = simulation_time
                    previous_target = target_key
                sample = {
                    "simulation_time_s": simulation_time, "simulation_frame": frame,
                    "capture_frame": receipt.frame if receipt else None, "capture_time_s": capture_time,
                    "capture_receipt_monotonic_s": receipt.received_monotonic_s if receipt else None,
                    "apply_monotonic_s": applied if receipt else None,
                    "capture_to_apply_ms": (applied - receipt.received_monotonic_s) * 1000 if receipt else None,
                    "frame_age": frame - receipt.frame if receipt else None,
                    "real_time_factor": (simulation_time - started) / max(applied - wall_started, 1e-9),
                    "x_m": state.x_m, "y_m": state.y_m, "yaw_rad": state.yaw_rad,
                    "captured_x_m": captured_state.x_m if captured_state else None,
                    "captured_y_m": captured_state.y_m if captured_state else None,
                    "speed_mps": state.speed_mps, "target_speed_mps": internal_target,
                    "signed_cte_m": signed_cross_track(path, state.x_m, state.y_m),
                    "phase": phase, "stable_cruise": phase == "cruise" and simulation_time - target_started >= stable_target_duration,
                    "in_goal_window": (
                        result.diagnostics.distance_to_goal_m <= mission.controller_config.goal_tolerance_m
                        and result.diagnostics.remaining_distance_m <= mission.controller_config.goal_tolerance_m
                    ),
                    "progress_m": result.diagnostics.progress_m,
                    "remaining_distance_m": result.diagnostics.remaining_distance_m,
                    "steering_angle_rad": result.command.steering_angle_rad,
                    "throttle": result.command.throttle, "brake": result.command.brake,
                    "status": result.status.value,
                }
                samples.append(sample)
                writer.writerow(sample)
                sample_file.flush()
                snapshots = {key: value for key, value in snapshots.items() if key >= frame - 8}
                if result.status == ControlStatus.FINISHED:
                    completed = True
                    break
                if simulation_time - started >= runtime["max_simulation_seconds"]:
                    raise TimeoutError("Mission exceeded the frozen simulation-time limit")
                if sensors.callback_errors:
                    raise RuntimeError("Sensor callback failed: " + sensors.callback_errors[-1])
            adapter.stop()
            for _ in range(6):
                owner.tick()
            manifest["terminal_frame"] = world.get_snapshot().frame
        except BaseException as error:
            failure = f"{type(error).__name__}: {error}"
            manifest["errors"].append(traceback.format_exc())
            if episode is not None:
                episode.fail(failure)
        finally:
            reachable = True
            if client is not None and world is not None:
                try:
                    client.set_timeout(2.0)
                    # get_world() can return a cached episode facade. Settings
                    # require an actual synchronous server RPC on this facade.
                    world.get_settings()
                    manifest["cleanup_server_reachable"] = True
                except Exception as error:
                    reachable = False
                    manifest["cleanup_server_reachable"] = False
                    cleanup_errors.append(f"Server unreachable; actor/settings/environment cleanup unknown: {error}")
            if owner is not None and reachable:
                cleanup_errors.extend(owner.emergency_stop())
                if not completed and owner._active and sensors is not None and sensors.opened:
                    owner.timing = WorldTiming(
                        fixed_delta_seconds=owner.timing.fixed_delta_seconds,
                        tick_timeout_seconds=2.0,
                    )
                    try:
                        for _ in range(6):
                            owner.tick()
                    except Exception as error:
                        cleanup_errors.append(f"Terminal sensor drain incomplete: {error}")
            if sensors is not None:
                cleanup_errors.extend(sensors.close(reachable=reachable))
            if environment is not None and reachable:
                environment_errors = environment.restore(advance=owner.tick if owner is not None and owner._active else None)
                manifest["environment_restored"] = not environment_errors
                manifest["environment_restore_evidence"] = environment.restore_evidence
                cleanup_errors.extend(environment_errors)
            if owner is not None:
                try:
                    if not reachable:
                        # Release only this process's ownership bookkeeping.
                        # Live actor/settings restoration remains explicitly unknown.
                        owner._actors.clear()
                        owner._original_settings = None
                    owner.close()
                except Exception as error:
                    cleanup_errors.append(str(error))
            if not reachable:
                manifest.update(settings_restored=None, actor_inventory_restored=None, environment_restored=None)
            if reachable and world is not None and initial_settings is not None:
                try:
                    manifest["settings_restored"] = _settings_record(world.get_settings()) == initial_settings
                    manifest["actor_inventory_restored"] = {a.id for a in world.get_actors()} == initial_ids
                    if not manifest["settings_restored"] or not manifest["actor_inventory_restored"]:
                        cleanup_errors.append("World settings or actor inventory did not restore")
                except Exception as error:
                    cleanup_errors.append(f"Cleanup verification: {error}")
            if sensors is not None:
                manifest["camera_receipts"] = sensors.camera.received_count
                manifest["camera_discarded_receipts"] = sensors.camera.discarded_count
            manifest["cleanup_errors"] = cleanup_errors
            final_provenance = execution_provenance(mission, mission_path)
            manifest["final_execution_provenance"] = final_provenance
            manifest["source_unchanged_during_run"] = (
                manifest["execution_provenance"]["source_bundle_sha256"] == final_provenance["source_bundle_sha256"]
            )
            if not manifest["source_unchanged_during_run"]:
                failure = failure or "Execution source or input files changed during the episode"
                manifest["errors"].append("Execution source or input files changed during the episode")
            callback_errors = []
            if sensors is not None:
                events, callback_errors = sensors.snapshot_events()
                manifest["sensor_callback_errors"] = callback_errors
            if episode is not None:
                events.extend({"kind": event.event, "step_index": event.step_index, "route_sha256": event.route_sha256} for event in episode.events)
            streams_complete = sensors is not None and sensors.opened and not callback_errors and not cleanup_errors
            events.append({"kind": "event_stream_status", "collision_complete": streams_complete,
                           "lane_invasion_complete": streams_complete})
            manifest["sensor_finalization_fence"] = "Terminal braking/drain; stop/destroy acknowledgement; callback-lock barrier and bounded quiescence; immutable snapshot after full cleanup"
            manifest["failure_reason"] = failure
            manifest["route_install_count"] = episode.route_install_count if episode else 0
            summary = evaluate_s0_run(
                samples, events, completed=completed, failure_reason=failure,
                route_install_count=manifest["route_install_count"], cleanup_ok=not cleanup_errors,
            )
            summary.update(run_id=output.name, route_sha256=mission.route_sha256,
                           config_sha256=config_sha256, formal_acceptance=manifest["formal_acceptance"])
            manifest["status"] = summary["status"]
            _write_json(output / "events.json", events)
            _write_json(output / "manifest.json", manifest)
            _write_json(output / "summary.json", summary)
    return summary


def run_s0_suite(mission, output, *, runs=1, **kwargs):
    """Run independent episodes sequentially and preserve every failed trial."""
    if isinstance(runs, bool) or not isinstance(runs, int) or not 1 <= runs <= 5:
        raise ValueError("runs must be an integer from 1 to 5")
    if runs == 5 and (kwargs.get("hold_at_progress") is not None or kwargs.get("max_simulation_seconds") is not None
                      or kwargs.get("navigation_only") or not mission.config["runtime"].get("rgb_enabled", True)):
        raise ValueError("Formal five-run acceptance requires the frozen configuration")
    if runs != 1 and (kwargs.get("navigation_only") or not mission.config["runtime"].get("rgb_enabled", True)):
        raise ValueError("Navigation-only diagnostics require exactly one run")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for index in range(runs):
        summary = run_s0_episode(mission, output / f"run-{index + 1:02d}", **kwargs)
        summaries.append(summary)
        _write_json(output / "suite-progress.json", {"requested_runs": runs, "completed_runs": len(summaries), "runs": summaries})
    aggregate = aggregate_s0_runs(summaries)
    _write_json(output / "summary.json", aggregate)
    return aggregate
