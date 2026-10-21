"""Evidence retention, callback timing and bounded pipeline tests without CARLA."""

import copy
import csv
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from carla_adapter import s0_runner
from navigation_control import ControllerConfig, VehicleState, Waypoint
from navigation_control.path import ReferencePath


def test_signed_cte_continuous_projection_and_endpoints():
    path = ReferencePath([Waypoint(0, 0), Waypoint(10, 0), Waypoint(10, 10)])
    assert s0_runner.signed_cross_track(path, 4, 2) == pytest.approx(2)
    assert s0_runner.signed_cross_track(path, 4, -2) == pytest.approx(-2)
    assert s0_runner.signed_cross_track(path, 12, 5) == pytest.approx(-2)
    assert abs(s0_runner.signed_cross_track(path, -3, 4)) == pytest.approx(5)


def test_curve_speed_cap_depends_only_on_frozen_route():
    path = ReferencePath([Waypoint(0, 0), Waypoint(20, 0), Waypoint(20, 30)])
    config = dict(cruise_speed_mps=4, curve_speed_mps=2.5,
                  curve_preview_m=15, curve_heading_threshold_deg=15)
    assert s0_runner.route_speed_target(path, 0, config) == 4
    assert s0_runner.route_speed_target(path, 10, config) == 2.5


def test_receipt_timestamp_precedes_buffer_access(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(s0_runner.time, "monotonic", lambda: now[0])

    class Image:
        width, height, frame, timestamp = 2, 1, 10, 0.5

        @property
        def raw_data(self):
            now[0] = 110.0
            return bytes(8)

    queue = s0_runner.RGBReceiptQueue(2, 1)
    queue.receive(Image())
    receipt = queue.read_latest(10, 10, 1)
    assert receipt.received_monotonic_s == 100
    assert now[0] == 110


def test_receipts_select_newest_and_cannot_be_reused():
    queue = s0_runner.RGBReceiptQueue(2, 1, capacity=3)
    for frame in range(1, 6):
        queue.receive(SimpleNamespace(width=2, height=1, frame=frame,
                                      timestamp=frame * 0.05, raw_data=bytes(8)))
    assert queue.read_latest(3, 5, 0.01).frame == 5
    with pytest.raises(TimeoutError, match="No fresh RGB"):
        queue.read_latest(3, 5, 0.001)
    queue.close()
    with pytest.raises(RuntimeError, match="closed"):
        queue.read_latest(3, 5, 0.01)


def test_invalid_camera_buffer_fails_instead_of_reusing_old_image():
    queue = s0_runner.RGBReceiptQueue(2, 1)
    queue.receive(SimpleNamespace(width=2, height=1, frame=1, timestamp=0.05, raw_data=b""))
    with pytest.raises(RuntimeError, match="buffer is incomplete"):
        queue.read_latest(1, 1, 0.01)


def test_collision_deduplication_retains_raw_recontacts():
    sensors = s0_runner.RunSensors(None, None, dict(camera_width=2, camera_height=1), [])
    for frame, timestamp in [(10, 0.5), (11, 0.55), (25, 1.25)]:
        sensors._collision(SimpleNamespace(
            frame=frame, timestamp=timestamp,
            other_actor=SimpleNamespace(id=17, type_id="vehicle.other"),
            normal_impulse=SimpleNamespace(x=1, y=0, z=0),
        ))
    assert [event["contact_id"] for event in sensors.events] == ["17:10", "17:10", "17:25"]
    assert len(sensors.events) == 3


def test_closed_callback_is_retained_as_error_without_mutating_frozen_snapshot():
    events = []
    sensors = s0_runner.RunSensors(None, None, dict(camera_width=2, camera_height=1), events)
    sensors.close(reachable=False)
    late = SimpleNamespace(frame=10, timestamp=0.5)
    sensors._collision(late)
    snapshot, errors = sensors.snapshot_events()
    assert errors == ["Late collision callback after admission fence"]
    assert snapshot[0]["kind"] == "late_sensor_callback"
    assert events == []
    sensors._lane(SimpleNamespace(frame=11, timestamp=0.55))
    assert len(snapshot) == 1
    assert len(errors) == 1
    assert sensors.callback_errors[-1] == "Late lane_invasion callback after finalization"


def test_navigation_diagnostic_spawns_incident_sensors_without_rgb():
    requested = []
    owned = []

    def find(name):
        requested.append(name)
        return SimpleNamespace()

    sensor = SimpleNamespace(listen=lambda callback: None)
    owner = SimpleNamespace(
        world=SimpleNamespace(
            get_blueprint_library=lambda: SimpleNamespace(find=find),
            spawn_actor=lambda *args, **kwargs: sensor,
        ),
        _actors=owned,
    )
    sensors = s0_runner.RunSensors(owner, object(), dict(camera_width=2, camera_height=1,
                                                       rgb_enabled=False), [])
    carla = SimpleNamespace(Transform=lambda: object(), AttachmentType=SimpleNamespace(Rigid="Rigid"))
    sensors.open(carla)
    assert requested == ["sensor.other.collision", "sensor.other.lane_invasion"]
    assert len(owned) == 2
    assert sensors.opened is True
    sensors.close(reachable=False)


@pytest.mark.parametrize("original_frozen", [False, True])
def test_environment_restoration_refreshes_snapshot_before_state_verification(original_frozen):
    original_states = ["Red", "Yellow"]
    authoritative = list(original_states)
    snapshot_states = list(original_states)
    freeze_flags = [original_frozen, original_frozen]
    snapshot_flags = list(freeze_flags)
    freeze_calls = []
    frame = 0
    weather = ["OriginalWeather"]

    class Light:
        def __init__(self, index):
            self.index, self.id = index, index + 1

        def get_state(self):
            return snapshot_states[self.index]

        def is_frozen(self):
            return snapshot_flags[self.index]

        def set_state(self, state):
            authoritative[self.index] = state

        def freeze(self, frozen):
            freeze_calls.append(frozen)
            freeze_flags[:] = [frozen, frozen]

    lights = [Light(0), Light(1)]

    def advance():
        nonlocal frame
        frame += 1
        # Unfrozen lights may legitimately cycle before the next cached snapshot.
        if not freeze_flags[0]:
            authoritative[:] = ["AutomaticCycle", "AutomaticCycle"]
        snapshot_states[:] = authoritative
        snapshot_flags[:] = freeze_flags
        return frame

    world = SimpleNamespace(
        get_weather=lambda: weather[0],
        set_weather=lambda value: weather.__setitem__(0, value),
        get_actors=lambda: SimpleNamespace(filter=lambda pattern: lights),
    )
    carla = SimpleNamespace(TrafficLightState=SimpleNamespace(Green="Green"),
                            WeatherParameters=SimpleNamespace(ClearNoon="ClearNoon"))
    environment = s0_runner.FrozenEnvironment(world)
    environment.apply(carla)
    assert [flag for _, _, flag in environment.lights] == [original_frozen, original_frozen]
    advance()
    assert [light.get_state() for light in lights] == ["Green", "Green"]
    assert environment.restore(advance=advance) == []
    assert freeze_calls == [True, True, original_frozen]
    assert snapshot_flags == [original_frozen, original_frozen]
    assert weather[0] == "OriginalWeather"
    assert [record["actual_state"] for record in environment.restore_evidence["frozen_state_checks"]] == original_states
    assert environment.restore_evidence["refresh_frames"] == [2, 3]
    if original_frozen:
        assert snapshot_states == original_states
    else:
        assert snapshot_states == ["AutomaticCycle", "AutomaticCycle"]


def mission():
    config = {
        "runtime": dict(fixed_delta_seconds=0.05, camera_width=2, camera_height=1,
                        max_simulation_seconds=0.1, settle_ticks=2, capture_timeout_s=0.1,
                        seed=42, map_layers="NONE"),
        "cruise_speed_mps": 4,
    }
    return SimpleNamespace(
        config=config, config_sha256=s0_runner.configuration_digest(config),
        route_sha256="a" * 64, map_name="Town10HD",
        opendrive_sha256=s0_runner.hashlib.sha256(b"map").hexdigest(),
        controller_config=ControllerConfig(wheelbase_m=3.00465, max_steering_angle_rad=0.3),
        waypoints=(Waypoint(0, 0), Waypoint(100, 0)),
        spawn_pose=dict(x_m=1.38641, y_m=0, z_m=0.5, yaw_rad=0),
    )


def test_connection_failure_still_retains_all_evidence_files(tmp_path, monkeypatch):
    def unavailable(host, port):
        raise ConnectionError("Test server unavailable")

    monkeypatch.setitem(sys.modules, "carla", SimpleNamespace(Client=unavailable))
    output = tmp_path / "connection-failure"
    summary = s0_runner.run_s0_episode(mission(), output)
    assert summary["status"] == "FAIL"
    assert "ConnectionError" in summary["failure_reason"]
    assert {p.name for p in output.iterdir()} == {"manifest.json", "events.json", "samples.csv", "summary.json"}
    marker = json.loads((output / "events.json").read_text())[-1]
    assert marker["collision_complete"] is False
    assert marker["lane_invasion_complete"] is False
    assert "Test server unavailable" in (output / "manifest.json").read_text()
    with pytest.raises(FileExistsError):
        s0_runner.run_s0_episode(mission(), output)


def test_formal_suite_rejects_smoke_or_hold_overrides_before_creating_output(tmp_path):
    for override in ({"max_simulation_seconds": 1}, {"hold_at_progress": 10}, {"navigation_only": True}):
        with pytest.raises(ValueError, match="frozen configuration"):
            s0_runner.run_s0_suite(mission(), tmp_path / "formal", runs=5, **override)
    assert not (tmp_path / "formal").exists()


@dataclass
class Geometry:
    wheelbase_m: float = 3.00465


class FakeWorld:
    def __init__(self):
        self.id = 1
        self.settings = SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None,
                                        substepping=False, max_substep_delta_time=0.01,
                                        max_substeps=10, no_rendering_mode=False)
        self.weather = "OriginalWeather"
        self.actors = []
        self.frame = 0
        self.transform = None

    def get_settings(self):
        return copy.deepcopy(self.settings)

    def apply_settings(self, settings):
        self.settings = copy.deepcopy(settings)

    def get_weather(self):
        return self.weather

    def set_weather(self, weather):
        self.weather = weather

    def get_actors(self):
        class Actors(list):
            def filter(self, pattern):
                return []
        return Actors(a for a in self.actors if a.is_alive)

    def get_map(self):
        return SimpleNamespace(name="/Game/Maps/Town10HD_Opt", to_opendrive=lambda: "map")

    def get_blueprint_library(self):
        return SimpleNamespace(find=lambda _: SimpleNamespace(set_attribute=lambda *args: None, id="vehicle.tesla.model3"))

    def try_spawn_actor(self, blueprint, transform):
        self.transform = transform
        actor = SimpleNamespace(id=1, type_id="vehicle.tesla.model3", is_alive=True)
        actor.apply_control = lambda control: None

        def destroy():
            actor.is_alive = False
            return True
        actor.destroy = destroy
        self.actors.append(actor)
        return actor

    def tick(self, timeout):
        self.frame += 1
        return self.frame

    def get_snapshot(self):
        return SimpleNamespace(frame=self.frame, timestamp=SimpleNamespace(elapsed_seconds=self.frame * 0.05))

    def load(self, map_name, *, reset_settings, map_layers):
        assert self.settings.synchronous_mode
        assert self.settings.fixed_delta_seconds == 0.05
        assert not self.actors
        self.reload_call = (map_name, reset_settings, map_layers)
        self.id += 1
        self.frame = 0
        return self


def fake_live(monkeypatch, world):
    client = SimpleNamespace(get_world=lambda: world, set_timeout=lambda _: None,
                             get_server_version=lambda: "0.9.16", get_client_version=lambda: "0.9.16",
                             load_world=world.load)
    carla = SimpleNamespace(
        Client=lambda host, port: client,
        Transform=lambda location=None, rotation=None: SimpleNamespace(location=location, rotation=rotation),
        Location=lambda **kwargs: SimpleNamespace(**kwargs),
        Rotation=lambda **kwargs: SimpleNamespace(**kwargs),
        VehicleControl=lambda **kwargs: SimpleNamespace(**kwargs),
        WeatherParameters=SimpleNamespace(ClearNoon="ClearNoon"),
        MapLayer=SimpleNamespace(NONE="NONE", All="ALL"),
    )
    monkeypatch.setitem(sys.modules, "carla", carla)

    class Adapter:
        geometry = Geometry()

        def __init__(self, vehicle, limit):
            pass

        def state(self):
            return VehicleState(0, 0, 0, 0)

        def apply(self, command, *, client=None):
            pass

        def stop(self):
            pass

    class Sensors:
        callback_errors = []
        opened = True

        def __init__(self, owner, vehicle, runtime, events):
            self.events = events
            self.camera = SimpleNamespace(read_latest=self.read_latest, received_count=0,
                                          discarded_count=0)

        def open(self, carla):
            pass

        def read_latest(self, min_frame, max_frame, timeout_s):
            return s0_runner.CameraReceipt(max_frame - 2, (max_frame - 2) * 0.05, time.monotonic())

        def close(self, *, reachable=True):
            return []

        def snapshot_events(self):
            return list(self.events), []
    monkeypatch.setattr(s0_runner, "FixedVehicleAdapter", Adapter)
    monkeypatch.setattr(s0_runner, "RunSensors", Sensors)


def test_timeout_keeps_samples_restores_world_and_hashes_override(tmp_path, monkeypatch):
    world = FakeWorld()
    before = copy.deepcopy(world.settings)
    fake_live(monkeypatch, world)
    frozen = mission()
    frozen.spawn_pose["y_m"] = 3
    frozen.spawn_pose["yaw_rad"] = math_pi = 0.5
    output = tmp_path / "timeout"
    summary = s0_runner.run_s0_episode(frozen, output, max_simulation_seconds=0.2)
    assert summary["status"] == "FAIL"
    assert summary["route_install_count"] == 1
    assert summary["sample_count"] >= 3
    assert summary["cleanup_ok"] is True
    assert vars(world.settings) == vars(before)
    assert world.weather == "OriginalWeather"
    assert not world.get_actors()
    assert summary["config_sha256"] != frozen.config_sha256
    assert frozen.config["runtime"]["max_simulation_seconds"] == 0.1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["formal_acceptance"] is False
    assert manifest["world_episode_id_before"] == 1
    assert manifest["world_episode_id_after"] == 2
    assert world.reload_call == ("Town10HD_Opt", False, "NONE")
    assert manifest["source_unchanged_during_run"] is True
    provenance = manifest["execution_provenance"]
    assert provenance["python_version"]
    assert provenance["source_sha256"]["carla_adapter.s0_runner"]
    assert provenance["source_bundle_sha256"] == manifest["final_execution_provenance"]["source_bundle_sha256"]
    assert set(provenance["installed_distributions"]) == {"carla", "numpy"}
    assert world.transform.location.y == -3
    assert world.transform.rotation.yaw == pytest.approx(-math_pi * 180 / 3.141592653589793)
    events = json.loads((output / "events.json").read_text())
    assert sum(event["kind"] == "route_installed" for event in events) == 1
    assert events[-1]["collision_complete"] is True


def test_map_hash_mismatch_refuses_spawn_and_retains_failure(tmp_path, monkeypatch):
    world = FakeWorld()
    fake_live(monkeypatch, world)
    frozen = mission()
    frozen.opendrive_sha256 = "b" * 64
    summary = s0_runner.run_s0_episode(frozen, tmp_path / "wrong-map")
    assert "OpenDRIVE hash" in summary["failure_reason"]
    assert world.transform is None
    assert world.weather == "OriginalWeather"


def test_navigation_only_completion_retains_missing_latency_evidence(tmp_path, monkeypatch):
    world = FakeWorld()
    fake_live(monkeypatch, world)
    base_adapter = s0_runner.FixedVehicleAdapter

    class GoalAdapter(base_adapter):
        calls = 0

        def state(self):
            self.calls += 1
            return VehicleState(100 if self.calls > 6 else 0, 0, 0, 0)

    def unexpected_read(*args):
        raise AssertionError("Navigation diagnostic must not read RGB")

    monkeypatch.setattr(s0_runner, "FixedVehicleAdapter", GoalAdapter)
    monkeypatch.setattr(s0_runner.RunSensors, "read_latest", unexpected_read)
    frozen = mission()
    output = tmp_path / "navigation-diagnostic"
    summary = s0_runner.run_s0_episode(frozen, output, navigation_only=True)
    assert summary["completed"] is True
    assert summary["status"] == "INSUFFICIENT_EVIDENCE"
    assert summary["formal_acceptance"] is False
    assert summary["config_sha256"] != frozen.config_sha256
    assert summary["metrics"]["capture_to_apply_p95_ms"]["value"] is None
    with (output / "samples.csv").open(newline="") as sample_file:
        sample = next(csv.DictReader(sample_file))
    for field in ("capture_frame", "capture_time_s", "capture_receipt_monotonic_s", "apply_monotonic_s",
                  "capture_to_apply_ms", "frame_age", "captured_x_m", "captured_y_m"):
        assert sample[field] == ""
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["navigation_only_diagnostic"] is True
    assert manifest["effective_config"]["runtime"]["rgb_enabled"] is False


def test_unreachable_cleanup_probes_real_rpc_and_skips_further_restoration(tmp_path, monkeypatch):
    world = FakeWorld()
    fake_live(monkeypatch, world)
    original_settings = world.get_settings
    world.unreachable = False
    calls_after_disconnect = []

    def settings_rpc():
        if world.unreachable:
            calls_after_disconnect.append("get_settings")
            raise ConnectionError("Injected disconnected server")
        return original_settings()

    world.get_settings = settings_rpc
    base_sensors = s0_runner.RunSensors

    class DisconnectSensors(base_sensors):
        def read_latest(self, min_frame, max_frame, timeout_s):
            world.unreachable = True
            raise TimeoutError("Injected camera disconnect")

        def close(self, *, reachable=True):
            world.close_reachable = reachable
            return []

    monkeypatch.setattr(s0_runner, "RunSensors", DisconnectSensors)
    output = tmp_path / "server-disconnected"
    summary = s0_runner.run_s0_episode(mission(), output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert calls_after_disconnect == ["get_settings"]
    assert world.close_reachable is False
    assert world.actors[0].is_alive
    assert world.weather == "ClearNoon"
    assert manifest["cleanup_server_reachable"] is False
    assert manifest["settings_restored"] is None
    assert manifest["actor_inventory_restored"] is None
    assert manifest["environment_restored"] is None
    assert summary["cleanup_ok"] is False
    assert summary["status"] == "FAIL"
    assert json.loads((output / "events.json").read_text())[-1]["collision_complete"] is False


def test_cli_invalid_values_fail_before_connection(monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_scenario.py"
    spec = importlib.util.spec_from_file_location("s0_cli_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    for value in ("nan", "inf", "0", "-1"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--output", "unused", "--max-sim-seconds", value])
    with pytest.raises(SystemExit):
        module.main(["--output", "unused", "--runs", "5", "--hold-at-progress", "10"])
    monkeypatch.setattr(module, "load_frozen_mission", lambda _: mission())
    monkeypatch.setattr(module, "run_s0_suite", lambda *args, **kwargs: {
        "status": "INSUFFICIENT_EVIDENCE", "passed_runs": 1,
    })
    assert module.main(["--output", "unused", "--runs", "1"]) == 0
