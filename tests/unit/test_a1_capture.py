"""TEST-01 raw logging, independent CTE checks and provenance guards."""

import csv
import math
from pathlib import Path

import pytest

from evaluation.a1_capture import (
    COLUMNS,
    CaptureLog,
    metrics,
    normalized_hash,
    signed_lane_cte,
    verify_sources,
)


@pytest.mark.parametrize(
    "yaw,x,y,expected",
    [
        (0, 3, 2, 2),
        (math.pi / 2, 2, 3, -2),
        (-math.pi / 2, 2, 3, 2),
        (math.pi, 3, 2, -2),
    ],
)
def test_cte_known_axes(yaw, x, y, expected):
    assert signed_lane_cte(x, y, 0, 0, yaw) == pytest.approx(expected)


def test_absolute_cte_rms_and_missing_metrics():
    result = metrics([-3, 4, None], absolute=True)
    assert result["mean"] == 3.5
    assert result["rms"] == pytest.approx(math.sqrt(12.5))
    assert metrics([])["mean"] is None


def test_logging_and_separate_controller_measurements(tmp_path):
    log = CaptureLog(tmp_path)
    row = dict.fromkeys(COLUMNS)
    row.update(
        sample=0,
        nearest_lane_cte_signed_m=-2,
        speed_mps=3,
        controller_diagnostic_current=False,
        controller_cte_m=99,
    )
    log.append(row)
    report = log.summary()
    log.close()
    assert report["nearest_lane_absolute_cte_m"]["mean"] == 2
    assert report["controller_cte_m_current_updates_only"]["count"] == 0
    assert report["supplied_prompt_results"]["status"].startswith("UNVERIFIED")
    with (tmp_path / "samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["speed_mps"] == "3"


def test_invalid_log_schema_and_nonfinite_values_rejected(tmp_path):
    log = CaptureLog(tmp_path)
    try:
        with pytest.raises(ValueError):
            log.append({"speed_mps": 3})
        row = dict.fromkeys(COLUMNS)
        row["speed_mps"] = float("nan")
        with pytest.raises(ValueError):
            log.append(row)
    finally:
        log.close()


def test_source_identity_normalizes_windows_newlines(tmp_path):
    source = tmp_path / "module.py"
    source.write_bytes(b"x = 1\r\n")
    expected = normalized_hash(source)
    source.write_bytes(b"x = 1\n")
    verify_sources(tmp_path, {"module.py": expected})
    source.write_text("x = 2\n")
    with pytest.raises(ValueError, match="Source differs"):
        verify_sources(tmp_path, {"module.py": expected})


def test_no_changes_to_driving_demo_in_bundle():
    root = Path(__file__).resolve().parents[2]
    # The manifest pins the inherited demo; the observer never writes to it.
    import json

    manifest = json.loads((root / "configs/test01_a1_sources.json").read_text())
    assert (
        normalized_hash(root / "src/demo_fused_lane_following.py")
        == manifest["src/demo_fused_lane_following.py"]
    )


@pytest.mark.parametrize("fail_after_frame", [False, True])
def test_capture_pipeline_and_exception_cleanup(
    tmp_path, monkeypatch, fail_after_frame
):
    import importlib.util
    import json
    import sys
    import time
    from dataclasses import dataclass
    from types import ModuleType, SimpleNamespace

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "test01_recorder_unit", root / "src/carla_adapter/a1_capture.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    controls = []
    actors = []
    location = SimpleNamespace(x=0.0, y=1.0, z=0.7)
    rotation = SimpleNamespace(yaw=0.0, pitch=0.0, roll=0.0)
    pose = SimpleNamespace(location=location, rotation=rotation)
    road_map = SimpleNamespace(
        name="Carla/Maps/Town10HD_Opt",
        to_opendrive=lambda: "fake map",
        get_waypoint=lambda *a, **k: SimpleNamespace(
            transform=SimpleNamespace(
                location=SimpleNamespace(x=0, y=0), rotation=rotation
            ),
            road_id=1,
            lane_id=-1,
        ),
    )

    class Actor:
        id = 7
        type_id = "vehicle.tesla.model3"
        attributes = {}
        is_alive = True

        def get_transform(self):
            return pose

        def get_velocity(self):
            return SimpleNamespace(x=1, y=0, z=0)

        def get_physics_control(self):
            return SimpleNamespace(wheels=[])

        def apply_control(self, control):
            controls.append(control)

        def destroy(self):
            self.is_alive = False
            return True

    def spawn(*args, **kwargs):
        actor = Actor()
        actors.append(actor)
        return actor

    world = SimpleNamespace(
        get_map=lambda: road_map,
        get_settings=lambda: SimpleNamespace(synchronous_mode=False),
        get_weather=lambda: SimpleNamespace(cloudiness=10.0),
        get_actors=lambda: [a for a in actors if a.is_alive],
        spawn_actor=spawn,
        get_snapshot=lambda: SimpleNamespace(
            frame=1, timestamp=SimpleNamespace(elapsed_seconds=0.05)
        ),
    )
    fake_carla = ModuleType("carla")
    fake_carla.Client = lambda *a, **k: SimpleNamespace(
        set_timeout=lambda t: None,
        get_world=lambda: world,
        get_client_version=lambda: "0.9.16",
        get_server_version=lambda: "0.9.16",
    )
    fake_carla.LaneType = SimpleNamespace(Driving=1)
    fake_carla.VehicleControl = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_cv = ModuleType("cv2")
    fake_cv.imshow = lambda *args: None
    fake_cv.waitKey = lambda delay: -1
    fake_cv.VideoWriter_fourcc = lambda *args: 0
    fake_cv.VideoWriter = lambda path, *args: SimpleNamespace(
        isOpened=lambda: True,
        write=lambda image: Path(path).write_bytes(b"stub video"),
        release=lambda: None,
    )
    fake_cv.VideoCapture = lambda path: SimpleNamespace(
        read=lambda: (True, b"stub frame"), release=lambda: None
    )
    fake_torch = ModuleType("torch")
    fake_torch.__version__ = "stub"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    for name, module in (
        ("carla", fake_carla),
        ("cv2", fake_cv),
        ("torch", fake_torch),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        runner,
        "_command",
        lambda args: {"returncode": 0, "stdout": "stub", "stderr": ""},
    )
    monkeypatch.setattr(
        runner,
        "_asset",
        lambda path, expected=None: {
            "path": str(Path(path).resolve()),
            "sha256": "stub",
            "size_bytes": 1,
        },
    )

    @dataclass
    class Config:
        wheelbase_m: float = 2.5
        max_steering_angle_rad: float = 0.6

    command = SimpleNamespace(steering_angle_rad=0.1, throttle=0.2, brake=0.0)
    fake_demo = SimpleNamespace(
        MAX_DURATION_S=120,
        ObjectDetector=lambda c: None,
        _save_failure_snapshot=lambda *args: None,
    )
    fake_demo.apply_command = lambda vehicle, cmd, scale: SimpleNamespace(
        steer=-cmd.steering_angle_rad / scale, throttle=cmd.throttle, brake=cmd.brake
    )

    def fake_main():
        vehicle = (
            fake_demo.carla.Client("localhost", 2000)
            .get_world()
            .spawn_actor(SimpleNamespace(id=Actor.type_id), pose)
        )
        config = Config()
        collision_events = []
        cycle_start = time.time()
        actual_dt_s = 0.1
        driving_mode = "lane_following"
        consecutive_misses = 0
        lane_path = None
        should_stop_for_light = False
        result = SimpleNamespace(
            diagnostics=SimpleNamespace(
                cross_track_error_m=0.25, internal_target_speed_mps=3
            )
        )
        # The recorder deliberately reads this inherited call frame's locals.
        assert all(
            v is not None
            for v in (
                collision_events,
                cycle_start,
                actual_dt_s,
                driving_mode,
                consecutive_misses,
                should_stop_for_light,
                result,
            )
        )
        assert lane_path is None
        fake_demo.apply_command(vehicle, command, config.max_steering_angle_rad)
        fake_cv.imshow("Fused lane following", SimpleNamespace(shape=(600, 800, 3)))
        if fail_after_frame:
            raise RuntimeError("injected post-frame failure")

    fake_demo.main = fake_main
    monkeypatch.setattr(
        runner.importlib.util,
        "spec_from_file_location",
        lambda *a: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m: None)),
    )
    monkeypatch.setattr(
        runner.importlib.util, "module_from_spec", lambda spec: fake_demo
    )
    output = tmp_path / "capture"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture",
            "--repo",
            str(root),
            "--detector-weights",
            str(tmp_path / "yolo26n.pt"),
            "--output",
            str(output),
        ],
    )
    if fail_after_frame:
        with pytest.raises(SystemExit):
            runner.main()
    else:
        runner.main()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["samples"] == 1
    assert summary["video_frames"] == 1
    assert summary["nearest_lane_absolute_cte_m"]["mean"] == 1
    assert summary["controller_cte_m_current_updates_only"]["mean"] == 0.25
    assert summary["capture_status"] == (
        "INCOMPLETE" if fail_after_frame else "CAPTURED"
    )
    assert not actors[0].is_alive
    assert controls[-1].brake == 1
    assert (output / "source_snapshot.zip").is_file()
