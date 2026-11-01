"""Paired capture uses real helpers and fake weather/light readback boundaries."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carla_adapter.weather_experiment import apply_capture_weather, select_capture_cell  # noqa: TID251

CONFIG = Path(__file__).resolve().parents[2] / "configs/weather/robustness_v1.json"


def test_cells_share_scene_seed_and_keep_night_lights_explicit():
    cells = [
        select_capture_cell(CONFIG, f"S1-{name}-r2")
        for name in ("ClearNoon", "HardRainNoon", "ClearNight")
    ]
    assert len({c["seed"] for c in cells}) == 1
    assert len({c["specification_sha256"] for c in cells}) == 1
    assert cells[2]["vehicle_lights"] == ["Position", "LowBeam"]
    assert cells[2]["weather"]["sun_altitude_angle"] < 0
    with pytest.raises(ValueError, match="Unknown"):
        select_capture_cell(CONFIG, "S1-CloudyNoon-r1")


@pytest.mark.parametrize("drift", ("none", "weather", "lights"))
def test_actual_weather_and_light_readback_is_mandatory(drift):
    cell = select_capture_cell(CONFIG, "S2-ClearNight-r1")

    class World:
        def set_weather(self, value):
            self.value = value

        def get_weather(self):
            if drift == "weather":
                self.value.precipitation = 42
            return self.value

    class Flags:
        def __init__(self, value):
            self.value = int(value)

        def __int__(self):
            return self.value

        def __or__(self, other):
            return self.value | int(other)  # CARLA Boost.Python enum behavior

    Flags.NONE, Flags.Position, Flags.LowBeam = Flags(0), Flags(1), Flags(2)

    class Ego:
        def set_light_state(self, flags):
            assert isinstance(flags, Flags)
            self.flags = flags

        def get_light_state(self):
            return self.flags if drift != "lights" else 0

    carla = SimpleNamespace(
        WeatherParameters=lambda **kw: SimpleNamespace(**kw),
        VehicleLightState=Flags,
    )
    if drift == "none":
        assert apply_capture_weather(carla, World(), Ego(), cell)["vehicle_light_flags"] == 3
    else:
        with pytest.raises(ValueError):
            apply_capture_weather(carla, World(), Ego(), cell)


def test_plan_only_never_runs_collector(tmp_path, monkeypatch):
    from scripts import capture_weather_dataset as script

    monkeypatch.setattr("sys.argv", ["capture", "--output", str(tmp_path / "queue")])
    monkeypatch.setattr(script.subprocess, "run", lambda *a, **kw: pytest.fail("No live execution"))
    assert script.main() == 0
    report = json.loads((tmp_path / "queue/queue.json").read_text())
    assert report["status"] == "PLAN_ONLY"
    assert len(report["cells"]) == 27
    assert all(c["status"] == "NOT_RUN" for c in report["cells"])


def test_combined_index_keeps_real_jsonl_and_original_asset_references(tmp_path, monkeypatch):
    from evaluation import dataset
    from scripts.capture_weather_dataset import combine_captures

    def read(folder):
        if folder == tmp_path:
            records = [
                json.loads(line) for line in (folder / "samples.jsonl").read_text().splitlines()
            ]
            assert len(records) == 2
            return {}, records
        identity = folder.name
        manifest = dict(
            weather_experiment=dict(cell_id=identity, specification_sha256="s"),
            settings_restored=True,
            no_actor_leaks=True,
            weather_readback={},
            camera={},
            image_size_wh=[4, 4],
            opendrive_sha256="map",
            label_tag=24,
        )
        row = dict(
            sample_id="a",
            run_id="run",
            sequence_id="scene",
            rgb="data/rgb.png",
            lane="data/lane.png",
            instance="data/instance.png",
            depth="data/depth.npy",
        )
        return manifest, [row]

    monkeypatch.setattr(dataset, "read_dataset", read)
    combine_captures(tmp_path, [dict(cell_id="cell1"), dict(cell_id="cell2")], "s")
    rows = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert rows[0]["rgb"] == "cell1/data/rgb.png"
    assert rows[1]["sequence_id"] == "cell2/scene"
    assert not (tmp_path / "cell1").exists()  # no image copy was made


def test_cleanup_waits_for_async_destroy_snapshot_without_own_tick():
    from carla_adapter.dataset_capture import verify_owned_cleanup  # noqa: TID251

    class World:
        pending = True

        def get_actors(self):
            return [SimpleNamespace(id=7)] if self.pending else []

        def wait_for_tick(self, timeout):
            self.pending = False

    assert verify_owned_cleanup(World(), {7})
    stuck = SimpleNamespace(get_actors=lambda: [SimpleNamespace(id=7)])
    assert verify_owned_cleanup(stuck, {7}, timeout_s=0) is False


def test_resume_reuses_complete_cells_and_archives_failed_attempt(tmp_path, monkeypatch):
    from evaluation import dataset
    from scripts import capture_weather_dataset as script

    output = tmp_path / "queue"
    common = ["capture", "--output", str(output), "--cells", "S0-ClearNoon-r1", "S0-ClearNoon-r2"]
    monkeypatch.setattr("sys.argv", common)
    assert script.main() == 0
    report = json.loads((output / "queue.json").read_text())
    report["cells"][0]["status"] = "COLLECTED"
    report["cells"][1]["status"] = "FAILED"
    (output / "queue.json").write_text(json.dumps(report))
    for cell in report["cells"]:
        folder = output / cell["cell_id"]
        folder.mkdir()
        (folder / "sentinel").write_text(cell["status"])
    (output / "S0-ClearNoon-r2.log").write_text("original failure")
    monkeypatch.setattr(
        dataset,
        "read_dataset",
        lambda folder: (dict(sample_count=600, settings_restored=True, no_actor_leaks=True), []),
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        Path(command[command.index("--output") + 1]).mkdir()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(script.subprocess, "check_output", lambda *args, **kwargs: "fixture-sha")
    monkeypatch.setattr(script.subprocess, "run", run)
    monkeypatch.setattr(script, "combine_captures", lambda *args: None)
    monkeypatch.setattr("sys.argv", common + ["--resume", "--execute"])
    assert script.main() == 0
    assert len(calls) == 1 and "S0-ClearNoon-r2" in calls[0]
    assert (output / "S0-ClearNoon-r1/sentinel").read_text() == "COLLECTED"
    archive = next((output / "attempt-history").glob("S0-ClearNoon-r2-attempt-*"))
    assert (archive / "capture/sentinel").read_text() == "FAILED"
    assert (archive / "capture.log").read_text() == "original failure"


def test_cleanup_does_not_skip_owned_actors_missing_from_cached_snapshot():
    from carla_adapter.dataset_capture import destroy_owned_actors  # noqa: TID251
    calls=[]
    actor=SimpleNamespace(id=7,is_alive=False,destroy=lambda:calls.append(7) or True)
    assert destroy_owned_actors([actor])==[]
    assert calls==[7]
