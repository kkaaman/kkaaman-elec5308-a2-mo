"""Weather evaluation must be paired, explicit and never claim unmeasured results."""

import json
from pathlib import Path

import pytest

from scripts.prepare_weather_matrix import prepare

CONFIG = Path(__file__).resolve().parents[2] / "configs/weather/robustness_v1.json"


def test_weather_matrix_pairs_all_scenes_and_never_claims_capture():
    result = prepare(CONFIG)
    assert result["count"] == 27
    assert len({c["cell_id"] for c in result["cells"]}) == 27
    assert {c["status"] for c in result["cells"]} == {"NOT_RUN"}
    for scenario in ("S0", "S1", "S2"):
        for repetition in range(1, 4):
            paired = [c for c in result["cells"] if c["scenario"] == scenario
                      and c["repetition"] == repetition]
            assert len(paired) == 3 and len({c["seed"] for c in paired}) == 1
    assert result["sensor_effects"]["depth"] == "NOT_OBSERVED"


@pytest.mark.parametrize("defect", ["missing", "nan", "inherited", "night", "seed", "duplicate"])
def test_invalid_or_unpaired_experiments_are_refused(tmp_path, defect):
    doc = json.loads(CONFIG.read_text())
    if defect == "missing":
        del doc["conditions"][0]["weather"]["dust_storm"]
    elif defect == "nan":
        doc["conditions"][0]["weather"]["cloudiness"] = float("nan")
    elif defect == "inherited":
        doc["conditions"][0]["weather"]["sun_azimuth_angle"] = -1
    elif defect == "night":
        doc["conditions"][2]["vehicle_lights"] = []
    elif defect == "seed":
        doc["paired_seeds"][0] = True
    else:
        doc["paired_seeds"][1] = doc["paired_seeds"][0]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        prepare(path)


@pytest.mark.parametrize("defect", [None, "preset", "field", "light", "drift"])
def test_installed_version_settings_and_lights_are_verified_without_world_access(defect):
    from types import SimpleNamespace

    from carla_adapter.weather_experiment import verify_settings

    doc = json.loads(CONFIG.read_text())

    class Weather:
        def __init__(self, **values):
            self.__dict__.update(values)

    for condition in doc["conditions"]:
        values = dict(condition["weather"], sun_azimuth_angle=-1)
        setattr(Weather, condition["name"], Weather(**values))
    module = SimpleNamespace(WeatherParameters=Weather,
                             VehicleLightState=SimpleNamespace(Position=1, LowBeam=2))
    if defect == "preset":
        del Weather.ClearNight
    elif defect == "field":
        del Weather.ClearNoon.dust_storm
    elif defect == "light":
        del module.VehicleLightState.LowBeam
    elif defect == "drift":
        Weather.ClearNoon.precipitation = 50
    if defect:
        with pytest.raises(ValueError):
            verify_settings(module, doc)
    else:
        rows = verify_settings(module, doc)
        assert len(rows) == 3
        assert rows[2]["vehicle_lights"] == ["Position", "LowBeam"]
        assert rows[0]["configured_values"]["sun_azimuth_angle"] == 0
