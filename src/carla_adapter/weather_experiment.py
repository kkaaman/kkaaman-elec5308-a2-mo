"""Read-only installed CARLA weather/light preflight for E1-01/02.

Records all 14 numeric weather fields and explicit light names. It never applies
weather, spawns actors, ticks the world or claims that weather affects a sensor.
"""

import argparse
import json
import math
from pathlib import Path

from autonomy.paths import REPO_ROOT


def verify_settings(carla_module, document):
    """Verify installed presets, explicit overrides and light flags without mutation."""
    reports = []
    for condition in document["conditions"]:
        preset = getattr(carla_module.WeatherParameters, condition["name"], None)
        if preset is None:
            raise ValueError(f"Installed CARLA lacks preset {condition['name']}")
        expected = condition["weather"]
        preset_values = {}
        for field, wanted in expected.items():
            actual = getattr(preset, field, None)
            if not isinstance(actual, (int, float)) or not math.isfinite(actual):
                raise ValueError(f"Missing/nonfinite weather field {field}")
            preset_values[field] = actual
            # The source preset's -1 azimuth means inherited state; experiment fixes 0.
            reference = -1 if field == "sun_azimuth_angle" else wanted
            if not math.isclose(actual, reference, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(f"Installed preset differs: {condition['name']}.{field}")
        configured = carla_module.WeatherParameters(**expected)
        for field, wanted in expected.items():
            if not math.isclose(getattr(configured, field), wanted, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(f"Weather override did not round-trip: {field}")
        for name in condition["vehicle_lights"]:
            if not hasattr(carla_module.VehicleLightState, name):
                raise ValueError(f"Installed CARLA lacks vehicle light {name}")
        reports.append(
            dict(
                name=condition["name"],
                preset_values=preset_values,
                configured_values=expected,
                vehicle_lights=condition["vehicle_lights"],
            )
        )
    return reports


def select_capture_cell(path, cell_id):
    """Select one immutable paired cell without importing or contacting CARLA."""
    import hashlib

    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if document.get("scenarios") != ["S0", "S1", "S2"] or document.get("repetitions") != 3:
        raise ValueError("Expected the approved S0-S2 three-repetition matrix")
    if [c["name"] for c in document["conditions"]] != ["ClearNoon", "HardRainNoon", "ClearNight"]:
        raise ValueError("Expected the approved weather conditions")
    seeds = document["paired_seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Three paired nonnegative seeds required")
    for scenario in document["scenarios"]:
        for condition in document["conditions"]:
            for repetition, seed in enumerate(seeds, 1):
                identity = f"{scenario}-{condition['name']}-r{repetition}"
                if cell_id == identity:
                    return dict(
                        cell_id=identity,
                        scenario=scenario,
                        condition=condition["name"],
                        repetition=repetition,
                        seed=seed,
                        weather=condition["weather"],
                        vehicle_lights=condition["vehicle_lights"],
                        specification_sha256=hashlib.sha256(raw).hexdigest(),
                    )
    raise ValueError("Unknown paired capture cell")


def apply_capture_weather(carla_module, world, ego, cell):
    """Apply explicit weather/lights and retain measured readback; no sensor claim."""
    weather = carla_module.WeatherParameters(**cell["weather"])
    world.set_weather(weather)
    flags = int(carla_module.VehicleLightState.NONE)
    for name in cell["vehicle_lights"]:
        flags |= int(getattr(carla_module.VehicleLightState, name))
    flags = carla_module.VehicleLightState(flags)
    ego.set_light_state(flags)
    actual = world.get_weather()
    readback = {name: getattr(actual, name) for name in cell["weather"]}
    if any(
        not math.isclose(readback[name], value, rel_tol=1e-5, abs_tol=1e-5)
        for name, value in cell["weather"].items()
    ):
        raise ValueError("Applied weather differs from the paired specification")
    if int(ego.get_light_state()) != int(flags):
        raise ValueError("Applied vehicle lights differ from the paired specification")
    return dict(
        weather=readback,
        vehicle_light_flags=int(flags),
        sensor_effects="NOT_INFERRED_FROM_SETTINGS",
    )


def main():
    import carla

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/weather/robustness_v1.json"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    doc = json.loads(args.config.read_text())
    client = carla.Client(args.host, args.port)
    client.set_timeout(10)
    client_version, server_version = client.get_client_version(), client.get_server_version()
    if client_version != doc["carla_version"] or server_version != doc["carla_version"]:
        raise ValueError("Matching specified CARLA client/server required")
    map_name = client.get_world().get_map().name
    if map_name.split("/")[-1] != doc["map"]:
        raise ValueError("Specified weather-experiment map required")
    report = dict(
        status="INSTALLED_SETTINGS_VERIFIED_NOT_CAPTURED",
        client_version=client_version,
        server_version=server_version,
        map=map_name,
        conditions=verify_settings(carla, doc),
        world_mutated=False,
        dataset="NOT_CAPTURED",
        sensor_effects=doc["sensor_effects"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
