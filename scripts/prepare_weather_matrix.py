"""Prepare the paired 27-cell E1 weather experiment; never connects to CARLA."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = frozenset("cloudiness precipitation precipitation_deposits wind_intensity "
                   "sun_azimuth_angle sun_altitude_angle fog_density fog_distance fog_falloff "
                   "wetness scattering_intensity mie_scattering_scale rayleigh_scattering_scale "
                   "dust_storm".split())


def prepare(path):
    """Validate complete explicit settings and emit paired scenario/condition/seed cells."""
    path = Path(path)
    raw = path.read_bytes()
    doc = json.loads(raw)
    if (doc["schema_version"] != 1 or doc["carla_version"] != "0.9.16"
            or doc["scenarios"] != ["S0", "S1", "S2"] or doc["repetitions"] != 3
            or len(doc["paired_seeds"]) != 3 or len(set(doc["paired_seeds"])) != 3
            or any(type(seed) is not int or seed < 0 for seed in doc["paired_seeds"])):
        raise ValueError("Expected S0-S2, three paired nonnegative integer seeds, CARLA 0.9.16")
    if not math.isfinite(doc["fixed_delta_seconds"]) or doc["fixed_delta_seconds"] <= 0:
        raise ValueError("Positive finite timestep required")
    conditions = doc["conditions"]
    if [c["name"] for c in conditions] != ["ClearNoon", "HardRainNoon", "ClearNight"]:
        raise ValueError("Expected the three approved weather conditions")
    for condition in conditions:
        weather = condition["weather"]
        if set(weather) != FIELDS or any(type(v) not in (int, float) or not math.isfinite(v)
                                         for v in weather.values()):
            raise ValueError("Every weather field must be explicit and finite")
        for name in ("cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
                     "fog_density", "wetness", "dust_storm"):
            if not 0 <= weather[name] <= 100:
                raise ValueError(f"Invalid percentage: {name}")
        if not 0 <= weather["sun_azimuth_angle"] < 360 or not -90 <= weather["sun_altitude_angle"] <= 90:
            raise ValueError("Explicit sun angles required; inherited -1 azimuth is refused")
        if any(weather[name] < 0 for name in ("fog_distance", "fog_falloff", "scattering_intensity",
                                              "mie_scattering_scale", "rayleigh_scattering_scale")):
            raise ValueError("Negative distance/scattering setting")
        required_lights = ["Position", "LowBeam"] if condition["name"] == "ClearNight" else []
        if condition["vehicle_lights"] != required_lights:
            raise ValueError("Night requires position lights and low beam; daytime lights off")
    cells = [dict(cell_id=f"{scenario}-{condition['name']}-r{rep + 1}", scenario=scenario,
                  condition=condition["name"], repetition=rep + 1, seed=doc["paired_seeds"][rep],
                  weather=condition["weather"], vehicle_lights=condition["vehicle_lights"],
                  status="NOT_RUN")
             for scenario, condition, rep in itertools.product(doc["scenarios"], conditions, range(3))]
    return dict(status="PLANNED_NOT_CAPTURED", specification_sha256=hashlib.sha256(raw).hexdigest(),
                cells=cells, count=len(cells), live_verification="NOT_RUN",
                sensor_effects=doc["sensor_effects"], required_artifacts=doc["required_artifacts"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/weather/robustness_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"Prepared {result['count']} cells; every live cell remains NOT_RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
