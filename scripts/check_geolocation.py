"""Which way is north, on the live server? Compares three things at two places 100 m apart in y:

- ``map.transform_to_geolocation`` (what the sat-nav route is built from);
- a GNSS sensor on a parked car, noise off (what the car receives);
- ``evaluation.satnav_route.carla_geolocation`` (the pure copy of CARLA's formula).

Prints the latitude change per metre of CARLA +y for each, and whether the route and the
receiver agree. The live curve run of 23 Oct showed the receiver's latitude growing with
CARLA +y, the opposite of the compass's north; the sat-nav route must use the same
convention as the receiver, and the suite's ``gnss.north_sign`` the sign measured here.

Usage: python scripts/check_geolocation.py [--host 127.0.0.1 --port 2000]
"""

import argparse
import json
import math
import sys
import time

from evaluation.satnav_route import carla_geolocation

EARTH_RADIUS_M = 6371000.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args(argv)
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    carla_map = world.get_map()
    spawn = carla_map.get_spawn_points()[0]
    library = world.get_blueprint_library()
    vehicle = world.try_spawn_actor(library.find("vehicle.tesla.model3"), spawn)
    if vehicle is None:
        raise RuntimeError("Could not spawn the probe car at spawn point 0")
    bp = library.find("sensor.other.gnss")
    for name in ("noise_lat_stddev", "noise_lon_stddev", "noise_lat_bias", "noise_lon_bias", "noise_alt_stddev", "noise_alt_bias"):
        bp.set_attribute(name, "0.0")
    bp.set_attribute("sensor_tick", "0.0")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
    readings = []
    sensor.listen(lambda data: readings.append((data.latitude, data.longitude)))
    try:
        results = []
        for dy in (0.0, 100.0):
            t = carla.Transform(carla.Location(spawn.location.x, spawn.location.y + dy, spawn.location.z + 0.5), spawn.rotation)
            vehicle.set_transform(t)
            vehicle.set_simulate_physics(False)
            readings.clear()
            deadline = time.time() + 5.0
            while len(readings) < 3 and time.time() < deadline:
                world.wait_for_tick(1.0)
            if not readings:
                raise RuntimeError("No GNSS readings arrived")
            loc = vehicle.get_location()
            fix = carla_map.transform_to_geolocation(loc)
            results.append(dict(carla_y=loc.y, sensor=readings[-1], map=(fix.latitude, fix.longitude), x=loc.x))
        lat0, lon0 = carla_map.transform_to_geolocation(carla.Location(0.0, 0.0, 0.0)).latitude, carla_map.transform_to_geolocation(carla.Location(0.0, 0.0, 0.0)).longitude
        formula = carla_geolocation(lat0, lon0)
        dy = results[1]["carla_y"] - results[0]["carla_y"]
        per_m = math.radians(1.0) * EARTH_RADIUS_M / dy

        def north_per_carla_y(a, b):
            return round((b[0] - a[0]) * per_m, 4)

        sensor_rate = north_per_carla_y(results[0]["sensor"], results[1]["sensor"])
        map_rate = north_per_carla_y(results[0]["map"], results[1]["map"])
        formula_rate = north_per_carla_y(formula(results[0]["x"], results[0]["carla_y"]), formula(results[1]["x"], results[1]["carla_y"]))
        out = dict(
            map=carla_map.name,
            server=client.get_server_version(),
            metres_north_per_metre_of_carla_plus_y=dict(gnss_sensor=sensor_rate, map_transform=map_rate, formula=formula_rate),
            compass_north_is="CARLA -y (core +y), per the IMU docs",
            gnss_north_sign=-1 if sensor_rate > 0.5 else 1 if sensor_rate < -0.5 else 0,
            route_matches_receiver=abs(sensor_rate - map_rate) < 0.1,
            note="north_sign -1 means the receiver's latitude grows towards the compass's south",
        )
        print(json.dumps(out, indent=2))
    finally:
        sensor.stop()
        sensor.destroy()
        vehicle.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
