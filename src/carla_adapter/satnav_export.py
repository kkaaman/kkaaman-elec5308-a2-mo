"""Write the sat-nav route (GPS points + commands) for a saved mission, from the live map.

This is evaluator-side tooling, in the role of a phone sat-nav. It reads the map once to
place the points. The mapless car receives only the written route.

It also checks ``evaluation.satnav_route.carla_geolocation`` against
``map.transform_to_geolocation``. Once that agrees, routes can be made offline from the
recorded geo reference.
"""

import argparse
import json
from pathlib import Path

from carla_adapter.topology import export_lane_topology
from evaluation.satnav_route import carla_geolocation, satnav_from_mission
from planning.mission import Mission


def main(argv=None):
    import carla

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mission", type=Path, required=True, help="Saved mission JSON")
    parser.add_argument("--output", type=Path, required=True, help="Route JSON to write")
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-spacing-m", type=float, default=50.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; choose a new file")
    mission = Mission.load(args.mission)
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    carla_map = client.get_world().get_map()
    topology = export_lane_topology(carla_map, sampling_distance_m=1.0, navigation_speed_cap_mps=5.0)
    if topology.opendrive_sha256 != mission.opendrive_sha256:
        raise RuntimeError("The server map is not the mission's map")
    if topology.sha256 != mission.source_topology_sha256:
        raise RuntimeError("Lane topology differs from the one the mission was built on")

    def geolocate(x, y):
        fix = carla_map.transform_to_geolocation(carla.Location(x=x, y=y, z=0.0))
        return fix.latitude, fix.longitude

    lat0, lon0 = geolocate(0.0, 0.0)
    formula = carla_geolocation(lat0, lon0)
    worst = 0.0
    for x, y in ((500, 0), (0, 500), (-500, -500), (300, -200), (-150, 420)):
        a, b = geolocate(float(x), float(y)), formula(float(x), float(y))
        # degrees -> metres, roughly; only the size of the disagreement matters here
        worst = max(worst, abs(a[0] - b[0]) * 111_000, abs(a[1] - b[1]) * 111_000)
    route = satnav_from_mission(mission, topology, geolocate, name=args.name, max_spacing_m=args.max_spacing_m)
    provenance = dict(
        route.provenance,
        geo_reference=dict(latitude_deg=lat0, longitude_deg=lon0),
        carla_formula_max_disagreement_m=worst,
        server=client.get_server_version(),
        map=carla_map.name,
    )
    route = type(route)(route.name, route.points, provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    route.save(args.output)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                points=len(route.points),
                commands=[p.command.value for p in route.points],
                route_sha256=route.sha256,
                carla_formula_max_disagreement_m=worst,
            ),
            indent=2,
        )
    )
