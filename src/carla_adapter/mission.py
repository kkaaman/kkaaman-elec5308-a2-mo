"""BL-A-04: read-only mission selection and validation against the live map."""

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from carla_adapter.topology import export_lane_topology
from planning.corridor import RouteGeometry
from planning.mission import select_mission


def prepare(client, output):
    import carla

    world = client.get_world()
    map_ = world.get_map()
    if (
        client.get_client_version() != client.get_server_version()
        or client.get_server_version() != "0.9.16"
    ):
        raise ValueError("Matching CARLA 0.9.16 client/server required")
    if map_.name.rsplit("/", 1)[-1] not in {"Town10HD", "Town10HD_Opt"}:
        raise ValueError("Expected Town10HD")
    topology = export_lane_topology(map_, sampling_distance_m=1.0, navigation_speed_cap_mps=5.0)
    mission = select_mission(topology)
    allowed = {tuple(map(int, key.split(":"))) for key in mission.segment_ids}
    max_error = 0.0
    widths = []
    checked = 0
    for a, b in zip(mission.points_xyz_m, mission.points_xyz_m[1:]):
        for ratio in np.linspace(0, 1, max(2, math.ceil(math.dist(a, b) / 0.25) + 1)):
            p = np.asarray(a) + ratio * (np.asarray(b) - a)
            wp = map_.get_waypoint(
                carla.Location(x=float(p[0]), y=float(-p[1]), z=float(p[2])),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            key = (wp.road_id, wp.section_id, wp.lane_id)
            error = math.hypot(wp.transform.location.x - p[0], -wp.transform.location.y - p[1])
            if key not in allowed or error > 0.20:
                raise ValueError(
                    f"Mission sample is outside its declared legal lane: {key}, offset {error:.3f}"
                )
            max_error = max(max_error, error)
            widths.append(wp.lane_width)
            checked += 1
    route = RouteGeometry(mission.points_xy_m)
    x, y, yaw = route.at(1.38641)
    report = dict(
        status="MAP_GEOMETRY_VALIDATED",
        server=client.get_server_version(),
        samples=checked,
        max_lane_center_error_m=max_error,
        minimum_lane_width_m=min(widths),
        spawn_actor_pose=dict(
            x=x, y=-y, z=mission.points_xyz_m[0][2] + 0.5, yaw_deg=-math.degrees(yaw)
        ),
        scope="Live map legality and centreline geometry; obstacle clearance and driving remain unverified",
    )
    if min(widths) < 2.5:
        raise ValueError("Lane narrower than fixed vehicle corridor")
    mission = replace(mission, map_validation=report)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mission.save(output / "mission.json")
    topology.save(output / "topology.json")
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    for lane in topology.lanes:
        p = np.asarray(lane.points_xyz_m)
        ax.plot(p[:, 0], p[:, 1], color="0.85", linewidth=0.5)
    p = np.asarray(mission.points_xyz_m)
    ax.plot(p[:, 0], p[:, 1], "b-", label="fixed mission")
    ax.scatter(*p[0, :2], c="green", label="start")
    ax.scatter(*p[-1, :2], c="red", label="goal")
    ax.set_aspect("equal")
    ax.set_xlabel("core X (m)")
    ax.set_ylabel("core Y (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "mission.png", dpi=150)
    plt.close(fig)
    return dict(report, mission_sha256=mission.sha256, output=str(output))


def assert_current_map(map_, mission):
    digest = hashlib.sha256(map_.to_opendrive().replace("\r\n", "\n").encode()).hexdigest()
    if map_.name.rsplit("/", 1)[-1] != mission.map_name or digest != mission.opendrive_sha256:
        raise ValueError("Mission was generated for another map/build")
    if (
        not mission.map_validation
        or mission.map_validation.get("status") != "MAP_GEOMETRY_VALIDATED"
    ):
        raise ValueError("Run prepare_mission.py on this CARLA build first")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(60)
    print(json.dumps(prepare(client, args.output), indent=2))
