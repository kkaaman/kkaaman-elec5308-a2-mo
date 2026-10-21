"""Live BL-B-06/07 checks: scenario placement, evaluation truth and event sensors.

All checks run on an otherwise empty Town10HD_Opt with one synchronous tick owner.
They validate the scoring path, not driving behaviour.

1. **Placement repeatability (BL-B-06).** Each of S1-S5 is spawned twice from its
   scenario file in a fresh placement. Both poses must agree within 1 cm and 0.1°.
   The obstacle must be inside the ego's corridor for S1, S2, S4 and S5, and
   outside it for S3.
2. **Stop-margin truth (BL-B-07).** A car or pedestrian is placed so the gap from
   the ego's front face to the obstacle's near face is known along the straight
   mission start. ``ego_truth`` must agree within 5 cm. A placement beside the
   corridor must be reported as not in the corridor.
3. **Collision deduplication (BL-B-07).** The ego touches a stationary car,
   backs off, waits and touches it again. The raw callbacks must reduce to exactly
   two contacts under the documented 0.5 s rule.
4. **Lane invasion (BL-B-07).** ``PathFollower`` drives a smooth lane change across
   a broken marking. The crossing must be reported and counted as unplanned. It is
   excluded only when a legal-manoeuvre declaration covers it.

Raw callbacks and placement records are written for every check, failed or
passed. Evaluation truth is never given to a controller here: the drives in checks
3 and 4 are scripted from the ego's own pose.
"""

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

from evaluation.scenario_events import count_contacts, count_unplanned_invasions
from evaluation.scenarios import load_scenarios
from navigation_control import PathFollower
from navigation_control.path import ReferencePath
from navigation_control.types import ControllerConfig, Waypoint
from planning.baseline_mission import load_frozen_mission

from .scenario_world import (
    ScenarioEventSensors,
    ScenarioObstacle,
    core_xy,
    ego_front_offset,
    ego_truth,
    event_records,
)
from .vehicle import FixedVehicleAdapter
from .world import SynchronousWorldSession, WorldTiming, _settings_record

POSE_TOLERANCE_M = 0.01
YAW_TOLERANCE_DEG = 0.1
GAP_TOLERANCE_M = 0.05
EXPECT_IN_CORRIDOR = {"S1": True, "S2": True, "S3": False, "S4": True, "S5": True}
GAP_CASES = (
    ("S1", 3.0, 0.0),
    ("S1", 6.0, 0.0),
    ("S1", 10.0, 0.0),
    ("S1", 6.0, 0.8),
    ("S1", 8.0, 3.5),
    ("S2", 3.0, 0.0),
    ("S2", 6.0, 0.0),
)


def _local_point_core(transform, box, x):
    import carla

    point = carla.Location(x=x, y=box.location.y, z=box.location.z)
    transform.transform(point)
    return core_xy(point)


def _forward_speed(vehicle):
    velocity = vehicle.get_velocity()
    yaw = math.radians(vehicle.get_transform().rotation.yaw)
    return velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw)


def _foreign_actors(world):
    return [
        a for a in world.get_actors() if a.type_id.startswith(("vehicle.", "walker.", "sensor."))
    ]


class Verification:
    def __init__(self, owner, mission, scenarios):
        self.owner, self.mission, self.scenarios = owner, mission, scenarios
        self.world = owner.world
        self.path = ReferencePath(mission.waypoints)
        self.points_xyz = [tuple(p) for p in mission.route["points_xyz_m"]]

    def tick(self, count=1):
        for _ in range(count):
            self.owner.tick()

    def now(self):
        return self.world.get_snapshot().timestamp.elapsed_seconds

    def spawn_ego(self, transform=None):
        import carla

        if transform is None:
            spawn = self.mission.spawn_pose
            transform = carla.Transform(
                carla.Location(x=spawn["x_m"], y=-spawn["y_m"], z=spawn["z_m"]),
                carla.Rotation(yaw=-math.degrees(spawn["yaw_rad"])),
            )
        blueprint = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        blueprint.set_attribute("role_name", "scenario-verification")
        ego = self.owner.try_spawn(blueprint, transform)
        if ego is None:
            raise RuntimeError("Ego spawn site is occupied")
        ego.apply_control(carla.VehicleControl(brake=1.0))
        self.tick(20)
        adapter = FixedVehicleAdapter(ego)
        front, box = ego_front_offset(ego, adapter.geometry.rear_axle_offset_m)
        return ego, adapter, front, box

    def destroy(self, *actors):
        for actor in actors:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.tick(2)

    # 1. Placement repeatability -------------------------------------------------
    def placement_case(self, scenario_id, ego, adapter, front, ego_box):
        definition = self.scenarios[scenario_id]
        records = []
        truth = None
        for attempt in (1, 2):
            obstacle = ScenarioObstacle(self.owner, definition, self.path, self.points_xyz)
            try:
                placement = obstacle.spawn()
                self.tick(3)
                actual = obstacle.actor.get_transform()
                placement["after_tick"] = {
                    "x": actual.location.x,
                    "y": actual.location.y,
                    "z": actual.location.z,
                    "yaw_deg": actual.rotation.yaw,
                }
                if attempt == 1:
                    truth = ego_truth(self.path, adapter.state(), front, ego_box.extent.y, obstacle)
                records.append(placement)
            finally:
                self.destroy(obstacle.actor)
        a, b = (r["after_tick"] for r in records)
        position_error = math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))
        yaw_error = abs((a["yaw_deg"] - b["yaw_deg"] + 180) % 360 - 180)
        settled = [
            math.dist(
                (r["after_tick"]["x"], r["after_tick"]["y"]),
                (r["carla_transform"]["x"], r["carla_transform"]["y"]),
            )
            for r in records
        ]
        in_corridor_ok = truth["truth_in_corridor"] == EXPECT_IN_CORRIDOR[scenario_id]
        ok = (
            position_error <= POSE_TOLERANCE_M
            and yaw_error <= YAW_TOLERANCE_DEG
            and max(settled) <= POSE_TOLERANCE_M
            and in_corridor_ok
        )
        return {
            "check": f"placement {scenario_id}",
            "status": "PASS" if ok else "FAIL",
            "repeat_position_difference_m": position_error,
            "repeat_yaw_difference_deg": yaw_error,
            "requested_vs_after_tick_m": settled,
            "expected_in_corridor": EXPECT_IN_CORRIDOR[scenario_id],
            "truth_from_mission_start": truth,
            "placements": records,
        }

    # 2. Known-distance stop-margin truth -----------------------------------------
    def gap_case(self, scenario_id, distance_m, lateral_m, ego, adapter, front, ego_box):
        ego_pose = ego.get_transform()
        ego_front_s = self.path.project(
            *_local_point_core(ego_pose, ego_box, ego_box.location.x + ego_box.extent.x)
        ).progress_m
        base = self.scenarios[scenario_id]
        probe = ScenarioObstacle(
            self.owner,
            dataclasses.replace(
                base,
                obstacle=dataclasses.replace(
                    base.obstacle, route_s_m=ego_front_s + distance_m + 6.0, lateral_m=0.0
                ),
            ),
            self.path,
            self.points_xyz,
        )
        probe.spawn()
        near_offset = probe.box.location.x - probe.box.extent.x  # origin -> rear face
        self.destroy(probe.actor)
        definition = dataclasses.replace(
            base,
            obstacle=dataclasses.replace(
                base.obstacle,
                route_s_m=ego_front_s + distance_m - near_offset,
                lateral_m=lateral_m,
                yaw_offset_deg=0.0,
            ),
        )
        obstacle = ScenarioObstacle(self.owner, definition, self.path, self.points_xyz)
        try:
            obstacle.spawn()
            self.tick(5)
            state = adapter.state()
            truth = ego_truth(self.path, state, front, ego_box.extent.y, obstacle)
            pose = obstacle.actor.get_transform()
            near_s = self.path.project(
                *_local_point_core(pose, obstacle.box, near_offset)
            ).progress_m
            front_s = self.path.project(
                *_local_point_core(
                    ego.get_transform(), ego_box, ego_box.location.x + ego_box.extent.x
                )
            ).progress_m
            ego_reference_s = self.path.project(state.x_m, state.y_m).progress_m + front
            expected_inside = (
                abs(lateral_m) < truth["truth_corridor_half_width_m"] + obstacle.box.extent.y
            )
            result = {
                "check": f"truth gap {base.obstacle.kind} {distance_m:g} m lateral {lateral_m:+g} m",
                "expected_in_corridor": expected_inside,
                "truth": truth,
                "face_to_face_s_m": near_s - front_s,
                "front_reference_consistency_m": ego_reference_s - front_s,
                "obstacle_footprint_core_m": [list(p) for p in obstacle.corners],
            }
            if expected_inside:
                gap = truth["truth_front_gap_m"]
                error = None if gap is None else gap - (near_s - front_s)
                result["error_m"] = error
                ok = (
                    error is not None
                    and abs(error) <= GAP_TOLERANCE_M
                    and abs(result["front_reference_consistency_m"]) <= GAP_TOLERANCE_M
                )
            else:
                ok = not truth["truth_in_corridor"] and truth["truth_front_gap_m"] is None
            result["status"] = "PASS" if ok else "FAIL"
            return result
        finally:
            self.destroy(obstacle.actor)

    # 3. Collision deduplication --------------------------------------------------
    def drive(self, vehicle, *, reverse, target_mps, stop_when, max_ticks):
        import carla

        for _ in range(max_ticks):
            if stop_when():
                break
            speed = abs(_forward_speed(vehicle))
            throttle = max(0.0, min(0.5, 0.4 * (target_mps - speed) + 0.25))
            vehicle.apply_control(
                carla.VehicleControl(throttle=throttle, brake=0.0, reverse=reverse)
            )
            self.tick()
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, reverse=False))
        self.tick(30)  # 1.5 s braked: separates contacts by more than the 0.5 s rule

    def collision_case(self):
        ego, _, front, ego_box = self.spawn_ego()
        sensors = ScenarioEventSensors(self.owner, ego)
        ego_front_s = self.path.project(
            *_local_point_core(ego.get_transform(), ego_box, ego_box.location.x + ego_box.extent.x)
        ).progress_m
        base = self.scenarios["S1"]
        target = ScenarioObstacle(
            self.owner,
            dataclasses.replace(
                base,
                obstacle=dataclasses.replace(
                    base.obstacle, route_s_m=ego_front_s + 8.4, lateral_m=0.0
                ),
            ),
            self.path,
            self.points_xyz,
        )
        try:
            target.spawn()
            self.tick(5)

            def touched(since):
                return lambda: any(
                    r["channel"] == "collision" and r["simulation_time_s"] >= since
                    for r in sensors.snapshot()
                )

            first = self.now()
            self.drive(ego, reverse=False, target_mps=1.5, stop_when=touched(first), max_ticks=400)
            back = ego.get_transform().location
            self.drive(
                ego,
                reverse=True,
                target_mps=1.0,
                stop_when=lambda: ego.get_transform().location.distance(back) >= 1.5,
                max_ticks=200,
            )
            second = self.now()
            self.drive(ego, reverse=False, target_mps=1.5, stop_when=touched(second), max_ticks=400)
            raw = sensors.snapshot()
            records = event_records(raw, {}, ())
            collisions = [r for r in records if r["kind"] == "collision"]
            contacts = count_contacts(records)
            first_touch = [r for r in collisions if first <= r["simulation_time_s"] < second]
            second_touch = [r for r in collisions if r["simulation_time_s"] >= second]
            ok = (
                contacts == 2
                and bool(first_touch)
                and bool(second_touch)
                and all(r["other_actor_id"] == str(target.actor.id) for r in collisions)
                and not sensors.callback_errors
            )
            return {
                "check": "collision deduplication",
                "status": "PASS" if ok else "FAIL",
                "raw_collision_callbacks": len(collisions),
                "first_touch_callbacks": len(first_touch),
                "second_touch_callbacks": len(second_touch),
                "counted_contacts": contacts,
                "expected_contacts": 2,
                "rule": "same other actor, consecutive callbacks <= 0.5 s apart = one contact",
                "callback_errors": sensors.callback_errors,
                "raw_callbacks": raw,
                "records": records,
            }
        finally:
            sensors.close()
            self.destroy(target.actor if target.present else None, ego)

    # 4. Lane invasion ------------------------------------------------------------
    def lane_change_site(self):
        import carla

        world_map = self.world.get_map()
        for spawn in world_map.get_spawn_points():
            wp = world_map.get_waypoint(spawn.location)
            if wp is None or wp.is_junction or wp.lane_type != carla.LaneType.Driving:
                continue
            for side in ("left", "right"):
                neighbour = wp.get_left_lane() if side == "left" else wp.get_right_lane()
                marking = wp.left_lane_marking if side == "left" else wp.right_lane_marking
                if (
                    neighbour is None
                    or neighbour.lane_type != carla.LaneType.Driving
                    or neighbour.lane_id * wp.lane_id <= 0
                    or str(marking.type) != "Broken"
                ):
                    continue
                pairs, ok = [], True
                for d in range(0, 46):
                    base = wp
                    if d:
                        nxt = wp.next(float(d))
                        if len(nxt) != 1 or nxt[0].is_junction:
                            ok = False
                            break
                        base = nxt[0]
                    other = base.get_left_lane() if side == "left" else base.get_right_lane()
                    if other is None or other.lane_type != carla.LaneType.Driving:
                        ok = False
                        break
                    turn = math.radians(base.transform.rotation.yaw - wp.transform.rotation.yaw)
                    if abs(math.atan2(math.sin(turn), math.cos(turn))) > math.radians(3):
                        ok = False
                        break
                    pairs.append((base.transform.location, other.transform.location))
                if ok:
                    return wp, side, pairs
        raise RuntimeError("No straight two-lane same-direction site with a broken marking")

    def lane_case(self):
        wp, side, pairs = self.lane_change_site()
        points = []
        for d, (base, other) in enumerate(pairs):
            t = min(1.0, max(0.0, (d - 10) / 15))
            blend = t * t * (3 - 2 * t)
            points.append(
                (base.x + blend * (other.x - base.x), -(base.y + blend * (other.y - base.y)))
            )
        lane_path = ReferencePath(Waypoint(*p) for p in points)
        transform = wp.transform
        transform.location.z += 0.5
        ego, adapter, _, _ = self.spawn_ego(transform)
        sensors = ScenarioEventSensors(self.owner, ego)
        follower = PathFollower(
            ControllerConfig(
                wheelbase_m=adapter.geometry.wheelbase_m,
                max_steering_angle_rad=adapter.geometry.controller_steering_limit_rad,
                lookahead_m=5.0,
            )
        )
        follower.set_path(Waypoint(*p) for p in points)
        progress = {}
        try:
            for _ in range(800):
                state = adapter.state()
                frame = self.world.get_snapshot().frame
                progress[frame] = lane_path.project(state.x_m, state.y_m).progress_m
                if progress[frame] >= 40:
                    break
                adapter.apply(follower.update(state, 3.0, 0.05).command)
                self.tick()
            adapter.stop()
            self.tick(10)
            raw = sensors.snapshot()
            declared = (
                {
                    "id": "verification-lane-change",
                    "from_s_m": 8.0,
                    "to_s_m": 30.0,
                    "markings": ("Broken",),
                    "reason": "scripted lane change in this check",
                },
            )
            undeclared_records = event_records(raw, progress, ())
            declared_records = event_records(raw, progress, declared)
            lane_raw = [r for r in raw if r["channel"] == "lane_invasion"]
            crossed_broken = any("Broken" in r["markings"] for r in lane_raw)
            ok = (
                bool(lane_raw)
                and crossed_broken
                and count_unplanned_invasions(undeclared_records) >= 1
                and count_unplanned_invasions(declared_records) == 0
                and not sensors.callback_errors
            )
            return {
                "check": "lane invasion and declared legal manoeuvre",
                "status": "PASS" if ok else "FAIL",
                "site": {"road_id": wp.road_id, "lane_id": wp.lane_id, "side": side},
                "raw_lane_callbacks": len(lane_raw),
                "counted_without_declaration": count_unplanned_invasions(undeclared_records),
                "counted_with_declaration": count_unplanned_invasions(declared_records),
                "callback_errors": sensors.callback_errors,
                "raw_callbacks": raw,
                "records_with_declaration": declared_records,
            }
        finally:
            sensors.close()
            self.destroy(ego)


def collect(client, mission_path, scenario_dir, output):
    import carla

    mission = load_frozen_mission(mission_path)
    scenarios = load_scenarios(scenario_dir)
    for definition in scenarios.values():
        if (Path(scenario_dir) / definition.mission_config).resolve() != Path(
            mission_path
        ).resolve():
            raise ValueError(f"{definition.scenario_id} refers to a different mission")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    world = client.get_world()
    if client.get_server_version() != "0.9.16" or client.get_client_version() != "0.9.16":
        raise RuntimeError("Matching CARLA 0.9.16 client and server required")
    actual_map = world.get_map()
    if actual_map.name.rsplit("/", 1)[-1].removesuffix("_Opt") != mission.map_name.removesuffix(
        "_Opt"
    ):
        raise RuntimeError(f"Expected map {mission.map_name}, received {actual_map.name}")
    opendrive = hashlib.sha256(actual_map.to_opendrive().encode("utf-8")).hexdigest()
    if opendrive != mission.opendrive_sha256:
        raise RuntimeError("Installed map OpenDRIVE hash differs from the frozen route")
    if _foreign_actors(world):
        raise RuntimeError("Use an empty world: stop other clients and their actors first")
    settings_before = _settings_record(world.get_settings())
    actors_before = {a.id for a in world.get_actors()}
    weather = world.get_weather()
    checks, error = [], None
    try:
        with SynchronousWorldSession(client, WorldTiming(fixed_delta_seconds=0.05)) as owner:
            world.set_weather(carla.WeatherParameters.ClearNoon)
            run = Verification(owner, mission, scenarios)
            ego, adapter, front, ego_box = run.spawn_ego()
            for scenario_id in ("S1", "S2", "S3", "S4", "S5"):
                print(f"Placement {scenario_id}", flush=True)
                checks.append(run.placement_case(scenario_id, ego, adapter, front, ego_box))
            for scenario_id, distance, lateral in GAP_CASES:
                print(f"Truth gap {scenario_id} {distance} m lateral {lateral} m", flush=True)
                checks.append(
                    run.gap_case(scenario_id, distance, lateral, ego, adapter, front, ego_box)
                )
            run.destroy(ego)
            print("Collision deduplication", flush=True)
            checks.append(run.collision_case())
            print("Lane invasion", flush=True)
            checks.append(run.lane_case())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        world.set_weather(weather)
    expected = 5 + len(GAP_CASES) + 2
    status = (
        "INCOMPLETE"
        if error or len(checks) != expected
        else "MEASURED_PASS"
        if all(c["status"] == "PASS" for c in checks)
        else "MEASURED_FAIL"
    )
    report = {
        "status": status,
        "task_ids": ["BL-B-06", "BL-B-07"],
        "server_version": client.get_server_version(),
        "client_version": client.get_client_version(),
        "map": actual_map.name,
        "opendrive_sha256": opendrive,
        "mission_route_sha256": mission.route_sha256,
        "scenario_sha256": {sid: d.sha256 for sid, d in scenarios.items()},
        "error": error,
        "settings_restored": _settings_record(world.get_settings()) == settings_before,
        "no_actor_leaks": {a.id for a in world.get_actors()} == actors_before,
        "checks": checks,
        "scope": (
            "Static-scene checks of scenario placement, evaluation truth and event sensors."
            " They validate scoring and scenario setup, not driving behaviour or S1-S5 acceptance."
        ),
    }
    if not (report["settings_restored"] and report["no_actor_leaks"]):
        report["status"] = "INCOMPLETE"
    (output / "scenario_verification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for check in checks:
        detail = check.get("error_m", check.get("repeat_position_difference_m", ""))
        print(f"  {check['check']}: {check['status']} {detail if detail is not None else ''}")
    print(f"{status} | {output / 'scenario_verification.json'}" + (f" | {error}" if error else ""))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--mission", type=Path, default=root / "configs/mission_baseline.json")
    parser.add_argument("--scenarios", type=Path, default=root / "configs/scenarios")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    report = collect(client, args.mission, args.scenarios, args.output)
    return 0 if report["status"] == "MEASURED_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
