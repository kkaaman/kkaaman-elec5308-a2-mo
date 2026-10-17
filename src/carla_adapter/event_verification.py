"""BL-B-07: live checks of evaluation truth and event sensors before scoring runs.

Three fixtures on an otherwise empty Town10HD_Opt, one synchronous owner:

1. Known-distance truth gap. A parked Tesla and a physics-disabled car/walker
   placed so the ego front face to obstacle near face distance along the
   straight mission start is known. ``truth_record`` (the acceptance metric's
   bumper-to-surface gap) must agree within 5 cm, and a roadside placement
   must be reported outside the corridor.
2. Collision deduplication. The ego drives into a stationary car at walking
   speed, backs off, waits, and touches it again. Raw callbacks must reduce to
   exactly two contacts with the same rule that scores runs.
3. Lane invasion. The production PathFollower drives a smooth lane change
   across a broken marking. The crossing must be reported, counted when
   undeclared, and excluded only by an explicit route-interval declaration.

Ground truth here is evaluation-only. Nothing is fed to the autonomy stack.
"""

import argparse
import hashlib
import json
import math
import threading
from pathlib import Path

from carla_adapter.ground_truth import footprint_xy, truth_record
from carla_adapter.mission import assert_current_map
from carla_adapter.vehicle import FixedVehicleAdapter
from carla_adapter.world import SynchronousWorldSession, _settings_record
from evaluation.event_checks import (
    annotate_invasion,
    check_gap,
    count_contacts,
    count_unplanned_invasions,
    verification_status,
)
from navigation_control import PathFollower
from navigation_control.types import ControllerConfig, Waypoint
from planning.corridor import RouteGeometry
from planning.mission import Mission

HALF_WIDTH_M = 1.08173 + 0.15


def _core(location):
    return (location.x, -location.y)


def _local_point(actor, x, y=0.0):
    """World core XY of an actor-local point at bounding-box centre height."""
    import carla

    box = actor.bounding_box
    point = carla.Location(x=x, y=y, z=box.location.z)
    actor.get_transform().transform(point)
    return _core(point)


def _front_face(actor):
    box = actor.bounding_box
    return _local_point(actor, box.location.x + box.extent.x, box.location.y)


def _rear_face(actor):
    box = actor.bounding_box
    return _local_point(actor, box.location.x - box.extent.x, box.location.y)


class EventLog:
    """Thread-safe collision/lane-invasion callbacks attached to one vehicle."""

    def __init__(self, owner, world, vehicle):
        import carla

        self.rows, self.lock, self.sensors = [], threading.Lock(), []
        for kind in ("collision", "lane_invasion"):
            bp = world.get_blueprint_library().find("sensor.other." + kind)
            sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
            owner._actors.append(sensor)
            self.sensors.append(sensor)
            sensor.listen(lambda event, channel=kind: self._receive(channel, event))

    def _receive(self, channel, event):
        row = dict(type=channel, frame_id=event.frame, sim_time_s=event.timestamp)
        if channel == "collision":
            row.update(
                other_actor_id=event.other_actor.id,
                other_type=event.other_actor.type_id,
                impulse=[event.normal_impulse.x, event.normal_impulse.y, event.normal_impulse.z],
            )
        else:
            row.update(markings=[str(m.type) for m in event.crossed_lane_markings])
        with self.lock:
            self.rows.append(row)

    def snapshot(self):
        with self.lock:
            return [dict(r) for r in self.rows]

    def close(self):
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()


class Fixture:
    def __init__(self, client, mission, follow=False):
        self.client, self.mission, self.follow = client, mission, follow
        self.world = client.get_world()
        self.map = self.world.get_map()
        self.route = RouteGeometry(mission.points_xy_m)
        self.library = self.world.get_blueprint_library()

    def tick(self, owner, count=1, vehicle=None):
        for _ in range(count):
            owner.tick()
            if self.follow and vehicle is not None and vehicle.is_alive:
                import carla

                pose = vehicle.get_transform()
                yaw = math.radians(pose.rotation.yaw)
                self.world.get_spectator().set_transform(
                    carla.Transform(
                        carla.Location(
                            x=pose.location.x - 8 * math.cos(yaw),
                            y=pose.location.y - 8 * math.sin(yaw),
                            z=pose.location.z + 5,
                        ),
                        carla.Rotation(pitch=-22, yaw=pose.rotation.yaw),
                    )
                )

    def spawn_ego(self, owner, transform=None):
        import carla

        if transform is None:
            pose = self.mission.map_validation["spawn_actor_pose"]
            transform = carla.Transform(
                carla.Location(x=pose["x"], y=pose["y"], z=pose["z"]),
                carla.Rotation(yaw=pose["yaw_deg"]),
            )
        ego = owner.try_spawn(self.library.find("vehicle.tesla.model3"), transform)
        if ego is None:
            raise RuntimeError("Ego spawn site occupied")
        ego.apply_control(carla.VehicleControl(brake=1.0))
        self.tick(owner, 20, ego)
        return ego

    def place(self, owner, kind, origin_s, lateral):
        """Physics-disabled obstacle with its actor origin at route s/lateral."""
        import carla

        x, y, yaw = self.route.at(origin_s, lateral)
        ground = self.world.ground_projection(carla.Location(x=x, y=-y, z=10.0), 30)
        if ground is None:
            raise RuntimeError("No ground at fixture site")
        blueprint = self.library.find(
            "walker.pedestrian.0001" if kind == "person" else "vehicle.tesla.model3"
        )
        transform = carla.Transform(
            carla.Location(x=x, y=-y, z=ground.location.z + 1.0),
            carla.Rotation(yaw=-math.degrees(yaw)),
        )
        actor = owner.try_spawn(blueprint, transform)
        if actor is None:
            raise RuntimeError(f"Could not place {kind} at s={origin_s:.2f}")
        actor.set_simulate_physics(False)
        bottom = min(p.z for p in actor.bounding_box.get_world_vertices(transform))
        transform.location.z += ground.location.z + 0.01 - bottom
        actor.set_transform(transform)
        return actor

    def gap_case(self, owner, ego, kind, distance_m, lateral_m):
        adapter = FixedVehicleAdapter(ego)
        ego_front_s = self.route.project(*_front_face(ego)).s_m
        # Spawn once to read the real bounding box, then move to the exact site.
        probe = self.place(owner, kind, ego_front_s + distance_m + 4.0, lateral_m)
        box = probe.bounding_box
        near_offset = box.location.x - box.extent.x
        probe.destroy()
        self.tick(owner, 2, ego)
        actor = self.place(owner, kind, ego_front_s + distance_m - near_offset, lateral_m)
        self.tick(owner, 5, ego)
        try:
            front_s = self.route.project(*_front_face(ego)).s_m
            near_s = self.route.project(*_rear_face(actor)).s_m
            truth = truth_record(ego, actor, self.route, adapter.state())
            in_corridor = abs(lateral_m) < HALF_WIDTH_M + box.extent.y
            if in_corridor:
                result = check_gap(near_s - front_s, truth["truth_gap_m"])
                result["status"] = result["status"] if truth["truth_in_corridor"] else "FAIL"
            else:
                ok = not truth["truth_in_corridor"] and truth["truth_gap_m"] is None
                result = dict(status="PASS" if ok else "FAIL", expected_m=None)
            result.update(
                case=f"{kind}@{distance_m:g}m,lateral{lateral_m:+g}m",
                kind=kind,
                requested_distance_m=distance_m,
                lateral_m=lateral_m,
                truth_in_corridor=truth["truth_in_corridor"],
                obstacle_corners_world_m=[list(p) for p in footprint_xy(actor)],
                obstacle_blueprint=actor.type_id,
            )
            return result
        finally:
            actor.destroy()
            self.tick(owner, 2, ego)

    def forward_speed(self, vehicle):
        velocity, yaw = vehicle.get_velocity(), math.radians(vehicle.get_transform().rotation.yaw)
        return velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw)

    def drive_until(self, owner, vehicle, log, *, reverse, target_mps, stop_when, max_ticks):
        import carla

        for _ in range(max_ticks):
            if stop_when():
                break
            speed = abs(self.forward_speed(vehicle))
            throttle = max(0.0, min(0.5, 0.4 * (target_mps - speed) + 0.25))
            vehicle.apply_control(
                carla.VehicleControl(throttle=throttle, brake=0.0, reverse=reverse)
            )
            self.tick(owner, 1, vehicle)
        vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0, reverse=False))
        self.tick(owner, 30, vehicle)  # 1.5 s at 20 Hz: separates contact episodes

    def collision_case(self, owner):
        ego = self.spawn_ego(owner)
        log = EventLog(owner, self.world, ego)
        ego_front_s = self.route.project(*_front_face(ego)).s_m
        car = self.place(owner, "car", ego_front_s + 6.0 + 2.4, 0.0)
        self.tick(owner, 5, ego)
        try:
            start = ego.get_transform().location

            def touched(phase_start):
                return lambda: any(
                    r["type"] == "collision" and r["sim_time_s"] >= phase_start
                    for r in log.snapshot()
                )

            def now():
                return self.world.get_snapshot().timestamp.elapsed_seconds

            first = now()
            self.drive_until(
                owner,
                ego,
                log,
                reverse=False,
                target_mps=1.5,
                stop_when=touched(first),
                max_ticks=400,
            )
            back = ego.get_transform().location
            self.drive_until(
                owner,
                ego,
                log,
                reverse=True,
                target_mps=1.0,
                stop_when=lambda: ego.get_transform().location.distance(back) >= 1.5,
                max_ticks=200,
            )
            second = now()
            self.drive_until(
                owner,
                ego,
                log,
                reverse=False,
                target_mps=1.5,
                stop_when=touched(second),
                max_ticks=400,
            )
            events = [r for r in log.snapshot() if r["type"] == "collision"]
            first_hits = [e for e in events if first <= e["sim_time_s"] < second]
            second_hits = [e for e in events if e["sim_time_s"] >= second]
            contacts = count_contacts(events)
            ok = contacts == 2 and first_hits and second_hits
            ok = ok and all(e["other_actor_id"] == car.id for e in events)
            return dict(
                status="PASS" if ok else "FAIL",
                raw_collision_callbacks=len(events),
                first_contact_callbacks=len(first_hits),
                second_contact_callbacks=len(second_hits),
                counted_contacts=contacts,
                expected_contacts=2,
                travel_m=start.distance(ego.get_transform().location),
                events=events,
                rule="callbacks per actor merge until a >= 1 s gap",
            )
        finally:
            log.close()
            car.destroy()
            ego.destroy()
            self.tick(owner, 2)

    def lane_change_site(self):
        import carla

        for spawn in self.map.get_spawn_points():
            wp = self.map.get_waypoint(spawn.location)
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
                    base = wp if d == 0 else wp.next(float(d))
                    if d:
                        if len(base) != 1 or base[0].is_junction:
                            ok = False
                            break
                        base = base[0]
                    other = base.get_left_lane() if side == "left" else base.get_right_lane()
                    if other is None or other.lane_type != carla.LaneType.Driving:
                        ok = False
                        break
                    turn = base.transform.rotation.yaw - wp.transform.rotation.yaw
                    if abs(
                        math.atan2(math.sin(math.radians(turn)), math.cos(math.radians(turn)))
                    ) > math.radians(3):
                        ok = False
                        break
                    pairs.append((base.transform.location, other.transform.location))
                if ok:
                    return wp, side, pairs
        raise RuntimeError("No straight two-lane same-direction site with a broken marking")

    def lane_case(self, owner):

        wp, side, pairs = self.lane_change_site()
        points = []
        for d, (base, other) in enumerate(pairs):
            t = min(1.0, max(0.0, (d - 10) / 15))
            blend = t * t * (3 - 2 * t)
            points.append(
                (
                    base.x + blend * (other.x - base.x),
                    -(base.y + blend * (other.y - base.y)),
                )
            )
        path = RouteGeometry(points)
        transform = wp.transform
        transform.location.z += 0.5
        ego = self.spawn_ego(owner, transform)
        adapter = FixedVehicleAdapter(ego)
        log = EventLog(owner, self.world, ego)
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
                progress[frame] = path.project(state.x_m, state.y_m).s_m
                if progress[frame] >= 40:
                    break
                adapter.apply(follower.update(state, 3.0, 0.05).command)
                self.tick(owner, 1, ego)
            adapter.stop()
            self.tick(owner, 10, ego)
            raw = [r for r in log.snapshot() if r["type"] == "lane_invasion"]

            def s_at(event):
                frames = [f for f in progress if f <= event["frame_id"]]
                return progress[max(frames)] if frames else 0.0

            declaration = (
                dict(from_s_m=8.0, to_s_m=30.0, markings=("Broken",), reason="fixture lane change"),
            )
            undeclared = [annotate_invasion(e, s_at(e), ()) for e in raw]
            declared = [annotate_invasion(e, s_at(e), declaration) for e in raw]
            crossed_broken = any("Broken" in e.get("markings", ()) for e in raw)
            ok = (
                bool(raw)
                and crossed_broken
                and count_unplanned_invasions(undeclared) >= 1
                and count_unplanned_invasions(declared) == 0
            )
            return dict(
                status="PASS" if ok else "FAIL",
                site=dict(road_id=wp.road_id, lane_id=wp.lane_id, side=side),
                raw_invasions=len(raw),
                counted_without_declaration=count_unplanned_invasions(undeclared),
                counted_with_declaration=count_unplanned_invasions(declared),
                events=declared,
            )
        finally:
            log.close()
            ego.destroy()
            self.tick(owner, 2)


def collect(client, mission_path, output, follow=False):
    import carla

    mission = Mission.load(mission_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    world = client.get_world()
    if client.get_server_version() != "0.9.16" or client.get_client_version() != "0.9.16":
        raise RuntimeError("Matching CARLA 0.9.16 client/server required")
    assert_current_map(world.get_map(), mission)
    if any(a.type_id.startswith(("vehicle.", "walker.", "sensor.")) for a in world.get_actors()):
        raise RuntimeError("Use an empty world: close other clients and their actors")
    before = _settings_record(world.get_settings())
    initial_ids = {a.id for a in world.get_actors()}
    weather, view = world.get_weather(), world.get_spectator().get_transform()
    gaps, collision, lane, error = [], None, None, None
    fixture = Fixture(client, mission, follow)
    try:
        with SynchronousWorldSession(client) as owner:
            world.set_weather(carla.WeatherParameters.ClearNoon)
            ego = fixture.spawn_ego(owner)
            for kind, distance, lateral in (
                ("car", 3.0, 0.0),
                ("car", 6.0, 0.0),
                ("car", 10.0, 0.0),
                ("car", 6.0, 0.8),
                ("car", 8.0, 4.5),
                ("person", 3.0, 0.0),
                ("person", 6.0, 0.0),
            ):
                print(f"Truth gap: {kind} {distance} m lateral {lateral} m", flush=True)
                gaps.append(fixture.gap_case(owner, ego, kind, distance, lateral))
            ego.destroy()
            fixture.tick(owner, 2)
            print("Collision deduplication", flush=True)
            collision = fixture.collision_case(owner)
            print("Lane invasion and declared crossing", flush=True)
            lane = fixture.lane_case(owner)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        world.set_weather(weather)
        world.get_spectator().set_transform(view)
    report = dict(
        status=verification_status(gaps, collision, lane),
        server=client.get_server_version(),
        map=world.get_map().name.rsplit("/", 1)[-1],
        opendrive_sha256=hashlib.sha256(
            world.get_map().to_opendrive().replace("\r\n", "\n").encode()
        ).hexdigest(),
        mission_sha256=mission.sha256,
        truth_gap_checks=gaps,
        collision=collision,
        lane_invasion=lane,
        error=error,
        settings_restored=_settings_record(world.get_settings()) == before,
        no_actor_leaks={a.id for a in world.get_actors()} == initial_ids,
        scope=(
            "Live evaluation-truth and event-sensor checks on a static scene; they"
            " validate scoring, not driving behaviour."
        ),
    )
    if error or not report["settings_restored"] or not report["no_actor_leaks"]:
        report["status"] = "INCOMPLETE"
    (output / "event_verification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"truth_gap_checks", "collision", "lane_invasion"}
            },
            indent=2,
        )
    )
    for row in gaps:
        print(f"  {row['case']}: {row['status']} {row.get('error_m', '')}")
    for name, part in (("collision", collision), ("lane", lane)):
        if part:
            print(f"  {name}: {part['status']}")
    if report["status"] != "MEASURED_PASS":
        raise RuntimeError("Event verification did not pass; inspect event_verification.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(60)
    collect(client, args.mission, args.output, args.follow)
