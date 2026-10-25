"""Live S1-S5 obstacles, evaluation truth and event sensors (BL-B-06/07).

- ``ScenarioObstacle`` places the scenario's obstacle where the scenario file says.
  The pose comes from ``evaluation.scenarios.placement_pose`` and is snapped to the
  ground. Physics is off, so contact cannot move it. The placement record is
  written at spawn, and the scripted removal is recorded when it happens.
- ``ScenarioEventSensors`` attaches CARLA's collision and lane-invasion sensors to
  the ego. It keeps every raw callback and builds S0-schema event records from
  them (see ``evaluation.scenario_events``).
- ``ego_truth`` is the per-tick evaluation-only stop-margin truth.

Everything here is evaluation-side. Callers log it beside the run and must never
pass it into ``AutonomyStack.step``: the stack only ever sees ``FrameBundle``.
Read bounding boxes before cameras stream. ``actor.bounding_box`` is a server call
that holds the Python GIL, so it can stall while camera callbacks run.
"""

import math
import threading

from evaluation.scenario_events import (
    COLLISION_DEDUP_S,
    CollisionContacts,
    collision_record,
    lane_invasion_record,
)
from evaluation.scenarios import placement_pose
from evaluation.stop_truth import (
    EVALUATION_CLEARANCE_M,
    ego_front_offset_m,
    route_surface_gap,
)

GROUND_SEARCH_ABOVE_ROAD_M = 1.5
GROUND_SEARCH_DISTANCE_M = 4.0


def core_xy(location):
    """CARLA (left-handed, y right) to the core frame (x forward, y left)."""
    return (location.x, -location.y)


def route_z(points_xyz_m, path, progress_m):
    """Road height at a route arc length, interpolated from the frozen route points."""
    s = min(max(progress_m, 0.0), path.length_m)
    for (a, b), (sa, sb) in zip(
        zip(points_xyz_m, points_xyz_m[1:]), zip(path.arc_lengths, path.arc_lengths[1:])
    ):
        if s <= sb:
            t = 0.0 if sb == sa else (s - sa) / (sb - sa)
            return a[2] + t * (b[2] - a[2])
    return points_xyz_m[-1][2]


def footprint_core(transform, box):
    """Bounding-box XY corners in the core frame for a given actor transform.

    Uses the box's local centre and yaw. Pass the transform the actor was placed
    with: in synchronous mode ``get_transform()`` reports the old pose until the
    world ticks.
    """
    import carla

    yaw = math.radians(box.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    corners = []
    for x, y in (
        (box.extent.x, -box.extent.y),
        (box.extent.x, box.extent.y),
        (-box.extent.x, box.extent.y),
        (-box.extent.x, -box.extent.y),
    ):
        point = carla.Location(
            x=box.location.x + c * x - s * y, y=box.location.y + s * x + c * y, z=box.location.z
        )
        transform.transform(point)
        corners.append(core_xy(point))
    return corners


def _box_record(box):
    return {
        "location_m": [box.location.x, box.location.y, box.location.z],
        "extent_m": [box.extent.x, box.extent.y, box.extent.z],
        "yaw_deg": box.rotation.yaw,
    }


class ScenarioObstacle:
    """Spawn, record and remove one scenario obstacle through the world's tick owner."""

    def __init__(self, owner, definition, path, points_xyz_m):
        self.owner, self.definition, self.path = owner, definition, path
        self.points_xyz_m = points_xyz_m
        self.actor = None
        self.box = None
        self.corners = None
        self.placement = None
        self.removal = None
        self.removal_failures = []  # attempts whose destroy() returned False: the obstacle stayed

    def spawn(self):
        import carla

        world = self.owner.world
        obstacle = self.definition.obstacle
        x, y, yaw = placement_pose(self.path, obstacle)
        road_z = route_z(self.points_xyz_m, self.path, obstacle.route_s_m)
        ground = world.ground_projection(
            carla.Location(x=x, y=-y, z=road_z + GROUND_SEARCH_ABOVE_ROAD_M),
            GROUND_SEARCH_DISTANCE_M,
        )
        if ground is None:
            raise RuntimeError(
                f"{self.definition.scenario_id}: no ground within {GROUND_SEARCH_DISTANCE_M} m"
                f" below the road at s={obstacle.route_s_m:.1f} m, lateral {obstacle.lateral_m:+.1f} m"
            )
        transform = carla.Transform(
            carla.Location(x=x, y=-y, z=ground.location.z + 1.0),
            carla.Rotation(yaw=-math.degrees(yaw)),
        )
        blueprint = world.get_blueprint_library().find(obstacle.blueprint)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", f"scenario-{self.definition.scenario_id}")
        actor = self.owner.try_spawn(blueprint, transform)
        if actor is None:
            raise RuntimeError(
                f"{self.definition.scenario_id}: obstacle site is occupied at s={obstacle.route_s_m:.1f} m"
            )
        self.actor = actor
        actor.set_simulate_physics(False)
        box = actor.bounding_box  # before any camera streams
        bottom = min(v.z for v in box.get_world_vertices(transform))
        transform.location.z += ground.location.z + 0.01 - bottom
        actor.set_transform(transform)
        self.box = box
        self.corners = footprint_core(transform, box)
        waypoint = world.get_map().get_waypoint(transform.location, project_to_road=True)
        self.placement = {
            "scenario_id": self.definition.scenario_id,
            "scenario_sha256": self.definition.sha256,
            "actor_id": actor.id,
            "blueprint": actor.type_id,
            "requested": {
                "route_s_m": obstacle.route_s_m,
                "lateral_m": obstacle.lateral_m,
                "yaw_offset_deg": obstacle.yaw_offset_deg,
            },
            "pose_core": {"x_m": x, "y_m": y, "yaw_rad": yaw},
            "carla_transform": {
                "x": transform.location.x,
                "y": transform.location.y,
                "z": transform.location.z,
                "yaw_deg": transform.rotation.yaw,
            },
            "ground_z_m": ground.location.z,
            "bounding_box": _box_record(box),
            "footprint_core_m": [list(p) for p in self.corners],
            "map_lane": None
            if waypoint is None
            else {
                "road_id": waypoint.road_id,
                "lane_id": waypoint.lane_id,
                "is_junction": waypoint.is_junction,
                "lane_type": str(waypoint.lane_type),
            },
        }
        return self.placement

    def remove(self, *, frame, sim_time_s, reason):
        """Destroy the obstacle (scripted removal) and record when.

        The obstacle counts as removed only once ``destroy()`` has succeeded: a failed
        destruction leaves it present (the truth keeps reporting it) and is recorded in
        ``removal_failures`` with the attempt's frame and time (review of PR #98).
        """
        if self.actor is None or self.removal is not None:
            raise RuntimeError("Obstacle is not present")
        record = {"frame": frame, "simulation_time_s": sim_time_s, "reason": reason}
        destroyed = bool(self.actor.destroy())
        if not destroyed:
            self.removal_failures.append(dict(record, destroyed=False))
            return dict(record, destroyed=False)
        self.removal = dict(record, destroyed=True)
        return self.removal

    @property
    def present(self):
        return self.actor is not None and self.removal is None


def ego_front_offset(vehicle, rear_axle_offset_m):
    """Rear axle to front bumper for the ego; read before cameras stream."""
    box = vehicle.bounding_box
    return ego_front_offset_m(rear_axle_offset_m, box.location.x, box.extent.x), box


def ego_truth(path, ego_state, front_offset_m, ego_half_width_m, obstacle):
    """Per-tick evaluation-only truth; never an input to the autonomy stack."""
    half_width = ego_half_width_m + EVALUATION_CLEARANCE_M
    if obstacle is None or not obstacle.present:
        return {
            "truth_front_gap_m": None,
            "truth_in_corridor": False,
            "truth_obstacle_present": False,
            "truth_corridor_half_width_m": half_width,
        }
    record = route_surface_gap(
        path, ego_state.x_m, ego_state.y_m, front_offset_m, obstacle.corners, half_width
    )
    record["truth_obstacle_present"] = True
    return record


class ScenarioEventSensors:
    """Collision and lane-invasion sensors on the ego; keeps every raw callback."""

    def __init__(self, owner, vehicle):
        import carla

        self.owner = owner
        self.raw = []
        self.callback_errors = []
        self._lock = threading.Lock()
        self.sensors = []
        library = owner.world.get_blueprint_library()
        for kind in ("collision", "lane_invasion"):
            # SynchronousWorldSession.try_spawn refuses sensors; register for cleanup
            # exactly as carla_adapter.s0_runner.RunSensors does.
            sensor = owner.world.spawn_actor(
                library.find("sensor.other." + kind),
                carla.Transform(),
                attach_to=vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
            owner._actors.append(sensor)
            self.sensors.append(sensor)
            sensor.listen(lambda event, channel=kind: self._receive(channel, event))

    def _receive(self, channel, event):
        try:
            row = {"channel": channel, "frame": event.frame, "simulation_time_s": event.timestamp}
            if channel == "collision":
                other = event.other_actor
                impulse = event.normal_impulse
                row.update(
                    other_actor_id=None if other is None else other.id,
                    other_actor_type=None if other is None else other.type_id,
                    impulse=[impulse.x, impulse.y, impulse.z],
                )
            else:
                row["markings"] = [str(m.type) for m in event.crossed_lane_markings]
            with self._lock:
                row["arrival"] = len(self.raw)
                self.raw.append(row)
        except Exception as error:  # a callback must never raise into CARLA
            with self._lock:
                self.callback_errors.append(f"{channel} callback: {error}")

    def snapshot(self):
        with self._lock:
            return [dict(r) for r in self.raw]

    def close(self):
        for sensor in self.sensors:
            try:
                if sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            except Exception as error:
                self.callback_errors.append(f"sensor cleanup {sensor.id}: {error}")


def event_records(raw, progress_by_frame, exclusions, window_s=COLLISION_DEDUP_S):
    """S0-schema records from raw callbacks, in simulation order.

    ``progress_by_frame`` maps a world frame to the ego's route arc length. A lane
    callback takes the latest progress at or before its frame.
    """
    contacts = CollisionContacts(window_s)
    frames = sorted(progress_by_frame)
    records = []
    sequence = 0
    for row in sorted(raw, key=lambda r: (r["simulation_time_s"], r["frame"], r["arrival"])):
        if row["channel"] == "collision":
            records.append(
                collision_record(
                    contacts,
                    frame=row["frame"],
                    sim_time_s=row["simulation_time_s"],
                    other_actor_id=row["other_actor_id"],
                    other_actor_type=row["other_actor_type"],
                    impulse=row["impulse"],
                )
            )
        else:
            sequence += 1
            earlier = [f for f in frames if f <= row["frame"]]
            records.append(
                lane_invasion_record(
                    sequence=sequence,
                    frame=row["frame"],
                    sim_time_s=row["simulation_time_s"],
                    markings=row["markings"],
                    route_s_m=progress_by_frame[earlier[-1]] if earlier else None,
                    exclusions=exclusions,
                )
            )
    return records
