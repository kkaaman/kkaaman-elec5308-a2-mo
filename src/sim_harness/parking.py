"""P1 synthetic depth rendering and independent body-contact evaluation.

Scene boxes are harness-only inputs. The controller receives FrameBundle depth,
never boxes. A downward-looking ideal sensor intentionally establishes full
near-field coverage; it is not the baseline CARLA front camera or a calibration.
"""

import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from autonomy.interfaces import CameraCalibration
from autonomy.parking import ParkingEpisode
from autonomy.parking_contracts import (
    DriveMode,
    ParkingBay,
    ParkingConfig,
    ParkingGoal,
    ParkingPhase,
    ParkingVehicle,
    angle_error,
)
from navigation_control import VehicleState
from planning.parking_geometry import bay_polygon, contains, footprint, intersects, rectangle
from sim_harness.kinematic import KinematicConfig, KinematicWorld
from sim_harness.synthetic import SyntheticScenario


@dataclass(frozen=True)
class DepthBox:
    corners: tuple[tuple[float, float], ...]
    height_m: float = 0.8
    first_frame: int = 0
    last_frame: int = 100000

    def active(self, frame_id):
        return self.first_frame <= frame_id <= self.last_frame


def render_parking_depth(frame, boxes=(), *, unknown=False):
    """Conservative box rasterization into an ideal top-down axial-depth image.

    The background is an explicitly observed flat ground plane at z=0. Every
    pixel touched by a projected box bound gets its nearest depth. Ego rendering
    is omitted in this fixture. An actual rig needs ego masking and independent
    coverage; unknown pixels stay unknown in the production clearance provider.
    """
    height, width = frame.rgb.shape[:2]
    focal = width / (2 * math.tan(math.radians(120) / 2))
    calibration = CameraCalibration(focal, focal, width/2, height/2,
                                   ((1, 0, 0, 0), (0, -1, 0, 0),
                                    (0, 0, -1, 8), (0, 0, 0, 1)))
    depth = np.full((height, width), np.nan if unknown else 8.0)
    state = frame.ego.state
    c, s = math.cos(state.yaw_rad), math.sin(state.yaw_rad)
    for box in boxes:
        if not box.active(frame.frame_id) or unknown:
            continue
        if not 0 < box.height_m < 6:
            raise ValueError("Synthetic box height must be between 0 and 6 m")
        local = (np.asarray(box.corners) - (state.x_m, state.y_m)) @ np.array([[c, -s], [s, c]])
        us, vs = [], []
        for z in (8.0, 8.0 - box.height_m):
            us.extend(width/2 + focal * local[:, 0] / z)
            vs.extend(height/2 - focal * local[:, 1] / z)
        x0, x1 = max(0, math.floor(min(us))), min(width-1, math.ceil(max(us)))
        y0, y1 = max(0, math.floor(min(vs))), min(height-1, math.ceil(max(vs)))
        if x0 <= x1 and y0 <= y1:
            depth[y0:y1+1, x0:x1+1] = np.minimum(
                depth[y0:y1+1, x0:x1+1], 8.0 - box.height_m
            )
    return replace(frame, calibration=calibration, depth_m=depth, depth_frame_id=frame.frame_id)


def motion_envelope(a, b, vehicle):
    """Independent conservative bound for one constant-curvature bicycle step."""
    turn = angle_error(b.yaw_rad, a.yaw_rad)
    yaw = a.yaw_rad + turn / 2
    c, s = math.cos(yaw), math.sin(yaw)
    positions = np.array([[a.x_m, a.y_m], [b.x_m, b.y_m]])
    local = positions @ np.array([[c, -s], [s, c]])
    low, high = local.min(axis=0), local.max(axis=0)
    centre = (low + high) / 2
    world_centre = centre @ np.array([[c, s], [-s, c]])
    travel = math.hypot(b.x_m-a.x_m, b.y_m-a.y_m)
    padding = 2 * vehicle.corner_radius_m * math.sin(abs(turn)/4) + travel * abs(turn)/2
    return rectangle(*world_centre, yaw,
                     vehicle.front_m + (high[0]-low[0])/2,
                     vehicle.rear_m + (high[0]-low[0])/2,
                     vehicle.half_width_m + (high[1]-low[1])/2, padding)


def standard_case(name):
    """Explicit synthetic fixtures; dimensions and dynamics are illustrative."""
    vehicle = ParkingVehicle(3.00465, math.radians(20), 4.0, 1.0, 0.9, 1.7)
    start = VehicleState(0, 0, 0, 0)
    x, y, yaw = 16.0, 0.0, 0.0
    if name in ("left", "rotated"):
        x, y, yaw = 20.0, 20.0, math.pi/2
    elif name == "right":
        x, y, yaw = 20.0, -20.0, -math.pi/2
    elif name not in ("straight", "blocked", "resume", "depth_dropout", "side_blocked"):
        raise ValueError("Unknown P1 fixture")
    if name == "rotated":
        turn = math.radians(130)
        c, s = math.cos(turn), math.sin(turn)
        x, y = c*x - s*y + 5, s*x + c*y - 3
        yaw += turn
        start = VehicleState(5, -3, turn, 0)
    bay = ParkingBay(x + 1.5*math.cos(yaw), y + 1.5*math.sin(yaw), yaw, 7.0, 3.8)
    goal = ParkingGoal(x, y, yaw, bay)
    area = ((-50., -50.), (50., -50.), (50., 50.), (-50., 50.))
    boxes = ()
    if name in ("blocked", "resume"):
        boxes = (DepthBox(rectangle(13, 0, 0, 0.5, 0.5, 0.6),
                          last_frame=320 if name == "resume" else 100000),)
    if name == "side_blocked":
        boxes = (DepthBox(rectangle(11, 1.3, 0, 0.5, 0.5, 0.25)),)
    return start, goal, vehicle, area, boxes


def run_parking_case(name, *, config=None, sample_sink=None):
    config = ParkingConfig() if config is None else config
    start, goal, vehicle, area, boxes = standard_case(name)
    world = KinematicWorld(
        KinematicConfig(dt_s=config.dt_s, wheelbase_m=vehicle.wheelbase_m,
                        max_steering_angle_rad=vehicle.steering_limit_rad),
        initial_state=start, scenario=SyntheticScenario(image_width=256, image_height=256),
        clock=lambda: 100.0,
    )
    raw, _ = world.reset(seed=7)
    frame = render_parking_depth(raw, boxes)
    episode = ParkingEpisode(frame, goal, vehicle, area, config)
    rows, contacts, boundary_violations = [], 0, 0
    blocked_case = name in ("blocked", "side_blocked")
    max_ticks = 700 if blocked_case else math.ceil(config.max_duration_s / config.dt_s) + 2
    parked_ticks = 0
    try:
        for _ in range(max_ticks):
            unknown = name == "depth_dropout" and 140 <= raw.frame_id <= 165
            frame = render_parking_depth(raw, boxes, unknown=unknown)
            result = episode.step(frame, now_monotonic_s=100.0)
            state, command = frame.ego.state, result.packet.command
            rows.append(dict(
                frame_id=frame.frame_id, sim_time_s=frame.sim_time_s,
                x_m=state.x_m, y_m=state.y_m, yaw_rad=state.yaw_rad, speed_mps=state.speed_mps,
                position_error_m=result.position_error_m, heading_error_rad=result.heading_error_rad,
                target_speed_mps=result.target_speed_mps, progress_m=result.progress_m,
                steering_angle_rad=command.steering_angle_rad, throttle=command.throttle, brake=command.brake,
                mode=result.motion.mode.value, phase=result.phase.value, clearance=result.clearance,
                obstacle_gap_m=result.obstacle_gap_m, reason=result.reason,
            ))
            if sample_sink is not None:
                sample_sink(rows[-1])
            if result.phase == ParkingPhase.FAULT:
                break
            next_raw = world.step(result.packet)
            envelope = motion_envelope(state, next_raw.ego.state, vehicle)
            contacts += sum(intersects(envelope, box.corners) for box in boxes if box.active(frame.frame_id))
            boundary_violations += not contains(area, envelope)
            raw = next_raw
            if episode.mode == DriveMode.PARK:
                parked_ticks += 1
                if parked_ticks >= 10:
                    break
    finally:
        world.close()
    last = rows[-1]
    transitions = sum(event.event == "drive_to_park" for event in episode.events)
    final_state = VehicleState(last["x_m"], last["y_m"], last["yaw_rad"], last["speed_mps"])
    final_body_fits = contains(bay_polygon(goal.bay), footprint(final_state, vehicle, goal.clearance_m))
    common = (contacts == 0 and boundary_violations == 0
              and max(row["speed_mps"] for row in rows) <= 1.0 + 1e-9)
    if blocked_case:
        passed = common and transitions == 0 and last["phase"] == "hold" and last["speed_mps"] <= config.stopped_speed_mps
    else:
        passed = (common and final_body_fits and transitions == 1 and last["mode"] == "park"
                  and last["speed_mps"] <= config.stopped_speed_mps
                  and last["position_error_m"] <= goal.position_tolerance_m
                  and last["heading_error_rad"] <= goal.heading_tolerance_rad)
    summary = dict(
        case=name, passed=passed, scope="synthetic_depth_kinematic_P1",
        contacts=contacts, drive_to_park_count=transitions, route_install_count=episode.route_install_count,
        boundary_violations=boundary_violations, final_body_fits=final_body_fits,
        events=[asdict(event) for event in episode.events],
        max_speed_mps=max(row["speed_mps"] for row in rows),
        max_curvature_bound=max(segment.curvature_bound for segment in episode.plan.segments),
        curvature_limit=vehicle.curvature_limit, final=last,
    )
    return summary, rows, episode.plan, goal, vehicle
