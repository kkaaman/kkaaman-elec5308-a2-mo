"""Lane-following trajectory with stops for obstacles and the route goal (G1b).

The trajectory follows the corridor centreline in the odometry frame. Its speed is the
lowest of:

- the cruise speed;
- the curve limit sqrt(lateral_accel / |curvature|);
- a braking ramp sqrt(2 a d) to a stop point, which is ``stop_gap_m`` short of an
  obstacle's nearest point for the front bumper, or the goal for the rear axle.

Past a stop point every speed is zero. A goal stop is sent as a final-goal trajectory
that ends at the goal, so the controller latches FINISHED there.

**Stop hold.** One sweep can miss an obstacle that the previous ones saw (occlusion,
drop-off). The last obstacle stop point is kept in the odometry frame and holds in one
of three states:

- BLOCKED: an obstacle is seen; the stop point follows it.
- CLEAR: the tube is seen clear (ground observed, nothing standing on it) to
  ``clear_beyond_m`` past where the obstacle was. Only ``resume_clear_s`` of continuous
  CLEAR releases the stop.
- UNKNOWN: neither. The stop stays. A missed or occluded obstacle is not clearance
  (review of 94b5caa; live run e released after 1 s of missed detections and crept 0.6 m).
"""

import math
from dataclasses import dataclass

import numpy as np

from mapless.corridor import Corridor
from mapless.obstacles import PathObservation, project_onto_polyline
from mapless.terrain import VehicleBody
from navigation_control.local import LocalTrajectory, TrajectorySample


@dataclass(frozen=True)
class PlannerConfig:
    cruise_mps: float = 2.0
    lateral_accel_mps2: float = 1.5
    decel_mps2: float = 1.0
    stop_gap_m: float = 2.5
    validity_s: float = 0.5
    resume_clear_s: float = 1.0
    clear_beyond_m: float = 2.0
    clear_gap_s: float = 0.15  # clear observations further apart than this are not continuous

    def __post_init__(self):
        for name in ("cruise_mps", "lateral_accel_mps2", "decel_mps2", "validity_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("stop_gap_m", "resume_clear_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class Plan:
    trajectory: LocalTrajectory | None
    stop_reason: str  # "", "obstacle", "obstacle_hold", "goal"
    stop_distance_m: float  # rear-axle distance along the path to the stop point (inf if none)
    curve_speed_mps: float  # lowest curve limit on the trajectory
    hold_state: str = ""  # "", "blocked", "clear", "unknown"


def curvature(coefficients, x):
    d1 = np.polyval(np.polyder(coefficients), x)
    d2 = np.polyval(np.polyder(coefficients, 2), x) if len(coefficients) > 2 else np.zeros_like(x)
    return d2 / (1.0 + d1**2) ** 1.5


def polyline_curvature(points):
    """Signed curvature at each vertex of a polyline (finite differences, 1/m)."""
    if len(points) < 3:
        return np.zeros(len(points))
    dx, dy = np.gradient(points[:, 0]), np.gradient(points[:, 1])
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    return (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-9)


class LanePlanner:
    def __init__(self, config, body):
        if not isinstance(config, PlannerConfig) or not isinstance(body, VehicleBody):
            raise TypeError("LanePlanner(PlannerConfig, VehicleBody)")
        self.config = config
        self.body = body
        self._hold_point = None  # odometry xy of the rear-axle stop point
        self._obstacle_point = None  # odometry xy of the held obstacle's nearest point
        self._clear_since_s = None
        self._clear_last_s = None  # the last clear observation: continuity is judged against it

    def reset(self):
        self._hold_point = None
        self._obstacle_point = None
        self._clear_since_s = self._clear_last_s = None

    def plan(self, corridor, observation, goal_body, now_s, hold_at_m=None):
        """``observation``: this sweep's ``PathObservation`` (None without a sweep this tick).

        ``goal_body``: the route goal in the vehicle frame, when it is within reach.
        ``hold_at_m``: stop the rear axle this far along the path (a junction the car may
        not enter yet); reported as stop reason "junction".
        """
        if not isinstance(corridor, Corridor):
            raise TypeError("corridor must be a Corridor")
        if observation is not None and not isinstance(observation, PathObservation):
            raise TypeError("observation must be a PathObservation or None")
        c = self.config
        if not corridor.ok:
            self._clear_since_s = self._clear_last_s = None  # no corridor: clearance is not being observed
            return Plan(None, "", math.inf, math.inf, "unknown" if self._hold_point is not None else "")
        line = corridor.centre_body
        arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))))
        s_car = float(project_onto_polyline(np.zeros((1, 2)), line)[0][0])

        stop_s, reason, state = math.inf, "", ""
        obstacle = None if observation is None else observation.obstacle
        if obstacle is not None:
            stop_s = s_car + obstacle.distance_m - self.body.front_m - c.stop_gap_m
            reason, state = "obstacle", "blocked"
            point = np.interp(stop_s, arcs, line[:, 0]), np.interp(stop_s, arcs, line[:, 1])
            self._hold_point = corridor.pose.to_world(np.array([point]))[0]
            nearest = s_car + obstacle.distance_m
            point = np.interp(nearest, arcs, line[:, 0]), np.interp(nearest, arcs, line[:, 1])
            self._obstacle_point = corridor.pose.to_world(np.array([point]))[0]
            self._clear_since_s = self._clear_last_s = None
        elif self._hold_point is not None:
            held = corridor.pose.to_body(self._hold_point[None])
            stop_s = float(project_onto_polyline(held, line)[0][0])
            where = corridor.pose.to_body(self._obstacle_point[None])
            needed = float(project_onto_polyline(where, line)[0][0]) - s_car + c.clear_beyond_m
            if observation is not None and observation.clear_to_m >= needed:
                state = "clear"
                # Continuous means every tick: a gap (a fault, a missing sweep, the planner
                # not called) restarts the count (review of 2b7e14d: one clear observation,
                # 3 s of nothing, one more, and the stop released).
                if self._clear_since_s is None or now_s - self._clear_last_s > c.clear_gap_s:
                    self._clear_since_s = now_s
                self._clear_last_s = now_s
            else:
                state = "unknown"
                self._clear_since_s = self._clear_last_s = None
            if state == "clear" and now_s - self._clear_since_s >= c.resume_clear_s:
                self._hold_point = self._obstacle_point = self._clear_since_s = self._clear_last_s = None
                stop_s, state = math.inf, "clear"
            else:
                reason = "obstacle_hold"

        if hold_at_m is not None and math.isfinite(hold_at_m) and s_car + hold_at_m < stop_s:
            stop_s, reason = s_car + hold_at_m, "junction"

        final_goal = False
        end = len(line)
        if goal_body is not None:
            goal_s, _, outside = project_onto_polyline(np.asarray(goal_body, dtype=float)[None], line)
            if not outside[0] and goal_s[0] < stop_s:
                stop_s, reason, final_goal = float(goal_s[0]), "goal", True
                end = int(np.searchsorted(arcs, stop_s)) + 1

        k = np.abs(polyline_curvature(line))
        curve = np.sqrt(c.lateral_accel_mps2 / np.maximum(k, 1e-6))
        speed = np.minimum(c.cruise_mps, curve)
        speed = np.minimum(speed, np.sqrt(2.0 * c.decel_mps2 * np.maximum(0.0, stop_s - arcs)))
        speed[arcs >= stop_s] = 0.0
        points, speeds = line[:end], speed[:end]
        if final_goal:
            goal_xy = (np.interp(stop_s, arcs, line[:, 0]), np.interp(stop_s, arcs, line[:, 1]))
            keep = arcs[:end] < stop_s - 0.05
            points = np.vstack((points[keep], goal_xy))
            speeds = np.append(speeds[keep], 0.0)
            if len(points) < 2:
                points = np.vstack((line[:1], goal_xy))
                speeds = np.array([0.0, 0.0])
        odometry = corridor.pose.to_world(points)
        samples = tuple(TrajectorySample(float(x), float(y), float(v)) for (x, y), v in zip(odometry, speeds))
        trajectory = LocalTrajectory(
            samples=samples,
            created_s=now_s,
            valid_until_s=now_s + c.validity_s,
            manoeuvre_id="goal_stop" if final_goal else "lane_follow",
            final_goal=final_goal,
        )
        return Plan(
            trajectory,
            reason,
            stop_s - s_car if math.isfinite(stop_s) else math.inf,
            float(curve.min()),
            state,
        )
