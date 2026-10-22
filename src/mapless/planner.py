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
drop-off). The last obstacle stop point is kept in the odometry frame and still applies
until no obstacle has been seen for ``resume_clear_s``.
"""

import math
from dataclasses import dataclass

import numpy as np

from mapless.corridor import Corridor
from mapless.obstacles import PathObstacle, project_onto_polyline
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


def curvature(coefficients, x):
    d1 = np.polyval(np.polyder(coefficients), x)
    d2 = np.polyval(np.polyder(coefficients, 2), x) if len(coefficients) > 2 else np.zeros_like(x)
    return d2 / (1.0 + d1**2) ** 1.5


class LanePlanner:
    def __init__(self, config, body):
        if not isinstance(config, PlannerConfig) or not isinstance(body, VehicleBody):
            raise TypeError("LanePlanner(PlannerConfig, VehicleBody)")
        self.config = config
        self.body = body
        self._hold_point = None  # odometry xy of the rear-axle stop point
        self._last_seen_s = -math.inf

    def reset(self):
        self._hold_point = None
        self._last_seen_s = -math.inf

    def plan(self, corridor, obstacle, goal_body, now_s):
        """``goal_body``: the route goal in the vehicle frame, when it is within reach."""
        if not isinstance(corridor, Corridor):
            raise TypeError("corridor must be a Corridor")
        if obstacle is not None and not isinstance(obstacle, PathObstacle):
            raise TypeError("obstacle must be a PathObstacle or None")
        c = self.config
        if not corridor.ok:
            return Plan(None, "", math.inf, math.inf)
        line = corridor.centre_body
        arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))))
        s_car = float(project_onto_polyline(np.zeros((1, 2)), line)[0][0])

        stop_s, reason = math.inf, ""
        if obstacle is not None:
            stop_s = s_car + obstacle.distance_m - self.body.front_m - c.stop_gap_m
            reason = "obstacle"
            self._last_seen_s = now_s
            point = np.interp(stop_s, arcs, line[:, 0]), np.interp(stop_s, arcs, line[:, 1])
            self._hold_point = corridor.pose.to_world(np.array([point]))[0]
        elif self._hold_point is not None and now_s - self._last_seen_s <= c.resume_clear_s:
            held = corridor.pose.to_body(self._hold_point[None])
            stop_s = float(project_onto_polyline(held, line)[0][0])
            reason = "obstacle_hold"
        else:
            self._hold_point = None

        final_goal = False
        end = len(line)
        if goal_body is not None:
            goal_s, _, outside = project_onto_polyline(np.asarray(goal_body, dtype=float)[None], line)
            if not outside[0] and goal_s[0] < stop_s:
                stop_s, reason, final_goal = float(goal_s[0]), "goal", True
                end = int(np.searchsorted(arcs, stop_s)) + 1

        k = np.abs(curvature(np.array(corridor.coefficients), line[:, 0]))
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
        return Plan(trajectory, reason, stop_s - s_car if math.isfinite(stop_s) else math.inf, float(curve.min()))
