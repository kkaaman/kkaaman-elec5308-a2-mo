"""Junction-aware truth and gates for the G1c turn scenarios (review of 697d53c, point F).

The G1b lane-footprint gate cannot be transferred to a junction unchanged: the car must
leave the incoming lane corridor while turning. Here the truth is the *permitted area*:
the approach lane, the junction's connecting lane for the commanded exit, and the exit
lane, each as a centreline polyline with its own width (CARLA's connecting lanes are
wider than the lanes they join). A body corner is in the permitted area when its
distance to the nearest centreline segment is within that segment's half width plus a
tolerance; crossing into the pavement or the opposing lane is a corner outside it.

Gates (``score_turn``):
- ``turn_executed``: the driver entered its TURNING mode.
- ``junction_footprint``: no body corner left the permitted area, whole run.
- ``correct_exit``: the run ends on the commanded exit lane, within its half width.
- ``exit_lane_keeping``: lane offset on the exit leg (8 m past the junction on) p95
  within ``exit_p95_m``.
- ``approach_lane_keeping``: as G1b, on the approach up to the junction box.
The map is read by the evaluator only; nothing here is visible to the car.
"""

import math
from dataclasses import dataclass

import numpy as np

from mapless.obstacles import project_onto_polyline

SEGMENTS = ("approach", "junction", "exit")


@dataclass(frozen=True)
class TurnSite:
    command: str  # "LEFT" or "RIGHT"
    approach: np.ndarray  # N x 2 centreline, evaluator core frame, ending at the junction box edge
    junction: np.ndarray  # M x 2 connecting-lane centreline through the junction
    exit: np.ndarray  # K x 2 exit-lane centreline, starting where the junction lane ends
    approach_width_m: float
    junction_width_m: float
    exit_width_m: float
    junction_tolerance_m: float = 0.5  # extra half width allowed inside the junction

    def __post_init__(self):
        for name in ("approach", "junction", "exit"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
                raise ValueError(f"{name} must be an N x 2 polyline with at least two points")
            object.__setattr__(self, name, arr)
        if self.command not in ("LEFT", "RIGHT"):
            raise ValueError("command must be LEFT or RIGHT")

    extend_m: float = 10.0  # the exit lane is carried on this far so the body's front corners project onto it

    @property
    def path(self):
        """The truth path: approach, junction and exit joined (duplicate joints dropped), the
        exit carried ``extend_m`` beyond its last point along its final direction."""
        parts = [self.approach, self.junction[1:] if np.allclose(self.junction[0], self.approach[-1]) else self.junction]
        parts.append(self.exit[1:] if np.allclose(self.exit[0], parts[-1][-1]) else self.exit)
        direction = self.exit[-1] - self.exit[-2]
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        parts.append((self.exit[-1] + self.extend_m * direction)[None])
        return np.vstack(parts)

    @property
    def arcs(self):
        p = self.path
        return np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))))

    def segment_bounds(self):
        """Arc length at which the junction starts and ends along ``path``."""
        a = self.arcs
        n_app = len(self.approach)
        start = float(a[n_app - 1])
        n_junc = len(self.junction) - (1 if np.allclose(self.junction[0], self.approach[-1]) else 0)
        end = float(a[n_app - 1 + n_junc])
        return start, end

    wind_out_m: float = 5.0  # the junction tolerance carries this far into the exit lane (the turn winds out)

    def half_width_at(self, s):
        start, end = self.segment_bounds()
        s = np.asarray(s, dtype=float)
        half = np.where(
            s < start,
            self.approach_width_m / 2,
            np.where(s <= end, self.junction_width_m / 2 + self.junction_tolerance_m, self.exit_width_m / 2 + np.where(s <= end + self.wind_out_m, self.junction_tolerance_m, 0.0)),
        )
        return half

    def segment_at(self, s):
        start, end = self.segment_bounds()
        s = np.asarray(s, dtype=float)
        return np.where(s < start, "approach", np.where(s <= end, "junction", "exit"))


def heading_at(path, arcs, s):
    i = int(np.clip(np.searchsorted(arcs, float(s)) - 1, 0, len(path) - 2))
    d = path[i + 1] - path[i]
    return math.atan2(d[1], d[0])


def truth_sample(site, rear_axle_xy, yaw_rad, body):
    """Truth for one tick: arc length, lane offset, segment, heading error and corner excess."""
    path, arcs = site.path, site.arcs
    xy = np.asarray(rear_axle_xy, dtype=float)
    s, lateral, _ = project_onto_polyline(xy[None], path)
    c, sn = math.cos(yaw_rad), math.sin(yaw_rad)
    corners = np.array(
        [
            (xy[0] + c * u - sn * v, xy[1] + sn * u + c * v)
            for u, v in ((body.front_m, body.half_width_m), (body.front_m, -body.half_width_m), (body.rear_m, -body.half_width_m), (body.rear_m, body.half_width_m))
        ]
    )
    cs, c_lateral, _ = project_onto_polyline(corners, path)
    excess = np.abs(c_lateral) - site.half_width_at(cs)
    return dict(
        s_m=float(s[0]),
        lane_offset_m=float(lateral[0]),
        segment=str(site.segment_at(s[0])),
        heading_error_rad=(yaw_rad - heading_at(path, arcs, s[0]) + math.pi) % (2 * math.pi) - math.pi,
        corner_excess_m=float(excess.max()),
    )


def score_turn(log, site, modes_turning, exit_p95_m=0.35, approach_p95_m=0.35, approach_max_m=0.7, end="finished"):
    """Gate rows (name, passed, values) from per-tick rows holding ``truth`` as ``truth_sample`` returns."""
    start, junction_end = site.segment_bounds()
    rows = []
    rows.append(("turn_executed", bool(modes_turning), dict(turning_ticks=int(modes_turning))))
    excess = max(r["truth"]["corner_excess_m"] for r in log) if log else math.inf
    rows.append(("junction_footprint", excess <= 0.0, dict(max_corner_excess_m=excess)))
    last = log[-1]["truth"] if log else None
    on_exit = last is not None and last["segment"] == "exit" and abs(last["lane_offset_m"]) <= site.exit_width_m / 2
    rows.append(("correct_exit", bool(on_exit) and end == "finished", dict(end=end, segment=None if last is None else last["segment"], offset_m=None if last is None else last["lane_offset_m"])))
    exit_leg = [r for r in log if r["truth"]["segment"] == "exit" and r["truth"]["s_m"] > junction_end + 8.0 and r["truth"]["speed_mps"] > 0.3]
    if len(exit_leg) >= 40:
        off = np.abs([r["truth"]["lane_offset_m"] for r in exit_leg])
        p95 = float(np.percentile(off, 95))
        rows.append(("exit_lane_keeping", p95 <= exit_p95_m, dict(p95_m=p95, max_m=float(off.max()), ticks=len(exit_leg))))
    else:
        rows.append(("exit_lane_keeping", None, dict(ticks=len(exit_leg))))
    approach = [r for r in log if r["truth"]["segment"] == "approach" and r["truth"]["s_m"] > 15.0 and r["truth"]["s_m"] < start - 5.0 and r["truth"]["speed_mps"] > 0.3]
    if len(approach) >= 40:
        off = np.abs([r["truth"]["lane_offset_m"] for r in approach])
        p95, worst = float(np.percentile(off, 95)), float(off.max())
        rows.append(("approach_lane_keeping", p95 <= approach_p95_m and worst <= approach_max_m, dict(p95_m=p95, max_m=worst, ticks=len(approach))))
    else:
        rows.append(("approach_lane_keeping", None, dict(ticks=len(approach))))
    return rows
