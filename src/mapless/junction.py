"""Commanded junction turns from the terrain grid and the sat-nav route (G1c).

The kerb corridor ends where the kerb does. At a junction the car needs a centreline of
its own: the entry lane up to the turn, an arc, and the exit lane. Each tick, from the
car's frame:

1. **Entry line.** The kerb corridor's lane centre at the car, continued straight.
2. **Exit line.** Its heading is the route's: the direction from the junction's exit point
   to the point after it, rotated into the odometry frame. Its lateral position is the
   exit road's right kerb, found in the grid with the corridor's own edge detector along
   a reference line through the route's exit point, plus ``keep_right``; until that kerb
   is seen, the route's exit point itself (GNSS-level, a few metres) stands in.
3. **Arc.** Tangent to both lines. Candidate radii in [``min_radius_m``,
   ``max_radius_m``] are tried nearest a nominal first (``nominal_fillet_m`` plus the
   lane's measured kerb distance), and the radius is latched while it
   stays feasible so the path does not jump between ticks. A candidate is feasible when
   the swept body footprint (plus ``margin_m``) crosses no BARRIER or OCCUPIED cell.
   None feasible: the caller stops before the junction. The car's steering lock sets the
   minimum (about 8.3 m here), so a right turn uses the junction box rather than
   hugging the corner.
4. **Verified space.** Unknown cells do not count as road: the plan says how far along
   the path the footprint lies entirely on *observed* road (``verified_m``). The caller
   holds the car at that point and lets it creep as more is seen, so the car never drives
   into space it has not observed (review of 697d53c).
5. The result is a ``Corridor`` (centreline every ``step_m`` along the path), so the
   planner, controller and obstacle check are the same as for lane following.

The turn is *commanded*: it begins only on a junction command from the route, and for a
right turn only once the right kerb is seen to end ahead (``kerb_opening``). The
route's exit point only selects the exit: the car does not drive past the junction
mouth until the exit road's right kerb has been seen (``exit_source == "kerb"``). It
hands back to the kerb corridor once the car's heading is within
``exit_tolerance_rad`` of the exit heading and the arc is behind it.
"""

import math
from dataclasses import dataclass

import numpy as np

from mapless.corridor import Corridor, CorridorEstimator
from mapless.ego_motion import Pose2D
from mapless.geometry import wrap
from mapless.terrain import CellLabel, ClassifiedSweep, TerrainGrid, VehicleBody


@dataclass(frozen=True)
class JunctionConfig:
    min_radius_m: float = 8.5  # the car's rear axle cannot turn tighter (wheelbase / tan(lock))
    max_radius_m: float = 16.0
    # Tried first: the kerb fillet plus the lane centre's distance from the kerb (keep_right,
    # measured at start-up), so a shoulder lane (4.4 m on Town10HD) gets the wider arc its
    # connecting lane has; 7.75 + 1.75 = 9.5 m on a plain lane.
    nominal_fillet_m: float = 7.75
    radius_step_m: float = 0.5
    latch_m: float = 1.0  # keep the last radius while it stays within this of the best
    refine_shift_m: float = 0.5  # a committed path follows a re-measured exit line that moved by more than this
    refine_persist_s: float = 0.5  # ... and has held (within refine_candidate_m of itself) for this long
    refine_candidate_m: float = 0.3
    refine_max_jump_m: float = 0.25  # a re-anchored path further than this from the car is rebuilt from the car's pose instead
    margin_m: float = 0.2  # around the body when choosing a turn
    follow_margin_m: float = 0.05  # around the body when re-verifying the latched turn (hysteresis)
    commit_clearance_m: float = 0.3  # a turn is entered only with this much clearance from kerbs
    commit_within_m: float = 3.0  # ... and once its tangent point is this close
    step_m: float = 1.0
    exit_search_near_m: float = 3.0  # exit kerb looked for from this far along the exit line ...
    exit_search_far_m: float = 25.0  # ... to this far
    exit_min_edges: int = 3  # stations with a kerb point ...
    exit_min_span_m: float = 2.0  # ... over at least this length (the direction is the prior's unless many more agree), agreeing ...
    # ... less this: three stations a metre apart span 2 m only nominally. Re-projected into a
    # probe frame that moves by centimetres tick to tick, they read 1.99-2.06 m, and a car at a
    # left turn's mouth, where only the exit kerb's first three stations are in view, lost its
    # measurement on the 1.99 and held there for good (PC Temp, Town03, 36b49ef and dd9344c).
    exit_span_tolerance_m: float = 0.25
    exit_max_mad_m: float = 0.4  # ... this well
    exit_use_fitted_heading: bool = False  # apply the fitted kerb direction to the exit heading (off: reported only)
    exit_near_band_m: float = 8.0  # the lateral position is read from the stations within this of the nearest
    exit_min_heading_deg: float = 1.5  # a fitted kerb direction is used only beyond this (range bias reads as skew) ...
    exit_max_heading_deg: float = 5.0  # ... and up to this far from the prior (a larger fit is two station clusters levering, not a skew)
    exit_max_distance_m: float = 7.5  # the kerb must be this close to the route's exit line (4.4 m shoulder + 3 m route error)
    exit_max_unseen_m: float = 3.0  # unseen strip allowed between the exit road and its kerb (ring spacing at 25-30 m)
    exit_kerb_edge: str = "raised"  # "raised": the first cell with fresh raised evidence beyond the road run (live Town10HD); "barrier": the first BARRIER cell; "midpoint": halfway between it and the road's last cell; "road": the road's last cell when the strip is wide
    exit_min_road_width_m: float = 0.5  # a run of road cells this wide counts as the exit road (far lanes are sampled by a ring or two)
    median_max_m: float = 2.5  # road again within this beyond a raised strip: the strip is a median, the kerb lies further out
    refine_max_shift_m: float = 1.5  # a re-measured exit line further than this from the latched is another road, not a correction
    memory_s: float = 2.0  # exit-kerb stations remembered for the line fit, for this long in motion (a standstill does not age them)
    road_memory_s: float = 10.0  # road cells count as observed for this long
    observed_share: float = 0.4  # of a pose's footprint cells seen as road: the pose is observed
    gap_m: float = 1.0  # unobserved poses allowed over this length ...
    gap_per_m: float = 0.3  # ... or this fraction of their distance (LiDAR rings spread out)
    near_cover_m: float = 8.0  # within this along the path (the stopping envelope) verified means covered across the body's width ...
    near_column_share: float = 0.8  # ... : this share of the footprint's lateral columns hold observed road, gaps allowed gap_m only
    after_turn_m: float = 15.0  # centreline continues this far past the arc
    exit_hazard_after_m: float = 2.0  # a hazard this far past the arc's end limits the verified distance, not the radius
    exit_tolerance_rad: float = math.radians(10)  # hand back to kerb following within this of the exit heading
    replan_max_heading_rad: float = math.radians(5)  # a blocked latched path is re-planned from the entry line while the car has turned less than this (beyond: mid-arc)
    opening_min_m: float = 3.0  # a right kerb gap at least this long is a junction mouth
    max_entry_offset_m: float = 10.0  # the route's entry point may be this far from the kerb opening (the opening is read ~4 m into the fillet)
    # Oncoming traffic on a left turn (full review of 73ae759, S5): the body may enter the
    # opposing lane (lane_width_m beside the car's own, which is lane_width_m wide) only once
    # its approach is seen clear for as long as the car takes to cross it (from here, at
    # gate_accel_mps2 up to the planner's cruise speed) plus oncoming_margin_s, at
    # oncoming_mps, as far as oncoming_view_m (the LiDAR's reach; the stack's tracks see
    # further). Fewer than oncoming_min_cells OCCUPIED cells hit within oncoming_age_s is clear.
    oncoming_check: bool = True
    lane_width_m: float = 3.5
    oncoming_mps: float = 8.5  # 30 km/h, town traffic
    oncoming_margin_s: float = 1.5
    oncoming_view_m: float = 40.0
    oncoming_min_cells: int = 2
    oncoming_age_s: float = 0.3
    gate_accel_mps2: float = 1.0
    gate_gap_m: float = 0.3  # the body stops this far short of the opposing lane

    def __post_init__(self):
        if not 0 < self.min_radius_m <= self.max_radius_m:
            raise ValueError("Need 0 < min_radius_m <= max_radius_m")
        if self.radius_step_m <= 0 or self.step_m <= 0 or self.margin_m < 0:
            raise ValueError("radius_step_m and step_m must be positive, margin_m non-negative")


@dataclass(frozen=True)
class TurnPlan:
    corridor: Corridor
    radius_m: float
    exit_source: str  # "kerb" (the exit lane is measured) or "route" (only selected)
    arc_start_m: float  # along the path from the rear axle
    arc_end_m: float
    exit_heading_rad: float  # in the vehicle frame
    verified_m: float  # along the path, how far the footprint lies on observed road
    exit_fit: dict | None = None
    clearance_m: float = math.inf  # the footprint's nearest approach to a kerb or obstacle cell


def kerb_opening(corridor, grid, min_length_m=3.0, probe_m=(0.3, -0.3, -0.9), max_age_s=3.0):
    """Distance ahead at which the right kerb ends for at least ``min_length_m``, or None.

    A station with a right edge followed by ``min_length_m`` of stations without one, where
    the grid shows observed ROAD at and beyond the kerb's expected offset (``probe_m``
    relative to the last kerb, positive towards the lane): the junction mouth. A kerb that
    runs out of the observed range, or stations whose kerb band is unseen, are unknown
    ground, not an opening (found on the fake-junction test: the car confirmed a "kerb
    opening" at the edge of its sight line and drove on).
    """
    if not corridor.ok:
        return None
    x, ref, right = corridor.station_x_m, corridor.reference_y_m, corridor.right_edge_m
    has, blocked = np.isfinite(right), corridor.blocked
    n = len(x)
    for i in range(n - 1):
        if x[i] <= 0 or not has[i] or has[i + 1] or blocked[i + 1]:
            continue
        kerb = float(np.median(right[max(0, i - 5) : i + 1][has[max(0, i - 5) : i + 1]]))
        j = i + 1
        while j < n and not has[j] and not blocked[j]:
            probes = np.array([[x[j], ref[j] + kerb + p] for p in probe_m])
            if not np.all(grid.labels(corridor.pose.to_world(probes), max_age_s) == CellLabel.ROAD):
                break
            j += 1
        if x[j - 1] - x[i] >= min_length_m:
            return float(x[i])
    return None


def _nearest_road_run(is_road, j0, merge_cells, min_cells):
    """(first column, last column) of the run of road cells nearest column ``j0`` (the one
    containing it when there is one), runs split by up to ``merge_cells`` unseen cells
    joined, at least ``min_cells`` wide; None when there is no such run."""
    cols = np.flatnonzero(is_road)
    if cols.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(cols) > merge_cells + 1)
    starts = np.concatenate(([cols[0]], cols[breaks + 1]))
    ends = np.concatenate((cols[breaks], [cols[-1]]))
    best = None
    for a, b in zip(starts, ends):
        if b - a + 1 < min_cells:
            continue
        distance = 0 if a <= j0 <= b else min(abs(a - j0), abs(b - j0))
        if best is None or distance < best[0]:
            best = (distance, int(a), int(b))
    return None if best is None else (best[1], best[2])


def _arc_path(y_entry, exit_point, exit_heading, radius, step, after_m):
    """Entry line y = y_entry (heading 0) -> arc of ``radius`` -> exit line; None if the
    tangent point lies behind the car. Returns (points, arc_start, arc_end)."""
    theta = wrap(exit_heading)
    if abs(theta) < 1e-3 or abs(abs(theta) - math.pi) < 1e-3:
        return None
    ex, ey = exit_point
    c, s = math.cos(theta), math.sin(theta)
    # Intersection of y = y_entry with the exit line through (ex, ey) along (c, s).
    if abs(s) < 1e-9:
        return None
    t = (y_entry - ey) / s
    px = ex + t * c
    tangent = radius * math.tan(abs(theta) / 2)
    x1 = px - tangent  # tangent point on the entry line
    if x1 < -1.5:
        return None  # the tangent point is behind the car: a tighter radius is needed
    # Just behind: the arc is entered where it passes the car (a few centimetres off the
    # entry line and a few degrees on, which the controller absorbs), its centre kept so the
    # exit straight lies on the exit line. Clamping the tangent point to the car instead
    # moved the whole arc forward and put the exit straight up to 1.5 m off the measured
    # line (full review of 73ae759: the exit offset).
    u0 = math.asin(min(1.0, -x1 / radius)) if x1 < 0.0 else 0.0
    start = max(x1, 0.0)
    # The entry straight runs up to the tangent point; when that is at the car (a mid-arc
    # re-plan solves the radius so it is, up to rounding) there is no straight: a point at
    # x = 1e-15 duplicated the arc's first point and gave it an arbitrary heading, which
    # the swept check then swung over the kerb (fresh-eyes review of 58adaf3).
    xs = np.arange(0.0, start - 1e-6, step) if start > 1e-6 else np.zeros(0)
    entry = np.column_stack((xs, np.full(len(xs), y_entry)))
    sign = 1.0 if theta > 0 else -1.0
    cx, cy = x1, y_entry + sign * radius
    sweep = abs(theta) - u0
    n_arc = max(2, int(math.ceil(sweep * radius / step)))
    angles = -sign * math.pi / 2 + sign * np.linspace(u0, abs(theta), n_arc + 1)
    arc = np.column_stack((cx + radius * np.cos(angles), cy + radius * np.sin(angles)))
    x2, y2 = arc[-1]
    n_out = int(after_m / step)
    out = np.column_stack((x2 + c * step * np.arange(1, n_out + 1), y2 + s * step * np.arange(1, n_out + 1)))
    points = np.vstack((entry, arc, out))
    return points, float(start), float(start + sweep * radius)


class _HazardField:
    """Distance (up to ``max_m``) to the nearest BARRIER/OCCUPIED cell around the car.

    On the grid's own cells over ``reach_m``, by repeated 8-neighbour dilation of the
    hazard mask: ring k holds cells first reached at step k, at about k cells' distance.
    Clearances beyond ``max_m`` are all the same to the turn's scoring.
    """

    def __init__(self, grid, pose, barrier_age_s, reach_m=30.0, max_m=2.0):
        self.cell = grid.config.cell_m
        self.origin = np.array([pose.x_m - reach_m, pose.y_m - reach_m])
        n = int(round(2 * reach_m / self.cell))
        xs = self.origin[0] + self.cell * (np.arange(n) + 0.5)
        ys = self.origin[1] + self.cell * (np.arange(n) + 0.5)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        labels = grid.labels(np.column_stack((gx.ravel(), gy.ravel())), max_age_s=barrier_age_s).reshape(n, n)
        hazard = (labels == CellLabel.BARRIER) | (labels == CellLabel.OCCUPIED)
        self.n = int(hazard.sum())
        steps = int(math.ceil(max_m / self.cell))
        self.distance = np.full((n, n), max_m)
        if self.n:
            reached = hazard.copy()
            self.distance[reached] = 0.0
            for k in range(1, steps + 1):
                grown = reached.copy()
                grown[1:, :] |= reached[:-1, :]
                grown[:-1, :] |= reached[1:, :]
                grown[:, 1:] |= reached[:, :-1]
                grown[:, :-1] |= reached[:, 1:]
                grown[1:, 1:] |= reached[:-1, :-1]
                grown[1:, :-1] |= reached[:-1, 1:]
                grown[:-1, 1:] |= reached[1:, :-1]
                grown[:-1, :-1] |= reached[1:, 1:]
                new = grown & ~reached
                self.distance[new] = k * self.cell
                reached = grown

    def query(self, xy):
        ij = np.floor((np.asarray(xy) - self.origin) / self.cell).astype(int)
        n = self.distance.shape[0]
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < n) & (ij[:, 1] >= 0) & (ij[:, 1] < n)
        out = np.full(len(ij), float(self.distance.max()))
        out[inside] = self.distance[ij[inside, 0], ij[inside, 1]]
        return out


def _swept_check(grid, pose, points, body, c, barrier_age_s, hazard_xy=None, margin_m=None, clearance_from_m=0.0, clearance_to_m=math.inf, road_memory_s=None, hazard_tolerated_beyond_m=None, tolerate_kerb_only=False):
    """(feasible, verified_m, clearance_m): no BARRIER/OCCUPIED under the body swept
    along ``points``, how far along the path the footprint lies on observed road, and the
    body's nearest approach to a hazard cell (``hazard_xy``, odometry frame), margin excluded.

    With ``hazard_tolerated_beyond_m`` a hazard first met further along the path than that
    does not make it infeasible: the path is verified only up to a metre short of it, so
    the car slows towards it and keeps measuring (fresh-eyes review of 58adaf3: a kerb cell
    under the exit straight 28 m out stopped the car dead at the junction mouth with 20 m
    of clear road ahead, and nothing at a standstill could ever release it).

    Observed means at least ``observed_share`` of the pose's footprint cells are ROAD
    (within ``road_memory_s``): LiDAR rings are metres apart at range, so a far pose is
    never fully covered. Unobserved poses are allowed over ``gap_m`` or ``gap_per_m``
    of their distance, as the corridor's support is.
    """
    if len(points) < 2:
        return False, 0.0, 0.0
    d = np.diff(points, axis=0)
    lengths = np.hypot(d[:, 0], d[:, 1])
    # A zero-length segment has no direction: take the next real one's (then the last's).
    headings = np.arctan2(d[:, 1], d[:, 0])
    real = lengths > 1e-6
    if real.any() and not real.all():
        idx = np.arange(len(headings))
        nxt = np.where(real, idx, len(headings))
        nxt = np.minimum.accumulate(nxt[::-1])[::-1]
        headings = np.where(real, headings, headings[np.minimum(nxt, idx[real][-1])])
    headings = np.append(headings, headings[-1])
    arcs = np.concatenate(([0.0], np.cumsum(lengths)))
    margin = c.margin_m if margin_m is None else margin_m
    # The planning margin applies over the arc (``clearance_from_m`` to ``clearance_to_m``);
    # the straights before and after are lane keeping, which runs with none: in a 3.5 m lane
    # the body has 0.7 m to the kerb, kerb cells reach 0.25 m into the road, and with the
    # margin on the exit lane too a lane placed 0.3 m off left no feasible radius.
    on_arc = (arcs >= clearance_from_m - 1e-6) & (arcs <= clearance_to_m + 1e-6)
    nu = int(round((body.front_m - body.rear_m + 2 * margin) / 0.25)) + 1
    nv = int(round((2 * body.half_width_m + 2 * margin) / 0.25)) + 1

    def footprint(m):
        us = np.linspace(body.rear_m - m, body.front_m + m, nu)
        vs = np.linspace(-body.half_width_m - m, body.half_width_m + m, nv)
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        return uu.ravel(), vv.ravel()

    # One footprint for the arc (planning margin), one for the straights (lane-keeping margin),
    # placed at every pose at once: a Python loop over the poses cost 70 ms a tick here.
    uu_arc, vv_arc = footprint(margin)
    uu_line, vv_line = footprint(min(margin, c.follow_margin_m))
    uu = np.where(on_arc[:, None], uu_arc[None, :], uu_line[None, :])
    vv = np.where(on_arc[:, None], vv_arc[None, :], vv_line[None, :])
    ch, sh = np.cos(headings)[:, None], np.sin(headings)[:, None]
    sx = points[:, 0][:, None] + ch * uu - sh * vv
    sy = points[:, 1][:, None] + sh * uu + ch * vv
    world = pose.to_world(np.column_stack((sx.ravel(), sy.ravel())))
    per_pose = nu * nv
    hazards = grid.labels(world, max_age_s=barrier_age_s).reshape(len(points), per_pose)
    hit = ((hazards == CellLabel.BARRIER) | (hazards == CellLabel.OCCUPIED)).any(axis=1)
    verified_cap = math.inf
    if hit.any():
        first_hazard_m = float(arcs[int(np.argmax(hit))])
        if hazard_tolerated_beyond_m is None or first_hazard_m <= hazard_tolerated_beyond_m:
            return False, 0.0, 0.0
        if tolerate_kerb_only and (hazards == CellLabel.OCCUPIED).any():
            return False, 0.0, 0.0  # something standing on the exit is waited for at the mouth, not tolerated
        verified_cap = max(0.0, first_hazard_m - 1.0)
    clearance = math.inf
    if hazard_xy is not None and hazard_xy.n:
        # Clearance is scored from ``clearance_from_m`` on (the arc and exit): the entry
        # straight runs beside the same kerb for every candidate and would tie them all.
        # The entry and exit straights run beside kerbs alike for every candidate.
        lo = int(np.searchsorted(arcs, clearance_from_m - 1e-6))
        hi = int(np.searchsorted(arcs, clearance_to_m + 1e-6))
        scored = world[lo * per_pose : max(hi, lo + 1) * per_pose] if lo < len(points) else world
        # The samples sit ``margin`` outside the body: report the body's own nearest approach
        # (in a 3.5 m lane the body passes a kerb at 0.7 m; with the margin counted twice a
        # correctly placed exit lane read 0.25-0.5 m and the commit test was a knife edge).
        clearance = float(hazard_xy.query(scored).min()) + margin
    road = grid.labels(world, max_age_s=c.road_memory_s if road_memory_s is None else road_memory_s).reshape(len(points), per_pose) == CellLabel.ROAD
    observed = road.mean(axis=1) >= c.observed_share
    # Sparse support is not verified free space (review of 6773e24: a footprint with ROAD on
    # one half and UNKNOWN on the other passed as observed at 40%). Within the stopping
    # envelope a pose counts only when observed road spans the body's width: a share of
    # the footprint's lateral columns (each the body's length long) holds a ROAD cell.
    # Beyond it the share rule stands, as far LiDAR rings never cover a footprint, so a
    # far pose is tentative support that the car grows into, never drives into blind.
    near = arcs <= c.near_cover_m
    columns = road.reshape(len(points), nu, nv).any(axis=1).mean(axis=1)
    observed = np.where(near, observed & (columns >= c.near_column_share), observed)
    observed[0] = True  # under the car: the LiDAR's blind ring
    verified = 0.0
    for x, seen, is_near in zip(arcs, observed, near):
        if seen:
            verified = float(x)
        elif x - verified > (c.gap_m if is_near else max(c.gap_m, c.gap_per_m * x)):
            break
    return True, min(verified, verified_cap), clearance


@dataclass(frozen=True)
class OncomingGate:
    """The opposing lane a left turn crosses, and whether its approach is clear (S5).

    ``frame`` is the entry lane (odometry pose on its centre, along it); in it the opposing
    lane is ``band`` (y range) and oncoming traffic moves towards -x. The body enters the
    band at ``entry_m`` and has left it at ``exit_m`` (rear axle, along the path from the
    car); while in it, the body spans ``zone_x`` along the lane. ``required_s`` is the time
    the car needs to be across, margin included; the approach is checked over ``checked_m``
    beyond the zone.
    """

    frame: Pose2D
    band: tuple
    zone_x: tuple
    entry_m: float
    exit_m: float
    required_s: float
    checked_m: float
    cells: int
    clear: bool
    note: str


def _time_to_cover(distance_m, speed_mps, accel_mps2, max_mps):
    """Seconds to drive ``distance_m`` from ``speed_mps``, accelerating at ``accel_mps2`` to ``max_mps``."""
    if distance_m <= 0:
        return 0.0
    v0 = min(max(speed_mps, 0.0), max_mps)
    ramp = (max_mps**2 - v0**2) / (2 * accel_mps2)
    if distance_m <= ramp:
        return (math.sqrt(v0**2 + 2 * accel_mps2 * distance_m) - v0) / accel_mps2
    return (max_mps - v0) / accel_mps2 + (distance_m - ramp) / max_mps


def body_corners(points, body):
    """(N, 4, 2) corners of the body at each pose of a rear-axle path (heading along the path)."""
    d = np.diff(points, axis=0)
    heading = np.arctan2(d[:, 1], d[:, 0])
    heading = np.append(heading, heading[-1] if len(heading) else 0.0)
    u = np.array([body.front_m, body.front_m, body.rear_m, body.rear_m])
    v = np.array([body.half_width_m, -body.half_width_m, -body.half_width_m, body.half_width_m])
    ch, sh = np.cos(heading)[:, None], np.sin(heading)[:, None]
    return np.stack((points[:, 0][:, None] + ch * u - sh * v, points[:, 1][:, None] + sh * u + ch * v), axis=-1)


def oncoming_gate(grid, pose, path_body, frame, body, c, speed_mps, cruise_mps, excused=()):
    """The opposing-lane gate for a left turn along ``path_body`` (rear axle, vehicle frame).

    ``excused``: footprints ((x, y) odometry centre, length, width, yaw) of things known not
    to be oncoming traffic (the stack's static or departing tracks), whose cells do not count.
    None when the path never enters the opposing lane.
    """
    pts = np.asarray(path_body, dtype=float)
    if len(pts) < 2:
        return None
    seg = np.hypot(*np.diff(pts, axis=0).T)
    arcs = np.concatenate(([0.0], np.cumsum(seg)))
    fine = np.arange(0.0, arcs[-1], 0.25)
    pts = np.column_stack((np.interp(fine, arcs, pts[:, 0]), np.interp(fine, arcs, pts[:, 1])))
    if len(pts) < 2:
        return None
    car = int(np.argmin(np.hypot(pts[:, 0], pts[:, 1])))
    s = fine - fine[car]
    in_frame = frame.to_body(pose.to_world(pts))
    corners = body_corners(in_frame, body)
    lo = c.lane_width_m / 2
    hi = lo + c.lane_width_m
    top, bottom = corners[:, :, 1].max(axis=1), corners[:, :, 1].min(axis=1)
    inside = (top > lo) & (bottom < hi)
    if not inside.any():
        return None
    first = int(np.argmax(inside))
    beyond = np.flatnonzero((np.arange(len(pts)) > first) & (bottom >= hi))
    last = int(beyond[0]) if beyond.size else len(pts) - 1
    zone = corners[first : last + 1][inside[first : last + 1]]
    x_lo, x_hi = float(zone[:, :, 0].min()), float(zone[:, :, 0].max())
    entry_m, exit_m = float(s[first]), float(s[last]) if beyond.size else math.inf
    distance = exit_m if math.isfinite(exit_m) else float(s[-1])  # from the car until the body has left the lane
    required = _time_to_cover(max(0.0, distance), speed_mps, c.gate_accel_mps2, cruise_mps) + c.oncoming_margin_s
    checked = min(c.oncoming_mps * required, c.oncoming_view_m)
    cell = grid.config.cell_m
    xs = np.arange(x_lo - 1.0, x_hi + checked, cell)
    ys = np.arange(lo + cell / 2, hi, cell)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    region = frame.to_world(np.column_stack((gx.ravel(), gy.ravel())))
    occupied = grid.labels(region, max_age_s=c.oncoming_age_s) == CellLabel.OCCUPIED
    for (cx, cy), length, width, yaw in excused:
        rel = region - np.array([cx, cy])
        along = np.abs(rel[:, 0] * math.cos(yaw) + rel[:, 1] * math.sin(yaw))
        across = np.abs(-rel[:, 0] * math.sin(yaw) + rel[:, 1] * math.cos(yaw))
        occupied &= ~((along <= length / 2 + 0.3) & (across <= width / 2 + 0.3))
    cells = int(np.unique(grid.cells(region[occupied])[0], axis=0).shape[0]) if occupied.any() else 0
    clear = cells < c.oncoming_min_cells
    if clear:
        note = f"oncoming lane clear for {checked:.0f} m ({checked / c.oncoming_mps:.1f} s at {c.oncoming_mps:g} m/s; crossing needs {required:.1f} s)"
    else:
        nearest = float(gx.ravel()[occupied].min()) - x_hi
        note = f"oncoming lane occupied {max(0.0, nearest):.0f} m from the crossing ({cells} cells)"
    return OncomingGate(frame, (lo, hi), (x_lo, x_hi), entry_m, exit_m, required, checked, cells, clear, note)


class JunctionTurn:
    def __init__(self, config, body, estimator):
        if not isinstance(body, VehicleBody) or not isinstance(estimator, CorridorEstimator):
            raise TypeError("JunctionTurn(JunctionConfig, VehicleBody, CorridorEstimator)")
        self.config = config or JunctionConfig()
        self.body = body
        self.estimator = estimator  # for keep_right and the grid ages
        self._radius = None
        self._kerb = []  # (time, moving clock, odometry xy of exit-kerb points found per tick)
        self._now = 0.0
        self._moving_s = 0.0  # time spent moving (standing_s 0), the exit kerb's memory's clock
        self._last_now = None
        # Time the car has stood still (set by the driver each tick, 0 while moving). At a
        # standstill the LiDAR's rings hit the same cells over and over and the cells between
        # them are never refreshed, so a freshness window measured from now would let the
        # observed road ahead "expire" while nothing changed (live run g1c_right: the
        # verified distance shrank from 35 m to 14 m in four seconds standing at the mouth,
        # and the car could not creep forward). Freshness is measured from the last motion.
        self.standing_s = 0.0
        self.still_credit_s = 0.0  # the last standstill's length, until the car has moved on (driver)
        self._committed = None  # (odometry points, arc end index, exit heading in odometry) once the turn has begun
        self._exit_odo = None  # the exit lane point and heading the latched path was built on, in the odometry frame
        self._probe_odo = None  # the route's exit point in the odometry frame: where the exit road is searched for
        self._entry = None  # (pose at commit, y_entry, radius): the entry frame the latched path was built in
        self._exit_fit = None  # the exit-line fit the latched path rests on; a refinement must read the kerb from nearer
        self.refine_note = None  # why the last refinement did or did not move the path (console diagnostics)
        self._refine_candidate = None  # (odometry exit point, first seen) of a re-measured line that differs from the latched
        self.last_measured_odo = None  # (exit lane point, kerb stations) in odometry from the last kerb measurement, for the drive log
        self.last_exit = None  # (exit point, exit heading, entry line y) in the vehicle frame of the last plan

    @property
    def barrier_age_s(self):
        return self.estimator.config.edge_max_age_s + self.standing_s

    @property
    def road_memory_s(self):
        # Observed road stays observed for the standstill's length after the car moves off
        # (``still_credit_s``): a car that waited at the mouth found the road ahead
        # "unseen" the moment it moved (S7). Kerbs do not get that credit: a kerb read from
        # 22 m while waiting, 0.9 m out, was kept fresh and the refinement never corrected it.
        return self.config.road_memory_s + max(self.standing_s, self.still_credit_s)

    def observe(self, sweep, pose, time_s):
        """Note the time; the exit search reads the terrain grid, which holds the sweeps."""
        if sweep is not None and not isinstance(sweep, ClassifiedSweep):
            raise TypeError("sweep must be a ClassifiedSweep or None")
        self._now = time_s

    def exit_line(self, grid, pose, exit_point_body, exit_heading_body, use_fitted_heading=False):
        """(point on the exit lane centre, source, heading, fit) in the vehicle frame.

        The exit road is looked for in the terrain grid along a probe line through the
        route's exit point in the exit heading: at each 1 m station the cell labels across
        the line (the corridor estimator's own profile and edge search) give the first kerb
        right of the line that has observed road right up to it. Cells beyond a longer
        unseen stretch are not an edge: at 25 m the LiDAR's rings are metres apart and a
        pavement top seen at a grazing angle reads as road, so raw sweep points placed the
        kerb 1-2 m off until the car was close. The grid accumulates the sweeps instead.
        The kerb stations are fitted as a line (scatter under ``exit_max_mad_m``), kept
        ``memory_s`` in the odometry frame. Source "kerb" when found, with the lane centre
        ``keep_right`` left of it; otherwise "route", which only selects the exit and never
        places the lane (review of 697d53c): the route's point can lie on the pavement.
        """
        c = self.config
        keep_right = self.estimator.keep_right_m
        if keep_right is None:
            return tuple(exit_point_body), "route", exit_heading_body, None
        probe = Pose2D(*pose.to_world(np.array([exit_point_body]))[0], pose.yaw_rad + exit_heading_body)
        stations = np.arange(c.exit_search_near_m, c.exit_search_far_m + 1e-9, 1.0)
        cc = self.estimator.config
        labels, offsets, j0, _ = self.estimator._profiles(grid, probe, stations, np.zeros_like(stations), self.barrier_age_s)
        raised = None
        if c.exit_kerb_edge == "raised":
            # Raised evidence along the same profile: the kerb face returns raised points in
            # the cell where the road's returns outvote them (the cell reads ROAD).
            px = stations[:, None] + offsets[None, :] * 0.0
            py = np.zeros_like(stations)[:, None] + offsets[None, :]
            raised = grid.raised_evidence(probe.to_world(np.column_stack((px.ravel(), py.ravel()))), self.barrier_age_s).reshape(px.shape)
        merge = int(round(0.5 / cc.lateral_step_m))  # road runs split by an unseen cell or two are one road
        min_run = int(round(c.exit_min_road_width_m / cc.lateral_step_m))
        median_cells = int(round(c.median_max_m / cc.lateral_step_m))
        found = []  # (station, kerb y) in the probe frame, this tick
        for i in range(len(stations)):
            # The exit road is the run of ROAD cells nearest the probe line, on either side of
            # it: with a 3 m route error the line can lie on the pavement beside the exit lane
            # (GNSS bias on seed 3), where a search outward from the line found a barrier and
            # no road and the kerb was never measured. Its right end, and the first raised
            # cell beyond, bracket the kerb.
            run = _nearest_road_run(labels[i] == CellLabel.ROAD, j0, merge, min_run)
            if run is None:
                continue
            right_end, _ = run
            jb = None
            for _ in range(3):
                if raised is not None:
                    # The first cell with fresh raised evidence right of (or at) the run's end.
                    lookback = int(round(0.5 / cc.lateral_step_m))
                    hits = np.flatnonzero(raised[i, : min(len(offsets), right_end + lookback + 1)] >= grid.config.barrier_min_evidence)
                else:
                    hits = np.flatnonzero(labels[i, :right_end] == CellLabel.BARRIER)
                if not len(hits):
                    jb = None
                    break
                jb = int(hits[-1])  # the first raised cell right of the run
                # Road again just beyond it: a median, not the kerb. The car keeps right, so
                # the exit carriageway is the outermost one (full review of 73ae759, S6: with
                # the route's point 4 m out, the run nearest it was the opposing carriageway
                # and the car drove 20 m the wrong way).
                lo = max(0, jb - median_cells)
                beyond = _nearest_road_run(labels[i, lo:jb] == CellLabel.ROAD, jb - lo, merge, min_run)
                if beyond is None:
                    break
                right_end = lo + beyond[0]
            if jb is None:
                continue
            strip = max(0.0, (right_end - jb) * cc.lateral_step_m)
            if strip > c.exit_max_unseen_m:
                continue
            # The kerb lies between the road's outer edge and the first raised cell. At 20 m
            # the LiDAR's rings are a metre apart and the strip between them up to 2 m wide;
            # in both geometries measured the strip lies on the pavement side (the kerb's
            # shadow seen from the road; sparse pavement returns seen end-on), so the road's
            # edge is taken when the strip is wide, else the midpoint.
            barrier_edge = float(offsets[jb] + cc.lateral_step_m / 2 - grid.config.cell_m / 2)
            road_edge = float(offsets[right_end] - grid.config.cell_m / 2)
            if c.exit_kerb_edge == "road" and strip > 2 * grid.config.cell_m:
                kerb = road_edge
            elif c.exit_kerb_edge in ("barrier", "raised"):
                kerb = barrier_edge
            else:
                kerb = 0.5 * (barrier_edge + road_edge)
            if abs(kerb) <= c.exit_max_distance_m:
                found.append((float(stations[i]), kerb))
        self.last_measured_odo = None
        if self._last_now is not None and self.standing_s <= 0.0:
            self._moving_s += max(0.0, self._now - self._last_now)
        self._last_now = self._now
        if found:
            self._kerb.append((self._now, self._moving_s, probe.to_world(np.array(found))))
        # Remembered for memory_s of motion, not of standing (as the grid's freshness is): at
        # a left turn's mouth the exit kerb's far stations are out of view, and a car that
        # stopped there forgot what it had measured on the approach and never measured it
        # again (PC Temp, Town03: held 25 s at the mouth, "exit lane not measured"). Once it
        # moves on, all of it ages as before and is replaced from nearer within memory_s.
        self._kerb = [(t, m, k) for t, m, k in self._kerb if self._moving_s - m <= c.memory_s]
        # The stations of the last memory_s first, as before; only if they make no line (a
        # car standing at the mouth with the far stations out of view) the ones kept from
        # before the standstill too. Read from farther away they carry more of the range
        # bias, so a car that has a line from where it stands keeps reading it from there
        # (Stage 3, S7: the standstill's view, 1.3 m, was right where the approach read 1.1).
        recent = [k for t, _, k in self._kerb if self._now - t <= c.memory_s]
        retained = [k for *_, k in self._kerb]
        for entries in [recent, retained] if len(retained) > len(recent) else [recent]:
            if not entries:
                continue
            remembered = probe.to_body(np.vstack(entries))
            stations, kerb_y = remembered[:, 0], remembered[:, 1]
            ok = np.isfinite(kerb_y)
            if ok.sum() >= c.exit_min_edges and np.ptp(stations[ok]) >= c.exit_min_span_m - c.exit_span_tolerance_m:
                x, y = np.asarray(stations[ok], dtype=float), np.asarray(kerb_y[ok], dtype=float)
                # The exit heading prior (a right angle to the entry lane's fitted direction) can be
                # a few degrees off: the kerb stations then run skewed across the probe frame, and a
                # median over them (centred ~14 m out) would absorb the skew into the lateral
                # position (0.95 m on a 2.4 degree error, read by the review of 1931070). So a line
                # is fitted when there are enough stations and its value at the exit point itself
                # anchors the lateral position whatever the skew; its direction corrects the
                # heading when it is clearly not the stations' range bias (>= exit_min_heading_deg).
                # The exit heading is the prior's (a right angle to the entry lane's direction,
                # read from the kerb beside the car, good to well under a degree). A line fitted
                # through the stations reads a slope anyway: the kerb estimate's bias grows with
                # range (far stations are sampled by a ring or two), and on the left turn of seed
                # 5308 a fitted -4 degrees sent the exit straight 1.6 m off a straight kerb. So
                # the fitted direction is only reported (``heading_fix_deg``) unless
                # ``exit_use_fitted_heading`` is set or ``use_fitted_heading`` asks for it (a repair
                # of a latched path blocked at its exit straight, where a prior a few degrees
                # off is the likeliest cause), and the lateral position comes from the
                # nearest stations (within ``exit_near_band_m`` of the nearest), where the
                # estimate is best and a residual skew costs least.
                slope, anchored = 0.0, None
                fitted_deg = 0.0
                if ok.sum() >= 8 and np.ptp(x) >= 10.0:
                    fitted_slope, fitted_intercept = np.polyfit(x, y, 1)
                    inliers = np.abs(y - (fitted_slope * x + fitted_intercept)) <= 0.6
                    if inliers.sum() >= c.exit_min_edges and not inliers.all():
                        fitted_slope, fitted_intercept = np.polyfit(x[inliers], y[inliers], 1)
                    fitted_deg = math.degrees(math.atan(fitted_slope))
                    if (c.exit_use_fitted_heading or use_fitted_heading) and c.exit_min_heading_deg <= abs(fitted_deg) <= c.exit_max_heading_deg:
                        slope, anchored = float(fitted_slope), float(fitted_intercept)
                residual = y - slope * x
                near = x <= x.min() + c.exit_near_band_m
                intercept = float(np.median(residual[near])) if anchored is None else anchored
                keep = np.abs(residual - intercept) <= 0.6
                if keep.sum() >= c.exit_min_edges:
                    if anchored is None:
                        intercept = float(np.median(residual[keep & near])) if (keep & near).sum() >= c.exit_min_edges else float(np.median(residual[keep]))
                    mad = float(np.median(np.abs(residual[keep] - intercept)))
                    distance = -intercept / math.hypot(1.0, slope)  # +: the kerb is right of the probe
                    if mad <= c.exit_max_mad_m and abs(distance) <= c.exit_max_distance_m:
                        lateral = intercept + keep_right * math.hypot(1.0, slope)
                        heading = exit_heading_body + math.atan(slope)
                        ch, sh = math.cos(exit_heading_body), math.sin(exit_heading_body)
                        point = (exit_point_body[0] - sh * lateral, exit_point_body[1] + ch * lateral)
                        kept_body = pose.to_body(probe.to_world(np.column_stack((x[keep], y[keep]))))
                        self.last_measured_odo = (pose.to_world(np.array([point]))[0], probe.to_world(np.column_stack((x[keep], y[keep]))))
                        fit = dict(
                            stations=int(keep.sum()),
                            mad=round(mad, 3),
                            kerb_from_probe_m=round(float(distance), 2),
                            heading_fix_deg=round(math.degrees(math.atan(slope)), 1),
                            fitted_skew_deg=round(fitted_deg, 1),
                            range_m=round(float(np.median(np.hypot(kept_body[:, 0], kept_body[:, 1]))), 1),  # the kerb stations' distance from the car
                        )
                        return point, "kerb", heading, fit
        return tuple(exit_point_body), "route", exit_heading_body, None

    def plan(self, grid, pose, corridor, exit_point_body, exit_heading_body, time_s):
        """A turn centreline as a Corridor, or None when no radius clears the terrain."""
        if not isinstance(grid, TerrainGrid) or not isinstance(pose, Pose2D) or not isinstance(corridor, Corridor):
            raise TypeError("plan(TerrainGrid, Pose2D, Corridor, exit point, exit heading, time)")
        c = self.config
        y_entry = corridor.lateral_offset_at(0.0) if corridor.ok else 0.0
        exit_point, source, exit_heading_body, fit = self.exit_line(grid, pose, exit_point_body, exit_heading_body)
        self.last_exit = (tuple(float(v) for v in exit_point), float(exit_heading_body), float(y_entry))
        radii = np.arange(c.min_radius_m, c.max_radius_m + 1e-9, c.radius_step_m)
        hazards = _HazardField(grid, pose, self.barrier_age_s)
        # The best feasible turn is the nearest to the nominal radius among those with the
        # commit clearance (the widest clearance swung 2.6 m into the exit road's opposing
        # lane: beyond what lane keeping has beside a kerb, more room buys nothing); when
        # none has it, the widest. So the radii are tried nearest the nominal first (the
        # latched one first of all) and the search stops at the first roomy one: one to three
        # swept checks a tick instead of sixteen (the full sweep cost 70 ms a tick).
        nominal = c.nominal_fillet_m + (self.estimator.keep_right_m or 1.75)
        order = sorted(radii, key=lambda r: (0 if self._radius is not None and abs(r - self._radius) < 1e-6 else 1, abs(r - nominal)))
        candidates = []
        for radius in order:
            built = _arc_path(y_entry, exit_point, exit_heading_body, float(radius), c.step_m, c.after_turn_m)
            if built is None:
                continue
            points, arc_start, arc_end = built
            # A kerb cell under the exit straight, past the arc, does not rule a radius out:
            # the exit line read from 15 m is a few decimetres off, the car is held short
            # of the cell (verified), and the line is refined as it closes. Ruled out, a car
            # standing at the mouth had no radius at all and never moved (Stage 3, S7).
            feasible, verified, clearance = _swept_check(
                grid, pose, points, self.body, c, self.barrier_age_s, hazards, clearance_from_m=arc_start, clearance_to_m=arc_end, road_memory_s=self.road_memory_s,
                hazard_tolerated_beyond_m=arc_end + c.exit_hazard_after_m, tolerate_kerb_only=True,
            )
            if feasible:
                candidates.append((float(radius), points, arc_start, arc_end, verified, clearance))
                if clearance >= c.commit_clearance_m:
                    break
        if not candidates:
            self._radius = None
            return None
        # The last radius is kept while it stays within latch_m of the best and nearly as
        # clear, so the path does not jump tick to tick.
        roomy = [cand for cand in candidates if cand[5] >= c.commit_clearance_m]
        if roomy:
            best = min(roomy, key=lambda cand: abs(cand[0] - nominal))
        else:
            best = max(candidates, key=lambda cand: (round(cand[5], 1), -abs(cand[0] - nominal)))
        if self._radius is not None:
            kept = [cand for cand in candidates if abs(cand[0] - self._radius) < 1e-6]
            if kept and abs(kept[0][0] - best[0]) <= c.latch_m and kept[0][5] >= best[5] - 0.3:
                best = kept[0]
        chosen = best
        radius, points, arc_start, arc_end, verified, clearance = chosen
        self._radius = radius
        if source == "kerb" and arc_start <= c.commit_within_m and clearance >= c.commit_clearance_m:
            # Near the tangent point, the exit measured and room to spare: commit the path.
            arc_index = int(np.searchsorted(np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T)))), arc_end - 1e-6))
            self._committed = (pose.to_world(points), arc_index, pose.yaw_rad + exit_heading_body)
            # The latched exit line is the measured one the path was built on, not the route
            # point (external review of 7b1e3c6: the refinement compared new measurements
            # against the route point, 2.7 m from the lane, so its first verdict was a
            # spurious "2.7 m off" and the re-anchor a no-op). The route point stays the
            # probe the exit road is searched around.
            self._exit_odo = (pose.to_world(np.array([exit_point]))[0], pose.yaw_rad + exit_heading_body)
            self._probe_odo = pose.to_world(np.array([exit_point_body]))[0]
            self._entry = (pose, y_entry, radius)  # the frame the path was built in, for a same-arc re-anchor
            self._exit_fit = fit
        n = len(points)
        length = float(arc_end + c.after_turn_m)
        turn = Corridor(
            time_s=time_s,
            pose=pose,
            centre_body=points,
            centre_odometry=pose.to_world(points),
            valid_length_m=length,
            coefficients=(0.0, float(y_entry)),
            station_x_m=points[:, 0],
            reference_y_m=points[:, 1],
            right_edge_m=np.full(n, np.nan),
            left_edge_m=np.full(n, np.nan),
            blocked=np.zeros(n, dtype=bool),
            evidence_body=np.zeros((0, 2)),
            evidence_time_s=time_s,
            support_m=length,
        )
        return TurnPlan(turn, radius, source, arc_start, arc_end, float(exit_heading_body), verified, fit, clearance)

    def latest_start_m(self):
        """How far along the entry line (rear axle, vehicle frame of the last ``plan``) the
        nominal arc (``nominal_fillet_m`` plus the lane's kerb distance) begins on its way to
        the last exit line, or None without one: from there the tighter radii lie ahead and
        the wider ones within the 1.5 m clamp.

        A car held at the junction mouth must not stand beyond it: at a 6 m kerb fillet the
        mouth (where the kerb ends) lies past the tangent point of the car's tightest arc,
        and a car that crept up to it could no longer turn (full review of 73ae759, S7)."""
        if self.last_exit is None:
            return None
        (ex, ey), heading, y_entry = self.last_exit
        theta = wrap(heading)
        if abs(math.sin(theta)) < 1e-3:
            return None
        px = ex + (y_entry - ey) / math.sin(theta) * math.cos(theta)
        nominal = self.config.nominal_fillet_m + (self.estimator.keep_right_m or 1.75)
        return float(px - nominal * math.tan(abs(theta) / 2))

    def reset(self):
        self._radius = None
        self.last_exit = None
        self._kerb = []
        self._committed = None
        self._exit_odo = None
        self._probe_odo = None
        self._entry = None
        self._exit_fit = None
        self._refine_candidate = None

    @property
    def committed(self):
        return self._committed is not None

    def exit_diagnostics(self):
        """The latched exit point and the last measured one (odometry frame) for the drive log."""
        out = {}
        if self._exit_odo is not None:
            out["turn_exit_odo"] = [round(float(v), 3) for v in self._exit_odo[0]]
            out["turn_exit_heading_odo"] = round(float(self._exit_odo[1]), 4)
        if self.last_measured_odo is not None:
            point, stations = self.last_measured_odo
            out["turn_exit_measured_odo"] = [round(float(v), 3) for v in point]
            # The remembered stations run to 800 pairs a tick (11 KB): the log keeps at most 60.
            keep = max(1, int(math.ceil(len(stations) / 60)))
            out["turn_exit_stations_odo"] = np.round(stations[::keep], 2).tolist()
            out["turn_exit_stations_n"] = int(len(stations))
        return out

    def exit_heading_body(self, pose):
        """The latched exit heading in the vehicle frame (None when not committed)."""
        return None if self._committed is None else wrap(self._committed[2] - pose.yaw_rad)

    def uncommit(self):
        """Drop the latched path but keep the exit-kerb memory and the latched radius: the
        caller re-plans from the current pose (allowed only while the car is still near the
        entry heading)."""
        self._committed = None

    def follow(self, grid, pose, time_s):
        """The committed turn from the current pose: re-verified, never re-planned.

        Once the car has entered the arc the path is latched in the odometry frame
        (review of 697d53c: a path re-planned from a turning car jumps and breaks down).
        Returns a TurnPlan, or None when the latched path no longer clears the terrain.
        """
        c = self.config
        points_odo, arc_end_index, exit_heading_odo = self._committed
        body = pose.to_body(points_odo)
        ahead = np.flatnonzero(body[:, 0] > -1.0)
        if ahead.size < 2:
            return None
        start = int(ahead[0])
        points = body[start:]
        hazards = _HazardField(grid, pose, self.barrier_age_s)
        feasible, verified, clearance = _swept_check(
            grid,
            pose,
            points,
            self.body,
            c,
            self.barrier_age_s,
            hazards,
            margin_m=c.follow_margin_m,
            road_memory_s=self.road_memory_s,
            hazard_tolerated_beyond_m=c.near_cover_m,
        )
        if not feasible:
            return None
        arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))))
        arc_end = float(arcs[max(0, min(arc_end_index - start, len(arcs) - 1))]) if arc_end_index >= start else -1.0
        n = len(points)
        turn = Corridor(
            time_s=time_s,
            pose=pose,
            centre_body=points,
            centre_odometry=points_odo[start:],
            valid_length_m=float(arcs[-1]),
            coefficients=(0.0, float(points[0, 1])),
            station_x_m=points[:, 0],
            reference_y_m=points[:, 1],
            right_edge_m=np.full(n, np.nan),
            left_edge_m=np.full(n, np.nan),
            blocked=np.zeros(n, dtype=bool),
            evidence_body=np.zeros((0, 2)),
            evidence_time_s=time_s,
            support_m=float(arcs[-1]),
        )
        heading_body = wrap(exit_heading_odo - pose.yaw_rad)
        return TurnPlan(turn, self._radius, "kerb", 0.0, arc_end, heading_body, verified, None, clearance)

    def replan_same_arc(self, grid, pose, time_s, min_shift_m=None):
        """Re-anchor the latched path on the exit line re-measured from here, keeping its
        radius and tangent point: the arc's end and the exit straight move with the exit
        lane estimate (a few decimetres, from closer by), the car's own heading does not
        change. For a block far along the path (the exit straight read against a kerb cell
        at 20 m) this is the right remedy; a block at the car's own position needs
        ``replan_mid_arc``. With ``min_shift_m`` it is the routine refinement while the car
        drives the arc: the exit kerb is read better from closer by (the first live turn
        committed on a reading from 30 m that was 1.1 m off, and nothing revisited it until
        the kerb corridor took over on the exit lane). Returns a TurnPlan and re-latches,
        or None."""
        if self._entry is None or self._exit_odo is None:
            return None
        c = self.config
        entry_pose, y_entry, radius = self._entry
        exit_odo, heading_odo = self._exit_odo
        exit_point, source, heading, fit = self.exit_line(grid, pose, pose.to_body(self._probe_odo[None])[0], wrap(heading_odo - pose.yaw_rad))
        if source != "kerb":
            self.refine_note = "exit kerb not read from here"
            return None
        exit_odo_new = pose.to_world(np.array([exit_point]))[0]
        heading_odo_new = pose.yaw_rad + heading
        if min_shift_m is not None:
            # Refinement, not a repair: only when the exit line has moved by more than the
            # latch width across its own direction (``range_m`` in the fit is reported, not
            # a condition: the stations' range barely changes as the car approaches, and a
            # range rule held a 1.15 m correction back for 3 s on the synthetic right turn).
            normal = np.array([-math.sin(heading_odo_new), math.cos(heading_odo_new)])
            shift = float((exit_odo - exit_odo_new) @ normal)
            if abs(shift) > c.refine_max_shift_m:
                # Further than a measurement can be off from close by: another carriageway
                # or road, not this exit lane (S6: a 4.6 m "refinement" onto the opposing
                # carriageway). The latched path stands; its own check stops the car if blocked.
                self._refine_candidate = None
                self.refine_note = f"exit line {shift:+.2f} m off: beyond {c.refine_max_shift_m:g} m, not taken"
                return None
            if abs(shift) < min_shift_m:
                # A reading hovering about the threshold (-0.41, -0.51, -0.41 ... on the
                # synthetic right turn) must not reset the candidate every other tick: it
                # is dropped only once the line reads clearly close to the latched.
                if abs(shift) < min_shift_m - c.refine_candidate_m:
                    self._refine_candidate = None
                self.refine_note = f"exit line within {shift:+.2f} m of the latched"
                return None
            # ... and only once the new reading has held for refine_persist_s: the kerb
            # estimate jumps by a metre from one tick to the next as a far station enters or
            # leaves the fit, and the path must not follow every jump.
            candidate = self._refine_candidate
            if candidate is not None and abs(float((candidate[0] - exit_odo_new) @ normal)) <= c.refine_candidate_m:
                held = self._now - candidate[1]
            else:
                self._refine_candidate, held = (exit_odo_new, self._now), 0.0
            if held < c.refine_persist_s:
                self.refine_note = f"exit line {shift:+.2f} m off for {held:.1f} s (read from {fit['range_m']} m)"
                return None
            self._refine_candidate = None
            self.refine_note = f"exit line {shift:+.2f} m off, read from {fit['range_m']} m"
        refining = min_shift_m is not None

        def fresh_arc(note):
            # The fallback for both modes (fresh-eyes review of 58adaf3: the repair had
            # no jump guard and the refinement skipped the fallback when no same-radius
            # arc existed): one arc from the car's own pose to the re-measured line.
            if refining:
                self.refine_note = self.refine_note.replace("re-anchored", note) if "re-anchored" in self.refine_note else self.refine_note + "; " + note
            mid = self._mid_arc(grid, pose, time_s, exit_point, heading, fit)
            if mid is None and refining:
                self.refine_note += " (not clear)"
            return mid

        if refining:
            self.refine_note += "; re-anchored"
        built = _arc_path(y_entry, entry_pose.to_body(exit_odo_new[None])[0], wrap(heading_odo_new - entry_pose.yaw_rad), radius, c.step_m, c.after_turn_m)
        if built is None:
            return fresh_arc("fresh arc from here")  # no same-radius arc reaches the moved line from the entry straight
        points_entry, arc_start, arc_end = built
        points_odo = entry_pose.to_world(points_entry)
        body = pose.to_body(points_odo)
        ahead = np.flatnonzero(body[:, 0] > -1.0)
        if ahead.size < 2:
            return fresh_arc("fresh arc from here")
        # Keeping the tangent point moves the arc's start along the entry straight: once the
        # car is in the arc the re-anchored path no longer passes under it (on the synthetic
        # left turns it jumped 0.8 m sideways and the car left the lane). Then the path is
        # a fresh arc from the car's own pose to the new exit line.
        nearest = int(np.argmin(np.abs(body[:, 0])))
        if abs(float(body[nearest, 1])) > c.refine_max_jump_m:
            return fresh_arc("fresh arc from here")
        hazards = _HazardField(grid, pose, self.barrier_age_s)
        feasible, _, _ = _swept_check(grid, pose, body[int(ahead[0]) :], self.body, c, self.barrier_age_s, hazards, margin_m=c.follow_margin_m, road_memory_s=self.road_memory_s)
        if not feasible:
            if refining:
                self.refine_note += " (not clear)"
            return None
        arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(points_entry, axis=0).T))))
        arc_index = int(np.searchsorted(arcs, arc_end - 1e-6))
        return self._relatch((points_odo, arc_index, heading_odo_new), (exit_odo_new, heading_odo_new), fit, self._radius, self._entry, grid, pose, time_s)

    def _relatch(self, committed, exit_odo, fit, radius, entry, grid, pose, time_s):
        """Latch a re-planned path and return it followed from here; if the follow rejects
        it (a path that passed its own swept check must not leave the car on a latch that
        refuses every tick after — fresh-eyes review of 58adaf3), the old latch stands."""
        kept = (self._committed, self._exit_odo, self._exit_fit, self._radius, self._entry)
        self._committed, self._exit_odo, self._exit_fit, self._radius, self._entry = committed, exit_odo, fit, radius, entry
        plan = self.follow(grid, pose, time_s)
        if plan is None:
            self._committed, self._exit_odo, self._exit_fit, self._radius, self._entry = kept
        return plan

    def replan_mid_arc(self, grid, pose, time_s):
        """A fresh latched path from the current pose when the latched one is blocked
        mid-turn: one arc tangent to the car's heading now and to the exit line, re-measured
        from here (the exit kerb is read better from close by). Returns a TurnPlan and
        re-latches, or None (then the caller stops along the old path).

        For a right turn the circle's centre lies at (0, -R) in the body frame and must be
        R from the exit line through e along u, i.e. ``(c - e) . n = -+R`` with n the line's
        normal: linear in R, one positive root.
        """
        if self._exit_odo is None:
            return None
        _, heading_odo = self._exit_odo
        probe = pose.to_body(self._probe_odo[None])[0]
        exit_point, source, heading, fit = self.exit_line(grid, pose, probe, wrap(heading_odo - pose.yaw_rad))
        if source != "kerb":
            return None
        plan = self._mid_arc(grid, pose, time_s, exit_point, heading, fit)
        if plan is None:
            # The prior heading failed: a prior a few degrees off is the likeliest cause of
            # a block at the exit straight (the stations' fitted skew says so), so the
            # fitted heading is tried second — second, because on straight kerbs the fit's
            # slope is range bias and a path on it failed where the prior's passed.
            exit_point, source, heading, fit = self.exit_line(grid, pose, probe, wrap(heading_odo - pose.yaw_rad), use_fitted_heading=True)
            if source == "kerb" and abs(float(fit["heading_fix_deg"])) > 0.0:
                plan = self._mid_arc(grid, pose, time_s, exit_point, heading, fit)
        return plan

    def _mid_arc(self, grid, pose, time_s, exit_point, heading, fit):
        c = self.config
        e = np.asarray(exit_point, dtype=float)
        theta = wrap(heading)
        if abs(theta) < math.radians(10):
            return None  # nearly aligned already: the straight hand-back handles it
        sign = 1.0 if theta > 0 else -1.0
        n = np.array([-math.sin(theta), math.cos(theta)])
        radius = None
        for root in (-1.0, 1.0):
            denominator = sign * n[1] - root
            if abs(denominator) < 1e-6:
                continue
            candidate = float(e @ n) / denominator
            if candidate >= c.min_radius_m and (radius is None or candidate < radius):
                radius = candidate
        if radius is None or radius > 4 * c.max_radius_m:
            return None
        built = _arc_path(0.0, e, theta, radius, c.step_m, c.after_turn_m)
        if built is None:
            return None
        points, arc_start, arc_end = built
        hazards = _HazardField(grid, pose, self.barrier_age_s)
        feasible, verified, clearance = _swept_check(
            grid, pose, points, self.body, c, self.barrier_age_s, hazards, margin_m=c.follow_margin_m, clearance_from_m=arc_start, clearance_to_m=arc_end, road_memory_s=self.road_memory_s
        )
        if not feasible:
            return None
        arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))))
        arc_index = int(np.searchsorted(arcs, arc_end - 1e-6))
        # The entry frame of this path is the car's pose now, with no entry straight: the
        # next refinement rebuilds from here, not from the original entry (review).
        return self._relatch((pose.to_world(points), arc_index, pose.yaw_rad + theta), (pose.to_world(np.array([exit_point]))[0], pose.yaw_rad + theta), fit, radius, (pose, 0.0, radius), grid, pose, time_s)
