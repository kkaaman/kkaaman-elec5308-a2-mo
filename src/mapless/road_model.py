"""The rolling local road model (plan v2.1, section 9). G2 framework.

``RoadModelBuilder`` keeps semantic evidence (road, lane marking, pavement, road user)
from painted LiDAR points on the same lattice as the ``TerrainGrid`` and follows its
recentring, and assembles a ``LocalRoadModel`` from the two each tick:

- ``occupancy`` / ``free`` / ``unknown``: from the terrain labels (BARRIER or OCCUPIED,
  ROAD, UNKNOWN) within the freshness window;
- ``ground_height``: not measured yet (zeros; the terrain classifier works on a local
  surface, section 9 item 4 is the place to add it);
- ``road`` / ``marking`` / ``pavement``: painted evidence, decayed with
  ``semantic_tau_s`` and cleared to unknown when stale (no 15 s memory: stale free space
  decays to unknown);
- ``age_s``: time since each cell was last hit.

Lane candidates are polylines of marking evidence in bands beside the car; stop-line
candidates are transverse marking runs ahead; exit candidates come from the junction
module and are passed through. Dynamic objects are kept off the model: a cell painted
``road_user`` adds nothing to the static layers.
"""

import math
from dataclasses import dataclass

import numpy as np

from mapless.contracts import LocalRoadModel
from mapless.ego_motion import Pose2D
from mapless.road_segmentation import ROAD_CLASSES
from mapless.terrain import CellLabel, TerrainGrid


@dataclass(frozen=True)
class RoadModelConfig:
    semantic_tau_s: float = 4.0  # painted evidence decays with this time constant
    semantic_max_age_s: float = 3.0  # older painted cells are unknown
    terrain_max_age_s: float = 3.0
    min_evidence: float = 1.0
    lane_band_m: float = 3.5  # marking bands searched this far either side of the lane centre
    lane_ahead_m: float = 30.0
    stop_line_min_width_m: float = 2.0  # a transverse marking run at least this wide ...
    stop_line_max_depth_m: float = 1.0  # ... and this thin along the lane

    def __post_init__(self):
        if self.semantic_tau_s <= 0 or self.semantic_max_age_s <= 0:
            raise ValueError("RoadModelConfig: positive time constants")


class RoadModelBuilder:
    def __init__(self, grid, config=None):
        if not isinstance(grid, TerrainGrid):
            raise TypeError("RoadModelBuilder(TerrainGrid)")
        self.grid = grid
        self.config = config or RoadModelConfig()
        n = grid.n
        self.origin = grid.origin.copy()
        self.evidence = {k: np.zeros((n, n), dtype=np.float32) for k in ("road", "marking", "pavement")}
        self.seen = np.full((n, n), -np.inf)
        self.time_s = None

    def reset(self):
        for v in self.evidence.values():
            v.fill(0.0)
        self.seen.fill(-np.inf)
        self.origin = self.grid.origin.copy()
        self.time_s = None

    def _follow_grid(self):
        """Shift the semantic layers when the terrain grid has recentred."""
        shift = np.round((self.grid.origin - self.origin) / self.grid.config.cell_m).astype(int)
        if not shift.any():
            return
        n = self.grid.n
        di, dj = shift
        src_i, dst_i = slice(max(0, di), min(n, n + di)), slice(max(0, -di), min(n, n - di))
        src_j, dst_j = slice(max(0, dj), min(n, n + dj)), slice(max(0, -dj), min(n, n - dj))
        for name, old in list(self.evidence.items()):
            new = np.zeros_like(old)
            new[dst_i, dst_j] = old[src_i, src_j]
            self.evidence[name] = new
        new_seen = np.full_like(self.seen, -np.inf)
        new_seen[dst_i, dst_j] = self.seen[src_i, src_j]
        self.seen = new_seen
        self.origin = self.grid.origin.copy()

    def add(self, points_ego, painted, pose, time_s):
        """Add painted LiDAR points (vehicle frame) seen from ``pose`` at ``time_s``."""
        if not isinstance(pose, Pose2D):
            raise TypeError("pose must be a Pose2D")
        c = self.config
        self._follow_grid()
        if self.time_s is not None:
            decay = math.exp(-max(0.0, time_s - self.time_s) / c.semantic_tau_s)
            for v in self.evidence.values():
                v *= decay
        self.time_s = time_s
        p = np.asarray(points_ego, dtype=float)
        if len(p) == 0:
            return
        xy = pose.to_world(p[:, :2])
        ij, inside = self.grid.cells(xy)
        flat = ij[:, 0] * self.grid.n + ij[:, 1]
        probs = painted.probabilities
        mask = inside & painted.painted
        for name in self.evidence:
            k = ROAD_CLASSES.index(name)
            weight = probs[:, k]
            take = mask & (weight > 0.2)
            np.add.at(self.evidence[name].ravel(), flat[take], weight[take])
        self.seen.ravel()[np.unique(flat[mask])] = time_s

    def build(self, pose, time_s, exit_candidates=()):
        """Assemble the LocalRoadModel for ``time_s`` (odometry frame)."""
        c, g = self.config, self.grid
        self._follow_grid()
        n = g.n
        ij = np.indices((n, n)).reshape(2, -1).T
        xy = g.origin + (ij + 0.5) * g.config.cell_m
        labels = g.labels(xy, max_age_s=c.terrain_max_age_s).reshape(n, n)
        occupancy = ((labels == CellLabel.BARRIER) | (labels == CellLabel.OCCUPIED)).astype(np.float32)
        free = (labels == CellLabel.ROAD).astype(np.float32)
        unknown = (labels == CellLabel.UNKNOWN).astype(np.float32)
        fresh = (time_s - self.seen) <= c.semantic_max_age_s
        layers = dict(
            occupancy=occupancy,
            free=free,
            unknown=unknown,
            ground_height=np.zeros((n, n), dtype=np.float32),
            age_s=np.where(np.isfinite(g.seen), time_s - g.seen, np.inf).astype(np.float32),
        )
        for name, evidence in self.evidence.items():
            layers[name] = np.where(fresh & (evidence >= c.min_evidence), evidence, 0.0).astype(np.float32)
        visibility = self._visibility(labels, pose)
        return LocalRoadModel(
            time_s=time_s,
            origin_xy=(float(g.origin[0]), float(g.origin[1])),
            cell_m=g.config.cell_m,
            layers=layers,
            lane_candidates=tuple(self.lane_candidates(layers["marking"], pose)),
            exit_candidates=tuple(exit_candidates),
            stop_lines=tuple(self.stop_lines(layers["marking"], pose)),
            visibility_m=visibility,
        )

    def _visibility(self, labels, pose):
        """How far ahead along the car's x axis the terrain is observed (road or occupied)."""
        g = self.grid
        xs = np.arange(0.0, g.config.size_m / 2, g.config.cell_m)
        body = np.column_stack((xs, np.zeros_like(xs)))
        ij, inside = g.cells(pose.to_world(body))
        far = 0.0
        for x, (i, j), ok in zip(xs, ij, inside):
            if not ok or labels[i, j] == CellLabel.UNKNOWN:
                # The LiDAR's blind ring under and beside the car, then gaps between far rings.
                if x > 8.0 and x - far > max(2.0, 0.3 * x):
                    break
                continue
            far = float(x)
        return far

    def _cells_in_body(self, layer, pose, x_range, y_range):
        """(x, y, value) in the vehicle frame of layer cells with evidence inside the ranges."""
        g = self.grid
        i, j = np.nonzero(layer > 0)
        if i.size == 0:
            return np.zeros((0, 3))
        xy = g.origin + (np.column_stack((i, j)) + 0.5) * g.config.cell_m
        body = pose.to_body(xy)
        keep = (body[:, 0] >= x_range[0]) & (body[:, 0] <= x_range[1]) & (body[:, 1] >= y_range[0]) & (body[:, 1] <= y_range[1])
        return np.column_stack((body[keep], layer[i[keep], j[keep]]))

    def lane_candidates(self, marking, pose):
        """Marking polylines beside the car: cells binned by lateral offset, each band with
        enough cells over enough length fitted as y(x) (quadratic). Returns (polyline, score)."""
        c = self.config
        cells = self._cells_in_body(marking, pose, (-5.0, c.lane_ahead_m), (-2 * c.lane_band_m, 2 * c.lane_band_m))
        out = []
        if len(cells) == 0:
            return out
        bins = np.floor(cells[:, 1] / 0.5).astype(int)
        for b in np.unique(bins):
            band = cells[np.abs(bins - b) <= 1]
            if len(band) < 8 or np.ptp(band[:, 0]) < 6.0:
                continue
            coefficients = np.polyfit(band[:, 0], band[:, 1], 2, w=np.sqrt(band[:, 2]))
            xs = np.arange(band[:, 0].min(), band[:, 0].max() + 1e-9, 1.0)
            line = np.column_stack((xs, np.polyval(coefficients, xs)))
            residual = np.abs(np.polyval(coefficients, band[:, 0]) - band[:, 1])
            score = float(len(band) / (1.0 + residual.mean()))
            if out and abs(out[-1][0][0, 1] - line[0, 1]) < 1.0:
                if score > out[-1][1]:
                    out[-1] = (line, score)
                continue
            out.append((line, score))
        return out

    def stop_lines(self, marking, pose):
        """Transverse marking runs ahead in the lane: (x, y_left, y_right, score)."""
        c = self.config
        cells = self._cells_in_body(marking, pose, (2.0, c.lane_ahead_m), (-c.lane_band_m, c.lane_band_m))
        out = []
        if len(cells) == 0:
            return out
        rows = np.floor(cells[:, 0] / 0.5).astype(int)
        for r in np.unique(rows):
            run = cells[np.abs(rows - r) <= int(c.stop_line_max_depth_m / 0.5)]
            width = float(np.ptp(run[:, 1])) if len(run) else 0.0
            if width >= c.stop_line_min_width_m and len(run) >= 4:
                out.append((float(run[:, 0].mean()), float(run[:, 1].max()), float(run[:, 1].min()), float(run[:, 2].sum())))
        return out
