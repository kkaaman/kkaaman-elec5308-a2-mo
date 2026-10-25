# LiDAR obstacle-perception path — E2-01–04

Status: offline implementation with synthetic evidence. Live CARLA capture and
fusion replay on recorded sweeps remain to be run (see "What remains live"
below). Nothing here changes the RGB-D path,
the frozen baseline thresholds or any existing public symbol; the LiDAR path is
additive and opt-in through `autonomy.lidar_fusion.FusedObservationProvider`.

## Contract (E2-01)

`obstacle_perception.lidar.LidarSweep` is the only LiDAR object that enters the
core. No `carla` type, raw byte buffer or actor identity crosses the boundary
(`scripts/check_simulator_boundary.py --smoke` imports every core module with
simulator imports blocked).

| Field | Contract |
|---|---|
| `points_xyzi` | N×4 `float32`, finite; columns X forward, Y left, Z up in metres (rear-axle ego frame), intensity dimensionless in [0, 1]. N may be 0. Copied to a read-only snapshot. |
| `episode_id`, `frame_id`, `sim_time_s` | Same identity scheme as `FrameBundle`; simulation seconds reset per episode. |
| `received_monotonic_s` | Host clock seconds; never compared with simulation time. |
| `sensor_to_ego` | Rigid 4×4 sensor→rear-axle transform already applied once in the adapter; kept for provenance, azimuth coverage and the range check. |
| `config` | `LidarConfig`: `channels`, `range_m`, `rotation_frequency_hz`, `points_per_second`, `upper_fov_deg`, `lower_fov_deg`, `horizontal_fov_deg`. `expected_points_per_sweep = points_per_second / rotation_frequency_hz`; `full_sweep_per_tick(dt_s)` checks `rotation_frequency_hz × dt_s == 1`. |

Z = 0 in the ego frame is the nominal ground under the rear axle; the road
height is re-estimated per sweep (below), so this is an anchor, not an
assumption of flatness. Range is checked from the sensor origin, not the rear
axle. Density is `point_count / expected_points_per_sweep`.

Recommended live configuration (`carla_adapter.lidar.LidarRig` defaults): 32
channels, 50 m, 640 000 points/s, −30° to +10°, 360°, roof mount 2.4 m above the
actor origin, no simulated noise or drop-off so that the measured density
reflects geometry only. At `fixed_delta_seconds = 0.05` the rotation frequency
is 20 Hz and a sweep nominally holds 32 000 returns.

## Capture and transform verification (E2-02)

`carla_adapter.lidar.LidarCapture` spawns one `sensor.lidar.ray_cast` rigidly
attached to the ego with `rotation_frequency = 1 / fixed_delta_seconds`,
`sensor_tick = 0` and channels/range/points_per_second/FOV from the rig. The
callback decodes the float32 `(x, y, z, i)` buffer, reflects CARLA's +Y-right to
+Y-left, applies the sensor-to-rear-axle transform **once**, drops nonfinite or
out-of-range rows, clips intensity to [0, 1] and stores the sweep in a bounded
per-frame queue (`SweepQueue`). `read(frame)` returns a `LidarSweep` whose
timestamp the capture script checks against the tick snapshot, and appends a
record with frame ID, simulation time, point count, expected count, density
fraction, azimuth coverage and dropped rows. Timeouts call the owner's
emergency stop and never deliver another frame's sweep.

`known_object_check(sweep, center_ego_m, extent_m)` verifies a known box: at
least `min_points` returns inside the box grown by `search_margin_m`, the
nearest observed X face within `tolerance_m` (0.3 m) of the expected near face,
and every intensity in [0, 1]. `scripts/capture_lidar_sweeps.py` parks the ego on
a checked 60 m straight, places a target vehicle at 10/20/30 m, captures
`--frames` sweeps per distance, computes the target's expected rear-axle box from
its bounding vertices (evaluation-only truth) and writes `sweeps.json` plus
vehicle-frame `.npy` arrays, capture records, queue events, cleanup state and a
`capture_status` that is `COMPLETE` only when every known-object check passed.

Offline, the fake-CARLA test (`tests/unit/test_lidar_capture.py`) drives the
real `LidarCapture` with a synthetic sensor-frame buffer and checks the
blueprint attributes (`rotation_frequency = 20.0` at dt 0.05), the single
transform (recovered ego coordinates within 1e-4 m), the known-object check and
the density record. **The live known-object coordinates, frame IDs and scan
density are not yet recorded**; they are produced by the capture script.

## Ground removal, corridor crop and clustering (E2-03)

`obstacle_perception.lidar_obstacles.extract_obstacles(sweep, route, ...)`:

1. **Body and range removal.** Returns inside `EgoBody` (front 3.8 m, rear −1.0 m,
   half-width 1.05 m, height 1.6 m, margin 0.3 m) and beyond `max_range_m`
   (40 m planar) are dropped.
2. **Road profile.** Bands of `band_length_m` (2 m) along X are grown outward
   from the anchor band at Z = 0. A band's height is the median of its lowest
   returns (within `surface_depth_m` = 0.05 m of the third-lowest), taken under
   the ego's own path (`|y| < 1.5 m`) when at least three returns are seen
   there, otherwise within `|y| < 3.0 m`. It is accepted only if it continues
   the grade of the previous accepted bands within
   `0.07 m + 0.015 m/m × distance` (plus `max_grade` = 0.12 m/m × distance for
   the first band, whose grade is unknown) and lies within `max_band_gap_m`
   (4 m) of the last accepted band. Heights between accepted bands are linear;
   beyond them the last accepted grade is extrapolated.
3. **Classification** by height above the road: ROAD in [−0.4, 0.10) m, RAISED
   (kerb, pavement, sills) in [0.10, 0.30) m, OBSTACLE in [0.30, 2.4) m,
   ignored otherwise.
4. **Corridor crop.** Points are projected onto the shared
   `planning.corridor.RouteGeometry` (vectorised, same semantics as
   `RouteGeometry.project`), optionally through an `EgoRoutePose` when the route
   is not in the ego frame. A point is in the corridor when its distance to the
   route is ≤ `corridor_half_width_m` (1.5 m default) and its arc coordinate is
   in `(ego_s, ego_s + corridor_ahead_m]`.
5. **Bounded clustering.** OBSTACLE points are grouped by 8-connected 0.5 m
   occupancy cells (`cv2.connectedComponents`), clusters whose footprints come
   within `cluster_merge_m` (1.5 m) are merged, clusters below
   `min_cluster_points` (6) are noise, and at most `max_clusters` (32) nearest
   clusters are kept (`truncated` records the cut). RAISED points are clustered
   only inside the corridor with `raised_corridor_min_points` (12): a kerb beside
   the lane is the road edge, a kerb-high object in the lane is not driven over.
   A raised cluster under an obstacle cluster is merged into it so one object
   yields one track.
6. **Footprint and nearest surface.** Each `LidarCluster` carries its
   axis-aligned XY footprint, centre and extent, the point nearest the rear axle
   (`nearest_point_ego_m`, `nearest_surface_m` planar range) and, for in-corridor
   clusters, `corridor_gap_m`: the front-bumper gap from
   `RouteGeometry.gap(...)` over the cluster's corridor points, so the front
   offset is subtracted exactly once by the same code the RGB-D path uses.
7. **Seen-clear distance.** ROAD returns inside the corridor are binned every
   0.5 m from the bumper across three lateral strips; a bin counts when its
   centre strip and at least two strips hold two returns. `seen_clear_m` is the
   end of the last seen bin reached without an unseen stretch longer than
   `1.0 m + 0.3 × distance` (one sweep sees the road in rings), starting within
   8 m. `visible_distance_m = min(nearest gap, seen_clear_m)`.

### Validity rules

| Condition | Result |
|---|---|
| `point_count < min_points` (500) or density `< min_density_fraction` (0.1) | batch invalid, reason `sparse_sweep_points` / `sparse_sweep_density`; no clusters, no seen-clear distance |
| fewer than two accepted road bands | invalid, `road_not_established` |
| valid batch, no in-corridor cluster | `nearest_gap_m = None`; **not** clearance (`usable_for_clearance` is always False) |
| raised-only in-corridor cluster | reported with `raised=True`, class `lidar_raised` |

`LidarTrackFilter` confirms clusters N-of-M (reusing `TrackPolicy`: 2 of 3, 0.5 s
expiry, 5 gap samples). Association is greedy by footprint centre within
`gate_m + closing_speed_mps × elapsed` (1.5 m + 10 m/s × Δt) or by overlapping
footprints. Track IDs are integers scoped by `TrackIdentity` to the episode,
`source_id = "lidar"` and `model_version = "lidar-obstacles-v1"`, so they cannot
collide with RGB-D tracker IDs. Invalid sweeps count as misses for every track;
a track exposes `gap_m` only when confirmed, current, in the corridor and from a
valid sweep. `unknown_tracks` is true when a corridor-relevant track lacks a
valid current gap; tracks that only ever stood outside the corridor are ignored.

## Fusion into the common governor input (E2-04)

`obstacle_perception.fusion.fuse_corridor_reports(reports, required_sources)`
combines one `SourceCorridorReport` per source:

- **Nearer wins.** `obstacle_gap_m` is the minimum valid gap; `disagreement_m`
  records the spread, never an average. `visible_distance_m` is the minimum
  visibility claim.
- **Loss of a source is reported, never cleared.** Any required source that is
  missing or invalid makes the fused report invalid with reasons such as
  `lidar:stale_sweep`, `lidar:sparse_sweep_points`, `lidar:unconfirmed_tracks`
  or `front-rgbd:depth_unavailable`; the gap is withheld so the governor HOLDs.
- **No double counting.** Duplicate source IDs, identities not namespaced by
  their source and identities repeated across sources raise. One object seen by
  both sensors yields one gap (the nearer), not two.
- **Clearance needs every source.** `clearance_valid` requires every report to
  assert it. LiDAR asserts it only from positive road coverage
  (`seen_clear_m` reaching `clear_start_m`, 8 m by default, with no in-corridor
  cluster); a single clear bin or an empty sweep never does.

`autonomy.lidar_fusion.LidarCorridorSource` binds the physical LiDAR stream to
the `lidar` namespace and answers a report per frame (stale, missing or
wrong-episode sweeps are invalid). `FusedObservationProvider(rgbd_provider,
lidar_source)` is a drop-in `observation_provider` for the existing
`TrackedGovernancePolicy`: it keeps `observation.metadata == batch.metadata` and
returns a `CorridorObservation` carrying the fused gap, visibility and
clearance. The RGB-D temporal filter, governor, clearance guard and frozen
thresholds are untouched.

## Replay results — synthetic fixture

`tests/fixtures/lidar/` holds ten deterministic ray-cast sweeps
(`obstacle_perception.lidar_synthetic.build_fixture`, seed 7; 32 channels,
forward 120° sector, 6000 nominal returns per sweep, one sweep per 0.05 s tick,
540 KB): a car approaching from a 24 m bumper gap at 4 m/s with a parked car on
the pavement, a pole and a wall as roadside clutter, then a 95 % drop-out sweep
and an empty sweep. The manifest's `depth_gap_m` rows simulate the RGB-D channel
(a dropped frame, agreement, a far disagreement, a nearer disagreement). Expected
gaps are fixture inputs used only to score the replay. Rerun with:

```bash
PYTHONPATH=src python scripts/replay_lidar_obstacles.py --output docs/evidence/E2-01-04/replay-synthetic.json
```

| Frame | Note | Points / density | Road / raised / obstacle | In-corridor clusters | LiDAR gap (expected) | Seen clear | Depth gap | Fused | Nearest source |
|---:|---|---|---|---:|---|---:|---|---|---|
| 5 | depth dropped | 3954 / 0.66 | 1858 / 1458 / 574 | 1 | unconfirmed (24.0) | 23.0 | — | HOLD: `front-rgbd:depth_unavailable`, `lidar:unconfirmed_tracks` | — |
| 10 | agree | 3954 / 0.66 | 1858 / 1456 / 576 | 1 | 22.99 (23.0) | 23.0 | 23.9 | 22.99 | lidar |
| 15 | depth far | 3954 / 0.66 | 1858 / 1457 / 575 | 1 | 21.98 (22.0) | 18.0 | 26.0 | 21.98 | lidar |
| 20 | depth nearer | 3954 / 0.66 | 1851 / 1465 / 574 | 1 | 20.98 (21.0) | 18.0 | 20.5 | 20.5 | front-rgbd |
| 25 | agree | 3954 / 0.66 | 1851 / 1464 / 575 | 1 | 19.97 (20.0) | 18.0 | 19.9 | 19.9 | front-rgbd |
| 30 | agree | 3954 / 0.66 | 1847 / 1457 / 588 | 1 | 18.97 (19.0) | 18.0 | 19.0 | 18.97 | lidar |
| 35 | agree | 3954 / 0.66 | 1847 / 1458 / 587 | 1 | 17.98 (18.0) | 18.0 | 18.0 | 17.98 | lidar |
| 40 | agree | 3954 / 0.66 | 1838 / 1467 / 587 | 1 | 16.98 (17.0) | 15.0 | 17.0 | 16.98 | lidar |
| 45 | 95 % drop-out | 208 / 0.035 | — | 0 | invalid `sparse_sweep_points` | — | 17.0 | HOLD: `lidar:sparse_sweep_points` | — |
| 50 | empty sweep | 0 / 0.0 | — | 0 | invalid `sparse_sweep_points` | — | 17.0 | HOLD: `lidar:sparse_sweep_points` | — |

Summary ([JSON](evidence/E2-01-04/replay-synthetic.json)): 10 frames, 8 valid
sweeps, 7 fused valid, 3 HOLD; maximum |LiDAR gap − expected| = 0.035 m; maximum
depth/LiDAR disagreement 4.02 m resolved to the nearer source; 1 depth-loss
frame and 2 LiDAR-loss frames, none treated as clear; four roadside/target
clusters tracked under IDs 0–3 in the `lidar` namespace only. The three
roadside objects are reported but never produce a corridor gap. Seen-clear
distances of 15–23 m reflect the ring structure of a single sweep; the governor
receives `min(gap, seen_clear)` as visibility, which is conservative.

## Evidence

| Item | Location | Status |
|---|---|---|
| LiDAR interface fixture | `tests/fixtures/lidar/sweeps.json` + `sweep_*.npy` | checked in, synthetic |
| LiDAR dataset | `scripts/capture_lidar_sweeps.py` → `runs/<dir>/sweeps.json` | **NOT RUN** (needs CARLA) |
| LiDAR unit/replay tests | `tests/unit/test_lidar_contract.py`, `test_lidar_capture.py`, `test_lidar_obstacles.py`, `test_lidar_fusion.py`, `test_lidar_replay.py` | 55 tests pass offline |
| Fusion replay results | `docs/evidence/E2-01-04/replay-synthetic.json` | synthetic; recorded replay NOT RUN |
| Boundary check | `scripts/check_simulator_boundary.py --smoke` | 86 core modules imported with simulator imports blocked |

## What remains live

With CARLA 0.9.16 / Town10HD_Opt running and an otherwise empty world:

```bash
PYTHONPATH=src python scripts/capture_lidar_sweeps.py --host <carla-host> --output runs/lidar-capture-v1
PYTHONPATH=src python scripts/replay_lidar_obstacles.py --input runs/lidar-capture-v1/sweeps.json \
    --output runs/lidar-capture-v1/replay.json
```

Attach to the issue: the `sweeps.json` known-object checks (coordinates and
intensity column), the recorded frame IDs, simulation times and scan density
(`capture_records`), and the recorded replay summary. The recorded manifest has
no depth channel (`depth_valid = false`), so its replay reports every frame as
HOLD for `front-rgbd:depth_unavailable` while still scoring the LiDAR gaps; a
joint RGB-D + LiDAR capture is a follow-up. Thresholds in `LidarObstaclePolicy`
are proposed defaults validated on synthetic data only and are not frozen.

## Verification

Offline, 25 October 2026: `ruff check .` passed; `python
scripts/check_simulator_boundary.py --smoke` passed (86 core modules);
`python -m pytest` — 887 passed, 1 skipped, 3 subtests passed (55 new LiDAR
tests); `sphinx-build -W` passed. No live CARLA evidence is claimed.
