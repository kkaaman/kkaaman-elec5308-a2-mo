# Wayfinder v1 — E3-01, E3-02, E3-03

Wayfinder plans a shortest legal route between named destinations on the
exported Town10HD lane graph (BL-A-02), hands the route manager one continuous
lane sequence and polyline, and reports turn-by-turn guidance with an ETA. It
lives in `planning.wayfinder` and imports no simulator; the live HUD overlay
and the click-to-select destination remain later, adapter-side work.

## E3-01 A* on the exported lane graph

`planning.wayfinder.astar.shortest_path(graph, start_id, goal_id)` searches the
`autonomy.interfaces.LaneGraph` produced by `LaneTopology.graph`.

| Aspect | Rule |
|---|---|
| Nodes | Lane boundary nodes `road:section:lane:entry` / `:exit`, with XYZ |
| Edges | Only the directed edges in the graph: one polyline edge per lane section and one short edge per explicit legal successor |
| Cost | Planar polyline length of each edge in metres (`edge_length_m`) |
| Heuristic | Planar Euclidean distance from a node to the goal node. Every edge polyline is at least as long as its XY chord, so the heuristic never overestimates (admissible) and is consistent; closed-set A* is therefore optimal. Node elevation is carried but not costed, matching the planar controller |
| Tie-breaking | Frontier ordered by `(f, g, node_id)`; adjacency sorted by end node ID; a parent changes only on a strictly shorter tentative cost. Results do not depend on node/edge listing order |
| Unreachable | A `SearchResult` with `reachable=False`, empty path and `length_m = inf`. No exception, no `None`. Unknown node IDs raise `ValueError` |
| Lane changes | **Excluded.** The exporter adds no lateral links and the search never invents an edge, so a route can only continue along a lane or through a listed successor. Town10HD's 16-lane outer circuit (for example `0:0:-2`) is therefore reachable from the main component but cannot return; the registry's `outer_circuit` POI records this one-way property and the tests assert it |

Tests (`tests/unit/test_wayfinder_astar.py`) compare against a plain Dijkstra
reference on grids with one-way edges, a curved edge longer than its chord, a
disconnected island, start = goal, shuffled listings for tie-break
determinism, and 40 random digraphs.

## E3-02 Named destinations and continuous routes

`configs/wayfinder/pois_town10hd.json` is a `PoiRegistry`: map name, topology
SHA256 and named `PointOfInterest` records (`name`, `node_id`, `xyz_m`,
`description`). `PoiRegistry.validate(topology)` refuses a registry built for
another topology fingerprint, a node that does not exist, a node on a junction
lane, or a recorded position more than 5 cm from the node. Clicking a
destination on the mini-map is optional in the issue and is not implemented;
named POIs are the selection mechanism.

`check_destination(topology, origin, destination)` returns a `DestinationCheck`
with a reason code (`ok`, `unknown_node`, `junction_node`, `same_node`,
`unreachable`, `lane_loop`) without raising. `plan_route` / `plan_named_route` then build a
`WayfinderRoute`:

1. A* node path from origin to destination.
2. Lane sequence: the lanes touched by the node path, in order. A* visits no
   node twice, but a path from `X:exit` round a block to `X:entry` touches lane
   `X` at both ends; that is refused as `lane_loop`, so an accepted sequence is
   an open route with no repeated lane.
3. `planning.route_manager.stitch_route` stitches the lane-centre points and
   checks each transition is a listed successor with a boundary heading jump of
   at most 30°. A refusal surfaces as `RoutePlanningError("route_manager_refused")`.
4. The stitched polyline is trimmed to the origin and destination nodes and
   recorded as `segment_ids`, `point_slice` and `points_xyz_m`, the same form
   as `configs/routes/town10hd_baseline_v1.json`.
5. Manoeuvres: for each run of junction lanes, the heading change from the
   approach lane's end to the exit lane's start is classified with the
   BL-A-03 rule (`classify_exit`: |Δ| ≤ 25° straight, 45°–135° left,
   −45°–−135° right; anything else is `ambiguous_maneuver` and refused). Each
   manoeuvre carries its route position `route_s_m` and junction ID; the list
   ends with `arrive`.

The route is installed once per episode (`MissionEpisode` keeps
`route_install_count == 1`); manoeuvres are announcements along that single
polyline, never endpoints that reset the path.

`configs/wayfinder/fixture_pairs_town10hd.json` records five origin/destination
pairs with expected lane counts, lengths and manoeuvre lists, plus one
unreachable pair:

| Origin | Destination | Lanes | Length | Manoeuvres |
|---|---|---|---|---|
| baseline_start | baseline_goal | 6 | 175.7 m | straight, left, arrive |
| baseline_start | south_end | 23 | 798.7 m | straight, left, left, left, straight, right, straight, arrive |
| north_end | west_end | 13 | 367.4 m | left, straight, straight, right, arrive |
| south_end | northwest_block | 10 | 282.6 m | straight, right, left, arrive |
| west_end | baseline_goal | 11 | 267.9 m | right, straight, straight, left, arrive |
| outer_circuit | baseline_start | — | — | unreachable (no lane change) |

`tests/unit/test_wayfinder_route.py` asserts for every pair: consecutive lanes
are explicit successors, the polyline is one slice of the stitched route
manager geometry with no gap larger than one sample, manoeuvre positions
increase, endpoints are outside junctions, and
`validate_open_polyline` (the frozen-mission check) accepts the geometry. The
`baseline_start → baseline_goal` pair reproduces the frozen baseline
`segment_ids`, and the frozen route points are a slice of the Wayfinder
polyline. The frozen baseline mission itself is not changed.

## E3-03 Guidance, mini-map and ETA

`planning.wayfinder.guidance.RouteTracker.update(time_s, x_m, y_m, speed_mps)`
projects the rear-axle pose onto the route and returns a `GuidanceState`:

| Field | Meaning | Unit |
|---|---|---|
| `progress_m`, `remaining_m` | Arc length driven / left; progress is monotonic and searches forward from the last match, so a nearby earlier segment (the opposite lane) cannot pull it back | m |
| `off_route_m` | Lateral distance from the matched segment | m |
| `next_maneuver`, `next_maneuver_m` | First manoeuvre at or ahead of the progress, and its distance | m |
| `eta_state`, `eta_s` | `moving` with a time, `paused` with `None`, or `arrived` with 0 | s |
| `speed_mps` | Speed used for the ETA | m/s |

`SpeedPolicy` (defaults: cruise 4.0 m/s, paused threshold 0.1 m/s, arrival
radius 1.0 m, reference `measured`):

- **paused**: speed ≤ 0.1 m/s. The ETA is reported as paused; there is no
  division by a zero or near-zero speed. The test stops the vehicle mid-route
  at 0.0 and 0.05 m/s and asserts `eta_s is None` while progress is unchanged.
- **moving**: ETA = remaining ÷ min(speed, cruise) for `measured`, or
  remaining ÷ cruise for `cruise` (a planning ETA that ignores controller
  speed fluctuations). The divisor is never below the paused threshold.
- **arrived**: remaining arc ≤ arrival radius and lateral offset from the route ≤ arrival radius (level with the goal but beside it is not arrived).

Display helpers: `format_distance` gives `NNN m` below 1 km and `N.NN km`
above; `format_eta` gives `ETA m:ss`, `ETA paused` or `arrived`. Internal
units are always metres, metres per second and seconds.

`planning.wayfinder.minimap.render_minimap(route, state, yaw_rad, topology)`
draws, with OpenCV (an existing dependency), the map lanes in grey, the route
in blue, the driven part in grey, the ego pose and heading in yellow, the next
manoeuvre point in red, the goal in white, and a text panel with origin →
destination, next manoeuvre and distance, remaining distance, speed and ETA.
`+Y` (left) points up on the map. The result is an RGB uint8 array; `save_png`
writes it.

### Replay

```powershell
python scripts/wayfinder_replay.py --origin baseline_start --destination baseline_goal --output runs/wayfinder/baseline --frames 4
```

The script plans the named route, walks poses along it at the policy cruise
speed with a 0.3 m lateral offset and one scripted 4 s stop at 45 % of the
route, and writes `route.json`, `guidance_log.json` and `frame_NN.png`. Its
summary reports monotonic progress and the number of paused steps.

## Evidence

| Criterion | Evidence |
|---|---|
| E3-01 small-graph tests match Dijkstra, including unreachable destinations and one-way edges | `tests/unit/test_wayfinder_astar.py` |
| E3-02 five fixture pairs yield continuous legal paths without endpoint resets | `configs/wayfinder/fixture_pairs_town10hd.json`, `tests/unit/test_wayfinder_route.py` (`assert_continuous`, `MissionEpisode` install count) |
| E3-03 recorded replay shows route, current pose and next turn; stopped ETA shown as paused | `docs/evidence/E3-wayfinder/` ({download}`frame_01.png <evidence/E3-wayfinder/frame_01.png>`, `guidance_log.json`), `tests/unit/test_wayfinder_guidance.py` |
| Live HUD screenshots | Pending a CARLA session: overlay `render_minimap` on the run's camera frames or save frames beside the run log |

## Limits

- Planar costs and heuristic; elevation is recorded, not costed. Town10HD is flat.
- No lane changes, so some pairs are unreachable by design; callers must check
  `DestinationCheck.legal` rather than assume.
- The manoeuvre classifier is the BL-A-03 rule; heading changes between 25°
  and 45° or beyond 135° are refused rather than guessed.
- Guidance assumes the first pose is on or near the route and that the vehicle
  follows the route; a large `off_route_m` is reported, not acted on.
- The mini-map is a frame renderer, not a HUD integration; the live HUD overlay
  and click-to-select destination are outside this change.
