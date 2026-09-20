# Autonomous Perception and Navigation Library (ELEC5308 Assignment 1)

A simulator-agnostic Python library providing lane/driving-area detection,
object detection & tracking, semantic segmentation, traffic condition
detection, and basic path-following/steering/speed control — demonstrated
live in CARLA.

**Team members:** *(add your names here)*

## Repository layout

```
.
├── pyproject.toml            # dependency + package spec
├── setup_environment.py      # one-shot installer for the CARLA demos (see below)
├── lane_unet_best.pth        # trained lane-detection weights
├── segmentation_model.pth    # trained segmentation weights
├── training/                 # model training scripts (not needed to run the demos)
├── tests/                    # automated test suite
├── examples/, docs/          # additional usage examples and docs
└── src/
    ├── perception/            # lane detection + ego-path extraction
    ├── segmentation/          # BiSeNetV2 semantic segmentation
    ├── object_detection/      # YOLO-based object detection/tracking
    ├── traffic_perception/    # traffic light/sign detection + distance estimation
    ├── navigation_control/    # Pure-Pursuit + PID path-following controller
    ├── carla_adapter/         # the only files that import `carla` directly
    ├── demo_combined_perception.py   # perception showcase (autopilot-driven)
    ├── demo_fused_lane_following.py  # lane detection actually drives the car
    ├── overlay_enhancements.py       # display-only smoothing/HUD helpers
    ├── turn_decision.py              # junction detection + turn selection
    ├── traffic_light_color.py        # HSV traffic-light color classifier
    ├── ground_projection.py          # image-pixel -> real-world ground projection
    └── lane_to_waypoints.py          # perception-to-control fusion point
```

## 1. Base install (library + tests)

This is the minimum needed to install the library and run its automated
test suite — no simulator required.

Create and activate a virtual environment, then install the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

Run the complete automated test suite from the repository root:

```bash
python -m pytest tests -v
```

## 2. Running the live CARLA demos (optional, builds on the base install)

The steps above are enough to use the library and run its tests. To also
run the two live CARLA demos, a few additional things are needed on top.

### Assumptions

- **CARLA is already installed** (server running via `CarlaUE4.exe`,
  version matching what you'll install the Python client for below).
- **The correct Python version is already installed** (this project was
  built and tested against Python 3.12).

### Additional install step

From the repository root, with your Python 3.12 environment active
(the same one from step 1 above):

```powershell
python setup_environment.py --carla-version <your CARLA version, e.g. 0.9.16>
```

This installs torch/torchvision (CUDA build if you have an NVIDIA GPU),
the `carla` Python client (from a local wheel if found, matching your
CARLA install, or falls back to PyPI), and the rest of this project's
dependencies from `pyproject.toml` (numpy, opencv-python, ultralytics,
pandas, pytest) in one pass. It's idempotent — safe to re-run if a step
fails partway through.

If CARLA's Python client is *already* installed and working, skip that
step:

```powershell
python setup_environment.py --skip-carla
```

### Run a demo

With `CarlaUE4.exe` already running, from `src/`:

```powershell
cd src
python demo_combined_perception.py --town Town01
```

Toggleable perception overlay (segmentation/objects/traffic/lane), driven
by CARLA's own autopilot. Keys: `1`/`2`/`3`/`4` toggle each layer, `0`
toggles all, `q` quits.

```powershell
python demo_fused_lane_following.py --town Town01
```

Lane detection actually drives the vehicle, via a Pure-Pursuit + PID
controller — no autopilot involved. Same toggle keys for the display
overlay; driving itself only ever depends on the lane layer, regardless of
what's toggled on screen.

`--town` is optional on both — omit it to use whatever map `CarlaUE4.exe`
already has loaded. `Town01` is a good default: simple, single-lane-per-
direction roads that avoid a known limitation in the current lane-
boundary-selection logic on wider, multi-lane maps (e.g. Town10HD).

## `demo_fused_lane_following.py`'s driving behaviour

Beyond ordinary lane-following, this demo also:

- **Turns at junctions.** Pure lane-following can't decide to turn (a
  turn's destination road usually isn't visible until already partway
  through it) — so when a junction is detected ahead (using CARLA's own
  map waypoint graph, see `turn_decision.py`), control briefly switches to
  a short, fixed map-based route through that junction, then hands back to
  lane-following. Priority: **straight always wins if available**;
  otherwise alternates right, then left, then right, ... This is a
  hardcoded rule, not navigation — there's no destination, and no
  awareness of the road beyond the current junction.
- **Stops for red traffic lights** (see `traffic_light_color.py`) — a
  simple HSV classifier on the detected "traffic light" box, using box
  *height* as a proximity proxy (not a real metric distance — traffic
  lights are elevated off the road plane, so this project's ground-plane
  projection technique doesn't apply to them, and no established real-world
  width exists for them either). Red only, not yellow (yellow previously
  mis-fired on nearby signage sharing the detected box's crop).

## Known limitations

- No obstacle/collision avoidance and no destination/route planning —
  turning is a per-junction, hardcoded rule (see above), not navigation.
  No awareness of other agents when choosing or executing a turn, and a
  red-light stop isn't specially suppressed mid-turn.
- The ego-lane boundary-selection heuristic (`src/perception/lane_path.py`)
  assumes exactly one marking line per side of the lane. Wide, multi-lane
  roads (3+ parallel lines — a double-yellow centreline, one or more
  internal dividers, a far edge line) can cause it to lock onto the wrong
  line. Confirmed on Town10HD; not fixed as of this release — use a
  simple single-lane-per-direction town for a reliable demo.
- The junction-turning and red-light logic were verified via isolated unit
  tests against synthetic/fake data (not a live CARLA run) at the time
  this was last assembled — treat the first real run as the actual test,
  same as any new feature in this project.
