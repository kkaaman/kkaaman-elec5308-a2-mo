# Autonomous Perception and Navigation Library

A simulator-agnostic Python library providing lane/driving-area detection,
object detection and tracking, semantic segmentation, traffic-condition
detection, and basic path-following, steering and speed control, demonstrated
live in CARLA.

**Team members:**

- Saviour Kwofie
- Alan Tran
- Mohammad Judeh
- Zhaolin Huang
- Abdullah Almutairi
- Ricky Purani

## Repository layout

```text
.
├── pyproject.toml            # dependency + package specification
├── setup_environment.py      # one-shot installer for CARLA demo dependencies
├── lane_unet_best.pth        # trained lane-detection weights
├── segmentation_model.pth    # trained segmentation weights
├── training/                 # model training scripts
├── tests/                    # automated test suite
├── examples/                 # additional usage examples
├── docs/                     # Sphinx documentation and worked examples
├── .github/
│   └── workflows/
│       └── ci.yml            # lint, test and documentation CI
└── src/
    ├── perception/            # object detection/tracking + lane processing
    ├── segmentation/          # BiSeNetV2 semantic segmentation
    ├── traffic_perception/    # traffic light/sign detection + distance estimation
    ├── navigation_control/    # Pure Pursuit + PID path-following controller
    ├── carla_adapter/         # CARLA-specific adapter layer
    ├── demo_combined_perception.py   # perception showcase
    ├── demo_fused_lane_following.py  # perception-driven lane following
    ├── overlay_enhancements.py       # display smoothing and HUD helpers
    ├── turn_decision.py              # junction detection + turn selection
    ├── traffic_light_color.py        # HSV traffic-light colour classifier
    ├── ground_projection.py          # image-pixel -> ground-plane projection
    └── lane_to_waypoints.py          # perception-to-control fusion
```

## 1. Base install

This is the minimum needed to install the library and run the automated
tests. A running simulator is not required.

Python 3.10 or later is supported. The CARLA demonstrations were developed
and tested using Python 3.12.

### Linux / WSL

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,docs]"
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,docs]"
```

## 2. Quick start

After installation, the public modules can be imported directly:

```python
from perception import ObjectDetector, PerceptionConfig
from traffic_perception import TrafficPerception
from navigation_control import PathFollower
```

Example object-detector setup:

```python
from perception import ObjectDetector, PerceptionConfig

config = PerceptionConfig(
    model_name="yolo26n.pt",
    confidence_threshold=0.25,
)

detector = ObjectDetector(config)
detections = detector.detect(image)

for detection in detections:
    print(detection)
```

The detector expects an RGB NumPy image.

## 3. Automated tests

Run the complete automated test suite from the repository root:

```bash
python -m pytest
```

For verbose output:

```bash
python -m pytest tests -v
```

The current integrated version contains 106 automated tests covering areas
including:

- object detection and tracking
- lane-path extraction
- traffic perception
- navigation and control
- CARLA adapter behaviour

A running CARLA simulator is not required for the unit tests.

## 4. Linting

The project uses Ruff for linting and import checking.

From the repository root:

```bash
ruff check .
```

## 5. Documentation

The project uses Sphinx to generate HTML documentation from source
docstrings.

The documentation includes:

- a Getting Started guide
- automatically generated API reference
- worked usage examples
- CARLA demonstration instructions

Build the documentation from the repository root:

```bash
sphinx-build -W docs docs/_build/html
```

The generated documentation can then be opened at:

```text
docs/_build/html/index.html
```

## 6. Continuous integration

GitHub Actions automatically runs:

1. Ruff linting
2. pytest automated tests
3. the Sphinx documentation build

The workflow is defined in:

```text
.github/workflows/ci.yml
```

## 7. Running the live CARLA demos

The base installation above is sufficient for the library, tests and
documentation. Running the live demos additionally requires CARLA.

### Assumptions

- CARLA is already installed.
- The CARLA server can be started with `CarlaUE4.exe`.
- The Python client version should match the CARLA simulator version.
- The project was tested with CARLA 0.9.16 and Python 3.12.

### Additional install step

From the repository root, with the Python environment active:

```powershell
python setup_environment.py --carla-version 0.9.16
```

The setup script installs the dependencies needed by the live demo,
including PyTorch, torchvision and the CARLA Python client.

If the CARLA Python client is already installed and working:

```powershell
python setup_environment.py --skip-carla
```

### Start CARLA

Start `CarlaUE4.exe` before launching either demo.

### Combined perception demo

From the repository root:

```powershell
cd src
python demo_combined_perception.py --town Town10HD
```

This demo runs the perception modules while CARLA autopilot controls the
vehicle.

The overlay includes:

- semantic segmentation
- object detection
- traffic-condition detection
- lane detection and ego-path extraction

Display controls:

| Key | Action                       |
| --- | ---------------------------- |
| `1` | Toggle semantic segmentation |
| `2` | Toggle object detections     |
| `3` | Toggle traffic detections    |
| `4` | Toggle lane overlay          |
| `0` | Toggle all overlays          |
| `q` | Quit                         |

The display controls only affect visualisation. They do not disable the
underlying perception modules.

### Fused perception and navigation demo

From `src/`:

```powershell
python demo_fused_lane_following.py --town Town10HD
```

This demo does not use CARLA autopilot.

Instead, the control pipeline is:

```text
RGB camera
    ↓
lane segmentation
    ↓
ego-lane centreline extraction
    ↓
ground-plane projection
    ↓
local world waypoints
    ↓
PathFollower
    ↓
steering / throttle / brake
```

The route is regenerated from the latest lane estimate so perception
directly influences vehicle control.

`--town` is optional on both demos. If omitted, the currently loaded CARLA
map is used.

Different CARLA towns and road sections produce different lane-detection
quality, so a known working route should be rehearsed before the live demo.

## `demo_fused_lane_following.py` driving behaviour

Beyond ordinary lane-following, the fused demo also includes junction
handling and red-light response.

### Junction handling

Pure camera-based lane-following cannot always determine which road branch
to follow through an intersection because the destination lane may not yet
be visible.

When a junction is detected ahead using CARLA's waypoint graph,
`turn_decision.py` constructs a short map-based route through the junction.
Control then returns to perception-driven lane-following afterwards.

The current selection rule is:

1. continue straight when a straight exit is available
2. otherwise alternate between right and left turns

This is local junction handling rather than destination-based route
planning.

### Red traffic lights

`traffic_light_color.py` applies an HSV-based colour classifier to detected
traffic-light regions.

When a sufficiently confident nearby red traffic light is detected, the
current driving command is overridden by reducing throttle and applying
the brake.

Bounding-box height is used as a proximity proxy rather than a calibrated
metric distance.

Yellow-light behaviour is not used in the current demo because colour
classification was less reliable for those cases.

## Traffic perception

General object detection and traffic perception are separate layers.

The object detector identifies general road objects using YOLO.

Traffic perception then filters traffic-relevant detections, including
traffic lights and stop signs, and can optionally attach monocular distance
estimates.

For supported objects, distance is estimated using:

```text
distance = focal_length × real_object_width
           ────────────────────────────────
                observed_pixel_width
```

Objects without a configured real-world width can still be detected but do
not receive a metric distance estimate.

## Known limitations

- The system does not currently provide obstacle or collision avoidance.
- It does not perform destination-based route planning.
- Junction selection uses a simple local rule rather than a global route.
- Lane extraction depends on the quality and visibility of road markings.
- Dashed, missing or ambiguous markings can cause the lane centreline to
  become unstable or temporarily unavailable.
- Multi-lane roads and intersections may produce ambiguous lane-boundary
  detections.
- Ground projection assumes a flat road and fixed camera geometry.
- Traffic-light proximity uses bounding-box size rather than calibrated 3D
  localisation.
- Red-light handling operates independently of the junction state.
- Real-time performance depends on available hardware and model inference
  speed.

These limitations are expected for a baseline system and provide clear
areas for future improvement.

## Reproducing the project

From a fresh clone:

```bash
git clone <repository-url>
cd 26-S2-Assignment1-Group7

python -m venv .venv
source .venv/bin/activate

python -m pip install -e ".[test,docs]"

ruff check .
python -m pytest
sphinx-build -W docs docs/_build/html
```

For the live CARLA demo:

```powershell
python setup_environment.py --carla-version 0.9.16
cd src
python demo_combined_perception.py --town Town10HD
```

or:

```powershell
python demo_fused_lane_following.py --town Town10HD
```

A recorded CARLA demonstration should also be kept as a backup for the live
presentation.

