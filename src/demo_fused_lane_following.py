"""Fused demo: lane detection drives the vehicle during ordinary driving,
handing off to a short map-based route through each junction encountered
-- via navigation_control -- for turning, since pure lane-following can't
decide to turn on its own.

Two driving modes, alternating automatically:

  "lane_following" (the ORIGINAL mechanism, unchanged): a receding-horizon
  local planner. Every cycle:
    1. Run lane detection on the latest camera frame -> image-space centerline.
    2. Back-project it onto the (assumed flat) road plane using known camera
       geometry -> a short (~2-20m) real-world local route, in the vehicle's
       current frame (see ground_projection.py for the geometry, verified via
       forward/inverse round-trip tests before this script was written).
    3. Feed that momentary local route into PathFollower fresh each cycle
       (lane_to_waypoints.py) and take its immediate next command.
    This is different from test_navigation_control_carla.py, which drives one
    long FIXED route set once at the start. Here the route is continuously
    replaced by whatever the camera currently sees.

  "junction_maneuver" (ADDED): pure lane-following has no way to DECIDE to
  turn -- a turn's destination road usually isn't visible in the forward
  camera until the vehicle is already partway through the turn. While in
  lane_following mode, every cycle also checks (via turn_decision.py,
  using CARLA's own map waypoint graph -- the same waypoint.next()/
  is_junction mechanism already used elsewhere in this project) whether a
  junction is coming up within JUNCTION_LOOKAHEAD_M. If so, its available
  exits are classified as straight/right/left by heading change, one is
  picked per the priority rule below, and a SHORT FIXED route through that
  specific junction is built and given to PathFollower ONCE (the normal,
  intended way to use PathFollower -- set_path once, update() repeatedly
  against the same route -- unlike lane_following's every-cycle replan,
  which was always a workaround). Control stays in this mode until
  PathFollower reports the route FINISHED (or a safety timeout), then
  hands back to lane_following.

  Turn priority (explicit choice, not a general planner): if the junction
  has a "straight" exit, ALWAYS take it. Otherwise alternate right, then
  left, then right, ... -- a hardcoded, deliberately simple rule; this is
  NOT navigation (no destination, no route beyond "the next junction").

ADDED: stops for a detected red traffic light (see traffic_light_color.py),
applied regardless of driving mode -- including mid-turn. This is a real,
disclosed limitation: a genuine self-driving system would avoid stopping
halfway through a committed turn; this one doesn't specially handle that
case. Box HEIGHT (pixels) is used as a proximity proxy, not a real metric
distance -- traffic lights are mounted well above the road plane, so
ground_projection.py's ground-plane assumption doesn't apply to them, and
no known real-world width exists for them either (see traffic_perception's
own OBJECT_WIDTHS_M comment: "Traffic lights have no single standard
width"). This was built once already this session, reverted after a run
where the light's own red phase ended before the car got close enough to
react (a timing coincidence, not a logic bug -- see the project's own
development history), and is restored here essentially unchanged, red only
(not yellow -- yellow previously mis-fired on a nearby sign sharing the
detected box's crop, not the light's own bulb).

ADDED (showcase polish -- see overlay_enhancements.py): the DISPLAYED lane
overlay is EMA-smoothed across frames and fades out over a short hold on a
miss instead of vanishing (or flashing a big "NO LANE DETECTED"); a small
HUD gives a persistent status badge + live FPS. Control never depends on
this smoothed version -- only the raw lane_path is ever used to drive.

ADDED: the same segmentation/objects/traffic/lane toggle keys as
demo_combined_perception.py (1/2/3/4 individually, 0 for all), purely for
display -- all four are still computed every cycle regardless of toggle
state, so flipping a layer on mid-demo never causes a stutter.

ADDED: spawns 3 junctions further along the SAME straight route
demo_combined_perception.py picks for a given town, rather than at a
default/first-available spawn point -- so segment 2 of the live demo
starts somewhere visually distinct from where segment 1 left off.

Known limitations (see report):
  - Assumes a perfectly flat road and a fixed, uncalibrated camera mount.
  - Lane-following miss-handling is a bounded hold -> slow creep -> stop
    chain (see HOLD_MISSES/CREEP_MISSES below), enough to drive through a
    short gap in lane markings without getting stuck indefinitely.
  - The ego-lane boundary-selection heuristic (perception/lane_path.py)
    assumes exactly one marking line per side -- wide, multi-lane roads
    (e.g. Town10HD) can cause it to lock onto the wrong line. Use a simple
    single-lane-per-direction town (e.g. Town01) for a reliable demo.
  - Turn selection is a hardcoded straight-then-alternate rule, not
    navigation -- there is no destination and no route beyond the next
    junction. Obstacle/collision avoidance and other traffic are not
    considered when choosing or executing a turn.
  - Red-light stopping uses box height as a distance proxy (see the ADDED
    note above), and is not specially suppressed mid-turn.

Run from src/ (so all packages resolve), with CarlaUE4.exe running.
Needs a real (non-headless) opencv-python install, since this displays a
live window -- see the project's opencv-python-headless note if
cv2.imshow fails with a "not implemented" error. Press 'q' in the window
to quit early.

Keys during the live window:
    1 -- toggle segmentation
    2 -- toggle object detection
    3 -- toggle traffic detection
    4 -- toggle lane overlay
    0 -- toggle all layers on/off together
    q -- quit

Usage:
    python demo_fused_lane_following.py
    python demo_fused_lane_following.py --town Town01
"""

import argparse
import math
import sys
import time
from typing import Iterable

import carla
import cv2
import numpy as np

sys.path.insert(0, ".")
from carla_adapter.adapter import apply_command, carla_vehicle_to_state
from ground_projection import (  # noqa: F401 (re-exported for reference)
    pixel_to_vehicle_local,
    vehicle_local_to_world,
)
from lane_to_waypoints import build_local_route
from navigation_control.controller import PathFollower
from navigation_control.types import ControlCommand, ControllerConfig, ControlStatus
from overlay_enhancements import (
    LaneTemporalSmoother,
    PerceptionHUD,
    blend_overlay,
    confidence_thickness,
)
from perception import ObjectDetector, PerceptionConfig
from perception.lane_inference import LaneSegmenter
from perception.lane_path import draw_ego_path, extract_ego_path
from segmentation.segmentation import SemanticSegmenter
from traffic_light_color import classify_traffic_light_color
from traffic_perception import BoundingBox, ModelPrediction, TrafficCategory, TrafficPerception
from turn_decision import (
    build_turn_route,
    find_junction_ahead,
    get_junction_exit_candidates,
    select_turn_direction,
)

SEGMENTATION_WEIGHTS = "../segmentation_model.pth"
LANE_WEIGHTS = "../lane_unet_best.pth"
CAMERA_HEIGHT_M = 2.4
CAMERA_FORWARD_OFFSET_M = 1.5
TARGET_SPEED_MPS = 3.0  # CHANGED: was 5.0 -- slower cruise for a more
                          # controlled drive. Also directly helps the red-
                          # light reaction (more time/less distance needed
                          # to stop at a lower approach speed) and likely
                          # the lane-tracking smoothness (more time for the
                          # control loop to correct between updates at the
                          # same loop rate). At stopping_deceleration_mps2
                          # =1.0 (ControllerConfig's default), stopping
                          # distance from 3.0 m/s is ~4.5m -- comfortably
                          # inside the 2-20m local horizon, same margin
                          # logic as the original 5.0 m/s/12.5m choice.
DT_S = 0.1
MAX_DURATION_S = 120.0  # CHANGED: was 60.0 -- a run reported not reaching
                          # any junction within 60s. Likely just Town01's
                          # road layout combined with the JUNCTIONS_AFTER_START
                          # spawn placement (3 junctions past the straight-
                          # route start), not a bug -- doubled to give a
                          # better chance of actually witnessing a turn.
                          # Tune further/shorten back down once you've
                          # confirmed the turning logic itself works.
FPS_LOG_INTERVAL = 20
SEGMENTATION_SKIP_INTERVAL = 3  # see the FIXED note in the main loop --
                                  # segmentation is skipped most cycles and
                                  # its last mask reused, since it has zero
                                  # functional role (pure display) unlike
                                  # lane/object detection.
JUNCTIONS_AFTER_START = 3  # spawn this many junctions further along the
                            # same straight route demo_combined_perception.py
                            # picks, so segment 2 starts somewhere visually
                            # distinct rather than retreading segment 1's road.
STRAIGHT_PATH_DISTANCE_M = 250.0  # mirrors demo_combined_perception.py's
STRAIGHT_PATH_STEP_M = 10.0        # constants, so both scripts deterministically
                                     # agree on the same "best straight start"
                                     # for a given town.

# ADDED: junction turn-maneuver tuning. See turn_decision.py for the
# actual classification/selection logic these feed into.
JUNCTION_LOOKAHEAD_M = 15.0        # start looking for a turn-maneuver
                                     # opportunity once a junction is this
                                     # close, while still in lane_following.
STRAIGHT_HEADING_THRESHOLD_DEG = 25.0  # a candidate exit within this many
                                          # degrees of the approach heading
                                          # counts as "straight".
POST_JUNCTION_ROUTE_M = 20.0       # how far past the junction the built
                                     # turn route extends, before handing
                                     # back to lane_following.
TURN_SPEED_MPS = 2.0               # CHANGED: was 3.0 -- lowered alongside
                                     # TARGET_SPEED_MPS, keeping turns
                                     # slower than cruise.
JUNCTION_MAX_DURATION_S = 12.0     # safety cutoff -- if a turn maneuver
                                     # hasn't finished by then, hand back to
                                     # lane_following anyway rather than get
                                     # stuck (same "bounded, not unbounded"
                                     # philosophy as HOLD_MISSES/CREEP_MISSES).

TRAFFIC_LIGHT_STOP_HEIGHT_PX = 20  # FIXED: was 40 -- a real run showed the
                                     # car slowing but not fully stopping
                                     # before reaching the light. A SMALLER
                                     # threshold here means the box is still
                                     # small (the light still far away) when
                                     # the trigger fires, giving more real
                                     # distance/time to decelerate -- a
                                     # LARGER threshold (the previous 40)
                                     # only fires once the box has already
                                     # grown big, i.e. once already close.
                                     # Still a proximity PROXY, not a real
                                     # metric distance (see the module
                                     # docstring's ADDED note).

# FIXED: a run showed red objects that AREN'T actually traffic lights
# triggering a stop -- YOLO's own class="traffic light" detection was
# being trusted at ObjectDetector's default confidence_threshold=0.25,
# a low bar that lets weak/ambiguous detections through. This is the
# "detect a traffic light [confidently], THEN classify its colour" gate:
# every "traffic light" detection is still SHOWN on screen regardless of
# confidence (see traffic_light_detections_for_display below -- useful
# for seeing what YOLO is picking up, including weak/wrong ones), but
# only a detection at or above this stricter bar is trusted enough to
# actually influence driving.
MIN_TRAFFIC_LIGHT_STOP_CONFIDENCE = 0.6

OBJECT_WIDTHS_M = {"stop sign": 0.75}
CATEGORY_MAP = {
    "traffic light": TrafficCategory.LIGHT,
    "stop sign": TrafficCategory.SIGN,
}
GENERAL_BOX_COLOR_BGR = (255, 200, 0)
TRAFFIC_BOX_COLOR_BGR = (0, 140, 255)

# Layer toggle keys -- identical scheme to demo_combined_perception.py.
LAYER_KEYS = {
    ord("1"): "segmentation",
    ord("2"): "objects",
    ord("3"): "traffic",
    ord("4"): "lane",
}
ALL_LAYERS_KEY = ord("0")
LAYER_DISPLAY_NAMES = {
    "segmentation": "1:Segmentation",
    "objects": "2:Objects",
    "traffic": "3:Traffic",
    "lane": "4:Lane",
}
HUD_ON_COLOR_BGR = (0, 255, 0)
HUD_OFF_COLOR_BGR = (90, 90, 90)


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference b - a, wrapped to (-180, 180]."""
    d = (b - a) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def measure_straightness(start_waypoint, distance_m=STRAIGHT_PATH_DISTANCE_M, step_m=STRAIGHT_PATH_STEP_M):
    """Walk forward from start_waypoint for distance_m, returning
    (total_heading_change_deg, junction_count, path_waypoints). Mirrors
    demo_combined_perception.py's function of the same name exactly, so
    both scripts pick the same "straightest" starting point for a town.
    """
    current = start_waypoint
    total_change = 0.0
    junction_count = 0
    path = [current]
    steps = int(distance_m / step_m)
    for _ in range(steps):
        next_options = current.next(step_m)
        if not next_options:
            break
        next_wp = next_options[0]
        if next_wp.is_junction:
            junction_count += 1
        total_change += abs(_angle_diff_deg(
            current.transform.rotation.yaw, next_wp.transform.rotation.yaw
        ))
        path.append(next_wp)
        current = next_wp
    return total_change, junction_count, path


def rank_straight_spawn_candidates(carla_map, spawn_points, distance_m=STRAIGHT_PATH_DISTANCE_M):
    """Rank spawn points by how straight/turn-free the road ahead is, best
    first -- mirrors demo_combined_perception.py's function of the same
    name exactly.
    """
    scored = []
    for sp in spawn_points:
        wp = carla_map.get_waypoint(sp.location)
        total_change, junction_count, path = measure_straightness(wp, distance_m)
        score = junction_count * 1000 + total_change
        scored.append((score, sp, path))
    scored.sort(key=lambda item: item[0])
    return scored


def advance_past_junctions(start_waypoint, num_junctions, step_m=10.0, max_steps=2000):
    """Walk forward from start_waypoint, returning a waypoint located
    shortly after passing through num_junctions distinct junctions.
    """
    current = start_waypoint
    passed_junctions = 0
    was_in_junction = current.is_junction
    for _ in range(max_steps):
        next_options = current.next(step_m)
        if not next_options:
            break
        next_wp = next_options[0]
        if was_in_junction and not next_wp.is_junction:
            passed_junctions += 1
        was_in_junction = next_wp.is_junction
        current = next_wp
        if passed_junctions >= num_junctions:
            break
    return current


HOLD_MISSES = 10
CREEP_MISSES = 60         # ~6s at DT_S=0.1
CREEP_THROTTLE = 0.25     # slow and deliberate -- just enough to clear a
                           # short markless gap, not a driving decision.


def make_cached_backend(shared_state: dict):
    """A traffic_perception DetectionBackend that reads a shared cache
    instead of re-running YOLO -- identical to demo_combined_perception.py's
    helper of the same name.
    """

    def backend(image: np.ndarray) -> Iterable[ModelPrediction]:
        predictions = []
        for detection in shared_state["detections"]:
            category = CATEGORY_MAP.get(detection.class_name)
            if category is None:
                continue
            box = detection.bounding_box
            predictions.append(
                ModelPrediction(
                    category=category,
                    label=detection.class_name,
                    confidence=detection.confidence,
                    bounding_box=BoundingBox(
                        x_min=box.x_min, y_min=box.y_min, x_max=box.x_max, y_max=box.y_max
                    ),
                )
            )
        return predictions

    return backend


def draw_object_detections(image_bgr: np.ndarray, detections) -> np.ndarray:
    """Draw general object-detection boxes (any class). Box thickness
    scales with confidence -- purely a display cue.
    """
    output = image_bgr.copy()
    for d in detections:
        box = d.bounding_box
        p1 = (int(box.x_min), int(box.y_min))
        p2 = (int(box.x_max), int(box.y_max))
        thickness = confidence_thickness(d.confidence, min_px=1, max_px=3)
        cv2.rectangle(output, p1, p2, GENERAL_BOX_COLOR_BGR, thickness)
        label = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(
            output, label, (p1[0], max(p1[1] - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GENERAL_BOX_COLOR_BGR, 2,
        )
    return output


def draw_traffic_detections(image_bgr: np.ndarray, detections) -> np.ndarray:
    """Draw traffic-relevant boxes (light/sign), including distance if known."""
    output = image_bgr.copy()
    for d in detections:
        box = d.bounding_box
        p1 = (int(box.x_min), int(box.y_min))
        p2 = (int(box.x_max), int(box.y_max))
        thickness = confidence_thickness(d.confidence, min_px=2, max_px=4)
        cv2.rectangle(output, p1, p2, TRAFFIC_BOX_COLOR_BGR, thickness)
        distance_str = f" {d.distance_m:.1f}m" if d.distance_m is not None else ""
        label = f"[{d.category.value}] {d.label}{distance_str}"
        cv2.putText(
            output, label, (p1[0], min(p2[1] + 20, image_bgr.shape[0] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, TRAFFIC_BOX_COLOR_BGR, 2,
        )
    return output


def draw_layer_hud(image_bgr: np.ndarray, visible_layers: dict) -> np.ndarray:
    """Small top-right legend showing which layers are currently shown vs.
    hidden -- identical to demo_combined_perception.py's helper of the
    same name.
    """
    output = image_bgr.copy()
    x = image_bgr.shape[1] - 190
    y = 25
    for key in ("segmentation", "objects", "traffic", "lane"):
        color = HUD_ON_COLOR_BGR if visible_layers[key] else HUD_OFF_COLOR_BGR
        cv2.putText(
            output, LAYER_DISPLAY_NAMES[key], (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )
        y += 24
    return output


def _save_failure_snapshot(frame_rgb: np.ndarray, lane_mask: np.ndarray, name: str) -> None:
    """Save the raw frame and the (empty/failed) lane mask side by side,
    for visually inspecting what the scene looked like when detection failed.
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    mask_visual = cv2.cvtColor((lane_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    side_by_side = np.hstack([frame_bgr, mask_visual])
    filename = f"{name}.png"
    cv2.imwrite(filename, side_by_side)
    print(f"Saved failure snapshot to {filename}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--town", default=None,
        help="Load this town instead of using whatever CarlaUE4 already has "
             "open (e.g. --town Town01). Town10HD's wide, multi-lane "
             "boulevards can have 3+ parallel lines per direction -- more "
             "than lane_path.py's ego-lane heuristic can reliably "
             "disambiguate. A simple single-lane-per-direction town (e.g. "
             "Town01) avoids this entirely for a smoother demo.",
    )
    args = parser.parse_args()

    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    world = client.load_world(args.town) if args.town else client.get_world()
    carla_map = world.get_map()

    for actor in world.get_actors().filter("vehicle.*"):
        actor.destroy()
    for actor in world.get_actors().filter("sensor.*"):
        actor.destroy()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.*")[0]
    spawn_points = world.get_map().get_spawn_points()

    ranked_candidates = rank_straight_spawn_candidates(carla_map, spawn_points)

    vehicle = None
    for _, sp, path in ranked_candidates:
        base_wp = carla_map.get_waypoint(sp.location)
        advanced_wp = advance_past_junctions(base_wp, JUNCTIONS_AFTER_START)
        candidates_along_path = [advanced_wp] + advanced_wp.next(5.0) + advanced_wp.next(10.0)
        spawned = False
        for wp in candidates_along_path:
            try:
                vehicle = world.spawn_actor(vehicle_bp, wp.transform)
                spawned = True
                break
            except RuntimeError:
                continue
        if spawned:
            break
    if vehicle is None:
        raise RuntimeError("No free spawn point found")

    # No autopilot -- navigation_control drives the vehicle directly.

    image_width, image_height, fov_deg = 800, 600, 90.0
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(image_width))
    camera_bp.set_attribute("image_size_y", str(image_height))
    camera_bp.set_attribute("fov", str(fov_deg))
    camera_transform = carla.Transform(
        carla.Location(x=CAMERA_FORWARD_OFFSET_M, z=CAMERA_HEIGHT_M)
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    focal_length_px = image_width / (2 * math.tan(math.radians(fov_deg) / 2))

    collision_events = []
    collision_bp = blueprint_library.find("sensor.other.collision")
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

    def on_collision(event):
        collision_events.append((time.time(), event.other_actor.type_id))
        print(f"\n*** COLLISION with {event.other_actor.type_id} ***\n")

    collision_sensor.listen(on_collision)

    captured = {}

    def on_frame(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        bgr = array[:, :, :3]
        rgb = bgr[:, :, ::-1]
        captured["frame"] = np.ascontiguousarray(rgb)

    camera.listen(on_frame)

    print("Loading models (this can take a moment)...")
    lane_segmenter = LaneSegmenter(checkpoint_path=LANE_WEIGHTS)
    segmenter = SemanticSegmenter(weights_path=SEGMENTATION_WEIGHTS, input_size=(512, 256))
    detector = ObjectDetector(
        PerceptionConfig(
            model_name="yolo26n.pt",
            confidence_threshold=0.20,
        )
    )
    shared_detection_state = {"detections": ()}
    traffic_perception = TrafficPerception(
        make_cached_backend(shared_detection_state),
        confidence_threshold=0.4,
        object_widths_m=OBJECT_WIDTHS_M,
    )

    config = ControllerConfig(wheelbase_m=2.5, max_steering_angle_rad=0.6, lookahead_m=5.0)
    follower = PathFollower(config)

    display_smoother = LaneTemporalSmoother()
    hud = PerceptionHUD(
        legend=(
            ("General object", GENERAL_BOX_COLOR_BGR),
            ("Traffic light/sign", TRAFFIC_BOX_COLOR_BGR),
        )
    )
    print("Models loaded. Starting live overlay.")
    print("Keys: 1=segmentation 2=objects 3=traffic 4=lane 0=toggle all  q=quit")

    visible_layers = {"segmentation": False, "objects": False, "traffic": False, "lane": False}

    # ADDED: driving-mode state machine (see module docstring). Starts in
    # lane_following; turn_preference starts "right" per the stated
    # priority (straight always wins; otherwise right then left).
    driving_mode = "lane_following"
    turn_preference = "right"
    turn_maneuver_start_time = None
    turn_direction_taken = None

    last_command = None
    consecutive_misses = 0
    frames_processed = 0
    lanes_found = 0
    failure_snapshots_taken = {"early": False, "late": False}
    frame_times: list = []
    cached_seg_mask = {"value": None}

    try:
        print("Waiting for the first camera frame...")
        timeout = time.time() + 15
        while "frame" not in captured and time.time() < timeout:
            time.sleep(0.05)
        if "frame" not in captured:
            raise RuntimeError("No camera frame received within timeout")

        print("Starting fused lane-following loop...")
        start_time = time.time()
        last_print = 0.0
        last_cycle_time = [time.time()]

        while time.time() - start_time < MAX_DURATION_S:
            cycle_start = time.time()
            frame = captured["frame"]
            state = carla_vehicle_to_state(vehicle)
            vehicle_location = vehicle.get_transform().location
            current_carla_waypoint = carla_map.get_waypoint(vehicle_location)

            # CHANGED: dt is now measured once per cycle regardless of
            # driving mode (previously only computed inside a successful
            # lane_following route-build) -- junction_maneuver's own
            # follower.update() calls need a real dt too.
            now = time.time()
            actual_dt_s = max(0.02, min(0.5, now - last_cycle_time[0]))
            last_cycle_time[0] = now

            lane_mask = lane_segmenter.predict(frame)
            lane_path = extract_ego_path(lane_mask)

            # FIXED: segmentation used to run every cycle, same as the
            # other three -- but unlike lane detection (drives the car)
            # and object detection (feeds the red-light safety check
            # below), segmentation is PURE cosmetic display with zero
            # functional role, and BiSeNetV2 is the heaviest of the four
            # models. Running it every cycle made the whole loop
            # noticeably slower -- and since CARLA's simulation runs in
            # real time independent of this script's own processing
            # speed (not synchronous/client-gated), a slower loop means
            # control commands get issued less often relative to the
            # vehicle's continuous real motion. This plausibly explains
            # both observed lane-swaying (stale/infrequent corrections)
            # and a too-late red-light reaction. Only segmentation is
            # skipped -- traffic_perception.detect() is already cheap
            # (no separate model call, just filters detections YOLO
            # already computed via the cached backend), so it stays.
            if frames_processed % SEGMENTATION_SKIP_INTERVAL == 0 or cached_seg_mask["value"] is None:
                cached_seg_mask["value"] = segmenter.predict(frame)
            seg_mask = cached_seg_mask["value"]

            detections = detector.detect(frame)  # UNCHANGED: every cycle -- feeds the red-light check
            shared_detection_state["detections"] = detections
            traffic_detections = traffic_perception.detect(frame)  # UNCHANGED: cheap, no separate model call

            # ADDED (restored): red-light detection -- see the module
            # docstring's ADDED note for why this is red-only and uses box
            # height as a proximity proxy. Independent of driving_mode --
            # checked every cycle regardless of lane_following/junction_maneuver.
            should_stop_for_light = False
            traffic_light_detections_for_display = []
            for detection in detections:
                if detection.class_name != "traffic light":
                    continue
                color = classify_traffic_light_color(frame, detection.bounding_box)
                box_height = detection.bounding_box.y_max - detection.bounding_box.y_min
                # Shown regardless of confidence -- lets you SEE what YOLO
                # is picking up as "traffic light" at any confidence,
                # including weak/wrong ones, even though only a confident
                # one below is trusted to actually influence driving.
                traffic_light_detections_for_display.append((detection.bounding_box, color, detection.confidence))
                if (
                    color == "red"
                    and box_height >= TRAFFIC_LIGHT_STOP_HEIGHT_PX
                    and detection.confidence >= MIN_TRAFFIC_LIGHT_STOP_CONFIDENCE
                ):
                    should_stop_for_light = True

            if lane_path is None:
                if consecutive_misses == 2 and not failure_snapshots_taken["early"]:
                    _save_failure_snapshot(frame, lane_mask, "failure_early")
                    failure_snapshots_taken["early"] = True
                if consecutive_misses == 29 and not failure_snapshots_taken["late"]:
                    _save_failure_snapshot(frame, lane_mask, "failure_late")
                    failure_snapshots_taken["late"] = True

            command = None

            # ADDED: junction-maneuver mode -- a FIXED route was set ONCE
            # when this mode was entered (below); just update() against it
            # repeatedly, the normal intended PathFollower usage pattern,
            # unlike lane_following's every-cycle replan.
            if driving_mode == "junction_maneuver":
                result = follower.update(state, TURN_SPEED_MPS, actual_dt_s)
                command = result.command
                timed_out = (time.time() - turn_maneuver_start_time) > JUNCTION_MAX_DURATION_S
                if result.status == ControlStatus.FINISHED or timed_out:
                    reason = "timed out" if timed_out else "finished"
                    print(f"[frame {frames_processed}] Junction maneuver ({turn_direction_taken}) {reason} -- resuming lane following")
                    driving_mode = "lane_following"

            if driving_mode == "lane_following":
                if lane_path is not None:
                    route = build_local_route(
                        lane_path["centerline"], image_width, image_height, focal_length_px,
                        CAMERA_HEIGHT_M, CAMERA_FORWARD_OFFSET_M,
                        state.x_m, state.y_m, state.yaw_rad,
                    )
                    if route is not None:
                        try:
                            saved_integral = follower._speed._integral
                            saved_previous_speed = follower._speed._previous_speed
                            follower.set_path(route)
                            follower._speed._integral = saved_integral
                            follower._speed._previous_speed = saved_previous_speed

                            result = follower.update(state, TARGET_SPEED_MPS, actual_dt_s)

                            command = result.command

                            lanes_found += 1
                            consecutive_misses = 0
                        except ValueError:
                            pass  # degenerate route this cycle -- fall through to the miss-handling below

                # ADDED: junction-ahead check -- only while genuinely in
                # lane_following. If a maneuver just finished above this
                # same cycle, driving_mode is already back to
                # "lane_following" here, so this correctly looks for the
                # NEXT junction immediately too (continuous multi-junction
                # driving), overriding whatever command lane-following
                # just computed with the turn route's first command instead.
                pre_junction_wp = find_junction_ahead(current_carla_waypoint, JUNCTION_LOOKAHEAD_M)
                if pre_junction_wp is not None:
                    approach_yaw, candidates = get_junction_exit_candidates(pre_junction_wp)
                    chosen_wp, direction_taken = select_turn_direction(
                        approach_yaw, candidates, turn_preference, STRAIGHT_HEADING_THRESHOLD_DEG
                    )
                    if chosen_wp is not None:
                        turn_route = build_turn_route(pre_junction_wp, chosen_wp, POST_JUNCTION_ROUTE_M)
                        if turn_route is not None:
                            try:
                                follower.set_path(turn_route)  # deliberate reset() here -- a real
                                                                  # phase change, not the receding-
                                                                  # horizon PID-preservation case above.
                                result = follower.update(state, TURN_SPEED_MPS, actual_dt_s)
                                command = result.command
                                driving_mode = "junction_maneuver"
                                turn_maneuver_start_time = time.time()
                                turn_direction_taken = direction_taken
                                if direction_taken in ("right", "left"):
                                    turn_preference = "left" if direction_taken == "right" else "right"
                                print(f"[frame {frames_processed}] Junction ahead -- taking {direction_taken}")
                            except ValueError:
                                pass  # degenerate turn route -- keep whatever lane_following command (if any) was computed above

            if command is None:
                consecutive_misses += 1
                if last_command is None:
                    command = ControlCommand(steering_angle_rad=0.0, throttle=0.0, brake=0.5)
                elif consecutive_misses <= HOLD_MISSES:
                    command = last_command
                elif consecutive_misses <= CREEP_MISSES:
                    command = ControlCommand(
                        steering_angle_rad=last_command.steering_angle_rad,
                        throttle=CREEP_THROTTLE,
                        brake=0.0,
                    )
                else:
                    command = ControlCommand(steering_angle_rad=0.0, throttle=0.0, brake=0.5)

            # ADDED (restored): red-light override -- an application-level
            # safety policy on top of whatever driving_mode decided, same
            # safety override on top of the selected driving command. Keeps
            # whatever steering was already chosen but forces throttle off
            # and a real brake. Applies regardless of driving_mode -- see
            # the module docstring's known limitation about not specially
            # handling a stop mid-turn.
            # FIXED: brake floor was 0.4 -- ControllerConfig's own default
            # stopping_deceleration_mps2=1.0 (see TARGET_SPEED_MPS's
            # comment: "~12.5m braking-distance...at a 5 m/s target speed")
            # implies the project's own assumed comfortable deceleration is
            # around 1.0 m/s^2; brake=0.4 is a noticeably gentler
            # deceleration than that, so the ACTUAL stopping distance at
            # 0.4 likely exceeded what TRAFFIC_LIGHT_STOP_HEIGHT_PX's
            # trigger point left available -- matching the observed "slows
            # but doesn't stop" run. Raised to 0.7 for a firmer stop,
            # alongside triggering earlier (see TRAFFIC_LIGHT_STOP_HEIGHT_PX
            # above) -- both address the same miss from two angles.
            if should_stop_for_light:
                command = ControlCommand(
                    steering_angle_rad=command.steering_angle_rad,
                    throttle=0.0,
                    brake=max(0.7, command.brake),
                )

            apply_command(vehicle, command, config.max_steering_angle_rad)
            last_command = command
            frames_processed += 1
            frame_times.append(time.time() - cycle_start)
            if len(frame_times) > FPS_LOG_INTERVAL:
                frame_times.pop(0)
            current_fps = (len(frame_times) / sum(frame_times)) if frame_times and sum(frame_times) > 0 else None

            if visible_layers["segmentation"]:
                overlay_rgb = segmenter.visualize(frame, seg_mask, alpha=0.35)
                display_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            else:
                display_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            hud.mark("segmentation", "ok")

            if visible_layers["objects"]:
                display_bgr = draw_object_detections(display_bgr, detections)
            hud.mark("detection", "ok")

            if visible_layers["traffic"]:
                display_bgr = draw_traffic_detections(display_bgr, traffic_detections)
            hud.mark("traffic", "ok")

            smoothed_path, lane_alpha, lane_status = display_smoother.update(lane_path)
            hud.mark("lane", lane_status)
            if visible_layers["lane"] and smoothed_path is not None:
                lane_drawn_bgr = draw_ego_path(display_bgr.copy(), smoothed_path)
                display_bgr = blend_overlay(display_bgr, lane_drawn_bgr, lane_alpha)

            _LIGHT_DISPLAY_COLOR_BGR = {
                "red": (0, 0, 255), "yellow": (0, 220, 255),
                "green": (0, 200, 0), "unknown": (200, 200, 200),
            }
            for box, color, confidence in traffic_light_detections_for_display:
                p1, p2 = (int(box.x_min), int(box.y_min)), (int(box.x_max), int(box.y_max))
                cv2.rectangle(display_bgr, p1, p2, _LIGHT_DISPLAY_COLOR_BGR[color], 2)
                # Confidence shown alongside color -- a detection below
                # MIN_TRAFFIC_LIGHT_STOP_CONFIDENCE is visibly marked
                # "(ignored)" so it's obvious on screen why a red-looking
                # box didn't trigger a stop.
                trust_note = "" if confidence >= MIN_TRAFFIC_LIGHT_STOP_CONFIDENCE else " (ignored)"
                label = f"{color} {confidence:.2f}{trust_note}"
                cv2.putText(
                    display_bgr, label, (p1[0], max(p1[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _LIGHT_DISPLAY_COLOR_BGR[color], 2,
                )

            display_bgr = draw_layer_hud(display_bgr, visible_layers)
            display_bgr = hud.draw(display_bgr, fps=current_fps)
            status_text = (
                f"mode={driving_mode} steer={command.steering_angle_rad:+.3f} "
                f"throttle={command.throttle:.2f} brake={command.brake:.2f}"
            )
            cv2.putText(
                display_bgr, status_text, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
            )
            if should_stop_for_light:
                cv2.putText(
                    display_bgr, "STOPPING FOR LIGHT", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
            cv2.imshow("Fused lane following", display_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in LAYER_KEYS:
                layer = LAYER_KEYS[key]
                visible_layers[layer] = not visible_layers[layer]
                print(f"Layer '{layer}' {'ON' if visible_layers[layer] else 'OFF'}")
            elif key == ALL_LAYERS_KEY:
                turn_on = not any(visible_layers.values())
                for layer in visible_layers:
                    visible_layers[layer] = turn_on
                print(f"All layers {'ON' if turn_on else 'OFF'}")

            if time.time() - last_print > 1.0:
                hit_rate = lanes_found / frames_processed * 100
                fps_str = f"{current_fps:.1f}" if current_fps is not None else "n/a"
                print(
                    f"frames={frames_processed:4d}  mode={driving_mode:16s}  lane hit rate={hit_rate:5.1f}%  "
                    f"fps={fps_str}  "
                    f"steer={command.steering_angle_rad:+.3f}  throttle={command.throttle:.2f}  "
                    f"brake={command.brake:.2f}  consecutive_misses={consecutive_misses}"
                )
                last_print = time.time()

            time.sleep(DT_S)

        print(f"\nStopped after {MAX_DURATION_S}s. Lane detection hit rate: "
              f"{lanes_found / frames_processed * 100:.1f}% over {frames_processed} frames.")

        if collision_events:
            first_collision_time = collision_events[0][0] - start_time
            print(
                f"\n{len(collision_events)} collision(s) detected during this run. "
                f"First collision at t={first_collision_time:.1f}s (with "
                f"{collision_events[0][1]})."
            )
        else:
            print("\nNo collisions detected during this run.")

    finally:
        cv2.destroyAllWindows()
        collision_sensor.stop()
        collision_sensor.destroy()
        camera.stop()
        camera.destroy()
        vehicle.destroy()


if __name__ == "__main__":
    main()
