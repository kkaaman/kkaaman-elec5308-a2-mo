"""Combined live perception overlay: segmentation + object detection +
lane detection + traffic detection, all drawn on one CARLA camera feed --
each layer individually toggleable so each module can be showcased on its
own before combining them, without needing separate demo runs.

This demonstrates all four perception modules' real, live output together.
It does NOT use any of them to control the vehicle -- the vehicle drives
via autopilot. Path following/control is demonstrated separately via
navigation_control (see test_navigation_control_carla.py / demo_fused_
lane_following.py), since that's a driving behaviour to watch, not a
visual layer to draw on a frame -- it isn't part of this toggle set.

All four models still run every frame regardless of which layers are
currently shown (see run_lane_detection's existing every-other-frame
skip for the one exception, kept for FPS -- see main()'s note): toggling
only changes what gets DRAWN, so flipping a layer on mid-demo never
causes a stutter from a model suddenly starting up.

ADDED (showcase polish -- see overlay_enhancements.py): the lane overlay
is EMA-smoothed across frames and fades out over a short hold on a miss
instead of vanishing instantly; a persistent HUD (status badges + legend
+ FPS) replaces the old console-only FPS print; object/traffic box
thickness scales with detection confidence. None of this changes any
model's output, only how it's displayed.

Keys during the live window:
    1 -- toggle segmentation
    2 -- toggle object detection
    3 -- toggle traffic detection
    4 -- toggle lane detection
    0 -- toggle all layers on/off together
    q -- quit

Run from src/ (so all four packages resolve), with CarlaUE4.exe running.
Needs a real (non-headless) opencv-python install, since this displays a
live window -- see the project's opencv-python-headless note if `cv2.imshow`
fails with a "not implemented" error.

Usage:
    python demo_combined_perception.py
    python demo_combined_perception.py --town Town01

    # ADDED (A2, BL-B-05 prep): choose the segmentation model for layer 1
    python demo_combined_perception.py --seg-backend segformer_b0
    python demo_combined_perception.py --seg-backend yolo_seg
    python demo_combined_perception.py --seg-backend yolo_seg --seg-weights ../yolov8n-seg.pt
"""

import argparse
import sys
import time
from collections import deque
from typing import Iterable

import carla
import cv2
import numpy as np

sys.path.insert(0, ".")
from demo_segmentation import (
    add_segmentation_args,
    build_demo_segmenter,
    require_real_weights,
    required_weight_files,
)
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
from traffic_perception import BoundingBox, ModelPrediction, TrafficCategory, TrafficPerception

SEGMENTATION_WEIGHTS = "../segmentation_model.pth"
LANE_WEIGHTS = "../lane_unet_best.pth"  # UPDATED: retrained on TuSimple +
                                              # CARLA-derived Town01/03/05/04
                                              # data (see train_lane_segmentation.py) --
                                              # held-out Town10HD eval went
                                              # from 0.1836 -> 0.6402 average
                                              # IoU vs. the original
                                              # lane_unet_best.pth, still on
                                              # disk unchanged for comparison/
                                              # rollback if needed.
TRAFFIC_MANAGER_PORT = 8001
FPS_LOG_INTERVAL = 20

OBJECT_WIDTHS_M = {"stop sign": 0.75}
CATEGORY_MAP = {
    "traffic light": TrafficCategory.LIGHT,
    "stop sign": TrafficCategory.SIGN,
}

# General object-detection boxes (any class) vs. traffic-relevant ones
# (light/sign) get visually distinguished, since the traffic-relevant subset
# additionally carries category + distance information the general boxes don't.
GENERAL_BOX_COLOR_BGR = (255, 200, 0)
TRAFFIC_BOX_COLOR_BGR = (0, 140, 255)

# Layer toggle keys -> visible_layers dict key. '0' toggles all four
# together (the "everything combined" finale); each digit toggles one
# layer on its own so the demo can walk through modules individually.
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


def make_cached_backend(shared_state: dict):
    """A traffic_perception DetectionBackend that reads a shared cache
    instead of re-running YOLO.

    Without this, calling perception.detect(frame) each frame would invoke
    ObjectDetector.detect(frame) a SECOND time internally (once for the
    general object-detection overlay, once inside TrafficPerception's own
    backend call) -- doubling YOLO inference cost for no benefit, since
    both calls would produce identical results on the same frame. The main
    loop below runs detect() once per frame and stores the result here;
    this backend just reads it back (ignoring its own `image` argument,
    since the shared cache is already keyed to the current frame by the
    caller's own per-frame update order).
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
    """Draw general object-detection boxes (any class) on the image.

    Box thickness scales with confidence (confidence_thickness) so a
    near-threshold detection visually reads as tentative rather than
    identical to a confident one -- purely a display cue.
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
    """Draw a small top-right legend showing which layers are currently
    shown vs. hidden, so the active toggle state is legible on-screen --
    both to whoever's presenting live and in any recorded clip -- rather
    than only visible to whoever's at the keyboard.
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


def build_combined_overlay(
    frame_rgb: np.ndarray,
    segmenter: SemanticSegmenter,
    detector: ObjectDetector,
    lane_segmenter: LaneSegmenter,
    traffic_perception: TrafficPerception,
    shared_detection_state: dict,
    cached_lane_path: dict,
    smoother: LaneTemporalSmoother,
    hud: PerceptionHUD,
    run_lane_detection: bool = True,
    visible_layers: dict | None = None,
    fps: float | None = None,
) -> np.ndarray:
    """Run all four perception modules on one frame and return a combined
    BGR overlay image, ready for cv2.imshow/cv2.imwrite.

    Pulled out of the main loop as its own function so it can be unit
    tested with fake/stand-in models, independent of a live CARLA
    connection -- this is the most complex integration point in the
    project (four models' outputs drawn together), worth verifying in
    isolation before trusting it live.

    Args:
        cached_lane_path: mutable holder for the most recent RAW lane path
            result (pre-smoothing), read/written across calls so a skipped
            frame (run_lane_detection=False) can reuse the last real result
            instead of re-running the model.
        smoother: EMA-smooths the raw path across frames and turns a run of
            misses into a brief fade instead of an instant disappearance
            (see overlay_enhancements.LaneTemporalSmoother). Owned by the
            caller so its state persists across calls -- and is updated
            EVERY frame regardless of visible_layers["lane"], so toggling
            the lane layer on mid-demo shows an already-warmed-up smoothed
            path rather than a cold-started one.
        hud: accumulates per-module status this frame and draws the status
            badges/legend/FPS panel (see overlay_enhancements.PerceptionHUD),
            positioned top-left -- distinct from the existing toggle-state
            legend (draw_layer_hud, top-right), so neither overlaps.
        run_lane_detection: if False, skips the (expensive) lane model
            entirely and reuses cached_lane_path's existing value -- see
            the FPS-optimization note in main() for why this exists.
        visible_layers: dict of {"segmentation"/"objects"/"traffic"/"lane":
            bool} controlling what gets DRAWN this frame. All four models
            still run regardless (see run_lane_detection for the one
            skip-frame exception, which is about compute cadence, not
            visibility) -- only drawing is gated, so toggling a layer on
            mid-demo never causes a stutter from a model cold-starting.
            Defaults to all layers visible if not given, matching the
            original always-combined behaviour.
        fps: current rolling FPS to display in the HUD, or None to omit it.
    """
    if visible_layers is None:
        visible_layers = {"segmentation": True, "objects": True, "traffic": True, "lane": True}

    # 1. Segmentation: translucent color wash as the base layer.
    seg_mask = segmenter.predict(frame_rgb)
    if visible_layers["segmentation"]:
        overlay_rgb = segmenter.visualize(frame_rgb, seg_mask, alpha=0.35)
    else:
        overlay_rgb = frame_rgb
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    hud.mark("segmentation", "ok")

    # 2. Object detection: general boxes, computed once and cached for
    # traffic_perception to reuse (see make_cached_backend's docstring).
    # Always computed (traffic detection depends on the cache below even
    # when the general-object boxes themselves aren't drawn this frame).
    detections = detector.detect(frame_rgb)
    shared_detection_state["detections"] = detections
    if visible_layers["objects"]:
        overlay_bgr = draw_object_detections(overlay_bgr, detections)
    hud.mark("detection", "ok")

    # 3. Traffic detection: reuses the cache from step 2, adds distance info.
    traffic_detections = traffic_perception.detect(frame_rgb)
    if visible_layers["traffic"]:
        overlay_bgr = draw_traffic_detections(overlay_bgr, traffic_detections)
    hud.mark("traffic", "ok")

    # 4. Lane detection: boundary + centerline lines drawn last, on top,
    # so they stay legible over the segmentation color wash underneath.
    # The raw path is smoothed (and, on a miss, faded rather than dropped)
    # before drawing -- see overlay_enhancements module docstring for why.
    if run_lane_detection:
        lane_mask = lane_segmenter.predict(frame_rgb)
        cached_lane_path["value"] = extract_ego_path(lane_mask)
    smoothed_path, lane_alpha, lane_status = smoother.update(cached_lane_path["value"])
    hud.mark("lane", lane_status)
    if visible_layers["lane"] and smoothed_path is not None:
        lane_drawn_bgr = draw_ego_path(overlay_bgr.copy(), smoothed_path)
        overlay_bgr = blend_overlay(overlay_bgr, lane_drawn_bgr, lane_alpha)

    overlay_bgr = draw_layer_hud(overlay_bgr, visible_layers)
    overlay_bgr = hud.draw(overlay_bgr, fps=fps)

    return overlay_bgr


# ADDED: for a calmer, more legible showcase drive -- picks a spawn point
# on a long, low-turn stretch of road and constrains the autopilot's
# route to it, purely cosmetic (doesn't touch perception or control).
STRAIGHT_PATH_DISTANCE_M = 250.0
STRAIGHT_PATH_STEP_M = 10.0


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference b - a, wrapped to (-180, 180]."""
    d = (b - a) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def measure_straightness(start_waypoint, distance_m=STRAIGHT_PATH_DISTANCE_M, step_m=STRAIGHT_PATH_STEP_M):
    """Walk forward from start_waypoint for distance_m, returning
    (total_heading_change_deg, junction_count, path_waypoints).
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
    """Rank spawn points by how straight/turn-free the road ahead is,
    best (straightest, fewest junctions) first. Junction count dominates
    the score -- each junction is a point where autopilot might choose to
    turn, which matters more for "fewer turns" than gentle road curvature.
    """
    scored = []
    for sp in spawn_points:
        wp = carla_map.get_waypoint(sp.location)
        total_change, junction_count, path = measure_straightness(wp, distance_m)
        score = junction_count * 1000 + total_change
        scored.append((score, sp, path))
    scored.sort(key=lambda item: item[0])
    return scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--town", default=None,
        help="Load this town instead of using whatever CarlaUE4 already has "
             "open (e.g. --town Town01). See demo_fused_lane_following.py's "
             "--town help for why a simple single-lane-per-direction town "
             "avoids the ego-lane heuristic's multi-lane ambiguity.",
    )
    add_segmentation_args(parser)  # ADDED (A2)
    args = parser.parse_args()

    # CHANGED (A2): check weights and load every model BEFORE spawning the car
    # and sensors, so a model failure exits cleanly instead of leaving actors
    # in the world.
    for weights_path in required_weight_files(args, LANE_WEIGHTS):
        require_real_weights(weights_path)
    print("Loading models (this can take a moment)...")
    # CHANGED (A2): segmentation backend is selectable (--seg-backend); the
    # default is still the A1 BiSeNetV2 with the same weights and input size.
    segmenter = build_demo_segmenter(args)
    window_title = f"Combined perception overlay [{segmenter.name}]"
    detector = ObjectDetector(
        PerceptionConfig(
            model_name="yolo26n.pt",
            confidence_threshold=0.20,
        )
    )
    lane_segmenter = LaneSegmenter(checkpoint_path=LANE_WEIGHTS)
    shared_detection_state = {"detections": ()}
    traffic_perception = TrafficPerception(
        make_cached_backend(shared_detection_state),
        confidence_threshold=0.4,
        object_widths_m=OBJECT_WIDTHS_M,
    )
    print("Models loaded. Connecting to CARLA and spawning the car...")

    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    world = client.load_world(args.town) if args.town else client.get_world()

    for actor in world.get_actors().filter("vehicle.*"):
        actor.destroy()
    for actor in world.get_actors().filter("sensor.*"):
        actor.destroy()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.*")[0]
    spawn_points = world.get_map().get_spawn_points()

    # CHANGED: rank spawn points by how straight/turn-free the road ahead
    # is (see rank_straight_spawn_candidates above) instead of just taking
    # the first spawnable point, for a calmer showcase drive. Still tries
    # candidates in ranked order and falls through on a blocked spawn,
    # same robustness as the original plain loop.
    ranked_candidates = rank_straight_spawn_candidates(world.get_map(), spawn_points)

    vehicle = None
    chosen_path = None
    for _, sp, path in ranked_candidates:
        try:
            vehicle = world.spawn_actor(vehicle_bp, sp)
            chosen_path = path
            break
        except RuntimeError:
            continue
    if vehicle is None:
        raise RuntimeError("No free spawn point found")

    traffic_manager = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
    vehicle.set_autopilot(True, traffic_manager.get_port())

    # ADDED: constrain the route to the straight stretch chosen above.
    # Best-effort -- set_path()'s exact availability/behaviour isn't
    # guaranteed identical across every CARLA build, so a failure here
    # just falls back to normal (possibly turn-taking) autopilot routing
    # rather than crashing the demo.
    try:
        traffic_manager.set_path(vehicle, [wp.transform.location for wp in chosen_path])
    except Exception as exc:
        print(f"[WARNING] Could not force a straight route via traffic_manager.set_path(): {exc}")
        print("Falling back to normal autopilot routing (may still take turns at junctions).")

    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    captured = {}
    processed_frame_id = {"value": None}

    def on_frame(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        bgr = array[:, :, :3]
        rgb = bgr[:, :, ::-1]
        captured["frame"] = np.ascontiguousarray(rgb)
        captured["frame_id"] = image.frame

    camera.listen(on_frame)

    print("Keys: 1=segmentation 2=objects 3=traffic 4=lane 0=toggle all  q=quit")

    # CHANGED: start with everything OFF -- the presenter toggles each
    # layer on progressively during the walkthrough, rather than starting
    # combined and toggling down.
    visible_layers = {"segmentation": False, "objects": False, "traffic": False, "lane": False}

    frame_times: deque[float] = deque(maxlen=FPS_LOG_INTERVAL)
    frames_processed = 0
    # FIXED (Mohammad): lane detection (ResNet18 + U-Net decoder) is
    # almost certainly the heaviest of the four models -- the only one not
    # specifically designed for real-time use. Combined FPS was sitting
    # right at the 15 FPS target with no margin; lane geometry doesn't
    # meaningfully change frame-to-frame at video rate, so running it every
    # other frame (and reusing the previous result in between) trades an
    # imperceptible lag in the drawn lane lines for real FPS headroom.
    cached_lane_path = {"value": None}

    # ADDED (showcase polish): EMA-smooths the lane path across frames and
    # turns a run of misses into a brief fade rather than an instant
    # disappearance; the HUD gives a persistent, unobtrusive per-module
    # status/legend/FPS panel instead of a console-only FPS print. See
    # overlay_enhancements.py.
    smoother = LaneTemporalSmoother()
    hud = PerceptionHUD(
        legend=(
            ("General object", GENERAL_BOX_COLOR_BGR),
            ("Traffic light/sign", TRAFFIC_BOX_COLOR_BGR),
        )
    )

    try:
        while True:
            if "frame" in captured and captured.get("frame_id") != processed_frame_id["value"]:
                frame_start = time.time()
                frame_rgb = captured["frame"]
                processed_frame_id["value"] = captured["frame_id"]

                current_fps = (len(frame_times) / sum(frame_times)) if frame_times else None
                overlay_bgr = build_combined_overlay(
                    frame_rgb, segmenter, detector, lane_segmenter,
                    traffic_perception, shared_detection_state, cached_lane_path,
                    smoother, hud,
                    run_lane_detection=(frames_processed % 2 == 0),
                    visible_layers=visible_layers,
                    fps=current_fps,
                )

                frames_processed += 1
                frame_times.append(time.time() - frame_start)
                if frames_processed % FPS_LOG_INTERVAL == 0 and frame_times:
                    print(
                        f"[{frames_processed} frames] combined pipeline FPS: {current_fps:.1f}"
                        f"  {segmenter.timing_text()}"  # ADDED (A2)
                    )

                cv2.imshow(window_title, overlay_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in LAYER_KEYS:
                    layer = LAYER_KEYS[key]
                    visible_layers[layer] = not visible_layers[layer]
                    print(f"Layer '{layer}' {'ON' if visible_layers[layer] else 'OFF'}")
                elif key == ALL_LAYERS_KEY:
                    # Toggle all together: if any are currently on, this
                    # turns everything off (a clean blank base frame to
                    # build back up from); if all are already off, this
                    # turns everything back on (the "combined" finale) --
                    # matching how a single all/none switch reads to an
                    # audience, rather than depending on per-layer history.
                    turn_on = not any(visible_layers.values())
                    for layer in visible_layers:
                        visible_layers[layer] = turn_on
                    print(f"All layers {'ON' if turn_on else 'OFF'}")

            time.sleep(0.001)
    finally:
        cv2.destroyAllWindows()
        camera.stop()
        camera.destroy()
        vehicle.destroy()


if __name__ == "__main__":
    main()