r"""Run a minimal CARLA path-following demonstration.

Start CARLA first, then run:
    $env:PYTHONPATH = "$PWD\src"
    python examples/run_carla_demo.py
"""

import argparse
import math
import sys
import threading
import time
from queue import Empty, Queue

import carla
import cv2
import numpy as np

from autodrive.perception.lane_inference import LaneSegmenter
from autodrive.perception.lane_path import draw_ego_path, extract_ego_path
from carla_adapter import apply_command, carla_vehicle_to_state
from navigation_control.controller import PathFollower
from navigation_control.types import (
    ControllerConfig,
    ControlStatus,
    Waypoint,
)
from perception import ObjectDetector, PerceptionConfig
from segmentation.segmentation import SemanticSegmenter
from traffic_perception import (
    BoundingBox as TrafficBoundingBox,
)
from traffic_perception import (
    ModelPrediction,
    TrafficCategory,
    TrafficPerception,
)


def connect_to_carla(
    host: str, port: int, timeout_s: float
) -> tuple[carla.Client, carla.World]:
    """Connect to a running CARLA server or raise a clear error."""

    client = carla.Client(host, port)
    client.set_timeout(timeout_s)

    try:
        world = client.get_world()
    except RuntimeError as error:
        raise RuntimeError(
            f"Could not connect to CARLA at {host}:{port}. "
            "Start CarlaUE4.exe first, then retry."
        ) from error

    return client, world


def build_route(
    world: carla.World,
    start_location: carla.Location,
    waypoint_spacing_m: float,
    waypoint_count: int,
) -> list[Waypoint]:
    """Build a simple lane-centre route starting from a CARLA spawn point."""

    carla_waypoint = world.get_map().get_waypoint(start_location)
    route = [
        Waypoint(
            x_m=carla_waypoint.transform.location.x,
            y_m=-carla_waypoint.transform.location.y,
        )
    ]

    for _ in range(waypoint_count - 1):
        next_waypoints = carla_waypoint.next(waypoint_spacing_m)
        if not next_waypoints:
            break

        carla_waypoint = next_waypoints[0]
        location = carla_waypoint.transform.location
        route.append(Waypoint(x_m=location.x, y_m=-location.y))

    if len(route) < 2:
        raise RuntimeError("Could not build a route with at least two waypoints.")

    return route


def spawn_vehicle(world: carla.World) -> carla.Vehicle:
    """Spawn one vehicle at the first available CARLA spawn point."""

    blueprints = world.get_blueprint_library().filter("vehicle.tesla.model3")
    if not blueprints:
        blueprints = world.get_blueprint_library().filter("vehicle.*")

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("The current CARLA map has no vehicle spawn points.")

    vehicle = world.try_spawn_actor(blueprints[0], spawn_points[0])
    if vehicle is None:
        raise RuntimeError("Could not spawn the demo vehicle.")

    return vehicle


CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
CAMERA_FOV = 90.0

TRAFFIC_CATEGORIES = {
    "traffic light": TrafficCategory.LIGHT,
    "stop sign": TrafficCategory.SIGN,
}


def spawn_camera(world: carla.World, vehicle: carla.Vehicle):
    """Attach a first-person RGB camera to the vehicle."""

    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(CAMERA_WIDTH))
    blueprint.set_attribute("image_size_y", str(CAMERA_HEIGHT))
    blueprint.set_attribute("fov", str(CAMERA_FOV))

    transform = carla.Transform(
        carla.Location(x=1.5, z=1.6),
        carla.Rotation(pitch=0.0),
    )

    return world.spawn_actor(
        blueprint,
        transform,
        attach_to=vehicle,
    )


def carla_image_to_rgb(image: carla.Image) -> np.ndarray:
    """Convert CARLA BGRA image data into RGB."""

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))

    bgr = array[:, :, :3]
    return bgr[:, :, ::-1].copy()


def make_traffic_backend(shared_detections: dict):
    """Convert general object detections into traffic-perception predictions."""

    def backend(_image):
        predictions = []

        for detection in shared_detections["detections"]:
            category = TRAFFIC_CATEGORIES.get(detection.class_name)

            if category is None:
                continue

            box = detection.bounding_box

            predictions.append(
                ModelPrediction(
                    category=category,
                    label=detection.class_name,
                    confidence=detection.confidence,
                    bounding_box=TrafficBoundingBox(
                        x_min=box.x_min,
                        y_min=box.y_min,
                        x_max=box.x_max,
                        y_max=box.y_max,
                    ),
                )
            )

        return predictions

    return backend


def build_perception_overlay(
    frame_rgb,
    detector,
    traffic_perception,
    segmenter,
    lane_segmenter,
    shared_detections,
):
    """Run all perception modules and draw their output."""

    # Semantic segmentation
    semantic_mask = segmenter.predict(frame_rgb)
    overlay_rgb = segmenter.visualize(
        frame_rgb,
        semantic_mask,
        alpha=0.30,
    )

    overlay_bgr = cv2.cvtColor(
        overlay_rgb,
        cv2.COLOR_RGB2BGR,
    )

    # Object detection and tracking
    detections = detector.track(frame_rgb)
    shared_detections["detections"] = detections

    for detection in detections:
        box = detection.bounding_box

        p1 = (int(box.x_min), int(box.y_min))
        p2 = (int(box.x_max), int(box.y_max))

        cv2.rectangle(
            overlay_bgr,
            p1,
            p2,
            (255, 200, 0),
            2,
        )

        track_text = (
            f" ID:{detection.track_id}" if detection.track_id is not None else ""
        )

        cv2.putText(
            overlay_bgr,
            f"{detection.class_name} {detection.confidence:.2f}{track_text}",
            (p1[0], max(20, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            2,
        )

    # Traffic perception
    traffic_detections = traffic_perception.detect(frame_rgb)

    for detection in traffic_detections:
        box = detection.bounding_box

        p1 = (int(box.x_min), int(box.y_min))
        p2 = (int(box.x_max), int(box.y_max))

        cv2.rectangle(
            overlay_bgr,
            p1,
            p2,
            (0, 140, 255),
            3,
        )

        distance = (
            f" {detection.distance_m:.1f}m" if detection.distance_m is not None else ""
        )

        cv2.putText(
            overlay_bgr,
            f"{detection.category.value}: {detection.label}{distance}",
            (p1[0], min(p2[1] + 20, CAMERA_HEIGHT - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 140, 255),
            2,
        )

    # Lane perception
    frame_bgr = cv2.cvtColor(
        frame_rgb,
        cv2.COLOR_RGB2BGR,
    )

    lane_mask = lane_segmenter.predict(frame_bgr)
    lane_path = extract_ego_path(lane_mask)

    if lane_path is not None:
        overlay_bgr = draw_ego_path(
            overlay_bgr,
            lane_path,
        )

    return overlay_bgr


def perception_worker(
    frame_queue,
    overlay_queue,
    stop_event,
    detector,
    traffic_perception,
    segmenter,
    lane_segmenter,
    shared_detections,
):
    """Process the newest camera frame without blocking vehicle control."""

    while not stop_event.is_set():
        try:
            frame_rgb = frame_queue.get(timeout=0.1)
        except Empty:
            continue

        # Always prefer the newest available frame.
        while True:
            try:
                frame_rgb = frame_queue.get_nowait()
            except Empty:
                break

        overlay = build_perception_overlay(
            frame_rgb,
            detector,
            traffic_perception,
            segmenter,
            lane_segmenter,
            shared_detections,
        )

        if overlay_queue.full():
            try:
                overlay_queue.get_nowait()
            except Empty:
                pass

        overlay_queue.put_nowait(overlay)


def run_demo(args: argparse.Namespace) -> None:
    """Run the CARLA path-following control loop."""

    _, world = connect_to_carla(args.host, args.port, args.timeout)

    print("Loading perception models...")

    detector = ObjectDetector(
        PerceptionConfig(
            model_name="yolo26n.pt",
            confidence_threshold=0.20,
        )
    )

    segmenter = SemanticSegmenter("segmentation_model.pth")

    lane_segmenter = LaneSegmenter("lane_unet_best.pth")

    shared_detections = {
        "detections": (),
    }

    focal_length_px = CAMERA_WIDTH / (2.0 * math.tan(math.radians(CAMERA_FOV) / 2.0))

    traffic_perception = TrafficPerception(
        backend=make_traffic_backend(shared_detections),
        confidence_threshold=0.4,
        focal_length_px=focal_length_px,
        object_widths_m={
            "stop sign": 0.75,
        },
    )

    print("Perception models loaded.")

    overlay_queue = Queue(maxsize=1)
    stop_event = threading.Event()

    vehicle = None
    camera = None
    perception_thread = None
    try:
        vehicle = spawn_vehicle(world)

        # Wait for CARLA to update the newly spawned actor state.
        world.wait_for_tick()

        camera = spawn_camera(world, vehicle)

        frame_queue = Queue(maxsize=1)

        def receive_frame(image):
            frame = carla_image_to_rgb(image)

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass

            frame_queue.put_nowait(frame)

        camera.listen(receive_frame)

        perception_thread = threading.Thread(
            target=perception_worker,
            args=(
                frame_queue,
                overlay_queue,
                stop_event,
                detector,
                traffic_perception,
                segmenter,
                lane_segmenter,
                shared_detections,
            ),
            daemon=True,
        )

        perception_thread.start()

        route = build_route(
            world,
            vehicle.get_location(),
            args.waypoint_spacing,
            args.waypoint_count,
        )

        config = ControllerConfig(
            wheelbase_m=2.875,
            max_steering_angle_rad=math.radians(35.0),
            lookahead_m=5.0,
            kp=0.6,
            ki=0.02,
            stopping_deceleration_mps2=1.5,
        )
        follower = PathFollower(config)
        follower.set_path(route)

        print(
            f"Demo started: {len(route)} waypoints, "
            f"target speed {args.target_speed_mps:.1f} m/s."
        )

        previous_time = time.monotonic()
        end_time = previous_time + args.max_duration_s

        while time.monotonic() < end_time:
            now = time.monotonic()
            dt_s = max(0.01, min(now - previous_time, 0.2))
            previous_time = now

            state = carla_vehicle_to_state(
                vehicle,
                rear_axle_offset_m=args.rear_axle_offset_m,
            )

            result = follower.update(
                state=state,
                target_speed_mps=args.target_speed_mps,
                dt_s=dt_s,
            )

            apply_command(
                vehicle,
                result.command,
                config.max_steering_angle_rad,
            )

            try:
                overlay = overlay_queue.get_nowait()

                cv2.putText(
                    overlay,
                    f"Speed: {state.speed_mps:.1f} m/s",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    overlay,
                    (
                        f"Steer: {result.command.steering_angle_rad:.2f}  "
                        f"Throttle: {result.command.throttle:.2f}  "
                        f"Brake: {result.command.brake:.2f}"
                    ),
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(
                    "Group 7 - Autonomous Perception and Navigation",
                    overlay,
                )

            except Empty:
                pass

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Demo stopped by user.")
                return

            if result.status == ControlStatus.FINISHED:
                print("Demo finished: route goal reached.")
                return

            time.sleep(0.05)

        print("Demo stopped: maximum duration reached.")

    finally:
        if stop_event is not None:
            stop_event.set()

        if perception_thread is not None:
            perception_thread.join(timeout=2.0)

        if camera is not None:
            camera.stop()
            camera.destroy()

        cv2.destroyAllWindows()

        if vehicle is not None:
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
            time.sleep(0.5)
            vehicle.destroy()


def parse_arguments() -> argparse.Namespace:
    """Read the demo configuration from the command line."""

    parser = argparse.ArgumentParser(description="Run the CARLA navigation demo.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--target-speed-mps", type=float, default=5.0)
    parser.add_argument("--waypoint-spacing", type=float, default=2.0)
    parser.add_argument("--waypoint-count", type=int, default=40)
    parser.add_argument("--max-duration-s", type=float, default=60.0)
    parser.add_argument("--rear-axle-offset-m", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_demo(parse_arguments())
    except RuntimeError as error:
        print(f"CARLA demo error: {error}", file=sys.stderr)
        raise SystemExit(1)

