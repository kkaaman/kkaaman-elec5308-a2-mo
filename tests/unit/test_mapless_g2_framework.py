"""G2 framework: segmentation interface and provenance, LiDAR painting with depth order,
the road model, detections joined to clusters, the stack end to end on the synthetic
world, and the training loop on a two-sample synthetic capture run."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from autonomy.interfaces import CameraCalibration
from mapless.corridor import CorridorConfig
from mapless.detections import DetectionsConfig, ImageBox, cluster_obstacles, join
from mapless.driver import DriveMode, DriverConfig
from mapless.ego_motion import Pose2D
from mapless.frame import CameraImage, RadarReturns, SensorFrame, SensorState, SensorStatus
from mapless.geometry import camera_calibration
from mapless.painting import PaintingConfig, paint
from mapless.road_model import RoadModelBuilder
from mapless.road_segmentation import (
    ROAD_CLASSES,
    ConstantSegmenter,
    Segmentation,
    TorchRoadSegmenter,
    carla_tags_to_classes,
    check_provenance,
)
from mapless.sensor_suite import load_sensor_suite
from mapless.stack import Admission, MaplessStack, StackConfig
from mapless.terrain import (
    ClassifiedSweep,
    PointClass,
    TerrainConfig,
    TerrainGrid,
    VehicleBody,
    classify_sweep,
)
from navigation_control.local import LocalControlConfig
from navigation_control.types import ControllerConfig
from sim_harness.mapless_road import KerbLine, MaplessRoadWorld, ParkedCar, RoadScene, Straight

ROOT = Path(__file__).resolve().parents[2]
SUITE = load_sensor_suite(ROOT / "configs" / "sensor_suites" / "mapless_v2.json")
BODY = VehicleBody()
DT = 0.05


def forward_camera(width=64, height=48, fx=40.0):
    # Optical frame at the rear axle, 1.5 m up, looking along +x: optical +Z = body +x, +X = body -y, +Y = body -z.
    m = np.eye(4)
    m[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    m[:3, 3] = (0.0, 0.0, 1.5)
    return CameraCalibration(fx, fx, width / 2, height / 2, tuple(tuple(r) for r in m))


def grey(name, calibration, width=64, height=48):
    return CameraImage(name, np.full((height, width, 3), 128, dtype=np.uint8), calibration)


# -- segmentation ------------------------------------------------------------------------
def test_constant_segmenter_and_label_mapping():
    image = grey("front", forward_camera())
    seg = ConstantSegmenter(0.5, 0.4).segment(image)
    assert seg.probabilities.shape == (48, 64, len(ROAD_CLASSES))
    assert seg.labels[40, 10] == ROAD_CLASSES.index("other")  # 0.4 road vs 0.6 other
    assert seg.probability("road")[40, 10] == pytest.approx(0.4) and seg.probability("road")[5, 10] == 0.0
    tags = np.array([[1, 24, 2], [14, 12, 99]], dtype=np.uint8)
    assert carla_tags_to_classes(tags).tolist() == [[0, 1, 2], [3, 3, 4]]


def test_provenance_needs_the_attestation_and_the_classes():
    good = dict(training_towns=["Town03", "Town04"], classes=list(ROAD_CLASSES), training_runs=["r"], label_mapping={}, created="x", test_towns_excluded=True)
    assert check_provenance(good) is good
    with pytest.raises(ValueError, match="held-out"):
        check_provenance(good | dict(test_towns_excluded=False))
    with pytest.raises(ValueError, match="held-out"):
        check_provenance({k: v for k, v in good.items() if k != "test_towns_excluded"})
    with pytest.raises(ValueError, match="classes"):
        check_provenance(good | dict(classes=["road"]))


# -- painting ----------------------------------------------------------------------------
def test_painting_follows_depth_order():
    calibration = forward_camera()
    image = grey("front", calibration)
    # Every pixel is "pavement" on the top half and "road" on the bottom half.
    p = np.zeros((48, 64, len(ROAD_CLASSES)), dtype=np.float32)
    p[:24, :, ROAD_CLASSES.index("pavement")] = 1.0
    p[24:, :, ROAD_CLASSES.index("road")] = 1.0
    seg = Segmentation("front", p)
    # Two points on the same ray: one at 10 m on the ground, one at 60 m on the ground
    # project to different rows, so take two on one ray instead: the near one at 10 m and a
    # far one straight behind it (same bearing, same height), which the camera cannot see.
    near = np.array([10.0, 0.0, 0.0])
    far = near * 6.0
    far[2] = near[2] * 6.0 - 1.5 * 5.0  # keep it on the same optical ray through the camera at z = 1.5
    pts = np.array([near, far, [10.0, 0.0, 2.5]])  # the third is above the horizon: pavement row
    painted = paint(pts, [image], [seg], PaintingConfig(bucket_px=4))
    assert painted.painted[0] and not painted.painted[1] and painted.painted[2]
    assert painted.labels[0] == ROAD_CLASSES.index("road")
    assert painted.labels[2] == ROAD_CLASSES.index("pavement")
    assert painted.probabilities[1].sum() == pytest.approx(1.0)  # unpainted: uniform


# -- road model --------------------------------------------------------------------------
def synthetic_sweep(parked=()):
    scene = RoadScene(KerbLine([Straight(150)]), parked=parked)
    world = MaplessRoadWorld(scene, SUITE, start_s_m=15.0, noise=False)
    frame = world.frame()
    sweep = frame.lidar(SUITE.lidars[0].name)
    return world, frame, classify_sweep(sweep.xyz, BODY, TerrainConfig())


def test_road_model_builds_layers_lane_candidates_and_stop_lines():
    world, frame, classified = synthetic_sweep()
    grid = TerrainGrid()
    pose = Pose2D(0.0, 0.0, 0.0)
    grid.add(classified, pose, 0.0)
    builder = RoadModelBuilder(grid)
    # Paint: road points marked "road"; a strip beside the car "marking" along x, and a
    # transverse "marking" run 12 m ahead (a stop line).
    probs = np.zeros((len(classified.xyz), len(ROAD_CLASSES)), dtype=np.float32)
    probs[:, ROAD_CLASSES.index("road")] = 1.0
    x, y = classified.xyz[:, 0], classified.xyz[:, 1]
    marking = (np.abs(y - 1.6) < 0.15) & (x > 0) & (x < 25) | (np.abs(x - 12.0) < 0.3) & (np.abs(y) < 1.5)
    probs[marking] = 0.0
    probs[marking, ROAD_CLASSES.index("marking")] = 1.0
    from mapless.painting import PaintedPoints

    painted = PaintedPoints(probs, np.ones(len(probs), dtype=bool), np.ones(len(probs), dtype=int))
    builder.add(classified.xyz, painted, pose, 0.0)
    model = builder.build(pose, 0.0)
    assert set(model.layers) >= set(model.REQUIRED)
    assert model.layers["free"].sum() > 100 and model.layers["road"].sum() > 0
    assert model.visibility_m > 15.0
    assert model.lane_candidates, "the marking strip should give a lane candidate"
    line, score = max(model.lane_candidates, key=lambda c: c[1])
    assert abs(np.median(line[:, 1]) - 1.6) < 0.4 and score > 0
    assert model.stop_lines and abs(model.stop_lines[0][0] - 12.0) < 1.0
    # Stale semantics vanish: build 10 s later with nothing added.
    later = builder.build(pose, 10.0)
    assert later.layers["marking"].sum() == 0.0


# -- detections --------------------------------------------------------------------------
def test_clusters_and_boxes_join_on_the_foreground():
    car = ParkedCar(45.0, 1.75)
    world, frame, classified = synthetic_sweep(parked=(car,))
    clusters = cluster_obstacles(classified, DetectionsConfig())
    assert clusters, "the parked car's points should form a cluster"
    centres = [classified.xyz[m, :2].mean(axis=0) for m in clusters]
    nearest = min(range(len(centres)), key=lambda k: abs(centres[k][0] - 30.0))
    assert abs(centres[nearest][0] - 30.0) < 3.0  # 45 m along the kerb, car at 15 m: 30 m ahead
    calibration = camera_calibration(SUITE.cameras[0], 1.38641)
    image = grey("front", calibration, SUITE.cameras[0].width, SUITE.cameras[0].height)
    # A box around the projected cluster.
    from mapless.painting import project

    u, v, depth, visible = project(classified.xyz[clusters[nearest]], calibration, image.rgb.shape[1], image.rgb.shape[0], PaintingConfig())
    assert visible.any()
    box = ImageBox("front", "car", 0.9, u[visible].min() - 2, v[visible].min() - 2, u[visible].max() + 2, v[visible].max() + 2)
    measurements, camera_only = join(classified, clusters, [image], {"front": [box]})
    classed = [m for m in measurements if m.class_probabilities is not None]
    assert classed and max(classed[0].class_probabilities, key=classed[0].class_probabilities.get) == "car"
    assert classed[0].extent is not None and 1.0 < classed[0].extent[0] < 7.0  # the rear face, seen from behind
    # A box nothing matches is camera-only class evidence with a bearing hint.
    _, camera_only = join(classified, clusters, [image], {"front": [ImageBox("front", "person", 0.8, 0, 0, 10, 10)]})
    assert len(camera_only) == 1 and camera_only[0][1]["pedestrian"] > 0.5


# -- stack -------------------------------------------------------------------------------
def control():
    return LocalControlConfig(
        ControllerConfig(wheelbase_m=3.00465, max_steering_angle_rad=math.radians(20), lookahead_m=4.0, kp=0.45, ki=0.15, stopping_deceleration_mps2=1.0, hold_brake=0.5)
    )


def with_camera_and_radar(frame, world, car_xy_world):
    """The synthetic world has no cameras or radar: add a grey front camera and a front radar
    seeing the parked car (range, bearing, elevation, range rate from the ego's own speed)."""
    cam_spec = SUITE.cameras[0]
    calibration = camera_calibration(cam_spec, world.rear_axle_offset_m)
    image = CameraImage(cam_spec.name, np.full((cam_spec.height, cam_spec.width, 3), 100, dtype=np.uint8), calibration)
    st = world.state
    c, s = math.cos(st.yaw), math.sin(st.yaw)
    dx, dy = car_xy_world[0] - st.x, car_xy_world[1] - st.y
    body = np.array([c * dx + s * dy, -s * dx + c * dy])
    radar_spec = next(r for r in SUITE.radars if "front" in r.name)
    sensor_to_ego = np.eye(4)
    sensor_to_ego[:3, 3] = (radar_spec.mount.x_m, radar_spec.mount.y_m, radar_spec.mount.z_m)
    rel = body - sensor_to_ego[:2, 3]
    r = float(np.hypot(*rel))
    rate = -st.speed * rel[0] / r  # static car seen from a moving ego
    returns = np.array([[r + d, math.atan2(rel[1], rel[0]) + a, 0.0, rate] for d, a in ((0.0, 0.0), (0.4, 0.01), (-0.3, -0.01))], dtype=np.float32)
    radar = RadarReturns(radar_spec.name, returns, sensor_to_ego)
    status = [s_ for s_ in frame.status if s_.name not in (cam_spec.name, radar_spec.name)]
    status += [SensorStatus(cam_spec.name, "camera", SensorState.OK, frame.sim_time_s), SensorStatus(radar_spec.name, "radar", SensorState.OK, frame.sim_time_s)]
    return SensorFrame(
        frame.episode_id,
        frame.frame_id,
        frame.sim_time_s,
        frame.received_monotonic_s,
        (image,),
        frame.lidars,
        (radar,),
        frame.gnss,
        frame.imu,
        frame.speed_mps,
        tuple(status),
        frame.calibration_id,
    )


def test_stack_runs_end_to_end_and_tracks_the_parked_car():
    car = ParkedCar(55.0, 1.75)
    scene = RoadScene(KerbLine([Straight(150)]), parked=(car,))
    world = MaplessRoadWorld(scene, SUITE, start_s_m=15.0)
    car_xy = scene.kerb.pose_at(car.s_m)
    h = car_xy[2]
    car_world = (car_xy[0] - car.offset_m * math.sin(h), car_xy[1] + car.offset_m * math.cos(h))
    config = DriverConfig(body=BODY, control=control(), corridor=CorridorConfig(keep_right_m=1.75))
    stack = MaplessStack(SUITE, world.route(70.0), config, StackConfig(admission=Admission.preset("C")), DT)
    steps = []
    for _ in range(int(14 / DT)):
        step = stack.step(with_camera_and_radar(world.frame(), world, car_world))
        steps.append((world.truth(), step))
        world.apply(step.command)
    modes = {s.mode for _, s in steps}
    assert DriveMode.DRIVING in modes
    last_truth, last = steps[-1]
    assert last.tracks, "the parked car should be tracked"
    confirmed = [t for t in last.tracks if t.status.value == "confirmed"]
    assert confirmed
    track = min(confirmed, key=lambda t: abs(t.position[0] - (car_world[0] - 15.22)))
    assert "lidar" in track.support
    assert last.road_model is not None and last.road_model.visibility_m > 10.0
    assert last.decision.executed in ("FOLLOW", "WAIT", "STOP") and "drive" in last.decision.timing_ms
    assert len(stack.records) == len(steps)
    # The car stopped short of the parked car (the driver's own obstacle stop, or following).
    assert not any(t.collision for t, _ in steps)
    assert last_truth.front_gap_m > 1.5


def test_admission_presets():
    assert Admission.preset("A") == Admission(True, True, False, False)
    assert Admission.preset("B").front_radar and not Admission.preset("B").rear_radars
    assert Admission.preset("c").rear_radars


# -- training loop on a synthetic capture --------------------------------------------------
def write_capture_run(root, town, n=2, width=64, height=32):
    import cv2

    run = root / f"capture_{town}"
    (run / "rgb" / "front").mkdir(parents=True)
    (run / "labels" / "semantic" / "front").mkdir(parents=True)
    rows = []
    for k in range(n):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[height // 2 :] = 90
        tags = np.zeros((height, width), dtype=np.uint8)
        tags[height // 2 :] = 1  # road
        tags[: height // 2] = 2  # sidewalk
        tags[height // 2 + 2, :] = 24  # a road line
        cv2.imwrite(str(run / "rgb" / "front" / f"{k:08d}.jpg"), rgb)
        cv2.imwrite(str(run / "labels" / "semantic" / "front" / f"{k:08d}.png"), tags)
        rows.append(dict(frame=k, files=dict(rgb={"front": f"rgb/front/{k:08d}.jpg"}, semantic={"front": f"labels/semantic/front/{k:08d}.png"})))
    (run / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps(dict(map=f"Carla/Maps/{town}", status="COMPLETE", suite_sha256="abc", seed=1)), encoding="utf-8")
    return run


def test_training_loop_writes_a_loadable_checkpoint(tmp_path):
    from mapless_training.segmentation_data import index_runs, split
    from mapless_training.train_segmentation import train

    runs = [write_capture_run(tmp_path, "Town03"), write_capture_run(tmp_path, "Town02")]
    samples, manifests = index_runs(runs)
    train_samples, validation_samples, other = split(samples)
    assert len(train_samples) == 2 and len(validation_samples) == 2 and other == []
    with pytest.raises(ValueError, match="test-only"):
        index_runs([write_capture_run(tmp_path, "Town07")])
    path, info = train(runs, tmp_path / "model.pt", epochs=1, batch=2, size_wh=(64, 32), device="cpu")
    assert path.exists() and path.with_suffix(".provenance.json").exists()
    assert info["training_towns"] == ["Town03"] and info["validation_towns"] == ["Town02"]
    assert "iou" in info["history"][-1]["validation"]
    segmenter = TorchRoadSegmenter(path, device="cpu")
    image = grey("front", forward_camera(), 64, 32)
    seg = segmenter.segment(image)
    assert seg.probabilities.shape == (32, 64, len(ROAD_CLASSES))
    assert np.allclose(seg.probabilities.sum(axis=2), 1.0, atol=1e-4)
    assert segmenter.provenance["training_towns"] == ["Town03"]
    # A labels folder is never a checkpoint location.
    with pytest.raises(ValueError):
        TorchRoadSegmenter(tmp_path / "labels" / "model.pt")


def test_classified_sweep_helper_is_the_terrain_classifier():
    world, frame, classified = synthetic_sweep()
    assert isinstance(classified, ClassifiedSweep) and (classified.cls == PointClass.ROAD).sum() > 1000
