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
from mapless.detections import (
    DetectionsConfig,
    ImageBox,
    cluster_measurement,
    cluster_obstacles,
    join,
    plan_extent,
    split_oversized,
)
from mapless.driver import DriveMode, DriverConfig, DriverStep
from mapless.ego_motion import Pose2D
from mapless.frame import CameraImage, RadarReturns, SensorFrame, SensorState, SensorStatus
from mapless.geometry import camera_calibration
from mapless.painting import PaintedPoints, PaintingConfig, SegmentedFrame, paint
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
from mapless.tracking import LidarMeasurement
from navigation_control.local import LocalControlConfig
from navigation_control.types import ControlCommand, ControllerConfig
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
    for _ in range(int(30 / DT)):
        step = stack.step(with_camera_and_radar(world.frame(), world, car_world))
        steps.append((world.truth(), step))
        world.apply(step.command)
    modes = {s.mode for _, s in steps}
    assert DriveMode.DRIVING in modes
    last_truth, last = steps[-1]
    assert last.tracks, "the parked car should be tracked"
    confirmed = [t for t in last.tracks if t.status.value == "confirmed"]
    assert confirmed
    # The parked car is tracked from the LiDAR now (support expires 0.5 s after a sensor's
    # last update). In this world its side splits into short clusters, each a track, and
    # the radar keeps a separate track on its bumper return: a known tracking limitation.
    near = [t for t in confirmed if abs(t.position[0] - (car_world[0] - 15.22)) < 3.0]
    assert any("lidar" in t.support for t in near)
    assert last.road_model is not None and last.road_model.visibility_m > 10.0
    assert last.decision.executed in ("FOLLOW", "WAIT", "STOP") and "drive" in last.decision.timing_ms
    assert len(stack.records) == len(steps)
    # The car stopped short of the parked car on the driver's own obstacle stop and holds
    # there, inside the live gate's 1.5-5 m (first live stack stop run: the predicted
    # conflict with the parked lead stopped it 6.3 m short, before the driver's stop point,
    # so it never held).
    assert not any(t.collision for t, _ in steps)
    assert 1.5 <= last_truth.front_gap_m <= 5.0, last_truth.front_gap_m
    assert last.mode is DriveMode.STOPPING and last.reason.startswith("holding for obstacle"), last.reason
    # No brake taps on the way: a fresh tentative cluster (initial velocity variance 5 m/s,
    # inflated to a 30 m box at 3 s) capped the speed to zero for a few ticks every time
    # one appeared 20-30 m out (fresh-eyes review of 58adaf3). Only confirmed and coasting
    # tracks are predicted onto the path; tentative ones are the LiDAR obstacle stop's job.
    taps = [(t.front_gap_m, s.diagnostics.get("speed_cap_next_mps")) for t, s in steps if t.front_gap_m > 15.0 and s.diagnostics.get("speed_cap_next_mps") is not None and s.diagnostics["speed_cap_next_mps"] <= 0.0]
    assert not taps, taps[:5]


def test_painting_refuses_another_frames_segmentation():
    calibration = forward_camera()
    image = grey("front", calibration)
    seg = ConstantSegmenter(0.5, 0.6).segment(image)
    tagged = SegmentedFrame(7, 0.35, [seg])
    pts = np.array([[10.0, 0.0, 0.0]])
    assert paint(pts, [image], tagged, PaintingConfig(), frame_id=7).painted[0]
    with pytest.raises(ValueError, match="frame 7"):
        paint(pts, [image], tagged, PaintingConfig(), frame_id=8)


def test_stack_paints_only_on_segmentation_ticks_and_records_the_cap_as_next():
    # Review of the G3 framework: the stack segmented every 2nd tick but painted every tick
    # (projecting the current LiDAR into the previous image's segmentation), and the record
    # read as if the cap computed this tick had shaped this tick's command.
    car = ParkedCar(55.0, 1.75)
    scene = RoadScene(KerbLine([Straight(150)]), parked=(car,))
    world = MaplessRoadWorld(scene, SUITE, start_s_m=15.0)
    car_xy = scene.kerb.pose_at(car.s_m)
    h = car_xy[2]
    car_world = (car_xy[0] - car.offset_m * math.sin(h), car_xy[1] + car.offset_m * math.cos(h))
    config = DriverConfig(body=BODY, control=control(), corridor=CorridorConfig(keep_right_m=1.75))
    stack = MaplessStack(SUITE, world.route(70.0), config, StackConfig(admission=Admission.preset("A")), DT)
    adds = []
    previous = None
    for _ in range(int(6 / DT)):
        frame = with_camera_and_radar(world.frame(), world, car_world)
        step = stack.step(frame)
        world.apply(step.command)
        if stack.road_model.time_s is not None and stack.road_model.time_s != previous:
            adds.append((stack._tick, frame.frame_id, stack._segmentations.frame_id))
            previous = stack.road_model.time_s
    assert len(adds) > 5, "the road model should have received semantic evidence"
    assert all(tick % stack.config.segment_every == 0 for tick, _, _ in adds), "evidence added off a segmentation tick"
    assert all(seg_id == frame_id for _, frame_id, seg_id in adds), "painted with another frame's segmentation"
    # The cap computed on one tick is the one applied on the next; the first tick had none.
    records = stack.records
    assert records[0].cap_applied_mps is None
    assert all(r.cap_applied_mps == p.speed_cap_next_mps for p, r in zip(records, records[1:]))
    assert records[-1].as_dict()["frame_generation"] == 0 and "speed_cap_next_mps" in records[-1].as_dict()


def test_stack_clears_its_spatial_state_when_the_driver_rebuilds_the_frame():
    # Review of the G3 framework: after an odometry outage the driver rebuilds grid, corridor
    # and trajectory from a standstill, but tracks, semantic evidence, the following history
    # and the cached segmentation survived in the old frame.
    world, frame, classified = synthetic_sweep()
    config = DriverConfig(body=BODY, control=control(), corridor=CorridorConfig(keep_right_m=1.75))
    stack = MaplessStack(SUITE, world.route(70.0), config, StackConfig(), DT)
    pose = Pose2D(0.0, 0.0, 0.0)
    for k in range(4):
        stack.tracker.update(k * DT, pose, lidar=[LidarMeasurement(np.array([20.0, 0.0]), (4.5, 2.0, 0.0), {"car": 0.9, "unknown": 0.1})])
    probs = np.zeros((len(classified.xyz), len(ROAD_CLASSES)), dtype=np.float32)
    probs[:, ROAD_CLASSES.index("road")] = 1.0
    stack.driver.grid.add(classified, pose, 0.0)
    stack.road_model.add(classified.xyz, PaintedPoints(probs, np.ones(len(probs), dtype=bool), np.ones(len(probs), dtype=int)), pose, 0.0)
    stack.following.target_speed(4.0, 2.0, 20.0, 0.0, DT)
    stack._segmentations = SegmentedFrame(3, 0.15, [])
    stack.driver.speed_cap_mps = 1.0
    assert stack.tracker.tracks and stack.road_model.evidence["road"].sum() > 0 and stack.following._target is not None
    # The driver reports the rebuild in its diagnostics on the tick it happens.
    rebuilt = DriverStep(ControlCommand(0.0, 0.0, 0.6), DriveMode.FAULT, "local frame rebuilt from a standstill", {"frame_rebuilt": True, "rebuilding": True})
    stack.driver.step = lambda f: rebuilt
    step = stack.step(frame)
    assert stack.tracker.tracks == {} and stack.tracker.live() == []
    assert stack.road_model.evidence["road"].sum() == 0.0 and stack.road_model.time_s is None
    assert stack.following._target is None and stack._segmentations is None and stack._model is None
    assert step.decision.frame_generation == 1 and step.diagnostics["frame_generation"] == 1
    assert step.decision.cap_applied_mps == 1.0 and step.decision.speed_cap_next_mps is None
    # While the driver is still stopping to rebuild, nothing is placed in the stale frame.
    stopping = DriverStep(ControlCommand(0.0, 0.0, 0.6), DriveMode.FAULT, "odometry recovered: stopping to rebuild the local frame", {"rebuilding": True})
    stack.driver.step = lambda f: stopping
    before = stack.tracker._next_id
    step = stack.step(frame)
    assert stack.tracker.tracks == {} and stack.tracker._next_id == before and step.tracks == ()
    assert step.decision.frame_generation == 1


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
    assert info["selection"]["in_progress"] is False
    # The best checkpoint is on disk after every improving epoch, marked in progress, so a
    # run that dies keeps it; the final write clears the mark and adds the last epoch.
    saved = []
    import mapless_training.train_segmentation as ts

    original = ts.save_checkpoints

    def spy(*args, **kwargs):
        saved.append(kwargs.get("in_progress", False))
        return original(*args, **kwargs)

    ts.save_checkpoints = spy
    try:
        train(runs, tmp_path / "two.pt", epochs=2, batch=2, size_wh=(64, 32), device="cpu")
    finally:
        ts.save_checkpoints = original
    assert saved[0] is True and saved[-1] is False and (tmp_path / "two_last.pt").exists()
    assert "iou" in info["history"][-1]["validation"]
    segmenter = TorchRoadSegmenter(path, device="cpu")
    image = grey("front", forward_camera(), 64, 32)
    seg = segmenter.segment(image)
    assert seg.probabilities.shape == (32, 64, len(ROAD_CLASSES))
    assert np.allclose(seg.probabilities.sum(axis=2), 1.0, atol=1e-4)
    assert segmenter.provenance["training_towns"] == ["Town03"]
    assert info["validated"] is True and info["seed"] == 0 and info["selection"]["best_epoch"] == 1
    assert path.name == "model.pt" and path.with_name("model_last.pt").exists()
    assert info["history"][-1]["validation"]["timing"]["synchronised"] is False  # CPU: nothing to synchronise
    # A labels folder is never a checkpoint location.
    with pytest.raises(ValueError):
        TorchRoadSegmenter(tmp_path / "labels" / "model.pt")


def test_training_refuses_unaccepted_runs_and_names_smoke_checkpoints(tmp_path):
    # Review: index_runs documented refusing INCOMPLETE runs but accepted them, and a run
    # with no validation split was saved as if it were a trained model.
    from mapless_training.segmentation_data import class_weights, index_runs, weight_samples
    from mapless_training.train_segmentation import train

    incomplete = write_capture_run(tmp_path, "Town03")
    (incomplete / "manifest.json").write_text(json.dumps(dict(map="Carla/Maps/Town03", status="INCOMPLETE", suite_sha256="abc", seed=1)), encoding="utf-8")
    with pytest.raises(ValueError, match="INCOMPLETE"):
        index_runs([incomplete])
    samples, manifests = index_runs([incomplete], allow_unaccepted=True)
    assert manifests[str(incomplete)]["acceptance"] == dict(source="manifest", status="INCOMPLETE", integrity=None, geometry=None, accepted=False, reasons=["run status INCOMPLETE"])
    # The inspection's verdict outranks a COMPLETE manifest.
    rejected = write_capture_run(tmp_path, "Town04")
    (rejected / "inspection").mkdir()
    verdict = dict(status="REJECT", integrity=dict(status="PASS", reasons=[]), geometry=dict(status="FAIL", reasons=["front: ground projection 0.4"]), reasons=["front: ground projection 0.4"])
    (rejected / "inspection" / "summary.json").write_text(json.dumps(dict(acceptance=verdict)), encoding="utf-8")
    with pytest.raises(ValueError, match="ground projection"):
        index_runs([rejected])
    verdict["status"], verdict["geometry"] = "ACCEPT", dict(status="PASS", reasons=[])
    (rejected / "inspection" / "summary.json").write_text(json.dumps(dict(acceptance=verdict)), encoding="utf-8")
    _, manifests = index_runs([rejected])
    assert manifests[str(rejected)]["acceptance"]["source"] == "inspection" and manifests[str(rejected)]["acceptance"]["accepted"]
    # Class-weight images are drawn evenly across runs and cameras, not from the head of the index.
    import cv2

    many = index_runs([write_capture_run(tmp_path / "many", "Town05", n=6), write_capture_run(tmp_path / "few", "Town06", n=2)])[0]
    chosen = weight_samples(many, limit=4, seed=0)
    assert sorted(s.town for s in chosen) == ["Town05", "Town05", "Town06", "Town06"]
    assert [s.frame for s in weight_samples(many, limit=4, seed=0)] == [s.frame for s in chosen]
    assert class_weights(many, (64, 32), cv2, limit=4).shape == (len(ROAD_CLASSES),)
    # No validation split: a smoke checkpoint, named and marked as such, with its seed recorded.
    path, info = train([incomplete], tmp_path / "model.pt", epochs=2, batch=2, size_wh=(64, 32), device="cpu", seed=7, allow_unaccepted=True)
    assert path.name == "model_smoke.pt" and not path.with_name("model_smoke_last.pt").exists()
    assert info["validated"] is False and info["smoke"] is True and info["seed"] == 7 and info["allow_unaccepted"] is True
    assert info["selection"]["validated"] is False and info["selection"]["best_miou"] is None and info["selection"]["best_epoch"] in (1, 2)
    assert TorchRoadSegmenter(path, device="cpu").provenance["validated"] is False


def test_classified_sweep_helper_is_the_terrain_classifier():
    world, frame, classified = synthetic_sweep()
    assert isinstance(classified, ClassifiedSweep) and (classified.cls == PointClass.ROAD).sum() > 1000


def test_a_cluster_straddling_a_range_ring_stays_one_cluster():
    # Review of 36ba0ca: clustering per 10 m ring split a car across r = 20 m into two tracks.
    rng = np.random.default_rng(0)
    pts = np.column_stack((rng.uniform(18.5, 21.5, 120), rng.uniform(-0.9, 0.9, 120), rng.uniform(0.3, 1.2, 120)))
    sweep = ClassifiedSweep(pts, pts[:, 2], np.full(len(pts), PointClass.OBSTACLE), np.zeros(1), np.zeros(1))
    clusters = cluster_obstacles(sweep, DetectionsConfig())
    assert len(clusters) == 1 and len(clusters[0]) == 120
    # Two cars 4 m apart stay apart.
    far = pts + np.array([0.0, 4.0, 0.0])
    both = ClassifiedSweep(np.vstack((pts, far)), np.zeros(240), np.full(240, PointClass.OBSTACLE), np.zeros(1), np.zeros(1))
    assert len(cluster_obstacles(both, DetectionsConfig())) == 2


def test_segformer_b0_trains_and_loads_offline(tmp_path):
    # The pretrained encoder needs the network (the PCs have it); the architecture itself,
    # the training loop and the loader are exercised here from a fresh MiT-B0.
    from mapless_training.train_segmentation import train

    runs = [write_capture_run(tmp_path, "Town04"), write_capture_run(tmp_path, "Town02")]
    path, info = train(runs, tmp_path / "segformer.pt", epochs=1, batch=2, size_wh=(64, 32), device="cpu", architecture="segformer_b0", pretrained=False)
    assert info["architecture"] == "segformer_b0" and info["pretrained_encoder"] is False
    segmenter = TorchRoadSegmenter(path, device="cpu")
    assert segmenter.architecture == "segformer_b0"
    seg = segmenter.segment(grey("front", forward_camera(), 64, 32))
    assert seg.probabilities.shape == (32, 64, len(ROAD_CLASSES)) and np.allclose(seg.probabilities.sum(axis=2), 1.0, atol=1e-4)


def _obstacle_sweep(xyz):
    xyz = np.asarray(xyz, dtype=float)
    return ClassifiedSweep(xyz, xyz[:, 2], np.full(len(xyz), PointClass.OBSTACLE), np.zeros(1), np.zeros(1))


def _in_lane_ahead(sweep, groups, x_max=15.0, half_lane=1.75):
    centres = [cluster_measurement(sweep, m).xy for m in groups]
    return [tuple(np.round(c, 2)) for c in centres if 0.0 < c[0] < x_max and abs(c[1]) < half_lane]


def test_scenery_chained_into_one_cluster_is_never_a_lead():
    """PC 2 braking-lead runs on a Town05 grade (26 Oct): both kerb railings and the road
    rising ahead and behind chained into one 63 x 26 m cluster whose mean sat 2.8 m ahead
    in the lane; tracked as a 12 x 12 m vehicle, it was followed at a negative gap for
    most of both runs. The saved startup sweep's OBSTACLE points, replayed."""
    from autonomy.paths import FIXTURES_DIR

    xyz = np.load(FIXTURES_DIR / "town05_grade_obstacles.npz")["obstacle_cm"].astype(float) / 100.0
    sweep = _obstacle_sweep(xyz)
    config = DetectionsConfig()
    clusters = cluster_obstacles(sweep, config)
    giant = max(clusters, key=len)
    length, width = plan_extent(sweep.xyz[giant, :2])
    assert length > 60 and width > 20  # the chaining itself (single linkage)
    assert _in_lane_ahead(sweep, clusters)  # ... whose centre lands in the lane ahead
    objects, structures = split_oversized(sweep, clusters, config)
    def road_user_sized(m):
        length, width = plan_extent(sweep.xyz[m, :2])
        return length <= config.max_object_length_m and width <= config.max_object_width_m

    # Structure: too big, or a thin line split off it (a railing's stretch).
    assert structures and all(not road_user_sized(m) or plan_extent(sweep.xyz[m, :2])[1] < config.split_min_width_m for m in structures)
    assert all(road_user_sized(m) for m in objects)
    assert _in_lane_ahead(sweep, objects) == []
    # Every OBSTACLE point is still somewhere (objects, structures or a dropped speck):
    # the driver's LiDAR stop reads the classified points, not the clusters.
    kept = sum(len(m) for m in objects) + sum(len(m) for m in structures)
    assert kept >= 0.95 * sum(len(m) for m in clusters)


def test_a_car_beside_a_railing_is_split_out_of_it():
    """A car 0.6 m off a 30 m kerb railing at 10 m range links into the railing's cluster;
    the split measures the car on its own and leaves the railing as structure."""
    rng = np.random.default_rng(5)
    railing = np.column_stack((np.arange(0.0, 30.0, 0.05), np.full(600, -3.0), np.full(600, 0.9)))
    # The car's near face and rear (an L), 1.8 m wide, its right side 0.6 m from the railing.
    side = np.column_stack((rng.uniform(8.0, 12.5, 150), np.full(150, -2.4), rng.uniform(0.3, 1.4, 150)))
    rear = np.column_stack((np.full(80, 8.0), rng.uniform(-2.4, -0.6, 80), rng.uniform(0.3, 1.4, 80)))
    sweep = _obstacle_sweep(np.vstack((railing, side, rear)))
    config = DetectionsConfig()
    clusters = cluster_obstacles(sweep, config)
    assert len(clusters) == 1  # merged at the range-tuned link
    objects, structures = split_oversized(sweep, clusters, config)
    assert len(structures) == 1 and len(structures[0]) == 600
    assert len(objects) == 1 and len(objects[0]) == 230
    assert np.allclose(cluster_measurement(sweep, objects[0]).xy, (9.5, -1.95), atol=0.6)


def test_one_pass_cluster_and_split_matches_the_two_passes():
    from autonomy.paths import FIXTURES_DIR
    from mapless.detections import cluster_and_split

    xyz = np.load(FIXTURES_DIR / "town05_grade_obstacles.npz")["obstacle_cm"].astype(float) / 100.0
    rng = np.random.default_rng(8)
    for sweep in (_obstacle_sweep(xyz), _obstacle_sweep(np.column_stack((rng.uniform(-30, 40, 900), rng.uniform(-12, 12, 900), rng.uniform(0.3, 2, 900))))):
        objects, structures = split_oversized(sweep, cluster_obstacles(sweep))
        one_objects, one_structures = cluster_and_split(sweep)
        key = lambda groups: sorted(tuple(int(v) for v in g) for g in groups)  # noqa: E731
        assert key(one_objects) == key(objects) and key(one_structures) == key(structures)


def test_object_sized_clusters_are_untouched_by_the_split():
    rng = np.random.default_rng(6)
    car = np.column_stack((rng.uniform(18.0, 22.5, 200), rng.uniform(-0.9, 0.9, 200), rng.uniform(0.3, 1.4, 200)))
    walker = np.column_stack((rng.normal(12.0, 0.15, 40), rng.normal(3.0, 0.15, 40), rng.uniform(0.3, 1.7, 40)))
    sweep = _obstacle_sweep(np.vstack((car, walker)))
    clusters = cluster_obstacles(sweep)
    objects, structures = split_oversized(sweep, clusters)
    assert structures == [] and [list(m) for m in objects] == [list(m) for m in clusters]


def test_vectorised_clustering_matches_pairwise_single_linkage():
    """The numpy clustering (26 Oct, 7x faster live) gives the same partition as a
    brute-force pairwise single-linkage on random scenes with range-scaled links."""
    from mapless.detections import _components, _link_pairs

    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(1, 400))
        xy = np.column_stack((rng.uniform(-40, 40, n), rng.uniform(-15, 15, n)))
        link = 0.6 + 0.03 * np.hypot(xy[:, 0], xy[:, 1])
        labels = _components(n, *_link_pairs(xy, link, block=37))
        # Brute force: flood fill over the full pairwise adjacency.
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        adj = d <= np.maximum(link[:, None], link[None, :])
        seen, truth = np.full(n, -1), 0
        for s in range(n):
            if seen[s] >= 0:
                continue
            stack = [s]
            seen[s] = truth
            while stack:
                k = stack.pop()
                for m in np.flatnonzero(adj[k] & (seen < 0)):
                    seen[m] = truth
                    stack.append(m)
            truth += 1
        same = labels[:, None] == labels[None, :]
        assert np.array_equal(same, seen[:, None] == seen[None, :])


def test_speed_cap_without_a_corridor_is_no_cap_not_none():
    """First Town05 run (26 Oct): the stack's first tick had no corridor yet and the speed
    cap returned a bare None, which the caller unpacks; it crashed the run."""
    scene = RoadScene(KerbLine([Straight(150)]))
    world = MaplessRoadWorld(scene, SUITE, start_s_m=15.0)
    config = DriverConfig(body=BODY, control=control(), corridor=CorridorConfig(keep_right_m=1.75))
    stack = MaplessStack(SUITE, world.route(70.0), config, StackConfig(admission=Admission.preset("B")), DT)
    stack.driver.corridor.previous = None
    assert stack._speed_cap(None, None, (), (), 0.0) == (None, "")


def test_a_missed_lidar_sweep_keeps_the_cap_in_force():
    """External review of 405b056: a tick without a sweep left the cap at None, so a stop
    request lapsed for the tick after. The restriction is now held until perception runs
    again and re-evaluates it (one- and two-tick dropouts)."""
    import dataclasses

    from mapless.frame import SensorState, SensorStatus

    scene = RoadScene(KerbLine([Straight(150)]))
    world = MaplessRoadWorld(scene, SUITE, start_s_m=15.0)
    config = DriverConfig(body=BODY, control=control(), corridor=CorridorConfig(keep_right_m=1.75))
    stack = MaplessStack(SUITE, world.route(70.0), config, StackConfig(admission=Admission.preset("A")), DT)
    for _ in range(int(4 / DT)):  # start up and drive
        world.apply(stack.step(world.frame()).command)

    def without_lidar(frame):
        status = [SensorStatus(s.name, s.kind, SensorState.MISSING, None) if s.kind == "lidar" else s for s in frame.status]
        return dataclasses.replace(frame, lidars=(), status=tuple(status))

    for gap in (1, 2):
        stack.driver.speed_cap_mps, stack._cap_reason = 0.0, "predicted conflict on the path"
        for _ in range(gap):
            step = stack.step(without_lidar(world.frame()))
            world.apply(step.command)
            assert stack.driver.speed_cap_mps == 0.0
            assert step.diagnostics["speed_cap_reason"].startswith("predicted conflict on the path")
            assert step.diagnostics["speed_cap_reason"].endswith("(held: no fresh LiDAR)")
        # A fresh sweep re-evaluates: the empty road lifts the cap.
        step = stack.step(world.frame())
        world.apply(step.command)
        assert stack.driver.speed_cap_mps is None
