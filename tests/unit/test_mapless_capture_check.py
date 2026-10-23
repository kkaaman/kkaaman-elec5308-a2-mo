"""Capture map policy and the rig check's evaluator-side geometry (no CARLA server)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from carla_adapter.mapless_capture import capture_map_allowed, main, to_ego_frame, truth_record
from carla_adapter.sensor_rig_check import rear_axle_pose, world_to_ego
from evaluation.rig_calibration import box_corners, inside_box, project
from mapless.geometry import camera_calibration
from mapless.sensor_suite import CameraSpec, Mount


def test_capture_refuses_the_test_maps_before_touching_carla(tmp_path):
    assert capture_map_allowed("Carla/Maps/Town03") and capture_map_allowed("Town05_Opt")
    for name in ("Town07", "Town07_Opt", "Town10HD", "Town10HD_Opt", "Town12", "Town15"):
        assert not capture_map_allowed(name)
    for name in ("Town10HD_Opt", "Town07"):
        with pytest.raises(SystemExit):
            main(["--map", name, "--output", str(tmp_path / "x")])
    assert not (tmp_path / "x").exists()


class _V:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def _fake_actor(id_, x, y, yaw_deg, box_dx, extent, alive=True):
    """A CARLA-like actor whose bounding box sits ``box_dx`` ahead of its origin."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    ex, ey, ez = extent

    def vertices(transform):
        cx, cy = x + c * box_dx, y + s * box_dx
        out = []
        for dz in (0.0, 2 * ez):
            for u, v in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                lx, ly = u * ex, v * ey
                out.append(_V(cx + c * lx - s * ly, cy + s * lx + c * ly, dz))
        return out

    box = SimpleNamespace(location=_V(box_dx, 0.0, ez), extent=_V(ex, ey, ez), get_world_vertices=vertices)
    return SimpleNamespace(
        id=id_,
        type_id="vehicle.fake",
        is_alive=alive,
        bounding_box=box,
        get_transform=lambda: SimpleNamespace(location=_V(x, y, 0.0), rotation=SimpleNamespace(yaw=yaw_deg)),
        get_velocity=lambda: _V(0.0, 0.0, 0.0),
        get_angular_velocity=lambda: _V(0.0, 0.0, 0.0),
        get_control=lambda: SimpleNamespace(throttle=0.0, steer=0.0, brake=0.0),
        get_traffic_light_state=lambda: "TrafficLightState.Green",
        is_at_traffic_light=lambda: False,
    )


def test_truth_boxes_are_placed_from_their_world_vertices():
    # Ego at the CARLA origin facing +x (CARLA y is mirrored into the core frame).
    ego = _fake_actor(1, 0.0, 0.0, 0.0, 0.0, (2.4, 1.0, 0.75))
    waypoint = SimpleNamespace(
        road_id=1, section_id=0, lane_id=-1, is_junction=False, lane_width=3.5,
        left_lane_marking=SimpleNamespace(type="Broken"), right_lane_marking=SimpleNamespace(type="Solid"), lane_change="Both",
    )
    carla_map = SimpleNamespace(get_waypoint=lambda loc: waypoint)
    # A car 20 m ahead, 3 m to the CARLA right (core left is -y in CARLA => core y = +3), turned 90 deg,
    # whose bounding box sits 0.5 m ahead of its origin.
    other = _fake_actor(2, 20.0, -3.0, -90.0, 0.5, (2.0, 0.9, 0.7))
    gone = _fake_actor(3, 5.0, 0.0, 0.0, 0.0, (1, 1, 1), alive=False)
    truth = truth_record(ego, carla_map, [ego, other, gone], rear_axle_offset_m=1.4)
    assert [a["id"] for a in truth["actors"]] == [2]
    box = truth["actors"][0]
    # Core frame: actor origin at (20, 3) facing +y (CARLA yaw -90 => core +90); box 0.5 m further along +y.
    assert (box["x_m"], box["y_m"]) == pytest.approx((20.0 + 1.4, 3.5), abs=1e-6)
    assert box["yaw_rad"] == pytest.approx(math.pi / 2)
    assert box["z_m"] == pytest.approx(0.7)
    corners = np.array(box["corners_m"])
    assert corners.shape == (4, 2)
    assert np.sort(corners[:, 0]) == pytest.approx([20.5, 20.5, 22.3, 22.3], abs=1e-6)  # 0.9 half-width across x
    assert np.sort(corners[:, 1]) == pytest.approx([1.5, 1.5, 5.5, 5.5], abs=1e-6)  # 2.0 half-length along y


def test_world_to_vehicle_frame_round_trip():
    transform = SimpleNamespace(location=SimpleNamespace(x=10.0, y=-5.0, z=0.2), rotation=SimpleNamespace(yaw=-90.0))
    pose = rear_axle_pose(transform, 1.4)
    # CARLA yaw -90 = facing core +y (north); rear axle 1.4 m behind.
    assert pose == pytest.approx((10.0, 5.0 - 1.4, 0.2, math.pi / 2))
    ahead = world_to_ego([(10.0, 15.0, 0.2)], pose)[0]
    assert ahead == pytest.approx((11.4, 0.0, 0.0))
    assert to_ego_frame(pose, (10.0, 15.0), math.pi / 2) == pytest.approx((11.4, 0.0, 0.0))


def test_box_membership_tells_a_box_from_its_mirror_image():
    centre, yaw, extent = np.array([20.0, 3.5, 0.75]), 0.0, np.array([1.8, 0.9, 0.75])
    corners = box_corners(centre, yaw, extent)
    assert inside_box(corners, centre, yaw, extent, 0.01).all()
    assert not inside_box(corners, centre * (1, -1, 1), -yaw, extent, 0.15).any()


def test_projection():
    cal = camera_calibration(CameraSpec("c", Mount(1.5, 0, 2.0), 1280, 720, 90.0), 1.38641)
    px, front = project(np.array([[30.0, 0.0, 2.0], [-5.0, 0.0, 2.0]]), cal)
    assert len(px) == 1 and front.tolist() == [True, False]
    assert px[0] == pytest.approx((640, 360), abs=1e-6)


def test_repair_moves_schema1_boxes_onto_their_bounding_box_centres(tmp_path):
    import json

    from carla_adapter.mapless_capture_repair import main as repair
    from carla_adapter.mapless_capture_repair import repair_actor

    actor = dict(type_id="vehicle.fake", x_m=20.0, y_m=3.0, z_m=0.7, yaw_rad=math.pi / 2, extent_m=[2.0, 0.9, 0.7])
    fixed = repair_actor(actor, [0.5, 0.0, 0.7])
    assert (fixed["x_m"], fixed["y_m"]) == pytest.approx((20.0, 3.5))
    assert np.sort(np.array(fixed["corners_m"])[:, 1]) == pytest.approx([1.5, 1.5, 5.5, 5.5])
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps(dict(schema="group7-mapless-capture-1")), encoding="utf-8")
    (run / "samples.jsonl").write_text(json.dumps(dict(frame=1, truth=dict(actors=[actor]))) + "\n", encoding="utf-8")
    offsets = tmp_path / "offsets.json"
    offsets.write_text(json.dumps({"vehicle.fake": [0.5, 0.0, 0.7]}), encoding="utf-8")
    repair(["--run", str(run), "--offsets", str(offsets), "--no-carla"])
    sample = json.loads((run / "samples.jsonl").read_text(encoding="utf-8"))
    assert sample["truth"]["actors"][0]["y_m"] == pytest.approx(3.5)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "group7-mapless-capture-2" and (run / "samples.schema1.jsonl").exists()
    with pytest.raises(ValueError):
        repair(["--run", str(run), "--offsets", str(offsets), "--no-carla"])  # already schema 2
    (run / "manifest.json").write_text(json.dumps(dict(schema="group7-mapless-capture-1")), encoding="utf-8")
    offsets.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        repair(["--run", str(run), "--offsets", str(offsets), "--no-carla"])  # unknown type, no CARLA
