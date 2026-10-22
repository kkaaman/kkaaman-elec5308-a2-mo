"""The capture inspection on a tiny synthetic run whose labels are drawn from the rig geometry."""

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.capture_inspection import inspect
from mapless.geometry import camera_calibration
from mapless.sensor_suite import load_sensor_suite

ROOT = Path(__file__).resolve().parents[2]
SUITE = load_sensor_suite(ROOT / "configs" / "sensor_suites" / "mapless_v2.json")
OFFSET = 1.38641
CAR = dict(x_m=12.0, y_m=0.0, z_m=0.75, yaw_rad=0.0, extent_m=[2.0, 0.9, 0.75])


def render(spec, cv2):
    """RGB and semantic tags: road below the horizon, sky above, one car box ahead."""
    cal = camera_calibration(spec, OFFSET)
    m = np.asarray(cal.camera_to_ego)
    u, v = np.meshgrid(np.arange(spec.width), np.arange(spec.height))
    rays = np.stack(((u - cal.cx_px) / cal.fx_px, (v - cal.cy_px) / cal.fy_px, np.ones_like(u, float)), -1) @ m[:3, :3].T
    tags = np.where(rays[..., 2] < 0, 1, 11).astype(np.uint8)
    corners = np.array([[CAR["x_m"] + sx * 2.0, sy * 0.9, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (0.0, 1.5)])
    optical = (corners - m[:3, 3]) @ m[:3, :3]
    if np.all(optical[:, 2] > 0.1):
        px = np.column_stack((cal.cx_px + cal.fx_px * optical[:, 0] / optical[:, 2], cal.cy_px + cal.fy_px * optical[:, 1] / optical[:, 2]))
        cv2.fillConvexPoly(tags, cv2.convexHull(px.astype(np.int32)), 14)
    rgb = np.stack([tags * 9, tags * 5, 255 - tags * 9], -1).astype(np.uint8)
    return rgb, tags


def lidar():
    xs, ys = np.meshgrid(np.arange(5.0, 29.0, 0.5), np.arange(-6.0, 6.0, 0.5))
    ground = np.column_stack((xs.ravel(), ys.ravel(), np.zeros(xs.size)))
    # Nothing under or behind the car: its shadow, as seen from the front camera at x = 2.9 m.
    near_face = CAR["x_m"] - 2.0 - 2.9
    shadow = (ground[:, 0] > CAR["x_m"] - 2.2) & (np.abs(ground[:, 1]) < 1.2 * (ground[:, 0] - 2.9) / near_face)
    ground = ground[~shadow]
    face = np.array([[CAR["x_m"] - 2.0, y, z] for y in np.arange(-0.7, 0.71, 0.1) for z in np.arange(0.4, 1.4, 0.1)])
    points = np.vstack((ground, face))
    return np.column_stack((points, np.full(len(points), 0.5))).astype(np.float32)


def make_run(tmp_path, cv2, swap_labels=False):
    run = tmp_path / "run"
    samples = []
    for frame in (100, 110):
        files = {"rgb": {}, "semantic": {}, "lidar": {}, "radar": {}}
        for spec in SUITE.cameras:
            rgb, tags = render(spec, cv2)
            if swap_labels:
                tags = tags[::-1].copy()  # a label image that does not belong to its RGB image
            for kind, image, ext in (("rgb", rgb[:, :, ::-1], "jpg"), ("semantic", tags, "png")):
                rel = Path(kind if kind == "rgb" else "labels/semantic") / spec.name / f"{frame:08d}.{ext}"
                (run / rel).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(run / rel), image)
                files[kind][spec.name] = rel.as_posix()
        rel = Path("lidar/roof") / f"{frame:08d}.npz"
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(run / rel, points=lidar())
        files["lidar"]["roof"] = rel.as_posix()
        truth = dict(weather="ClearNoon", episode=0, traffic_light_state="Green", actors=[dict(type_id="vehicle.audi.a2", **CAR)])
        samples.append(dict(frame=frame, files=files, speed_mps=3.0, truth=truth))
    manifest = dict(map="Carla/Maps/Town03", status="COMPLETE", suite=SUITE.as_dict(), rig_records=[dict(rear_axle_offset_m=OFFSET)])
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in samples), encoding="utf-8")
    return run


def test_consistent_run_passes_both_projection_checks(tmp_path):
    cv2 = pytest.importorskip("cv2")
    run = make_run(tmp_path, cv2)
    summary = inspect(run, sheet_samples=2)
    front = summary["projection_checks"]["front"]
    assert front["ground_points_on_ground_labels"] > 0.95 and front["ground_points"] > 100
    assert front["vehicle_points_on_vehicle_labels"] > 0.9 and front["vehicle_points"] > 50
    assert summary["class_shares"]["front"]["road"] > 0.3
    assert (run / "inspection" / "contact_sheet.png").exists() and (run / "inspection" / "cameras_sheet.png").exists()


def test_mismatched_labels_show_up_in_the_projection_checks(tmp_path):
    cv2 = pytest.importorskip("cv2")
    summary = inspect(make_run(tmp_path, cv2, swap_labels=True), sheet_samples=1)
    assert summary["projection_checks"]["front"]["ground_points_on_ground_labels"] < 0.5
