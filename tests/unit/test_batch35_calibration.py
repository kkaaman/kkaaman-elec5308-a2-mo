"""Calibration checks from independent geometric/constant-deceleration fixtures."""

import math
from dataclasses import asdict

import numpy as np
import pytest

from autonomy.baseline import BaselineConfig
from autonomy.configuration import read_config, write_config
from carla_adapter.depth_calibration import choose_convention
from evaluation.longitudinal import braking_summary
from evaluation.perception_replay import axial_image, range_summary
from geometry.camera import CameraIntrinsics
from obstacle_perception.box_depth import estimate_box_depth
from obstacle_perception.types import PixelBox
from scripts.freeze_thresholds import derive
from visualisation.lane_monitor import monitor_lane


def test_narrow_pedestrian_roi_counts_pixel_centres():
    cal = CameraIntrinsics.from_horizontal_fov(640, 360, math.pi / 2).calibration(np.eye(4))
    result = estimate_box_depth(
        np.full((360, 640), 32.0, np.float32), PixelBox(318, 182, 323, 200), cal
    )
    assert result.valid and result.valid_pixels >= 9
    assert result.axial_depth_m == 32


@pytest.mark.parametrize("convention", ["axial", "radial"])
def test_off_axis_rays_discriminate_depth_definition(convention):
    rows = [
        dict(
            raw_depth_m=20 if convention == "axial" else 25,
            reference={"axial_m": 20, "radial_m": 25},
        )
        for _ in range(30)
    ]
    report = choose_convention(rows)
    assert report["status"] == "MEASURED_PASS" and report["convention"] == convention
    assert choose_convention(rows[:10])["status"] == "INCONCLUSIVE"
    for r in rows:
        r["raw_depth_m"] = 22.5
    assert choose_convention(rows)["status"] == "INCONCLUSIVE"


def test_radial_conversion_at_off_axis_pixel():
    cal = CameraIntrinsics(4, 4, 2, 2, 2, 2).calibration(np.eye(4))
    image = np.full((4, 4), 10, np.float32)
    assert axial_image(image, cal, "radial")[2, 0] == pytest.approx(10 / math.sqrt(2))
    assert axial_image(image, cal, "axial")[2, 0] == 10


def braking_rows():
    rows = []
    for speed in (2, 5, 8):
        for brake in (0.4, 0.7, 1):
            for i, t in enumerate(np.linspace(0, speed / 2, 20)):
                rows.append(
                    dict(
                        trial=f"{speed}-{brake}",
                        phase="brake",
                        target_mps=speed,
                        brake=brake,
                        speed_mps=max(0, speed - 2 * t),
                        sim_time_s=float(t),
                        distance_m=float(speed * t - t * t),
                    )
                )
    return rows


def test_measured_budget_uses_all_trials_and_conservative_reserve():
    report = braking_summary(braking_rows())
    assert report["status"] == "MEASURED_PASS"
    assert report["deceleration_mps2"] == pytest.approx(1.4)
    assert report["response_budget_s"] >= 0.4
    assert braking_summary(braking_rows()[:-20])["status"] == "INCOMPLETE"


def test_empty_range_never_claims_pedestrian_25m():
    report = range_summary([])
    assert not report["pedestrian_25m_covered"]
    assert report["usable_range_lower_bound_m"] == {"car": 0, "person": 0}


def test_threshold_derivation_reduces_cruise_and_rejects_test_set():
    perception = dict(
        split="val",
        fault=None,
        usable_range_lower_bound_m={"car": 12.5, "person": 12.5},
        warm_p95_s=0.05,
    )
    budget = dict(status="MEASURED_PASS", deceleration_mps2=1, response_budget_s=1)
    result = derive(BaselineConfig(), budget, perception, {"status": "MEASURED_PASS"})
    assert result.governor.cruise_mps < 5
    assert result.governor.deceleration_mps2 == 1
    perception["split"] = "test"
    with pytest.raises(ValueError):
        derive(BaselineConfig(), budget, perception, {"status": "MEASURED_PASS"})


def test_config_roundtrip_and_unknown_key_rejection(tmp_path):
    import json

    path = tmp_path / "config.json"
    write_config(path, BaselineConfig())
    config, _, _ = read_config(path)
    assert asdict(config) == asdict(BaselineConfig())
    record = json.loads(path.read_text())
    record["runtime"]["typo"] = 3
    path.write_text(json.dumps(record))
    with pytest.raises(TypeError):
        read_config(path)


def test_monitor_sign_and_no_control_output():
    left = monitor_lane([(0, -1), (10, -1)])
    right = monitor_lane([(0, 1), (10, 1)])
    assert left.side == "left" and left.signed_offset_m == pytest.approx(1)
    assert right.side == "right" and right.signed_offset_m == pytest.approx(-1)
    assert not monitor_lane([]).valid
    assert not hasattr(left, "command")
