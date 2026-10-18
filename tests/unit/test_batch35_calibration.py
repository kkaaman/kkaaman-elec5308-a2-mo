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


def _pose(sequence, name, distance, *, target, lateral=0.0, good=True, code=1):
    truth = dict(
        instance_code=code,
        class_name=name,
        xyxy=[300, 150, 340, 210],
        scenario_target=target,
        requested_lateral_m=lateral if target else None,
        distance_m=distance,
        detected=good,
        confirmed=good,
        valid_depth=good,
        track_id=1 if good else None,
        depth_error_m=0.1 if good else None,
    )
    return [
        dict(
            sequence_id=sequence,
            sim_time_s=i * 0.05,
            wall_s=0.02,
            truths=[truth],
            false_positives=0,
        )
        for i in range(12)
    ]


def test_range_gate_uses_in_path_targets_with_ninety_percent_of_poses():
    # Live 18 Oct capture: kerb-side map cars (oblique, cut off, invalid depth)
    # zeroed the car range although every placed in-lane car was detected.
    records = []
    for centre in (5, 10, 15, 20, 25):
        for k in range(10):
            records += _pose(
                f"car{centre}-{k}", "car", centre, target=True, good=(k != 0 or centre != 10)
            )
            records += _pose(
                f"car{centre}-{k}", "car", centre + 1, target=False, good=False, code=7
            )
        for k in range(10):
            records += _pose(f"p{centre}-{k}", "person", centre, target=True, good=centre <= 10)
    report = range_summary(records)
    # One failing pose in ten still passes the 10 m car bin.
    assert report["usable_range_lower_bound_m"] == {"car": 22.5, "person": 7.5}
    # Two failures in a bin of ten do not.
    worse = list(records) + _pose("car10-extra", "car", 10, target=True, good=False)
    assert range_summary(worse)["usable_range_lower_bound_m"]["car"] == 7.5 - 5
    assert report["all_visible_usable_range_m"]["car"] == 0.0
    # Side-lane targets are not in-path.
    side = []
    for k in range(6):
        side += _pose(f"s{k}", "car", 15, target=True, lateral=4.0)
    assert range_summary(side)["usable_range_lower_bound_m"]["car"] == 0.0


def test_scenario_target_matches_requested_pose_only():
    from evaluation.perception_replay import tag_scenario_target

    cal = CameraIntrinsics.from_horizontal_fov(640, 360, math.pi / 2).calibration(
        ((0, 0, 1, 2.88641), (-1, 0, 0, 0), (0, -1, 0, 2.0), (0, 0, 0, 1))
    )
    truths = [
        dict(class_name="car", distance_m=10.2, xyxy=[300, 160, 340, 200]),  # ahead, in lane
        dict(class_name="car", distance_m=9.8, xyxy=[560, 160, 640, 220]),  # kerb-side, right
    ]
    target = dict(
        spawned=True, kind_requested="car", requested_surface_range_m=10, requested_lateral_m=0
    )
    assert tag_scenario_target(truths, target, cal) == (0, "requested_pose")
    assert truths[0]["scenario_target"] and "scenario_target" not in truths[1]
    assert tag_scenario_target(truths, dict(target, spawned=False), cal) is None
    far = dict(target, requested_surface_range_m=25)
    assert tag_scenario_target([dict(truths[1])], far, cal) is None


def test_scenario_target_by_actor_id_includes_targets_round_a_bend():
    from evaluation.perception_replay import instance_actor_id, tag_scenario_target

    cal = CameraIntrinsics.from_horizontal_fov(640, 360, math.pi / 2).calibration(
        ((0, 0, 1, 2.88641), (-1, 0, 0, 0), (0, -1, 0, 2.0), (0, 0, 0, 1))
    )
    # G = 0x98, B = 0x01 -> actor 408; map props carry a large high byte (167).
    assert instance_actor_id(0x9801) == 408 and instance_actor_id(29184) == 114
    assert instance_actor_id(17575) > 40000
    truths = [
        dict(class_name="car", instance_code=17575, distance_m=24.0, xyxy=[300, 170, 330, 190]),
        dict(class_name="car", instance_code=0x9801, distance_m=23.0, xyxy=[560, 170, 600, 190]),
    ]
    target = dict(
        spawned=True,
        kind_requested="car",
        actor_id=408,
        requested_surface_range_m=25,
        requested_lateral_m=0,
    )
    assert tag_scenario_target(truths, target, cal) == (1, "actor_id")


def test_freeze_copies_measured_detector_settings():
    perception = dict(
        split="val",
        fault=None,
        usable_range_lower_bound_m={"car": 22.5, "person": 22.5},
        warm_p95_s=0.04,
        detector=dict(image_size=960, confidence=0.25),
    )
    budget = dict(status="MEASURED_PASS", deceleration_mps2=1.5, response_budget_s=0.75)
    result = derive(BaselineConfig(), budget, perception, {"status": "MEASURED_PASS"})
    assert result.detector_image_size == 960 and result.detector_confidence == 0.25
    del perception["detector"]  # reports from before the field: measured at the defaults
    assert (
        derive(
            BaselineConfig(), budget, perception, {"status": "MEASURED_PASS"}
        ).detector_image_size
        == 640
    )
    with pytest.raises(ValueError):
        BaselineConfig(detector_image_size=1000)
