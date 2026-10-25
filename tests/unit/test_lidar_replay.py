"""Fusion replay over the synthetic sweep set: determinism, HOLD accounting and JSON output."""

import json
import subprocess
import sys

import numpy as np
import pytest

from autonomy.lidar_replay import depth_report, replay_manifest, replay_path
from autonomy.paths import FIXTURES_DIR, REPO_ROOT
from obstacle_perception.lidar_synthetic import build_fixture, load_fixture_sweep

FIXTURE = FIXTURES_DIR / "lidar"


def test_fixture_regenerates_to_the_checked_in_arrays(tmp_path):
    manifest = build_fixture(tmp_path)
    checked = json.loads((FIXTURE / "sweeps.json").read_text())
    assert [row["frame_id"] for row in manifest["sweeps"]] == [
        row["frame_id"] for row in checked["sweeps"]
    ]
    assert (
        manifest["config"] == checked["config"]
        and manifest["sensor_to_ego"] == checked["sensor_to_ego"]
    )
    for fresh, old in zip(manifest["sweeps"], checked["sweeps"]):
        a = load_fixture_sweep(manifest, fresh, tmp_path).points_xyzi
        b = load_fixture_sweep(checked, old, FIXTURE).points_xyzi
        assert a.shape == b.shape
        assert np.allclose(a, b, atol=1e-4)
        assert fresh["expected_gap_m"] == old["expected_gap_m"] and fresh["note"] == old["note"]


def test_replay_scores_gaps_and_never_treats_sensor_loss_as_clear():
    result = replay_path(FIXTURE / "sweeps.json")
    summary, frames = result["summary"], result["frames"]
    assert summary["frames"] == 10 and summary["sweep_valid_frames"] == 8
    assert summary["max_abs_lidar_gap_error_m"] < 0.1
    assert summary["depth_loss_frames"] == 1 and summary["depth_loss_frames_treated_clear"] == 0
    assert summary["lidar_loss_frames"] >= 2 and summary["lidar_loss_frames_treated_clear"] == 0
    assert summary["lidar_track_sources"] == ["lidar"]
    by_note = {f["note"]: f for f in frames}
    sparse, empty = by_note["sparse_dropout"], by_note["empty_sweep"]
    assert (
        not sparse["sweep_valid"]
        and sparse["governor_input"]["hold"]
        and sparse["fused"]["obstacle_gap_m"] is None
    )
    assert empty["point_count"] == 0 and empty["governor_input"]["hold"]
    assert not by_note["depth_dropped"]["fused"]["valid"]
    assert "front-rgbd:depth_unavailable" in by_note["depth_dropped"]["fused"]["invalid_reasons"]
    far = by_note["depth_far_disagreement"]
    assert far["fused"]["valid"] and far["fused"]["nearest_source_id"] == "lidar"
    assert far["fused"]["obstacle_gap_m"] == pytest.approx(far["lidar_report"]["nearest_gap_m"])
    assert far["fused"]["disagreement_m"] > 3
    near = by_note["depth_nearer_disagreement"]
    assert (
        near["fused"]["nearest_source_id"] == "front-rgbd"
        and near["fused"]["obstacle_gap_m"] == 20.5
    )
    for f in frames:
        if f["fused"]["valid"]:
            gaps = [g for g in f["fused"]["source_gaps_m"].values() if g is not None]
            assert f["fused"]["obstacle_gap_m"] == min(gaps)
            assert f["fused"]["visible_distance_m"] <= f["fused"]["obstacle_gap_m"]
    json.dumps(result)  # the summary must be serialisable as written by the script


def test_depth_report_and_corridor_overrides():
    assert (
        depth_report(dict(depth_gap_m=None, depth_valid=True)).invalid_reason == "depth_unavailable"
    )
    assert (
        depth_report(dict(depth_gap_m=5.0, depth_valid=False)).invalid_reason == "depth_unavailable"
    )
    assert depth_report(dict(depth_gap_m=5.0, depth_valid=True)).nearest_gap_m == 5.0
    manifest = json.loads((FIXTURE / "sweeps.json").read_text())
    narrow = replay_manifest(manifest, FIXTURE, half_width_m=0.2, ahead_m=10.0)
    assert narrow["summary"]["policy"]["corridor_half_width_m"] == 0.2
    # A 10 m look-ahead cannot see the car that starts 24 m away: LiDAR reports no gap.
    first_valid = next(f for f in narrow["frames"] if f["lidar_report"]["valid"])
    assert first_valid["lidar_report"]["nearest_gap_m"] is None


def test_replay_script_writes_summary(tmp_path):
    output = tmp_path / "replay.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "replay_lidar_obstacles.py"),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "", "HOME": str(tmp_path)},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "frames" in completed.stdout
    data = json.loads(output.read_text())
    assert data["summary"]["frames"] == 10 and len(data["frames"]) == 10
