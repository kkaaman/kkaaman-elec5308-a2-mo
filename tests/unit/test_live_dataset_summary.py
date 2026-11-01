"""Repeated weather views must not inflate pairing or spatial-site uncertainty."""

from copy import deepcopy

import pytest

from scripts.summarize_live_dataset import clustered_interval, pairing


def test_recall_interval_clusters_and_retains_all_counts():
    result = clustered_interval([[10, 2, 0], [0, 1, 10]])
    assert result["spatial_sites"] == 2
    assert (result["tp"], result["fp"], result["fn"]) == (10, 3, 10)
    assert result["recall"] == 0.5
    assert result["recall_site_bootstrap_95"] == [0.0, 1.0]
    assert clustered_interval([[0, 2, 0]])["recall"] is None


def test_pairing_detects_pose_drift_and_keeps_failed_targets():
    rows = []
    for weather in ("ClearNoon", "HardRainNoon", "ClearNight"):
        rows.append(
            dict(
                weather_cell_id=f"S1-{weather}-r1",
                sample_id=f"{weather}/sample_0",
                sequence_id="scene",
                road_id=1,
                lane_id=-1,
                road_s_m=5.0,
                ego_world=dict(x=0.0),
                camera_world=dict(x=1.5),
                target_evaluation_only=dict(spawned=True, actor_id=len(rows)),
            )
        )
    assert pairing(rows)["status"] == "PAIRED"
    drift = deepcopy(rows)
    drift[1]["ego_world"]["x"] = 1.0
    assert pairing(drift)["status"] == "PAIRING_FAILED"
    rows[2]["target_evaluation_only"]["spawned"] = False
    assert pairing(rows)["failed_target_scenes"]
    with pytest.raises(ValueError, match="Duplicate"):
        pairing(rows + [rows[0]])
