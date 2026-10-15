"""Opt-in acceptance on the active Town10HD map, using a fresh SDK process."""
import json
import os
import subprocess
import sys

import pytest

from autonomy import paths


@pytest.mark.skipif(os.environ.get("CARLA_TOPOLOGY_LIVE") != "1", reason="set CARLA_TOPOLOGY_LIVE=1 with a live Town10HD server")
def test_live_town10hd_export(tmp_path):
    # Existing A1 tests install a fake carla module; keep live calls isolated.
    code = """
import sys
from carla_adapter.topology import connect_map, export_lane_topology
from planning.lane_graph import LaneTopology
map_, version = connect_map(sys.argv[1], int(sys.argv[2]), 10.0, 'Town10HD')
assert version == '0.9.16'
a = export_lane_topology(map_)
b = export_lane_topology(map_)
assert a.to_json() == b.to_json()
a.save(sys.argv[3])
assert LaneTopology.load(sys.argv[3]) == a
print(a.to_json())
"""
    result = subprocess.run([sys.executable, "-c", code, os.environ.get("CARLA_HOST", "127.0.0.1"),
                             os.environ.get("CARLA_PORT", "2000"), str(tmp_path / "Town10HD.json")],
                            env=dict(os.environ, PYTHONPATH=str(paths.SRC_DIR)), capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["lanes"]
    assert any(lane["junction_id"] is not None for lane in data["lanes"])
