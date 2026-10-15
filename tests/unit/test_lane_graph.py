"""Offline consumers keep explicit lane identity and legal connectivity."""
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from autonomy import paths
from planning.lane_graph import LaneSegment, LaneTopology


def segment(road=1, *, points=((0, 0, 0), (1, 0, 0)), successors=(), junction=None):
    return LaneSegment(road, 0, -1, 0.001, 0.999, points, successors, junction)


def document(*lanes):
    return LaneTopology("test-map", "a" * 64, 2.0, 5.0, lanes)


def fork():
    return document(segment(successors=("2:0:-1", "3:0:-1")),
                    segment(2, points=((1.01, 0, 0), (2, 1, 0)), junction=7),
                    segment(3, points=((1.01, 0.01, 0), (2, -1, 0)), junction=7))


def test_directed_fork_and_junction_endpoints():
    topology = fork()
    graph = topology.graph
    edges = {(e.start_id, e.end_id) for e in graph.edges}
    assert len(graph.nodes) == 6
    assert len(edges) == 5
    assert ("1:0:-1:exit", "2:0:-1:entry") in edges
    assert ("1:0:-1:exit", "3:0:-1:entry") in edges
    assert ("2:0:-1:entry", "1:0:-1:exit") not in edges
    assert topology.lanes[1].junction_id == 7
    assert all(edge.speed_limit_mps == 5 for edge in graph.edges)


def test_proximity_does_not_create_crossing_or_elevation_connections():
    topology = document(segment(), segment(2, points=((0, 0, 5), (1, 0, 5))),
                        segment(3, points=((0.5, -1, 0), (0.5, 1, 0))))
    assert len(topology.graph.edges) == 3
    assert len(topology.graph.nodes) == 6


def test_merge_does_not_authorize_an_unlisted_turn():
    a = segment(successors=("3:0:-1", "4:0:-1"))
    b = segment(2, points=((0, 0.01, 0), (1, 0.01, 0)), successors=("3:0:-1",))
    c = segment(3, points=((1.01, 0, 0), (2, 1, 0)))
    d = segment(4, points=((1.01, 0.01, 0), (2, -1, 0)))
    edges = {(e.start_id, e.end_id) for e in document(a, b, c, d).graph.edges}
    assert (b.exit_node_id, c.entry_node_id) in edges
    assert (b.exit_node_id, d.entry_node_id) not in edges


def test_cycle_is_retained():
    a = segment(points=((0.01, 0, 0), (1, 0, 0)), successors=("2:0:-1",))
    b = segment(2, points=((1.01, 0, 0), (0, 0, 0)), successors=("1:0:-1",))
    assert len(document(a, b).graph.edges) == 4


def test_serialization_order_independence_and_round_trip(tmp_path):
    original = fork()
    reordered = replace(original, lanes=tuple(reversed(original.lanes)))
    assert original.to_json() == reordered.to_json()
    assert original.sha256 == reordered.sha256
    path = tmp_path / "graph.json"
    original.save(path)
    assert LaneTopology.load(path) == original
    assert path.read_bytes() == original.to_json().encode("utf-8")
    assert b"\r\n" not in path.read_bytes()


@pytest.mark.parametrize("change", [
    {"schema_version": 2}, {"schema_version": True}, {"coordinate_frame": "carla"},
    {"map_name": ""}, {"opendrive_sha256": "bad"}, {"sampling_distance_m": 0},
    {"navigation_speed_cap_mps": float("nan")}, {"max_connection_gap_m": True},
])
def test_invalid_document_metadata(change):
    with pytest.raises(ValueError):
        replace(fork(), **change)


@pytest.mark.parametrize("change", [
    {"lane_id": 0}, {"road_id": True}, {"junction_id": -1}, {"entry_s_m": -1},
    {"points_xyz_m": ((0, 0, 0), (0, 0, 1))},
    {"points_xyz_m": ((0, 0, 0), (float("inf"), 0, 0))},
    {"successor_ids": ("2:0:-1", "2:0:-1")},
])
def test_invalid_lane_records(change):
    with pytest.raises(ValueError):
        replace(segment(), **change)


def test_unknown_duplicate_and_disconnected_lanes_rejected():
    with pytest.raises(ValueError, match="unknown successor"):
        document(segment(successors=("missing",)))
    with pytest.raises(ValueError, match="duplicate lane"):
        document(segment(), segment())
    with pytest.raises(ValueError, match="disconnected lane"):
        document(segment(successors=("2:0:-1",)), segment(2, points=((5, 0, 0), (6, 0, 0))))
    with pytest.raises(ValueError, match="coincident"):
        document(segment(successors=("2:0:-1",)), segment(2, points=((1, 0, 0), (2, 0, 0))))


def test_loader_rejects_unsupported_and_malformed_json():
    data = json.loads(fork().to_json())
    data["schema_version"] = 99
    with pytest.raises(ValueError, match="schema version"):
        LaneTopology.from_json(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        LaneTopology.from_json('{"map_name":"a","map_name":"b"}')
    with pytest.raises(ValueError):
        LaneTopology.from_json('{"schema_version":1,"coordinate_frame":"x","lanes":[{}]}')


def test_town10hd_fixture_integrity():
    topology = LaneTopology.load(paths.FIXTURES_DIR / "Town10HD.lane_graph.json")
    assert topology.map_name == "Town10HD"
    assert len(topology.lanes) == 168
    assert len(topology.graph.edges) == 368
    assert sum(lane.junction_id is not None for lane in topology.lanes) == 82
    # Every lane retains travel-order geometry, not just entry/exit chords.
    assert any(len(lane.points_xyz_m) > 3 and lane.junction_id is not None for lane in topology.lanes)


def test_offline_cli_loads_fixture_with_carla_import_forbidden():
    script = paths.REPO_ROOT / "scripts" / "export_lane_graph.py"
    fixture = paths.FIXTURES_DIR / "Town10HD.lane_graph.json"
    code = (
        "import importlib.abc, sys, runpy\n"
        "class RejectCarla(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self, fullname, path=None, target=None):\n"
        "  if fullname == 'carla' or fullname.startswith('carla.'):\n"
        "   raise AssertionError('core tried to import CARLA')\n"
        "sys.meta_path.insert(0, RejectCarla())\n"
        f"sys.argv = [{str(script)!r}, 'validate', {str(fixture)!r}]\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(sys.executable).parent,
                            env=dict(os.environ, PYTHONPATH=str(paths.SRC_DIR)), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["lanes"] == 168
