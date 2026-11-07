"""CARLA-like map queries exercise export without a simulator."""
import hashlib
import math
import struct
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

import carla_adapter.topology as exporter
from scripts import check_baseline_mission


class FakeWaypoint:
    def __init__(self, map_, road, s):
        self.map = map_
        self.road_id, self.section_id, self.lane_id, self.s = road, 0, -1, s
        self.lane_type = "Driving"
        self.is_junction = road == 2
        self.junction_id = 9 if self.is_junction else -1
        # A curved junction preserves an intermediate bulge, not a chord.
        x = {1: 0, 2: 4, 3: 4, 4: 12}[road] + (s if road != 4 else 4 - s)
        y = math.sin(s * math.pi / 4) if road == 2 else 0
        self.transform = SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=0.5))

    def next(self, distance):
        direction = -1 if self.road_id == 4 else 1
        s = self.s + distance * direction
        if 0 < s < 4:
            return [FakeWaypoint(self.map, self.road_id, s)]
        remaining = s - 4 if direction == 1 else -s
        targets = list(self.map.successors[self.road_id])
        if self.map.reverse:
            targets.reverse()
        return [FakeWaypoint(self.map, road, remaining) for road in targets]


class FakeMap:
    name = "fake-map"

    def __init__(self, successors=None, reverse=False):
        self.successors = successors or {1: (2, 3), 2: (), 3: (), 4: ()}
        self.reverse = reverse

    def to_opendrive(self):
        return '<OpenDRIVE>' + ''.join(
            f'<road id="{i}" length="4"><lanes><laneSection s="0"><right>'
            '<lane id="-1" type="driving"/></right></laneSection></lanes></road>' for i in sorted(self.successors)
        ) + '</OpenDRIVE>'

    def get_topology(self):
        result = []
        for road, targets in self.successors.items():
            for target in targets or (road,):
                result.append((FakeWaypoint(self, road, 0 if road != 4 else 4),
                               FakeWaypoint(self, target, 0 if targets else 4)))
        return list(reversed(result)) if self.reverse else result

    def get_waypoint_xodr(self, road_id, lane_id, s):
        return FakeWaypoint(self, road_id, s)


def test_export_preserves_all_branches_curve_axes_and_decreasing_s():
    topology = exporter.export_lane_topology(FakeMap(), sampling_distance_m=1.0)
    by_id = {lane.segment_id: lane for lane in topology.lanes}
    assert by_id["1:0:-1"].successor_ids == ("2:0:-1", "3:0:-1")
    curve = by_id["2:0:-1"]
    assert curve.junction_id == 9
    assert min(p[1] for p in curve.points_xyz_m) < -0.9
    assert all(p[2] == 0.5 for p in curve.points_xyz_m)
    assert by_id["4:0:-1"].entry_s_m > by_id["4:0:-1"].exit_s_m
    assert by_id["3:0:-1"].successor_ids == ()


def test_export_is_independent_of_topology_and_successor_order():
    a = exporter.export_lane_topology(FakeMap())
    b = exporter.export_lane_topology(FakeMap(reverse=True))
    assert a.to_json() == b.to_json()


def test_ambiguous_direction_is_rejected(monkeypatch):
    original = FakeWaypoint.next

    def ambiguous(self, distance):
        result = original(self, distance)
        return result * 2 if abs(self.s - 2) < 1e-6 else result

    monkeypatch.setattr(FakeWaypoint, "next", ambiguous)
    with pytest.raises(exporter.TopologyError, match="ambiguous travel"):
        exporter.export_lane_topology(FakeMap())


def test_missing_section_and_mismatched_successor_rejected(monkeypatch):
    map_ = FakeMap()
    pairs = map_.get_topology()
    monkeypatch.setattr(map_, "get_topology", lambda: pairs[:-1])
    with pytest.raises(exporter.TopologyError, match="missing driving"):
        exporter.export_lane_topology(map_)
    monkeypatch.setattr(map_, "get_topology", lambda: [p for p in pairs if p[1].road_id != 3 or p[0].road_id == 3])
    with pytest.raises(exporter.TopologyError, match="successor query disagrees"):
        exporter.export_lane_topology(map_)


@pytest.mark.parametrize("value", [0, -1, float("nan"), True])
def test_bad_sampling_parameters(value):
    with pytest.raises(ValueError, match="sampling_distance_m"):
        exporter.export_lane_topology(FakeMap(), sampling_distance_m=value)


class MultiSectionMap:
    """A short section followed by another section on the same road/lane."""

    name = "multi-section"

    def to_opendrive(self):
        return ('<OpenDRIVE><road id="1" length="4"><lanes>'
                '<laneSection s="0"><right><lane id="-1" type="driving"/></right></laneSection>'
                '<laneSection s="0.5"><right><lane id="-1" type="driving"/></right></laneSection>'
                '</lanes></road></OpenDRIVE>')

    def get_waypoint_xodr(self, road_id, lane_id, s):
        if not 0 <= s < 4:
            return None
        wp = SimpleNamespace(road_id=1, lane_id=-1, section_id=int(s >= 0.5), s=s,
                             lane_type="Driving", is_junction=False, junction_id=-1,
                             transform=SimpleNamespace(location=SimpleNamespace(x=s, y=0, z=0)))
        wp.next = lambda distance: [target] if (target := self.get_waypoint_xodr(1, -1, s + distance)) else []
        return wp

    def get_topology(self):
        def wp(s):
            return self.get_waypoint_xodr(1, -1, s)

        return [(wp(0), wp(0.5)), (wp(0.5), wp(3.999))]


def test_short_sections_are_not_skipped_and_same_road_sections_stay_distinct():
    topology = exporter.export_lane_topology(MultiSectionMap(), sampling_distance_m=2)
    assert len(topology.lanes) == 2
    first, second = topology.lanes
    assert first.successor_ids == (second.segment_id,)
    assert len(first.points_xyz_m) == 2
    assert first.exit_s_m < 0.5 < second.entry_s_m
    assert len(topology.graph.edges) == 3


def test_wrong_section_or_unreachable_sample_is_rejected(monkeypatch):
    map_ = MultiSectionMap()
    original = map_.get_waypoint_xodr

    def wrong_section(road, lane, s):
        waypoint = original(road, lane, s)
        if waypoint and abs(s - 0.001) < 1e-6:
            waypoint.section_id = 1
        return waypoint

    monkeypatch.setattr(map_, "get_waypoint_xodr", wrong_section)
    with pytest.raises(exporter.TopologyError, match="cannot sample"):
        exporter.export_lane_topology(map_)


def test_illegal_geometric_shortcut_is_rejected():
    map_ = FakeMap(successors={1: (4,), 4: ()})
    with pytest.raises(ValueError, match="disconnected lane boundary"):
        exporter.export_lane_topology(map_)


def test_live_asset_name_and_local_name_produce_same_document():
    local = FakeMap()
    live = FakeMap()
    live.name = "/Game/Carla/Maps/fake-map"
    assert exporter.export_lane_topology(local).to_json() == exporter.export_lane_topology(live).to_json()


def test_map_verification_rejects_tampered_geometry_and_successors():
    map_ = FakeMap()
    document = exporter.export_lane_topology(map_, sampling_distance_m=1)
    result = exporter.validate_lane_topology_on_map(map_, document)
    assert result["lanes_checked"] == 4
    assert result["successors_checked"] == 2
    first, curved, *rest = document.lanes
    points = list(curved.points_xyz_m)
    points[2] = (points[2][0], points[2][1] + 0.5, points[2][2])
    bad = replace(document, lanes=(first, replace(curved, points_xyz_m=points), *rest))
    with pytest.raises(exporter.TopologyError, match="leaves lane centre"):
        exporter.validate_lane_topology_on_map(map_, bad)
    bad = replace(document, lanes=(replace(first, successor_ids=("2:0:-1",)), curved, *rest))
    with pytest.raises(exporter.TopologyError, match="saved successors"):
        exporter.validate_lane_topology_on_map(map_, bad)
    with pytest.raises(exporter.TopologyError, match="hashes differ"):
        exporter.validate_lane_topology_on_map(map_, replace(document, opendrive_sha256="b" * 64))


def test_map_verification_tolerates_sdk_float32_s_quantization():
    class QuantizedMap:
        name = "quantized"
        length = 37.02883876824462

        def to_opendrive(self):
            return (f'<OpenDRIVE><road id="1" length="{self.length}"><lanes>'
                    '<laneSection s="0"><right><lane id="-1" type="driving"/></right>'
                    '</laneSection></lanes></road></OpenDRIVE>')

        def waypoint(self, s):
            wp = SimpleNamespace(road_id=1, lane_id=-1, section_id=0, s=s,
                                 lane_type="Driving", is_junction=False,
                                 transform=SimpleNamespace(location=SimpleNamespace(x=s, y=0, z=0)))
            wp.next = lambda distance: [self.waypoint(s + distance)] if s + distance < self.length else []
            return wp

        def get_waypoint_xodr(self, road, lane, s):
            return self.waypoint(struct.unpack("f", struct.pack("f", s))[0])

        def get_topology(self):
            return [(self.waypoint(0), self.waypoint(self.length - 0.001))]

    map_ = QuantizedMap()
    document = exporter.export_lane_topology(map_)
    assert exporter.validate_lane_topology_on_map(map_, document)["lanes_checked"] == 1


def test_live_connection_checks_versions_and_map_without_mutation(monkeypatch):
    versions = ["0.9.16", "0.9.15"]
    client = SimpleNamespace(set_timeout=lambda seconds: None, get_client_version=lambda: versions[0],
                             get_server_version=lambda: versions[1],
                             get_world=lambda: SimpleNamespace(get_map=lambda: SimpleNamespace(name="/Game/Carla/Maps/Town10HD")))
    monkeypatch.setitem(sys.modules, "carla", SimpleNamespace(Client=lambda host, port: client))
    with pytest.raises(exporter.TopologyError, match="version mismatch"):
        exporter.connect_map("localhost", 2000, 2, "Town10HD")
    versions[1] = "0.9.16"
    assert exporter.connect_map("localhost", 2000, 2, "Town10HD")[1] == "0.9.16"
    with pytest.raises(exporter.TopologyError, match="expected active map"):
        exporter.connect_map("localhost", 2000, 2, "Town01")


@pytest.mark.parametrize(("actual", "expected"), [
    ("Town10HD_Opt", "Town10HD"), ("Town10HD", "Town10HD_Opt"),
])
def test_town10hd_alias_requires_the_frozen_opendrive_fingerprint(monkeypatch, actual, expected):
    map_ = FakeMap()
    map_.name = f"Carla/Maps/{actual}"
    digest = hashlib.sha256(map_.to_opendrive().encode("utf-8")).hexdigest()
    client = SimpleNamespace(
        set_timeout=lambda seconds: None, get_client_version=lambda: "0.9.16",
        get_server_version=lambda: "0.9.16",
        get_world=lambda: SimpleNamespace(get_map=lambda: map_),
    )
    monkeypatch.setitem(sys.modules, "carla", SimpleNamespace(Client=lambda host, port: client))
    with pytest.raises(exporter.TopologyError, match="expected active map"):
        exporter.connect_map("localhost", 2000, 2, expected)
    assert exporter.connect_map(
        "localhost", 2000, 2, expected, expected_opendrive_sha256=digest
    )[0] is map_
    with pytest.raises(exporter.TopologyError, match="OpenDRIVE hash"):
        exporter.connect_map("localhost", 2000, 2, expected, expected_opendrive_sha256="b" * 64)
    with pytest.raises(exporter.TopologyError, match="expected active map"):
        exporter.connect_map("localhost", 2000, 2, "Town01", expected_opendrive_sha256=digest)


@pytest.mark.parametrize(("actual", "expected"), [
    ("Town10HD_Opt", "Town10HD"), ("Town10HD", "Town10HD_Opt"),
])
def test_topology_validation_accepts_town10hd_alias_only_after_hash_match(actual, expected):
    map_ = FakeMap()
    map_.name = expected
    document = exporter.export_lane_topology(map_)
    map_.name = f"Carla/Maps/{actual}"
    assert exporter.validate_lane_topology_on_map(map_, document)["lanes_checked"] == 4
    with pytest.raises(exporter.TopologyError, match="hashes differ"):
        exporter.validate_lane_topology_on_map(map_, replace(document, opendrive_sha256="b" * 64))
    with pytest.raises(exporter.TopologyError, match="map names differ"):
        exporter.validate_lane_topology_on_map(map_, replace(document, map_name="Town01"))


def test_baseline_verify_map_passes_frozen_fingerprint_to_live_connection(monkeypatch, capsys):
    observed = {}
    actual_map = object()

    def connect(host, port, timeout_s, expected_map, *, expected_opendrive_sha256):
        observed.update(map_name=expected_map, digest=expected_opendrive_sha256)
        return actual_map, "0.9.16"

    def validate(map_, document):
        assert map_ is actual_map
        assert observed["map_name"] == document.map_name
        assert observed["digest"] == document.opendrive_sha256
        return {"verified": True}

    monkeypatch.setattr(exporter, "connect_map", connect)
    monkeypatch.setattr(exporter, "validate_lane_topology_on_map", validate)
    assert check_baseline_mission.main(["--verify-map"]) == 0
    assert '"verified": true' in capsys.readouterr().out
