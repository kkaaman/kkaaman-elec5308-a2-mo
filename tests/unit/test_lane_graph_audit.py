"""Raw XML checks catch faults even when an exported graph is self-consistent."""
import hashlib
import math
from dataclasses import replace

import pytest

from planning.lane_graph import LaneSegment, LaneTopology
from planning.lane_graph_audit import audit_lane_topology


def linked_roads():
    source = '''<OpenDRIVE>
    <road id="1" length="2" junction="-1"><link><successor elementType="road" elementId="2" contactPoint="start"/></link>
    <planView><geometry s="0" x="0" y="0" hdg="0" length="2"><line/></geometry></planView>
    <lanes><laneSection s="0"><right><lane id="-1" type="driving"><link><successor id="-1"/></link>
    <width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane></right></laneSection></lanes></road>
    <road id="2" length="2" junction="-1">
    <planView><geometry s="0" x="2" y="0" hdg="0" length="2"><line/></geometry></planView>
    <lanes><laneSection s="0"><right><lane id="-1" type="driving">
    <width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane></right></laneSection></lanes></road>
    </OpenDRIVE>'''
    lanes = (LaneSegment(1, 0, -1, .001, 1.999, ((.001, -1.75, 0), (1.999, -1.75, 0)), ("2:0:-1",)),
             LaneSegment(2, 0, -1, .001, 1.999, ((2.001, -1.75, 0), (3.999, -1.75, 0))))
    return source, LaneTopology("test", hashlib.sha256(source.encode()).hexdigest(), 2, 5, lanes)


def test_raw_xml_connections_direction_geometry_and_reachability():
    source, document = linked_roads()
    report = audit_lane_topology(source, document)
    assert report["successors_checked"] == 1
    assert report["reachable_ordered_lane_pairs_excluding_self"] == 1
    assert report["strongly_connected_component_sizes"] == [1, 1]
    assert report["max_raw_xml_point_error_m"] < 1e-10


def test_xml_audit_detects_dropped_successor_in_an_otherwise_valid_graph():
    source, document = linked_roads()
    a, b = document.lanes
    bad = replace(document, lanes=(replace(a, successor_ids=()), b))
    with pytest.raises(ValueError, match="XML successors disagree"):
        audit_lane_topology(source, bad)


def test_xml_audit_detects_reversed_travel_and_mirrored_geometry():
    source, document = linked_roads()
    _, b = document.lanes
    reversed_lane = replace(b, entry_s_m=b.exit_s_m, exit_s_m=b.entry_s_m, points_xyz_m=tuple(reversed(b.points_xyz_m)))
    # Isolate a reversed terminal road so endpoint continuity is not the oracle.
    terminal_source = source[source.index('<road id="2"'):source.index('</OpenDRIVE>')]
    terminal_source = '<OpenDRIVE>' + terminal_source + '</OpenDRIVE>'
    bad = replace(document, lanes=(reversed_lane,), opendrive_sha256=hashlib.sha256(terminal_source.encode()).hexdigest())
    with pytest.raises(ValueError, match="travel direction disagrees"):
        audit_lane_topology(terminal_source, bad)
    mirrored = replace(b, points_xyz_m=tuple((p[0], -p[1], p[2]) for p in b.points_xyz_m))
    bad = replace(document, lanes=(mirrored,), opendrive_sha256=hashlib.sha256(terminal_source.encode()).hexdigest())
    with pytest.raises(ValueError, match="raw XML centre differs"):
        audit_lane_topology(terminal_source, bad)


def test_dense_probes_reject_a_curved_lane_replaced_with_an_endpoint_chord():
    source = '''<OpenDRIVE><road id="1" length="20" junction="-1">
    <planView><geometry s="0" x="0" y="0" hdg="0" length="20"><arc curvature="0.1"/></geometry></planView>
    <lanes><laneSection s="0"><right><lane id="-1" type="driving">
    <width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane></right></laneSection></lanes></road></OpenDRIVE>'''
    # Offset circle: the right lane centre has radius 10 + 1.75 metres.
    points = tuple((11.75 * math.sin(s / 10), 10 - 11.75 * math.cos(s / 10), 0) for s in (.001, 19.999))
    lane = LaneSegment(1, 0, -1, .001, 19.999, points)
    document = LaneTopology("curve", hashlib.sha256(source.encode()).hexdigest(), 20, 5, (lane,))
    with pytest.raises(ValueError, match="polyline shortcut leaves lane"):
        audit_lane_topology(source, document)


def test_hash_mismatch_and_unsupported_features_fail_explicitly():
    source, document = linked_roads()
    with pytest.raises(ValueError, match="source map hash differs"):
        audit_lane_topology(source + " ", document)
    changed = source.replace('junction="-1"', 'junction="-1" rule="LHT"')
    document = replace(document, opendrive_sha256=hashlib.sha256(changed.encode()).hexdigest())
    with pytest.raises(ValueError, match="LHT"):
        audit_lane_topology(changed, document)


def test_junction_lane_links_are_checked_directly():
    source, document = linked_roads()
    source = source.replace('elementType="road" elementId="2" contactPoint="start"', 'elementType="junction" elementId="7"')
    source = source.replace('<road id="2" length="2" junction="-1">', '<road id="2" length="2" junction="7">')
    source = source.replace('</OpenDRIVE>', '<junction id="7"><connection incomingRoad="1" connectingRoad="2" contactPoint="start">'
                            '<laneLink from="-1" to="-1"/></connection></junction></OpenDRIVE>')
    a, b = document.lanes
    document = replace(document, lanes=(a, replace(b, junction_id=7)), opendrive_sha256=hashlib.sha256(source.encode()).hexdigest())
    report = audit_lane_topology(source, document)
    assert report["junction_lane_counts"] == {7: 1}
    assert report["successors_checked"] == 1
    changed = source.replace('from="-1"', 'from="-2"')
    bad = replace(document, opendrive_sha256=hashlib.sha256(changed.encode()).hexdigest())
    with pytest.raises(ValueError, match="XML successors disagree"):
        audit_lane_topology(changed, bad)


def test_positive_lanes_follow_predecessors_in_decreasing_s_order():
    source, document = linked_roads()
    source = source.replace('id="-1"', 'id="1"').replace('<right>', '<left>').replace('</right>', '</left>')
    source = source.replace('<road id="2" length="2" junction="-1">',
                            '<road id="2" length="2" junction="-1"><link><predecessor elementType="road" elementId="1" contactPoint="end"/></link>')
    first, second = source.split('<road id="2"', 1)
    second = second.replace('<lane id="1" type="driving">', '<lane id="1" type="driving"><link><predecessor id="1"/></link>')
    source = first + '<road id="2"' + second
    lanes = (LaneSegment(1, 0, 1, 1.999, .001, ((1.999, 1.75, 0), (.001, 1.75, 0))),
             LaneSegment(2, 0, 1, 1.999, .001, ((3.999, 1.75, 0), (2.001, 1.75, 0)), ("1:0:1",)))
    document = replace(document, lanes=lanes, opendrive_sha256=hashlib.sha256(source.encode()).hexdigest())
    assert audit_lane_topology(source, document)["successors_checked"] == 1
