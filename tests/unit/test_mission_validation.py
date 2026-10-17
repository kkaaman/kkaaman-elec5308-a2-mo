"""BL-A-04 live validation: overlapping junction lanes on the fixed mission.

CARLA's get_waypoint returns one lane where junction lanes merge or diverge;
on the live server it returned 934:0:-2 (straight from road 13) at the exit of
the mission's right turn 939:0:-1, where both lanes join 14:0:-2.
"""

import math

import numpy as np

from autonomy.paths import FIXTURES_DIR
from planning.lane_graph import LaneTopology
from planning.mission import overlapping_junction_lane, select_mission

TOPOLOGY = LaneTopology.load(FIXTURES_DIR / "Town10HD.lane_graph.json")
MISSION = select_mission(TOPOLOGY)


def test_merge_into_the_turn_exit_is_accepted():
    assert MISSION.segment_ids[:3] == ("12:0:-1", "939:0:-1", "14:0:-2")
    assert overlapping_junction_lane(TOPOLOGY, MISSION.segment_ids, "934:0:-2")


def test_unrelated_lanes_are_still_rejected():
    by_id = {lane.segment_id: lane for lane in TOPOLOGY.lanes}
    assert not overlapping_junction_lane(TOPOLOGY, MISSION.segment_ids, "13:0:-2")
    assert not overlapping_junction_lane(TOPOLOGY, MISSION.segment_ids, "no:such:lane")
    junctions = {by_id[k].junction_id for k in MISSION.segment_ids if by_id[k].junction_id}
    rejected = [
        lane.segment_id
        for lane in TOPOLOGY.lanes
        if lane.junction_id in junctions
        and lane.segment_id not in MISSION.segment_ids
        and not overlapping_junction_lane(TOPOLOGY, MISSION.segment_ids, lane.segment_id)
    ]
    assert len(rejected) > 10


def nearest_lane(x, y):
    """Fake get_waypoint: nearest centreline; ties resolved by lane-id order."""
    best = None
    for lane in sorted(TOPOLOGY.lanes, key=lambda item: item.segment_id):
        points = np.asarray(lane.points_xyz_m)[:, :2]
        a, b = points[:-1], points[1:]
        d = b - a
        t = np.clip(((np.array([x, y]) - a) * d).sum(1) / (d**2).sum(1), 0, 1)
        distance = float(np.linalg.norm(a + t[:, None] * d - [x, y], axis=1).min())
        if best is None or distance < best[0] - 1e-6:
            best = (distance, lane.segment_id)
    return best


def test_every_mission_sample_passes_with_the_overlap_rule_only():
    allowed = set(MISSION.segment_ids)
    overlaps, rejected = set(), []
    for a, b in zip(MISSION.points_xyz_m, MISSION.points_xyz_m[1:]):
        for ratio in np.linspace(0, 1, max(2, math.ceil(math.dist(a, b) / 0.5) + 1)):
            x, y = (np.asarray(a[:2]) + ratio * (np.asarray(b[:2]) - a[:2])).tolist()
            distance, segment = nearest_lane(x, y)
            assert distance <= 0.20
            if segment in allowed:
                continue
            if overlapping_junction_lane(TOPOLOGY, MISSION.segment_ids, segment):
                overlaps.add(segment)
            else:
                rejected.append(segment)
    assert not rejected
    # The old strict rule would have failed: CARLA-style ties do pick overlaps.
    assert "934:0:-2" in overlaps
