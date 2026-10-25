"""BL-B-07: collision/lane-invasion records, deduplication and legal exclusions.

The S1-S5 records must be counted exactly as the live S0 runner's records are, so
the same 0.5 s contact rule and the same S0 incident metric apply to every scenario.
"""

from types import SimpleNamespace

import pytest

from carla_adapter.s0_runner import RunSensors
from evaluation.s0_metrics import _incident_metrics
from evaluation.scenario_events import (
    CollisionContacts,
    collision_record,
    count_contacts,
    count_unplanned_invasions,
    lane_invasion_record,
    validate_exclusions,
)

STREAM_OK = {
    "kind": "event_stream_status",
    "collision_complete": True,
    "lane_invasion_complete": True,
}
CALLBACKS = [  # (other actor, frame, time)
    (7, 100, 5.00),
    (7, 101, 5.30),
    (7, 110, 5.75),  # 0.45 s after the previous callback: same contact
    (8, 111, 5.80),  # another actor: its own contact
    (7, 122, 6.31),  # 0.56 s after the previous callback: new contact
    (None, 130, 7.00),
    (None, 130, 7.00),  # unknown actors never merge
]


def _records():
    contacts = CollisionContacts()
    return [
        collision_record(
            contacts,
            frame=frame,
            sim_time_s=t,
            other_actor_id=actor,
            other_actor_type="vehicle.tesla.model3",
            impulse=(1.0, 0.0, 0.0),
        )
        for actor, frame, t in CALLBACKS
    ]


def test_contact_rule_merges_callbacks_within_half_a_second():
    records = _records()
    assert [r["contact_id"] for r in records[:5]] == ["7:100", "7:100", "7:100", "8:111", "7:122"]
    assert records[5]["contact_id"] != records[6]["contact_id"]
    assert count_contacts(records) == 5
    assert len(records) == len(CALLBACKS)  # raw callbacks are kept, not replaced by counts


def test_contact_ids_match_the_live_s0_runner():
    events = []
    sensors = RunSensors(SimpleNamespace(), None, {"camera_width": 4, "camera_height": 4}, events)
    for actor, frame, t in CALLBACKS[:5]:
        sensors._collision(
            SimpleNamespace(
                frame=frame,
                timestamp=t,
                other_actor=SimpleNamespace(id=actor, type_id="vehicle.tesla.model3"),
                normal_impulse=SimpleNamespace(x=1.0, y=0.0, z=0.0),
            )
        )
    assert [e["contact_id"] for e in events] == [r["contact_id"] for r in _records()[:5]]


def test_window_must_be_positive():
    for bad in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            CollisionContacts(bad)


EXCLUSIONS = validate_exclusions(
    [{"id": "turn-532", "from_s_m": 105.9, "to_s_m": 151.6, "markings": ["Broken"], "reason": "x"}]
)


def _lane(sequence, s, markings):
    return lane_invasion_record(
        sequence=sequence,
        frame=200 + sequence,
        sim_time_s=10.0 + sequence,
        markings=markings,
        route_s_m=s,
        exclusions=EXCLUSIONS,
    )


def test_only_declared_crossings_are_legal():
    records = [
        _lane(1, 120.0, ["Broken"]),  # declared interval and marking
        _lane(2, 120.0, ["Broken", "Solid"]),  # one undeclared marking
        _lane(3, 40.0, ["Broken"]),  # outside the interval
        _lane(4, None, ["Broken"]),  # no route position: never legal
    ]
    assert [r["predeclared_legal"] for r in records] == [True, False, False, False]
    assert records[0]["legal_manoeuvre_id"] == "turn-532"
    assert count_unplanned_invasions(records) == 3


def test_s0_incident_metric_counts_these_records_the_same_way():
    records = _records() + [_lane(1, 120.0, ["Broken"]), _lane(2, 40.0, ["Solid"])]
    metrics = _incident_metrics(records + [STREAM_OK], 0)
    assert metrics["collisions"]["value"] == count_contacts(records) == 5
    assert metrics["collisions"]["raw_event_count"] == len(CALLBACKS)
    assert metrics["unplanned_lane_invasions"]["value"] == count_unplanned_invasions(records) == 1
    assert metrics["unplanned_lane_invasions"]["excluded_legal_event_count"] == 1


@pytest.mark.parametrize(
    "entry",
    [
        {"id": "a", "from_s_m": 2, "to_s_m": 1, "markings": ["Broken"], "reason": "x"},
        {"id": "a", "from_s_m": 1, "to_s_m": 2, "markings": [], "reason": "x"},
        {"id": "a", "from_s_m": 1, "to_s_m": 2, "markings": ["Broken"], "reason": " "},
        {"id": "", "from_s_m": 1, "to_s_m": 2, "markings": ["Broken"], "reason": "x"},
        {"from_s_m": 1, "to_s_m": 2, "markings": ["Broken"], "reason": "x"},
    ],
)
def test_bad_exclusions_are_refused(entry):
    with pytest.raises(ValueError):
        validate_exclusions([entry])


def test_baseline_route_declares_no_legal_crossings():
    import json
    from pathlib import Path

    route = json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs/routes/town10hd_baseline_v1.json"
        ).read_text()
    )
    assert validate_exclusions(route["legal_lane_invasion_exclusions"]) == ()
