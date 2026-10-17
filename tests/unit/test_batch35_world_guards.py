"""Control-expiry and exception-cleanup guards without a running simulator."""

from types import SimpleNamespace

import pytest

from autonomy.interfaces import ControlPacket, Diagnostics
from carla_adapter.baseline_world import CarlaWorld
from navigation_control.types import ControlCommand


def fixture():
    world = CarlaWorld.__new__(CarlaWorld)
    calls = []
    world._world = SimpleNamespace(
        get_snapshot=lambda: SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=1.0))
    )
    world._episode = "episode"
    world._last_bundle = SimpleNamespace(frame_id=5, sim_time_s=0.8)
    world.adapter = SimpleNamespace(
        stop=lambda: calls.append("stop"), apply=lambda command: calls.append(command)
    )
    world._scenario_events = lambda now: calls.append("events")
    world._tick = lambda: calls.append("tick")
    world._read = lambda: "next-observation"
    return world, calls


def packet(**changes):
    values = dict(
        episode_id="episode",
        source_frame_id=5,
        source_sim_time_s=0.8,
        valid_until_sim_time_s=1.1,
        command=ControlCommand(0, 0.2, 0),
        diagnostics=Diagnostics(),
    )
    values.update(changes)
    return ControlPacket(**values)


@pytest.mark.parametrize(
    "changes",
    [
        {"episode_id": "old"},
        {"source_frame_id": 4},
        {"valid_until_sim_time_s": 0.9},
        {"source_sim_time_s": 0.75},
    ],
)
def test_stale_or_foreign_control_stops_before_application(changes):
    world, calls = fixture()
    with pytest.raises(ValueError):
        world.step(packet(**changes))
    assert calls == ["stop"]


def test_current_control_is_applied_once_before_one_tick():
    world, calls = fixture()
    command = packet()
    assert world.step(command) == "next-observation"
    assert calls == [command.command, "events", "tick"]
    assert world.last_simulation_age_s == pytest.approx(0.2)
    assert world.last_applied_monotonic_s > 0
