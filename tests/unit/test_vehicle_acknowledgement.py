"""Verify server-acknowledged commands without advancing simulation frames."""

import math
import sys
from types import SimpleNamespace

import pytest

from carla_adapter.vehicle import FixedVehicleAdapter  # noqa: TID251
from navigation_control import ControlCommand


@pytest.fixture
def adapter(monkeypatch):
    def vehicle_control(**values):
        defaults = dict(throttle=0.0, brake=0.0, steer=0.0, reverse=False)
        defaults.update(values)
        return SimpleNamespace(**defaults)

    monkeypatch.setitem(
        sys.modules,
        "carla",
        SimpleNamespace(
            VehicleControl=vehicle_control,
            command=SimpleNamespace(
                ApplyVehicleControl=lambda actor_id, control: (actor_id, control)
            ),
        ),
    )
    value = FixedVehicleAdapter.__new__(FixedVehicleAdapter)
    value.geometry = SimpleNamespace(
        physical_steering_limit_rad=math.radians(70),
        controller_steering_limit_rad=math.radians(20),
    )
    controls = []
    value.vehicle = SimpleNamespace(id=17, is_alive=True, apply_control=controls.append)
    value.state = lambda: None
    return value, controls


def test_acknowledged_path_waits_for_response_and_does_not_request_tick(adapter):
    value, ordinary_controls = adapter
    batch_calls = []
    received_response = []

    def apply_batch_sync(commands, do_tick):
        batch_calls.append((commands, do_tick))
        received_response.append(True)
        return [SimpleNamespace(error="")]

    client = SimpleNamespace(apply_batch_sync=apply_batch_sync)
    control = value.apply(ControlCommand(math.radians(10), 0.4, 0), client=client)
    assert received_response == [True]
    assert batch_calls == [([(17, control)], False)]
    assert control.steer == pytest.approx(-10 / 70)
    assert control.throttle == 0.4
    assert control.brake == 0
    assert control.reverse is False
    assert ordinary_controls == []


@pytest.mark.parametrize(
    "responses",
    [
        [],
        [SimpleNamespace(error="Actor not found")],
        [SimpleNamespace(error=""), SimpleNamespace(error="")],
    ],
)
def test_missing_failed_or_extra_acknowledgements_attempt_full_braking(adapter, responses):
    value, ordinary_controls = adapter
    calls = []

    def apply_batch_sync(commands, do_tick):
        calls.append((commands, do_tick))
        return responses

    with pytest.raises(RuntimeError, match="not acknowledged"):
        value.apply(
            ControlCommand(0, 0.4, 0), client=SimpleNamespace(apply_batch_sync=apply_batch_sync)
        )
    assert len(calls) == 1
    assert calls[0][1] is False
    assert len(ordinary_controls) == 1
    assert ordinary_controls[0].brake == 1.0
    assert ordinary_controls[0].throttle == 0.0


def test_acknowledgement_timeout_is_preserved_and_attempts_braking(adapter):
    value, ordinary_controls = adapter

    def apply_batch_sync(commands, do_tick):
        raise TimeoutError("Server acknowledgement timed out")

    with pytest.raises(TimeoutError, match="acknowledgement timed out"):
        value.apply(
            ControlCommand(0, 0.4, 0), client=SimpleNamespace(apply_batch_sync=apply_batch_sync)
        )
    assert len(ordinary_controls) == 1
    assert ordinary_controls[0].brake == 1.0


def test_invalid_state_never_sends_an_acknowledged_drive_command(adapter):
    value, ordinary_controls = adapter

    def invalid_state():
        raise ValueError("Reverse motion is unsupported")

    value.state = invalid_state
    with pytest.raises(ValueError, match="Reverse motion"):
        value.apply(ControlCommand(0, 0.4, 0), client=SimpleNamespace())
    assert len(ordinary_controls) == 1
    assert ordinary_controls[0].brake == 1.0


def test_oversized_steering_never_sends_an_acknowledged_drive_command(adapter):
    value, ordinary_controls = adapter
    with pytest.raises(ValueError, match="steering cap"):
        value.apply(ControlCommand(math.radians(21), 0.4, 0), client=SimpleNamespace())
    assert len(ordinary_controls) == 1
    assert ordinary_controls[0].brake == 1.0
