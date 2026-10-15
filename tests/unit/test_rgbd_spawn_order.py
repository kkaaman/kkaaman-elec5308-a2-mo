"""Wheel geometry is read only after the spawned vehicle has physics frames."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_script():
    path = Path(__file__).resolve().parents[2] / "scripts/check_rgbd_capture.py"
    spec = importlib.util.spec_from_file_location("rgbd_smoke_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spawn_brakes_then_ticks_then_reads_geometry(monkeypatch):
    module = load_script()
    events = []
    vehicle = SimpleNamespace(
        apply_control=lambda control: events.append(("brake", control))
    )
    owner = SimpleNamespace(tick=lambda: events.append("tick"))
    result = object()

    def construct(value):
        assert value is vehicle
        assert len(events) == 21 and events[1:] == ["tick"] * 20
        events.append("geometry")
        return result

    monkeypatch.setattr(module, "FixedVehicleAdapter", construct)
    control = object()
    assert module.prepare_parked_adapter(owner, vehicle, control) is result
    assert events[0] == ("brake", control)
    assert events[-1] == "geometry"


def test_invalid_settled_geometry_is_not_bypassed(monkeypatch):
    module = load_script()

    def invalid(vehicle):
        raise ValueError("invalid wheel geometry")

    monkeypatch.setattr(module, "FixedVehicleAdapter", invalid)
    with pytest.raises(ValueError, match="invalid wheel geometry"):
        module.prepare_parked_adapter(
            SimpleNamespace(tick=lambda: None),
            SimpleNamespace(apply_control=lambda c: None),
            object(),
        )
