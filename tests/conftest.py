"""Shared pytest configuration."""

import socket
import sys
from pathlib import Path

import pytest

SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIRECTORY))


# Unit/integration CI is offline: real model downloads must be an explicit
# asset-preparation step outside pytest. Existing model fakes remain usable.

@pytest.fixture(autouse=True)
def forbid_network_in_offline_tests(monkeypatch, request):
    if request.node.get_closest_marker("carla"):
        return

    def blocked(*args, **kwargs):
        raise AssertionError("Network disabled in offline tests; use a fake or a local asset")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
