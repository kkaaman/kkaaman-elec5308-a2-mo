"""Tests that need a live CARLA 0.9.16 server.

Run on a CARLA machine with ``python -m pytest tests/carla``. Each module here
must begin with ``carla = pytest.importorskip("carla")`` so it is reported as
skipped, not failed or silently dropped, where CARLA is not installed.
"""

from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    """Tag every test in this folder with the ``carla`` marker."""
    for item in items:
        if _HERE in Path(item.path).resolve().parents:
            item.add_marker(pytest.mark.carla)
