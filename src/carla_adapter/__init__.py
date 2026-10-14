"""CARLA integration adapter: the only package allowed to import ``carla``.

Converts simulator objects to the plain contracts in ``autonomy.interfaces``.
Planned split of ``adapter.py``:

* ``world.py``: synchronous world and cleanup lifecycle (BL-B-01 Mohammad)
* ``vehicle.py``: vehicle geometry and actuator translation
  (BL-E-01 Abdullah, E5-05 Zhaolin)
* ``sensors.py``: aligned RGB-D (and LiDAR) capture, bounded queues
  (BL-B-02, E2-02 Mohammad)
* ``topology.py``: lane topology exported as a plain graph (BL-A-02 Zhaolin)
* ``scenario.py``: deterministic obstacle scenarios (BL-B-06 Mohammad)
* ``ground_truth.py``: evaluation-only truth packets (BL-B-03, BL-B-07 Mohammad)
* ``carla_world.py``: CarlaWorld wiring the stack to CARLA (BL-E-04 Abdullah)

The A1 functions below stay importable from here.
"""

__all__ = [
    "apply_command",
    "carla_vehicle_to_state",
    "command_to_carla_control",
]


def __getattr__(name):
    """Keep A1 exports while deferring the SDK until they are requested."""
    if name in __all__:
        from . import adapter

        return getattr(adapter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
