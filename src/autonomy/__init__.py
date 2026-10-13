"""Contracts, the autonomy stack and safety arbitration (workstream F).

Everything that crosses a package boundary is defined here as plain data, so
core code never sees a simulator type.

Planned modules:

* ``interfaces.py``: FrameBundle, ego state, Detector/ObstacleSource protocols,
  ControlCommand, MotionState, Graph/Mission, Diagnostics
  (BL-F-01 Saviour, BL-C-01 Ricky, E5-01 Zhaolin)
* ``stack.py``: AutonomyStack lifecycle and ``step``, timing, emergency stop
  (BL-F-02, BL-F-06 Saviour)
* ``paths.py``: repository locations that do not depend on the working
  directory (available now)
"""
