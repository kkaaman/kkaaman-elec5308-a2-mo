"""Routes, corridor, speed governor and behaviour (workstreams A, D and E3/E5).

Planned modules:

* ``route_manager.py``: continuous mission route (BL-A-04 Zhaolin)
* ``junction_planner.py``: legal junction exits stitched into lane-centre
  paths (BL-A-03 Zhaolin)
* ``corridor.py``: route-frame footprints and swept corridor (BL-D-04 Alan)
* ``speed_governor.py``: cruise, curve and obstacle speed caps (BL-D-07 Alan)
* ``behaviour.py``: CRUISE/APPROACH/STOPPING/HOLD/RESUME/EMERGENCY_STOP
  (BL-D-08 Alan)
* ``wayfinder.py``: A* on the exported lane graph (E3-01, E3-02 Abdullah)
* ``parking.py``: forward bay parking, optional reverse (E5-02 to E5-04
  Zhaolin)
"""
