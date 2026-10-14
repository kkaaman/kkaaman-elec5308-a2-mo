"""Routes, corridor, speed governor and behaviour (workstreams A, D and E3/E5).

Implemented:

* ``lane_graph.py``: plain lane topology, deterministic JSON and conversion to
  the existing LaneGraph contract (BL-A-02 Zhaolin)
* ``lane_graph_audit.py``: independent raw OpenDRIVE connectivity/geometry
  checks for the Town10HD fixture (BL-A-02)

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
