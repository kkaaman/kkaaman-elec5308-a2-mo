"""Obstacles from detections, depth and LiDAR (workstreams B, C and D).

Turns detector boxes and sensor data into typed, filtered obstacles in the
vehicle, world and route frames. Ground truth is never an input here.

Planned modules:

* ``types.py``: Obstacle and ObstacleSource result types (BL-C-01 Ricky)
* ``box_depth.py``: inner-box median depth and validity checks
  (BL-D-03 Alan, BL-B-05 Mohammad)
* ``association.py``: joining tracks to depth observations
  (BL-C-06 Ricky, BL-D-04 Alan)
* ``track_filter.py``: N-of-M confirmation, median filter, expiry
  (BL-C-04, BL-C-07 Ricky)
* ``lidar_obstacles.py``: ground removal, corridor crop, clustering
  (E2-01, E2-03 Mohammad)
"""
