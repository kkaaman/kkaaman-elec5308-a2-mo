"""Mapless, truth-free driving stack (A2 extension).

The car sees only what its sensors report. It receives a sat-nav route: sparse GPS points,
each with a high-level command. This follows the rules of the CARLA Leaderboard 2.0
SENSORS track:
- no HD map;
- no simulator pose, actor or waypoint queries;
- no ground-truth sensors (depth, segmentation, semantic LiDAR, obstacle detector).

Nothing in this package imports the simulator, the evaluator or the map tooling.
``tests/unit/test_mapless_firewall.py`` enforces this.

Vehicle frame: right-handed, +x forward, +y left, +z up, origin at the rear axle.
"""
