ELEC5308 FINAL WORKING LIVE DEMO
================================

Frozen: 2 November 2026

DEMO ONLY - NOT ACCEPTANCE EVIDENCE

LIVE-VALIDATED BEHAVIOUR
------------------------
- Town10HD_Opt 900 m route.
- Real lane segmentation HUD working correctly.
- Lane display stable through route/junctions.
- Stop sign:
  - perception-armed
  - route-based stop-line target
  - front-bumper correction
  - stops before line
  - holds for 2 seconds
  - resumes correctly
  - no erroneous second stop after intersection
- Traffic-light colour is image-perception-derived.
- First intersection: GREEN correctly detected.
- Second intersection: GREEN correctly detected.
  - confirmation can occur relatively late after the preceding right turn,
    close to the line.
- Third intersection: GREEN correctly detected.
- Later neighbouring-lane return to first intersection:
  GREEN correctly detected.
- Traffic-light source locking and junction confirmation behaved correctly.
- No traffic-light braking/control is implemented.
- CARLA traffic-light state is not used to determine displayed colour.

FINAL VIEWER
------------
colour_candidate_junction_confirmation_viewer_3000.py

FINAL RUNNER
------------
FROZEN_PERFECT_stop_sign_runner_3000.py

TRAFFIC-LIGHT CLASSIFIER
------------------------
traffic_light_color_candidate.py

CARLA
-----
CARLA 0.9.16
Town10HD_Opt
Host: 172.28.176.1
RPC port: 3000

Known stable launch configuration during testing:
DX12
Low quality
normal rendering
port 3000

IMPORTANT
---------
Do not modify these frozen copies.
Make another copy before any further experimentation.
