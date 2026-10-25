"""HUD, overlays, lane-departure monitor and mini-map (workstream E, E3).

Subscribes to diagnostics only and never owns control state. The A1
``overlay_enhancements`` helpers move here behind a compatibility shim
(RV-06, E3-03 Abdullah).

Implemented: :mod:`visualisation.lane_departure` (image offset versus map CTE
monitor) and :mod:`visualisation.hud` (overlay hook). See
``docs/lane_departure_monitor.md``.
"""
