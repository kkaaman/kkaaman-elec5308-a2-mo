"""HUD, overlays, lane-departure monitor and mini-map (workstream E, E3).

Subscribes to diagnostics only and never owns control state. The A1
``overlay_enhancements`` helpers move here behind a compatibility shim
(RV-06, E3-03 Abdullah).
"""
