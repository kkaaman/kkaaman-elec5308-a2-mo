# configs/

YAML configuration for the stack, sensors, vehicles, scenarios and releases.
Load files through `autonomy.paths.config_path("stack")` rather than relative
paths.

**Conventions:**
- SI units throughout: metres, seconds, radians, m/s. Put the unit in the
  key name when it isn't obvious, for example `stop_margin_m` or `dt_s`.
- No absolute paths, and no secrets.
- Every flag for an optional feature defaults to off. The baseline must run
  offline.
- A threshold frozen at a gate changes only with a dated entry in
  `docs/decisions.md`.

| File / folder | Contents | Task (owner) |
|---|---|---|
| `stack.yaml` | Stack lifecycle, timing and controller settings | BL-F-02 (Saviour), BL-A-05 (Zhaolin), BL-D-09 (Alan) |
| `acceptance.yaml` | Frozen speed and acceptance thresholds | BL-D-06 (Alan) |
| `models.yaml` | Model manifest: files, checksums, versions | BL-C-03 (Ricky) |
| `sensors.yaml` | Camera, depth and LiDAR rig | E2-01 (Mohammad) |
| `vehicle_<blueprint>.yaml` | Measured vehicle geometry and actuator response | BL-E-02/03 (Abdullah) |
| `destinations.yaml` | Named Wayfinder destinations | E3-03 (Abdullah) |
| `release.yaml` | Demo and release configuration | G3-01 (Alan), G4-01 (Mohammad) |
| `scenarios/` | S0–S5, L1, P1/P2, WF1 and fault scenarios | BL-B-06 (Mohammad), scenario owners |
| `weather/` | Weather presets for E1 | E1-01 (Alan) |
