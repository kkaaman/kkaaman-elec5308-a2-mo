# E1 weather robustness matrix — issue #49

This is an explicit experiment specification and read-only preflight, not a
captured dataset or robustness claim. Every planned cell remains **NOT RUN**.

The matrix pairs S0/S1/S2 with ClearNoon, HardRainNoon and ClearNight, using three
identical scene seeds across conditions: 27 cells total. The JSON records all
14 weather fields, fixed 0.05 s steps and explicit vehicle-light names. Night
uses Position + LowBeam; daytime vehicle lights are off. Numeric values follow
[CARLA 0.9.16 source](https://github.com/carla-simulator/carla/blob/0.9.16/LibCarla/source/carla/rpc/WeatherParameters.cpp).
Sun azimuth is explicitly overridden from the inherited -1 sentinel to 0 degrees
in every condition. Installed attributes/presets must pass preflight; mismatches
are errors rather than silent substitutions. This is not installed-version evidence.

## Prepare offline

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py scripts/prepare_weather_matrix.py --output runs/weather-v1/matrix.json
& $py -m pytest tests/unit/test_weather_matrix.py
```

Output uses exclusive creation to preserve previous attempts. Each cell includes
scenario, condition, repetition, paired seed, full weather/light settings and
NOT_RUN status. Twelve tests cover pairing, missing/nonfinite fields, inherited
azimuth, night lighting, bad seeds and fake installed preset/attribute drift.

## Live preflight and dataset queue

Prerequisites: matching CARLA 0.9.16 client/server, Town10HD_Opt, local admitted
models, final source-admission review. When CARLA is available:

```powershell
& $py -m carla_adapter.weather_experiment --host 127.0.0.1 --output runs/weather-v1/installed-settings.json
```

This reads actual versions/map and verifies all preset fields, explicit override
round trips and light flags; it never applies weather, ticks or spawns actors.
The output explicitly remains NOT_CAPTURED. Actual paired sensor capture needs
the final S1/S2 runner and positive clearance/stop-resume path from #40/#41.
Do not run the current S0 runner and label it rain/night: its baseline environment
owner deliberately installs ClearNoon. Wire the verified condition into an
extension-only environment owner with restoration checks after the runner
prerequisites are complete; baseline settings stay frozen.

For every cell, retain configuration/source/model hashes, actual weather and
light readback, seed, timestamps/frame/session IDs, RGB and depth frames,
verified car/person boxes with visibility/occlusion, control/event logs and
cleanup/restoration results. Repeat identical scene placement and visibility
windows across conditions; do not tune confidence/model/tracker on this held-out
set. Failed captures and invalid-depth cases remain visible with denominators.
Document measured RGB/depth/LiDAR effects separately: rain/night appearance
cannot establish a depth or LiDAR degradation. LiDAR is optional and stays
NOT_RUN when absent. Collection and labels are not implemented/verified here;
issue #49 remains open for that work after its runner prerequisites.

## Admission and attribution

The new config/adapter changes the exact G1 source inventory. The trusted record
is unchanged; reviewed rebinding is needed before integrated operational use.
AI-assisted with OpenAI Codex, 28 October 2026; no human or live approval claimed.
