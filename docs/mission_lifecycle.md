# Fixed mission lifecycle (BL-A-06)

`autonomy.mission.MissionEpisode` owns one `PathFollower` for one episode. Its
constructor installs the validated reference exactly once. It exposes route
identity, installation count, progress, failure reason and immutable lifecycle
events; it has no path replacement or reset method. Construct another instance
for the next episode.

```python
from autonomy.mission import MissionEpisode
from planning.baseline_mission import load_frozen_mission

mission = load_frozen_mission("configs/mission_baseline.json")
episode = MissionEpisode(mission.controller_config, mission.waypoints, mission.route_sha256)
command = episode.step(rear_axle_state, mission.config["cruise_speed_mps"], 0.05)
paused = episode.step(rear_axle_state, mission.config["cruise_speed_mps"], 0.05, hold=True)
resumed = episode.step(rear_axle_state, mission.config["cruise_speed_mps"], 0.05)
```

The caller supplies a fresh `VehicleState` for each step and stops actuators if
an update raises. A hold requests zero speed without clearing progress. It also
suppresses new completion near the endpoint; releasing the hold restores normal
completion eligibility. A previously finished episode remains latched and
braked. Endpoint distance, stopping speed and the existing endpoint braking
ceiling remain unchanged.

`PathFollower.update(..., completion_allowed=True)` adds a keyword-only gate.
The default preserves existing caller behavior. Mission orchestration passes
`False` for an explicit temporary obstacle hold, including inside goal tolerance.
This API verifies the control boundary; obstacle detection and the source of
the hold request remain separate responsibilities.

```powershell
python -m pytest -q tests/unit/test_mission.py tests/unit/test_forward_mission.py tests/navigation_control
```

Regression coverage includes straight/curved holds, near-goal holds, resume
without route replacement, ordinary endpoint braking, completed-state latching,
fresh episodes, rejected inputs and retained failure history.
