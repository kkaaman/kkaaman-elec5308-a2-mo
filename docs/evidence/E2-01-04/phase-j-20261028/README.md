# PR103 / issue52 LiDAR continuation

AI-assisted with OpenAI Codex. Original contributor history is preserved.
Complete production review and supported loader-only rebinding are recorded
adjacent, including the intermediate binding before capture-schema completion.

Corrected unsupported positive clearance: sparse road returns at 7.6-7.7m
could report 8m extent despite an unobserved near prefix. Descriptive extent
now never supplies positive clearance or releases HOLD. Clustering grid and
component work are bounded; overload invalidates the sweep. Capture COMPLETE
requires known-object checks and settings restoration/no leaked actors.

Capture now covers 10/20/30m at center/left/right offsets in one parked
session: 9 cases, 10 frames each, 90 sweeps. Recorded target footprint geometry
is evaluation-only, never policy input. Replay measures target-to-cluster
corridor classification; missing geometry is unknown, never fabricated.

Final offline validation: 67 focused and 1067 full tests passed, 1 CARLA
deselected, 3 subtests. Admission and all 19 negative probes pass; Ruff,
boundary (96 modules), strict Sphinx and diff checks pass. Synthetic fixture
evidence remains synthetic. The prior pending source blocker is resolved.

Status: CARLA BLOCKED. Issue52 stays open for actual known-object capture and
recorded replay. CARLA stayed OFF. No #40 positive-clearance or #41 sensor
false-positive resolution is claimed.

When explicitly authorized later, use CARLA0.9.16 Town10HD_Opt, empty world,
no competing synchronous client, fixed dt0.05:
python scripts/capture_lidar_sweeps.py --host 127.0.0.1 --distances 10 20 30 --lateral-offsets 0 4.5 -4.5 --frames 10 --output runs/lidar-capture-v1
python scripts/replay_lidar_obstacles.py --input runs/lidar-capture-v1/sweeps.json --output runs/lidar-capture-v1/replay.json
Retain every sweep, identity, coordinate/intensity/density/coverage check,
target classification, source hash and cleanup record. Known-object face error
must meet the configured 0.3m tolerance. Without RGB-D, fused HOLD is expected.
