# Phase J: LiDAR validation and remaining acceptance

AI-assisted with OpenAI Codex, 28 October 2026. Canonical `ae6a83d` merged
normally; original Mohammad commits preserved.

Freshness fix: future simulation timestamps cannot supply a gap or clearance;
nonfinite maximum ages are rejected. Sixty focused LiDAR tests pass; Ruff,
boundary smoke (90 modules) and strict Sphinx pass. Synthetic replay reports
10 frames, 8 valid sweeps, 7 fused valid, 3 HOLD, maximum gap error 0.034831 m
and zero depth-loss frames treated clear. This is synthetic evidence only.

Before the freshness fix, full integrated tests: 972 passed, 6 failed,
1 CARLA deselected, 3 subtests. All failures are exact-source G1 admission
mismatches. A final full regression log is recorded alongside this note.

The adjacent source-binding proposal does not modify the trusted record or
claim designated corridor-owner/admission review. Issue #52 acceptance remains
incomplete: corridor-owner protocol acceptance, live known-object capture,
recorded road/roadside/in-corridor replay and admission review are outstanding.
Issue #40 must not privately duplicate this unmerged capability.

## CARLA run queue: first session, parked LiDAR capture

Prerequisites: reviewed/admitted candidate, matching CARLA 0.9.16 client/server,
Town10HD_Opt, empty world, no other synchronous client; local project installed.
Keep the ego parked. From this PR checkout, in PowerShell:

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py scripts/capture_lidar_sweeps.py --host 127.0.0.1 --output runs/lidar-capture-v1
& $py scripts/replay_lidar_obstacles.py --input runs/lidar-capture-v1/sweeps.json --output runs/lidar-capture-v1/replay.json
```

Retain all sweeps/arrays, script/source identity, frame IDs, simulation times,
point counts/density/coverage, known-object coordinate/intensity results at
10/20/30 m, cleanup/restoration status and replay rows. Preserve failures.
This capture has no RGB-D channel: fused HOLD is expected and must not be
reported as joint RGB-D/LiDAR clearance or stop/resume acceptance.
