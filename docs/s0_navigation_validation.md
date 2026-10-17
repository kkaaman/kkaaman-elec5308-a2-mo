# S0 navigation execution and evidence (TEST-02)

This entry point runs the frozen Town10HD mission only. It reuses the fixed
Tesla adapter, Pure Pursuit and synchronous world owner. It does not implement
the S1-S5 obstacle/perception stack.

Current status: RGB-enabled five-run S0 acceptance passed on the minimum-spec
laptop using the isolated Windows readback/thread-protection repair. The
original executable still has the diagnosed camera defect. See the
[repair instructions](carla_rgb_readback_fix.md) and retained failed trials.

## Reproduce

Use the project environment with CARLA client/server 0.9.16 and the verified
isolated runtime. Close existing CARLA instances before launching. This
machine has the existing MinGW-w64 compiler shown below.

```powershell
$py = '.\.venv\Scripts\python.exe'
$runtime = 'D:\26-S2-Assignment2-Group7\runs\carla-rgb-runtime'
& $py scripts/patch_carla_rgb_readback.py --carla-root 'D:\CARLA_0.9.16' --runtime $runtime --cxx 'D:\mingw64\bin\g++.exe'
if ($LASTEXITCODE -ne 0) { throw 'Runtime verification failed.' }
Start-Process -FilePath "$runtime\CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe" -WorkingDirectory $runtime -ArgumentList '-dx11 -windowed -ResX=640 -ResY=360 -quality-level=Low -nosound' -WindowStyle Normal
# Wait for the map before running the following commands.
& $py scripts/check_baseline_mission.py --verify-map
$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
& $py scripts/run_scenario.py --runs 5 --output "runs/S0/acceptance-$runId"
```

Every output directory must be new. Five-run mode rejects smoke time limits,
injected holds, navigation-only execution and RGB-disabled configurations.
It evaluates every run; a failed run remains in the suite and its directory.
Never collect five successful trials by omitting failed trials. Changes to
route/configuration require a new complete suite with a new identity.

The frozen minimum-memory scene uses the optimized map with optional layers
`NONE`; base roads, sidewalks, traffic lights and signs remain. Each run first
checks for foreign actors, enables synchronous 0.05 s timing, and loads a fresh
world with the same map and declared layers while preserving those settings.
It then owns its actor/controller/tracker state. Initial world settings, actor
inventory, clear weather and traffic-light changes are verified during cleanup.
State restoration uses owned ticks before releasing synchronization because
CARLA traffic-light getters read simulation snapshots.

## Measurement contract

The limits mirror `tests/fixtures/acceptance_definitions.json` and
[the acceptance definitions](acceptance.md). Five real runs must each finish
with zero collisions, unplanned invasions and false stops. Worst-run mean
absolute rear-axle CTE must be <=0.3 m, stable-cruise speed RMS <=0.5 m/s and
capture-to-apply p95 <=100 ms.

CTE is measured against the continuous fixed polyline while moving. Stable
cruise is selected by phase and unchanged target for two simulation seconds;
measured speed error never selects the window. Signed CTE and sample counts
remain in the raw trace. Percentiles use linear interpolation at `(n-1)*0.95`.
Collision contacts are grouped by other actor while callbacks remain within
0.5 simulation seconds; every raw callback remains available. No lane event
is automatically excluded merely because it occurred in a junction.

RGB callbacks timestamp receipt before buffer access. Exact camera frame/time
is joined to saved ego state. Navigation uses current rear-axle state; camera
frame age is reported separately. The application timestamp follows checked
`client.apply_batch_sync(..., False)` acknowledgement. Ordinary
`vehicle.apply_control()` dispatches asynchronously and cannot establish this
endpoint. See the pinned [CARLA client implementation](https://github.com/carla-simulator/carla/blob/0.9.16/LibCarla/source/carla/client/detail/Client.cpp).

The runner never advances the world through the command batch. Missing
observations, missing metric windows, incomplete incident streams and missing
provenance are insufficient evidence, not zero errors. Callback streams are
drained and fenced before an immutable event snapshot is written. A lost
server causes a bounded health check and explicit unknown cleanup state.

Each run retains `manifest.json`, `samples.csv`, `events.json` and `summary.json`.
The manifest records effective configuration, map/route/source hashes, versions,
world episode IDs, vehicle geometry and cleanup checks. The suite summary
retains every per-run result. Source edits during a run invalidate its evidence.

## Separate diagnostics

```powershell
python scripts/run_scenario.py --runs 1 --navigation-only --output runs/S0/navigation-NEW-ID
python scripts/run_scenario.py --runs 1 --navigation-only --hold-at-progress 127 --hold-duration 2 --output runs/S0/hold-NEW-ID
```

Navigation-only mode keeps collision/lane sensors and actual vehicle physics,
but records capture/latency fields as missing. Its result cannot pass complete
S0 acceptance. A diagnostic may exit nonzero because evidence is insufficient
or the injected hold intentionally violates S0's no-false-stop rule. Inspect
the per-run summary and hold lifecycle evidence rather than treating exit code
as proof of controller failure or successful full acceptance.

Injected HOLD verifies the mission/controller boundary; it is not a claim that
obstacle perception or a clearance governor has been validated.

## Current result

Implementation checks and real navigation diagnostics are recorded in
[the evidence report](evidence/TEST-02/README.md). Full five-run S0 acceptance
has now passed on the repaired Windows runtime. Independent camera probes and
thread stacks reproduced and localized the original package's stall without
importing the navigation stack. The report retains all earlier failures and
does not substitute navigation-only or kinematic results for real RGB trials.
