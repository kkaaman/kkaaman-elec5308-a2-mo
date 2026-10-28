# Phase J continuation execution and consolidated CARLA queue

28 October 2026, Australia/Sydney. CARLA remained OFF. Original #40 dirty work
was preserved. Normal merges/pushes preserved contributor histories: no force
push, rebase, squash or contributor branch deletion. This supersedes the
[initial snapshot](evidence/phase-j-88/initial-phase-j-report.txt).

## Completed merges

| PR | Merge commit | Issue | Validation |
|---|---|---|---|
| #100 parking harness | 7f0a80d35fd2288a8ee61a02c0c0f5bf56cf2b26 | #59 CLOSED; #60 OPEN | 77 focused/972 full; P1 8/8, S0 5/5, archive/member hashes |
| #108 weather preparation | 86a7e189776d996bc6a6e9fe4ca80ead3bbac5d7 | #49 OPEN | 12 focused/984 full; all27 paired cells NOT_RUN |
| #107 aggregation scaffold | d0ceb4f2134b8b3a2eca4e422c1e42e532a29113 | #47 OPEN | 16 focused/1000 full; identity/duplicate/fault checks, empty register BLOCKED |
| #105 client/fallback implementation scaffold | 29aed99b7c1a6c22228c26db0e40f4b2a28ae80b | #57 OPEN | 37 focused/1037 full; canonical optional control composition, HOLD retained across source/session changes |
| #109 inference service | deeefde82fee14257d5b9e4dd0cf915f89ec99fe | #56 CLOSED | 55 focused/1055 full; fresh real-model two-client/duplicate/repeated-session Docker smoke |

All full runs deselected one CARLA test and passed three subtests; server runs
retain one dependency deprecation warning. Ruff, simulator boundary, strict
Sphinx and diff checks pass. Docker smoke used network none, read-only root/
weights, 4GiB/2CPU/128PIDs/nonroot/no capabilities; temporary container removed.
Black images are protocol evidence, not accuracy/LAN performance.

CI failures were classified billing-only only after actual payment/spending
annotation, no runner and zero steps were verified (#105/#107). Missing checks
were NOT_REPORTED. No unexecuted CI check is represented as passed.

## Remaining PR dispositions

| PR | Classification | Remaining criterion |
|---|---|---|
| #99/#41 | TECHNICALLY INCOMPLETE; validated correction needs further sensor evidence | Learned detector still emits car for trash-can region, confidence0.55528. Crop verifier rejects genuine car too; width interval overlaps583/599 genuine observations. No validated sensor-only correction. Runner/site and mission runner fingerprint differ from historical admission; no blind rebind. |
| #101/#42,#45 | CARLA BLOCKED; final runner dependencies explicit | Recorded lane/curve/HUD, S3/S4/S5 and final full-workload timing. Substantive review fixes complete; 41 focused/1041 full and19 negative probes pass. |
| #102/#54 | READY component/replay scope; RV-06 integration dependency remains for whole issue | 40 focused/1095 full and19 negative probes pass. Hash-verified merged TEST-02 archive supplies24 actual pose samples,3 paused and24 new HUDs. Original HOLD false-stop FAIL retained. No live camera-overlay claim. |
| #103/#52 | CARLA BLOCKED | Actual90-sweep known-object dataset and recorded road/roadside/in-corridor replay. 67 focused/1067 full and19 negative probes pass. Sparse road extent never supplies positive clearance. |
| #88+#106/#62 | Documentation scope READY after final publication checks | Contributor PR106 merged normally into88, useful preflight/tests retained. Complete75-wheel Linux CPU profile and fresh disconnected installation verified. Whole62 remains open for final upstream evidence. |

No new human approval was invented. Complete production review uses explicit
user delegation. Source binding is separate from live/sensor acceptance.

## Admission

| Branch | Old source fingerprint | Reviewed new fingerprint |
|---|---|---|
| #100 | bb8f19fedc92a330b90ee090a928bd836612390019104d823a44482ff242203d | b0f3edc49691ce9033bc85226a2473854122a9d386f2c822d161064903f40b90 |
| #108 | b0f3edc49691ce9033bc85226a2473854122a9d386f2c822d161064903f40b90 | 664c4d019ad385f1a6a6b360bf8a776ed4fa47c157230da3c751853f21984f5c |
| #105 | 664c4d019ad385f1a6a6b360bf8a776ed4fa47c157230da3c751853f21984f5c | f34a21469f2207fdbefe53eee91b7d3913cb7e522081e25747bddd70021ffea3 |
| #109 | f34a21469f2207fdbefe53eee91b7d3913cb7e522081e25747bddd70021ffea3 | e036881ecd771d85698915075c3f6c5c7134a36de30d02f62cc3418ceff32924 |
| #88/#106 | e036881ecd771d85698915075c3f6c5c7134a36de30d02f62cc3418ceff32924 | 520921912f1f43bc6743d01ad176167d901bf07678377dfaa3382ff3cc9c7875 |
| #101 candidate | 664c4d019ad385f1a6a6b360bf8a776ed4fa47c157230da3c751853f21984f5c | b6967c2a16de3fd3b3f5b2d7ffecd2bd601992d5b3f8f6b88bf3cb5a2a4cc47e |
| #102 candidate | e036881ecd771d85698915075c3f6c5c7134a36de30d02f62cc3418ceff32924 | bf72ba45692595a89810d36f7859f1417f0321b96555b21fbdbb885f1610d496 |
| #103 candidate | 664c4d019ad385f1a6a6b360bf8a776ed4fa47c157230da3c751853f21984f5c | 639a8bc367cdf66a7921807ba978b396f397cd0353b1314d52326dac9c95d51b |

Per-branch applied-review JSON records complete inventory/classification,
config semantic/file SHA, actual model, camera/depth and record SHA. Historical
records are preserved. Open candidates require canonical refresh/revalidation
before future merge; their bindings admit no live extension operation.

Frozen values stay4m/s,1.497566177392398m/s²,0.9500000052154064s,6m;
stationary_vehicle_g2_only, pedestrian admission deferred, supported depth25m,
existing confirmation/freshness/all-track contracts. Mission semantic SHA:
2903124878485c8030e160f3598c381bf0197329f79d74773d84048a96c2e9ef.
Mission file SHA:
a4b5e669f9426db24d42d9a1d20114981c95bb15302af006b1196044696fe187.
Model SHA:
9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef.
Historical runner fingerprint:
cac8bc9c4f131b16b871191f69562d87517d3f1b956849b9950543e582ff400a.

All19 probes still reject modified/missing/added source, aux config, mission
values/scope/profile, model identity/actual bytes, camera, confirmation/depth,
runner revision/hash and record checksum. No validator was disabled.

## Official graph and issues remaining open

[Refreshed metadata](evidence/phase-j-88/continuation/official-state.json) retains
official issue bodies, dependencies and PR heads. Cycles42↔43,43↔44,43↔45,
56↔57,57↔58,66↔72 represent integration feedback.105→109 implementation landed
without inventing57 results.

- #40: supported action/positive-clearance bounds and repeated stop/HOLD/resume.
- #41/#43: validated detector correction/held-out audit. #42/#44/#45/#46:
  complete-stack lane/sensor/fault/cost and repeated live evidence.
- #47/#48: real acceptance register, failures/uncertainty, final scope and freeze.
- #49/#50/#51: actual paired weather captures/labels, robustness table/admission.
- #52/#53: recorded LiDAR evidence, performance/cost and extension admission.
- #54/#55: recorded Wayfinder HUD and route/integration/admission criteria.
- #57/#58: >=200 labelled held-out real frames per local/LAN configuration,
  hardware/model/config identity, latency/accuracy/miss tables; available cloud
  or NOT_RUN; live fault/operational extension admission.
- #60/#61: actual near-field/low-speed parking qualification; reverse evidence or
  explicitly authorized cut decision, not an invented owner decision.
- #62: final41/45/46/57 evidence and final report/claim verification. Linux CPU
  reproduction passes; historical runtime/backup CUDA remain separate.
- #63/#67/#70: selected admitted release, all-flags/combined/final sensors
  depend on missing G3 and extension decisions.
- #64/#65/#66/#68/#71/#72/#73/#74: stable measured report/media/release/demo,
  real backup-machine/hardware and archive access depend on selected release.
- #75/#76/#78/#79: actual rehearsals, Q&A, peer review and cross-training.
- #80–#85: respective humans' private submissions.

Locally actionable client/control, server, source review, capture/replay,
aggregation and reproduction tooling were completed. Remaining official work
depends on missing validated sensor contracts/data, live runs, hardware,
release decisions or human participation. No claim that all work is complete.

## CARLA RUN QUEUE — ordered to minimise simulator sessions

No new simulator run is requested until the listed preconditions are satisfied.
A NOT_READY item is a technical prerequisite, not an executable acceptance claim.
Use one installed CARLA 0.9.16/Town10HD_Opt environment, retain all failed attempts,
and record candidate/source/config/model identities for every batch.

### Session A — parked sensor qualification, #103/#52 (conditional live queue)

Offline protocol/production review and source rebinding are complete. Refresh
canonical source/admission before the future run. Require matching client/server, empty world and no
other synchronous client. Install the selected PR checkout locally, then:

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py scripts/capture_lidar_sweeps.py --host 127.0.0.1 --distances 10 20 30 --lateral-offsets 0 4.5 -4.5 --frames 10 --output runs/lidar-capture-v1
& $py scripts/replay_lidar_obstacles.py --input runs/lidar-capture-v1/sweeps.json --output runs/lidar-capture-v1/replay.json
```

Parked target: 10/20/30m × center/+4.5/-4.5m ×10 frames = 90 sweeps, dt0.05. Require face-gap error <=0.3m and all intensity/frame/density/cleanup checks.
Retain all `.npy` sweeps, `sweeps.json`, source script/hash, coordinates/intensity
checks, frame/simulation timestamps, point counts, density/coverage, cleanup/
restoration and recorded replay. Verify road/roadside/in-corridor classification,
not just file existence. This collector has no RGB-D: fused HOLD is expected and
is not joint clearance or stop/resume evidence. Inspect results before deciding
whether #52/#103 meets its actual definition of done.

In the same installed environment, #108/#49 can verify presets without driving:

```powershell
& $py -m carla_adapter.weather_experiment --host 127.0.0.1 --output runs/weather-v1/installed-settings.json
```

Weather preparation #108 is canonical; check its current reviewed identity;
do not copy private capability duplicates into another branch. It reads versions,
map, presets and lighting flags only. Actual weather dataset stays NOT_CAPTURED.

### Session B — first stop/faults, #99/#41 (NOT_READY)

Prerequisites: address the actual recorded detector false-positive defect through
#99/#43; establish supported positive clearance where needed; merge prerequisite
sensor/behaviour capabilities normally; bind the final runner/config/source.
Do not choose confidence/shape cuts from one failure example or claim the old
all-track guard is a route-clearance certificate. Pedestrian admission is deferred.

Offline admission preflight on the final #99 checkout (no motion):

```powershell
& $py -m carla_adapter.first_stop_runner --output runs/issue41/admission-final
```

Only after that preflight genuinely passes and its technical preconditions hold,
collect the bounded stationary-car stop and three fault cases in one server session:

```powershell
& $py -m carla_adapter.first_stop_runner --execute --fault none --output runs/issue41/stop-final
& $py -m carla_adapter.first_stop_runner --execute --fault capture_timeout --output runs/issue41/capture-timeout-final
& $py -m carla_adapter.first_stop_runner --execute --fault missing_depth --output runs/issue41/missing-depth-final
& $py -m carla_adapter.first_stop_runner --execute --fault invalid_depth --output runs/issue41/invalid-depth-final
```

Retain report/config/source snapshots, raw per-cycle decisions, sensor frames,
physical stop/post-fault braking samples, watchdog/fault state, cleanup and images.
These commands collect evidence; exit 0 is not automatic acceptance. They are
first-stop-only and cannot establish #40 removal/resume or general S1–S5 admission.

### Session C — final baseline matrix and workloads (NOT_READY)

Prerequisites: #40 supported clearance contract plus integrated/admitted final
runner, #41 fixes, held-out detector audit and final sensor/depth validity rules.
The original #40 workspace contains an offline planning command:

```powershell
& $py scripts/prepare_issue40_matrix.py
```

It is explicitly PLAN_ONLY_NOT_AUTHORIZED_TO_EXECUTE, not a live runner.
It currently lists S1/S3/S4/S5 and keeps S2 deferred. Its historical dense/full-volume
assumptions must not be promoted into official #40 acceptance requirements.
No different physical rig has been proved mandatory. Preserve the later staged-
resume review; require supported action/clearance bounds rather than blind probing.
There is no complete admitted final S1–S5 acceptance command yet; do not invent one.

After that runner exists, batch all 30 official baseline trials (five each S0–S5),
including S2 only after pedestrian operational admission. Retain every attempt,
per-tick outputs, safe-completion metric, HOLD speed/margin, clear-frame/duration,
removal/resume delay, collisions/lane events and capture-to-apply timing. S4 route
completion is N/A. Add the 60 s occlusion/missing-depth/ID-switch HOLD fault,
controller invalid-state/path-loss faults, S3 roadside negative and S5 bend
visibility/speed envelope. Compare headless versus final optional HUD workload
with the same warm models/config. #101's `docs/live_trial_plan.md` contains current
profiling/overlay commands; those commands are not complete-stack acceptance.

Once the real attempt register is populated on #107's checkout:

```powershell
& $py scripts/aggregate_baseline_evidence.py docs/evidence/TEST-08/register.json --output runs/TEST-08/final
```

An empty/missing register must remain BLOCKED. #47/#48 cannot close until complete
metrics, failures, uncertainty, final candidate identity and actual acceptance exist.

### Session D — admitted extension evidence (NOT_READY; batch with C where compatible)

- #49/#50: matched S0/S1/S2 × three weather conditions × three repetitions;
  preserve scene pairing, all weather/light readbacks, labels/visibility, RGB/depth,
  failures/denominators and actual observed sensor effects. Current S0 forces
  ClearNoon; extension capture integration is still needed and cannot be relabelled.
- #102/#54/#55: legal named route pairs, current pose/next-turn HUD, stopped ETA
  paused and actual recorded curves. Existing `wayfinder_replay.py` makes scripted
  poses; it is not live HUD integration. Complete the adapter path after RV-06.
- #100/#59 → #60: live forward-parking trials only after near-field depth/LiDAR
  coverage and low-speed steering/braking qualification; harness results alone do
  not admit P1. #61's reverse admission/cut decision must be evidenced explicitly.
- #52 → #53: extension obstacle/performance/cost evidence and admission on the
  final candidate; parked sweep capture alone does not certify driving behavior.

Use official extension acceptance criteria to set counts and thresholds. A final
extension-run command cannot be supplied honestly before these runner/admission
prerequisites exist. #67 combined regression/G4 and #70 final constraints follow,
then backup capture, rehearsal, report and release packaging.


## Recorded Wayfinder input

After a later authorized real capture:
python scripts/wayfinder_replay.py --origin baseline_start --destination baseline_goal --poses runs/wayfinder-poses.json --output runs/wayfinder-recorded-v1
Require one complete recorded route including stop; retain provenance and
actual poses. Schema1 origin=recorded,run_id,steps(time_s,x_m,y_m,yaw_rad,speed_mps).
The CLI renders each supplied pose and preserves hashes; declared origin alone
does not establish acceptance. Actual live HUD integration depends on RV-06.

## External deployment/reproduction queue

Canonical service recipe and75-wheel CPU reproduction profile are available.
No service is running.57 requires actual LAN hardware and labelled replay,
not loopback/black-image replacement. Cloud unavailable stays NOT_RUN.
Actual backup-machine/CUDA validation remains71. Retain every failed attempt.
