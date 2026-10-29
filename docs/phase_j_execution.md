# Phase J final continuation report and CARLA RUN QUEUE

29 October2026, Australia/Sydney. **CARLA remained OFF.** Original #40 uncommitted
work remains intact. Contributor histories preserved through normal merges and
pushes; no force push/rebase/squash/branch deletion. This supersedes the
[previous report](evidence/phase-j-final/pre-final-report.txt).
[Official snapshot](evidence/phase-j-final/official-state.json) preserves exact
heads, merge commits, issue states/bodies and dependencies.

## Merged PRs

| PR/title | Final merge commit | Issue disposition | Tests/evidence |
|---|---|---|---|
| [#100](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/pull/100) E5-01\u201303: Build the forward-parking capability  | 7f0a80d35fd2288a8ee61a02c0c0f5bf56cf2b26 | #59 CLOSED; #60 OPEN | 77 focused;972 full;P1 8/8,S0 5/5,archive/hash checks |
| [#108](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/pull/108) feat(weather): specify paired experiment and installed preset checks for #49 | 86a7e189776d996bc6a6e9fe4ca80ead3bbac5d7 | #49 OPEN | 12 focused;984 full;27 paired cells NOT_RUN |
| [#107](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/pull/107) test: add baseline evidence aggregation scaffold for #47 | d0ceb4f2134b8b3a2eca4e422c1e42e532a29113 | #47 OPEN | 16 focused;1000 full;empty register BLOCKED |
| [#105](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/pull/105) Add remote detector and local fallback scaffold for issue 57 | 29aed99b7c1a6c22228c26db0e40f4b2a28ae80b | #57 OPEN | 37 focused;1037 full;canonical control/HOLD/session fixtures |
| [#109](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/pull/109) feat(server): add isolated offline inference service for #56 | deeefde82fee14257d5b9e4dd0cf915f89ec99fe | #56 CLOSED | 55 focused;1055 full;real-model offline Docker smoke |
| [#88](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/pull/88) [DOC-01/03/05] Add report evidence index and reproducible documentation | 0eaae9716e8d8b2ebaf74e6463ef6bed17d86461 | #62 OPEN | 15 reproduction;1070 full;75-wheel fresh disconnected profile |
| [#106](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/pull/106) docs: add report evidence map and reproduction checks for #62 | 2eaa653dbc47688f448155c31d48198d5a8a4391 | #62 OPEN; incorporated via88 | Contributor history included by normal merge, no duplicate content |
| [#102](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/pull/102) Feat/e3 01 03 wayfinder v1 | 9cc45d6de9f0c00d33b3fd66acadca74e3e5bbe3 | #54 OPEN; RV-06 dependency | 40 focused;1110 full;24 recorded HUDs,3 paused;archive/member SHA verified |
| [#111](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/pull/111) feat(e1-03): add weather detection evaluator | ed07850a9818d9d55e00b37fcd3eb26ed5aa27cb | #50 OPEN; measured dataset missing | 12 focused;1122 full;invalid-frame/identity/time counterexamples |
| [#112](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/pull/112) test: add offline fault matrix and diagnostics for #63 | 9de375502e42304100c0617bf18c6727f9989daa | #63 OPEN; G3/E4-10 missing | 216 synthetic fault cases;1349 full;12 release combinations BLOCKED |

All full runs passed3 subtests and deselected1 CARLA test. Server-related suites
retain1 dependency deprecation warning. Ruff, simulator boundary, generated
module notes, strict Sphinx and diff checks passed on each final scope.
The latest canonical full regression is **1349 passed**. Container smoke used
real mounted weights with networking disabled/read-only root,2CPU/4GiB/128PIDs,
nonroot/no capabilities; container stopped/removed. It proves protocol, not
accuracy/LAN latency. Fresh Linux CPU reproduction used75 fully hashed wheels
in a clean venv without system packages, network disabled; pip check, imports,
package/asset profile, reproduction tests and strict Sphinx passed.
Windows/CUDA/backup-machine evidence remains separate.

## Open PRs

| PR | Classification | Exact remaining criterion |
|---|---|---|
| #99/#41 | TECHNICALLY INCOMPLETE; correction requires further sensor validation | Known learned-detector trash-can false positive persists (car confidence0.55528). Crop verification also rejects the genuine car; width proxy overlaps583/599 genuine observations. No validated sensor-only correction/held-out data. Runner site/scope and mission binding differ from historical admission; unsupported execution was not rebound. |
| #101/#42,#45 | CARLA BLOCKED; final runner dependencies remain | Actual recorded lane/curve/model overlay, S3/S4/S5 and final complete workload/cadence/timing. Offline observer/evidence tooling and substantive review fixes complete. Latest1163 full tests,41 focused,admission+19 negative probes pass. |
| #103/#52 | CARLA BLOCKED | Actual90-sweep known-object capture and recorded road/roadside/in-corridor classification/replay with cleanup. Latest1189 full tests,67 focused,admission+19 negative probes pass. Sparse road extent cannot establish positive clearance; that unsupported predicate was removed. |

No READY PR is intentionally left open. #102's actual recorded-replay criterion
was satisfied by reusing the merged TEST-02 archive, not new CARLA execution:
24 sampled real ego poses,3 stopped samples,24 rendered mini-map HUDs, full
archive/member hashes and selected row/frame IDs. Original injected-HOLD run
retains FAIL for false_stops; never relabelled S0 PASS. Its original controller
did not run Wayfinder; these are postprocessed recorded HUDs, not live overlays.
Issue54 stays open for documented RV-06 integration; no dependency silently waived.

## Admission and production review

Complete production deltas classified per entry before supported loader-only
rebinding under explicit user delegation. No new human approval fabricated;
no validator disabled or test changed merely to accept a fingerprint.

| Branch | Old fingerprint | Final reviewed fingerprint |
|---|---|---|
| #100 | bb8f19fedc92a330b90ee090a928bd836612390019104d823a44482ff242203d | b0f3edc49691ce9033bc85226a2473854122a9d386f2c822d161064903f40b90 |
| #108 | b0f3edc49691ce9033bc85226a2473854122a9d386f2c822d161064903f40b90 | 664c4d019ad385f1a6a6b360bf8a776ed4fa47c157230da3c751853f21984f5c |
| #105 | 664c4d019ad385f1a6a6b360bf8a776ed4fa47c157230da3c751853f21984f5c | f34a21469f2207fdbefe53eee91b7d3913cb7e522081e25747bddd70021ffea3 |
| #109 | f34a21469f2207fdbefe53eee91b7d3913cb7e522081e25747bddd70021ffea3 | e036881ecd771d85698915075c3f6c5c7134a36de30d02f62cc3418ceff32924 |
| #88/#106 | e036881ecd771d85698915075c3f6c5c7134a36de30d02f62cc3418ceff32924 | 520921912f1f43bc6743d01ad176167d901bf07678377dfaa3382ff3cc9c7875 |
| #102 canonical refresh | 520921912f1f43bc6743d01ad176167d901bf07678377dfaa3382ff3cc9c7875 | f2143c7ca680f2657bb69193698b2402c18f0ac273b592a9dc283b0d0d619ceb |
| #111 current canonical | f2143c7ca680f2657bb69193698b2402c18f0ac273b592a9dc283b0d0d619ceb | 9c9d4d0718ac841fb466f9d31c8de0cbcd871236e8b50b86018abdf30dc1c38b |
| #101 refreshed candidate | 9c9d4d0718ac841fb466f9d31c8de0cbcd871236e8b50b86018abdf30dc1c38b | d6ebdd26c001451af8bb30b40c2802f2ed123b685d412ae0d18199811ef9cf95 |
| #103 refreshed candidate | 9c9d4d0718ac841fb466f9d31c8de0cbcd871236e8b50b86018abdf30dc1c38b | 2b8daedd77c0977eb17eaebbae87a83ec0cf07a98bcbd43edbe8e5363fe3834b |

#107/#112 and this report change no protected production inventory; no rebind
needed. Intermediate/previous records remain in docs/gates/history and applied
review evidence. Fresh records include inventory/classification, config semantic/
file/model/camera/depth/record identities. Candidates101/103 contain canonical
production foundation111; later112 is scripts/tests only. Refresh again before
future merge if canonical protected source changes.

Frozen values unchanged: cruise4m/s,deceleration1.497566177392398m/s²,
response0.9500000052154064s,stop margin6m,stationary_vehicle_g2_only,
pedestrian admission deferred,depth25m,confirmation/freshness/all-track contracts.
Mission semantic SHA2903124878485c8030e160f3598c381bf0197329f79d74773d84048a96c2e9ef;
file SHAa4b5e669f9426db24d42d9a1d20114981c95bb15302af006b1196044696fe187;
actual model SHA9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef.
Historical runner fingerprintcac8bc9c4f131b16b871191f69562d87517d3f1b956849b9950543e582ff400a
and revision897bd126 remain unchanged.

All19 negative probes still reject changed/missing/added source, auxiliary config,
mission values/scope/profile, model identity/actual bytes, camera/confirmation/
depth policy, runner hash/revision and record checksum. Canonical checks pass.
CI102/112 failures were explicitly payment/spending, no runner, zero steps.
111 had no reported checks; it was never labelled passed/billing by assumption.
Local evidence is recorded separately.

## Issues completed and remaining

Only **#59 and #56 closed** during continuation: harness-scoped parking8/8 plus
baseline5/5 and archive provenance; offline server protocol/isolation/reset/
fault/source-switch fixtures plus actual real-model local Docker smoke.

- #40/#41/#43: supported clearance/action bounds and validated detector/held-out
  evidence; original40 dirty work preserved, no private copy of unmerged LiDAR.
- #42/#44/#45/#46: final complete-stack repeated lane/sensor/HOLD/fault/cost data.
- #47/#48: actual30-trial register,all failures/uncertainty,final scope and G3 freeze.
- #49/#50/#51: paired weather dataset/labels and measured robustness/admission.
  Preparation and evaluator now canonical; no result fabricated.
- #52/#53: actual LiDAR capture/replay then extension performance/cost/admission.
- #54/#55: RV-06-dependent Wayfinder integration and extension admission; recorded
  replay component evidence completed.
- #57/#58: >=200 labelled held-out real frames per local/LAN configuration with
  hardware/model/config identities,latency/accuracy/miss table,available cloud or
  NOT_RUN and final operational failover/admission. Synthetic216-case harness
  and black-image Docker smoke do not substitute.
- #60/#61: actual low-speed/near-field/braking parking qualification; reverse
  evidence or explicitly authorized cut decision.
- #62: final41/45/46/57 evidence and report/claim verification; CPU reproduction
  complete,actual backup-machine/CUDA remains separate.
- #63/#67/#70: admitted selected release/all-flags/combined/final sensors depend
  on missing G3 and extension decisions.63 harness retains12 BLOCKED combinations.
- #64/#65/#66/#68/#71/#72/#73/#74: stable measured report/media/release/demo,
  archive access and backup hardware depend on final selected release.
- #75/#76/#78/#79: real rehearsals,Q&A,peer review,cross-training.
- #80–#85: respective humans' private submissions.

The dependency graph's literal feedback cycles42↔43,43↔44,43↔45,56↔57,57↔58,
66↔72 are retained. The client/server implementation stack was completed without
pretending dependent measurement issues were closed. No further dependency-ready
offline issue can be accepted from the available evidence. Missing validated
sensor contracts/data,hardware,live execution,release choices and human actions
remain explicit; this is not a claim that all official work is complete.

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
  paused and actual recorded curves. Recorded replay acceptance now uses real TEST-02 traces. The remaining RV-06-dependent live integration/demo is separate; do not repeat the completed pose replay as a missing live criterion.
- #100/#59 → #60: live forward-parking trials only after near-field depth/LiDAR
  coverage and low-speed steering/braking qualification; harness results alone do
  not admit P1. #61's reverse admission/cut decision must be evidenced explicitly.
- #52 → #53: extension obstacle/performance/cost evidence and admission on the
  final candidate; parked sweep capture alone does not certify driving behavior.

Use official extension acceptance criteria to set counts and thresholds. A final
extension-run command cannot be supplied honestly before these runner/admission
prerequisites exist. #67 combined regression/G4 and #70 final constraints follow,
then backup capture, rehearsal, report and release packaging.



## Additional precise commands and scope

For101 timing preparation, once the final runner/workload is admitted:
python scripts/run_scenario.py --runs 5 --output runs/S0/final-candidate
python scripts/run_scenario.py --runs 1 --output runs/TEST-22/headless
python scripts/run_scenario.py --runs 1 --output runs/TEST-22/hud
python scripts/measure_operational_governance.py --host 127.0.0.1 --frames 120 --episodes 2 --output runs/TEST-22/warm
python scripts/profile_loop_timing.py --output runs/TEST-22/profile --run headless=runs/TEST-22/headless/run-01 --run hud=runs/TEST-22/hud/run-01 --run warm=runs/TEST-22/warm/timing.jsonl --baseline headless --final hud
Same warm/config identity required; >=20 valid latency samples,final p95<=100ms
or explicitly BLOCKED. Windowed server/spectator is not an in-loop HUD timing
certificate. Recorded lane-model curves need actual video and matched poses;
see101 docs/live_trial_plan.md,not a substituted synthetic video.

Final S0–S5 full-stack runner is not admitted yet: no honest exact execution
command exists for30 trials. S2 remains deferred until pedestrian admission.
This is a NOT_READY prerequisite,not an instruction to improvise live execution.

No service/container remains running.57 needs actual LAN hardware and >=200
labelled frames/configuration.51/53/55/58/60 admission decisions require their
measured criteria; no artificial single repeated frame or local loopback result
is accepted. Actual backup-machine/CUDA and human participation remain external.
