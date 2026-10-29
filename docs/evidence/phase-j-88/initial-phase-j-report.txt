# Phase J official execution graph and CARLA RUN QUEUE

28 October 2026, Australia/Sydney. Canonical main remains `ae6a83d`.
[Machine-readable official graph](evidence/phase-j-88/project-graph.json) records
refreshed open PRs, official Task-ID owners/dependencies and issue bodies' textual
prerequisites. No PR was merged and no issue closed in this execution. No CARLA
server was started. Existing issue #40 uncommitted work was preserved.

## Work published through official owners

| Official PR / issue | Published outcome | Validation / remaining acceptance |
|---|---|---|
| #103 / #52 | Canonical main merged normally; future sweep timestamps and nonfinite freshness limits rejected; exact binding proposal retained | 60 focused tests pass; 976 full-suite passes, 6 exact-source admission failures; live known-object capture/recorded replay and corridor-owner review remain |
| #101 / #42,#45 | All three requested evidence defects fixed; canonical main merged normally; exact binding proposal retained | 14 scenario tests pass; 949 full-suite passes, 6 admission failures; recorded curves, S3/S4/S5 and timing evidence remain |
| #100 / #59 | Canonical-main integration validated; complete source-delta proposal and logs added to original contributor branch | Original head: 882 tests, P1 8/8 and S0 5/5 pass; canonical integration: 966 passes, 6 admission failures; no live parking claim |
| #105 / #57 | Existing client updated with explicit run/session identity, response echoes, capture/send client-clock metadata, bearer credentials and worker-side old-session cleanup | 27 focused tests pass; 945 full-suite passes, 4 admission failures; controller integration and real labelled local/LAN/cloud comparison remain |
| #109 / #56 | Focused server PR stacked on #105; bounded isolated trackers, FastAPI routes, complete CPU wheel hash lock and actual offline-runtime image smoke | 45 combined focused tests pass; 961 full-suite passes, 6 admission failures; real-model network-none/read-only smoke passes; no accuracy or deployment-performance claim |
| #108 / #49 | Explicit 27-cell paired weather plan and installed-preset/light preflight | 12 tests pass; 930 full-suite passes, 4 admission failures; every capture cell remains NOT_RUN; dataset/labels and installed verification remain |
| #88 / #62 | Requested stale module notes/evidence index fixed; failed LFS pull tolerated only after pinned local asset validation; explicit binding proposal retained | 9 reproduction tests pass; 927 full-suite passes, 4 admission failures; fresh disconnected install, complete platform lock and upstream evidence remain |
| #40 local branch | Existing implementation/evidence preserved; no private copy of #103 introduced | 54 targeted tests pass; no physical clearance/resume certificate or live acceptance claimed |

Ruff, simulator-boundary smoke and strict Sphinx pass on the changed PR branches.
Full regression logs retain every admission failure; the validator was not weakened
and the trusted record was not silently rebound. Original contributor histories
were preserved by normal merges/pushes. No force push, squash or branch deletion.
PR #106 overlaps #88's ownership of #62; both remain Draft and must reconcile
useful documentation/reproduction work while preserving both authors.

## Immediate external prerequisite: exact-source admission review

The production loader hashes every Python source and auxiliary config, including
optional extensions. Consequently a feature addition changes baseline identity
even while off. See [the governing repository process](gates/G1-admission-notes.md)
and [issue #38's designated review process](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/issues/38#issuecomment-6395116981).
A designated reviewer/owner must review the published candidate binding; this
execution cannot invent that human review. Full-team unanimous sign-off is not
required. No permission is inferred from a stale approval of different bytes.

Concrete review proposals are already committed to the corresponding PRs:

| PR | Exact candidate proposal |
|---|---|
| #100 | `docs/evidence/E5-P1/phase-j-20261028/admission-proposal.json` |
| #103 | `docs/evidence/E2-01-04/phase-j-20261028/admission-proposal.json` |
| #101 | `docs/evidence/phase-j-101/admission-proposal.json` |
| #105 | `docs/evidence/E4-04-05-07-09/phase-j-20261028/admission-proposal.json` |
| #109 | `docs/evidence/E4-01-03-06/phase-j-20261028/admission-proposal.json` |
| #108 | `docs/evidence/E1-01-02/phase-j-20261028/admission-proposal.json` |
| #88 | `docs/evidence/phase-j-88/admission-proposal.json` |

Each proposal preserves mission values/scope and the historical #41 runner binding;
it does not grant live or general obstacle admission. These are independent branch
snapshots. After any prerequisite merge, merge the new canonical main into each
dependent branch, regenerate its exact proposal, validate again and obtain the
required review of changed bytes before merging with a merge commit.

#100 is the shortest path to a completed independent PR: its #59 acceptance is
harness-scoped. Once its exact rebind is reviewed and final integrated checks pass,
merge normally and inspect/close #59 only if all its criteria still hold. Live
parking qualification remains #60. Do not treat a live #60 gap as a hidden #59
requirement, and do not waive genuine G1 source identity failures.

## CI infrastructure evidence

For actual run [40652652932](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/actions/runs/40652652932),
all three jobs had empty runner names, zero steps and the payment/spending-limit
annotation. This matches the authorised infrastructure exception. It does not
excuse the separately reproduced local admission failures. Local lint/docs/tests
were run instead of representing unexecuted CI as passed.

## Dependency order and feedback cycles

The critical path is #103/#52's official sensor evidence, then supported #40
clearance/stop-hold-resume, then #99/#41's integrated sensor-driven stop and
#42–#46's repeated/fault evidence, then #107/#47 aggregation and #48 freeze.
#103 must genuinely complete and merge before #40 imports that capability.
A second sensor alone does not fix #41: its recorded learned-detector false
positive is still a substantive defect. Nearer-wins conservative fusion can
retain an RGB-D false positive. Neither #41 nor #43 is declared passed.

The literal graph has feedback cycles: #42↔#43, #43↔#44, #43↔#45,
#56↔#57, #57↔#58 and #66↔#72. They describe integration/validation feedback;
PR-number sorting is not a valid execution order. The #56/#57 code is exposed as
a focused stack (#109 based on #105), rather than a competing client. Neither
whole issue is closed before its acceptance evidence exists. #63 waits for
#48/G3 and #58/E4-10; it cannot claim all-flags acceptance before selected flags,
source identities and baseline decisions are established.

## CARLA RUN QUEUE — ordered to minimise simulator sessions

No new simulator run is requested until the listed preconditions are satisfied.
A NOT_READY item is a technical prerequisite, not an executable acceptance claim.
Use one installed CARLA 0.9.16/Town10HD_Opt environment, retain all failed attempts,
and record candidate/source/config/model identities for every batch.

### Session A — parked sensor qualification, #103/#52 (conditional live queue)

Before starting: corridor owner accepts the protocol/fixture; exact candidate
source is reviewed/admitted. Require matching client/server, empty world and no
other synchronous client. Install the selected PR checkout locally, then:

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py scripts/capture_lidar_sweeps.py --host 127.0.0.1 --output runs/lidar-capture-v1
& $py scripts/replay_lidar_obstacles.py --input runs/lidar-capture-v1/sweeps.json --output runs/lidar-capture-v1/replay.json
```

Scenarios: parked ego, known target at 10/20/30 m on the checked straight.
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

That command requires the #108 source/config checkout and its reviewed identity;
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

## Other external/environment evidence — no CARLA required

#109's actual CPU container is available locally as `group7-inference:issue56`;
its recipe/hash lock and smoke commands are in [the service guide](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/cf52257/docs/inference_service.md)
on that PR. It does not replace #62's full target-platform/CUDA lock.
#57/#58 need an actual selected local/LAN deployment, a labelled held-out replay
of at least 200 measured frames per configuration, matching model/hardware notes,
accuracy/miss-rate/latency outputs, and controlled-stop integration through the
final runner. Cloud stays NOT_RUN when unavailable; loopback and black-image
smokes are not substituted for LAN or accuracy measurements.

#62/#71 need a real fresh disconnected install and backup-machine environment;
#65/#66/#72 need stable measured report claims/evidence; #73/#74 need the selected
release and actual fallback media. #75/#76 require the real six-person rehearsals
and cross-module Q&A, #78/#79 actual peer/cross-training records, and #80–#85 each
person's own private submission material. This agent cannot fabricate human
participation, reflection, sign-off or evidence. Those official issues remain open.

The next integration action is designated review of the already published source
proposals, starting with the dependency-ready candidates. After each genuinely
complete merge, fetch canonical main, refresh this graph, merge main into dependent
branches and continue. No billing exception, partial scaffold or synthetic evidence
is used to close an incomplete issue.

```{toctree}
:hidden:

evidence/phase-j-88/README
```
