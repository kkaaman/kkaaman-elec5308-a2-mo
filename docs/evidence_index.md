# Report evidence index — DOC-01 / DOC-03 / DOC-05

Snapshot: **28 October 2026**, Australia/Sydney. Canonical team main inspected:
`deeefde82fee14257d5b9e4dd0cf915f89ec99fe`. Current GitHub metadata and committed admission records were inspected.
This is a claim-to-source map, not approval of safety thresholds or final gates.
**Documentation tooling is updated; #62 stays open for final upstream evidence.**

## Evidence and acceptance status

| Area | Current classification | Evidence and reporting limit |
|---|---|---|
| BL-A / TEST-02, PR #89 | **MERGED / ACCEPTED ON MAIN**; review APPROVED | [Five-run navigation report](evidence/TEST-02/README.md), [results](evidence/TEST-02/rgb-repair-results.json), [archive/checksum](evidence/TEST-02/rgb-repair-evidence.sha256). RGB-enabled `packaged-acceptance`, five distinct episodes. This is navigation evidence, not final complete-perception S0 acceptance. |
| BL-F / TEST-14, PR #87 | **MERGED / ACCEPTED ON MAIN**; review APPROVED | [Offline acceptance logging](acceptance-logging.md), retained `offline-s0-acceptance` CI artifact. Offline S0 tests do not complete BL-F-07's live sensor-driven stop. |
| BL-B / TEST-13, PR #92 | **MERGED ON MAIN**; no separate aggregate reviewer approval asserted | [Depth report](evidence/TEST-13/README.md), [calibration](evidence/TEST-13/depth_calibration.json), [analysis](evidence/TEST-13/depth_analysis.json). Axial convention: 107 off-axis samples, median axial error ~0.050 m vs radial hypothesis ~1.886 m. Reviewed car/person bins through 25 m have median error <0.5 m and 0% invalid plain-depth target boxes. **Depth accuracy is not confirmed obstacle/bumper visibility.** |
| BL-E-03 / TEST-03, PR #91 | **MERGED ON MAIN**; comparison qualifications retained | [Braking report](evidence/BL-E-03/README.md), [measurements](evidence/BL-E-03/longitudinal.json), [A1 comparison](evidence/TEST-03/README.md). Conservative deceleration 1.497566177392398 m/s²; vehicle lag 0.3500000052154064 s. Route/protocol/timing intervals differ; ratios alone are not proof of overall improvement. |
| BL-C-04–06, PR #93 | **IMPLEMENTATION MERGED; HISTORICAL SCREENING QUALIFIED** | [Tracking report](track_stabilisation.md), [timing audit](evidence/BL-C-04-06/sensor-wait-timing-review.json), [current G1 record](gates/G1-admission-notes.md). Old 4.9 m/s and untimed confirmation claims remain withdrawn. Later bounded stationary-vehicle admission does not establish pedestrian or general obstacle performance. |
| BL-D-04–07, PR #86 | **MERGED; BOUNDED STATIONARY-VEHICLE PROFILE RECORDED** | [Historical report][bl-d-report], [admission record](gates/G1-admission-notes.md). Canonical config records 4.0 m/s, 1.497566177392398 m/s² deceleration, 0.9500000052154064 s response and 6 m margin for the exact bounded G2 source profile. General clearance/resume and pedestrian admission remain unestablished. |
| G1, issue #38 / PR #90 | **#38 CLOSED; #90 MERGED; EXACT-SOURCE ADMISSION** | [Gate record](gates/G1-admission-notes.md) and [trusted binding](gates/G1-admission.json). Later source additions require explicit reviewed rebinding. Reviewed source-only rebinding is recorded per branch. Canonical #100/#108/#105/#109 inventories pass production admission and 19 negative probes each; no downstream acceptance is inferred. |

PR #86 links retain the historical reviewed head. Current bounded admission is recorded separately; historical blocked decisions are not rewritten as measurements.

## Remaining dependencies for #62

| Item | Verified current state | What is still needed |
|---|---|---|
| PROC-01 / BL-C-01 | Process/contracts are available; retain each decision's actual approval state | Cite [decisions](decisions.md) and [contracts](detector_contracts.md); do not infer unanimous review. |
| BL-F-07 — [issue #41](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/issues/41) | **OPEN / BLOCKED**; final live integration acceptance not established | Same core harness/CARLA path, first sensor-driven stationary-car stop, raw logs and reproducible fault/acceptance command. Navigation TEST-02 and offline TEST-14 are existing evidence, not completion of this task; G1 PASS is required. |
| E4-07 — [issue #57](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/issues/57) | **OPEN / MISSING MEASURED SWEEP** | Latency sweep/breakdown and comparable local/LAN/available-cloud results. Main now contains the validated optional client/control adapter and server; no measured extension improvement is claimed. Unavailable modes stay NOT RUN. |
| RV-07 — [issue #46](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/issues/46) | **OPEN / PARTIAL FOUNDATIONS** | A1 wrappers/mission lifecycle and endpoint behavior records exist, but the task checklist, complete-stack five-run S0 and controller-fault acceptance are not closed. RV-07 means archive/demo/endpoint documentation, not a generic peer-review label. |
| RV-08 — [issue #45](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/issues/45) | **OPEN / PARTIAL FOUNDATIONS** | Rig/depth/lifecycle docs exist, but complete sensor/segmentation documentation, S3/S4 evidence and full sensor-cost/cadence acceptance remain outstanding. RV-08 is a sensor documentation task. |
| Full runtime lock | **LINUX CPU PROFILE VALIDATED**; 75 hashed runtime/test/docs/build wheels | See deploy/reproduction; CARLA/CUDA and the actual backup-machine profile remain separate. Historical version subsets are preserved. |
| Fresh disconnected installation | **LINUX CPU PASS** | Fresh venv in network-disabled container: hashed offline install, pip check, imports, reproduction tests and strict docs passed. Actual backup-machine/CUDA validation remains open. |
| Final report verification | **PENDING**; #62 remains OPEN | Final word/page count, reviewed numeric claims/figures, raw archive access and final admitted configuration. No final report or extension performance is fabricated. |

## Four-section claim map

| ID / report section | Claim or reserved item | Source / raw identity | State and limit |
|---|---|---|---|
| A1 Architecture | F1: sensor → frame/packet → stack → control, simulator boundary | [Interfaces](interfaces.md), [contracts](detector_contracts.md), [boundary](simulator_boundary.md), [API](api.rst), [module notes](module_notes.rst) | Implemented contracts; final F1 drawing pending. Evaluation truth stays separate from control. |
| A2 Architecture | Offline world and live synchronous owner | [Kinematic world](kinematic_world.md), [world lifecycle](world_lifecycle.md) | Synthetic offline world is not live driving validation. PR #86 integration is merged; runtime acceptance remains separate. |
| D1 Decisions | Units, rear axle/front bumper, freshness, single tick owner | [Decision register](decisions.md), [G1 record](gates/G1-admission-notes.md) | Bounded stationary-vehicle admission is source-specific; general stop/resume acceptance remains incomplete. |
| D2 Decisions | Local model identity and reproducibility | [Assets](assets.md), [manifest](../configs/reproduction.json), [reproduction](reproduction.md) | Asset hash/sample inference is not accuracy, CUDA validation or a full lock. |
| R1 Evaluation | Inherited TEST-01 baseline and limitations | [A1 record](evidence/TEST-01-baseline.md); `TEST-01-summary.json`, manifest, samples and video-frame CSVs | Historical run with failures; retain metric definitions and archive access caveat. |
| R2 Evaluation | Existing navigation/braking/depth measurements | TEST-02 `packaged-acceptance` 5-run archive, [TEST-03](evidence/TEST-03/README.md), [TEST-13](evidence/TEST-13/README.md) | Available merged results can populate labelled panels/trace F2. Navigation-only S0 is not the missing full sensor-to-control stop. |
| R3 Evaluation | Final integrated obstacle acceptance | BL-F-07 / issue #41; raw stop-run ID **pending** | **NOT ESTABLISHED / BLOCKED**; no final stop/HOLD/resume PASS. |
| R4 Evaluation | Optional extension comparison | E4-07 / issue #57; measured run IDs **pending** | **NOT RUN**; no local/LAN/cloud improvement claim. |
| L1 Limitations | A1 failures, repaired-runtime scope, metric/depth/timing limits | [TEST-01](evidence/TEST-01-baseline.md), [TEST-02](evidence/TEST-02/README.md), [TEST-03](evidence/TEST-03/README.md), [BL-C audit](evidence/BL-C-04-06/sensor-wait-timing-review.json) | Old BL-C timing/cap is withdrawn as current screening evidence; axial accuracy is not usable confirmed range. |
| L2 Limitations | Final integration, task closure and designated review | Issues #41/#45/#46/#57, [G1 record](gates/G1-admission-notes.md) | Open deliverables and exact-source review remain explicit; unanimous individual sign-off is not required. |

## Archive identity and new-run registration

TEST-01 source record names `Group7_TEST01_baseline-01_20261015_143228.zip`,
43,951,797 bytes, SHA256
`f30b5ca894af2d71c9c2d9bf3ace3c733a77b267b969042ff73b6799272b8ce8`.
This is a recorded identifier, not new local archive verification. Its shared
access URL remains pending in the source record. For TEST-02, use the separately
committed RGB-repair archive/checksum and run IDs; do not conflate the two cohorts.

Register each new run without overwriting history: immutable run ID/time, source
SHA/dirty state, scenario/seed, effective config/model hashes, actual platform/
GPU/package and CARLA versions, metric windows/thresholds, raw logs/video/summary,
archive checksum/access link, failures and reviewer disposition. Reference
configuration and observed environment are separate records.

## Documentation, validation and AI assistance

Saviour Kwofie's original commit `17a6224ad7e0fd8a224c30e787759d3cf062f27c`
and AI disclosure are preserved. His original report records 533 passed/1 skipped;
that is historical validation, not this refreshed branch's result. ChatGPT/Codex
assisted his implementation/docs/tests and local validation was run by the author.

20 October 2026 refresh: OpenAI Codex inspected all 17 PR files and live GitHub
states, integrated canonical main with a normal merge and updated this mapping,
reproduction notes and generated inventory. The original setup/runtime checks,
nonzero failures, module-note check and HTML CI artifact remain. No new driving,
fresh capture, extension performance or disconnected-install experiment was run.
Automated checks do not independently approve safety thresholds or acceptance.

CI retains both main's `offline-s0-acceptance` logs and #88's `documentation-html`
artifact. A configured artifact step is not a new successful CI-run record. Keep
PR #88 Draft until its remaining evidence/review obligations are resolved.

### Refresh validation (20 October 2026)

- Setup/reproduction focused tests: **8 passed**.
- Full suite: **728 passed, 1 skipped, 3 subtests passed**.
- Ruff, generated-module-note `--check`, strict Sphinx `-W` and
  `git diff --check`: passed.
- Simulator-boundary AST/import smoke: passed (72 core modules); five offline
  S0 acceptance trials passed. These are offline results, not new CARLA evidence.
- CI YAML parsed, ten shell `run` blocks passed Bash syntax checks, and both
  artifact upload steps remain present. `actionlint` was unavailable locally.
  GitHub CI was not triggered and no new successful remote run is claimed.
- Read-only checks of the shared local model files passed all three identities.
  The active reused Python environment has torch **2.6.0+cu124** and numpy
  **2.5.2**, so the default historical-subset check correctly fails those two
  comparisons against **2.14.1 / 2.5.4**. This is not a validated fresh install,
  full runtime lock or claim that the historical and live profiles are identical.
  The reused environment includes CARLA; CI's CARLA-absent installation condition
  was not independently recreated. Simulator imports were blocked during the
  separate core smoke check.

[bl-d-report]: https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/fcfa3d8fe592c2516472fbf45f80540796c0162e/docs/baseline_threshold_freeze.md
[bl-d-review]: https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/fcfa3d8fe592c2516472fbf45f80540796c0162e/docs/evidence/BL-D-06/review.json
[corridor-fix]: https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/commit/61e2d1a40c4b65229621b2f18b052861e975974f
[g1-pr]: https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/pull/90
