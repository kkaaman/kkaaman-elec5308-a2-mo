# Report evidence index — DOC-01 / DOC-03 / DOC-05

Snapshot for task #62. Completion states below reflect the owner's task update;
this index does not independently approve other tasks. This is a maintained
claim-to-source map, not a declaration that final evaluation has passed.

## Dependencies and completion gate

| Dependency | Reported state | Effect on #62 |
|---|---|---|
| PROC-01 | Completed | Process prerequisite available; cite the actual decision record for each claim |
| BL-C-01 | Completed | Detector contracts available; see [contract record](detector_contracts.md) |
| BL-F-07 | Not completed | Final baseline/acceptance run evidence remains pending |
| E4-07 | Not completed | E4 evaluation and its reproduction evidence remain pending |
| RV-07 | Not completed | Required review evidence remains pending |
| RV-08 | Not completed | Required review evidence remains pending |

**#62 remains open.** Documentation scaffolding, existing evidence mapping and
API notes can proceed now. Final report/result sign-off waits for the four open
dependencies. Confirm their precise deliverables in the task tracker; this file
does not infer their implementation from task IDs. PR #87 is not assumed merged
and its offline S0 artifacts are not treated as final BL-F-07 evidence.

## Four-section claim map

| ID / report section | Claim or reserved item | Tasks / decision record | Source and raw run IDs | State and limit |
|---|---|---|---|---|
| A1 Architecture | F1: sensor → packet → stack → control; adapter boundary | BL-C-01; shared contract/task ownership in source | [Interfaces](interfaces.md), [detector contracts](detector_contracts.md), [boundary](simulator_boundary.md), [API](api.rst) | Implemented contracts; F1 drawing pending. Not driving evidence |
| A2 Architecture | Fixed-step offline world vs live world ownership | BL-F-03; [world lifecycle](world_lifecycle.md) | [Kinematic world](kinematic_world.md), `src/sim_harness/kinematic.py`, `src/carla_adapter/world.py` | Document actual limitations; kinematic world is not CARLA validation |
| D1 Decisions | Units, frame identity, freshness, single tick owner | [Shared contract review](interfaces.md), [decision register](decisions.md) | Source and task records; no raw run required for describing contracts | Preserve each record's approval status; do not imply unanimous approval |
| D2 Decisions | Local weights and reproducible runtime | DOC-03; RV-01/RV-02 records | [Assets](assets.md), [manifest](../configs/reproduction.json), [runtime sample](evidence/baseline-batch/detector.json) | Identity/sample inference only; no accuracy claim; lane inference review pending |
| R1 Evaluation | T1 inherited A1 baseline; raw metrics | TEST-01 | [Measured baseline](evidence/TEST-01-baseline.md); `TEST-01-summary.json`, `TEST-01-manifest.json`, `TEST-01-samples.csv`, `TEST-01-video-frames.csv`; archive ID below | Existing measurement with failures; reviewer/archive access pending in source record |
| R2 Evaluation | F2 and T1 final baseline/acceptance values | BL-F-07 | **NOT RUN / raw run ID pending** | Blocked; no values or pass claim |
| R3 Evaluation | E4 comparison and configuration | E4-07 | **NOT RUN / raw run ID pending** | Blocked; no cloud/hybrid improvement claim |
| L1 Limitations | Baseline junction timeout/lane misses, metric scope | TEST-01; relevant design decisions | [Baseline limitations](evidence/TEST-01-baseline.md), [batch baseline](batch_baseline.md) | Measured baseline limitations only; sample checks do not prove accuracy |
| L2 Limitations | Final caveats and peer review | RV-07, RV-08 | **Review record IDs pending** | Required reviews not completed |

## Existing archive identity

TEST-01 source record names
`Group7_TEST01_baseline-01_20261015_143228.zip`, 43,951,797 bytes,
SHA-256 `f30b5ca894af2d71c9c2d9bf3ace3c733a77b267b969042ff73b6799272b8ce8`.
These are recorded identifiers, not a new local verification. The team's shared
archive URL remains pending in the [source record](evidence/TEST-01-baseline.md).
Do not replace it with an inaccessible machine path or describe it as published.

## Register each new run

Add a new row without overwriting historical results. Record: immutable run ID,
UTC time, commit SHA and dirty-tree status, scenario/seed, configuration hashes,
model SHA-256, Python/package/platform/GPU details, actual CARLA client/server
versions, metric definitions and thresholds, raw-log/summary/video locations,
archive SHA-256, access link, failures and reviewer disposition. Link F2/T1 to
those rows only after measurement. Keep reference configurations separate from
observed runtime metadata.

## Documentation evidence

Run the commands in [reproduction](reproduction.md), retain their logs and record
the tested commit. CI builds with `sphinx-build -W` and publishes the
`documentation-html` artifact. A configured job is not a successful build record;
add its actual CI URL after the branch passes. Fresh-machine and disconnected
reproduction evidence, complete transitive lock and final reviews are pending.
