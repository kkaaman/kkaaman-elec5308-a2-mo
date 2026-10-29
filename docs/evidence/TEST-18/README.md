# TEST-18: offline fault matrix (issue #63)

Status: **PARTIAL / BLOCKED**, pending G3-01 (#48) baseline/flag admission and
E4-10 (#58) outage/restart evidence. No release gate or baseline acceptance is
claimed. This adds runnable command-level integration tests now; keep #63 open.

## What actually runs

The uploaded source tree contains no admitted production extension-flag registry
and no remote/hybrid detector implementation. The suite therefore enumerates an
explicit **harness profile matrix**, not a claim to enumerate approved release
flags. It runs the real `SkeletonStack`, `FrameDetector`,
`OperationalFreshnessGate`, `TrackDepthFilter`, `TrackedGovernancePolicy`,
`ClearanceGuard` and `SpeedGovernor`. No production control logic is replaced or
modified. The harness lives under `scripts` so it does not expand the frozen
`src` admission inventory. Controller outputs, observations, clock progression and corridor
geometry are synthetic fixture inputs.

| Axis | Available cases | Scope |
|---|---|---|
| Sensor/backend | local_fixture, delayed_fixture | Synthetic image/depth, image-backend protocol; no learned model |
| Behaviour | cruise, approach, hold | Inputs drive the existing governance/clearance code; not production CLI mode flags |
| Dispatch | detect, track | Exercises existing `use_tracking=False/True`; synthetic IDs are supplied in both modes |
| Fault | 18 per combination | Sensor, result identity, delay, restart and controller faults |
| Remote/hybrid | 12 blocked combinations | Two missing profiles x three behaviours x two dispatches; not silently substituted by fakes |

Available count: **2 x 3 x 2 x 18 = 216 cases**. Blocked combinations are counted
separately from executed fault cases. The fake delayed backend uses virtual time,
not HTTP, a cloud service or a latency benchmark. Restart is a backend generation
change, not a killed/restarted server. Old-session response-envelope injection
checks the stack identity boundary; it does not validate asynchronous transport
cleanup or replace E4-10.

Faults: missing depth, all-NaN depth, missing capture clock, old capture,
expired simulation frame, duplicate frame, foreign observation session,
old-session result, wrong result frame, wrong result source, detector exception,
processing deadline miss, backend generation restart, missing track ID, invalid
ego state, controller exception, controller delay and invalid controller output.

Each isolated case first proves that positive commands are possible with valid
observations. HOLD cases then establish a blocker and two partial clear frames.
Faults persist for three consecutive injected ticks. On each fault tick the suite requires zero throttle, full brake and zero steer;
records source/session/frame/reason; and checks that sensor/deadline faults do
not advance clearance. Controller exceptions and invalid controller outputs occur
after a fresh perception update, so their prior valid perception count can
advance; they must still select braking. Existing HOLD/emergency/recovery gates
must stop the immediate next step. An unblocked scene with valid fresh recovered
observations may resume according to the current policy. This is not a new
release-level recovery rule.

Virtual `perf_counter` and host clocks are patched only inside single-threaded
fixture cases; do not run the harness concurrently with production control in the
same interpreter. No real waits are used. Socket connect/connect_ex and
create_connection are blocked during the matrix. A fresh-process regression
checks that CARLA, torch and ultralytics are not imported. No weights, GPU or
CARLA server are needed.

## Run and retain evidence

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/unit/test_all_flags_faults.py
.\.venv\Scripts\python.exe -m ruff check scripts/offline_fault_suite.py scripts/run_all_flags_faults.py tests/unit/test_all_flags_faults.py
.\.venv\Scripts\python.exe scripts/run_all_flags_faults.py --output runs/TEST-18/local-01
```

Use a **new output directory** each time. `report.json` preserves every case,
check and trace, fixture parameters, Python/NumPy versions, source/configuration
byte hashes and commit/dirty state. `diagnostics.jsonl` includes all tick records,
including failing cases and both input and returned result identities.
`report.md` is the readable matrix, with blocked profiles listed explicitly.
Synthetic measurements are not physical sensor/actuator times.

Exit codes: **1** if any available fixture fails; **2** if fixtures pass but full
scope is blocked. For CI, `--check-fixtures` returns **0** when available fixtures
pass while leaving the report's overall status **BLOCKED**. That switch cannot
hide fixture failures. The separate CI job uploads diagnostics with `always()`;
a configured workflow is not evidence of a successful CI execution.

Reviewers should inspect failed-check names and full diagnostic traces. Do not
delete failed attempts, choose only passing profiles, relax frozen thresholds or
label this as all-flags release approval. When G3 supplies the admitted flag list
and E4-10 supplies the real remote/session behaviour, extend this matrix to
exercise their actual implementations and rerun at the candidate commit.
Unlisted/admitted extensions are outside current coverage and must remain blocked.

## Local development history

The first exploratory run stopped correctly on every injected fault but recorded
16 failed **additional recovery assertions**: they had incorrectly required
another stopped tick in an unblocked scene after valid recovery, and forbidden
valid perception-count updates preceding controller exceptions/invalid outputs.
Those assertions were corrected against the existing policy semantics above;
no runtime code or baseline criterion was changed. A later sustained-fault run
recorded eight assertion failures because recovery was required to be valid
immediately after missing IDs; the real filter conservatively remained stopped
while reconfirming tracks. The recovery oracle now accepts continued full braking
or a valid policy command after the required fresh-result count. That exploratory
report is retained too. The initial report is retained
in the [development evidence archive](development-evidence.zip), alongside the final report and diagnostics.
These are local synthetic development records, not Windows or CI measurements.

AI assistance: ChatGPT/Codex prepared the harness, tests, workflow and analysis.
The user must run the commands in the project checkout and retain their results.
