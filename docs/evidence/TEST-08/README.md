# TEST-08: baseline acceptance aggregation (issue #47)

Status: **BLOCKED / evidence pending**. This is an aggregation scaffold, not a
claim that any of the 30 final integrated acceptance runs passed. The uploaded
snapshot contains no registered final candidate run records. Existing TEST-01,
offline S0, sensor setup and scenario-placement checks are supporting evidence;
they do not fill these final acceptance slots.

| Task | Input owner issue | Scenarios | Required trials |
|---|---|---|---:|
| TEST-06 | #46 | S0 full-stack mission | 5 |
| TEST-04 | #44 | S1 car and S2 pedestrian stop/resume | 10 |
| TEST-05 | #45 | S3 roadside negative and S4 persistent HOLD | 10 |
| TEST-07 | #42 | S5 bend obstacle | 5 |

The generated [baseline table](current/baseline_table.md) reserves all 30 slots,
includes all 55 scenario/metric definitions, and records missing evidence.
The adjacent `summary.json` retains all submitted attempt records and the failure
log. The authoritative metric source remains
`tests/fixtures/acceptance_definitions.json`, currently proposed/review pending as
explained in [acceptance](../../acceptance.md). Do not silently change thresholds.
Owner-run acceptance and fault evidence is required before #47 can close.

## Register owner-reviewed evidence

Append attempts to `register.json`; never overwrite a failed attempt with a
successful retry. Keep the original register and generated reports in Git.
Use a new run ID for each attempt, including aborted runs. `trial` identifies
one of five planned trial slots; a retry repeats that slot but remains another
attempt. Distinct slots require distinct recorded seeds. No selected subset is
admitted here: a retained failure keeps the report blocked until the team
explicitly reviews the failure and establishes a new candidate evidence set,
while retaining and linking the earlier set.

Set top-level `candidate_commit` to the tested full commit SHA and
`configuration_sha256` to the actual frozen configuration digest. Every attempt
must match. Keep separate registers for different candidates/configurations.
The aggregator does not independently verify Git, configuration provenance,
physical measurements or review authenticity; reviewers must inspect raw logs,
configuration/model hashes, hardware/software, clock definitions and run archive.

An attempt has these fields (illustration only; do not register this as a run):

```json
{
  "run_id": "unique-UTC-run-id",
  "scenario": "S1",
  "trial": 1,
  "seed": 5308,
  "candidate_commit": "actual tested full SHA",
  "configuration_sha256": "actual configuration SHA-256",
  "environment": "live_carla",
  "complete_stack": true,
  "outcome": "completed",
  "failure_reason": "",
  "safe_completion_verified": true,
  "reviewer": "reviewer identity",
  "review_record": "review permalink or retained record path",
  "artifacts": [{"path": "raw/run.zip", "sha256": "actual artifact SHA-256"}],
  "metrics": {
    "Route completion (minimum per run)": {
      "value": 1.0,
      "n": 1,
      "method": "endpoint proximity and continuous stopped-window evaluation",
      "evidence": "raw/run.zip"
    }
  }
}
```

Add **every applicable metric**, using its exact name from the definitions JSON.
Each numeric measurement needs a positive sample count, measurement/window method,
and an evidence path listed in hashed artifacts. Paths are relative to the
register. Keep raw per-tick data, events, evaluation output and video in that
archive; SHA-256 verifies supplied bytes, not the truth of measurements. Keep a
shared archive access link in the review record. This script reads local artifacts
and does not download remote evidence. Missing/corrupt artifacts block review.
Outcome is `completed`, `failed` or `aborted`; completion does not imply acceptance.

Route progress is not safe completion: for S0/S1/S2/S3/S5 the reviewer must verify
legal endpoint arrival and the stopped window as well as all applicable checks.
S4 route completion is **N/A**, with HOLD duration/speed/margin still required.
No sample is never zero. Report-only resume delay is still required evidence,
without an invented pass threshold. Preserve clear-frame counts, removal and
perceived-clear timestamps, frame age, real-time factor, S5 visibility/speed and
curve-governor behaviour in the raw evidence and review record, alongside the
numeric metrics. S3 slowdown needs matched S0 windows and matching conditions.

## Fault tests and failure log

The separate `fault_tests` array covers these dependency checks; they do not
replace any of the 30 scenario trials:

- #44 TEST-15: `hold_occlusion`, `hold_missing_depth`, `hold_track_expiry`,
  `hold_id_switch`. Verify physical obstacle presence, stopped speed for 60 s
  despite track expiry, and resume only after validated clear observations.
- #46 TEST-16: `controller_invalid_state`, `controller_behind_target`,
  `controller_path_loss`. Verify latched/logged emergency stop, no seven-second
  command reuse, and documented restart/goal behaviour.

Append every fault attempt with `test` (one identifier above), unique `run_id`,
`candidate_commit`, `outcome` (`meets`, `failed`, `aborted` or `blocked`),
`observations` (actual measured behaviour and limits), `reviewer`, `review_record`
and `artifacts` in the same path/hash format. All attempts remain visible in the
JSON and Markdown. Reviewers must validate fault-specific assertions from logs;
the aggregator only checks the recorded disposition and evidence availability.

Per-scenario attempt counts, missing slots, failures and blocked records are
reported separately. All supplied numeric attempts contribute min/max/median and
sample standard deviation. These are descriptive summaries, not independent-run
confidence intervals: retries are correlated, sample counts are small, and missing
measurements are omitted explicitly. No pooled mean can override a failed trial.

## Generate and verify

From the repository root:

```powershell
python -m pytest -q tests/unit/test_baseline_aggregation.py
python -m ruff check scripts/aggregate_baseline_evidence.py tests/unit/test_baseline_aggregation.py
python scripts/aggregate_baseline_evidence.py docs/evidence/TEST-08/register.json --output runs/test08-review-01
```

Choose a new output directory on every invocation; existing reports are never
overwritten. The script writes `baseline_table.md` and `summary.json`. Exit 1 is
expected for the supplied empty register: it means **BLOCKED**, not a script test
failure. Exit 0 means only **READY_FOR_REVIEW**, never baseline approval or PASS.
Keep #47 open until owners supply all final candidate evidence, outstanding
failures are resolved with retained history, and required reviews are recorded.

AI assistance: ChatGPT/Codex prepared the aggregation scaffold and automated
checks. The user and task owners must run and review actual acceptance trials.
