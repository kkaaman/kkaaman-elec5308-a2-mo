# BL-B-06/07 live verification — 2026-10-25

Historical attempts 01–03: **BLOCKED / MEASURED_FAIL**. No scenario geometry or threshold was changed.
These are evaluation/scenario checks, not issue-41 perception/control acceptance.

Superseded by [attempt04 MEASURED_PASS](accepted-20261025.md), with an explicit
S3 topology correction. The failed archive and manifest below remain unchanged.

Canonical main: `ce801258ddbadf3911916c4ed32e0e0e3a8b4981` (G1 published).
PR98 source base: `18e891f517d435da01927582f1f276f4b9644071`, with the exact uncommitted
source snapshots, hashes and patch retained in the archive. Existing authorship/history is intact.
The scenario additions change G1's source/config inventory; operational use needs reviewed
rebinding, not bypassing admission. The standalone evaluation verifier does not load that gate.

## Reproduce

From the repository root with the CARLA SDK and project dependencies installed:

```bash
PYTHONPATH=src python scripts/verify_scenarios.py --output runs/issue39/verification-NEW --host 172.28.176.1
```

Use a new directory and an otherwise empty CARLA world. Host/port: `172.28.176.1:2000`.
Client/server: 0.9.16; map: `Carla/Maps/Town10HD_Opt`;
OpenDRIVE SHA256: `5d883b799f634030af92be1e9d79d107845540ba04338e8c60e095be1aef7be7`.
The verifier owns synchronous 0.05 s ticks and uses ClearNoon. All three attempts restored
settings/weather and leaked no actors. The archive retains all three reports, raw callbacks,
stdout, exact mission/scenario files, source bytes and read-only geometry probes.

## Results and failure classification

| Check | Result |
|---|---|
| S1/S2/S4/S5 repeatability | PASS: each spawned twice; zero measured position/yaw differences |
| S3 +3.5 m at s=40 | FAIL: spawn rejected; exact non-projected Any-lane lookup is road 20, lane 2, Median |
| Car gaps 3/6/10 m; lateral +0.8 m | PASS in attempt 03: maximum absolute gap error 0.000001364 m |
| Car lateral +3.5 m | FAIL: spawn rejected; adjacent-road evidence not established |
| Pedestrian gaps 3/6 m | PASS: errors +0.000000923 / -0.000000127 m |
| Two-contact collision check | PASS in attempt 03: two contacts against the target, raw callbacks retained |
| Lane invasion | PASS: four raw Broken callbacks; four undeclared events, zero with explicit legal declaration |

Attempt 01 stopped at S3 (INCOMPLETE); attempt 02 continued independent checks and
reported MEASURED_FAIL (7 PASS, 7 FAIL). Attempt 03 is MEASURED_FAIL (12 PASS, 2 FAIL).
Failed reports were not rewritten. Attempt 03 followed a justified verifier cleanup fix:
a new actor may not be alive in the client snapshot until a synchronous tick. Checking
liveness before that tick skipped destruction of geometry probes. Those probes blocked
later vehicle spawns and contaminated later ground projections with their roofs.
Ticking before cleanup fixes this lifecycle bug; a regression covers the pending snapshot.
Car gaps and the live collision check then passed without geometry/threshold changes.
The exact S3 location remains on a median, not an adjacent driving lane: this is a
scenario feasibility defect requiring owner-reviewed geometry, not a tolerance adjustment.
The +3.5 m lateral car site also still rejects spawning; its outside-corridor truth check
remains unverified. No live data establishes an adjacent driving lane for either case.
No environment/map/route changes were made to obtain a pass.

## Next required evidence

Agree a defensible S3 adjacent-driving-lane scenario on the frozen route, resolve the +3.5 m lateral test-site feasibility without relaxing truth geometry, then rerun the full verifier.
A genuine MEASURED_PASS and complete adjacent-lane/lateral truth evidence are still required for #39.
Pedestrian evaluation truth checks do not admit pedestrian operational control.

## Verification of packaging

`manifest.json` records archive SHA256 and every uncompressed member's size/hash.
The ZIP uses the project's compressed-raw-evidence convention. Verify all hashes before reuse;
the source snapshot reproduces the measured dirty run, not merely its committed base.

## Final local validation

- Focused scenario/world/event/stop-truth/isolation tests: 61 passed.
- Full pytest: 875 passed, 1 skipped, 3 subtests passed.
- Ruff: passed.
- Strict Sphinx (`sphinx-build -W docs docs/_build/html`): passed.
- Simulator boundary (`python scripts/check_simulator_boundary.py --smoke`): AST passed;
  83 core modules imported with simulator imports blocked.
- Ordinary `git diff --check`: passed.
- Archive SHA256, all 40 member sizes/hashes, attempt02/03 source and mission hashes: passed.
- Live command: executed three times in distinct output directories; final status
  MEASURED_FAIL, so #39 acceptance/merge readiness remains blocked.

The live run produced two raw collision callbacks, one per separated contact. Continuous
multi-callback deduplication is covered offline; this run does not separately demonstrate
a callback burst during a single contact.
