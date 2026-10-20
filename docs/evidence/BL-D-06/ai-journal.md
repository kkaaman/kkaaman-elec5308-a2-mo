# AI assistance journal — BL-D-06

OpenAI Codex assisted with code inspection, implementation, offline checks and
CARLA tooling/evidence analysis at the user's direction. **Human review of the
current D-006 decision remains pending.** Passing automated checks does not imply
AI approval of safety thresholds, gate admission or closed-loop stopping.

## 20 October 2026: D-004 historical cold-path review

Inspected the corridor, guard/governor, lifecycle and measured inputs. Retained
hashed JSON snapshots instead of importing unrelated implementation history.
The then-reported BL-C response produced the historical 31.9508 m requirement at
4.0 m/s; that argument blocked the initial freeze. The later sensor-wait correction
makes the old timing/cap a lower-bound diagnostic, not valid current evidence.
D-004's arithmetic and source fingerprints remain traceable in cold-review.json.

## 20 October 2026: D-005 historical operational timing/inference

Reused SkeletonStack, ObjectDetector.track(), TrackDepthFilter, ClearanceGuard,
SpeedGovernor and PathFollower. Added an opt-in capture-age/warm-up gate, reset
handling and host/simulation command-expiry checks. Missing/invalid/stale results
HOLD without becoming clearance; evaluation truth stays outside runtime control.

Real CARLA 0.9.16 / Town10HD_Opt trials used the existing RGB-D rig and local
YOLO/BoT-SORT settings. Failed/held runs and the same-frame sensor probe were
retained. The selected parked-ego run captured 240 frames across two resets,
122 valid positive shadow decisions and zero cold/injected-stale positive outputs.
The exact executed uncommitted source snapshot and raw timing ledger remain in
warm-source.zip; warm-operational.json is unchanged raw measured evidence.

The D-005 proposal inferred a 4.0 m/s gated freeze from the 0.9500000052 s response
composition and assumed 25 m usable visibility. **That inference was withdrawn by
D-006**: axial-depth accuracy does not establish usable confirmed obstacle range.
The source measurements survive; the old freeze conclusion is historical only.
Original D-005 decision/review bytes are retained under history/.

## 20 October 2026: corrected BL-C timing and D-006

Inspected PR #93's correction: capture records actual sensor read/conversion time
and replay charges it once, including failures. The legacy 600-frame capture lacks
that metadata, so its old confirmation latencies and 4.9 m/s screen do not establish
current speed support. CARLA was unavailable for the requested recapture; no new
range/timing value was invented.

The independent operational bound remains 0.5 s capture age + 0.1 s distinct
command-delivery reserve + 0.3500000052 s measured vehicle lag. It does not prove
25 m bumper visibility, timely first confirmation or closed-loop braking.
D-006 is BLOCKED pending fresh BL-C capture and usable-range review. Cruise and
response frozen values are null, implementation policy is separate, and the
navigation cruise and acceptance criteria remain unchanged.

## 20 October 2026: human-authored corridor review fixes

The reviewer identified missing outer-turn sweep and endpoint front-overhang
coverage. After inspecting history/patch equivalence, integrated only Zhaolin-Huang's
fix as 61e2d1a40c4b65229621b2f18b052861e975974f, preserving original author/message.
His regression tests and documentation were retained. The outer-turn example now
intersects and the endpoint example returns -1 m. This addresses geometry defects;
it supplies no new BL-C range/timing evidence and does not unblock D-006.

## Evidence hygiene and validation

Audited every artifact, committed-source fingerprint, archive member, raw timing
row and XML case set. Kept the measured source archive because normal committed
history does not reproduce the executed uncommitted snapshot. Removed temporary
PR drafts, generated file lists, repository-operation audits and XML superseded by
the final focused suite. Historical decision/measurement bytes were preserved.
Clarified stale selection/config metadata and separated current from historical
claims. No runtime behaviour, Git history, staging or GitHub state was changed
by this hygiene pass. Human D-006 review and physical acceptance remain pending.

Final hygiene validation: **154 focused tests passed**; **814 full-suite tests
passed**, 1 skipped, 3 subtests passed. Ruff, Sphinx `-W`, whitespace checks and
byte-identical review reproduction passed. Runtime/script/test files, effective
configuration and navigation, Git HEAD/index, and raw/historical evidence were
verified unchanged. Automated validation is not human safety approval.
