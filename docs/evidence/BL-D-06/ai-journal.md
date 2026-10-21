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

21 October 2026 follow-up: reviewed fresh 600-frame BL-C capture without rerunning
CARLA. Verified current runtime against the archived warm snapshot: ten identical
runtime files, adapter expiry AST unchanged, only post-decision documentation/error
detail differs. Cold backlog cannot authorise a new positive packet; operational
0.95 s budget is independently composed, not 4.199442 s of admitted latency.
Offline all-track replay with the actual 25 m depth policy and optimistic freshness
still produces 600 HOLDs, despite valid target depth. Missing fresh capture is
resolved; joint operational range/admission evidence is not. Numeric thresholds
and baseline candidates were preserved, frozen cruise/response remain null.
G1 remains technically blocked, independently of team review. No GitHub/PR #90
state, commits, pushes or authorship/history were changed.

Fresh-range follow-up validation: **158 focused tests passed**; **818 full-suite tests passed, 1 skipped, 3 subtests passed**. Ruff, Sphinx `-W`, local decision-document links, numeric/boolean/null configuration preservation, unchanged acceptance bytes, exact review reproduction and `git diff --check` passed. No simulator recapture was performed.

21 October rejection investigation: extended only offline diagnostic accounting,
replayed the existing 600 raw observations, separated policy predicates/instances,
TTL facts and reconstructed versus missing host timing. Persistent background car
beyond the validated depth range and off-axis mixed-depth bus explain current
all-track HOLDs; no runtime false-rejection bug or justified threshold change was
found. Recorded exact combinations and a minimum same-frame operational approach
capture specification. No new CARLA run, G1/PR #90 edit or GitHub change. Validation:
162 focused tests; 822 full-suite passed, 1 skipped, 3 subtests; Ruff, strict Sphinx,
review reproduction and diff checks passed.

21 October replacement operational capture: reused current RGBDCapture/owner,
SkeletonStack/freshness/all-track policy, actual governor and PathFollower. Retained
site-152 failed preflight and all four additional background-site checks, selected
site 93 before final collection. Recorded 800 real same-frame source observations,
254 admitted observations and 64 positive shadow outputs; actuators stayed braked.
Injected one labelled post-read dropout and explicit reset recovery. No evaluation
labels entered control, no unrelated detections or validity thresholds were removed.
Archived source/config, full metadata, per-stage clocks and raw artifact hashes.
Deterministic policy/provider/controller replay has zero mismatches. The bounded
4.0 m/s technical candidate fits 22.348600 m at the enforced 0.9500000052 s response;
5.7 m/s is only the calculated envelope cap. Updated local freeze candidate and
review, retaining old BL-C/rejection evidence. Human review/publication/G1 admission
and closed-loop acceptance are separate; no PR #90 or GitHub state changed.

Geometry audit corrected the initial range attempt: kinematic targets must use
road elevation rather than elevated spawn-point Z. Preserved that complete run
as superseded, plus two setup failures (ego spawn placement and helper NameError),
all cleanup verified. Final same-frame-04 has normal road-height targets, 800 rows,
345 admitted observations, 260 matching-target admissions, 22 positive shadow
packets and zero invalid positives. Full output p95/max 45.949/60.974 ms; common
conservative range remains 22.348600 m. Only this final run supplies the ordinary-
road candidate freeze. Earlier timings/counts above describe the superseded run.
No G1/PR #90 or GitHub state was changed.

Final corrected-run validation: 170 focused tests passed; full pytest 830 passed,
1 skipped, 3 subtests passed; Ruff, strict Sphinx, exact review/config-source
reproduction and diff checks passed. Local G1 and PR #90 remained untouched.
The source-position regression prevents the setup-helper error from recurring.
The 5.7 m/s mathematical envelope is not a driven-speed claim; local candidate
ceiling remains 4.0 m/s, with publication/human admission still pending.

## PR #97 reviewer follow-up

Preserved fff2c4a and measured captures. Changed current status to technically
validated candidate, with null cruise/response frozen values: the existing loader
rejects it pending PR #90 / #38 and team admission. Scoped 4.0 m/s to stationary
vehicle baseline; pedestrian operational admission remains deferred and depth
validation unchanged. Corrected root module commands and CSV CRLF; original raw
CSV hash remains in its raw manifest with normalized-copy provenance. Following
TEST-02, byte-verified capture/history ZIP plus checksum/member manifest replaces
large JSON duplicates; failed attempts remain archived. Added command, disabled
loader and archive-corruption regressions. Historical captured configuration and
publication wording are immutable measurement metadata, not current admission.
