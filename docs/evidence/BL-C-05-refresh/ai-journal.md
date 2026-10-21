# AI assistance journal — BL-C-05 follow-up

21 October 2026: OpenAI Codex inspected freshly fetched canonical main, created
an isolated follow-up branch and reviewed schema-3 capture, the RGB-D joiner,
filter failure semantics, measured waits/replay and regression tests. No runtime
implementation, threshold, model/tracker policy or historical evidence was changed.

Verified the local model SHA256 and invoked the actual collector for three
200-frame scenarios using existing calibration/braking inputs. CARLA timed out
at client.get_world with its configured 20000 ms wait. No actor/sensor or fresh
frame existed; requested server/map/rig were not passed off as observed data.
Exact failure, whole-process duration and local provenance are retained; process
duration is not a per-frame sensor wait. No historical/synthetic benchmark was
substituted for new live evidence.

Documented the blocked attempt and ran offline checks. These tests do not approve
4.0 m/s, establish usable range or certify stopping. D-006/G1/#38 are unchanged;
human review and successful live measurement remain necessary. No commit, push,
merge, issue closure or GitHub-state change was performed.

Validation: focused BL-C/RGB-D 60 passed; Ruff passed; full pytest 814 passed,
1 skipped, 3 subtests passed; strict Sphinx and git diff --check passed.

Continuation, 21 October 2026: user confirmed CARLA was previously not started.
Reclassified the first attempt as setup-only, retained it under setup-attempt/,
and reran the unchanged collector successfully: 600 valid reads, settings restored,
no actor leaks. Ran actual replay and per-frame filter/depth analysis. Preserved
cold timing, misses, ID switch, queue statistics and all measured read durations.
Versioned source rows/reports/hash manifest; raw arrays remain locally retained.

The raw first-event screen is 5.5 m/s. Explicitly separated its out-of-reviewed-scope
first-event depths from a common 24.013 m sampled bumper range within axial 25 m.
Combining that range with all-frame worst response and vehicle lag once gives a
3.2 m/s conservative screen, insufficient for unchanged 4.0 m/s. No runtime code,
threshold, baseline, BL-D/G1 decision or GitHub state was changed. This remains
screening, not safety certification; human review and operational evidence remain.

Fresh-run validation rerun: 60 focused passed; Ruff passed; full pytest 814 passed,
1 skipped, 3 subtests passed; strict Sphinx and git diff --check passed.

G1 follow-up: BL-C-05 capture/reporting scope passes; all-track operational-range
audit remains blocked. Existing captured data was replayed, not recaptured.
Fresh-range follow-up validation: **158 focused tests passed**; **818 full-suite tests passed, 1 skipped, 3 subtests passed**. Ruff, Sphinx `-W`, local decision-document links, numeric/boolean/null configuration preservation, unchanged acceptance bytes, exact review reproduction and `git diff --check` passed. No simulator recapture was performed.
