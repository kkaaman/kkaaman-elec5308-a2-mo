# AI assistance — issue 39 / PR 98

2026-10-25: Codex inspected canonical G1 and PR98, preserved original authored commits,
and added focused regressions for removal qualification, destroy failure, and interrupted
HOLD. It added strict non-projected S3 lane verification, per-case failure retention and
source provenance; no scenario geometry, control policy or acceptance threshold changed.
Three live verifier commands ran against the user's CARLA server. All failed and are retained
with raw outputs. Geometry probing identified S3's exact requested point on a Median.
Independent checks continued in attempt 02, including real lane-invasion callbacks.
Attempt03 corrected snapshot-dependent probe cleanup after a focused regression.
Car distance/lateral-overlap and real two-contact collision checks then passed.
S3 and the +3.5 m lateral case remain blocked; no MEASURED_PASS was fabricated.
This work does not perform issue41 physical-stop acceptance or change G1 admission.
No commits, pushes, merges, approvals or GitHub state changes were performed.


Follow-up: live topology identified a genuine right adjacent Driving lane at unchanged
s=40 m. With explicit user authorization, S3 changed only lateral +3.5 to −3.5 m;
S1/S2/S4/S5 and the ego route stayed unchanged. The lateral truth probe now derives its
offset from S3. Strict centre/four-corner lane checks and a negative footprint regression
were added. Fresh attempt04 generated MEASURED_PASS (14/14), retaining all prior failures.
Exact source/config snapshots and topology are packaged separately. Canonical main was
fetched and compared; no merge or admission bypass was performed. This does not run #41.
