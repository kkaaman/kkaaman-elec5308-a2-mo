# Operational admission rejection investigation — 21 October 2026

The existing 600-frame BL-C capture was reanalysed; no CARLA experiment, runtime
policy/threshold change, PR #90 or G1 checklist edit was made. All pre-existing
uncommitted work was protected in a byte-verified archive before this investigation.
[Expanded machine report](operational-range-review.json) retains every source row,
all-track diagnostics, predicate failures and overlapping combinations.

## Scope and correction to interpretation

The prior audit explicitly calls `TrackedGovernancePolicy.evaluate` with
`fresh=True, ready=True`. It is a **policy-only diagnostic**, not a reconstruction
of SkeletonStack admission or actual vehicle control. The zero result is real for
that policy diagnostic: every frame contains a current invalid track. Missing
host clocks do not cause that policy-layer zero, because clocks are bypassed.
The replay's synthetic zero ego pose and throttle stub are not measured control
inputs/results and cannot certify complete operational motion. They do not cause
the observed HOLDs, which precede controller invocation.

No false-rejection bug was found in the policy replay. Its reporting was incomplete:
noncurrent/expired tracks and provider conjuncts were not separately counted, track
instance counts were not frame counts, and the full timing state could not be
reconstructed. The audit now exposes those distinctions and tests their accounting.
It must not call zero forced-warmup failures evidence that real warm-up passed.

## Exact frame counts (conditions overlap)

| Failed policy conjunct | Approach | Short occlusion | Continuity risk | Total |
|---|---:|---:|---:|---:|
| Invalid depth or confirmation in any retained track | 200 | 200 | 200 | 600 |
| Noncurrent retained track | 23 | 25 | 36 | 84 |
| Expired track emitted this frame | 2 | 2 | 3 | 7 |
| Provider invalid | 1 | 6 | 3 | 10 |
| Missing visible distance | 1 | 6 | 3 | 10 |
| Neither obstacle gap nor validated clearance | 1 | 6 | 3 | 10 |
| Missing/None track ID | 0 | 0 | 0 | 0 |

Source batches/depth arrays are present and valid on all 600 frames; there are no
sensor-read failure rows. Source/model/frame matching and settings checks pass.
Warm-up is forced ready in this diagnostic; its zero failure count is **bypassed**,
not measured. No observation reaches valid governance, emergency/normal speed
limiting or controller invocation. Invalidity returns HOLD before those stages.

Exact mutually exclusive **policy predicate combinations** (sum=600):

| Combination | Frames |
|---|---:|
| Invalid track only | 503 |
| Invalid track + noncurrent track | 80 |
| Invalid track + expiry | 7 |
| Invalid track + provider invalid + missing visibility + missing obstacle/clearance | 6 |
| Same four failures + noncurrent track | 4 |

All combinations also have **recorded source TTL failure** if this four-tick
capture schedule is offered to the operational gate. Three of the six frames in
the fourth combination also exceed processing budget from detector+filter alone.
The JSON separately records those joint-known combinations and every per-sequence
combination; no overlapping categories are added as independent frame totals.

Current invalid-track reason counts:

| Track reason | Frames containing reason | Track instances |
|---|---:|---:|
| insufficient_valid_depth | 596 | 786 |
| mixed_or_spread_depth | 373 | 373 |
| unconfirmed | 19 | 25 |

Exact current-track reason combinations (sum=600): insufficient only 217;
insufficient+mixed 363; insufficient+mixed+unconfirmed 9;
insufficient+unconfirmed 7; mixed only 1; unconfirmed only 3.

## What the scene actually contains

At least one invalid current detection **not evaluation-labelled as the target**
exists in every frame (600/600). The label is used for reporting only, after
policy execution; an unlabelled detection cannot be discarded in control.

- Approach car ID 1: current/invalid **200/200**, never target-labelled. Raw
  central-ROI median **31.632–31.662 m**. First frame unconfirmed; subsequent
  **199** fail sufficient depth because the validated operational depth policy
  excludes pixels at/above **25 m**.
- Short-occlusion car ID 2: invalid **200/200** (1 unconfirmed, 199 insufficient),
  raw median **31.582–31.699 m**. Bus ID 1: invalid **197** current observations,
  including **195 mixed-depth** rejections, raw median about **8.046–8.104 m**.
- Continuity car ID 2: invalid **199** current observations (1 unconfirmed,
  198 insufficient), median **31.567–31.662 m**. Bus ID 1: invalid **180** current
  observations, including **178 mixed-depth** failures. Other invalid tracks cover
  the remaining frame as well.

Existing RGB frames were visually inspected: the capture contains a background
car on the left and the scripted bus remains visible on the right when moved
laterally off the target axis. Off-axis does **not** mean outside the camera FOV.
The all-track policy does not discard off-route/off-axis detections based on
image position alone. No route/footprint exclusion evidence authorises doing so.
Whether every detection corresponds to a genuine physical obstacle is immaterial
to this guard: uncertain perception cannot be called clear space.

Raw-ROI diagnostics find depth support beyond the 25 m cutoff on **599 frames**,
and mixed-depth support on **377** (overlap; raw estimates can be masked by the
track's earlier unconfirmed reason). Three frames have other insufficient support
near the cutoff; eight include confirmation-only failures with otherwise valid
instantaneous depth. These are diagnostic raw depths, **not accepted measurements
beyond the reviewed range**. Neither zeros nor missing depth files cause the
persistent 31.6 m failure.

## Timing and missing-evidence classification

| Condition | Count | Interpretation |
|---|---:|---|
| Recorded source delay >= 0.1 s TTL | 600 | Real capture scheduling mismatch: four ticks = about 0.2 s. Correct operational rejection; not a replay bug. |
| Reconstructed serial age > 0.5 s | 200 | All approach rows; diagnostic backlog reconstruction, not a measured host capture-age trace. |
| Detector+filter alone > 0.1 s | 3 | Index 0 of each sequence: 3.992022448 / 0.162457243 / 0.122157230 s. Full processing necessarily also fails. |
| Missing owner capture-origin host timestamp | 600 | Evidence field absent; actual host-age gate cannot be reconstructed. Not an observed live sensor failure. |
| Missing full governor/controller processing boundaries | 600 | Full processing/decision timing unknown, even where detector+filter fit. |

Actual detector generation/readiness, current simulation time at final decision,
packet expiry/currentness, policy observation validity/reason, controller result,
provider geometry provenance and per-frame ego/route geometry were not captured
as operational inputs/outputs. Their historical operational states cannot be
fabricated. The existing warm run supplies those timing boundaries at another
site, but cannot supply them retrospectively for these source observations.

This means there are two independent all-frame blockers: the forced-fresh policy
rejects every row for current invalid tracks; the recorded capture schedule is
already source-expired for operational use. Missing timing fields additionally
prevent a complete live admission trace. Reducing cruise speed fixes none of them.

## Source-of-truth rules and classification

| Rule / source | Classification |
|---|---|
| `src/autonomy/obstacle_policy.py`, `TrackedGovernancePolicy.evaluate`: unknown means any untracked/expired/not-current/invalid-depth track; valid also requires provider visibility and gap/clearance | Expected current conservative policy, including scenario background/occluder tracks. `test_fresh_confirmed_target_does_not_override_other_invalid_track` locks this behaviour. |
| `src/obstacle_perception/tracking.py`, `TrackDepthFilter.update`: 2-of-3, current visibility, expiry and aligned valid depth | Expected startup/flicker/occlusion/expiry semantics; tested in tracking/freshness suites. |
| `src/obstacle_perception/box_depth.py`, `estimate_box_depth`; `configs/mission_baseline.json` depth policy | Expected 25 m cutoff, minimum pixel/fraction and mixed-surface checks; no threshold adjustment justified. |
| `src/autonomy/freshness.py` and `src/autonomy/stack.py`: 0.5 s host age, 0.1 s processing/source TTL, warm-up and final check | Correct guards; source TTL is known failed, full host-age/readiness evidence missing. Cold 4.199442 s cannot authorise new positive output. |
| `scripts/capture_obstacle_tracks.py`: SENSOR_LAG_TICKS=4, recorded sensor_delay_s, detector/filter/read durations | Correct BL-C benchmark design, unsuitable operational source schedule and incomplete operational evidence fields. |
| `scripts/review_operational_range.py`: forced fresh/ready, shadow straight-point provider | Diagnostic-only scope; reporting coverage corrected, no policy threshold or rejection-semantic bug found. |
| `tests/unit/test_operational_range_audit.py` | Frame/instance separation, overlapping conjuncts and raw out-of-scope-depth diagnostics verified. |

## Reuse and smallest correct next step

**Reuse the existing capture** for BL-C target confirmation, read-wait replay,
ID/occlusion/depth regressions and this exact diagnostic. It is **not reusable as
a complete operational first-admission range run**. Removing non-target detections,
raising the 25 m limit or relabelling a 0.2 s-old packet fresh would weaken policy;
missing host timing cannot be recovered from the old arrays. No capture defect
requires repeating the same BL-C experiment.

To obtain operational range evidence, a **new same-frame operational approach
measurement is required**, using the existing RGBDCapture/SkeletonStack/policy
and timing path, not the four-tick BL-C schedule. The smallest evidence collector
change is to combine the approach scenario with that same-frame timing path:

1. Preflight the unchanged model/640/.25/BoT-SORT/rig at a controlled site (the
   existing site 152 has demonstrated warm positives, but is not pre-approved
   for an approach). Ensure unrelated visible detections have valid support, or
   choose/place the scene so they are genuinely outside the FOV. An empty actor
   list alone does not exclude background/map vehicles. Do not filter results
   using evaluation labels or cherry-pick successful frames.
2. Keep one evaluation target approaching through the validated depth scope.
   Preserve expected startup/out-of-range/occlusion HOLDs and every failed read.
   Move the scripted occluder truly out of view between occlusions, or retain its
   valid operational observations; merely offsetting it 7 m is insufficient.
3. Process the current frame without the four-tick delay; stamp host capture
   origin **before owner tick**, retain receipt/read/conversion, detector/filter,
   actual provider/governor/controller and final decision/dispatch boundaries,
   current simulation time, detector generation, fresh-result count, gate/policy
   reasons, packet expiry/currentness and actual shadow target/command.
4. Record real ego state, vehicle/front-bumper reference and calibrated mount,
   route/corridor/provider inputs and geometry. Reuse measured rig/deceleration
   provenance with verified identity. No zero-pose/always-throttle substitutes
   may be called a full operational timing result. Ego actuators may remain
   physically braked if the measurement is explicitly shadow decision evidence.
5. Apply target labels/gaps only after all operational computations. Establish
   the first **actually admissible** confirmed target/visibility range and pair
   it with its measured timing, resets/HOLDs and unchanged stopping envelope.
   Retain all attempts; it is not necessary or expected for every frame to pass.

No broader G1 changes or new CARLA run were performed during this investigation.
The next action is the narrowly scoped same-frame approach collector/evidence,
not a threshold change or another identical BL-C capture.

## Validation

162 focused tests passed, including four new diagnostic-accounting regressions.
Full pytest: 822 passed, 1 skipped, 3 subtests passed. Ruff, strict Sphinx and
`git diff --check` passed after connecting this report to the evidence toctree.
The saved BL-D review reproduces exactly; G1/PR #90 source files remain unchanged
from the start of this investigation. [JUnit](rejection-investigation-tests.xml).
