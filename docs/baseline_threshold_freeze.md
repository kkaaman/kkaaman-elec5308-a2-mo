# BL-D-06 operational first-admission review — 21 October 2026

**25 October G1 update:** the working tree now contains a strict-identity frozen
stationary-vehicle admission proposal. See [G1 admission](gates/G1.md) and its
[versioned record](gates/G1-admission.json). Publication/final designated review
remain pending. The original technical candidate review below and all recorded
measurements are retained unchanged as provenance, not rewritten as live acceptance.

**Technical decision D-008: PASS for the bounded freshness-gated stationary-vehicle candidate.**
The new [same-frame operational capture](evidence/BL-D-06/operational-approach/README.md)
resolves the previous joint range/admission blocker. **4.0 m/s** is supported with
**0.9500000052154064 s** response, **1.497566177392398 m/s²** deceleration and
unchanged **6 m** stopping margin. The highest calculated envelope ceiling is
**5.7 m/s**, but the proposed configuration retains the unchanged 4.0 m/s candidate.

This is a **published PR #97 technical candidate**, not full-team G1/G2 admission,
closed-loop stopping acceptance or demonstrated driving at the calculated cap.
PR #97 publishes this technical candidate; PR #90 / issue #38 retain responsibility for G1 admission after canonical publication and full-team review.

## What the replacement proves

800 source frames at preflighted site 93 on CARLA 0.9.16 / Town10HD_Opt use the
same-frame RGB-D path, real ObjectDetector.track(), unchanged tracker/filter,
all-track guard, governor and PathFollower. The final road-height run supersedes an elevated-target attempt and two setup
failures, all retained in the attempt index. All detections remain operational
inputs. Target labels/geometry are evaluated only after control computation.
The ego remains physically braked; positive commands are measured shadow outputs.

There are **345 admissible observations**, **260 matching-target admissions**,
**22 positive shadow packets** and **zero invalid/cold/stale positive packets**.
The observed first-admission operational gaps are 22.350923, 22.350923, 22.350923
and 22.348600 m. The common conservative operating range is
**22.348599722608924 m**, selected as the minimum perceived/evaluation bumper gap
at each scenario's first actual target admission. A supported unrelated traffic
light remains included and makes the operational gap shorter than target truth.
No 25 m axial value is substituted for bumper range.

Actual source simulation age at every decision is **0.0 s**. Source host capture
origin is measured before owner tick; complete output p95/max are
**45.948829 / 60.973706 ms**. Detector/tracker, filter, provider, governor,
controller and remaining overhead partition the interval without overlap/omission.
The cold maximum **3.609765 s** returns HOLD. Short/long occlusion and one explicitly
injected missing-depth input are retained; reset/warm-up recovery is explicit,
not a claim that disappearance or expiry clears the road.

## Operational response and stopping consistency

The existing gate allows at most 0.5 s capture age and 0.1 s processing, requires
two fresh results, checks source simulation expiry and rechecks after control.
Adapter dispatch checks both expiry clocks. The previous BL-C serial backlog
**4.199442 s** remains valid evidence for that different capture schedule but
cannot authorise positive output under these bounds.

Use the **full enforced age**, not just the measured warm maximum:

```text
response = 0.5 s admitted capture age
         + 0.1 s distinct delivery reserve
         + 0.3500000052154064 s measured vehicle lag
         = 0.9500000052154064 s
required_gap(4) = 15.142001009378749 m
measured operational first-admission range = 22.348599722608924 m
calculated maximum speed = 5.7180586377280065 m/s (report 5.7)
```

Measured post-decision brake acknowledgement max **10.138 ms** fits the separate
100 ms delivery reserve. Vehicle lag is used once; sensor/detector/filter/governor/
controller are already inside capture-to-output. The 100 ms acceptance definition
is unchanged. Receipt-to-brake-ack p95 is **31.999 ms**, explicitly for physical
brake commands, not a positive driving/stopping trial.

## Exact local candidate configuration

[configs/mission_baseline.json](../configs/mission_baseline.json), subsection
`baseline_threshold_freeze`, is `TECHNICALLY_VALIDATED_CANDIDATE` with the values below under `candidate_values`; cruise/response remain null in `frozen_values`:

| Technically validated candidate value | Value |
|---|---:|
| Cruise ceiling | 4.0 m/s |
| Conservative deceleration | 1.497566177392398 m/s² |
| Response budget | 0.9500000052154064 s |
| Upper stop margin | 6.0 m |

All gate, tracking, depth-validity, detector/model/tracker, rig and implementation
policy values are unchanged. The unchanged clearance guard still requires five
fresh clear frames and 0.4 s; this provider never fabricates valid empty clearance.
The frozen operational loader rejects this candidate status. PR #90 / issue #38 must explicitly publish the admitted freeze after required team review. Legacy 320×180 S0
navigation configuration does not replace the **640×360 operational rig**.
[Provenance](evidence/BL-D-06/operational-approach/provenance.json) pins the local
candidate config SHA256 and base commit plus exact measured source archive.
The candidate is published in PR #97 but is not G2-admitted; canonical publication and full-team G1 review remain outstanding.

A lower speed is permitted by D-003/issue #34; 4.0 m/s is not mandatory. The old
BL-C 3.2 m/s recommendation remains a correct conservative target-only screen
for its serial schedule, not a ceiling on the separately verified bounded path.
No withdrawn historical 4.9 m/s result is used.

## Scope, preserved counterexamples and remaining review

The provider includes every visible track conservatively as supported surface
geometry and performs no off-corridor exclusion. It does not prove full unseen
corridor coverage, arbitrary bends or physical stop margins. Unknown tracks,
missing depth, cold/backlogged output, source expiry and interrupted obstacle
HOLD still prevent positive output. The timing interpretation requires the
exclusive synchronous owner/checked dispatch, not an asynchronous last-throttle
loop without a watchdog. Closed-loop S1–S5 acceptance remains separate.

The [600-frame BL-C rejection investigation](evidence/BL-D-06/admission-rejection-investigation.md)
remains unchanged and valid for that dataset: out-of-range background tracks and
its 0.2 s schedule still reject all rows. The new scene/schedule/evidence does not
rewrite that result. Site-152 and site-22 unsuccessful preflights are retained.
[Previous blocked decision](evidence/BL-D-06/history/D007-rejection-decision.txt)
and [review](evidence/BL-D-06/history/D007-rejection-review.json), D-004/D-005 and
pre-refresh D-006 remain historical. Reviewer corridor fixes and tests are retained.

The range/admission **technical blocker is resolved for this bounded candidate**.
Human BL-D review, durable raw evidence access, publication of the exact candidate
configuration and full-team G1/G2 admission are still required. The G1/PR #90
checklists have not been modified by this capture task.

```{toctree}
:maxdepth: 1

evidence/BL-D-06/README
evidence/BL-D-06/ai-journal
```

## Operational scope and admission boundary (PR #97 review)

The **4.0 m/s technically validated candidate is limited to the stationary-vehicle
obstacle profile for the G2 stationary-car baseline**. The 22.348600 m operational
first-admission evidence uses a stationary vehicle target with a controlled approach.
Pedestrian operational first-admission range is **unestablished / deferred**.
Existing car/person depth-validation evidence is preserved: pedestrian depth
accuracy does not establish pedestrian operational detection/admission range.
No new pedestrian capture or #39–#41 implementation is claimed here.

`TECHNICALLY_VALIDATED_CANDIDATE` is not `FROZEN_GATED_OPERATIONAL`:
the existing frozen loader rejects it. The measured candidate inputs are retained,
but cruise/response frozen fields remain null. Only PR #90 / issue #38 and required
team review may publish the actual G1/G2 admission. The 5.7 m/s arithmetic ceiling
is not an admitted or certified driving speed.

## G1 production identity correction — 25 October 2026

The prior status-only loader accepted altered frozen values or broadened scope.
It now verifies the exact versioned config path, deterministic full-config
identity and source/config/calibration inventory against the G1 admission record.
Candidate-only, wrong binding, modified source/config, altered numeric values and
pedestrian/general scope changes fail closed before operational construction.
This startup-only validation changes no measured detector/control processing path.
The PR #97 evidence review intentionally remains a technical-candidate result;
formal admission is a separate record, preserving the original candidate history.

Prepared frozen inputs remain 4.0 m/s stationary-vehicle only, 1.497566177392398
m/s², 0.9500000052154064 s and 6.0 m. No pedestrian first-admission, general
visibility or downstream live stop is certified. BL-C 3.2 m/s, withdrawn 4.9 m/s
and non-admitted 5.7 m/s retain their distinct meanings. See the G1 proposal for
current identity/source hashes and the revised designated-review process.
