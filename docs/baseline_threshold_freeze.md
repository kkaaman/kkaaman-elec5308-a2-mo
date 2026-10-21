# BL-D-06 operational first-admission review — 21 October 2026

**Technical decision D-008: PASS for the bounded freshness-gated candidate.**
The new [same-frame operational capture](evidence/BL-D-06/operational-approach/README.md)
resolves the previous joint range/admission blocker. **4.0 m/s** is supported with
**0.9500000052154064 s** response, **1.497566177392398 m/s²** deceleration and
unchanged **6 m** stopping margin. The highest calculated envelope ceiling is
**5.7 m/s**, but the proposed configuration retains the unchanged 4.0 m/s candidate.

This is an **uncommitted local technical candidate**, not full-team G1/G2 admission,
closed-loop stopping acceptance or demonstrated driving at the calculated cap.
No GitHub state or PR #90 source branch has changed.

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
`baseline_threshold_freeze`, is `FROZEN_GATED_OPERATIONAL` with:

| Frozen envelope value | Value |
|---|---:|
| Cruise ceiling | 4.0 m/s |
| Conservative deceleration | 1.497566177392398 m/s² |
| Response budget | 0.9500000052154064 s |
| Upper stop margin | 6.0 m |

All gate, tracking, depth-validity, detector/model/tracker, rig and implementation
policy values are unchanged. The unchanged clearance guard still requires five
fresh clear frames and 0.4 s; this provider never fabricates valid empty clearance.
The baseline operational loader enforces the bounded profile. Legacy 320×180 S0
navigation configuration does not replace the **640×360 operational rig**.
[Provenance](evidence/BL-D-06/operational-approach/provenance.json) pins the local
candidate config SHA256 and base commit plus exact measured source archive.
A new committed G2-admitted source revision does not yet exist: publication and
human review are explicitly outstanding.

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
