# BL-D-06 integration re-review — 20 October 2026

**Decision D-006: BLOCKED pending fresh BL-C capture and range review.**
Canonical main now includes PR #93's measured sensor-wait correction. Its old
4.9 m/s screening cap and first-confirmation times are superseded and do not
establish support for 4.0 m/s. The proposed navigation cruise and acceptance
criteria are unchanged; the operational frozen-profile loader now rejects the
blocked baseline.

D-004's [cold review](evidence/BL-D-06/cold-review.json) is preserved as historical
lower-bound arithmetic, not valid current BL-C timing. D-005's previous gated
freeze decision is preserved [verbatim](evidence/BL-D-06/history/D005-decision.txt)
with its [original review](evidence/BL-D-06/history/D005-review.json). D-006
withdraws that freeze claim after re-examining the usable-range assumption; it
does not discard the independently measured operational timing.

## What changed in the runtime

The existing SkeletonStack remains the stack. Its default latched-fault behaviour
is preserved; the tracked operational entry point opts into the new gate.
The composition reuses ObjectDetector.track(), the unchanged committed BL-C
TrackDepthFilter, ClearanceGuard, SpeedGovernor and PathFollower.

The baseline loader (`frozen_governor_config` / `baseline_operational_stack`)
rejects this blocked configuration. Unit tests exercise its startup gate only
with explicitly hypothetical accepted fixtures; they do not admit the real baseline.
Reset installs the episode, reloads detector history, resets confirmation, and
starts with zero target/full brake. The gate requires two consecutive fresh
completed detector results, taken from the existing TrackPolicy.hits value.
Track confirmation/depth validity and corridor validity must independently pass.

The gate derives its limits from the unchanged configs/stack.yaml contracts:

| Check | Existing value | Enforced meaning |
|---|---:|---|
| Capture age | 0.5 s | Host time from before the owner tick through final control availability |
| Processing step | 0.1 s | Detector/filter/governor/controller processing budget |
| Source command expiry | 0.1 simulation s | A result cannot authorise motion after its source packet expires |
| Warm-up | 2 fresh results | Existing confirmation hit requirement, not a new tuning threshold |

Capture owners supply `FrameBundle.capture_started_monotonic_s` from the same host
before requesting its world tick. This is a conservative lower bound on capture
time; it is never reset on dequeue. The RGB-D joiner now preserves actual sensor
receipt time separately. Frame IDs, episode, source/model and exact depth alignment
retain their existing checks. The simulation expiry clock is separate from host age.

Cold/over-budget results return HOLD and reset warm-up; they do not permanently
latch the ordinary cold-start case. Existing structural faults still latch.
The gate checks again after the control policy computes. Direct detector reset
changes its generation, invalidating readiness even within the same episode.
A ControlPacket carries an additional host expiry; missing host clock fails closed
when checking such a packet. FixedVehicleAdapter.apply_packet checks both expiry
clocks immediately before dispatch. A stale result therefore cannot sit in a
queue and later become a positive command while simulation time is paused.

Invalid/expired/missing-ID/depth observations do not become clearance. Partial
clearance history is invalidated without clearing the blocked latch. Startup
warming is distinct from an interrupted obstacle HOLD. Normal HOLD/resume after
a blocker still requires the existing fresh-clear hysteresis. Emergency overrides
the governor/controller result in the same cycle.

If later accepted, the baseline loader checks the governor budget, model asset hash, resolved
tracker settings, confidence/image size/class selection, confirmation/depth policy, timestep and
capture dimensions/intrinsics. The calibrated mount/source binding remains the
capture adapter's responsibility. Legacy navigation-only S0 and A1 demos are not
silently relabelled as the gated perception path.

## Live measurement and provenance

CARLA 0.9.16 / Town10HD_Opt, existing 640x360 / 90-degree RGB-D rig at actor mount
(1.5, 0, 2.0) m; matching OpenDRIVE and axial convention evidence. Local yolo26n
hash `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`,
BoT-SORT with the existing resolved settings, 640 inference size, confidence 0.25,
unchanged 2-of-3 confirmation, CUDA:0.

The final run retained 240 frames across two detector resets at existing
depth-calibration site 152. Ego actuators remained braked. Real perceived box/depth
points, with the ego front offset applied once, supplied a conservative straight-
scene shadow governor input. No target actor geometry or labels entered detection,
filtering, governance or control. This is control-decision timing, not closed-loop
driving or full route-footprint validation.

| Measurement | Result |
|---|---:|
| Fresh, valid positive decisions | 122 |
| Positive cold/injected-stale decisions | 0 |
| Warm capture-to-decision p95 | 0.1604129467 s |
| Warm capture-to-decision maximum | 0.1855280180 s |
| Warm sensor-receipt-to-decision p95 | 0.0708841830 s |
| First cold result, episode 0 | 2.3370834930 s, HOLD |
| First result after reset, episode 1 | 0.3044119390 s, HOLD |

All frames, including remaining perception HOLDs and two explicit stale injections,
are retained in [warm-operational.json](evidence/BL-D-06/warm-operational.json).
Settings were restored and no actors leaked. Each row records host boundaries for
sensor availability, bundle conversion, detector, filter, controller and remaining
governor/decision overhead. The contiguous non-overlapping partition sums exactly
to capture-to-decision time. Source hashes and raw source snapshots identify the
executed uncommitted runtime; the baseline loader was then added and tested to
select that same gated path and budget. [Measured source and raw timing archive](evidence/BL-D-06/warm-source.zip)
retains the exact executed code and timing ledger. Raw logs/snapshots also remain under
`runs/BL-D-06/warm-operational-v4/`.

[All-run index](evidence/BL-D-06/warm-run-index.json) retains the failed adapter
attempt and site-10 runs. Site 10 included another car with invalid validated-range
depth, so policy HOLDs there were not accepted as successful positive-governance
timing. No validity rule was relaxed. The [same-frame sensor probe](evidence/BL-D-06/sensor-probe.json)
confirmed the four-tick queue was unnecessary and retained its startup timing.

The 70.9 ms metric is receipt-to-decision, not completed actuator application.
The 160.4 ms metric includes sensor waiting before receipt. These are different
quantities; neither rewrites the existing 100 ms acceptance definition. Full
capture-to-actuator/closed-loop acceptance remains separate.

## Independent timing composition; unresolved operating range

The measured warm path timestamps **before the owner tick**, then includes sensor
availability/read, bundle conversion, detector, filter, governor/controller and
final decision availability in non-overlapping host-clock intervals. It never
uses the BL-C replay script or first-confirmation latency in that composition.
Cold/stale outputs HOLD. A failed read cannot produce a positive command.

The existing enforced budgets still give:

`response = 0.5 s capture age + 0.1 s post-decision delivery reserve`
`           + 0.3500000052154064 s measured vehicle lag`
`         = 0.9500000052154064 s`

Vehicle lag is counted once. No detector/filter/controller term is added twice.
The observed warm maximum plus vehicle lag remains **0.5355280232151927 s**.
These are timing inputs; they do not establish when a previously unseen obstacle
will first become detectable/confirmed. The freshness gate prevents old results
from driving but cannot manufacture visibility or timely first detection.

Conservative BL-E-03 deceleration remains **1.497566177392398 m/s²**. With the
unchanged upper stopping margin **6.0 m**, the candidate budget at **4.0 m/s** gives:

`6 + 4*0.9500000052154064 + 4²/(2*1.497566177392398)`
`= 15.142001009378749 m`

The earlier **6.2540012234 m/s** cap is only the conditional result of assuming
**25 m usable bumper visibility**. That assumption is not established. TEST-13
supports axial-depth accuracy through 25 m for the reviewed boxes; it does not
prove 25 m front-bumper clearance, full corridor coverage or first-confirmation
range. Camera/rear-axle/front-bumper geometry must be applied once to real
perceived measurements. The parked timing target does not supply approach-range
or closed-loop stopping evidence. Therefore **4.0 m/s cannot honestly be frozen
for obstacle operation from this integrated evidence**. No lower speed is invented
without the missing range/confirmation evidence.

`frozen_values` now retains only measured deceleration and the unchanged margin,
with cruise/response null. The conditional 4.0 m/s and 0.9500000052 s inputs remain
under `candidate_values`. Lateral acceleration **1.5 m/s²**, acceleration policy
**1.0 m/s²** and emergency margin **2.0 m** live only in `implementation_policy`;
they are not new measured calibration. Clearance hysteresis and the **2.5–6.0 m**
acceptance band remain unchanged. Evaluation truth never enters runtime control.

## Source provenance and retained history

The [evidence guide](evidence/BL-D-06/README.md) describes the curated artifact set
and the source archive's exact identity. `warm-source.zip` is **not** a release or
single Git revision: the measurement used an uncommitted source snapshot. It
contains twelve Python files, the raw 240-row timing ledger and the byte-identical
measurement report. Its SHA256 is
`6e6d9b7742b7d8624d1cc79bb88c2a9f5cabe55680b9f1853ad316b020d07b42`.
The report pins executed source files by SHA256 and records model, rig, dependency
and clock provenance. The review script verifies this archive before accepting
the timing ledger as historical measurement evidence.

Most executed freshness/adapter sources are not reproduced by the current
committed HEAD. The archived vehicle adapter also predates upstream's
acknowledged-control documentation/error detail. Subsequent main integrations
and the reviewer corridor commit do not retroactively validate that source
snapshot as the current integrated driving stack. The raw measured timings remain
valid for the executed parked-ego path; D-005's usable-range/freeze inference is
withdrawn by D-006.

[Source provenance](evidence/BL-D-06/provenance.json) pins each retained JSON input
to its committed source path/blob and SHA256. These snapshots keep the offline
review self-contained. Historical BL-C response/cap values are used only to
reproduce the explicitly superseded D-004 arithmetic. They never supply current
obstacle-range support or a frozen cruise value. Current BL-C status is recorded
in the [sensor-wait timing review](evidence/BL-C-04-06/sensor-wait-timing-review.json).

## Corridor boundary review

Zhaolin's commit `3851a0e8014cf3df25645bfe9258f0d27ed123a6` was preserved by
cherry-pick `61e2d1a40c4b65229621b2f18b052861e975974f`:
`fix: cover corridor turns and front overhang at route end`.
Round joins cover outer-turn sweep and final-tangent extension covers vehicle
front overhang. The obstacle 0.282843 m from a right-angle vertex intersects at
arc interval `(10, 10)`; the endpoint example returns **-1 m**. Regression tests
also cover rotations, both turn directions, tangencies and arc limits.
These geometry defects are resolved. The commit and retained regression report
provide the permanent evidence; temporary fetch/merge/probe transcripts are not
part of this evidence set. Geometry fixes do not establish missing BL-C range or
closed-loop stopping acceptance. D-006 remains BLOCKED.

## Reproduction and remaining acceptance

[Current machine-readable review](evidence/BL-D-06/review.json) reproduces D-006
while retaining the historical cold review and independently measured warm timing.
The [all-run index](evidence/BL-D-06/warm-run-index.json) preserves failed/held
attempts and explicitly distinguishes historical selection from current timing
selection. No run currently authorises an obstacle freeze.

CARLA was unavailable during the integration's recapture check. No fresh BL-C
sequence or new timing value was fabricated. First recapture the BL-C sequences
using the merged measured-wait instrumentation (commands in
[track stabilisation](track_stabilisation.md)), then review first-confirmation
range with camera-to-bumper geometry and perceived corridor coverage.

```bash
.venv/bin/python scripts/measure_operational_governance.py \
  --host 172.28.176.1 --spawn-index 152 \
  --output runs/BL-D-06/warm-operational-next
.venv/bin/python scripts/review_baseline_freeze.py \
  --output /tmp/bl-d-06-review-next.json
```

The [retained regression report](evidence/BL-D-06/corridor-fix-regression.xml)
covers the final focused BL-D/BL-C interaction suite, including the corridor fixes,
freshness, blocked baseline and evidence reproduction. Earlier 129-case XML
reports were superseded by this superset. Test success verifies the stated code
contracts, not a safety-threshold approval or substitute for the missing live data.
Human review, exact final baseline admission and physical stop/HOLD/resume
acceptance remain outstanding. The **4.0 m/s navigation cruise is unchanged**;
obstacle-operation support is unestablished.

```{toctree}
:maxdepth: 1

evidence/BL-D-06/README
evidence/BL-D-06/ai-journal
```
