# Same-frame operational approach — 21 October 2026

**BL-D-06 technical PASS for the bounded freshness-gated profile.** The local
candidate freeze retains **4.0 m/s**, response **0.9500000052154064 s**, conservative
deceleration **1.497566177392398 m/s²** and unchanged **6 m** upper margin.
This is **not** a team G1 admission, a driven 5.7 m/s certificate, a full-corridor
visibility claim or closed-loop stopping/HOLD/resume acceptance.

## Capture and provenance

[Capture](capture.json), [exact command/log](capture-command.json), [verified review](review.json), [provenance](provenance.json),
[measured source archive](measured-source.zip) and [raw artifact hashes](artifact-manifest.json)
retain all **800 frames**. The existing BL-C 600-frame run and rejection report
remain unchanged. Real CARLA client/server **0.9.16**, map
**Carla/Maps/Town10HD_Opt**, site **93**, OpenDRIVE SHA256
`5d883b799f634030af92be1e9d79d107845540ba04338e8c60e095be1aef7be7`.
RGB-D rig **640×360 / 90°**, actor mount **(1.5,0,2.0) m**, axial convention,
0.05 s exclusive synchronous owner. Existing convention/rig evidence is reused,
not recalibrated. Full weather, model/tracker settings and dependency versions
are in the capture. All actual source simulation ages at decision are **0.0 s**;
no four-tick source lag was used.

YOLO26n SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`,
confidence **0.25**, inference **640**, resolved BoT-SORT unchanged, device
**cuda:0**. The actual all-track policy, 25 m depth policy, 2-of-3 confirmation,
0.5 s capture-age / 0.1 s processing / 0.1 s source TTL and real PathFollower
are unchanged. Controller/provider/governor outputs are measured shadow outputs;
physical ego commands are always full brake.

Source snapshots and file hashes identify the executed collector, reused helpers,
configuration and runtime. The frozen candidate config differs from the captured
blocked profile only in freeze metadata and population of existing candidate
values; runtime source and numeric operating policy were not changed.

## Preflight and all attempts

[All-attempt index](attempt-index.json), [site-152 preflight](preflight-01-report.json)
and [four-site preflight](background-preflight-01-report.json) retain all unsuccessful and
successful checks. Site 152 had 0/40 admissible observations because an unrelated
car remained invalid. Empty-background checks: site 22 unsuitable; site 93 clean;
sites 118/147 also had valid supported background detections. Site 93 was selected
before final approach recording, not by selecting successful final frames.
No detection was hidden/filtered by target label. In the final run, a traffic
light was also tracked with valid depth and sometimes set the shorter conservative
provider distance. It remains in every relevant track/provider record.
The bus stays **behind the camera** outside its scheduled occlusion; off-axis
7 m placement from the old run is not reused.

The first 800-frame attempt is retained under `elevated-target-attempt/` but
**excluded from the final ordinary-road range conclusion**: its target used the
spawn-point height rather than road height. Two subsequent setup attempts failed
before sampled frames (ego spawn placement, then helper variable error); both
restored settings and leaked no actors. The final accepted run is **same-frame-04**,
with kinematic actor road reference **z=0.0 m** and **0.05 m** placement offset.
The ego still uses its normal spawn-and-settle path. A regression covers the
placement helper; no model/threshold/guard was changed to obtain the result.

## Results

| Sequence | Frames | Admissible observations | Matching-target admissions | Positive shadow packets | First target-admitted evaluation gap | First operational gap |
|---|---:|---:|---:|---:|---:|---:|
| approach | 200 | 106 | 77 | 8 | 23.808215 m | 22.350923 m |
| short_occlusion | 200 | 71 | 63 | 3 | 24.008215 m | 22.350923 m |
| continuity_risk | 200 | 62 | 52 | 7 | 23.808215 m | 22.350923 m |
| failure_recovery | 200 | 106 | 68 | 4 | 23.208215 m | 22.348600 m |

Total **345 admissible**, **260 matching-target admissions**, **22 positive
shadow packets**, **zero invalid/cold/stale positive packets**. All frames,
including early out-of-range/unconfirmed detections, occlusion, missing depth,
recovery and later invalid observations remain recorded. Admission does not
necessarily imply positive target speed: HOLD/emergency can still override.
Only **10** of the 22 positive packets also have a matching valid target
(0 / 0 / 7 / 3 by sequence). Other target admissions legitimately remain HOLD
under the existing interruption latch. No automatic clearance/resume is claimed.

The occlusions cover **3** and **20** frames. Failure/recovery has one explicitly
**injected dropout after an actual successful read**, not a claimed natural
CARLA timeout. It withholds the source depth from the operational input and
causes the ordinary missing-depth structural latch: full brake. Index 141 has an
explicit episode/generation reset and warm-up HOLD; admission and positive output
return at index **143**. No expiry/miss is called clearance. The provider always
records `clearance_valid=False`; reset recovery is not automatic road-clear resume.

Rejections (455 total): **444 perception HOLD**, **5 warm-up HOLD**,
**4 processing-budget HOLD**, **1 capture-age HOLD**, **1 missing-depth fault**.
The initial cold capture-to-output maximum **3.609765 s** is retained and HOLD;
it is not admitted operational latency. All stages/clock values, all tracks and
provider/packet validity are in each row.

## Non-overlapping operational timing

Admissible rows, milliseconds:

| Stage | p95 (ms) | Maximum (ms) |
|---|---:|---:|
| Capture origin → sensor availability | 22.2045 | 33.0132 |
| Bundle conversion | 1.0712 | 1.7634 |
| Detector/tracker | 23.0504 | 35.9091 |
| Temporal/depth filter | 0.7940 | 1.2679 |
| Geometry provider | 0.5042 | 0.8842 |
| Governor | 0.0211 | 0.2560 |
| Controller | 0.0911 | 0.2607 |
| Remaining overhead | 0.8091 | 1.3189 |
| Complete capture → output | 45.9488 | 60.9737 |

The stage partition matches the complete interval with zero numerical residual.
Individual percentile/max values are not summed as though they occurred together.
Receipt-to-output p95/max: **26.016 / 38.175 ms**. The distinct post-decision
brake-ack delay p95/max: **6.618 / 10.138 ms**, below the existing 0.1 s delivery
reserve. Receipt-to-acknowledged **brake** application p95/max: **31.999 / 45.216 ms**.
Positive commands were not physically applied; that distinction is retained.

The envelope uses the **enforced 0.5 s admission bound**, not just the observed
60.974 ms maximum, plus **0.1 s separate delivery reserve** and measured vehicle
lag once. It does not add detector/filter/controller delays twice. The old BL-C
4.199442 s backlog remains valid for that separate serial capture and cannot
produce positive commands on the bounded operational path.

## Operational range and candidate freeze

For each scenario select the first actual packet-admitted, matching confirmed
target-depth observation after runtime computations. Use the **minimum of its
perceived operational gap and evaluation target bumper gap**, then the minimum
across scenarios. Evaluation labels never enter runtime policy. The common
operational range is therefore **22.348599722608924 m**, rather than 25 m axial
range or the larger target-only gap.

```text
response = 0.5 + 0.1 + 0.3500000052154064 = 0.9500000052154064 s
required_gap(4) = 6 + 4*response + 16/(2*1.497566177392398)
                = 15.142001009378749 m < 22.348599722608924 m
maximum envelope speed = 5.7180586377280065 m/s
reported downward to 0.1 m/s = 5.7 m/s
```

**Highest calculated envelope cap: 5.7 m/s. Proposed frozen operating ceiling:
4.0 m/s**, the unchanged tested candidate. No evidence of driving at 5.7 m/s is
claimed. A higher configuration was not substituted for the measured candidate.
[Current BL-D review](../review.json) and [candidate config](../../../../configs/mission_baseline.json)
identify the exact settings. Only `baseline_operational_stack` with the bounded
profile may use these values; legacy navigation/demo entry points are not
silently relabelled. G2 admission still requires publication and human/team review.

This provider conservatively includes every valid perceived surface, performs no
outside-corridor exclusion and makes no empty-road coverage claim. The result
supports a controlled straight-scene first-admission/stopping-envelope candidate,
not arbitrary bends, unseen obstacles or physical stop-margin certification.
Existing corridor unit tests remain separate geometry evidence. S1–S5 driving
acceptance and general operating-domain validation are not fabricated here.

## Reproduction and validation

Raw arrays/source rows remain under
`/home/alan/26-S2-Assignment2-Group7/runs/BL-D-06/operational-approach/same-frame-04`
(about 1.3 GB). All hashes are versioned; preserve durable team access with review.

```bash
PYTHONPATH=src .venv/bin/python -m scripts.review_operational_range \
  --input runs/BL-D-06/operational-approach/same-frame-04/report.json \
  --output /tmp/operational-approach-reviewed.json
PYTHONPATH=src .venv/bin/python scripts/review_baseline_freeze.py \
  --output /tmp/bl-d-freeze-reviewed.json
```

The operational reviewer verifies all raw artifact hashes, source snapshots,
actual clocks/TTL/partitions, all-track validity, provider and real controller
replay (zero mismatches), startup/fault recovery and the speed equation. Offline
unit tests verify corrupt clock/source/expiry/positive-output rejection. Final validation: **170 focused tests passed**, **830 full tests passed,
1 skipped, 3 subtests passed**. Ruff, strict Sphinx `-W`, `git diff --check`,
exact BL-D review reproduction and config/source hashes passed. [JUnit](regression.xml).
