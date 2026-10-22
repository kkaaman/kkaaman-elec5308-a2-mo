# Decision log

Record decisions that change scope, interfaces, thresholds, ownership or
process. Add new entries at the bottom.

A threshold change after gate G1 needs:
- a dated entry,
- the justification, and
- a rerun of the affected evidence.

**Status values:** Proposed, Agreed (record who agreed), Superseded (link the
replacing entry).

| ID | Date | Decision | Rationale | Status | Revisit |
|---|---|---|---|---|---|
| D-001 | 2026-10-13 | The A2 repository starts from A1 `main` (`a1f3112`) with full history, tagged `a1-final` | Preserves authorship and keeps the A1 tests as a regression baseline | Proposed (PROC-01 PR) | G0 review |
| D-002 | 2026-10-13 | `main` changes only through reviewed PRs with green CI, one task per PR, merged with a merge commit | Reviewable history and correct attribution for peer assessment | Proposed (PROC-01 PR) | G0 review |
| D-003 | 2026-10-16 | Keep baseline speed and stopping thresholds provisional until G1 calibration evidence is reviewed | BL-D-06 depends on measured depth accuracy (BL-B-04), detector range/latency (BL-C-05) and longitudinal response (BL-E-03). The 5.0 m/s cruise candidate, 25 m detection target, 2.5–6.0 m stop-margin band and 100 ms p95 latency target are acceptance constraints, not measured frozen parameters. If the measured detection/stopping envelope cannot safely support 5.0 m/s, the cruise speed must be reduced rather than weakening acceptance criteria. | Proposed (BL-D-06) | G1 review |

| D-004 | 2026-10-20 | BL-D-06 measured review blocks final threshold freeze; current navigation candidate is 4.0 m/s | Historical D-004 arithmetic used the then-reported BL-C response and gave 31.9508 m at 4.0 m/s with a 6 m margin. The omitted sensor waits make that timing a superseded lower-bound diagnostic, not current evidence. No acceptance criterion is weakened. See [the measured review](baseline_threshold_freeze.md). | Historical blocker; timing superseded by D-006 | Fresh timing/range evidence and reviewer approval |

| D-005 | 2026-10-20 | Freeze the 4.0 m/s ceiling only for the bounded, freshness-gated operational profile | Same-frame YOLO/filter/governor/PathFollower run retained 240 samples and 122 valid positive decisions; cold/stale positives zero. Enforced 0.5 s capture age + separate 0.1 s delivery reserve + measured 0.3500000052 s vehicle lag gives 0.9500000052 s response and 15.142 m required gap at 4 m/s, below 25 m. Existing 2.5–6 m acceptance band is unchanged. The assumed usable 25 m range was not established. See [historical D-005 evidence](evidence/BL-D-06/history/D005-decision.txt). | Historical proposal; superseded by D-006 | Timing retained; usable range/BL-C timing re-review required |

| D-006 | 2026-10-20 | Block BL-D-06 after integrating corrected BL-C timing | The independent 0.5 s age + 0.1 s delivery + measured vehicle-lag budget remains justified, but historical BL-C first-confirmation latency/4.9 m/s cap are invalid current evidence. TEST-13's 25 m axial accuracy is not validated 25 m bumper visibility or first-confirmation range. Retain 4.0 m/s as navigation candidate; do not freeze obstacle operation. D-004/D-005 are historical records. See [integration re-review](baseline_threshold_freeze.md). | Proposed; freeze blocked | Fresh measured-wait BL-C capture, usable-range review and integrated acceptance |

| D-007 | 2026-10-21 | Fresh BL-C capture resolves the timing-data gap; D-006 remains technically blocked on operational range admission | BL-C-05 passes its evidence scope and records a 3.2 m/s conservative target-only recommendation. Cold 4.199442 s replay age cannot authorise new positive output under current gating; 0.9500000052 s operational candidate is independently justified. But all 600 approach-source rows contain invalid current tracks under the actual operational policy, even granting fresh/warm status; no full governor observation is admitted. Lower cruise is allowed by D-003 but cannot fix invalid visibility. See [fresh re-review](baseline_threshold_freeze.md). | Proposed; freeze blocked on joint range/admission evidence | Valid perceived first-admission range on the bounded operational path, then exact configuration and team review |

| D-008 | 2026-10-21 | Technical freeze candidate: retain 4.0 m/s only as a bounded freshness-gated stationary-vehicle candidate | Stationary-vehicle-only replacement same-frame approach has 800 road-height frames, 345 admissible observations, 22 positive shadow packets and zero invalid/cold/stale positives. Common actual first-admission operational gap 22.348600 m exceeds the 15.142001 m required by the unchanged 0.9500000052 s response / BL-E deceleration / 6 m margin. Calculated envelope cap 5.7 m/s is not a driven-speed certificate. All-track guard, thresholds and historical rejection evidence are preserved. See [current review](baseline_threshold_freeze.md). | Published technical PASS candidate; G1/team admission pending | Exact candidate source/config publication, durable evidence access and full-team G1 admission; closed-loop acceptance remains separate |

## Open questions for the teaching team (PROC-03)

Record each request and response with its date. Don't assume approval.

- Are map topology and the simulator ego pose acceptable as navigation inputs?
  Obstacle ground truth would stay evaluation-only.
- Is the extension list E1–E5 acceptable, given that the reduced baseline is
  delivered first?
- What are the report deadline, the video deadline and the Week 13 lab slot?
