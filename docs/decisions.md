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

## Open questions for the teaching team (PROC-03)

Record each request and response with its date. Don't assume approval.

- Are map topology and the simulator ego pose acceptable as navigation inputs?
  Obstacle ground truth would stay evaluation-only.
- Is the extension list E1–E5 acceptable, given that the reduced baseline is
  delivered first?
- What are the report deadline, the video deadline and the Week 13 lab slot?
