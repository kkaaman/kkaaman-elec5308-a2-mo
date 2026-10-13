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
| D-003 | 2026-10-13 | Forward camera candidate: 1280×720, 90° FOV, detector input 1280 (`PerceptionConfig.image_size`) | At 800×450 no setting detected a pedestrian at 30 m. At 1280×720/1280 both models did, with the detector unchanged at about 26 ms. Tick p95 26.7 ms vs 22.5 ms (`experiments/detector_seg.md`, `machines.md`) | Proposed: needs Ricky's agreement (detector owner) | BL-B-02, TEST-17 full-stack timing |
| D-004 | 2026-10-13 | Keep `yolo26n.pt` as the detector; do not switch to `yolo26n-seg.pt` | The seg model never detected more, lost pedestrians at 20–25 m at the default threshold, and cost 4–7 ms more. Masks improved distance only by centimetres (`experiments/detector_seg.md`) | Proposed | BL-C-05 range benchmark |

## Open questions for the teaching team (PROC-03)

Record each request and response with its date. Don't assume approval.

- Are map topology and the simulator ego pose acceptable as navigation inputs?
  Obstacle ground truth would stay evaluation-only.
- Is the extension list E1–E5 acceptable, given that the reduced baseline is
  delivered first?
- What are the report deadline, the video deadline and the Week 13 lab slot?
