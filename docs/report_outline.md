# Report outline — DOC-01

Status: refreshed 20 October 2026; working outline. Merged navigation, braking
and depth results exist; final obstacle acceptance and review remain open. The
four topics below follow the technical-report deliverable in `plan.md`. Confirm
these against the final assignment brief before submission. This file is an
outline, not the submitted report. Keep module explanations in the API docs.

## Submission budget

Hard ceiling: **1,000 words and three pages**, including captions unless the brief
explicitly excludes them. Allocate 900 words and retain 100 words for captions
and references. Page allocations are planning limits: export the final report and
check its actual page count; this Markdown file cannot establish pagination.

| Page | Section | Word ceiling | Figure/table reservation |
|---|---|---:|---|
| 1 | System architecture | 220 | F1: sensor/core/adapter dataflow |
| 1 | Key design decisions | 220 | No additional figure |
| 2 | Evaluation results | 320 | T1: baseline/acceptance results; F2: measured trace |
| 3 | Limitations | 140 | Failure example only if measured and within space |
| All | Captions and references | 100 | Cite evidence IDs and decision IDs |

## 1. System architecture — 220 words

Explain the sensor-to-command path and the simulator boundary, citing
[shared interfaces](interfaces.md) and [module notes](module_notes.rst).
F1 is a placeholder: produce it from the reviewed implementation, label clocks,
coordinate conversion and evaluation-only truth. Link evidence rows A1–A2 in the
[evidence index](evidence_index.md); do not depict an unimplemented extension as
part of the measured baseline.

## 2. Key design decisions — 220 words

Choose the decisions that explain the design: physical units and rear-axle
reference, one world/tick owner, packet freshness, and offline model identity.
Cite the corresponding implementation/task record and
[decision register](decisions.md). D001/D002 are process records; check their
recorded approval status before calling them agreed architecture decisions.
Use E4 only when E4-07 supplies evidence. Map claims to rows D1–D2.

## 3. Evaluation results — 320 words

T1 columns: scenario/run ID, source revision, configuration/model identity,
metric and definition, measured value, acceptance threshold, status, raw evidence.
Start with TEST-01 as the inherited baseline with its recorded failures (R1).
Use accepted TEST-02 five-run navigation, TEST-03 braking/comparison and TEST-13
depth measurements as explicitly scoped existing results (R2). F2 may use their
retained traces; it is not globally NOT RUN. Distinguish nearest-lane and rear-axle
route CTE, different routes and timing intervals before comparing.

Reserve final integrated obstacle stop/HOLD/resume results as **NOT ESTABLISHED /
awaiting BL-F-07** (R3). BL-C's old confirmation latencies and 4.9 m/s cap are stale;
fresh measured-wait capture is required. D-006 remains BLOCKED, so 4.0 m/s obstacle
operation is not established; 25 m depth accuracy is not confirmed bumper visibility.
Leave E4 results **NOT RUN / awaiting E4-07** (R4). Unit tests/model sample inference
are not driving performance, and targets or screenshots must not fill missing data.

## 4. Limitations — 140 words

Use recorded failure modes, measurement scope and the dependency/review status
(rows L1–L2). Include the baseline junction timeout/lane misses where supported;
state any simulator, depth or GPU constraints that remain unverified. Generalise
only as far as the measured scenarios support. Final wording must reflect open RV-07 demo/endpoint documentation and RV-08
sensor/segmentation obligations, their partial merged foundations, and full-team
G1/final-report review. These task IDs are not generic peer-review approvals.

## Before submission

Resolve every pending evidence row, obtain reviews, verify archive access and
checksums, and check references against the final revision. Remove unsupported
claims rather than concealing missing runs. Count the complete report's words
and render it to confirm no more than three pages. The outline and evidence index
are supporting documentation and must not be pasted wholesale into the report.
