# Post-freeze offline continuation — 1 November 2026

The annotated tag release/presentation-20261031 still resolves to
9a283d41b19c3339aa5762468a0968b421a524d5. Its worktree and release artifacts
remain unchanged. New work uses codex/post-freeze-20261101 and a separate PR101
continuation. CARLA remained OFF.

## Reuse and priority deltas

Inspected current main, merged #105/#109/#111/#112, open #99/#101/#116, PR69
5942fab and Mohammad's standalone implementation before coding. The standalone
segmentation evaluator has alternative contracts; canonical dataset/metrics
were reused. PR69's recorded perception evaluator was adapted with attribution,
not merged wholesale.

| Issue | Completed offline | Exact remaining criterion |
|---|---|---|
| #43 | PR69 replay adapted to TrackDepthFilter; held-out road/run/image validation; class TP/FP/FN, invalid depth, first-confirmed distance, flicker/ID-reset records. Existing 600-frame failures retained. | Fresh independently held-out labelled scenes. Existing sequences were already used in development and cannot become held-out tests. No detector-quality or semantic-correction PASS. |
| #57 | Canonical nonblocking fallback retained; wrong-source defect fixed. RGB-file selection and prediction retention permit later label scoring without repeating inference. 200 real 640x360 frames after 5 warm-ups: p50 14.422 ms, p95 28.391 ms, max 35.652 ms, 0/200 deadline misses. | Actual available LAN deployment and identical held-out labels; cloud NOT RUN. One-machine detector-call timing is not LAN, accuracy, live timing or admission. |
| #61 | Conditional P2 cut documented below. | None for the permitted cut outcome; reverse stays future work. |
| #63 | 216 new remote/hybrid cases pass alongside 216 existing local/delayed cases. Real bounded workers, hybrid adapter and shared stack; synthetic sensors/backends. | Selected admitted all-flags release still depends on #48/#58; no all-flags driving certification. |
| #101/#42/#45 | Explicit frame mismatch invalidates lane diagnostics. Pristine held-out mask scoring reuses dataset leakage/hash checks and existing metrics. Workload/deadline instrumentation reused. | Pristine labels, lane-divider validation, full-stack S3/S4/S5 and applicable baseline acceptance. #40 does not block label evaluation. |

### Fault defect and retained evidence

Before correction, 12/216 added cases failed: source metadata was overwritten
before validation. Wrong-source output could become a policy command or advance
fixture clearance. The adapter now validates the configured worker source (or
pins the first source for generic backends) before namespacing. Mismatch uses
the existing latched full-brake path. Model/session/frame and HOLD rules remain.

[All before/after traces](evidence/post-freeze-20261101/hybrid-faults.zip),
[summary](evidence/post-freeze-20261101/fault-summary.json),
[local replay](evidence/post-freeze-20261101/local-replay/comparison.md).
Original failures remain visible. Fixture workers do not establish HTTP
performance. Source-only admission rebind preserves mission/model/sensor and
historical runner identities; it adds no live acceptance.

## P2 reverse-parking cut — #61

E5-04/05/08/09 are conditional P2 work. The user authorised finishing or recording
the cut. Baseline #40/#41 and live P1 #60 remain unaccepted, so the prerequisite
for spending reverse-prototype/pad/regression budget is false. P2 is **CUT /
FUTURE WORK**, disabled in the frozen release. Forward-only A1 and P1 stay intact.
The existing [eight-case P1 harness record](evidence/E5-P1/README.md) supports the
scope decision, not reverse acceptance.

No reverse prototype, physical sign/brake/gear-interlock test, 5/5 tolerance pass
or P2-enabled baseline regression is claimed. Conditional tasks are superseded
by the permitted cut, not passed. Reopening P2 requires signed-speed/gear-safe
contracts, stopped gear changes, isolated pad qualification, 5/5 <=0.3 m /
<=5 degrees / zero collisions, and baseline regression.

## Canonical review and individual responsibilities

The official Assignment 2 brief's team/individual deliverables and AI-use rules
require disclosure, confidential journals/reflections, every member at the live
demonstration and able to answer any module, lab peer reflection and SparkPLUS.
It does not mandate unanimous freeze signatures, named weekly review pairings
or six duplicate evidence documents. Existing G1 admission already uses a
designated owner/reviewer rather than unanimous acknowledgement.

For #78/#79, internally invented weekly pair/attendance checklists are superseded
by canonical implementation/evidence review, recorded decisions, Git history
and existing cross-module Q&A material. Human review was reported by the project
owner for this workflow; no named/per-member approval is invented. Actual
demonstration attendance/knowledge remains #75/#76; individual duties stay
#80–#85. #64 uses one evidence-based owner readiness checklist; missing report,
video and actual submission checks remain mandatory.

[Central contribution links](contribution_evidence.md) prepare repository-side
evidence once while preserving attribution. Private contents stay private.

## Reproduction

Use the existing environment with PYTHONPATH=src:. and fresh output paths.

```bash
python -m scripts.hybrid_fault_suite --output runs/TEST-18/hybrid-new
python -m scripts.benchmark_perception --dataset runs/heldout --depth-calibration docs/evidence/BL-C-04-06/depth-calibration.json --output runs/TEST-09/heldout-new
python -m scripts.score_detector_replay --replay runs/E4/local-new --annotations heldout-annotations.json --output runs/E4/accuracy-new.json
```

The existing benchmark_remote_detector command now accepts --frame-pattern
'*-rgb.npy' and scripts.replay_local_detector:make_detector. Accuracy annotations
contain model_version, tuning_scene_ids, tuning_image_sha256 and frames. Each
filename maps to scene_id, split=test, original image sha256 and explicit labels
(class_name/xyxy), including empty lists. Every input is required. Unknown tuning
provenance cannot be treated as an empty tuning history.

```{toctree}
:hidden:

evidence/post-freeze-20261101/local-replay/comparison
```

## Shared parked capture queue for #43/#49/#50/#101

The existing collector's training presets were not the approved weather matrix.
The collector now accepts a selected canonical cell, preserves paired scene
seeds, applies all explicit weather fields/lights and verifies actual readbacks.
A plan-only wrapper enumerates all 27 cells without importing CARLA. Each cell
captures stationary S0 (no inserted target), S1 (car) or S2 (person) scenes.
These are **not** 27 driving acceptance runs for #51.

One combined index references original RGB/instance/depth/lane assets. The
existing dataset validator enforces road/run/image splits and hashes across
cells. Paired weather views share a pose identity and cannot inflate independent
range evidence. Held-out replay feeds the existing weather evaluator; missing
depth remains missing, contributes an explicit invalid-depth count, and cannot
produce a fabricated zero range or COMPLETE result without range evidence.

The same original PNGs feed --dataset mode in the latency sweep and PR101's
pristine-mask evaluator. No image copies or repeated live captures are required.
LiDAR weather effects remain NOT MEASURED; settings alone establish no effect.

## Final offline validation

Production state 60e8611, followed by source-only admission/documentation:
1557 pytest passed, 1 live test skipped, 3 subtests passed; one dependency
deprecation warning. Admission and all 19 negative probes pass. Ruff,
AST/import boundary (110 core imports), generated module inventory, model
asset identity and diff checks pass. The earlier general-environment run lacked
FastAPI (1548 passed / two dependency failures); retained service rerun passed
18/18, and the final full run above used the service-capable environment.

The service/deployment/wire protocol sources are unchanged, so existing real-model
Docker evidence is reused. The changed hybrid adapter is covered by the new
216-case matrix and existing client/server/control tests. No cloud or Docker
performance measurement is invented. See [remaining work and live queue](release_live_queue.md).
