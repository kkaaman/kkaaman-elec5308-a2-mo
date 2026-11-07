# BL-D-06 evidence guide

**Latest technical decision: PASS / published stationary-vehicle technical candidate; frozen loader disabled pending G1 admission.**
The new [same-frame operational approach](operational-approach/README.md) supplies
345 admitted observations and supports the unchanged 4.0 m/s candidate under the
0.9500000052 s enforced budget. Human review/publication/G1 admission remain pending.
The earlier guide and rejection investigation below remain historical for their
original datasets and are not overwritten by the new capture.

**Previous decision: D-006 BLOCKED_OPERATIONAL_RANGE_ADMISSION (superseded).**
Fresh BL-C capture is now complete; [current decision](../../baseline_threshold_freeze.md)
and [all-track replay](operational-range-review.json) supersede the missing-capture
rationale below. Even optimistically bypassing timing/warm-up, all 600 source rows
remain HOLD because other current tracks are invalid. A confirmed target-only gap
is not an admitted full obstacle/visibility observation. Neither 4.0 nor 3.2 m/s
is frozen. [Fresh review regressions](fresh-range-regression.xml) cover this update.

The rest of this guide records the **pre-refresh historical interpretation** of
the retained source/timing bundle; unchanged raw artifacts remain authoritative
for their stated measurement scope, not current admission. Current review.json
now consumes the fresh capture and the operational-range audit.

**Prior missing-capture rationale (superseded):**
No fresh sensor-wait-aware BL-C capture establishes first-confirmation/usable
obstacle range. TEST-13 accuracy through 25 m axial depth is not proof of 25 m
bumper visibility. **4.0 m/s obstacle-operation support is unestablished.** The
navigation cruise remains 4.0 m/s; acceptance criteria are unchanged.

Fresh-range follow-up validation: **158 focused tests passed**; **818 full-suite tests passed, 1 skipped, 3 subtests passed**. Ruff, Sphinx `-W`, local decision-document links, numeric/boolean/null configuration preservation, unchanged acceptance bytes, exact review reproduction and `git diff --check` passed. No simulator recapture was performed.

[Detailed rejection investigation](admission-rejection-investigation.md) classifies
every predicate, overlapping frame combinations, background tracks and missing
operational timing fields. It specifies the minimal replacement measurement.

## Current and historical authority

- `review.json`: current machine-readable D-006 review. Cruise/response frozen
  values are null, candidate support is false and no cruise is recommended.
  `screening.speed_cap_mps` (6.254...) is explicitly **conditional diagnostic
  arithmetic**, assuming unvalidated 25 m visibility; it is not a measured or
  admitted operating cap. The separately labelled historical cold review retains
  the old BL-C 4.9 m/s result only to reproduce D-004's superseded reasoning.
- `cold-review.json`: historical D-004 lower-bound calculation from the old BL-C
  timing methodology. It is not current timing evidence.
- `history/D005-decision.txt` and `history/D005-review.json`: original historical
  D-005 proposal, preserved byte-for-byte. D-006 withdraws its usable-range/freeze
  inference. Former links in the plain-text snapshot are historical context,
  not maintained navigation to current evidence.
- `warm-operational.json`: unchanged measured parked-ego timing, **not a cruise
  freeze or closed-loop stopping certificate**. 240 frames, two resets, 122 valid
  positive shadow decisions, zero cold/injected-stale positives. Warm capture-to-
  decision p95/max are 0.1604129467/0.1855280180 s. Raw timing remains valid for
  the source snapshot actually run; the earlier 25 m obstacle-range inference
  is not established. D-006 supersedes that inference.
- The independent **0.9500000052 s** candidate is the enforced 0.5 s capture-age
  bound + 0.1 s post-decision reserve + measured vehicle lag once. It is not an
  observed end-to-end sample or proof that an obstacle is first confirmed in time.

## Exact executed source snapshot

`warm-source.zip` identifies the executed **uncommitted** `warm-operational-v4`
source snapshot; no single committed Git revision is recorded for the whole run.
Archive SHA256:

`6e6d9b7742b7d8624d1cc79bb88c2a9f5cabe55680b9f1853ad316b020d07b42`

The archive has 14 members: twelve Python source files, `timing.jsonl` and
`report.json`. Its report is byte-identical to `warm-operational.json` (SHA256
`08e8a563b42a1f61ed35f8a8de7975d8fc9e4fbeb405c1c3d693b855da3cac57`), and all
240 JSONL rows equal the report's rows. `source_hashes` pins eleven runtime files;
`provenance["scripts/measure_operational_governance.py"]` pins the collector.
The eleven runtime hashes and collector hash match the archived bytes.

Only the tracking filter and speed governor snapshots are reproducible from
normal branch/tag history. The complete executed snapshot is not: freshness,
stack, adapters and collector included uncommitted changes. Local Codex backups
or stash objects are not durable team provenance. The archive therefore provides
unique reproducibility and is required by `scripts/review_baseline_freeze.py`.
It is retained because of that dependency and unique source identity, not its size.

The snapshot is historical. Its vehicle adapter predates later upstream
acknowledged-control documentation/error details, and it predates the reviewer
corridor fix `61e2d1a40c4b65229621b2f18b052861e975974f`. That fix changes geometry
outside the parked timing collector's path; it does not recertify the archive as
the current integrated obstacle-driving stack. Subsequent integration/loader
changes must not inherit an unmeasured operational PASS from these timings.

## Retained artifact purposes

| Artifact | Purpose |
|---|---|
| `review.json` | Current blocked review, independent timing composition and explicitly conditional diagnostics |
| `corridor-fix-regression.xml` | Final focused BL-D/BL-C regression evidence, including reviewer boundary cases |
| `warm-operational.json` | Raw historical live timing report and exact source/settings provenance |
| `warm-source.zip` | Unique executed source snapshot and raw timing ledger |
| `warm-run-index.json` | All attempt summaries/hashes, including failed/held runs; historical selection is distinct from current timing selection and no run supports a current freeze |
| `sensor-probe.json` | Distinct 12-frame same-frame RGB-D feasibility/cleanup measurement; not a response or range certificate |
| `provenance.json` | Source commit/path/blob/SHA256 for each retained review-input snapshot |
| `cold-review.json` | Historical D-004 calculation |
| `history/D005-decision.txt` | Verbatim superseded decision |
| `history/D005-review.json` | Verbatim superseded machine-readable review |
| `sources/benchmark.json` | Superseded BL-C benchmark used only for historical arithmetic/settings provenance |
| `sources/capture-summary.json` | Historical benchmark identity, capture/cleanup and model/rig provenance |
| `sources/depth-calibration.json` | Measured installed-build axial convention and rig input |
| `sources/depth-analysis.json` | Reviewed axial-depth accuracy scope; not confirmed obstacle/bumper visibility |
| `sources/longitudinal.json` | Measured vehicle deceleration/response input |
| `sources/acceptance.json` | Unchanged stopping/latency acceptance snapshot |
| `sources/navigation-baseline.json` | Navigation candidate/configuration snapshot for historical review |
| `sources/controller-timing.json` | Separate historical navigation timing provenance, not integrated perception timing |
| `ai-journal.md` | AI disclosure, chronological decision history and limits of automated validation |

The eight source JSON snapshots match their committed originals and recorded
hashes. Retaining this small pinned input bundle lets the offline reviewer run
without Git/network retrieval. Test-report success is code-validation evidence;
human decision review and missing live evidence remain separate.

## Reproduction and cleanup policy

```bash
PYTHONPATH=src .venv/bin/python -m scripts.review_baseline_freeze --output /tmp/bl-d-review.json
.venv/bin/python -m pytest \
  tests/unit/test_baseline_freeze_review.py \
  tests/unit/test_operational_freshness.py \
  tests/unit/test_corridor.py tests/unit/test_behaviour.py \
  tests/unit/test_speed_governor.py tests/unit/test_obstacle_tracking.py \
  tests/unit/test_track_capture.py tests/unit/test_track_benchmark.py \
  tests/unit/test_rgbd_capture.py \
  --junitxml=docs/evidence/BL-D-06/corridor-fix-regression.xml
```

The final XML case set supersedes both former 129-case reports; each earlier case
is present in the final set. PR-description drafts, generated file lists,
repository-management audits and synthetic intermediate bug probes are omitted.
Authorship, changes and integration are traceable in Git and the journal; the
reviewer's final boundary regressions retain the permanent code-level evidence.
The raw timing, archive and historical decisions have not been rewritten.

Human D-006 review remains pending. No AI-produced review or passing test suite
independently approves safety thresholds, G1 admission or closed-loop stopping.

Final hygiene validation: **154 focused tests passed**; **814 full-suite tests
passed**, 1 skipped, 3 subtests passed. Ruff, Sphinx `-W`, whitespace checks and
byte-identical review reproduction passed. Runtime/script/test files, effective
configuration and navigation, Git HEAD/index, and raw/historical evidence were
verified unchanged. Automated validation is not human safety approval.

```{toctree}
:maxdepth: 1

admission-rejection-investigation
operational-approach/README
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
