# Phase J: PR #100 canonical-main integration check

AI-assisted with OpenAI Codex, 28 October 2026. Original contributor commits
are preserved; canonical main was merged normally into the PR branch.

Original PR head `9c4f3ac`: 882 passed, 1 CARLA test deselected, 3 subtests passed;
Ruff, boundary smoke (86 modules), strict Sphinx, eight P1 cases and five S0
trials passed. These are fresh offline runs, not live parking evidence.

After merging canonical `ae6a83d`: 966 passed, 6 failed, 1 CARLA test deselected,
3 subtests passed. Every failure is an exact-source G1 admission mismatch. The
production validator and trusted admission record remain unchanged. The adjacent
proposal records the exact candidate inventory/delta for designated review; it
does not grant admission. Issue #59 and PR #100 remain open pending that review
and successful final integrated checks. Live parking qualification belongs to
#60 and is not established by harness results.

## Independent local rebind simulation — 28 October 2026

The existing proposal was independently checked against the exact `210d81e`
integration tree. Its complete source inventory and combined scope fingerprint
match: seven additions, one changed file, no removals. Mission content matches
canonical main after normalizing the archive export's line endings.

With the trusted record unchanged, focused G1/freeze/freshness tests reproduced
the same six failures: **51 passed, 6 failed**. In an isolated snapshot only,
substituting `source_variants.canonical_loader_only` with the proposed variant
produced **972 passed, 1 CARLA test deselected, 3 subtests passed**. This includes
the negative source/configuration drift guards and all eight parking scenarios.
No validator, tests, mission values or issue-41 runner binding were changed.

See [machine-readable local validation](local-rebind-validation.json),
[control-case failure log](local-control-pytest.log),
[candidate simulation log](local-candidate-pytest.log) and
[candidate simulation JUnit](local-candidate-regression.xml).

Ruff and the simulator boundary smoke passed for the integrated snapshot. The
first strict Sphinx build exposed this report's missing toctree entry; the
documentation fix adds that entry, and the strict build then passed. This simulation is technical evidence for
designated review, **not** an approved production rebind or a successful remote
CI run. The production admission record remains unchanged and retains its
source-inventory blocker until the designated review/publication step.

## Continuation: applied reviewed rebind and final validation

The explicit continuation request delegates production-diff review and supported
source rebinding. Codex reviewed the entire production delta and applied only the
canonical-loader variant; no new human sign-off is invented. Existing PR #100
feature approval is retained. Seven opt-in P1 inputs are added; the only existing
source change is a planning package docstring. Baseline runtime/config, values,
model/sensor identity, confirmation/freshness/all-track contracts, pedestrian
deferral and historical first-stop runner binding are unchanged. P1 live parking
is not admitted. The prior trusted record is archived byte-for-byte.

Production admission passes; all 19 independent source/config/model/sensor/scope
negative probes reject, and all existing admission tests pass. Full offline
suite: **972 passed, 1 CARLA deselected, 3 subtests passed**. Fresh P1 **8/8** and
S0 **5/5** pass. Ruff, boundary smoke (90 core modules), strict Sphinx and diff
checks pass. Adjacent logs, review identities and verified acceptance archive
retain the exact evidence. This supersedes the earlier pending-rebind blocker.
