# E1-01/02 offline preparation

AI-assisted with OpenAI Codex, 28 October 2026. Twelve targeted tests, Ruff,
simulator boundary (84 core modules), strict Sphinx and matrix generation pass.
Full suite: 930 passed, 4 failed, 1 CARLA deselected, 3 subtests. Four failures
are exact-source admission mismatches; the trusted record remains unchanged.
The adjacent proposal is for designated review, not operational admission.

Every one of the 27 planned cells remains NOT_RUN. No dataset, installed CARLA
verification, weather sensor-effect or robustness result is claimed. See
`docs/weather_experiment.md` for preflight commands and capture prerequisites.

## Continuation: preparation scope ready

Canonical PR100 main merged before rebinding. Complete production diff reviewed
under explicit owner delegation: only the optional read-only weather adapter
and NOT_RUN experiment config are new inventory entries. Frozen mission bytes,
values/scope, model/sensor settings and historical #41 runner binding are unchanged.
Applied source-only rebind is recorded in `applied-rebind-review.json`; no live
weather, pedestrian operation or robustness is admitted.

Production admission and 19 negative probes pass. Full offline suite: **984 passed,
1 CARLA deselected, 3 subtests passed**; 12 weather tests, Ruff, boundary smoke
(90 core modules), strict Sphinx and diff checks pass. All 27 cells stay NOT_RUN.
This completes the explicitly preparation-only PR scope; #49 remains open for
installed verification, capture/labels and sensor-effect evidence.
