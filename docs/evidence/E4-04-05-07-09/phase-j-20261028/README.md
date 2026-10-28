# Issue #57 protocol/session follow-up

AI-assisted with OpenAI Codex, 28 October 2026. Existing PR #105 is updated,
not replaced. Run/session echoes, UUID reset isolation, client-clock metadata,
optional env bearer credentials and worker-only best-effort old-session cleanup
support the separate official #56 service. The first request explicitly starts
a session; expired/failed service state requires explicit reset. Neither a
backend reset nor track-ID reuse supplies positive clearance.

Twenty-seven client/fallback/replay tests pass. Ruff, boundary smoke (88 core
modules) and strict Sphinx pass. Full integrated regression: 945 passed,
4 failed, 1 CARLA deselected, 3 subtests. All failures are G1 source-inventory
mismatches; trusted admission remains unchanged. Exact review proposal and log
are adjacent. Service/container integration evidence belongs to #56's branch.

Still incomplete: reviewed source binding, explicit control-runner integration,
>=200 comparable labelled real-model replay frames per local/LAN configuration,
latency/accuracy table and available-cloud evidence (otherwise NOT_RUN). No
synthetic/loopback timing is reported as LAN or detector-accuracy evidence.
