# E4-01/02/03/06 service evidence, 28 October 2026

AI-assisted with OpenAI Codex. No simulator was started. The adjacent image build
log, token-free container provenance, server log and black-image real-model smoke
are actual local executions. Networking was disabled, root filesystem read-only,
weights mounted read-only, memory capped at 4 GiB, CPUs at two, and PIDs at 128.
The two-client, duplicate-rejection and repeated-episode smoke passed.

Black images contain no labelled target: this is not an accuracy result,
200-frame deployment benchmark, actual LAN/cloud measurement or CARLA evidence.
Fake tracking/HTTP fixtures separately test isolation and faults. Full regression
and exact source-admission proposal are recorded alongside this note before
publication. The trusted G1 record remains unchanged; designated review/rebind is
required. Server code is on #56's focused branch, dependent on #105's existing
client implementation. Original contributor commits are preserved.

Final validation: 45 combined focused server/client tests pass (18 server,
27 existing-client cases), Ruff and simulator boundary pass (89 core modules),
and strict Sphinx passes. Full integrated suite: 961 passed, 6 failed,
1 CARLA deselected, 1 dependency deprecation warning, 3 subtests passed.
All six failures are the exact-source G1 inventory checks, not waived code
failures. The trusted record is unchanged; the adjacent proposal awaits review.
