# Issue56 offline inference service

Optional authenticated FastAPI service with strict protocol/timestamp identity,
bounded per-session trackers, ordering checks, explicit reset/TTL, health and
mounted hash-pinned weights. Original contributor history is preserved.
Canonical client PR105 is included.

Final validation: 55 focused server/client/control fixtures; 1055 full tests
passed, 1 CARLA deselected, 3 subtests, one dependency deprecation warning.
Production admission and 19 negative probes pass following complete production
review and supported loader-only rebind. Ruff, simulator boundary, strict
Sphinx and diff checks pass.

Fresh real-model Docker replay smoke PASSED: two clients, duplicate rejection,
repeated episode with a new session. Image/source/model/lock identities are
recorded in final-container-provenance.json. Network disabled, read-only root
and model mount, 4GiB, 2CPUs, 128PIDs, nonroot and no capabilities. The temporary
container was stopped and removed. CPU wheels are fully hashed and installed
offline. Black-image smoke is not accuracy, LAN latency or >=200-frame evidence.

E4-06 failure/recovery and backend-switch fixtures retain HOLD across source/
session ID changes, including empty detections; no track-ID change releases
safety state. Issue56's offline criteria are satisfied; #57 real deployment
comparisons and #58 extension operational acceptance remain open. CARLA stayed
OFF. Earlier failed source-binding logs/proposals remain historical.

AI-assisted with OpenAI Codex; delegated source review is not new human sign-off.
