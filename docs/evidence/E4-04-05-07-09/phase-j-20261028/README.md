# Issue #57 client/fallback implementation scaffold

AI-assisted with OpenAI Codex, 28 October 2026. Existing PR #105 is updated.
This completes the existing implementation-scaffold scope, not issue #57.

The optional adapter composes bounded detection with the canonical freshness
gate, expiry, emergency latch and tracked governance. Source/session changes
invalidate the switch frame and reset depth confirmation without releasing an
existing obstacle HOLD. HTTP 410/503 recovery explicitly resets on the worker
using a fresh UUID; reset cost is separately reported.

Final validation: 37 focused tests; 1,037 full tests and 3 subtests passed,
1 CARLA test deselected. Production admission and all 19 negative probes pass.
Ruff, simulator boundary (95 modules), strict Sphinx and diff checks pass.
Complete protected inventory review and supported loader-only rebind are
recorded in the adjacent applied review. Historical logs remain historical.
No new human approval is asserted.

Issue #57 remains OPEN for >=200 comparable labelled real-model frames per
local/LAN configuration, hardware/configuration identity, latency/accuracy/miss
tables, and available cloud measurements (otherwise NOT_RUN). Synthetic and
loopback tests are not deployment performance evidence. #58 operational
extension admission and live fault cases remain open. CARLA stayed off.
