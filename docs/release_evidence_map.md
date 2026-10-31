# Release evidence map and report handoff

Candidate source **d8425ad905eb5808fa24473378c8045173fc2219**, branch `codex/release-candidate-20261031` (assembled by normal dependency merges of existing PR99/101). Canonical main remains `cf970f9e3192c5f77c17db8c4e66b5f1d562bc5a`. No tag, new PR merge, issue closure, live execution or human approval is claimed.

The selected configuration is `configs/release_demo.json` on that candidate. Default run:

```bash
PYTHONPATH=src python scripts/run_release.py --output runs/release/demo-01
```

Modes: `demo` (default), `hud`, `lidar`, `wayfinder`, `parking`, `weather`, `faults`, `service`, `observe`. Observe defaults to offline preflight; explicit execution is parked/full-brake only. See the source-linked workflow below.

| Capability / owner | Result and denominator | Evidence / limitation |
|---|---|---|
| Release composition (#41/#64; PR99+101) — PASS offline; live pending | 80 synthetic cycles; 19 positive, 60 full-brake; one route installation | [Evidence](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/d8425ad905eb5808fa24473378c8045173fc2219/docs/evidence/release-20261031/composed-demo.json). Synthetic images/detections/ranges and illustrative bicycle; no physical latency/stop claim. |
| RGB-D tracking/governor/navigation (#41/#43/#46; PR99) — PARTIAL | Fresh source/track and command expiry tests pass; normal target-driven stop missing | [Evidence](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/d8425ad905eb5808fa24473378c8045173fc2219/docs/evidence/issue41-recovery-20261026/README.md). YOLO trash-can car false positive remains; no semantic correction, full-stack empty-road motion or normal-stop PASS. |
| STOP/HOLD/RESUME (#40; PR116) — BLOCKED RESUME / safe HOLD | Protected full-cell CLEAR prefix 0 m under the investigated contract | [Evidence](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/pull/116). PR116 evidence on its draft branch; physical restart acceleration bound absent; do not infer CLEAR from rays/expiry. |
| LiDAR sensor path (#52 closed; PR103 / PR115) — PASS bounded sensor geometry; driving pending | Final two matrices 90/90 each; max recorded in-corridor gap error 0.063779 m; all 360 attempted sweeps retained | [Evidence](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/d8425ad905eb5808fa24473378c8045173fc2219/docs/evidence/E2-01-04/live-20261029/README.md). Known-object sensor protocol, not driving stop or positive clearance certification. |
| LiDAR replay (#53; PR103) — PASS synthetic | 10 frames; 8 valid sweeps, 7 valid fused, 3 HOLD; max error 0.0348308563 m; depth-loss clear 0 | [Evidence](evidence/release-20261031/lidar-summary.json). Synthetic denominator and geometry distinct from recorded #103 matrix. |
| Wayfinder (#54/#55; PR102 merge 9cc45d6) — PASS core/recorded; live pending | Legal A*/Dijkstra fixtures; 24 recorded mini-maps, three paused ETA frames | [Evidence](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/d8425ad905eb5808fa24473378c8045173fc2219/docs/evidence/E3-wayfinder/recorded-test02/README.md). No lane-changing/dynamic rerouting; five reachable live pairs and regression absent. |
| P1 forward parking (#60; PR100 merge 7f0a80d) — PASS synthetic; live pending | 8/8 synthetic cases including obstacle, dropout and harness resume | [Evidence](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/d8425ad905eb5808fa24473378c8045173fc2219/docs/evidence/E5-P1/README.md). Downward synthetic depth and illustrative physics; harness resume not full-road #40 RESUME; no live P1 or reverse claim. |
| Lane monitor/HUD (#42/#45; PR101) — PARTIAL / disagreement retained | 573 recorded display frames; 262 valid nonjunction samples; signs 29/57 left, 69/152 right, 40/53 straight | [Evidence](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/d8425ad905eb5808fa24473378c8045173fc2219/docs/evidence/RV-06-08-TEST-22/live-20261029/README.md). Approximate matched poses/display RGB; outer marking selection defect; no pristine-label accuracy PASS. |
| Navigation component (#46; PR101) — PASS component; full baseline pending | S0 5/5; worst mean |CTE| 0.020446 m, cruise RMS 0.111257 m/s, capture-to-apply p95 13.648009 ms | [Evidence](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/d8425ad905eb5808fa24473378c8045173fc2219/docs/evidence/RV-06-08-TEST-22/live-20261029/navigation-summary.json). Historical navigation-only workload, not new candidate full perception S0. |
| Timing (#45; PR101) — PARTIAL | 234 warm shadow samples; capture-to-decision p95 34.234413 ms | [Evidence](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/d8425ad905eb5808fa24473378c8045173fc2219/docs/evidence/RV-06-08-TEST-22/live-20261029/warm-shadow-summary.json). Decision ends before actuation; final full-HUD capture-to-apply NOT_RUN; no cross-workload improvement ratio. |
| Weather (#49-#51; PR111 merge ed07850) — IMPLEMENTED / measurements deferred | 27 planned paired cells; all live cells NOT_RUN | [Evidence](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/d8425ad905eb5808fa24473378c8045173fc2219/docs/weather_experiment.md). No weather recall/range measurements or admission from a plan. |
| Inference/local fallback (#57/#58; PR105/106/109) — PASS protocol; hybrid unadmitted | Real pinned YOLO Docker smoke: isolated sessions, duplicate 409, reset valid | [Evidence](evidence/release-20261031/final-docker-smoke.log). CPU protocol requests are too slow for baseline; cloud/LAN accuracy and final failover NOT_RUN; default local operation. |
| All-flags fault suite (#63; PR112 merge 9de3755) — PASS available fixtures / scope BLOCKED | 216/216 available synthetic cases; 12 blocked combinations retained | [Evidence](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/d8425ad905eb5808fa24473378c8045173fc2219/docs/evidence/TEST-18/continuation/fixture-report/report.json). No G3 or live all-flags certification. |
| Release provenance/docs (#62/#64/#67/#73) — PROPOSED; human review pending | Frozen mission bytes/semantic identity unchanged; 19 negative probes reject drift | [Evidence](evidence/release-20261031/source-audit.json). Candidate branch, not canonical main acceptance or final release tag. |

## Source, environment and reproducibility

[Candidate workflow](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/d8425ad905eb5808fa24473378c8045173fc2219/docs/release_workflow.md), [configuration](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/d8425ad905eb5808fa24473378c8045173fc2219/configs/release_demo.json), [model manifest](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/d8425ad905eb5808fa24473378c8045173fc2219/configs/reproduction.json). The source audit retains every production hash/delta, exact sensors, governor, mission and model identities. `configs/mission_baseline.json` retains file SHA256 `a4b5e669f9426db24d42d9a1d20114981c95bb15302af006b1196044696fe187` and semantic identity `2903124878485c8030e160f3598c381bf0197329f79d74773d84048a96c2e9ef`. Loader-only rebind does not authorize the historical first-stop runner.

Validation uses a preinstalled Python 3.12 environment and verified local model bytes restored to a clean source checkout. This is not a claim that a fresh wheel acquisition occurred today. The existing hash-locked Linux CPU reproduction recipe remains separate from the preinstalled validation environment (Torch 2.6.0+cu124, CUDA device unavailable; execution on CPU). Environment details and exact commands are retained in the validation record. No user-private files are required by the composed offline demo; required model binaries and dependencies remain explicit prerequisites.

Durable raw outputs: `/home/alan/release-evidence-20261031/` (`runtime`, `hud`, `final-demo`, `final-preflight`, `final-recorded-lane`, `final-docker`). Compact results and hashes are retained here. Large historical raw sets remain in their versioned archives; do not cite a missing local pathname as a measurement.

## Figures worth using and suggested captions

1. `docs/evidence/E3-wayfinder/recorded-test02/frame_000012.png`: recorded route/current pose/next-turn example. Choose a frame identified as paused in its guidance log when claiming paused ETA; 24 frames/three paused states are the recorded denominator.
2. `docs/evidence/E5-P1/left.png`: “Illustrative P1 forward parking in the synthetic depth/kinematic harness; this does not establish CARLA parking tolerances.”
3. PR101 `docs/evidence/RV-06-08-TEST-22/live-20261029/model-mask115.png` and `monitor-frame115.png`: “Recorded display-frame model selects outer road markings; diagnostic disagreement is retained. Approximate pose alignment, not pristine-image accuracy.”
4. `docs/evidence/issue41-recovery-20261026/detector-deep-20261027/confidence-analysis.png`: known detector false-positive investigation. Retain conditions and failed evidence; do not label a threshold fix.
5. `docs/evidence/release-20261031/hud-synthetic-input.png` on candidate: composition/safe-unavailable illustration only, never a lane-quality figure.

Use at most the compact architecture plus one measured figure/table within the existing `report_outline.md` budget (900 body words + 100 captions/references; <=3 pages). Human authors must render and inspect the final report; no final PDF or page count is asserted here.

## Claim audit and honest cuts

Do not equate synthetic P1 resume with roadway RESUME; navigation-only S0 with full perception S0; decision latency with capture-to-apply; sensor gap accuracy with driving admission; Docker protocol success with cloud speedup; or a prepared weather matrix with measurements. Reverse remains disabled/future work. All positive-clearance loss, source loss and expiry remain conservative. New AI-assisted changes are pending actual human review before acceptance/merge.

See [all remaining issues](release_remaining.md), [minimal live queue](release_live_queue.md), [eight-minute script](release_presentation.md), machine evidence map and exact official issue bodies/dependencies.

## Final offline verification

Exact source `d8425ad`: **1,626 passed, one CARLA skip, three subtests**, one
third-party deprecation warning; clean checkout. Production loader and all
**19 negative probes** pass. Ruff, generated notes, AST boundary and **122 core
imports with simulator imports blocked**, strict Sphinx and diff checks pass.
All three model assets match. **26 evidence ZIPs** pass CRC and retain SHA256
records. All **32 original dirty files** match their preserved hashes.
Final network-none Docker real-model protocol smoke passes. The image was built
from the existing hash-locked CPU dependency image plus exact candidate source;
this does not claim a fresh offline OS/base-image/wheel acquisition.
See `evidence/release-20261031/validation.json` and individual retained logs.

No new PR was merged. Existing #99/#101/#116 remain open for substantive criteria
and actual human review. The report/presentation proposal is also pending human
review; no person is falsely recorded as having accepted it.

The current verification host reports no available CUDA device. Offline model processing is not evidence that the live 100 ms deadline can be met here. Tier 1 must retain deadline failures on the actual deployment hardware; optional HUD stays off unless its selected workload is demonstrated within budget.

Final-source recorded replay completed: **573/573 indexed display frames**, 115 diagnostic PNGs, 262 valid nonjunction samples. Sign results exactly reproduce the retained historical limitation (29/57 left, 69/152 right, 40/53 straight). The summary and compressed per-frame rows are retained under `evidence/release-20261031/recorded-lane-*`; no agreement PASS or model correction is asserted.

The proposed script/report package is [PR117](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/pull/117). PR99/101/116/117 all remain open; #117 awaits human action and the declared G4 prerequisite.
