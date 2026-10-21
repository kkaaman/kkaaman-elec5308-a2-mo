# TEST-08: 30 live baseline runs with one frozen configuration

Status: **`MEASURED_PASS_REVIEW_REQUIRED`**. All 30 runs pass. Live CARLA, frozen
configuration, full coverage, unique seeds and consistent provenance are all true, and
there are no errors. The report is `acceptance_amended.json` in this folder. The status
still needs the team's review; the tool does not grant acceptance.

- Runs: `Group7_batch35_Acceptance_20261018_174604_187/runs/S0-01..S5-05` (under
  Downloads), CARLA 0.9.16 server, Town10HD_Opt, synchronous at 0.05 s, full stack,
  CUDA, Python 3.12.0, torch 2.10.0+cu128, ultralytics 8.4.177.
- Configuration: `Group7_batch35_Freeze_20261018_174441_158/frozen.json`,
  `FROZEN_FOR_VALIDATION`, cruise 4.8 m/s, detector 1280 px and confidence 0.25.
- Provenance (identical in all 30 manifests): commit `d389c74` with a clean working tree;
  `source_sha256 757127763a7b…`, `config_sha256 a4ae63e98c5a…`,
  `model_sha256 9b09cc8bf347…`, `mission_sha256 7fee3ad2831260…`, and event verification
  `MEASURED_PASS` (`ed241134744f…`).
- Runtime recovered: `frozen_runtime.json` (this folder) is the frozen runtime rebuilt from the
  30 manifests. Its hash equals `config_sha256 a4ae63e98c5a…`. Its status is PROVISIONAL
  because the original `frozen.json` and its provenance files were lost with the old PC.
  Use it for diagnostic runs, demos and new routes, not to re-certify this acceptance.
- Re-scored offline on 2026-10-18 after the S4 amendment below, with no re-driving. The
  original `acceptance.json` (`NOT_ACCEPTED`, S4 cruise RMS missing) stays beside it in
  the runs folder.

## Results (worst of five runs unless stated)

| Scenario | Metric | Criterion | Worst of 5 (range) | Status |
|---|---|---|---|---|
| S0 — Open mission; no obstacles | Route completion | ≥ 1.0 | 1.00 | PASS |
|  | Collisions | = 0 | 0 | PASS |
|  | Unplanned lane invasions | = 0 | 0 | PASS |
|  | Cross-track mean (m) | ≤ 0.30 | 0.028 | PASS |
|  | Cruise speed RMS (m/s) | ≤ 0.50 | 0.003 | PASS |
|  | Capture-to-apply p95 (ms) | ≤ 100 | 46 (32–46) | PASS |
|  | False stops | = 0 | 0 | PASS |
| S1 — Car in corridor; HOLD then scripted removal | Route completion | ≥ 1.0 | 1.00 | PASS |
|  | Collisions | = 0 | 0 | PASS |
|  | Unplanned lane invasions | = 0 | 0 | PASS |
|  | Cross-track mean (m) | ≤ 0.30 | 0.025 (0.024–0.025) | PASS |
|  | Cruise speed RMS (m/s) | ≤ 0.50 | 0.002 (0.001–0.002) | PASS |
|  | Capture-to-apply p95 (ms) | ≤ 100 | 47 | PASS |
|  | Min stopped bumper margin (m) | ≥ 2.5 | 4.596 (4.596–4.611) | PASS |
|  | Max stopped bumper margin (m) | ≤ 6.0 | 4.611 (4.597–4.611) | PASS |
|  | HOLD max speed (m/s) | ≤ 0.1 | 0.0005 (0.0003–0.0005) | PASS |
|  | Resume delay (s) | report | 1.55 | Reported |
| S2 — Pedestrian in corridor; HOLD then removal | Route completion | ≥ 1.0 | 1.00 | PASS |
|  | Collisions | = 0 | 0 | PASS |
|  | Unplanned lane invasions | = 0 | 0 | PASS |
|  | Cross-track mean (m) | ≤ 0.30 | 0.025 | PASS |
|  | Cruise speed RMS (m/s) | ≤ 0.50 | 0.001 | PASS |
|  | Capture-to-apply p95 (ms) | ≤ 100 | 47 (32–47) | PASS |
|  | Min stopped bumper margin (m) | ≥ 2.5 | 4.785 | PASS |
|  | Max stopped bumper margin (m) | ≤ 6.0 | 4.786 | PASS |
|  | HOLD max speed (m/s) | ≤ 0.1 | 0.0053 | PASS |
|  | Resume delay (s) | report | 1.55 | Reported |
| S3 — Car outside corridor | Route completion | ≥ 1.0 | 1.00 | PASS |
|  | Collisions | = 0 | 0 | PASS |
|  | Unplanned lane invasions | = 0 | 0 | PASS |
|  | Cross-track mean (m) | ≤ 0.30 | 0.028 | PASS |
|  | Cruise speed RMS (m/s) | ≤ 0.50 | 0.003 | PASS |
|  | Capture-to-apply p95 (ms) | ≤ 100 | 32 | PASS |
|  | False stops | = 0 | 0 | PASS |
|  | Cruise drop vs matched S0 (m/s) | ≤ 0.5 | 0.0001 (0.0000–0.0001) | PASS |
| S4 — Persistent obstacle; 60 s HOLD | Route completion | not applicable | n/a | Not Applicable |
|  | Collisions | = 0 | 0 | PASS |
|  | Unplanned lane invasions | = 0 | 0 | PASS |
|  | Cross-track mean (m) | ≤ 0.30 | 0.001 (0.000–0.001) | PASS |
|  | Cruise speed RMS (m/s) | not applicable | n/a | Not Applicable |
|  | Capture-to-apply p95 (ms) | ≤ 100 | 47 | PASS |
|  | Min stopped bumper margin (m) | ≥ 2.5 | 4.584 (4.584–4.611) | PASS |
|  | Max stopped bumper margin (m) | ≤ 6.0 | 4.611 (4.584–4.611) | PASS |
|  | HOLD max speed (m/s) | ≤ 0.1 | 0.0005 (0.0003–0.0005) | PASS |
|  | Continuous HOLD (s) | ≥ 60 | 60.05 | PASS |
| S5 — Obstacle after bend; HOLD then removal | Route completion | ≥ 1.0 | 1.00 | PASS |
|  | Collisions | = 0 | 0 | PASS |
|  | Unplanned lane invasions | = 0 | 0 | PASS |
|  | Cross-track mean (m) | ≤ 0.30 | 0.035 (0.034–0.035) | PASS |
|  | Cruise speed RMS (m/s) | ≤ 0.50 | 0.005 | PASS |
|  | Capture-to-apply p95 (ms) | ≤ 100 | 47 (32–47) | PASS |
|  | Min stopped bumper margin (m) | ≥ 2.5 | 4.814 (4.814–4.828) | PASS |
|  | Max stopped bumper margin (m) | ≤ 6.0 | 4.828 (4.814–4.828) | PASS |
|  | HOLD max speed (m/s) | ≤ 0.1 | 0.0002 (0.0001–0.0002) | PASS |
|  | Resume delay (s) | report | 1.55 (1.55–1.60) | Reported |

For resume delay the median is taken over the five runs, with the range in brackets.

## Read with the results

1. **S4 amendment, made after the results.** S4's "Cruise speed RMS" was MISSING in all
   five runs, not failed. The persistent obstacle is confirmed from standstill, about
   30 m ahead while the car is still accelerating, so S4 has no stable cruise window (0
   cruise ticks in every run). The metric is marked not applicable for S4 only, in
   `tests/fixtures/acceptance_definitions.json` (see its `amendments` entry). Cruise
   tracking is still scored in S0, S1, S2, S3 and S5 and passes there. Because the change
   came after the results, the PR and report must state it.
2. **What five seeds mean here.** The seed labels the episode. The live world has no
   seed-dependent randomness: ClearNoon weather, no traffic, scripted obstacle placement.
   The five repeats therefore show repeatability (stop margins agree within 2.7 cm), not
   robustness across varied conditions. Weather and traffic variation are extension
   work.
3. **Capture-to-apply resolution.** The latency clock is Python's `time.monotonic`,
   which ticks every 15.6 ms on Windows. The full-stack p95 of 32-47 ms is 2-3 clock
   ticks. The 100 ms criterion is resolved; finer values are not.
4. **Stop margins.** The car stops 4.6-4.8 m from the obstacle's bounding box, inside the
   2.5-6.0 m window. The frozen governor targets a 4.0 m stop margin. The 4.8 m/s cruise
   is derived from the measured 17.5 m in-path detection range (car and person, BL-C-05)
   less the 2 m range reserve and that margin. The board's 25 m pedestrian target is
   **not** met (`pedestrian_25m_covered: false`).
5. **Resume delay** (report only) is the time from scripted removal to motion: 1.55 s,
   and up to 1.60 s in S5. The window from removal to motion is exempt from false-stop
   counting.
6. **Source hash portability.** `source_sha256` sorts paths with the host OS's ordering.
   The branch head reproduces `757127…` with Windows ordering, which confirms that no
   hashed file has changed since these runs. On Linux the same files hash differently.
   Making the order OS-independent is a source change, deferred until after the frozen
   validation.

Related evidence: perception range and depth (Perception stage), BL-C-07 fault replay
(`../BL-C-07-fault-replay/`), TEST-03 A1 versus A2 (`../TEST-03-a1-vs-a2/`).
