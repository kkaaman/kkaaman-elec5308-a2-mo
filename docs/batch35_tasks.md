# Batch task and acceptance record

The 35 tasks below are taken from the supplied board. This record maps implementation and evidence tooling; it does not change board status or certify live acceptance. Original board SHA-256: `5d9d6dfde5588c9ad5490dc927d127d0e7624bcdf525a97df646f0f823673a0d`.

| Task | Implementation / command | Remaining evidence or limit |
|---|---|---|
| BL-A-04 — Validate complete mission path in Town10HD | `planning/mission.py`, `carla_adapter/mission.py`; LiveSetup | Inspect saved legal-route overlay and junction/lane metadata in Town10HD. |
| BL-B-03 — Capture RGB-D and evaluation ground truth by frame | `carla_adapter/dataset_capture.py`; Capture | Collect and review the actual RGB-D, labels, ground truth and split manifest. |
| BL-D-04 — Implement route-frame obstacle footprints and corridor | `planning/corridor.py`, `obstacle_perception/association.py` | Synthetic corridor/footprint tests pass; confirm live depth geometry. |
| BL-C-04 — Implement track confirmation, median filter and expiry | `obstacle_perception/track_filter.py` | Synthetic confirmation/expiry/episode tests pass; inspect real tracking replays. |
| BL-E-03 — Measure throttle and brake steps for the fixed vehicle | `carla_adapter/longitudinal.py`, `evaluation/longitudinal.py`; LiveSetup | Complete all nine live brake trials and review the measured stopping budget. |
| BL-A-05 — Tune fixed-dt control without replacing Pure Pursuit | `scripts/tune_controller.py`; Tune | Select a qualified live candidate and inspect CTE/speed traces. |
| BL-B-04 — Validate depth convention and range accuracy in CARLA | `carla_adapter/depth_calibration.py`, `scripts/analyze_depth.py` | Establish axial/radial convention using off-axis rays and actual range errors. |
| BL-C-05 — Benchmark detection and track confirmation range offline | `evaluation/perception_replay.py`; Perception | Measure real car/person range, confirmation, depth errors and timing; 25 m pedestrian target remains explicit. |
| BL-D-05 — Specify behaviour guards and rate-limited speed policy | `planning/behaviour.py`, `planning/speed_governor.py` | Functional state/rate tests pass; live approach and resume need inspection. |
| BL-F-04 — Add harness S0 to CI and preserve A1 checks | `sim_harness/baseline.py`, CI S0 artifact | Six harness scenarios collected; CI must run on the actual PR. |
| TEST-02 — Run navigation S0 acceptance five times | Navigation stage, five S0 seeds | Five independent navigation-only CARLA runs must meet all fixture criteria. |
| TEST-03 — Measure repaired versus A1 navigation and loop cost | `scripts/compare_runs.py` | Agree comparable A1/A2 CTE definitions; review plot/provenance. Existing differing definitions cannot prove improvement. |
| BL-B-05 — Test optional segmentation masking for depth medians | `scripts/analyze_depth.py --segmentation-weights` | Measure real mask/depth ablation. Runtime mask remains off; false-stop benefit is unproven. |
| BL-D-06 — Freeze conservative speed and acceptance configuration | `scripts/freeze_thresholds.py`; Freeze | Freeze only from matching measured braking/depth/range and five live navigation runs. |
| G1-01 — Approve or block the navigation gate with evidence | `scripts/check_gates.py` | Team review of calibration, route, comparison, lane HUD and offline evidence; tool does not grant acceptance. |
| RV-06 — Repurpose lane estimates as departure monitor and HUD | `visualisation/lane_monitor.py`, `scripts/preview_lane_monitor.py` | Review live-derived lane overlays and sign/offset consistency; lane signal cannot command control. |
| BL-F-05 — Implement reproducible logger and metric aggregation | `evaluation/run_logger.py`, `evaluation/metrics.py` | Golden metric tests pass; review live manifests, timing and hashes. |
| BL-D-07 — Implement obstacle and curvature speed governor | `planning/speed_governor.py` | Functional stopping/curve envelope tests pass; validate measured braking and range assumptions in CARLA. |
| BL-C-06 — Join detector tracks to valid depth observations | Existing exact-frame join plus sensor receipt clocks; `perception/frame_detector.py` | Verify live paired frames, callback errors and deadline behaviour. |
| BL-B-06 — Implement deterministic obstacle scenarios and events | `configs/scenarios/S0.yaml` through `S5.yaml`, `CarlaWorld` | Review each actual actor placement and scripted removal condition. |
| BL-D-08 — Implement stop-hold-resume state machine | `autonomy/baseline.py`, behaviour/visibility/track modules | Harness stop/hold/resume passes; all live scenario margins and clearances remain pending. |
| BL-F-06 — Integrate perception-to-control stop in the harness | `scripts/run_fault_suite.py` | Missing/stale/invalid depth faults pass in harness; simulator faults still require relevant live evidence. |
| BL-E-04 — Wire the complete stack into CarlaWorld | `carla_adapter/baseline_world.py` | Mock guard tests pass; verify actual world lifecycle, rigid sensors, deadlines and cleanup. |
| G2-01 — Demonstrate the first perception-driven stationary-car stop | Smoke stage and gate report | First live perception-driven S1 stop with positive measured margin and clean cleanup. |
| BL-A-06 — Verify temporary stops do not latch mission completion | Policy state handling with existing PathFollower | Functional HOLD/resume/FINISHED separation passes; inspect live completion. |
| BL-D-09 — Tune stopping margin and visibility-aware clear hysteresis | Measured budget, configurable stop margin, raw margin metrics | Tune on development runs, retain failures, then freeze and repeat independent acceptance; no automatic success claim. |
| BL-C-07 — Fix detection flicker and ID-switch failures from first stop | ReplayFaults stage: flicker and ID reset | Inspect real RGB-D replay effects and clear-evidence behaviour; dynamic occlusion coverage remains limited. |
| BL-F-07 — Build one-command acceptance runner and fault suite | `scripts/run_scenario.py`, `run_acceptance_suite.py`, Windows stage runner | Six harness CLI runs pass; Windows/CARLA orchestration must be exercised. |
| BL-B-07 — Verify collision, lane invasion and ground-truth metrics | `carla_adapter/ground_truth.py`, `evaluation/ground_truth.py` | Geometry fixtures pass; validate live bumper-edge distances, collision and lane events. |
| TEST-04 — Run five S1 and five S2 stop-resume acceptance trials | Acceptance S1/S2, five each | Run and retain ten live vehicle/person stop-resume results. |
| TEST-05 — Run five S3 and five S4 acceptance trials | Acceptance S3/S4, five each | Match roadside S3 to S0 and verify continuous 60-second persistent-block holds. |
| TEST-06 — Repeat five S0 trials with the complete perception stack | Acceptance full-stack S0, five seeds | Measure five full perception/control clean-road runs, separate from navigation-only tests. |
| TEST-07 — Run five S5 bend-obstacle acceptance trials | Acceptance S5, five seeds | Measure five post-turn obstacle trials with route and video review. |
| TEST-08 — Aggregate all 30 baseline runs and fault tests | `evaluation/acceptance.py`; Acceptance | Thirty live runs with one frozen configuration and matching provenance must satisfy all criteria. |
| TEST-09 — Audit held-out detections and false-stop evidence | Audit stage and held-out perception report | Review untouched test split and failure examples; keep model/config unchanged after selecting on validation. |

See [batch35_baseline.md](batch35_baseline.md) for execution order and [batch35_training.md](batch35_training.md) for dataset/training instructions. Local verification is recorded separately in `docs/evidence/batch35-local-verification.txt`.

A task may be implemented while its acceptance remains pending. A synthetic pass is not a live CARLA pass, a collected run is not an accepted run, and an evidence-ready gate still requires the named reviewer.
