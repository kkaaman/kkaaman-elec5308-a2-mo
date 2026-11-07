# Baseline acceptance definitions — BL-D-01

Owner: Alan. Reviewer: Mohammad. Status: proposed; review pending.
Source: supplied task board, Scenarios & Metrics sheet. Machine-readable criteria
are preserved in `tests/fixtures/acceptance_definitions.json`. These are project
plan criteria, not a new claim of tutor approval or a replacement for official deadlines.
All actual scenario results remain NOT RUN.

| Scenario | Setup | Route completion | Stop/HOLD metrics |
|---|---|---|---|
| S0 | Open mission, no obstacles | Required | False stops; obstacle stop margin not applicable |
| S1 | In-corridor car, removal after HOLD | Required | Stopped margin, HOLD speed, resume delay |
| S2 | In-corridor pedestrian, removal | Required | Stopped margin, HOLD speed, resume delay |
| S3 | Car outside corridor | Required | False stops; intentional obstacle stop not applicable |
| S4 | Persistent obstacle, 60-second HOLD | NOT APPLICABLE | Stopped margin, HOLD speed, HOLD duration |
| S5 | Obstacle after bend, removal | Required | Stopped margin, HOLD speed, resume delay |

Apply only metrics explicitly assigned to each scenario in the JSON. Do not add
route-completion failure to S4. Missing evidence for an applicable metric is NOT
RUN/insufficient evidence, not zero or PASS. Report-only metrics are not pass/fail
thresholds. Preserve required run counts and evaluate each trial; average success
cannot hide a failed trial.

## Measurement definitions

- Completion: completed legal route to the goal according to endpoint criteria;
  projected arc-length alone cannot prove arrival. S4 intentionally cannot finish.
- Collisions: maximum per-run count; consistently deduplicate repeated callbacks
  for the same contact. Keep raw collision events and the deduplication rule.
- Unplanned invasions: maximum per-run count. Exclude only predeclared mapped
  legal manoeuvres; never exclude every junction automatically.
- Cross-track mean: per-run mean rear-axle distance to continuous reference,
  evaluated over the applicable motion windows; retain signed trace and tick count.
  Report the worst run, not a pooled mean hiding one unstable run.
- Cruise RMS speed error: stable cruise windows only; exclude approach, HOLD and
  goal transients by a predeclared rule. No cruise samples is not a zero error.
- Capture-to-apply p95: host-monotonic capture receipt to command application.
  This excludes unknown sensor/render/network latency before receipt; report
  simulation frame age and real-time factor separately. Use a fixed percentile
  calculation and state the number of samples.
- Stop margin: evaluation-truth bumper-to-obstacle surface gap after speed is
  <=0.1 m/s; report minimum and maximum steady-HOLD gaps across required trials.
- HOLD speed: maximum speed during the defined HOLD interval.
- Resume delay: scripted removal to resumed motion, using simulation timestamps;
  report separately from delay to perceived clearance. Median is report-only
  where the sheet supplies no threshold.
- False stop: unexpected target speed <0.1 m/s for >=1 second outside goal window;
  definition is frozen at G1 as stated in the board.

Thresholds and run counts are copied from the task board into the JSON. Examples:
S0 requires five runs, complete route, zero collisions/unplanned invasions,
worst-run mean CTE <=0.3 m, cruise RMS <=0.5 m/s and wall p95 <=100 ms.
The JSON is authoritative for the other scenario-specific limits; do not infer
limits from examples or silently relax thresholds after a failure.

## Review and later live evidence

Mohammad reviews decoding, optical convention test, extrinsics, missing-depth
policy, worked example and scenario applicability. Review is pending.
Numerical specification tests do not replace live alignment/convention checks,
vehicle calibration, obstacle-association tests or the five-run scenario suite.
The current BL-D-01 task is CARLA-free as marked in the board. Later sensor and
vehicle tasks must provide live CARLA evidence before pushing their acceptance claims.


## Offline acceptance logging

See [acceptance-logging.md](acceptance-logging.md) for the offline S0 logging implementation and its limitations.
