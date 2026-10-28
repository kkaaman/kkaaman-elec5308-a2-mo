# TEST-08 baseline evidence review

State: **BLOCKED**

Reviewed register aggregation only; no raw-log evaluation or admission.

## Thirty required trial slots

| Scenario | Trial | Attempts and state |
|---|---|---|
| S0 | 1 | NOT RUN / evidence pending |
| S0 | 2 | NOT RUN / evidence pending |
| S0 | 3 | NOT RUN / evidence pending |
| S0 | 4 | NOT RUN / evidence pending |
| S0 | 5 | NOT RUN / evidence pending |
| S1 | 1 | NOT RUN / evidence pending |
| S1 | 2 | NOT RUN / evidence pending |
| S1 | 3 | NOT RUN / evidence pending |
| S1 | 4 | NOT RUN / evidence pending |
| S1 | 5 | NOT RUN / evidence pending |
| S2 | 1 | NOT RUN / evidence pending |
| S2 | 2 | NOT RUN / evidence pending |
| S2 | 3 | NOT RUN / evidence pending |
| S2 | 4 | NOT RUN / evidence pending |
| S2 | 5 | NOT RUN / evidence pending |
| S3 | 1 | NOT RUN / evidence pending |
| S3 | 2 | NOT RUN / evidence pending |
| S3 | 3 | NOT RUN / evidence pending |
| S3 | 4 | NOT RUN / evidence pending |
| S3 | 5 | NOT RUN / evidence pending |
| S4 | 1 | NOT RUN / evidence pending |
| S4 | 2 | NOT RUN / evidence pending |
| S4 | 3 | NOT RUN / evidence pending |
| S4 | 4 | NOT RUN / evidence pending |
| S4 | 5 | NOT RUN / evidence pending |
| S5 | 1 | NOT RUN / evidence pending |
| S5 | 2 | NOT RUN / evidence pending |
| S5 | 3 | NOT RUN / evidence pending |
| S5 | 4 | NOT RUN / evidence pending |
| S5 | 5 | NOT RUN / evidence pending |

## Metric coverage and descriptive uncertainty

All attempts, including failures, contribute. Retries are not independent trials.
Sample standard deviation is descriptive spread, not a confidence interval.

### S0

| Metric | Rule | n attempts | Min | Max | Median | Sample SD |
|---|---|---:|---|---|---|---|
| Route completion (minimum per run) | >= 1.0 fraction | 0 | pending | pending | pending | pending |
| Collisions (maximum per run) | = 0.0 count | 0 | pending | pending | pending | pending |
| Unplanned lane invasions (max per run) | = 0.0 count | 0 | pending | pending | pending | pending |
| Cross-track mean (worst run) | <= 0.3 m | 0 | pending | pending | pending | pending |
| Cruise speed RMS (worst run) | <= 0.5 m/s | 0 | pending | pending | pending | pending |
| Capture-to-apply p95 (worst run) | <= 100.0 ms | 0 | pending | pending | pending | pending |
| False stops (maximum per run) | = 0.0 count | 0 | pending | pending | pending | pending |

### S1

| Metric | Rule | n attempts | Min | Max | Median | Sample SD |
|---|---|---:|---|---|---|---|
| Route completion (minimum per run) | >= 1.0 fraction | 0 | pending | pending | pending | pending |
| Collisions (maximum per run) | = 0.0 count | 0 | pending | pending | pending | pending |
| Unplanned lane invasions (max per run) | = 0.0 count | 0 | pending | pending | pending | pending |
| Cross-track mean (worst run) | <= 0.3 m | 0 | pending | pending | pending | pending |
| Cruise speed RMS (worst run) | <= 0.5 m/s | 0 | pending | pending | pending | pending |
| Capture-to-apply p95 (worst run) | <= 100.0 ms | 0 | pending | pending | pending | pending |
| Minimum stopped bumper margin | >= 2.5 m | 0 | pending | pending | pending | pending |
| Maximum stopped bumper margin | <= 6.0 m | 0 | pending | pending | pending | pending |
| HOLD maximum speed | <= 0.1 m/s | 0 | pending | pending | pending | pending |
| Resume delay (median) | report only | 0 | pending | pending | pending | pending |

### S2

| Metric | Rule | n attempts | Min | Max | Median | Sample SD |
|---|---|---:|---|---|---|---|
| Route completion (minimum per run) | >= 1.0 fraction | 0 | pending | pending | pending | pending |
| Collisions (maximum per run) | = 0.0 count | 0 | pending | pending | pending | pending |
| Unplanned lane invasions (max per run) | = 0.0 count | 0 | pending | pending | pending | pending |
| Cross-track mean (worst run) | <= 0.3 m | 0 | pending | pending | pending | pending |
| Cruise speed RMS (worst run) | <= 0.5 m/s | 0 | pending | pending | pending | pending |
| Capture-to-apply p95 (worst run) | <= 100.0 ms | 0 | pending | pending | pending | pending |
| Minimum stopped bumper margin | >= 2.5 m | 0 | pending | pending | pending | pending |
| Maximum stopped bumper margin | <= 6.0 m | 0 | pending | pending | pending | pending |
| HOLD maximum speed | <= 0.1 m/s | 0 | pending | pending | pending | pending |
| Resume delay (median) | report only | 0 | pending | pending | pending | pending |

### S3

| Metric | Rule | n attempts | Min | Max | Median | Sample SD |
|---|---|---:|---|---|---|---|
| Route completion (minimum per run) | >= 1.0 fraction | 0 | pending | pending | pending | pending |
| Collisions (maximum per run) | = 0.0 count | 0 | pending | pending | pending | pending |
| Unplanned lane invasions (max per run) | = 0.0 count | 0 | pending | pending | pending | pending |
| Cross-track mean (worst run) | <= 0.3 m | 0 | pending | pending | pending | pending |
| Cruise speed RMS (worst run) | <= 0.5 m/s | 0 | pending | pending | pending | pending |
| Capture-to-apply p95 (worst run) | <= 100.0 ms | 0 | pending | pending | pending | pending |
| False stops (maximum per run) | = 0.0 count | 0 | pending | pending | pending | pending |
| Cruise speed drop versus matched S0 | <= 0.5 m/s | 0 | pending | pending | pending | pending |

### S4

| Metric | Rule | n attempts | Min | Max | Median | Sample SD |
|---|---|---:|---|---|---|---|
| Route completion (minimum per run) | N/A | 0 | pending | pending | pending | pending |
| Collisions (maximum per run) | = 0.0 count | 0 | pending | pending | pending | pending |
| Unplanned lane invasions (max per run) | = 0.0 count | 0 | pending | pending | pending | pending |
| Cross-track mean (worst run) | <= 0.3 m | 0 | pending | pending | pending | pending |
| Cruise speed RMS (worst run) | <= 0.5 m/s | 0 | pending | pending | pending | pending |
| Capture-to-apply p95 (worst run) | <= 100.0 ms | 0 | pending | pending | pending | pending |
| Minimum stopped bumper margin | >= 2.5 m | 0 | pending | pending | pending | pending |
| Maximum stopped bumper margin | <= 6.0 m | 0 | pending | pending | pending | pending |
| HOLD maximum speed | <= 0.1 m/s | 0 | pending | pending | pending | pending |
| Continuous HOLD duration (minimum) | >= 60.0 s | 0 | pending | pending | pending | pending |

### S5

| Metric | Rule | n attempts | Min | Max | Median | Sample SD |
|---|---|---:|---|---|---|---|
| Route completion (minimum per run) | >= 1.0 fraction | 0 | pending | pending | pending | pending |
| Collisions (maximum per run) | = 0.0 count | 0 | pending | pending | pending | pending |
| Unplanned lane invasions (max per run) | = 0.0 count | 0 | pending | pending | pending | pending |
| Cross-track mean (worst run) | <= 0.3 m | 0 | pending | pending | pending | pending |
| Cruise speed RMS (worst run) | <= 0.5 m/s | 0 | pending | pending | pending | pending |
| Capture-to-apply p95 (worst run) | <= 100.0 ms | 0 | pending | pending | pending | pending |
| Minimum stopped bumper margin | >= 2.5 m | 0 | pending | pending | pending | pending |
| Maximum stopped bumper margin | <= 6.0 m | 0 | pending | pending | pending | pending |
| HOLD maximum speed | <= 0.1 m/s | 0 | pending | pending | pending | pending |
| Resume delay (median) | report only | 0 | pending | pending | pending | pending |

## Failure and insufficient-evidence log

No attempt records supplied; this is not a zero-failure result.

## Fault evidence

- hold_occlusion: BLOCKED (0 retained attempts)
- hold_missing_depth: BLOCKED (0 retained attempts)
- hold_track_expiry: BLOCKED (0 retained attempts)
- hold_id_switch: BLOCKED (0 retained attempts)
- controller_invalid_state: BLOCKED (0 retained attempts)
- controller_behind_target: BLOCKED (0 retained attempts)
- controller_path_loss: BLOCKED (0 retained attempts)
