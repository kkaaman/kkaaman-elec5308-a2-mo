# BL-E-03 longitudinal calibration evidence

This evidence records live CARLA longitudinal braking measurements for the fixed
baseline vehicle on Town10HD_Opt using CARLA 0.9.16.

## Measurement

Nine trials were executed across:

| Target speed | Brake commands |
| --- | --- |
| 2 m/s | 0.4, 0.7, 1.0 |
| 5 m/s | 0.4, 0.7, 1.0 |
| 8 m/s | 0.4, 0.7, 1.0 |

For each trial, the raw trace records speed and travelled distance from throttle
acceleration through brake application to the stopped state.

The measured worst response lag was:

- **0.350000 s**

The weakest measured effective distance deceleration was:

- **2.139380 m/s²**

A 30% conservative reserve is applied to that measured deceleration:

- **1.497566 m/s² conservative deceleration**

`response_budget_s` contains only the measured vehicle response lag. Perception,
processing and confirmation latency are not assumed here and must be supplied
separately from their own measured evidence before the final stopping budget is
frozen.

## Environment and integrity

- CARLA client/server: **0.9.16**
- map: **Carla/Maps/Town10HD_Opt**
- settings restored after calibration: **yes**
- actor inventory restored/no leaks: **yes**
- calibration status: **MEASURED_PASS**

## Evidence

- `longitudinal.csv` — retained raw nine-trial trace
- `longitudinal.json` — derived per-trial measurements and conservative result
- `braking-traces.png` — measured speed-versus-distance traces

The JSON result is derived from the retained CSV. No historical A1 value is
treated as a new measurement.
