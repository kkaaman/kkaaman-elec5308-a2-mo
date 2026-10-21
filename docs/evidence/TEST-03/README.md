# TEST-03 repaired navigation versus A1 evidence

TEST-03 compares the retained A1 TEST-01 capture with the five accepted live
CARLA S0 runs from TEST-02.

For cross-track error, the A1 nearest-driving-lane definition is recomputed for
the A2 poses so both sets are reported using the same geometric definition.

## Inputs

A1:

- `docs/evidence/TEST-01-samples.csv`

A2:

- five `packaged-acceptance` runs retained by TEST-02 RGB-repair evidence
- CARLA 0.9.16
- Town10HD_Opt
- RGB enabled
- fixed simulation step: 0.05 s
- mission cruise target: 4.0 m/s

## Measured comparison

| Metric | A1 TEST-01 | A2 retained TEST-02 |
| --- | ---: | ---: |
| Mean absolute CTE using A1 definition | 0.917850 m | 0.023864 m worst run |
| Timing p95 | 467.421 ms cycle-before-apply | 16.000 ms capture-to-apply worst run |
| Real-time factor | 1.001 | 9.841–10.900 |
| Live CARLA evidence | Yes | Yes |

The calculated A2/A1 mean-CTE ratio is approximately 0.026. This is retained as
a measured comparison value, **not as a direct claim of improvement**.

## Comparability limitations

The measurements differ in several important ways:

- A1 follows lane estimates using its local junction behaviour, while A2 follows
  one fixed legal mission. They therefore drive different routes.
- A1 is asynchronous with a variable control period. A2 is synchronous at a
  fixed 0.05 s simulation step.
- A1 targets 3 m/s, whereas these retained A2 runs use a 4.0 m/s mission cruise
  target. The later BL-D-06 safety freeze is separate.
- A1 and A2 use different vehicle/controller geometry assumptions.
- A1 `cycle_wall_before_apply` and A2 `capture_to_apply` measure different timing
  intervals and must not be interpreted as identical latency metrics.

Because of these differences, the CTE distributions and timing measurements can
be reported side by side, but the numerical ratios alone do not establish that
the A2 system is categorically better than A1.

The offline recomputation of the A1 CTE definition against the original TEST-01
capture differed by at most approximately 0.0303 m, supporting use of the same
definition for this comparison.

## Evidence

- `comparison.json` — measured metrics, provenance and limitations
- `comparison.png` — visual comparison generated from the retained traces

Status: `COMPARABLE_DEFINITION_MEASURED_REVIEW_REQUIRED`.
