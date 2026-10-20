# TEST-03: A1 versus A2 navigation on A1's cross-track definition, and loop cost

Date: 2026-10-20 (A2 runs from 2026-10-18). Status from the tool:
`COMPARABLE_DEFINITION_MEASURED_REVIEW_REQUIRED`. The numbers are measured; the
interpretation below is ours.

- A1: committed TEST-01 capture (`docs/evidence/TEST-01-samples.csv`), 258 moving samples.
- A2: the five live Navigation runs used to freeze the configuration,
  `Group7_batch35_Navigation_20261018_174322_751/runs/S0-01..05` (seeds 5308-5312), about
  655 moving samples each.
- Lane topology: `Group7_batch35_LiveSetup_20261017_164642_464/mission/topology.json`.

Reproduce:

```
python scripts/compare_runs.py --a2-run <Navigation>/runs/S0-01 ... S0-05 --topology <LiveSetup>/mission/topology.json --output <new folder>
```

Outputs: `comparison.json` and `comparison.png` (this folder). `traces.json` stays in
the output folder.

## Results

| Measure (moving samples) | A1 | A2, five runs (range) |
|---|---|---|
| Mean \|CTE\|, A1 definition | 0.918 m | 0.0353-0.0358 m |
| p95 \|CTE\|, A1 definition | 1.73 m | 0.200-0.203 m |
| Max \|CTE\|, A1 definition | 3.43 m | 0.315 m |
| Mean \|CTE\| outside junctions | n/a | 0.0122-0.0133 m (p95 0.104-0.114 m) |
| Mean \|CTE\|, A2 route definition (rear axle) | n/a | 0.0279-0.0282 m (p95 0.177 m) |
| Loop cost p95 | 467 ms (A1 cycle before apply) | 16 ms (capture to apply) |
| Control period | 0.43 s mean, variable | 0.05 s fixed |
| Mean speed | 2.08 m/s | 4.07-4.10 m/s (cruise 4.8 m/s) |

The five A2 runs agree to within a millimetre on mean CTE. About half of each A2 run is
inside junctions, where the A1 nearest-lane definition switches between overlapping
lanes. That is why A2's A1-definition p95 (0.20 m) is higher than its own route
definition p95 (0.18 m) and its outside-junction p95 (0.11 m).

The A1 offline definition reproduces A1's own logged CTE within 8 mm (max) on the
TEST-01 capture.

## Differences that stop the ratio being a controller comparison

The tool reports `cte_ratio_a2_over_a1` = 0.039. Do not quote it as "A2 is 26 times
better" without the following:

1. Pose source. A2 tracks a fixed map route using the simulator's vehicle pose. A1
   steered from camera lane estimates. Most of the gap is localisation and lane
   estimation, not control. The camera lane check for A2 is RV-06 (lane-monitor
   agreement, 5 cm median on straights), and it is not used for steering.
2. Route. Different roads and junctions are driven, so CTE is compared as a
   distribution, not sample by sample.
3. Timing. A1 ran asynchronously with a 0.34-0.5 s control period. A2 runs synchronously
   at a fixed 0.05 s step, reading the frame it just ticked (simulation age 0).
4. Speed. A1 averaged 2.1 m/s (3 m/s cruise). A2 averaged 4.1 m/s at the frozen 4.8 m/s
   cruise. The tool's text "provisional 5 m/s" is a fixed string; the frozen value is
   4.8 m/s.
5. Vehicle. A1 used its demo's first blueprint match with nominal geometry. A2 uses the
   measured Tesla Model 3 geometry (BL-E-01).

## Loop-cost notes

- Clock resolution. A2 latencies come from Python's `time.monotonic`, which ticks every
  15.6 ms on Windows. Every recorded capture-to-apply value is 0, 15/16 or 31/32 ms.
  "p95 16 ms" therefore means within one clock tick, and the maximum is two ticks
  (32 ms). The 100 ms criterion is resolved; values below 16 ms are not. A
  higher-resolution clock (`time.perf_counter`) would be a source change, deferred until
  after the frozen validation.
- One host stall. Tick-to-apply p95 is 31 ms in every run. S0-04 has a single 1.58 s
  stall. In synchronous mode the simulator waits for the client, so simulation age stays
  0 and control is unaffected. The run-level `timing` report records stalls over 500 ms
  (see `docs/batch35_baseline.md`).
- Real-time factor. The A2 simulation runs 2.6-2.9 times faster than wall time in
  synchronous mode; A1 ran at 1.0.
- Figure. In `comparison.png` the minor tick labels on the log-scale loop-cost axis
  overlap. This is cosmetic.

## Reading

On A1's own CTE definition, the A2 baseline keeps the car within 0.32 m of the lane
centre on the whole route (1.2 cm mean outside junctions). Its loop cost is under 32 ms
against a 100 ms limit, where A1's p95 was 467 ms. This shows that the A2 navigation
baseline is well inside its lane and inside its timing budget. Because of differences
1-5 above, it is not a like-for-like controller comparison with A1.
