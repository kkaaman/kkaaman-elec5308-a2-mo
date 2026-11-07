# Offline acceptance logging (issue #37)

The S0 acceptance path uses the shared `KinematicWorld` and `SkeletonStack`
with an `EmptyDetector` and the inherited `PathFollower` motion policy. No shared
interfaces are modified. The world starts at cruise speed with a small seeded
lateral offset on a straight 80 m legal graph edge. Completion requires endpoint
proximity and speed <=0.1 m/s for a continuous second, not progress alone.
Five distinct seeds are retained, including failures.

This is **offline stack/world integration evidence with synthetic observations**.
It does not validate learned perception, Town10HD, live collision/lane sensors,
vehicle calibration, obstacle association or the complete CARLA stack. This
kinematic world does not simulate physical contacts; its empty raw evaluation
records are explicitly labelled synthetic. No real vehicle safety claim follows
from their zero counts. No images/video are produced by this nonvisual harness.

From the repository root after the documented editable install:

```powershell
python scripts/run_s0_acceptance.py
python -m pytest
python -m ruff check .
python -m sphinx -W -b html docs docs/_build/html
```

`configs/acceptance.json` contains operational window settings and the S0
thresholds. Before a run, the evaluator checks them against the authoritative
`tests/fixtures/acceptance_definitions.json`; changed applicability, thresholds
or run counts require an explicit evaluator update. The full source definitions
are snapshotted in each manifest, so offline re-evaluation needs only the saved
run files. Missing evidence never becomes zero or PASS.

Each invocation creates a UTC-labelled batch under `runs/S0/`. Existing run
folders are never overwritten. Re-evaluate a complete batch with its five paths:

```powershell
python scripts/evaluate_runs.py runs/S0/<run-0> runs/S0/<run-1> runs/S0/<run-2> runs/S0/<run-3> runs/S0/<run-4> --output rechecked.json
```

All supplied attempts are evaluated. A batch requires at least five distinct
run directories and seeds, identical configuration and a passing result from
every attempt. Do not select only successful attempts from a larger batch.
Non-PASS exits nonzero. Malformed/incomplete logs raise an error.

## Log schema 2

- `manifest.json`: schema, complete configuration, source criteria, seed,
  git hash/dirty flag, dependency versions/metadata hashes, model hashes (empty
  because no model is used), Python and hardware. Git information is null in
  ZIP snapshots; execute in a checkout to record the actual commit.
- `ticks.jsonl`: matched frame/time, rear-axle state, actual `ControlPacket`,
  stack diagnostic reasons and stage durations, behaviour, governed target,
  speed caps, signed CTE, progress, gap and final-stop classification. The
  random anti-replay episode ID changes each reset.
- `evaluation.jsonl`: raw evaluation-only state/events, never passed to policy
  or stack. Synthetic event provenance is explicit.
- `summary.json`: recomputed metrics/checks. Re-evaluation never trusts it as
  input. A batch aggregate JSON retains each trial's result.

Schema 1 logs from the earlier interface are rejected. State, command values and
reason traces repeat for a given seed. Episode UUIDs, measured timings, metadata
and host timestamps naturally vary and are excluded from repeatability comparison.

## Clocks and metrics

Receipt and application use **the same `time.monotonic()` clock**. The application
boundary is sampled immediately after `world.step(packet)` returns: a conservative
upper bound on application, including the synchronous model advance. It is not a
physical actuator timestamp. Capture-to-apply latency is this boundary minus the
input frame's `received_monotonic_s`. Unknown latency before sensor receipt is
excluded, as required by the current `acceptance.md`. Stack per-stage durations
use their existing `perf_counter` durations and are never subtracted from the
monotonic timestamps. Metadata collection happens before the first observation.

Simulation frame age is application simulation time minus observation simulation
time. It is zero here because the current packet applies at the start of its
fixed-step interval; world advancement produces the next observation afterward.
Real-time factor uses elapsed simulation time divided by actual host elapsed
time. This unpaced fixture is not a CARLA performance benchmark.

Completion requires endpoint proximity and the continuous stopped window. Cruise
RMS starts after two seconds at a fixed target in CRUISE and excludes active
curve/endpoint caps. No stable samples is blocked. CTE averages absolute rear-axle
distance over all sampled S0 states through final hold; signed trace and sample
count are retained. False-stop episodes require a target below 0.1 m/s for one
continuous second outside the planned endpoint braking/hold window. Collision
callbacks are deduplicated per actor over 0.5 seconds; unknown actors count
separately. Lane events are counted conservatively with no junction exemptions.
Latency p95 uses linear interpolation; deadline misses use strictly greater than
dt. Missing frames, invalid clocks, nonfinite JSON and invalid observations
cannot pass. All run checks must pass separately, so pooled means cannot mask
a failed trial.

The golden fixture fixes independent projection geometry, sustained-stop timing,
a first-stop front-to-surface-gap helper, percentile, RMS, event and false-stop
expectations. The gap helper is not an implementation of the team's steady-HOLD
S1-S5 metrics. Tests also check source-criteria drift, actual stack/world runs,
repeatability, episode/frame mismatches, truncated logs and missing evidence.
Socket connections are blocked in offline pytest tests; explicitly marked live
CARLA tests retain their exemption.

S1-S5, depth-dataset evaluation, steady-HOLD margin/speed, resume and matched S3
slowdown remain outside this S0 evaluator. Those scenarios cannot receive PASS
here. CI uploads S0 logs as `offline-s0-acceptance`, including failed-run artifacts.
