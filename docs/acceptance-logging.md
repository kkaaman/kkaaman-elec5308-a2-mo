# Offline acceptance logging (issue #37)

The S0 component harness runs the inherited `PathFollower` against fixed-step
bicycle dynamics on a straight 80 m route, without CARLA, neural models or network
access. It starts at cruise speed, with a small seeded lateral offset, and must
stop within goal tolerance for a continuous second. Five distinct seeds are used.
This is **offline controller component evidence**, not a Town10HD result or
validation of the complete autonomy/perception stack. The supplied snapshot has
no `AutonomyStack` or `KinematicWorld`; integration with those upstream modules
must be checked when they become available. Synthetic evaluation events are empty
in this obstacle-free harness; collision sensing is not validated here.

From the repository root, after the documented editable install:

```powershell
python scripts/run_s0_acceptance.py
python -m pytest
ruff check .
sphinx-build -W docs docs/_build/html
```

The thresholds are in `configs/acceptance.json`, matching the provisional S0
values in `acceptance.md`. Each invocation creates a new UTC-labelled batch and
retains every trial. Existing directories are never overwritten. Rerun a saved
batch using its five directory paths:

```powershell
python scripts/evaluate_runs.py runs/S0/<run-0> runs/S0/<run-1> runs/S0/<run-2> runs/S0/<run-3> runs/S0/<run-4> --output rechecked.json
```

A non-PASS result exits nonzero. An incomplete or corrupt log raises an error;
a missing metric is blocked. All supplied attempts are evaluated, including failed
ones. A batch requires at least five distinct directories and seeds with identical
configuration. Never select only successful attempts from a larger batch.

Each directory contains:

- `manifest.json`: schema, configuration snapshot, seed, git hash/dirty state,
  installed dependency versions and metadata hashes, model hashes (empty because
  no models are used), Python and hardware details. Git information is null in
  ZIP snapshots; run in a checkout to capture the commit.
- `ticks.jsonl`: matched frame/time, rear-axle state, full command, reason, mode,
  target, speed caps, gap, signed cross-track error, progress and stage timings.
- `evaluation.jsonl`: raw evaluation-only truth and events, never supplied to
  the controller.
- `summary.json`: results recomputed from those files. No summary is trusted as
  an input to metric calculation. The batch also has an aggregate JSON file.

Host timestamps are actual `perf_counter` readings relative to a shared run
origin. Simulator time is separate. State/commands/reasons repeat for the same
seed; host timestamps and measured timing naturally vary. No simulated timing
is presented as measured performance. Real-time factor describes this unpaced
component harness, not CARLA throughput. No image/video is produced by this
nonvisual harness.

Metric rules: completion requires endpoint proximity and sustained stopped
speed, not monotonic progress. Cruise RMS starts after a target has stayed fixed
for two seconds in CRUISE, with curve/endpoint phases excluded. CTE uses absolute
rear-axle distance, with the signed trace retained. False-stop episodes require
one continuous second below the target-speed threshold outside the declared
final stop. Collision events are deduplicated by actor over 0.5 seconds; unknown
actors count separately. Lane events are counted conservatively, without blanket
junction exemptions. Latency uses linear-interpolated p95; deadline misses use
strictly greater than dt. Missing frames and invalid clock order are rejected.

`tests/fixtures/acceptance/golden.json` fixes independent geometry, stopped-window,
front-to-surface gap, percentile, RMS, collision and false-stop expectations.
Unit tests also check missing evidence, interruption, truncation, bad timestamps,
logger rejection and repeatability. Socket connections are blocked in offline
pytest tests. Live CARLA tests retain their explicit marker exemption.

S1-S5, depth-dataset evaluation, obstacle HOLD/resume and matched S3 slowdown are
not implemented by this S0 evaluator. Those scenarios cannot receive PASS here;
they require subsequent integration and evidence under the relevant issues.
CI uploads S0 logs as `offline-s0-acceptance`, including failed-run artifacts.
