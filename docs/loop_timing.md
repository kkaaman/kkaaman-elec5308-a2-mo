# Loop timing profile — TEST-22 / TEST-17

Owners: Abdullah (TEST-22), Mohammad (TEST-17). Status: offline profiler and
tests implemented; the headless-versus-HUD and warm-model measurements are live
trials listed in `TRIAL_PLAN.md`. Nothing here changes the frozen 100 ms budget.

## What is measured

`evaluation.loop_timing` turns retained diagnostics into one profile per
labelled configuration and compares them. Inputs it reads unchanged:

| Input | Written by | End-to-end column | Stage columns |
|---|---|---|---|
| S0 run folder (`samples.csv`) | `scripts/run_scenario.py` | `capture_to_apply_ms` (RGB callback receipt -> acknowledged apply) | none; `frame_age`, `real_time_factor` reported |
| `timing.jsonl` | `scripts/measure_operational_governance.py`, `scripts/capture_operational_approach.py` | `capture_to_decision_s` (host clock before tick -> decision); `receipt_to_decision_s` also summarised | `sensor_availability_s`, `bundle_s`, `detector_s`, `filter_s`, `governor_s`, `controller_s`, `decision_and_overhead_s`, `stack_s` |
| `Diagnostics` objects | any stack | caller-supplied seconds | `stage_durations_s` pairs (`detector`, `policy`, `total`, ...) |

Any further `<name>_s` column (for example `segmentation_s`, `lane_monitor_s`,
`lidar_s`, `hud_s`) is picked up automatically, so optional workloads only need
to log their wall cost per tick to appear in the per-stage table.

Per profile: p50 / p95 / max / mean in milliseconds and the sample count for
every stage and for the end-to-end latency; real-time factor (per-sample
column, or simulation span / wall span when only timestamps exist); maximum
frame age; fraction of valid observations. Percentiles use linear interpolation
at `(n - 1) p`, the same rule as `evaluation.s0_metrics`.

## Gate

The only verdict is the frozen capture-to-apply budget:

| Status | Rule |
|---|---|
| `PASS` | p95 <= 100 ms with at least 20 end-to-end samples |
| `FAIL` | p95 > 100 ms at any sample count |
| `INSUFFICIENT_EVIDENCE` | no end-to-end samples, or fewer than 20 |

`compare_profiles(profiles, baseline=..., final=...)` reports p95 deltas of the
end-to-end latency, each shared stage and the real-time factor against the
baseline label, and writes a `gate` block for the `final` label: `ADMIT` only
when its status is `PASS`, otherwise `BLOCKED` with the reason spelled out
(TEST-22 acceptance: "final baseline config p95 <= 100 ms or gate explicitly
blocked").

## Command

```powershell
& $py scripts/profile_loop_timing.py --output runs/TEST-22/profile `
    --run headless=runs/S0/headless-<id>/run-01 `
    --run hud=runs/S0/hud-<id>/run-01 `
    --run warm=runs/BL-C-05/governance/timing.jsonl `
    --baseline headless --final hud
```

Outputs `latency_profile.json` (sources, profiles, comparison, gate) and
`latency_profile.png` (end-to-end CDFs with the 100 ms line; per-stage p95
bars). The exit code is 0 only when the final configuration passes.

## Interpretation notes

- `capture_to_apply_ms` excludes render and transport before the callback;
  `capture_to_decision_s` starts earlier (before the tick request) and ends at
  the decision, before actuation. The two are not interchangeable; the profile
  records which definition each label used.
- Real-time factor below 1 means the synchronous loop runs slower than the
  simulated clock; it is reported, not judged.
- Headless versus HUD: the S0 runner has no in-loop HUD. The live comparison
  therefore varies server rendering (`-RenderOffScreen` versus windowed) and the
  RGB pipeline (`--navigation-only` versus RGB-enabled); a HUD rendering cost in
  the loop must be logged as `hud_s` once the E3 HUD consumer exists.
