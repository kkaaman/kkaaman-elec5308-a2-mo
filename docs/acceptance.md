# Acceptance scenarios and metrics (BL-D-01)

These definitions decide what "the baseline works" means. Thresholds are
provisional until gate G1. After that, they change only through a dated
entry in `decisions.md` and a rerun of the affected evidence.

All results start as **NOT RUN**. A blank result is not zero.

## Scenarios

Common setup:
- Town10HD and one fixed vehicle blueprint.
- The fixed sensor rig and one open, non-self-crossing mission route
  (`mission.md`).
- Traffic lights held green by the scenario, and no background traffic.
- A fixed seed per replicate.

Every scenario needs **5 trials**. All attempts are kept, not just the best 5.

| ID | Setup | Pass condition | Completes route? |
|---|---|---|---|
| S0 | Open mission, no obstacles | Finishes with no false stop | Yes |
| S1 | Stationary car in lane ahead; removed by script after the hold | Stop → hold → perceived clearance → resume → finish | Yes |
| S2 | Stationary pedestrian in lane ahead; removed after the hold | Same as S1 | Yes |
| S3 | Car parked at the roadside, outside the swept corridor | No false stop, no substantial slowdown | Yes |
| S4 | Persistent obstacle, never removed | Stops and holds for the whole interval (60 s) | **No: completion is N/A** |
| S5 | Stationary obstacle just after a bend, with a physically achievable sight distance | Stop → hold → removal → resume → finish | Yes |

**Clearance comes from perception.** The scenario removes the obstacle, and
the car must discover that through its own sensors. The removal event is
never fed to the stack.

**S5 must be achievable.** Its sight distance and pre-bend curve speed are
documented, so the stopping budget (`depth-association.md`, §7) is
physically possible.

## Metric applicability

| Metric | S0 | S1 | S2 | S3 | S4 | S5 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Route completion | ✓ | ✓ | ✓ | ✓ | N/A | ✓ |
| Collisions | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Unplanned lane invasions | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Stopped margin | – | ✓ | ✓ | – | ✓ | ✓ |
| Cruise speed error | ✓ | ✓ | ✓ | ✓ | – | ✓ |
| Mean cross-track error | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| End-to-end latency | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| False stops | ✓ | – | – | ✓ | – | – |
| Resume time | – | ✓ | ✓ | – | – | ✓ |
| HOLD stability | – | ✓ | ✓ | – | ✓ | ✓ |
| Added slowdown | – | – | – | ✓ | – | – |
| Depth error | Dataset-level (BL-B-04), not per scenario |||||

– means not measured in that scenario. N/A means measured but not
applicable to pass or fail.

## Definitions

| Metric | Target (provisional) | How it is measured |
|---|---|---|
| **Route completion** | 100% of runs (S0–S3, S5) | Ego reaches the route endpoint and satisfies the stopped criterion within goal tolerance. Monotonic progress alone is not proof |
| **Collisions** | 0 in every run | Unique physical contacts from the collision sensor (`EvaluationPacket`). Repeated events with the same actor within 0.5 s count once; raw events are logged |
| **Unplanned lane invasions** | 0 in every run | Raw lane-invasion events, excluding only crossings at mapped, declared manoeuvres on the mission. Junctions are never excluded wholesale |
| **Stopped margin** | 2.5–6.0 m | Ground truth: ego front bumper to the nearest surface of the obstacle's box, along the route, at the moment the **stopped criterion** first holds. Report the worst minimum and worst maximum across runs |
| **Stopped criterion** | – | Speed ≤ 0.1 m/s for ≥ 1.0 s continuously |
| **Cruise speed error** | RMS ≤ 0.5 m/s | Over stable cruise intervals only: behaviour CRUISE, target unchanged for ≥ 2 s, outside curve and endpoint caps. Compared with the active governed target, not a fixed 5 m/s |
| **Mean cross-track error** | ≤ 0.3 m | Rear-axle distance to the continuous route path, averaged per run. Report the signed trace and the maximum as context |
| **End-to-end latency** | p95 ≤ 100 ms | `host_command_s − host_tick_s` per frame (`interfaces.md`, clocks). Also report p50, max, frame age, missed control deadlines (> dt) and the real-time factor |
| **Depth error** | Median absolute error ≤ 0.5 m within 25 m | Measured surface distance against ground truth for the same visible surface, in range bins with counts and the invalid rate |
| **False stops** | 0 | Target speed < 0.1 m/s for ≥ 1 s at any time other than the planned final stop |
| **Resume time** | Report only | From the scripted removal to the ego exceeding 0.5 m/s, with the clear-counter trace |
| **HOLD stability** | ≤ 0.1 m/s throughout | Maximum speed while in HOLD. In S4, held for the full 60 s |
| **Added slowdown (S3)** | ≤ 0.5 m/s | Mean speed deficit against the matched S0 cruise window, excluding curve and goal intervals |

## Recording

Each run writes `runs/<scenario>/<run-id>/`:
- the config snapshot, git hash and dirty flag;
- model and dependency versions with hashes, hardware and seed;
- a per-tick log (frame, timings, behaviour, speeds, caps, gap, CTE) and the
  raw evaluation events;
- a summary JSON and selected images or video.

`scripts/evaluate_runs.py` (BL-F-05) computes every metric above from
those files alone.

A scenario passes only when all its applicable metrics pass in all 5 runs.
A failed criterion may be reported as blocked or conditional, never as a
pass.

## Offline logging implementation

See [acceptance-logging.md](acceptance-logging.md) for issue #37 commands,
record formats, golden fixtures and the limits of the offline S0 evidence.
