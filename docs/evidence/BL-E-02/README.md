# BL-E-02 steering-response measurements

I measured the installed Tesla Model 3 adapter in CARLA 0.9.16, Town10HD_Opt,
at spawn 152 with a fixed 0.05 s timestep on a dry road.
Each batch covers left/right 2, 4 and 6 degree commands at 2, 5 and 8 m/s.
Each steering window lasts 1 s. Commands use the unchanged nominal adapter
mapping: CARLA steer = -requested angle / physical maximum wheel angle.

## Results

| Target speed | Repeat inferred/requested gain |
|---|---|
| 2 m/s | 0.9392–0.9566 |
| 5 m/s | 0.8867–0.9026 |
| 8 m/s | 0.8631–0.8788 |

The first complete batch had 17/18 usable windows. Trial 2 (right 2 degrees,
2 m/s) had oscillating instantaneous yaw-rate estimates and remains flagged.
The independent repeat had 18/18 usable windows. Across the 17 initially
usable trials, the largest change in mean inferred angle was 0.0108 degrees.
The flagged trial repeated with gain 0.9540.
Both complete batches restored simulator settings and left no vehicle/sensor leaks.

## Interpretation and scope

The nominal mapping has the correct sign, but its motion-equivalent response
falls below the request and depends on speed and angle. It is not an exact
physical-angle calibration. Individual front-wheel readbacks and the raw
physics steering curve are recorded separately in the JSON/CSV evidence.
A bicycle-equivalent angle from yaw rate is not an individual Ackermann wheel angle.
No speed-dependent compensation is enabled by this task. A compensated mapping
would need separate held-out CARLA validation before integration.
Commands above the existing 20-degree controller cap were rejected while parked;
full-lock steering and silent actuator clipping were not dynamically tested.
No extrapolation beyond the sampled speeds/angles, tyre steady-state claim,
collision-free certification or route-following acceptance is made.

## Implementation and verification

The collection tool owns fixed-step ticks, applies travel/height guards, records
physical wheel readbacks and yaw-rate-derived steering, and restores the world.
The experiment speed servo uses proportional gain 0.35 and integral gain 0.3,
a bounded integral of +/-5, and conditional integration for saturation recovery.
The initial integral cap caused a 2 m/s settling shortfall. After correction,
slow 8 m/s settling hit the unchanged 48 m travel guard; the gain change
shortened settling sufficiently for both complete batches. Failed raw runs
remain in Downloads and are not presented as accepted calibration evidence.

I ran the Windows regression suite: 328 passed, 1 CARLA test deselected.
Live measurements are recorded in the first/ and repeat/ directories beside
this report. The original source commit is recorded in each manifest; collection
code was uncommitted during measurement. Only the live module and config hashes
are captured there, not a complete source snapshot.

## Review

Reviewer: Zhaolin. Review remains pending.
ChatGPT assisted with implementation, regression tests and analysis. I ran the
commands and CARLA experiments and inspected their reported results.
