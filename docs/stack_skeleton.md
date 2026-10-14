# BL-F-02: validated configuration and runnable stack skeleton

`autonomy.stack.SkeletonStack` implements the BL-F-01 reset/step/close protocol.
Its detector implements BL-C-01. The default motion policy returns full brake:
this skeleton never claims working navigation or obstacle avoidance.

`configs/stack.yaml` uses JSON syntax, which is a valid YAML subset. The loader
accepts that subset only, with no additional dependency. Unknown fields, wrong
types, nonfinite/negative timing, missing files and malformed data latch emergency
stop. Command TTL must not exceed observation-age allowance. The supplied depth
requirement is false for fake observations; depth-dependent association must
require depth when wired in a subsequent task.

Each step validates frame/ego state, episode, increasing frame IDs and simulation
time, local monotonic receipt age, optional depth and detector source metadata.
Invalid input or an inference/policy exception returns a full-brake ControlPacket
with a reason and stage timings. Emergency stop remains latched until explicit
reset; bad configuration requires a corrected stack instance. The default hold
reason is `skeleton_hold`. An injected callable motion policy may return an
existing ControlCommand and optionally implement reset/close hooks. A stage-time
budget or observation expiry after processing also selects full brake. These
checks occur after synchronous calls return; they cannot interrupt a hung model.

Control packets carry episode/source frame/time and finite simulation expiry.
The fake world rejects foreign, expired or wrong-frame commands and selects full
brake. Physical actuator dispatch remains a world adapter responsibility. The
stack does not own ticks or import CARLA or obstacle truth.

`sim_harness.skeleton.ContractWorld` supplies stationary fake RGB observations;
it is a protocol fixture, not the planned kinematic simulator. EmptyDetector
supplies same-frame, valid empty detections without loading weights or downloading
anything. Running the module emits one complete fake world/stack cycle as JSON.
No live CARLA test is required by the board for BL-F-02.

Tests cover emitted commands/reasons/timing, bad config, invalid state, stale
observations, missing depth, cross-episode results, duplicate frames, emergency
latching/reset, detector faults, policy errors, expiry and cleanup hooks.
Original owner: Saviour. Reviewer: Ricky.

## Verified results

The integrated non-CARLA suite passed all 227 tests.
The fake-world smoke run emitted a finite full-brake command,
skeleton_hold reason, stage timings and the next observation frame.
Evidence: docs/evidence/BL-F-02-skeleton-smoke.json.
