# BL-B-01: synchronous world lifecycle

`carla_adapter.world.SynchronousWorldSession` is a sensor-free lifecycle utility.
It does not yet produce `FrameBundle` objects or implement the final `World`
protocol; capture and full `CarlaWorld` wiring belong to subsequent tasks.

Use the session as a context manager. Opening sets synchronous mode, a 0.05 s
fixed step, physics substeps up to 0.01 s and a maximum of ten substeps. Timing
validation requires a positive finite fixed step no larger than the available
substep budget. Unrelated settings are preserved. Closing attempts full braking,
destroys only session-owned actors and restores the original settings even if
actor cleanup fails. Cleanup errors are raised; when another exception is
already active, cleanup errors are attached as exception notes on Python 3.11+.

A process-local endpoint registry rejects duplicate sessions, and only the
opening thread may tick or spawn. Opening rejects an already synchronous world:
its tick ownership is ambiguous. Other clients must not tick or modify settings
while the session is active. Consecutive-frame validation detects external
advances after the first session tick, but this is not a distributed lock and
cannot prevent races with independent processes or endpoint aliases.

`try_spawn` records ownership immediately and refuses sensor blueprints. No
sensors or Traffic Manager are enabled. Existing actors are never destroyed by
the session. On a tick timeout or frame discontinuity, full braking is attempted
on all owned live vehicles before the error is propagated. A disconnected
server can prevent control delivery; this utility cannot guarantee braking over
a broken connection. Stop failures remain available in `stop_errors`.

The module's command-line smoke test repeats three ordinary start/stop cycles
and a fourth cycle with an injected timeout while CARLA remains reachable. It
checks fixed timestep, owned actor destruction, complete actor ID inventory and
restoration of exposed scalar world settings. The injected timeout checks the
braking path; it does not claim delivery during a genuine server outage.
Run with other simulator clients stopped. Do not use a session in a world whose
synchronous tick owner is another client.

Unit tests cover restoration, repeated cleanup, owner conflicts, sensor rejection,
startup partial failure, tick timeout braking, frame jumps, thread ownership,
cleanup failure and invalid substep configurations. Live results must be recorded
before pushing. Owner in the plan: Mohammad. Reviewer: Alan. Implemented by
Abdullah with permission to cover these baseline tasks.

## Verified results

The integrated non-CARLA suite passed all 205 tests.
The live check passed on CARLA 0.9.16, Town10HD_Opt:
all four cycles restored world settings and left no actor leaks.
Simulation steps were approximately 0.05 seconds.
The fourth cycle verified full braking after an injected timeout
while the server remained reachable.

Evidence: docs/evidence/BL-B-01-world-smoke.json.
