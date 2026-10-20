# Baseline behaviour guards

This document defines the conservative obstacle-clearance rules introduced by
BL-D-05. It does not define the complete stop/hold/resume state machine; that
belongs to BL-D-08.

## Route-corridor boundaries

`planning.corridor.RouteGeometry` intersects obstacle footprints with swept
route segments and round joins at interior vertices. The joins include the
outside of left and right turns, including footprints whose edges intersect a
join even when all their corners lie outside it. A join contributes its vertex's
arc coordinate and obeys the requested arc interval.

The route goal is a rear-axle position. `gap()` therefore extends the final
segment along its tangent by `front_bumper_from_rear_axle_m` to cover the nose
at the goal. It then subtracts that offset exactly once to obtain the bumper
gap. An obstacle beyond the last waypoint can consequently have a negative gap
when it overlaps the nose.

Direct `footprint_interval()` calls keep the original route-length boundary by
default. Callers may request a finite nonnegative `end_extension_m`; returned arc
coordinates can then exceed `length_m`. An explicit `maximum_s` still clips the
extension.

## Clearance semantics

A route that has previously been blocked remains blocked until fresh perception
evidence confirms that the corridor is clear.

The following conditions are therefore not sufficient to authorize motion:

- an obstacle track expiring;
- an obstacle identifier disappearing;
- a stale observation;
- an invalid or unavailable perception frame;
- sensor or processing failure.

In particular:

> Track expiry is not evidence of clearance.

Once a blocker has been observed, clearance requires both:

1. a configured number of consecutive fresh, valid observations reporting no
   obstacle in the route corridor; and
2. a configured minimum clear duration.

If either condition is interrupted by a new obstacle or an invalid observation,
the accumulated clearance evidence is reset.

## Initial clear route

Before any blocker has been observed in the current episode, a fresh valid
observation reporting no obstacle may be treated as clear immediately.

This avoids imposing unnecessary hysteresis on normal cruising while preserving
conservative behaviour after a genuine blockage has been observed.

## Separation of responsibilities

`planning.behaviour.ClearanceGuard` answers only whether sufficient fresh
evidence exists to consider a previously blocked corridor clear.

It does not decide vehicle behaviour states such as:

- CRUISE;
- STOPPING;
- HOLD;
- RESUME;
- FINISHED; or
- FAULT_STOP.

Those transitions are implemented separately by BL-D-08.

Emergency-brake precedence and speed limiting are also handled separately by
the BL-D-07 speed governor.
