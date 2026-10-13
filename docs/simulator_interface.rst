Simulator Interface and Data Contracts
======================================

Purpose
-------

This document defines the boundary between the simulator and the reusable
perception and navigation library. Simulator-specific APIs must remain inside
an adapter so that the core library can be reused with a different simulator
or vehicle platform.

Boundary
--------

The simulator adapter is responsible for converting simulator data into the
library's standard data types and for applying the library's controller output
to the simulator vehicle.

::

   simulator telemetry -> adapter -> VehicleState -> navigation controller
   navigation ControlCommand -> adapter -> simulator vehicle controls

The core perception and navigation modules must not import simulator-specific
packages or depend on simulator objects.

Shared Navigation Contracts
---------------------------

The navigation controller uses the existing public types in
``navigation_control.types``:

* ``VehicleState(x_m, y_m, yaw_rad, speed_mps)`` represents the vehicle's
  rear-axle position, heading and forward speed in the shared planar world
  frame. Position is measured in metres, yaw in radians, and speed in metres
  per second.
* ``ControlCommand(steering_angle_rad, throttle, brake)`` represents the
  controller output. Positive steering denotes a left turn. Throttle and brake
  are each within ``[0, 1]`` and must not both be positive.

Adapter Responsibilities
------------------------

The simulator adapter must:

* read simulator vehicle telemetry;
* transform simulator position, heading and speed into ``VehicleState`` using
  the documented shared coordinate convention;
* translate ``ControlCommand`` into the simulator's steering, throttle and
  brake API values; and
* keep simulator imports and simulator-specific coordinate conversions out of
  the core library.

The navigation controller remains responsible for generating
``ControlCommand`` values from ``VehicleState`` and the planned path.

Deferred Decisions
------------------

This initial contract does not select a simulator or define camera, lidar,
perception-result, timing or sensor-synchronisation interfaces. Those details
will be added when the selected simulator harness and perception integration
are implemented.

Verification
------------

The contract will be verified through adapter tests that use representative
telemetry and commands, without requiring core navigation modules to import
the simulator SDK.
