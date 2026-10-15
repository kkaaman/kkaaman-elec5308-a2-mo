# BL-E-01: fixed Tesla geometry and actuator adapter

The baseline uses `vehicle.tesla.model3` on CARLA 0.9.16. The new
`carla_adapter.vehicle.FixedVehicleAdapter` leaves inherited A1 adapter
APIs available. Construct it after the simulator has published updated vehicle
physics and pose; a stale, zero or ambiguous geometry sample raises.

Wheel positions were verified as world centimetres by transforming them into
actor-local metres at approximately 0.159 and 90.159 degrees. Wheelbase readings
were 3.00464778 and 3.00464655 m; rear axle offsets were 1.38640756 and
1.38640519 m. Both samples and vehicle cleanups passed. Front steering settings
were approximately 70 degrees, with rear steering zero.

The state uses the rear axle, metres and counterclockwise yaw in a right-handed
world: x is CARLA x, y is negative CARLA y, yaw is negative CARLA yaw. Forward
speed is the longitudinal planar velocity projection. Reverse motion beyond
0.05 m/s raises instead of being converted to positive speed; the tolerance
covers small stationary numerical motion. Tilt beyond five degrees is rejected.

Positive core steering means left. Live CARLA input -0.2 produced a -2.87514
degree CARLA yaw change (positive core yaw); +0.2 produced +2.92169 degrees
(negative core yaw). Peak speeds were about 2.34 m/s. Both direction checks and
cleanups passed. These are direction checks, not collision or route acceptance.

The physical 70-degree configuration and provisional 20-degree controller cap
are separate. Nominal normalization divides the requested angle by the physical
limit and reverses its sign. It does not divide by the controller cap. Commands
above that cap raise. This mapping does not establish achieved wheel angle or
account for speed-dependent steering response: BL-E-02 must calibrate dynamic
magnitude and tune the cap before driving acceptance. The JSON configuration is
a recorded specification; the adapter currently accepts its cap explicitly and
does not implement a configuration loader (BL-F-02).

`apply()` validates state and commands; on failure it attempts full braking and
re-raises. `stop()` applies full brake. A disconnected simulator can prevent
braking; timeout/watchdog and tick ownership belong to the world lifecycle task.
The adapter never ticks, spawns, destroys or resets world settings.

Live raw evidence is recorded separately under `docs/evidence/` after copying
and inspecting the generated JSON files. Synthetic tests cover multiple headings,
units, rear axle position, reverse velocity, physical/capped steering separation,
nonfinite inputs, reflected transforms and invalid wheel layout. Live probes so
far inspected CARLA directly; a live run of this installed adapter remains
required before pushing. Reviewer: Zhaolin.

## Final verification

The installed adapter smoke check passed in CARLA 0.9.16 on Town10HD_Opt.
I also visually confirmed left and right turns, braking after each turn,
and vehicle cleanup using the installed adapter.

The integrated non-CARLA suite passed all 189 tests, and Ruff passed.
Evidence: BL-E-01-heading-check.json, BL-E-01-steering-check.json and
BL-E-01-adapter-smoke.json under docs/evidence/.

Dynamic steering magnitude calibration remains BL-E-02.
