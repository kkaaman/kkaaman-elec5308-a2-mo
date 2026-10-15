# BL-E-02: steering-response measurement

Owner Abdullah; reviewer Zhaolin. This tool collects measurements; calibration
is not accepted or applied to the adapter until live data and quality are reviewed.

The site probe found spawn 152 on road 10, lane -2, with a 62 m straight corridor
and 3.5 m lane width. Map geometry is not an obstacle-clearance guarantee. Use an
isolated dry Town10HD_Opt with no other tick owner or vehicle/sensor actors.
The runtime rechecks forward geometry and bounds travel to 48 m and lateral
movement to 1.4 m from the initial straight heading. Height/tilt and forward-only
state guards reject unsuitable motion. No collision sensor is enabled; these
bounds are not proof of no contact with static scene objects.

Each trial spawns a fresh Tesla Model 3 and uses the BL-B-01 synchronous owner
at 0.05 s per tick. Physics settings are restored on context exit. After settling,
the existing adapter validates wheel geometry. A stationary over-cap command is
required to raise, demonstrating rejection rather than silent clipping. The
speed servo stabilizes within 0.2 m/s for ten ticks, then requests a fixed angle
for one second. Full braking is issued and the vehicle destroyed after each
trial; this does not measure physical stopping performance (BL-E-03).

Smoke collection uses 2 m/s and +/-4 degrees. Full collection uses +/-2, +/-4 and
+/-6 degrees at 2, 5 and 8 m/s: 18 independent short trials. The rear-axle
longitudinal speed, signed core yaw rate, requested angle, actual normalized
control, physical front-wheel angles, pedals and displacement are logged per tick.
Raw physical steering limits and speed-dependent steering curve are archived.
No unit conversion of the curve's speed axis is assumed by this measurement tool.

Core inferred angle is atan(wheelbase * yaw_rate / forward_speed), with core
positive yaw/steering left. CARLA angular velocity Z is converted from degrees/s
and its sign reversed. Individual wheel angles remain in their raw reported
degree/sign convention to avoid assuming they equal a bicycle-equivalent angle.
CARLA 0.9.16 documents `get_wheel_steer_angle` as a physical rather than visual
angle: https://carla.readthedocs.io/en/0.9.16/python_api/#carla.Vehicle

Analysis uses the final 0.6 s, retaining samples within 0.25 m/s of the target.
At least six samples, a consistent turn sign and limited inferred-angle variation
are needed for USABLE_SHORT_WINDOW. This is not proof of steady state or a tyre
model. Output includes response gain and spread per speed/sign/angle and a plot
of requested versus inferred angles. No gain is applied automatically, and no
extrapolation to full lock or unmeasured speeds is justified. Live inspection must
compare measured wheel angles and inferred response to requested steering and
raw physics settings before declaring the adapter scale validated.

Run after an editable install:

```powershell
.\.venv\Scripts\python.exe scripts/calibrate_vehicle.py --smoke --follow --output "$HOME\Downloads\Group7_BL_E02_smoke01"
.\.venv\Scripts\python.exe scripts/calibrate_vehicle.py --follow --output "$HOME\Downloads\Group7_BL_E02_full01"
```

Output folders must be new. All raw outputs and plots go into the selected
Downloads folder: samples.csv, manifest.json, calibration.json, steering_response.png.
If collection aborts, retain partial CSV/manifest and fix the cause before retrying
with a new folder. `--analyze <folder>` regenerates analysis/plot without CARLA.
A COLLECTED status means acquisition/cleanup succeeded, not that calibration is
validated. Review-required windows and limitations must remain visible in evidence.
