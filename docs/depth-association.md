# RGB-D association specification — BL-D-01

Owner: Alan. Reviewer: Mohammad. Status: proposed; review pending.
This specifies behaviour for BL-B-02, BL-D-02 and BL-D-03; no live sensor
implementation or depth-accuracy result is claimed by this task.

## Decode and align

CARLA's raw camera buffer is BGRA uint8. RGB is channels [2,1,0], with alpha
removed. Never pass raw BGR/BGRA into the canonical lane or semantic models.
Decode depth before any display converter is applied. With raw bytes B,G,R,A:

`encoded = R + 256*G + 65536*B`
`depth_value_m = 1000 * encoded / 16777215`

Promote channels to uint32 before arithmetic to prevent uint8 overflow.
Zero and saturated 1000 m pixels are invalid for baseline obstacle association;
represent them as NaN. Also reject nonfinite values and values outside the chosen
validated sensor range. Do not change a valid far measurement into free-space
proof simply because it is outside the association range.

Source: https://carla.readthedocs.io/en/latest/ref_sensors/ (Depth camera).
Checked 14 October 2026; installed client/server in this project is 0.9.16.

Use matching RGB/depth resolution, FOV, mount and capture rate. Join exact frame
IDs and simulation capture timestamps within one episode, with ego from the same
world snapshot. Queue bounded frames; never combine nearest-by-arrival frames.
If cameras differ in mount, calibration or dimensions, explicitly reproject with
occlusion handling instead of silently pairing pixels.

## Axial depth versus ray range

The core FrameBundle contract requires optical axial Z depth. CARLA documentation
supplies the encoding but describes distance/depth-buffer terminology without a
sufficient off-axis numerical example to establish the installed build's behaviour.
Do not assume raw values are radial range and apply an unconditional correction.
Any pre-existing documentation that assumes raw CARLA depth is radial range
must be treated as provisional until the mandatory live convention test below
is completed. BL-D-02 must use the convention established by that test rather
than silently applying a range-to-axial conversion.

For normalized pixel ray `(a,b,1)`, where a=(u-cx)/fx and b=(v-cy)/fy:
- Axial Z: optical point is `(a*Z, b*Z, Z)`.
- Radial r: first compute `Z=r/sqrt(1+a*a+b*b)`.

Mandatory live convention test before accepting BL-B-04 sensor accuracy: use a
known flat surface normal to camera forward, sample centre and off-axis pixels,
and compare reconstruction against evaluation-only geometry. At a=0.5,b=0,Z=10,
axial depth is 10 while ray distance is 11.1803398875. Log both hypotheses and
select only the one supported by measured geometry, including a quantization
and surface tolerance. Do not use evaluation truth as an operational obstacle input.

## Optical, ego and world coordinates

Optical coordinates: X right, Y down, Z forward. Ego: X forward, Y left, Z up.
For an unrotated camera mounted at rear-axle-relative `(tx,ty,tz)`:

`p_ego = (Z+tx, -X+ty, -Y+tz)`

For a general mount use the calibrated rigid camera-to-ego 4x4 transform,
including pitch/roll/yaw. A mount measured relative to actor origin must first
be converted to the rear-axle reference using verified vehicle geometry.

Worked example: fx=fy=400, cx=480, cy=270, pixel (680,310), axial Z=10 m.
Optical point `(5,1,10)`. Mount `(1.5,0,1.7)` gives ego `(11.5,-5,0.7)`.
Rear axle world `(100,20,0)`, yaw +pi/2, gives world `(105,31.5,0.7)`.
This example tests sign, height, translation and rotation; pixel right becomes
negative ego Y. CARLA world Y reflection belongs exclusively to the adapter.

## Boxes, footprints and corridor relevance

Boxes use original-image XYXY continuous coordinates with exclusive maximum
edges. Clip to image bounds, shrink to an explicitly configured inner region,
require sufficient finite support and report support/spread. The inner-region
ratio, valid-pixel fraction and temporal filtering parameters are set and tested
in BL-D-03; do not invent defaults in this specification.

Use robust supported foreground depth, not the closest arbitrary noisy pixel
or a box-wide median dominated by background. Depth surfaces do not directly
measure object centre/full extent; mark extent assumptions and uncertainty.
Obstacle extents are full lengths, with yaw. A planar footprint is the four
rotated XY corners of the centred length/width rectangle. Corridor membership
uses polygon overlap with the swept vehicle corridor and uncertainty margin,
not only object-centre distance from the route. A rotated vehicle or pedestrian
at the edge can intersect even when its centre is outside.

Stopping gap is bumper-to-obstacle surface along the relevant route corridor,
not distance from camera, rear axle or obstacle centre. Vehicle front overhang
and obstacle extent must be accounted for. Controller planning uses perceived
geometry; acceptance measurement uses independent evaluation truth.

## Missing-depth policy

Missing, stale, wrong-episode, mismatched, invalid or insufficiently supported
required depth produces an invalid observation with a reason. No zero distance,
valid-empty result or previous-frame reuse may manufacture clearance. Request
controlled stop and retain HOLD until fresh valid clearance meets configured
coverage and hysteresis requirements. A tracker expiring is not proof of clearance.
Any ground-plane fallback must carry explicit validity and uncertainty and pass
its own acceptance tests; it is not silently enabled by this task.
