# BL-D-02: camera and world transform primitives

Canonical implementations live in `geometry.camera`, `geometry.transforms` and
`geometry.ground_projection`. The old top-level `ground_projection` module
re-exports the same three A1 functions. Existing valid signatures and numerical
results are preserved; invalid focal lengths, dimensions, nonfinite values or
camera heights now raise. Forward projection of a point at/behind the camera
returns None instead of a physically misleading pixel or division by zero.

All core points and translations use metres; yaw and horizontal field of view
use radians. Ego X is forward, Y left, Z up, with the rear axle as reference.
World yaw is counterclockwise. Optical X is right, Y down and Z forward. CARLA's
left-handed axes are converted only in its adapter package, never in geometry.

CameraIntrinsics validates image size and focal lengths, supports separate fx/fy
and principal points, and constructs the existing CameraCalibration contract.
Pinhole depth projection requires axial optical Z; it does not silently interpret
raw CARLA depth as axial depth. BL-D-01's installed-build depth-convention check
remains necessary before decoding and associating simulator depth.

Rigid transforms validate homogeneous shape, orthonormal rotation and positive
handedness. Camera projection uses supplied mount translation/rotation. Ground
intersection uses an explicit local horizontal plane and returns None for
parallel rays or intersections behind the camera. Flat-ground IPM does not model
slopes or object surfaces; the inherited claim that every CARLA road is flat has
been removed. No accuracy or driving acceptance is claimed by round-trip tests.

The BL-D-01 worked example is checked numerically: pixel (680,310), fx=fy=400,
principal point (480,270), axial depth 10 m and camera mount (1.5,0,1.7) produce
ego (11.5,-5,0.7). At rear-axle world (100,20), yaw pi/2, world point is
(105,31.5,0.7). Tests independently check this example, multiple-heading pixel
round trips, known ground intersection, legacy import identity, metres/radians,
invalid intrinsics/depth and reflected or nonrigid transforms.

Owner in plan: Alan. Reviewer: Mohammad. No live CARLA check is required for
BL-D-02; mounting/depth integration is validated in subsequent sensor tasks.

## Verified results

All 35 geometry tests and all 262 integrated regression tests passed.
Evidence: docs/evidence/BL-D-02-geometry-tests.txt.
