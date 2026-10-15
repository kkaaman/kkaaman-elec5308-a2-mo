# Baseline batch: capture, junctions, depth and offline fixtures

## Task ownership and scope

This batch includes G0-01, BL-B-02, BL-A-03, BL-D-03 and BL-C-03, plus
required BL-C-02 detector lifecycle and RV-04 compatibility dependencies.
BL-E-02 steering measurements remain a separate commit in the combined PR.
No learned model is invoked by unit fixtures. No automatic driving is enabled.

## Capture (BL-B-02)

`carla_adapter.sensors.RGBDCapture` owns matching rigid RGB/depth cameras with
same resolution, FOV and zero lens distortion. Callback queues are bounded;
only exact frame IDs and matching simulation timestamps join to ego state.
The owner records ego immediately after its fixed-step tick; delayed camera
callbacks can join that saved snapshot after subsequent ticks. Callbacks do
not tick or read current ego state. Timeout requests full braking and retires
the frame. Frames cannot cross episodes or be reused after retirement.

The live check parks a Tesla, records 24 aligned frames with a four-frame
pipeline, injects a missing observation, checks full brake, then destroys
sensors/vehicle and restores settings. It exports a raw sample, not a depth
accuracy claim. Other clients must not tick or change world settings.

Depth decoding follows CARLA 0.9.16's sensor reference:
https://carla.readthedocs.io/en/0.9.16/ref_sensors/
B,G,R,A bytes encode 1000*(R+256G+65536B)/(2^24-1).
Zero and saturation become NaN. Raw decoded depth remains in AlignedCapture;
conversion to FrameBundle requires explicitly established axial/radial mode.
The off-axis installed-build convention test remains BL-B-04 work, as specified
in depth-association.md. No operational clearance depends on unverified depth.

## Junctions (BL-A-03)

`junction_exits` enumerates all explicit legal successors through junction
sections, then classifies the outgoing lane heading relative to approach.
Straight is within 25 degrees; left/right lie between 45 and 135 degrees;
intermediate angles and U-turns fail as unsupported. Multiple exits for the
same requested maneuver fail as ambiguous. Cycles and traversal excess fail.
`stitch_route` retains interior lane geometry and explicit short topology
connections, rejects illegal transitions and >30 degree boundary tangent jumps.
Install the resulting immutable path once; no per-frame controller reset.
T/4-way fixtures include turns whose first junction segment starts straight.
This is route geometry, not a full mission executor or live tracking acceptance.

## Depth and boxes (BL-D-03 / RV-04)

Legacy object/traffic BoundingBox constructors inherit validated PixelBox.
TrafficDetection and ModelPrediction signatures remain unchanged; box adapters
never call a detector. Nonfinite, negative or inverted coordinates fail.
Boxes may extend beyond image bounds and are clipped before ROI sampling.
Central half-width/half-height is used; require 9 valid pixels and 60% support.
Depth must be finite, positive and below 100 m. P90-P10 spread exceeding
max(1 m, 10% median) rejects potential mixed surfaces. These are explicit
baseline policies, not experimentally optimized thresholds. Minority surfaces
below the robust quantile support can still be missed; no free-space guarantee
is made. Output is a supported surface point, not object centre/full extent.
Ground and known-size fallbacks are explicitly invalid/unvalidated and cannot
permit clearance. Old scalar traffic pinhole distance remains a legacy helper.

## Offline fixtures (BL-C-03)

`configs/models.yaml` is JSON-compatible YAML and pins the known local YOLO
asset SHA256/size. Package versions are captured by the offline validation
script, not guessed. Synthetic JSON fixtures explicitly label their origin;
no claim of real recorded detection accuracy is made. RecordedDetector supports
validated recorded/synthetic sequences, reset, detect/track and close; absent
frames are errors, valid empty differs from invalid/missing. Metadata and
source dimensions/time are checked at load and replay. Unit timing tests use
fakes. Real model check records construction, first call and subsequent warm
calls separately. It verifies local weights before initializing YOLO.

## Validation commands

Run offline tests before the live commands. Live outputs must use a new folder
under Downloads. Keep all raw ZIP exports in Downloads. No PR may describe
live checks as passed before their actual reports have been inspected.

## Local author checks (15 October 2026)

391 tests passed, 3 subtests passed. Two tests were deselected: live CARLA and
`test_model_path_finds_repository_yolo_weights` (the uploaded source archive
excludes model binaries). The repository weight check must run normally on the
university PC. Local author environment: Python 3.12.14, CPU PyTorch 2.5.1;
the user's Windows environment/version results must be retained separately.
Ruff checks are run on all changed Python files. Windows verification completed subsequently: 394 tests passed, 1 CARLA test
deselected, 3 subtests passed. The live aligned capture and offline detector
checks passed; see docs/evidence/baseline-batch/. Junction geometry audit accepted
60 routes and rejected 11 ambiguous configurations. Gate approval and raw-depth
convention/accuracy acceptance remain pending.
