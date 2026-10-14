# Detector and obstacle contracts — BL-C-01

Status: implementation proposed; reviewer Saviour; owner Ricky.
Uses the BL-F-01 FrameBundle and right-handed rear-axle ego convention.

## Detector lifecycle

The shared `Detector` protocol takes FrameBundle for both `detect` and `track`.
Both return DetectionBatch. Local and remote fake implementations are exercised
against the same protocol. No real detector, cloud transport or actor lookup is
introduced by this task.

`reset(episode_id=...)` clears all tracker history, frame-order state and ID
allocation. Calls require that episode. `detect` does not advance tracker history;
`track` accepts increasing frame IDs within the episode. Concrete episode-reset
and tracker-selection implementation belongs to BL-C-02. `close` is idempotent.
Existing A1 ObjectDetector detect/track APIs are unchanged; a concrete adapter to
this frame-based protocol will preserve them in BL-C-02.

## Observations and identities

| Field | Meaning |
|---|---|
| episode_id | New identity per reset; never reuse IDs across episodes |
| frame_id / sim_time_s | Original sensor frame/time, not remote completion time |
| source_id | Stable sensor/inference stream name, not a CARLA actor ID |
| model_version | Explicit checkpoint identity or release version; no unspecified default |
| track_id | Optional nonnegative tracker-local integer; zero is valid |
| TrackIdentity | Episode + source + model version + track ID |
| PixelBox | Continuous XYXY pixels in original image; max edges exclusive |
| confidence | Finite [0,1], subject to later calibrated thresholds |
| CallTiming | Total caller wall duration; optional inference and transport durations |

No remote absolute monotonic timestamp is compared with a local timestamp.
CallTiming stores durations only. Total includes local waiting/processing; optional
remote inference duration need not sum exactly with transport to that total.
Detect and track each return their own per-call timing.

Duplicate track IDs in a batch, invalid boxes, nonfinite confidence and missing
model versions fail clearly. Batch image dimensions refer to the original sensor
image; resize/letterbox reversal belongs to the concrete detector adapter.
PixelBox is the new boundary type. The A1 BoundingBox is retained unchanged;
compatibility conversions/unification belong to RV-04.

Valid empty detections mean processing succeeded with no detections. Invalid
batches require a reason and cannot contain detections. A timeout must not be
converted to a valid empty batch. Neither case proves that the road is clear.
The stack must also apply sensor completeness and freshness policy.

## Obstacles and association

Obstacle contains source-frame metadata, class, confidence, optional track ID,
centre in rear-axle ego coordinates, full XYZ lengths and yaw. +X forward,
+Y left, +Z up; distances are metres and yaw is counterclockwise radians.
Extents are full lengths, not CARLA half extents, and must be positive.
Sources may include camera-depth or LiDAR inference; obstacle ground truth is
reserved for evaluation and must never construct core perceived obstacles.

ObstacleSource.observe(frame, detections) requires matching episode, frame and
simulation capture time. Wrong-frame or missing required depth must raise a
clear error for safety arbitration, never return an apparently clear road.
Return an empty tuple only after valid processing. Depth policy and numerical
association definitions are elaborated by BL-D-01 and BL-D-03.

## Verification

Ten new unittest cases passed in the development environment, including fake
local/remote conformance, reset/frame ordering, track namespace isolation,
missing-depth/frame mismatch, invalid versus empty results and input bounds.
These fixtures demonstrate the contracts; they do not prove a real detector
implements those lifecycle rules. Project-PC lint and full pytest are required
before pushing. CARLA is not required because this task defines core contracts.

Protocol import path: `autonomy.detector_contracts`. The previously reviewed
`autonomy.interfaces` file is left unchanged. Sensor output types are defined
in `obstacle_perception.types`.
