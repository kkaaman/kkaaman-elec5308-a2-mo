# BL-C-02: offline detector lifecycle

A1 `ObjectDetector(config)`, `detect(RGB)` and `track(RGB)` remain available and
return tuples of the original Detection type. RGB inputs must now be nonempty
uint8 HxWx3. Added `reset(episode_id=...)`, `close()` and `last_call_timing`.
The legacy image API is not itself the FrameBundle Detector protocol; a future
integration adapter must wrap source metadata into DetectionBatch.

`PerceptionConfig` adds `tracker_config`, default `botsort.yaml`; `bytetrack.yaml`
is also selectable. Local custom YAML must specify a supported tracker type.
ReID is disabled for the baseline so custom configuration cannot silently load
an additional embedding model. Confidence, image size and class selection are
validated. Builtin tracker configuration is supplied by installed Ultralytics.

Model names now resolve to an existing, nonempty local .pt file before YOLO is
constructed. Missing weights, URLs interpreted as nonexistent local paths and
Git LFS pointers fail without requesting a model download. The old default
`yolo26n.pt` therefore works only when that local asset already exists. Supply an
explicit absolute model path when running from another working directory. This
changes missing-asset behaviour intentionally while preserving successful APIs.
Checks validate local availability, not model checksum or model accuracy.

Reset creates a fresh YOLO backend from the same local weights, discarding all
predictor and tracker history without private tracker internals. It is more
expensive than resetting a tracker in place, but occurs once per episode rather
than per frame. A failed reset makes the old backend unavailable. Track IDs can
restart numerically; consumers must scope them by episode as defined in BL-C-01.
Close releases this instance's backend reference; it does not promise to clear
process-global GPU caches. Use one caller per detector instance.

Per-call CallTiming records total wall duration and the predict/track call span,
including error cases. Total includes validation, RGB/BGR conversion and output
conversion. The call span is host-side timing, not a standalone GPU kernel
measurement; output conversion is included in total. Transport duration is absent
for this local detector. Reset clears the previous call timing.

Tests use local placeholder files and stub model factories: they cover old API
results/RGB handling, backend isolation, selected trackers, offline failure before
YOLO construction, invalid config/images, timing and failures. Placeholder files
are never loaded by real YOLO. Learned-model inference and tracking accuracy need
separate asset-backed checks; no live CARLA test is required for this task.
Owner in plan: Ricky. Reviewer: Saviour.
