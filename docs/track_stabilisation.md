# Track confirmation and live RGB-D evidence — BL-C-04–06

## Temporal filter

`obstacle_perception.tracking.TrackDepthFilter` consumes the existing FrameBundle
and DetectionBatch contracts without changing detector APIs, obstacle extents,
SkeletonStack's latched stop, or frozen acceptance thresholds.

Proposed implementation defaults: 2 hits in 3 supplied batches, 0.5 simulation
seconds expiry, 5 distance samples. Invalid detector batches count as misses.
Dropped frames are not invented; elapsed simulation time still expires history.
One filter owns one episode/source/model stream. Reset clears history. Wrong
episode/source/model, reversed frames/times and mismatched dimensions raise before
history changes. Integer zero is a valid tracker ID; None IDs remain untracked
unknown observations. ID/class changes start unconfirmed.

A short miss may retain confirmation but exposes no valid current distance. Expiry
is strictly greater than the timeout. Expiry and empty batches never authorise
clearance: TrackBatch.usable_for_clearance stays false. Independent fresh coverage
and hysteresis must establish clearance before HOLD can resume.

Only a current confirmed detection with valid aligned depth exposes distance.
Median history uses supported axial measurements, bounded by age and sample count;
it is descriptive axial depth, not a bumper gap or permission to ignore a smaller
current distance. FrameBundle has no source ID: callers must bind the physical
capture stream to the configured source, since metadata cannot detect upstream
mislabelled sensors. Same-ID reassignment inside the expiry interval still relies
on tracker association quality.

## Live capture

`scripts/capture_obstacle_tracks.py` reuses SynchronousWorldSession, RGBDCapture,
FrameJoiner, AlignedCapture.to_bundle(), FixedVehicleAdapter and the real local
ObjectDetector.track(). It does not create another sensor stack or feed target
geometry to inference, temporal filtering or vehicle control.

The capture requires an otherwise empty CARLA 0.9.16 Town10HD_Opt world and an
asynchronous initial world. It parks the ego, then moves a target along a checked
60 m straight road. Three separate episodes capture an approach, a three-frame
physical bus occlusion, and lateral motion plus a twenty-frame bus occlusion.
These are kinematic target sequences, not closed-loop driving/stopping acceptance.
Every sampled frame is retained, including valid detector misses and invalid
observations. Matching RGB/depth are saved as .npy, with per-frame hashes, timings,
detections and evaluation labels. JSONL rows are flushed incrementally.

The default rig is taken from the already measured BL-B-04 evidence: 640x360,
90 degree FOV, actor-relative mount (1.5, 0, 2.0) m. Capture checks the CARLA
client/server versions, map name, OpenDRIVE hash and complete rig before reusing
its axial convention. It does not recalibrate. Source evidence bytes/hashes and
the capture script are retained with each run.

BL-E-03 supplies 1.497566177392398 m/s² deceleration (including its recorded 30%
reserve) and 0.3500000052154064 s vehicle response. The proposed 4.0 m/s cruise
comes from configs/mission_baseline.json; conservative screening uses the 6.0 m
upper bound from configs/acceptance.json, without changing either configuration.
This does not add unmeasured downstream control/transport latency.

Install the project dependencies, including lap>=0.5.12 for tracking, before
capture. The script checks lap before changing CARLA settings so Ultralytics does
not install it inside a timed inference call. With CARLA running, reproduce into a
new directory:

```bash
.venv/bin/python scripts/capture_obstacle_tracks.py \
  --host 172.28.176.1 \
  --depth-evidence docs/evidence/BL-C-04-06/depth-calibration.json \
  --braking-evidence docs/evidence/BL-C-04-06/longitudinal.json \
  --output runs/track-depth-capture-v3

.venv/bin/python scripts/benchmark_obstacle_tracks.py \
  --input runs/track-depth-capture-v3/sequences.json \
  --output runs/track-depth-capture-v3/benchmark.json
```

The command uses the curated measured evidence included below; it does not require recreating calibration outputs. The model
path/hash comes from configs/models.yaml, with PerceptionConfig confidence/image
size/class selection/tracker defaults. Resolved tracker YAML, actual device,
dependency versions and capture resolution are recorded. Model/tracker history
resets per episode.

The existing owner/context managers attempt cleanup on success and failure.
The collector records owned IDs, retries destruction only for owned IDs, waits for
a fresh asynchronous snapshot, and verifies full settings and actor inventory.
Incomplete captures are retained for diagnosis and rejected by the benchmark.

## Replay schema and evaluation

Input contains origin, settings, track_policy, proposed_cruise_mps,
braking_deceleration_mps2, additional_response_s, stopping_margin_m and sequences.
Settings require confidence, image_size, tracker, tracker_settings, model_version,
model_sha256, device and versions. Each sequence contains name, episode_id,
source_id, start_sim_time_s, calibration (CameraCalibration fields), and frames.

Each frame contains frame_id, sim_time_s, total_wall_s, depth_npy, detections,
target_gap_m, target_detection_indices, and optional valid/invalid_reason.
Detections contain class_name, confidence, original-image xyxy and nullable track_id.
Live captures additionally retain rgb_npy, filter_wall_s, sensor_delay_s, artifact
hashes and evaluation-only target-box/occlusion information. A sensor failure has
null detector timing and NaN placeholder depth explicitly marked invalid; no
successful inference timing or depth observation is fabricated.

The operational filter runs before labels are computed. Labels use projected target
bounding geometry with IoU >=0.3 and valid supported depth within the target's
axial bounding interval plus 0.5 m tolerance. Nearer occluders and ambiguous
duplicate matches stay unlabelled. These are evaluation matching settings, not
acceptance thresholds. Geometry labels are not human-reviewed segmentation masks.
The gap is target rear surface minus ego front bumper along ego forward, measured
from independent actor bounding vertices. One evaluation target is used per sequence.

The report pairs first raw and first confirmed-with-valid-depth detection on the
same frames, with evaluation gap and latency from sequence start. Serial availability
is computed as max(capture time + sensor delay, previous result availability) plus
actual detector and filter wall durations. Slow earlier calls therefore remain in
the backlog. Missing events remain null. Invalid observations advance filtering as
misses and never create target events. Recorded depth hashes are verified on replay.

Screening solves v*response + v²/(2*deceleration) <= gap-margin using the first
confirmed gap and its capture-to-result response, plus measured vehicle lag.
The worst sequence cap is rounded down to 0.1 m/s, and the recommendation cannot
exceed the proposed cruise. A missing confirmation recommends 0 m/s (hold).
This is first-confirmation screening, not ongoing real-time or closed-loop safety
certification. A synchronous scripted capture does not reproduce live traffic
advancing while a slow model call blocks.

## Measured run — 20 October 2026

[Benchmark JSON](evidence/BL-C-04-06/benchmark.json) and
[capture/identity/provenance summary](evidence/BL-C-04-06/capture-summary.json).
Raw arrays, JSONL and the replay manifest remain locally in
`runs/track-depth-capture-v2/` (ignored raw evidence, approximately 1 GB).

| Sequence | Frames | First raw gap / latency | First confirmed gap / latency | Observed target IDs |
|---|---:|---:|---:|---|
| Approach | 200 | 45.25 m / 4.398 s | 39.65 m / 6.202 s | 3 |
| Short occlusion | 200 | 35.45 m / 2.685 s | 35.25 m / 2.734 s | 14 |
| Continuity risk | 200 | 43.06 m / 1.343 s | 42.86 m / 1.412 s | 8 → 31 |

600 frames were retained with zero invalid observations; misses remain in the data.
The three short-occlusion frames and twenty continuity-occlusion frames had no
target match. Short occlusion retained the target ID; the continuity case observed
one ID change. Detector: local yolo26n, confidence 0.25, inference size 640,
BoT-SORT, CUDA:0. The report records the full model SHA, resolved tracker settings
and installed package versions. Cold first-call time is included, not discarded.

Worst first-confirmation screening cap: **4.9 m/s**. Recommendation: **retain the
proposed 4.0 m/s** for this screening result; no increase is recommended and no
acceptance threshold is changed. Closed-loop HOLD/resume and stopping evidence
remain separate obligations.

Depth convention provenance:
[BL-B-04 source evidence](evidence/BL-C-04-06/depth-calibration.json).
Braking provenance:
[BL-E-03 source evidence](evidence/BL-C-04-06/longitudinal.json).
The summary records the source file hashes, OpenDRIVE hash and executed script
hashes; the benchmark records the exact input manifest SHA256.

The first attempt in `runs/track-depth-capture/` is explicitly INCOMPLETE: the
off-road bus spawn failed after one sequence. Its residual target was destroyed
and recovery recorded. The successful second run restored settings and left no
actors. Cleanup retry responses saying actor not found reflect stale proxies for
already destroyed actors; the final fresh inventory verifies cleanup. No failed
attempt is used in the benchmark.

## Verification

Focused tests cover confirmation/flicker, ID switches/expiry/None IDs, invalid
visibility/depth, association mismatches, capture provenance/schema and evaluation
occluder rejection, serial backlog, sensor delay and artifact corruption.
[JUnit evidence](evidence/BL-C-04-06/temporal-depth-tests.xml) is retained.
The full suite is run with python -m pytest so repository scripts are importable.
No commit, push, PR or issue update was performed.


Final local validation on 20 October 2026: **46 focused tests passed**;
**699 full-suite tests passed, 1 skipped, 3 subtests passed**. Repository Ruff,
Sphinx with -W and git diff --check passed. Test XML includes the focused capture,
track/depth and benchmark cases.


## Sensor-read failure semantics (PR #93 review)

Every attempted sampled source frame advances the temporal filter, including
RGB-D read/conversion failures. An invalid empty detection batch carries the
attempted episode, frame, simulation time, source and model, with the explicit
exception reason. A shape-only RGB carrier retains the captured ego pose and
calibration but has no depth; it is never passed to inference. The retained
evidence row has no detections, labels, RGB image or invented detector timing;
its depth artifact is all NaN. Replay advances the same invalid batch as a miss.
Neither loss of confirmation nor expiry authorises clearance or resume.

The live-helper/replay regression compares every TrackBatch through two
consecutive read failures and subsequent expiry after a confirmed object.
Additional tests reject mismatched episode/frame/time/source/model metadata
and verify invalid frames cannot authorise clearance. The measured 600-frame
run above contained no invalid observations, so its results are unchanged.

PR #93 review validation: **52 focused tests passed**, including RGB-D joiner
contracts; **705 full-suite tests passed, 1 skipped, 3 subtests passed**.
Ruff, Sphinx `-W` and `git diff --check` passed.
[Sensor-failure regression JUnit evidence](evidence/BL-C-04-06/sensor-failure-tests.xml).
