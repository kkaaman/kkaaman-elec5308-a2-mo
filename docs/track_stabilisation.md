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
Live captures additionally retain rgb_npy, filter_wall_s, sensor_delay_s,
sensor_read_wall_s (required for schema 3), artifact
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
actual sensor-read/conversion, detector and filter wall durations. Slow earlier calls therefore remain in
the backlog. Missing events remain null. Invalid observations advance filtering as
misses and never create target events. Recorded depth hashes are verified on replay.

Screening solves v*response + v²/(2*deceleration) <= gap-margin using the first
confirmed gap and its capture-to-result response, plus measured vehicle lag.
The worst sequence cap is rounded down to 0.1 m/s, and the recommendation cannot
exceed the proposed cruise. A missing confirmation recommends 0 m/s (hold).
This is first-confirmation screening, not ongoing real-time or closed-loop safety
certification. A synchronous scripted capture does not reproduce live traffic
advancing while a slow model call blocks.

## Historical measured run — 20 October 2026 (timing screen superseded)

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

The original first-confirmation screening cap was **4.9 m/s**, with a recommendation
then to retain **4.0 m/s**. That screening conclusion is **superseded by the
incomplete-timing audit below**: read waits were not instrumented, so the original
report is retained as historical evidence, not a current measured speed approval.
Closed-loop HOLD/resume and stopping evidence remain separate obligations.

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


## Sensor-read wall timing correction (PR #93 follow-up)

**Historical capture screening: BLOCKED_INCOMPLETE_TIMING.** The fresh schema-3
run and conservative range/speed interpretation are recorded in
[fresh-capture evidence](evidence/BL-C-05-refresh/README.md).
[Versioned timing review](evidence/BL-C-04-06/sensor-wait-timing-review.json)
retains the original report, audited replay and deterministic regression result.
The original benchmark JSON is unchanged so the historical result remains visible.

The live helper calls the existing join/read path with a **5.0 s** timeout.
FrameJoiner waits on its condition until the monotonic deadline; failures call
emergency stop and raise. Early retired-frame/alignment failures can return much
sooner. The configured timeout is not an observed duration and does not bound
all conversion/stop-callback overhead. No fixed five-second penalty is fabricated.

Schema **3** records `sensor_read_wall_s` on **every** attempted frame, including
successful reads and failures. A host `perf_counter()` interval spans read plus
axial-depth conversion, including exception/stop handling, and ends before the
tracking call. `total_wall_s` remains actual detector duration (null when no call
occurred); `filter_wall_s` remains separate. `sensor_delay_s` is the existing
simulator scheduling lag, not a host-wall timeout measurement. Episode/frame/time,
source/model context, invalid reason and failed NaN depth artifacts remain retained.
Invalid batches still advance the temporal filter and can never establish clearance.

Replay advances serial availability by:

`ready = max(source_elapsed + sensor_delay_s, previous_ready)`
`        + sensor_read_wall_s + detector_wall_s + filter_wall_s`

The read interval is charged exactly once, including on invalid frames. The first
confirmed event reports `ready` relative to sequence start. Its response duration
is `ready - source_elapsed` for that confirmed frame. The speed formula uses that
response plus the separately measured BL-E-03 vehicle lag once; it does not add
sequence-start latency again. Range remains the evaluation gap at the source
frame. This is serial first-confirmation screening, not a full live controller or
physical stopping certificate.

Legacy files remain replayable for temporal diagnostics. Missing read timing is
counted explicitly; diagnostic latencies are labelled **legacy lower bounds**,
`measured_speed_cap_mps` is null, cruise support is unknown, and the screening
recommendation is HOLD (0 m/s) until measured timing is available. That output
does not change the proposed 4.0 m/s baseline or acceptance thresholds. Schema 3
rejects missing timing, and all schemas reject negative/nonfinite/bool durations.

### Historical evidence audit and original recapture status

This section records the earlier timing-review state. The 21 October fresh
capture below supersedes its recapture-unavailable status; the old dataset
remains invalid for measured-wait screening.

The retained 600-frame capture contains **zero invalid frames**, but **all 600
lack measured read duration**. Consequently there is no measured timeout penalty
to reconstruct, and successful read/conversion durations are also unknown.
The old numeric events remain lower-bound diagnostics, not corrected measurements:

| Sequence | Historical first-confirmation latency | Corrected measured latency |
|---|---:|---|
| Approach | 6.202 s | Unavailable; recapture required |
| Short occlusion | 2.734 s | Unavailable; recapture required |
| Continuity risk | 1.412 s | Unavailable; recapture required |

Old screening cap: **4.9 m/s**. Corrected measured cap: **unavailable**, screening
**BLOCKED**. Retaining 4.0 m/s is **not established by corrected timing evidence**;
this audit does not claim that a new live measurement has proved it unsupported.
The earlier 4.9 result is retained only as an optimistic cap calculated from
lower-bound latency; it is not a conservative speed bound.

Fresh live capture is required and was **not performed**: a WSL TCP check of
`172.28.176.1:2000` timed out. No synthetic result is substituted for CARLA evidence.
With the server available, run from this branch with the existing local model:

```bash
PYTHONPATH=src .venv/bin/python scripts/capture_obstacle_tracks.py \
  --host 172.28.176.1 \
  --depth-evidence docs/evidence/BL-C-04-06/depth-calibration.json \
  --braking-evidence docs/evidence/BL-C-04-06/longitudinal.json \
  --output runs/track-depth-capture-timed-v3
PYTHONPATH=src .venv/bin/python scripts/benchmark_obstacle_tracks.py \
  --input runs/track-depth-capture-timed-v3/sequences.json \
  --output runs/track-depth-capture-timed-v3/benchmark.json
```

Capture verifies CARLA 0.9.16 / Town10HD_Opt and unchanged rig/convention provenance;
it records model/hash, 640 inference size, confidence 0.25, resolved BoT-SORT,
device/dependencies and unchanged 2-of-3 track policy. Preserve the new run alongside
the old run, and version its benchmark/provenance before restoring a speed claim.

### Deterministic regression evidence

Two consecutive failed reads wait **0.5 s** and **1.0 s**. The serial availability
at each frame is **0.125, 0.750, 1.750, 1.875, 2.000 s**. The final confirmation
therefore occurs at exactly **2.000 s**, versus the old omitted-wait **1.125 s**.
Its source-relative response is **1.000 s**, and the fixture's corrected speed cap
is **3.0 m/s** (synthetic parameters only). Tests independently apply the speed
formula to this corrected response. The live helper's fake monotonic clock also
proves measured failure waits survive serialization and per-frame temporal replay;
read timing excludes detector/filter work, and invalid frames never clear HOLD.


Timing-follow-up validation: **60 focused tests passed** (including RGB-D joins),
**713 full-suite tests passed, 1 skipped, 3 subtests passed**. Ruff, Sphinx `-W`
and `git diff --check` passed.
[Timing regression JUnit](evidence/BL-C-04-06/sensor-wait-tests.xml).

## Fresh-capture follow-up

On 21 October 2026 the unchanged schema-3 collector captured **600 fresh frames**
(200 approach, 200 short occlusion, 200 continuity risk) on verified CARLA 0.9.16 /
Town10HD_Opt with the matching RGB-D rig. All 600 reads are valid and timed;
457 target observations are current, confirmed and joined to valid depth.
The earlier connection timeout was an invalid setup attempt (CARLA not started),
not a BL-C failure, and is excluded from operational timing.

Fresh first-confirmation latencies are **5.303253 / 2.587205 / 0.882718 s**;
source-relative responses are **3.903253 / 0.237205 / 0.232718 s**. Actual read
wait/conversion is charged once, including any failure (none occurred in this
run). Median/max measured read duration is **0.595740 / 7.419415 ms**. Cold-start
backlog is retained, not discarded in favour of later warmed sequences.

The fresh benchmark's first-event-only cap is **5.5 m/s**, distinct from the
withdrawn historical 4.9 m/s result. Its first-confirmation axial depths exceed
the reviewed 25 m scope. Within that scope the common sampled confirmed bumper
gap is **24.013027 m**, not an assumed 25 m bumper gap. Combining this range with
the worst measured source-relative response **4.199442 s**, BL-E vehicle lag once,
unchanged deceleration and 6 m margin gives **3.205373 m/s**, rounded down to
**3.2 m/s**. **4.0 m/s is not supported by this conservative BL-C screen**:
it requires **29.539768 m**. This does not certify 3.2 m/s operational stopping.

The fresh measured-wait evidence satisfies BL-C-05's capture/reporting scope.
[BL-D re-review](baseline_threshold_freeze.md) confirms cold backlog cannot
supply a new positive operational decision, but the current all-track policy
admits none of these 600 source rows, even granting fresh/warm status. The
24.013027 m target-only range therefore cannot yet freeze operational visibility.
The separate [same-frame operational capture](evidence/BL-D-06/operational-approach/README.md)
subsequently supplies valid joint range/admission evidence and supports the bounded
BL-D-06 technical PASS / 4.0 m/s freeze candidate. It does not change this dataset's
3.2 m/s conservative screening result or its zero-admission finding. Final G1
checklist publication and full-team admission remain PR #90 responsibilities.
No G1 gate decision is published by this blocker-resolution work. Full provenance, all-sequence results,
occlusion/ID-switch findings, equations and raw artifact hashes follow.

```{toctree}
:maxdepth: 1

evidence/BL-C-05-refresh/README
```
