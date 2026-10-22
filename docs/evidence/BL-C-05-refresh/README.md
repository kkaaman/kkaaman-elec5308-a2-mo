# BL-C-05 sensor-wait-aware refresh — 21 October 2026

**Fresh capture COMPLETE: 600/600 sensor observations. Conservative BL-C screen
supports 3.2 m/s, not 4.0 m/s. D-006/G1 remain BLOCKED and unfrozen.**
This is parked-ego, kinematic-target detector evidence, not closed-loop stopping
certification. No implementation, acceptance threshold or proposed cruise value
was changed. The withdrawn historical 4.9 m/s result is not reused.

## Provenance and capture

Branch `fix/BL-C-05-sensor-wait-screening`, canonical base
`f777a6d0aec45feded3046749c90e384c6bdffcd`. The actual unchanged collector ran
200 frames per scenario. [Command and completion log](capture-attempt.json),
[provenance](provenance.json), [source manifest](sequences.json),
[benchmark](benchmark.json), [analysis](analysis.json),
[per-frame analysis](frame-analysis.csv) and [summary](capture-summary.json)
retain the measured results. Settings were restored and no actor leaks remained.

Observed client/server **0.9.16**, map **Carla/Maps/Town10HD_Opt**, OpenDRIVE SHA256
`5d883b799f634030af92be1e9d79d107845540ba04338e8c60e095be1aef7be7`, capture site 10.
Aligned RGB-D: **640×360**, **90° horizontal FOV**, actor-relative camera
**x=1.5 m, z=2.0 m**, axial depth, **0.05 s** fixed simulation step. Camera x in
rear-axle coordinates is 2.886406898442095 m; the actor-to-rear-axle offset is
already included there. Existing passing BL-B-04 convention evidence matches
server/map/rig hashes; no recalibration was needed. Weather is retained verbatim
in provenance/source manifest.

Real **yolo26n**, SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`, confidence
**0.25**, inference **640**, BoT-SORT resolved YAML SHA256
`8cd88930d3be5557be0777f384eecff0a36a7011850fe613fc0a68178bfd5b01`.
Resolved settings are versioned, including buffer 30, high/new threshold .25,
low .1, match .8, sparseOptFlow, ReID disabled. Policy is unchanged:
**2-of-3**, expiry **0.5 s**, median history **5 samples**. Actual device
**cuda:0**, NVIDIA GeForce RTX 4070 Laptop GPU; Python 3.12.11,
Ultralytics 8.4.177, Torch 2.6.0+cu124, NumPy 2.5.2, OpenCV 5.0.0.93, LAP 0.5.13.

The earlier connection timeout occurred because CARLA had not been started,
as confirmed by the user. It is **invalid infrastructure/setup evidence, not
a BL-C-05 failure**. Its original records remain in `setup-attempt/`; no duration
from that attempt enters the fresh timing or screen.

## Fresh first-event results

Latencies below are serial replay availability from each sequence start, including
actual read/conversion, detector, filter, source schedule and accumulated backlog.
Ranges are evaluation-only target near-face to ego-front-bumper gaps, not axial
depth interpreted as a bumper gap. Evaluation labels are applied after the
operational filter and never enter runtime control.

| Sequence | Sensor-valid / sampled | Confirmed + valid target depth | First raw gap / latency | First confirmed gap / latency | First-confirmed source age |
|---|---:|---:|---|---|---|
| Approach | 200/200 | 161 | 45.253 m / 4.199442 s | 39.654 m / 5.303253 s | 3.903253 s |
| Short occlusion | 200/200 | 144 | 36.054 m / 2.531997 s | 35.854 m / 2.587205 s | 0.237205 s |
| Continuity risk | 200/200 | 152 | 42.862 m / 0.836400 s | 42.662 m / 0.882718 s | 0.232718 s |

**600 sensor-valid frames; 457 current, confirmed, matching-valid-depth target
frames.** Valid sensor input is not equivalent to valid target visibility.
Short occlusion hides indices 60–62 (3 frames); confirmation with depth returns
at index 64, gap 32.454 m, same target ID 13. Continuity occlusion hides 80–99
(20 frames); confirmation with depth returns at index 104, gap 24.419 m,
new ID 42 following ID 6. No target confirmation occurs during either scripted
occlusion. No labelled target has a None ID in this capture; None-ID semantics
remain covered by tests, not a claimed live observation.

There are **zero sensor-read failures and zero callback errors**, and every
attempt has measured read duration. The configured read timeout remains **5 s**;
no fabricated five-second waits were inserted. Each joiner records overflow=1,
discarded=635, joined=200, retained rather than claimed as zero queue events.
Warm-up and bounded source queues discard non-requested observations; all 200
requested source frames per sequence were joined and retained. All 600 filter
outputs have `usable_for_clearance=False`.

## Wait-aware timing and cold backlog

Measured `sensor_read_wall_s` spans `capture.read()` and conversion, ends before
tracking, and is charged exactly once:

`ready = max(source_elapsed + sensor_delay, previous_ready) + read + detector + filter`

`response = ready - source_elapsed`. The source sensor schedule delay is
**0.200000003 s**. Read/conversion across 600 frames: minimum **0.451890 ms**,
median **0.595740 ms**, p95 **0.980792 ms**, maximum **7.419415 ms**, sum
**0.397346684 s**. Missing measured read durations: **0**.

| Sequence | Read median / max (ms) | Detector median / max (s) | Filter median / max (ms) | Serial response max (s) |
|---|---|---|---|---|
| Approach | 0.579938 / 7.419415 | 0.031833 / 3.981091 | 0.776867 / 10.931555 | 4.199442 |
| Short occlusion | 0.586682 / 4.025487 | 0.033937 / 0.161868 | 1.019580 / 1.884639 | 0.366483 |
| Continuity risk | 0.616120 / 2.182879 | 0.034590 / 0.121098 | 1.040816 / 2.990362 | 0.323289 |

The initial approach tracking call is cold: read+detector+filter **3.999442 s**.
It creates backlog retained by replay; even the final approach frame has
**1.064855 s** source age. Later sequences use the already loaded model but reset
tracker/filter per episode. Their smaller timings must not be substituted for
the cold sequence without a separately reviewed operational admission gate.
This capture measures through filter availability, **not governor/control or
command delivery**. Those BL-D components must be measured/composed separately
without double-counting. Startup connection and evaluation/serialization work
are not fabricated as sensor-read latency.

## Defensible sampled range and speed interpretation

The unchanged benchmark's first-event-only calculation gives **5.5 m/s** and
reports proposed 4.0 m/s supported. That is a real fresh result, but its first
confirmations have axial depth **40.662 / 36.837 / 43.647 m**, beyond the currently
reviewed **25 m axial** evidence scope. It is not sufficient for a BL-D freeze.
The old withdrawn 4.9 m/s result remains invalid; 5.5 is a different fresh screen.

For a more conservative downstream screen, retain only current confirmed target
observations whose measured axial depth is within that existing scope. First
such observations are at index 106 in all three sequences, with axial depths
**24.995567 / 24.995567 / 24.995626 m** and evaluation bumper gaps
**24.013188 / 24.013188 / 24.013027 m**. Thus the common **sampled confirmed bumper
range is 24.01302661659352 m**, not an assumed 25 m bumper visibility. There are
**270** such observations (88 / 89 / 93). This is sampled evaluation evidence,
not a universal visibility guarantee or a frozen clearance/footprint parameter.

Target usability still drops out later: after index 106 the unusable indices are
170,172–176 for approach; 170,172,173,175,176 for short occlusion; 173 for continuity.
These misses/invalid target observations remain unknown and cannot clear HOLD.
This evidence does not prove uninterrupted visibility or generalise beyond this
single site, vehicle, weather and scripted set.

To avoid choosing a favourable timing/range pairing or excluding cold backlog,
combine the minimum common sampled range with the **maximum source-relative
response across all 600 attempts, 4.199441865980589 s**. Add measured BL-E-03
vehicle lag **0.3500000052154064 s once**, deceleration
**1.497566177392398 m/s²**, unchanged margin **6.0 m**:

```text
response = 4.549441871195995 s
required_gap(4) = 6 + 4*response + 16/(2*1.497566177392398)
                = 29.539768473301102 m > 24.01302661659352 m
cap = sqrt((a*response)^2 + 2*a*(range-6)) - a*response
    = 3.205372578126002 m/s
rounded down to the existing 0.1 m/s reporting precision: 3.2 m/s
```

**3.2 m/s is the highest rounded-down cap on this stated conservative BL-C
basis; 4.0 m/s is not justified on that basis.** No baseline cruise setting was
changed. This is first-confirmation/perception screening only: unmeasured control,
delivery, range uncertainty and closed-loop stopping/HOLD/resume preclude calling
3.2 m/s a certified operational speed. The paired first-in-scope approach event
alone yields 4.011 m/s (only about 0.064 m slack at 4 m/s); it must not be presented
as a robust freeze while ignoring the other observed response ages.

## BL-D/G1 hand-off and remaining blockers

**BL-C-05 PASS for its capture/reporting scope**: a failed proposed-speed screen
with an explicit lower recommendation satisfies that task. The fresh-capture
blocker is resolved. [BL-D's current review](../../baseline_threshold_freeze.md)
confirms that 4.199442 s cannot authorise a new positive operational packet; the
0.9500000052 s gated response is independently justified. The conservative 3.2
screen is not the operational governor's timing model.

However, current-policy replay with the operational 25 m depth policy produces
zero valid full observations/positive commands across all 600 rows, even granting
fresh/warm status: other current invalid tracks force HOLD. Target-only confirmed
range does not establish operational first-admission range. D-006/G1 therefore
remain **technically BLOCKED**, not merely waiting for signatures. Lower speed
cannot validate unknown visibility; no configuration is admitted/frozen. Full-team
review and durable evidence/configuration publication also remain required.
No CARLA recapture or runtime/threshold change was made for this follow-up.

## Reproduction and raw artifacts

The raw run is retained locally at
`/home/alan/26-S2-Assignment2-Group7/runs/BL-C-05-refresh/20261021-wait-aware-02`.
It includes 600 depth and 600 RGB NumPy arrays, per-sequence JSONL, source/config
snapshots and benchmark output (about 970 MB before analysis artifacts).
[Artifact hashes](artifact-manifest.json) cover every retained file. Raw arrays
are not placed in Git; share/archive them alongside this evidence for independent
replay. Versioned sequences and per-frame analysis preserve all sampled misses.

```bash
PYTHONPATH=src /home/alan/26-S2-Assignment2-Group7/.venv/bin/python scripts/benchmark_obstacle_tracks.py \
  --input /home/alan/26-S2-Assignment2-Group7/runs/BL-C-05-refresh/20261021-wait-aware-02/sequences.json \
  --output /tmp/BL-C-05-replayed.json
```

For the per-frame analysis, apply the same filter and replay recurrence to every
source row, select evaluation-labelled target IDs only after filter update, record
`current && confirmed && depth_valid`, and count `distance_m <= 25` for the reviewed
scope. Take the minimum first-in-scope evaluation gap across sequences and maximum
response across all source rows, then apply the equation above. Full per-frame
inputs/results are versioned; no evaluation labels enter the runtime filter.

## Validation

Focused BL-C/RGB-D tests: **60 passed**, including sensor-failure temporal
equivalence and measured-wait accumulation regressions; see [JUnit](regression.xml).
`ruff check .`: passed. Full pytest: **814 passed, 1 skipped, 3 subtests passed**.
Sphinx `-W` and `git diff --check`: passed. These tests do not certify an
operational speed or replace the remaining BL-D/G1 review.

```{toctree}
:maxdepth: 1

ai-journal
```
