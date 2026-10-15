# TEST-01: A1-settings comparison baseline

## Configuration and provenance

I captured the inherited A1 driving settings on the recovered library at
commit 3a518cb87490509a55fc14044b0129a7892bef2f.

This is an A1-settings comparison, not a pristine historical A1 checkout.
The recorder leaves inherited driving source unchanged.

- CARLA 0.9.16, Town10HD_Opt.
- Python 3.12.10; PyTorch 2.14.1+cpu; CUDA unavailable.
- RTX 4000 SFF Ada Generation, 20475 MiB; driver 596.71.
- Actual A1 vehicle: vehicle.audi.a2.
- Controller wheelbase: 2.5 m; steering normalization: 0.6 rad.
- Cruise setting: 3 m/s; junction setting: 2 m/s.
- Default driving-loop duration: 120 seconds, asynchronous simulation.

Exact settings, spawn poses, model hashes and source hashes are recorded
in TEST-01-manifest.json.

## Measured results

| Measurement | Result |
|---|---:|
| Command samples / video frames | 274 / 274 |
| Mean absolute nearest-lane CTE | 0.911968 m |
| RMS nearest-lane CTE | 1.124706 m |
| Maximum nearest-lane CTE | 3.427656 m |
| Mean speed magnitude | 1.958256 m/s |
| Maximum speed magnitude | 4.983622 m/s |
| Recorded collisions | 0 |

Controller-path CTE is a separate measurement: mean 6.283274 m over
211 current controller updates. It must not be combined with nearest-lane CTE.

The run completed without early quit, capture errors or cleanup errors.
A junction manoeuvre timed out, followed by sustained lane misses.
Display layers were toggled during the run; rendering overhead therefore varied.
Zero recorded collisions does not establish successful lane following.

## Evidence

- TEST-01-summary.json: measured summary and capture status.
- TEST-01-manifest.json: hardware, settings, source and model provenance.
- TEST-01-samples.csv: raw command and measurement samples.
- TEST-01-video-frames.csv: video-to-wall/simulation timestamp mapping.
- TEST-01-archive.json: verified raw archive identity.

Archive: Group7_TEST01_baseline-01_20261015_143228.zip
Size: 43951797 bytes. All 9 archived files matched the run files.
SHA256: f30b5ca894af2d71c9c2d9bf3ace3c733a77b267b969042ff73b6799272b8ce8

The archive contains the video, failure snapshots, source snapshot,
environment record, configuration and raw logs.

Team-accessible archive link: PENDING.

## Limitations

Nearest-lane CTE uses actor origin and the nearest driving-lane tangent
for evaluation only. Lane identity can change, particularly at junctions.
It is not fixed-route completion error.

The original camera callback lacks source frame timestamps. Observations
are asynchronous; cycle wall time is not sensor age.

Video contains processed frames at nominal 10 fps, not real-time playback.
Use the timestamp CSV for actual timing. Recording adds wall-time overhead.

A1 selected an Audi A2; the repaired baseline uses a Tesla Model 3.
Future comparisons must match the vehicle or explicitly disclose this difference.

The supplied 0.22 m / 1.14 m figures remain UNVERIFIED SUPPLIED RESULTS.
They are not measurements or acceptance thresholds for this run.

## Validation and review

319 automated tests passed; the short recording check and full capture succeeded.
Reviewer: Zhaolin. Review and team archive access remain pending.
