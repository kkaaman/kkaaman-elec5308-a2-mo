# TEST-09: held-out test-split audit

Date: 2026-10-20. Town10HD_Opt **test** split, 2900 frames, 145 sequences of 20 frames.
These were not used for any selection. Detector `yolo26n.pt` at the frozen 1280 px and
confidence 0.25, with the production association, box depth and track filter. Model and
configuration are unchanged from the validation selection, as TEST-09 requires.
Report: `perception_test.json` (stage folder `Group7_batch35_Audit_…/heldout`).

## Gate result

| Bin (m) | Car in-path poses (pass) | Person in-path poses (pass) |
|---|---|---|
| 5 | 8 (8) PASS | 9 (8) PASS |
| 10 | 7 (5) FAIL | 4 (4) too few poses |
| 15 | 6 (6) PASS | 6 (6) PASS |
| 20 | 8 (8) PASS | 6 (4) FAIL |
| 25 | 4 (3) too few poses | 5 (4) PASS |
| 30 | 4 (4) too few poses | 6 (5) PASS |

The gate's usable range is 2.5 m for both classes, against 17.5 m on the validation split.
That figure comes from contiguity, not from a 2.5 m sight limit:
- The person 10 m bin has only 4 independent poses (5 are needed), even though all 4 pass.
- The car 10 m bin fails by one pose more than the one-miss allowance.

The useful reading is per pose, below.

## Failure examples (in-path targets)

**Cars: 37 poses, none missed.** Two poses at about 9.7 m (road 5) were detected at low
confidence (0.43-0.51) and confirmed one frame late: 0.20 s, and 0.15 s with 80% recall.
The limit is 0.15 s. One car at 27.1 m had 45% recall. Every car pose from 12.5 m to
22.5 m passes. Held-out car detection agrees with the validation result, apart from
one-frame-late confirmation near 10 m.

**People: 36 poses, 5 never detected.** Visible height is estimated from the truth box
(640×360 image, fx = fy = 320 px):

| Pose | Distance | Truth box | Visible height | Kind |
|---|---|---|---|---|
| road12 r000 pose000100 | 5.3 m | 23×68 px | ~1.13 m | child-sized |
| road12 r000 pose004300 | 20.2 m | 12×29 px | ~1.83 m | adult |
| road12 r001 pose010460 | 20.3 m | 6×17 px | ~1.08 m | child-sized |
| road5 r000 pose004420 | 25.7 m | 6×23 px | ~1.85 m | adult |
| road5 r001 pose011660 | 28.9 m | 5×13 px | ~1.17 m | child-sized |

The validation split's two person misses were an adult at 15.3 m (38 px tall) and a
child-sized walker at 20.6 m.

## What this means

1. **The 17.5 m person range is not confirmed on held-out data.** The 17.5-22.5 m
   person bin passes on validation but fails here (4 of 6). The 12.5-17.5 m bin passes
   (6 of 6). The frozen cruise of 4.8 m/s was derived from 17.5 m. With the same
   measured braking (1.5 m/s², 0.75 s response), a 12.5 m range would give about
   3.4 m/s.
2. **Child-sized pedestrians are missed, including one at 5.3 m.** Capture cycles through
   every CARLA walker model. The live scenarios spawn only the adult
   `walker.pedestrian.0001`, so the 30 accepted runs do not exercise child-sized
   pedestrians. The demo is valid for the adult walker it uses. The report must not
   claim general pedestrian detection.
3. **No retuning.** TEST-09 keeps the model and configuration fixed after the
   validation selection, so the frozen configuration is not changed on this evidence.
   Improving pedestrian detection belongs to post-baseline work, for example:
   - LiDAR, which detects by geometry rather than appearance;
   - fine-tuning the detector on walkers, including children, captured on another town;
   - a higher camera resolution.

   Each would be re-measured with the same Perception → Acceptance stages.
4. **Replay timing.** The warm p95 is 50 ms (median 39 ms), against 22 ms on validation
   with the same settings. The whole run was uniformly slower, which points to host load
   rather than these images. Offline replay timing is not the runtime criterion; live
   capture-to-apply p95 was at most 47 ms in all 30 acceptance runs.
5. **False positives.** There were 3906 unmatched predictions over 2900 frames.
   - 3448 (88%) are traffic lights and stop signs, which are not obstacle classes.
   - Most of the rest are trucks, buses and motorcycles (238). The truth labels cover
     only cars and people, so these are unlabelled rather than necessarily wrong.
   - There are also 139 person and 66 car boxes with no matching truth.

   Whether any of these would stop the car is measured live by the false-stop count,
   which was 0 in every S0 and S3 run.
