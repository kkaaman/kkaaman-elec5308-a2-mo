# Experiment: YOLO-seg masks vs box-only YOLO for obstacle distance

| Field | Value |
|---|---|
| Date and author | 12 October 2026, Mohammad |
| Machine | WCB3LJ64 (RTX 4000 SFF Ada, CARLA 0.9.16, Town10HD_Opt) |
| Script | `scripts/detector_seg_experiment.py` |
| Feeds | BL-B-02 (camera rig), BL-B-05 (segmentation masking), BL-D-03 (box depth) |
| Evidence | `docs/evidence/BL-B-05/det_seg/<run>/` (`capture.json`, `analysis.json`). Captured frames (216 MB) are kept off-repo |

## Question

Should the detector switch from `yolo26n.pt` to `yolo26n-seg.pt`, so that
per-object masks can be used to read obstacle depth? Does it detect at least
as well, give better distances, and fit the time budget?

## Setup

- **Scene.** The ego is parked on a straight lane in Town10HD, with one
  obstacle at a time in the ego lane: a parked car (`vehicle.tesla.model3`)
  or a standing, front-facing pedestrian (`walker.pedestrian.0001`).
- **Gap.** Ego front bumper to the obstacle's near surface (the stop-margin
  definition in `acceptance.md`): 5, 10, 15, 20, 25 and 30 m.
- **Camera.** x = 1.5 m, z = 1.7 m on the ego, 90° horizontal field of view.
  RGB, depth and instance cameras are co-located.
- **Ground truth** (evaluation only). The obstacle's own pixels from the
  instance camera. Truth distance is the median depth (median surface) or
  the 5th-percentile depth (nearest surface) over those pixels.
- **Detected.** A box of the right class with IoU ≥ 0.5 against the
  ground-truth box.
- **Distance readings** for a detection:
  - full-box median;
  - inner-box median (middle 50%);
  - mask median (seg model only);
  - the nearest-surface equivalents of each (5th percentile).
- **Timing.** Models were timed after capture, with CARLA open but not
  ticking. Times include full-resolution masks for the seg model.
- **Depth** is CARLA's ideal depth camera, so the distance results isolate
  *which pixels* are read, not sensor noise.

## Results

### Detection

Pedestrians found out of 6. Every setting found all 6 cars.

| Camera | Model input | Confidence | `yolo26n.pt` | `yolo26n-seg.pt` | ms/frame (det / seg) |
|---|---|---|---|---|---|
| 800×450 | 640 | 0.20 | 5/6 (misses 30 m) | 3/6 (misses 20, 25, 30 m) | 27.6 / 31.2 |
| 800×450 | 640 | 0.10 | 5/6 (misses 30 m) | 5/6 (misses 30 m) | 25.7 / 30.0 |
| 800×450 | 960 | 0.20 | 5/6 (misses 30 m) | 5/6 (misses 30 m) | 23.6 / 28.2 |
| **1280×720** | **1280** | 0.20 | **6/6** | **6/6** | 25.8 / 33.1 |

The pedestrian at 30 m is 95 pixels at 800×450 and 238 pixels at 1280×720.

### Distance error

Median absolute error over detected frames, in metres:

| Camera / input | Class | Box median | Inner-box median | Mask median | Nearest: box | Nearest: inner | Nearest: mask |
|---|---|---|---|---|---|---|---|
| 800×450 / 640 | car | 0.13 | 0.10 | 0.01 | 0.005 | 0.013 | 0.000 |
| 800×450 / 640 | pedestrian | 0.09 | 0.02 | 0.04 | 0.008 | 0.001 | 0.004 |
| 1280×720 / 1280 | car | 0.21 | 0.10 | 0.02 | 0.005 | 0.013 | 0.001 |
| 1280×720 / 1280 | pedestrian | 0.09 | 0.02 | 0.02 | 0.009 | 0.001 | 0.002 |

Box and inner-box columns are from `yolo26n.pt`; mask columns are from
`yolo26n-seg.pt`. The seg model's own box readings were within 0.03 m of
these.

**Full-box median failures**, where the reading landed on the background:

| Pedestrian gap | Error | Model and setting |
|---|---|---|
| 25 m | +4.08 m | det, 800×450 / 640 |
| 25 m | +1.62 m | both models, 800×450 / 960 |
| 25 m | +1.41 m | seg, 1280×720 |
| 10 m | +0.72 m | seg, 800×450 / 640 |

The inner-box and nearest-surface readings never failed.

## Findings

1. **Keep `yolo26n.pt`.** The seg model never detected more. At the default
   threshold it lost pedestrians at 20–25 m, because of lower confidence on
   small objects. It only matched the current model with a lower threshold
   or a larger input. It costs 4–7 ms more per frame.
2. **Masks do not improve distance.** With a tight box, the inner-box median
   and the nearest-surface reading are already within about 0.1 m, far
   inside the 0.5 m depth target. Masks improve this by centimetres.
3. **Do not use the full-box median for pedestrians.** It reads the
   background at range. Use the inner box or a low percentile, as BL-D-03
   plans (`depth-association.md`). In real scenes a low percentile also
   picks up anything partly blocking the obstacle, such as a pole. That errs
   on the safe side but can cause early stops.
4. **Camera resolution limits pedestrian range.**
   - At 800×450, no model or setting detected the pedestrian at 30 m.
   - At 1280×720 with model input 1280, both did, and the current
     detector's model time was unchanged (about 26 ms).
   - Confirmation over several frames means first detection must come
     before 25 m, so 800×450 is at the edge of the 25 m target.
5. **BL-B-05 (segmentation masking for depth) is likely to be cut** on this
   evidence, leaving segmentation as a display layer. Confirm on BL-B-03
   data with sideways offsets and partial occlusion.

## Recommendation for BL-B-02

Make **1280×720 at 90° (detector input 1280)** the candidate forward camera.
It is recorded as proposed decision D-003 in `decisions.md`.

**Frame cost.** From the PROC-02 smoke test, with four co-located cameras at
a 0.10 s step: 26.7 ms p95 at 1280×720, against 22.5 ms at 800×450
(`docs/evidence/PROC-02/WCB3LJ64-20261012T045031Z`).

**Rough budget at 10 Hz:**
- about 27 ms for the cameras;
- about 26 ms for the detector;
- leaving roughly 45 ms of the 100 ms target for depth association,
  filtering and control.

The live rig needs only the RGB and depth cameras (semantic and instance are
evaluation-only), so the camera cost should be lower. The parts were timed
separately, so the full stack still has to be profiled together (TEST-17).
The same sum (about 53 ms) does **not** fit a 0.05 s (20 Hz) step, which
further supports the 0.10 s step at G1.

**Still to do:**
- Ricky to agree to detector input 1280 (`PerceptionConfig.image_size`).
- Repeat at other locations and with other pedestrian models.

## Limitations

- One location.
- One car and one pedestrian model, front-facing poses only.
- Stationary ego, daylight, ideal depth.
- Obstacles centred in the lane, with no sideways offsets or occlusion.
- False positives not measured, which matters for the lower-threshold run.

## Reproduce

```powershell
.\.venv-carla\Scripts\python.exe scripts\detector_seg_experiment.py capture --then-analyze
.\.venv-carla\Scripts\python.exe scripts\detector_seg_experiment.py analyze runs\det_seg\<run> --conf 0.10
.\.venv-carla\Scripts\python.exe scripts\detector_seg_experiment.py analyze runs\det_seg\<run> --imgsz 960
.\.venv-carla\Scripts\python.exe scripts\detector_seg_experiment.py capture --width 1280 --height 720 --then-analyze --imgsz 1280
```

**Run folders.** The main runs are `20261012T043252Z` (800×450) and
`20261012T044650Z` (1280×720). All five capture runs from that day are
summarised in `docs/evidence/BL-B-05/det_seg/`.

Each `analyze` overwrites that run's `analysis.json`, so the 800×450 folder
holds only the last setting analysed. The tables above come from the console
output of each run.
