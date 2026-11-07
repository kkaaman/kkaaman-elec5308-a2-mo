# TEST-13 RGB-D depth validation evidence

This directory contains the reviewed evidence for BL-B-03, BL-B-04 and TEST-13.

## Environment

- CARLA 0.9.16
- Map: `Town10HD_Opt`
- RGB-D rig: 640x360, 90 degree FOV
- Evaluation-only ground truth is not consumed by the runtime control path.

## Depth convention calibration

The installed CARLA depth representation was measured using off-axis geometry rather
than assumed from documentation.

Command:

```bash
python scripts/calibrate_depth.py \
  --host 172.28.176.1 \
  --output runs/BL-B-04/depth-calibration
```

Result:

- status: `MEASURED_PASS`
- selected convention: `axial`
- off-axis samples: 107
- median absolute axial error: approximately 0.050 m
- median absolute radial error: approximately 1.886 m

See `depth_calibration.json`.

## TEST-13 dataset

The final reproducible capture contained 180 samples split by run/episode:

- train: 60
- validation: 60
- test: 60

Command:

```bash
python src/carla_adapter/dataset_capture.py \
  --host 172.28.176.1 \
  --output runs/BL-B-03/test13-final \
  --samples 180 \
  --rounds 6 \
  --sequence-frames 1 \
  --ray-reference
```

The full image dataset is not committed. The generated manifest/index and analysis
hashes provide provenance for the reviewed run.

## Validation analysis

Command:

```bash
python scripts/analyze_depth.py \
  --dataset runs/BL-B-03/test13-final \
  --depth-calibration runs/BL-B-04/depth-calibration/depth_calibration.json \
  --output runs/TEST-13/depth-analysis-final \
  --split val
```

The validation split produced:

- 117 target-geometry samples
- 39 target-box comparisons
- 0 invalid plain-depth target boxes
- median absolute depth error below 0.5 m for every measured vehicle and pedestrian
  range bin through 25 m

See `depth_analysis.json` and `depth_ranges.png`.

Optional segmentation masking remains disabled (`KEEP_OFF`) because detector-box
ablation and segmentation runtime evidence were not available. BL-B-05 therefore
remains optional and is not a baseline dependency.

## Failure handling

Existing RGB-D unit tests verify timeout, stale-frame and invalid-observation paths
fail safe rather than allowing stale or missing depth to become a valid clearance
observation.

Final repository validation:

- `ruff check .` passed
- `python -m pytest`: 666 passed, 1 skipped
- `sphinx-build -W docs docs/_build/html` passed
