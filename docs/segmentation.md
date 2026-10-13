# Semantic segmentation

The segmentation module (`src/segmentation/`, owner Mohammad) keeps the A1
BiSeNetV2 model as the default. Two alternatives, suggested in tutor
feedback, sit behind the same interface, so BL-B-05 can compare them on
identical frames. The rest of the stack switches backend by configuration
only.

RV-08 adds the sensor-rig, cadence and lifecycle sections to this page.

## Backends

| Name | Model | Labels | Weights | Offline? |
|---|---|---|---|---|
| `bisenetv2` (default) | A1 BiSeNetV2 (`SemanticSegmenter`) | All 19 Cityscapes classes | `segmentation_model.pth` in this repository (`assets.md`) | **Yes** |
| `segformer_b0` | SegFormer-B0 fine-tuned on Cityscapes (Hugging Face) | All 19, same order | `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` | After the first download. Pass `local_files_only=True` to forbid network access |
| `yolo_seg` | Ultralytics instance segmentation trained on COCO | Objects only: person, bicycle, car, motorcycle, bus, train, truck, traffic light, stop sign. Everything else is 255 | `yolo26n-seg.pt` (or `yolov8n-seg.pt`) | After the first download |

Every backend takes an `(H, W, 3)` uint8 RGB frame. It returns an `(H, W)`
uint8 mask of Cityscapes trainIds (0–18) at the input resolution, with 255
for unlabelled pixels.

`YoloSegSegmenter.predict_instances` also returns one boolean mask per
detected object. That is the form depth-ROI masking needs (BL-B-05).

```python
from segmentation import create_segmenter

seg = create_segmenter("bisenetv2")                  # A1 default, offline
seg = create_segmenter("segformer_b0", device="cuda")
mask = seg.predict(rgb)                              # HxW trainIds, 255 = unlabelled
```

`segformer_b0` needs `transformers`. Install it with
`pip install -e ".[seg]"`. Importing `segmentation` does not need it, because
heavy libraries load only when a backend is built.

### In the A1 demos

Both demos take `--seg-backend` (and `--seg-weights`). They now check the
weight files and load every model **before** spawning the car and sensors,
so a model failure exits cleanly instead of leaving actors in the world:

```bash
cd src
python demo_combined_perception.py --seg-backend segformer_b0
python demo_fused_lane_following.py --seg-backend yolo_seg
```

### Comparing backends

```bash
python scripts/evaluate_segmentation.py \
    --images "runs/smoke/*/rgb.png" --labels "runs/smoke/*/semantic.png" \
    --label-format carla-palette \
    --backend bisenetv2 --backend segformer_b0 --backend yolo_seg=yolo26n-seg.pt
```

**Outputs:**
- the 19-class mean IoU;
- the obstacle mean IoU (person, rider, car, truck, bus, motorcycle,
  bicycle);
- pixel accuracy and per-class IoU;
- inference timing;
- overlay images.

CARLA ground truth comes from the semantic camera. Its tags 1–19 are the
Cityscapes trainIds + 1 (`assets.md`).

## First comparison (12 Oct 2026, preliminary)

The first comparison ran on WCB3LJ64 (RTX 4000 SFF Ada, CUDA) against
CARLA's own semantic labels. Results are in
`docs/evidence/BL-B-05/seg_eval/`.

| Backend | Weights | GPU ms p50 / p95 | 19-class mIoU | Pixel accuracy | Car IoU |
|---|---|---|---|---|---|
| bisenetv2 | `segmentation_model.pth` | 12.2 / 12.7 | 0.345 | 0.876 | 0.00 |
| segformer_b0 (run 1) | Cityscapes 1024×1024 checkpoint | 47.0 / 49.6 | 0.401 | 0.937 | 0.00 |
| segformer_b0 (run 2, settings not recorded) | same | 15.1 / 16.5 | 0.364 | 0.923 | 0.02 |
| yolo_seg | `yolov8n-seg.pt` | 23.4 / 25.5 | – (objects only) | – | 0.00 |
| yolo_seg | `yolo26n-seg.pt` | 33.4 / 34.8 | – (objects only) | – | 0.67 |

**How far to trust this:**
- **Weak evidence.** The three frames come from the PROC-02 smoke runs at
  the **same spawn point**, so this is effectively one scene, with few
  obstacles. Treat it as a pipeline check plus first timings, not a
  benchmark.
- **Real CARLA frames help.** On them, the A1 BiSeNetV2 is the fastest, and
  its scene classes (road 0.96 IoU) are usable.
- **Vehicles are its weak point.** Only `yolo26n-seg` segmented the car.
- **Next steps.** The decision on depth-ROI masking needs the
  obstacle-distance experiment (`experiments/detector_seg.md`), followed by
  BL-B-05's ablation on the BL-B-03 dataset.
