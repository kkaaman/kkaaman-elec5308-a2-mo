# Model assets and provenance (RV-01)

The trained model files the A2 stack depends on: what they are, where they
came from, and how to check that a clone holds the real files rather than
Git LFS pointer stubs.

Expected sizes and checksums live in `assets/model_manifest.json`.
`scripts/verify_model_asset.py` checks a clone against it.

## Check a clone

```bash
git lfs install
git lfs pull                                    # fetch the real .pth files
python scripts/verify_model_asset.py            # size + SHA-256 of every asset
python scripts/verify_model_asset.py segmentation_bisenetv2 --load \
    --sample src/perception/street.jpg --record runs/assets/rv01.json
```

The script prints `[PASS]` or `[FAIL]` per asset and exits non-zero on any
failure. A failure means the file is:
- missing;
- still a Git LFS pointer (about 130 bytes), in which case the message says
  which `git lfs pull` to run;
- a different file (size or hash mismatch);
- unable to load.

`--record` writes a JSON provenance record: host, platform, Python, torch,
device, timing, predicted classes and git commit.

## Register

| Name | File | Storage | Bytes | SHA-256 | Owner |
|---|---|---|---:|---|---|
| `segmentation_bisenetv2` | `segmentation_model.pth` | Git LFS | 9,505,431 | `a631440d3fe2ccd2554631bb49b1352c0efbbaffc2316caa79c288757e8dde2a` | Mohammad (RV-01) |
| `lane_unet` | `lane_unet_best.pth` | Git LFS | 57,385,466 | `10dca01621fb8e110f1d5b8419e81fb616a8ce1b75d6aface11f5793af88e8ff` | Abdullah (RV-02) |
| `detector_yolo26n` | `yolo26n.pt` | Git | 5,544,453 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` | Ricky (BL-C-03) |

For the two LFS files, the manifest values are the `oid` and `size` from the
Git LFS pointers committed in the A1 repository.

## Segmentation model: `segmentation_bisenetv2`

| Field | Record |
|---|---|
| Architecture | BiSeNetV2 (`src/segmentation/bisenetv2.py`), 19 Cityscapes trainId classes |
| Input | 512×256 RGB (width × height), ImageNet mean/std normalisation. The mask is resized back to the input resolution |
| Checkpoint format | Bare `state_dict` saved by `torch.save`, with 315 tensors and 2,332,764 learnable parameters (float32). It includes the three auxiliary heads, so the strict load in `SemanticSegmenter` matches. No embedded metadata |
| Origin | Commit `1e48a83`, 2026-09-13, by kkaaman, in the A1 repository |
| Training (from that commit) | Best checkpoint at validation mIoU **0.4763**. Combined focal and Dice loss, rare-class oversampling with a capped sampling weight, horizontal-flip augmentation, and CARLA background traffic to address bus, motorcycle and train scarcity |
| Training code and data | Not in this repository. Add a link here if the notebook or dataset is published |
| Recovered | 12 Oct 2026, from the A1 repository's Git LFS object. The GitHub ZIP download contains only the 132-byte pointer |
| Structural check | 12 Oct 2026: tensor names and shapes compared with `BiSeNetV2(num_classes=19)` without executing checkpoint code. 0 missing, 0 unexpected, 0 shape mismatches |
| **Verified** | **12 Oct 2026** on a university Windows PC (WCB3LJ64, Windows 11, Python 3.13.12, torch 2.10.0+cpu). Hash matches, strict load OK. On `src/perception/street.jpg` it predicted road 43%, sky 28%, building 20%, car 5%, bus 1%, sidewalk 1%. Mean 37.9 ms/frame over 5 runs on that CPU. Record: [`evidence/RV-01/rv01.json`](evidence/RV-01/rv01.json) |

**Limits:**
- mIoU 0.4763 is the A1 validation figure. It has not been re-measured on A2
  frames.
- CPU timing says nothing about the demo GPU; TEST-17 profiles the full rig.

**CARLA labels.** CARLA 0.9.16's semantic camera lists tags 1–19 in the same
order as the 19 Cityscapes training classes, so CARLA tag = Cityscapes
trainId + 1. Tag 0 (Unlabeled) and tags 20–28 (Static, Dynamic, Other,
Water, RoadLine, Ground, Bridge, RailTrack, GuardRail) need an explicit
mapping, for example RoadLine → road, or are ignored. The model can
therefore be scored on captured CARLA frames without retraining (BL-B-05).

## Lane model: `lane_unet` (RV-02)

Added in A1 commit `4058963` on 2026-09-13 and re-committed unchanged in
`d9c0539`. Its size and hash pass, but there is no loader in the script yet.
Loading it currently needs internet access, because `LaneUNet` first builds
an ImageNet ResNet18 backbone (`repository_setup.md`). The fix and the
comparison of the two lane code trees belong to RV-02 and RV-03.

## Detector: `detector_yolo26n` (BL-C-03)

`yolo26n.pt` is the Ultralytics checkpoint used by `perception.ObjectDetector`.
It is a normal Git file and loads with 80 COCO classes. Its version and
source go into the model manifest under BL-C-03.

## Offline copy

Keep a copy outside Git, for example on OneDrive or a USB drive for the demo
laptop:

```bash
python scripts/verify_model_asset.py --export /path/to/backup/elec5308-models
```

This copies every verified file and writes `SHA256SUMS`. Nothing is exported
if any check fails. Check a copy later with `sha256sum -c SHA256SUMS` on
Linux or macOS, or `Get-FileHash -Algorithm SHA256 <file>` in PowerShell on
Windows.
