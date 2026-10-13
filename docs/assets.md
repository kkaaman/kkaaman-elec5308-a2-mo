# Model assets and provenance (RV-01)

The trained model files the A2 stack depends on: what they are, where they
came from, and how to check that a clone holds the real files rather than
Git LFS pointer stubs.

## Check a clone

```bash
git lfs install
git lfs pull                                # fetch the real .pth files
python scripts/verify_assets.py --load      # sizes, SHA256, and a test segmentation
```

The script fails (non-zero exit) if any file is missing, is still an LFS
pointer of about 130 bytes, or doesn't match the checksum below. Code should
locate these files with `autonomy.paths.model_path("segmentation")` and
friends, which also report LFS pointers clearly.

## Register

| Name | File | Storage | Bytes | SHA256 |
|---|---|---|---:|---|
| `segmentation` | `segmentation_model.pth` | Git LFS | 9,505,431 | `a631440d3fe2ccd2554631bb49b1352c0efbbaffc2316caa79c288757e8dde2a` |
| `lane` | `lane_unet_best.pth` | Git LFS | 57,385,466 | `10dca01621fb8e110f1d5b8419e81fb616a8ce1b75d6aface11f5793af88e8ff` |
| `yolo` | `yolo26n.pt` | Git (not LFS) | 5,544,453 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |

For both LFS files, the SHA256 of the downloaded file equals the `oid` in
its Git LFS pointer, and the size equals the pointer's `size`. Checked on
2026-10-13.

## Segmentation model (owner: Mohammad)

| Item | Record |
|---|---|
| Architecture | BiSeNetV2 (`src/segmentation/bisenetv2.py`), 19 Cityscapes classes, 2,346,687 parameters |
| Checkpoint format | Bare `state_dict`, 315 tensors, no embedded metadata. `SemanticSegmenter` also accepts a `{"model_state_dict": ...}` wrapper |
| Input | RGB, resized to 512×256 for inference. The mask is resized back to the input resolution |
| Added | Commit `1e48a83`, 2026-09-13, by kkaaman, in the A1 repository |
| Training (from that commit) | Best checkpoint at validation mIoU 0.4763. Combined focal and Dice loss, rare-class oversampling with a capped sampling weight, horizontal-flip augmentation, and CARLA background traffic to address bus, motorcycle and train scarcity |
| Training code and data | Not in this repository |
| Load check (CPU, 2026-10-13) | Loads without network access. On `src/perception/street.jpg` (335×597) it returns a full-size mask with 13 classes present: road 43%, sky 28%, building 20%, car 5%. Warm inference took about 138 ms on CPU |

**Known limits:**
- mIoU 0.4763 is a validation figure from A1 training. It has not been
  re-measured on A2 frames.
- Whether segmentation improves depth sampling is tested separately in
  BL-B-05.

## Lane model (owner: Abdullah, RV-02)

Added in commit `4058963` on 2026-09-13 and re-committed unchanged in
`d9c0539`. The checksum is recorded above.

Loading it currently needs internet access: `LaneUNet` builds an ImageNet
ResNet18 backbone first (`docs/repository_setup.md`). The comparison of the
two lane code trees and the offline fix belong to RV-02 and RV-03.

## YOLO detector (owner: Ricky, BL-C-03)

`yolo26n.pt` is the Ultralytics checkpoint used by `perception.ObjectDetector`.
It loads with 80 COCO classes. Its version and source go into the model
manifest under BL-C-03.

## Offline copy

To keep a copy outside Git, for example on OneDrive or a USB drive for the
demo laptop:

```bash
python scripts/verify_assets.py --export /path/to/backup/elec5308-models
```

This copies the three verified files and writes a `SHA256SUMS` file. On
Linux or macOS, check a copy later with `sha256sum -c SHA256SUMS`. On
Windows, use `Get-FileHash -Algorithm SHA256 <file>` in PowerShell.
