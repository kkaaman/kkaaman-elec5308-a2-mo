# Lane asset and implementation audit — RV-02

## Asset verification

The recovered lane checkpoint is 57,385,466 bytes. Its SHA256 matches
the committed Git LFS pointer:

10dca01621fb8e110f1d5b8419e81fb616a8ce1b75d6aface11f5793af88e8ff

An offline copy was verified, loaded on CPU and used for sample inference.
Evidence: `evidence/RV-02-assets.json`.

The evidence was captured on the combined working tree before splitting
the task branches. It verifies asset identity and inference output validity,
not detection accuracy.

## Original implementation differences

The comparison below records the pre-consolidation Git diff.

| Module | src/perception | src/autodrive/perception |
|---|---|---|
| lane_inference.py | Expects RGB without swapping channels | Converts BGR to RGB |
| lane_path.py | Adds component filtering, gap closing, horizontal-stripe suppression and global-median outlier filtering | Uses sequential nearest-boundary selection without those additions |
| lane_segmentation.py | ResNet18 LaneUNet | Identical model definition |

## Proposed consolidation

Keep `src/perception` canonical. Preserve legacy public imports through
compatibility wrappers, including the legacy BGR input convention.

Implementation and regression tests belong to RV-03. This task changes
documentation and evidence only; no lane code is deleted or replaced.

Reviewer: Zhaolin.
