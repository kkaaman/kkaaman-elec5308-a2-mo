# Lane recovery and consolidation — RV-02 / RV-03

## Source and comparison

A2 starts from A1 tag `a1-final` (README identifies commit `a1f3112`). The uploaded
working copy already included consolidation edits, so the original differences
below are recorded from the supplied pre-consolidation terminal diff, not a new
Git-history comparison performed in this environment.

| File | src/perception | Original src/autodrive/perception | Decision |
|---|---|---|---|
| lane_inference.py | RGB accepted without channel swap | Converts BGR to RGB | Canonical RGB inference; legacy wrapper preserves BGR |
| lane_path.py | Small-component removal, gap closing, horizontal-stripe suppression and global-median outlier filtering | Sequential nearest-boundary selection without those additions | Keep canonical implementation; preserve public extraction/drawing imports |
| lane_segmentation.py | ResNet18 LaneUNet | Identical according to supplied git comparison | One canonical model definition, legacy re-exports |
| lane_dataset.py | Not present | Dataset utilities used by the training pipeline | Retain under `autodrive.perception` for training compatibility |
| losses.py | Not present | Training loss functions | Retain under `autodrive.perception` for training compatibility |

Callers remain split across the compatibility boundary. Runtime demos such as
`src/demo_fused_lane_following.py` and `src/demo_combined_perception.py` use the
canonical `perception.*` modules, while `examples/run_carla_demo.py` and several
compatibility tests still use `autodrive.perception.*`. The training pipeline
also depends on `autodrive.perception.lane_dataset`,
`autodrive.perception.lane_segmentation` and
`autodrive.perception.losses`. These training-only modules therefore remain in
the legacy package rather than being deleted during consolidation.

The checkpoint identity reported on the university PC was
`10dca01621fb8e110f1d5b8419e81fb616a8ce1b75d6aface11f5793af88e8ff`,
size 57,385,466 bytes. The RV-02 asset record is preserved in
`evidence/RV-02-assets.json`. That evidence was captured on the combined
pre-split working tree and therefore is not treated as independent proof of the
final consolidated implementation. RV-03 subsequently verified the canonical
offline loading path. `verify_assets.py` independently checks the current Git
LFS pointer instead of trusting a hardcoded hash.

## Compatibility and deprecation

`src/perception` is canonical. Legacy `autodrive.perception` lane modules are
compatibility shims, retained for existing callers. They are deprecated for new
code; no removal date is set. Public classes/functions remain available.
Private helper imports were not supported APIs and are not re-exported.

Canonical inference accepts nonempty uint8 RGB images. Legacy inference accepts
nonempty uint8 BGR images and converts once. Existing A1 callers must keep their
original colour convention; new CARLA callers convert BGRA to RGB at the adapter
boundary. Array dtype/shape validation now rejects ambiguous input early.

Inference constructs ResNet18 without pretrained downloads before strict loading
of the complete lane checkpoint. Training's pretrained default remains available.
Both loaders use tensor-only checkpoint loading; arbitrary pickled objects are
not supported. No retraining, weight replacement or controller change is included.

## Verification and limits

Previously reported PC results: 12 lane tests passed; lane checkpoint loading
without download and RGB/BGR prediction equivalence passed. These are historical
session observations, not a replacement for rerunning the delivered tests.

The new colour tests use a channel-sensitive probe to detect a real channel swap,
not just equal argmax masks. Path tests cover straight, curved, broken, single-sided,
blank and invalid masks. Real sample loading/checksums are verified separately.

The inherited extractor uses fixed pixel thresholds and global-position median
filtering. Synthetic success does not establish accuracy on sharp curves,
intersections, different resolutions or missing markings. No live driving
acceptance is claimed. Lane estimates remain a planned departure monitor/HUD
input (RV-06), not a replacement for the baseline map route.

Reviewers: Zhaolin for RV-02 and RV-03. Document sample-run evidence before completion.
