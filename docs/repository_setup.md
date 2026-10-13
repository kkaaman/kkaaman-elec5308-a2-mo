# Repository setup record (PROC-01)

This page records how the A2 repository was created from Assignment 1, so the
starting point can be checked and reproduced.

## Source and identity

| Item | Value |
|---|---|
| A2 repository | <https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7> |
| A1 source | <https://github.com/USYD-ELEC5308/26-S2-Assignment1-Group7>, branch `main` |
| A1 final commit | `a1f31128ffa3d8150f8f011b901936799e245b41` (2026-09-20) |
| A1 tree hash | `db35b66f527089e3cd301355a3c5440c6a58f15b` |
| Tag | `a1-final` → A1 final commit |
| A1 source archive | `git archive --format=tar a1-final` → SHA256 `08ecee301aad61ef68d74f3ded1c12010b0871f79f4ac7a8f1337f31c4b35ee8` (72,826,880 bytes; LFS files appear as pointers) |
| History | All 72 A1 commits are kept with their original authors. The A2 repository's initial README commit is merged in so that `main` keeps both histories. |

The commit hash and tree hash are the authoritative identifiers. The archive
hash is a convenience check and can differ between Git versions.

## Model assets

| File | Storage | Size (bytes) | SHA256 | Load check (CPU, 2026-10-13) |
|---|---|---:|---|---|
| `lane_unet_best.pth` | Git LFS | 57,385,466 | `10dca01621fb8e110f1d5b8419e81fb616a8ce1b75d6aface11f5793af88e8ff` | Loads into `LaneUNet`, 360×640 mask. See issue 1 below |
| `segmentation_model.pth` | Git LFS | 9,505,431 | `a631440d3fe2ccd2554631bb49b1352c0efbbaffc2316caa79c288757e8dde2a` | Loads into `SemanticSegmenter` (BiSeNetV2), 360×640 mask |
| `yolo26n.pt` | Git (not LFS) | 5,544,453 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` | Loads in Ultralytics, 80 classes |

Run `git lfs pull` after cloning. Without Git LFS, the two `.pth` files are
pointer files of about 130 bytes and the models will not load.

## Fresh verification of the A1 baseline

These checks ran on 2026-10-13 on Linux with Python 3.12.3. Dependencies were
unpinned and resolved that day: torch 2.14.1, torchvision 0.29.1, ultralytics
8.4.176, opencv-python 5.0.0.93, numpy 2.5.4, pytest 9.1.1, ruff 0.17.0,
sphinx 9.1.0 and myst-parser 5.1.0.

| Check | Result |
|---|---|
| `ruff check .` | Pass |
| `python -m pytest` | 106 passed, which matches the A1 README's claim |
| `sphinx-build -W docs docs/_build/html` | Pass |

These results come from the A1 code alone. They are not A2 results and not a
CARLA run.

## Issues found during setup

1. **The lane model needs internet access at construction.** `LaneUNet`
   builds its backbone with `resnet18(weights=ResNet18_Weights.DEFAULT)`,
   which downloads ImageNet weights before the trained checkpoint replaces
   them. Offline startup fails. Fix under RV-02/RV-03: build with
   `weights=None` when loading a full checkpoint.
2. **Two lane implementations exist.** They are in `src/perception/` and
   `src/autodrive/perception/`. Reconcile them under RV-02/RV-03.
3. **Dependencies are unpinned.** `pyproject.toml` sets no version bounds. Pin
   validated versions once the CARLA machines are inventoried (PROC-02).

## GitHub handles

These come from A1 merge-commit messages and the first A2 pull request. Confirm them before assigning issues.

| Member | Handle |
|---|---|
| Mohammad | `kkaaman` |
| Alan | `at0941` |
| Abdullah | `Abdullah-Almutairi-uni` |
| Zhaolin | `Zhaolin-Huang` |
| Ricky | `RickyPurani` |
| Saviour | `skwo0237` |

## PROC-01 checklist

- [x] Repository URL recorded
- [x] A1 starting point preserved with full history, tag `a1-final`, and hashes recorded
- [x] A1 compatibility tests retained and passing (106)
- [x] Branch and PR policy (decision D-002), PR template with AI disclosure, task issue template
- [ ] Branch protection on `main` (admin): require a PR with 1 approval, require the `lint`, `test` and `docs` checks, block force pushes and deletions
- [ ] All six members confirmed able to open PRs
- [ ] Initial issues imported from the task board, one per Task ID
