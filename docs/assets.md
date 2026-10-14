# Model asset recovery — RV-01 / RV-02

Models must be recovered through the team's authorized Git LFS clone. The source
of truth for each object is the committed pointer's SHA256 and size. Do not commit
pointer text as though it were a recovered checkpoint, and do not put offline
backup copies into Git.

| Asset | Expected recovered size | Reviewer |
|---|---|---|
| segmentation_model.pth | 9,505,431 bytes (task board) | Alan |
| lane_unet_best.pth | 57,385,466 bytes (task board) | Zhaolin |

Run `scripts/verify_assets.py` on the university PC with the project environment.
It verifies the current committed pointer, actual checksum and size, creates a
separate offline copy, checks that copy, loads on CPU without pretrained downloads,
and predicts a saved sample. Results include the Git commit, origin, changed
tracked paths, sample hash, Python and PyTorch versions. A mismatch/load failure
produces FAIL and a nonzero exit code. Existing evidence JSON is never overwritten.

Example (PowerShell, repository root):

```powershell
.\.venv\Scripts\python.exe scripts/verify_assets.py --models segmentation --backup-dir "$HOME\Downloads\Group7_A2_model_backup" --output docs/evidence/RV-01-assets.json
.\.venv\Scripts\python.exe scripts/verify_assets.py --models lane --backup-dir "$HOME\Downloads\Group7_A2_model_backup" --output docs/evidence/RV-02-assets.json
```

Default sample is `src/perception/street.jpg`. To use a captured CARLA frame, pass
`--sample runs/lane_check/<run-id>/rgb.png`. The name rgb.png is stored as a normal
PNG; OpenCV reads BGR and the script converts it to RGB. Sample inference establishes
loading and output shape/labels, not segmentation accuracy. Keep the sample or its
reproducible source available to the reviewer.

Results for the actual supplied checkpoints have NOT RUN in the delivered
workspace: weights were excluded from the uploaded archive. Prior session output
verified the lane identity/loading; the segmentation file size alone is insufficient.
Update this status from successful evidence files, not from this document.

## Verified university-PC result — 14 October 2026

Segmentation asset verification passed: committed LFS checksum and size,
verified offline copy, CPU model loading and sample inference.
Evidence: `evidence/RV-01-assets.json`.

This result supersedes the earlier NOT RUN statement for segmentation.
Lane verification belongs to the separate RV-02 PR.
