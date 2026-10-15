# tests/fixtures/

Small, checked-in test data: recorded frame snippets, fake detections and
golden geometry cases. Keep each file small (well under 1 MB) and deterministic.
Raw CARLA runs belong in `runs/` (ignored by git), not here.

Find this folder with `autonomy.paths.FIXTURES_DIR`.

`Town10HD.lane_graph.json` is a real CARLA 0.9.16 OpenDRIVE export (BL-A-02),
not a synthetic graph. Provenance is in `docs/evidence/BL-A-02-validation.json`;
regeneration and map verification commands are in `docs/lane-topology.md`.
The independent second audit is in `docs/evidence/BL-A-02-double-check.json`:
raw OpenDRIVE links/geometry, dense SDK samples and all-origin reachability.
