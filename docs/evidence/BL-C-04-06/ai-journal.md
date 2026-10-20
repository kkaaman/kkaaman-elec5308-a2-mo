# AI assistance journal

## 20 October 2026: Track stabilisation and live RGB-D benchmark

Tool: ChatGPT (OpenAI) and OpenAI Codex. Human source review completed during implementation and validation.

- Reviewed the existing detector lifecycle, obstacle depth contracts and baseline
  acceptance requirements for BL-C-04–06 before changing the repository.
- Used Codex to inspect the local codebase, implement the temporal track filter and
  add focused tests while preserving the existing detector and RGB-D interfaces.
- Added N-of-M confirmation, bounded median depth history, episode/source/model/track
  identity namespacing, conservative handling of missing track IDs and stale-track
  expiry that cannot directly authorise road clearance or resume.
- Integrated confirmed tracks with the existing aligned depth estimator so distance is
  exposed only for matching current observations with valid depth. Missing, stale or
  invalid depth is not replaced with historical measurements.
- Added replay/benchmark tooling for paired first raw detection and first confirmed
  detection measurements on the same captured sequences, including detector settings,
  timing, model/tracker provenance and conservative cruise-speed screening.
- Used Codex with the local CARLA 0.9.16 server to implement and run the live capture.
  The successful run retained 600 frames across approach, short-occlusion and
  continuity-risk sequences, restored simulator settings and left no actor leaks.
- The measured sequences produced first-confirmed gaps of approximately 39.65 m,
  35.25 m and 42.86 m. The continuity-risk sequence observed a real tracker ID switch.
  Using the existing BL-E-03 braking evidence and frozen 6.0 m stopping margin, the
  screening cap was 4.9 m/s, so the existing proposed 4.0 m/s cruise was retained.
- Reused the already measured BL-B-04 axial depth convention evidence rather than
  recalibrating an unchanged CARLA build and RGB-D rig.
- Final validation passed 46 focused tests and 699 full-suite tests with 1 skipped
  test, together with Ruff, strict Sphinx documentation build and git diff checks.

No commit, push, PR merge, issue closure or acceptance-threshold change was performed
by the assistant or Codex. The user supervised the repository changes and live CARLA
validation. Evaluation-only target geometry and labels were kept outside the detector,
temporal filter and vehicle control path.