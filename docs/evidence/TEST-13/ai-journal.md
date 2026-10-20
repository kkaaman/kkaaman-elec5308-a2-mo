# AI assistance journal

## 18 October 2026: RGB-D depth validation and TEST-13 evidence

Tool: ChatGPT (OpenAI). Human source review completed during implementation and validation.

- Reviewed the current repository state and the earlier baseline implementation for
  BL-B-03/04/05 and TEST-13, then identified the minimum reusable depth-validation
  components that were still compatible with the current codebase.
- Adapted the recovered depth calibration and dataset-analysis tooling to the current
  repository, removing stale dependencies and adding focused tests for installed-build
  axial-versus-radial depth convention selection.
- Ran live CARLA 0.9.16 calibration in Town10HD_Opt. The installed depth representation
  measured as axial, with 107 off-axis samples and a median absolute axial error of
  approximately 0.050 m versus approximately 1.886 m for the radial hypothesis.
- Diagnosed invalid TEST-13 range results caused by analysing unrelated background
  actors and by using CARLA ray-cast geometry for pedestrian references. Reworked the
  evaluation path to isolate the spawned target using instance-segmentation identity
  and to use evaluation-only target bounding-box geometry for range reference.
- Repeated smoke captures and analysis until the target-only methodology was stable,
  then collected a final 180-sample dataset split evenly across train, validation and
  test sets. The validation analysis produced 117 geometry samples and 39 target-box
  comparisons.
- Final measured median absolute geometry errors remained below 0.5 m for every
  measured car and pedestrian range bin through 25 m. All 39 target-box comparisons
  produced valid depth. Optional segmentation masking remained disabled with a
  KEEP_OFF decision because no detector-box ablation/runtime evidence was supplied.
- Rechecked fail-safe RGB-D behaviour using the existing timeout, stale-frame,
  interface and detector-contract tests. Final repository validation passed Ruff,
  666 pytest tests with 1 skipped test, and the strict Sphinx documentation build.

No GitHub push, PR merge, issue closure or acceptance-threshold change was performed
by the assistant. The user ran all CARLA experiments and repository commands.
Evaluation-only ground truth was kept separate from the runtime control path.
