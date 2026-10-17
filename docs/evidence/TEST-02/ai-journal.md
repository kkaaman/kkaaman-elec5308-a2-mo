# AI assistance journal

## 17 October 2026: Town10HD RGB capture diagnosis and repair

Tool: OpenAI Codex. Human source review is pending.

- Reviewed user-run navigation/HOLD evidence and repeated the repository lint,
  full pytest suite and strict documentation build.
- Repeated RGB capture after the user updated the NVIDIA driver to 617.42.
  The original executable still stopped after five camera receipts.
- Built independent capture and native thread-stack diagnostics. Local PDB
  symbols identified a readback lambda running on StatsThread while the game
  thread waited for statistics. Checked the corresponding source and compiled
  task-target instruction.
- Generated a separately named, hash-verified executable copy that changes
  the task target from `HighTaskPriority` alone to `AnyNormalThreadHiPriTask`.
  Preserved the official executable. Added a reproducible guarded utility,
  source patch and safety tests.
- Verified independent capture and complete RGB missions. Retained the later
  plain-DX11 device-internal-error trial and the unsuccessful RHI-thread
  stability trial. Built an isolated D3D11 context-protection shim from
  source; checked that protection changed from disabled to enabled. Two
  five-run suites and all RGB HOLD regressions passed in one server session.
  The reproducible runtime builder then passed an additional complete five-run
  cohort. Final offline regression: 653 passed, 1 skipped; all 106 A1 legacy
  test cases passed. All earlier failed diagnostic artifacts were retained.
  Current measured results are in `evidence/TEST-02/README.md`.

No GitHub push, PR creation, issue closure, driver installation or registry
change was performed by the assistant during this repair. The user installed
the graphics driver. Navigation code, route data and frozen acceptance limits
were not changed by the RGB repair.
