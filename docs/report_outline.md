# Assignment 2 report outline

Task: DOC-01 / issue #62

Final submission: PDF, no more than 3 pages and 1,000 words,
excluding references, figures and captions.

This is a planning outline. Replace placeholders only with
verified evidence. Keep code explanations in Sphinx documentation.

## 1. Scope and rationale — target 150 words

- Define the autonomous perception and navigation problem.
- State the demonstrated operating scope and selected scenarios.
- Explain why this scope is appropriate for the assignment.
- Distinguish implemented features from proposed extensions.

Evidence: acceptance criteria, scenario definitions and reviewed
operating limits.

## 2. Extension — target 200 words

- Identify the Assignment 1 controller library reused.
- Explain how CARLA adapters connect vehicle state, perception
  and control.
- Summarise Assignment 2 additions and refactoring.
- Explain how interfaces preserve separation between simulator
  access and reusable logic.

Figure 1 placeholder: Assignment 1 reuse and Assignment 2 data flow.

Evidence: docs/interfaces.md, docs/simulator_boundary.md,
docs/vehicle_adapter.md and relevant merged changes.

## 3. Design decisions — target 250 words

- Describe the target platform and CARLA environment.
- Explain the selected perception and control architecture.
- Discuss freshness checks, controlled stopping and operating limits.
- Compare the main alternatives and explain the chosen trade-offs.
- Describe remote detection/local fallback as draft work until
  integration and measurements support stronger claims.

Evidence: docs/decisions.md, docs/detector_contracts.md,
docs/baseline_threshold_freeze.md and gate records.

## 4. Evidence and limitations — target 300 words

- Present results tied to a commit, configuration and raw run ID.
- Separate unit tests, offline replay and live CARLA evidence.
- Report failures, skipped checks and environmental restrictions.
- Explain what remains unimplemented and the next validation steps.
- Do not treat historical evidence as validation of changed code.

Figure 2 placeholder: measured scenario outcome.
Table 1 placeholder: test and scenario results with evidence links.

Remote-server benchmark: NOT RUN.
Integrated remote fallback demonstration: NOT RUN.
Final submission-commit verification: NOT RUN.

## Final review

- [ ] All four mandatory sections are present.
- [ ] Main text is at most 1,000 words.
- [ ] Exported report is at most 3 pages.
- [ ] Every result links to retained evidence.
- [ ] Figures identify their source run and configuration.
- [ ] Limitations reflect the submission commit.
- [ ] AI assistance is disclosed as required.